from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from quanxin_life.core import TrainingReadableSplit, TrainingTaskType
from quanxin_life.data.model_views.schemas import ModelViewArtifact, ModelViewManifest
from quanxin_life.models.upstream_compat.pbt import UPSTREAM_COMMIT
from quanxin_life.training.adapters.pbt import PBTAdapter, PBTTensorDataset


def _view(tmp_path: Path) -> ModelViewManifest:
    tensors_path = tmp_path / "pbt_tensors.safetensors"
    rows = 6
    save_file(
        {
            "curves": torch.randn(rows, 3, 3, 4),
            "condition_embeddings": torch.randn(rows, 6),
            "targets": torch.linspace(10, 15, rows),
            "valid_cycle_mask": torch.ones(rows, 3, dtype=torch.bool),
            "expert_mask": torch.ones(rows, 4, dtype=torch.bool),
        },
        str(tensors_path),
    )
    metadata_path = tmp_path / "pbt_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "entity_ids": [f"MATR-{i}" for i in range(rows)],
                "splits": ["train", "train", "train", "validation", "validation", "test"],
                "domains": ["MATR"] * rows,
                "seen": [True] * rows,
            }
        ),
        encoding="utf-8",
    )
    artifacts = (
        ModelViewArtifact(
            relative_path=tensors_path.name,
            size_bytes=tensors_path.stat().st_size,
            sha256=__import__("hashlib").sha256(tensors_path.read_bytes()).hexdigest(),
        ),
        ModelViewArtifact(
            relative_path=metadata_path.name,
            size_bytes=metadata_path.stat().st_size,
            sha256=__import__("hashlib").sha256(metadata_path.read_bytes()).hexdigest(),
        ),
    )
    return ModelViewManifest(
        schema_version="model-view-manifest-v2",
        view_id="pbt_multidomain_early_life",
        view_version="pbt-v1",
        task_type=TrainingTaskType.CYCLE_LIFE,
        target_semantics="eol_cycle_life",
        mask_semantics=("valid_cycle_mask", "expert_mask"),
        cutoff_cycle=3,
        canonical_sha256="0" * 64,
        split_sha256="1" * 64,
        builder_version="test",
        builder_code_sha256="2" * 64,
        config_sha256="3" * 64,
        normalization_sha256="4" * 64,
        training_entity_ids_sha256="5" * 64,
        row_count=rows,
        entity_key="cell_id",
        source_dataset_ids=("MATR", "HUST"),
        artifacts=tuple(sorted(artifacts, key=lambda item: item.relative_path)),
    )


def test_pbt_tiny_fit_forward_and_prediction(tmp_path: Path) -> None:
    manifest = _view(tmp_path)
    adapter = PBTAdapter(view_root=tmp_path)
    dataset = adapter.load_view(manifest)
    assert isinstance(dataset, PBTTensorDataset)
    config = type(
        "Config",
        (),
        {
            "model_family": "pbt",
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "curve_length": 4,
            "early_cycle_threshold": 3,
            "d_model": 8,
            "n_heads": 2,
            "e_layers": 1,
            "d_layers": 1,
            "d_ff": 7,
            "dropout": 0.0,
            "num_experts": 4,
            "num_general_experts": 1,
            "condition_embedding_dim": 6,
            "gate_d_ff": 5,
            "top_k": 2,
            "learning_rate": 0.003,
            "weight_decay": 0.02,
        },
    )()
    adapter.build_model(config)
    task = adapter.attach_training(config, device=torch.device("cpu"), seed=38)
    assert task.micro_batch_size == 2
    assert task.gradient_accumulation_steps == 1
    assert task.optimizer.param_groups[0]["lr"] == pytest.approx(0.003)
    assert task.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.02)
    transform = task.target_transform_metadata()
    train_logs = torch.log(torch.tensor([10.0, 11.0, 12.0]))
    assert transform["method"] == "train_log_zscore_v1"
    assert transform["fitted_split"] == "train"
    assert transform["log_mean"] == pytest.approx(float(train_logs.mean()))
    assert transform["log_std"] == pytest.approx(float(train_logs.std(unbiased=False)))
    raw_targets = torch.tensor([10.0, 15.0])
    assert torch.allclose(
        task.decode_predictions(task.encode_targets(raw_targets)),
        raw_targets,
    )
    metrics = task.train_epoch(1)
    assert metrics.loss >= 0
    validation = task.validate(1, split=TrainingReadableSplit.VALIDATION)
    assert validation.metrics["mae"] >= 0
    predictions = task.predict(split=TrainingReadableSplit.TEST)
    assert predictions.entity_ids == ("MATR-5",)
    assert predictions.values[0] == predictions.values[0]
    assert UPSTREAM_COMMIT
