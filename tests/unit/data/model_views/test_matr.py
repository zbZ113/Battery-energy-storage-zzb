from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from quanxin_life.core import sha256_canonical
from quanxin_life.data.model_views.builder import verify_model_view
from quanxin_life.data.model_views.matr import (
    build_tensor_model_view,
    matr_tensor_payload,
    tensor_partition_metadata,
    verify_existing_tensor_model_view,
)
from quanxin_life.data.schemas import SplitManifest


def test_tensor_partition_metadata_preserves_cells_and_explicit_masks() -> None:
    metadata = tensor_partition_metadata(
        partition="validation",
        cell_ids=("MATR_b1c12", "MATR_b2c12"),
        tensors={
            "validation.values": torch.zeros((2, 3, 2, 4, 3)),
            "validation.sample_mask": torch.ones((2, 3, 2, 4), dtype=torch.bool),
            "validation.target_soh": torch.ones((2, 7)),
            "validation.target_mask": torch.ones((2, 7), dtype=torch.bool),
        },
    )

    assert metadata["partition"] == "validation"
    assert metadata["cell_ids"] == ["MATR_b1c12", "MATR_b2c12"]
    assert metadata["tensor_shapes"]["validation.target_mask"] == [2, 7]


def test_tensor_view_is_idempotent_and_rejects_tampering(tmp_path: Path) -> None:
    output = tmp_path / "early-life-sequence-v1"
    tensors = {
        "train.values": torch.zeros((1, 2, 2, 3, 3)),
        "train.sample_mask": torch.ones((1, 2, 2, 3), dtype=torch.bool),
        "train.target_cycle_life": torch.tensor([100.0]),
    }
    metadata = {
        "schema_version": "matr-tensor-view-v1",
        "view_id": "early_life_sequence",
        "partitions": [
            tensor_partition_metadata(
                partition="train",
                cell_ids=("MATR_b1c0",),
                tensors=tensors,
            )
        ],
    }
    normalization = {
        "fitted_split": "train",
        "training_cell_ids_sha256": "f" * 64,
    }
    kwargs = {
        "output_root": output,
        "view_id": "early_life_sequence",
        "view_version": "early-life-sequence-v1",
        "task_type": "cycle_life",
        "target_semantics": "matr_official_cycle_life",
        "mask_semantics": ("explicit_tensor_masks",),
        "cutoff_cycle": 50,
        "canonical_sha256": "a" * 64,
        "split_sha256": "b" * 64,
        "builder_code_sha256": "c" * 64,
        "config_sha256": "d" * 64,
        "normalization_sha256": sha256_canonical(normalization),
        "training_entity_ids_sha256": "f" * 64,
        "metadata": metadata,
        "normalization": normalization,
        "tensors": tensors,
    }

    first = build_tensor_model_view(**kwargs)
    second = build_tensor_model_view(**kwargs)

    assert first.status == "BUILT"
    assert second.status == "SKIPPED_VALID"
    (output / "metadata.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_model_view(output)


def test_existing_tensor_view_is_reused_without_reloading_source_data(
    tmp_path: Path,
) -> None:
    output = tmp_path / "early-life-sequence-v1"
    tensors = {
        "train.values": torch.zeros((1, 2)),
        "train.sample_mask": torch.ones((1, 2), dtype=torch.bool),
    }
    metadata = {
        "schema_version": "matr-tensor-view-v1",
        "view_id": "early_life_sequence",
        "partitions": [
            tensor_partition_metadata(
                partition="train",
                cell_ids=("MATR_b1c0",),
                tensors=tensors,
            )
        ],
    }
    normalization = {"fitted_split": "train", "training_cell_ids_sha256": "f" * 64}
    build_tensor_model_view(
        output_root=output,
        view_id="early_life_sequence",
        view_version="early-life-sequence-v1",
        task_type="cycle_life",
        target_semantics="matr_official_cycle_life",
        mask_semantics=("explicit_tensor_masks",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256="b" * 64,
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
        normalization_sha256=sha256_canonical(normalization),
        training_entity_ids_sha256="f" * 64,
        metadata=metadata,
        normalization=normalization,
        tensors=tensors,
    )

    result = verify_existing_tensor_model_view(
        output,
        view_id="early_life_sequence",
        view_version="early-life-sequence-v1",
        task_type="cycle_life",
        target_semantics="matr_official_cycle_life",
        mask_semantics=("explicit_tensor_masks",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256="b" * 64,
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
    )

    assert result is not None
    assert result.status == "SKIPPED_VALID"


@dataclass(frozen=True)
class _Normalizer:
    dataset_id: str = "MATR"
    data_version: str = "matr-v1"
    feature_version: str = "feature-v1"
    cutoff_cycle: int = 1
    cycle_indices: tuple[int, ...] = (0, 1)
    sample_count: int = 3
    condition_names: tuple[str, ...] = ()
    variable_means: tuple[float, ...] = (0.0, 0.0, 0.0)
    variable_stds: tuple[float, ...] = (1.0, 1.0, 1.0)
    condition_means: tuple[float, ...] = ()
    condition_stds: tuple[float, ...] = ()
    training_cell_ids_sha256: str = "f" * 64
    statistics_sha256: str = "e" * 64


def _early(cell_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        cell_ids=(cell_id,),
        values=torch.zeros((1, 2, 2, 3, 3)),
        cycle_indices=torch.tensor([[0, 1]], dtype=torch.int64),
        cycle_mask=torch.ones((1, 2), dtype=torch.bool),
        sample_mask=torch.ones((1, 2, 2, 3), dtype=torch.bool),
        condition_values=torch.empty((1, 0)),
        condition_mask=torch.empty((1, 0), dtype=torch.bool),
    )


def test_matr_payload_keeps_frozen_partitions_and_real_task_targets() -> None:
    cells = {
        "train": "MATR_train",
        "validation": "MATR_validation",
        "calibration": "MATR_calibration",
        "test": "MATR_test",
    }
    split = SplitManifest(
        dataset_id="MATR",
        train=(cells["train"],),
        validation=(cells["validation"],),
        calibration=(cells["calibration"],),
        test=(cells["test"],),
    )
    values = {"source_split": split, "scalar_normalizer": _Normalizer()}
    for partition, cell_id in cells.items():
        values[f"scalar_{partition}"] = SimpleNamespace(
            cell_ids=(cell_id,),
            early_batch=_early(cell_id),
            raw_labels=torch.tensor([100.0]),
        )
    data = SimpleNamespace(**values)

    metadata, normalization, tensors, training_sha = matr_tensor_payload(
        data,
        view_id="early_life_sequence",
    )

    assert metadata["target_semantics"] == "matr_official_cycle_life"
    assert [item["partition"] for item in metadata["partitions"]] == [
        "train",
        "validation",
        "calibration",
        "test",
    ]
    assert "test.target_cycle_life" in tensors
    assert normalization["fitted_split"] == "train"
    assert training_sha == "f" * 64
