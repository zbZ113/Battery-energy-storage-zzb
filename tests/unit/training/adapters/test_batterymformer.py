from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from quanxin_life.core import TrainingBlockedReason
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.batterymformer import (
    BATTERY_MFORMER_LICENSE_STATUS,
    BATTERY_MFORMER_UPSTREAM_COMMIT,
    BatteryMFormerAdapter,
    batterymformer_memory_metrics,
    compute_batterymformer_losses,
    validate_batterymformer_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_batterymformer_is_bound_and_blocked_without_trajectory_view() -> None:
    adapter = BatteryMFormerAdapter()

    assert BATTERY_MFORMER_UPSTREAM_COMMIT == "febe174032ad4861fa057b9af23f5bcee8a8fb77"
    assert BATTERY_MFORMER_LICENSE_STATUS == "RESEARCH_ONLY_LICENSE_UNVERIFIED"
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.readiness(set()) is TrainingBlockedReason.BLOCKED_DATA_VIEW
    assert adapter.promotion_blocker is TrainingBlockedReason.BLOCKED_LICENSE


def test_batterymformer_safe_artifacts_are_parsed_not_just_suffix_checked(
    tmp_path: Path,
) -> None:
    embeddings = tmp_path / "condition_embeddings.safetensors"
    save_file({"embeddings": torch.ones((2, 4))}, embeddings)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"schema_version": "condition-metadata-v1"}), encoding="utf-8")
    history = tmp_path / "history.parquet"
    pq.write_table(pa.table({"cell_id": ["cell-1"], "cycle": [1]}), history)
    normalizer = tmp_path / "normalization.json"
    normalizer.write_text(json.dumps({"schema_version": "normalization-v1"}), encoding="utf-8")
    artifacts = {
        "condition_embeddings": (embeddings, _sha(embeddings)),
        "metadata": (metadata, _sha(metadata)),
        "history": (history, _sha(history)),
        "normalizer": (normalizer, _sha(normalizer)),
    }

    assert tuple(validate_batterymformer_artifacts(artifacts)) == tuple(sorted(artifacts))

    unsafe = tmp_path / "history.pkl"
    unsafe.write_bytes(b"forbidden")
    artifacts["history"] = (unsafe, _sha(unsafe))
    with pytest.raises(ValueError, match=r"Parquet|unsafe|format"):
        validate_batterymformer_artifacts(artifacts)


def test_batterymformer_loss_contract_matches_reviewed_memory_math() -> None:
    predictions = torch.tensor([[0.9, 0.8]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.7]])
    mask = torch.tensor([[True, False]])
    parameter_predictions = torch.tensor([[0.2, 0.4]], requires_grad=True)
    parameter_targets = torch.tensor([[0.1, 0.5]])
    recovered = torch.tensor([[0.5, 0.5]], requires_grad=True)
    recovered_targets = torch.tensor([[0.4, 0.6]])
    trajectory_embedding = torch.tensor([[0.8, 0.2]])
    slot_attention = torch.tensor([[0.75, 0.25]], requires_grad=True)
    memory = torch.eye(2, requires_grad=True)

    losses = compute_batterymformer_losses(
        trajectory_predictions=predictions,
        trajectory_targets=targets,
        trajectory_mask=mask,
        parameter_predictions=parameter_predictions,
        parameter_targets=parameter_targets,
        recovered_embeddings=recovered,
        target_embeddings=recovered_targets,
        trajectory_embeddings=trajectory_embedding,
        slot_attention=slot_attention,
        value_memory=memory,
    )
    losses["total_loss"].backward()

    assert set(losses) == {
        "trajectory_loss",
        "parameter_loss",
        "recovery_loss",
        "memory_alignment_loss",
        "memory_diversity_loss",
        "total_loss",
    }
    assert torch.allclose(losses["memory_diversity_loss"], torch.zeros(()))
    assert torch.allclose(losses["trajectory_loss"], torch.tensor(0.01))
    assert torch.allclose(losses["recovery_loss"], torch.tensor(0.01))
    assert torch.allclose(
        losses["total_loss"],
        losses["trajectory_loss"]
        + 0.1 * losses["recovery_loss"]
        + 0.1 * losses["memory_alignment_loss"],
    )
    assert predictions.grad is not None
    assert memory.grad is not None

    metrics = batterymformer_memory_metrics(
        slot_attention=slot_attention.detach(),
        parameter_predictions=parameter_predictions.detach(),
        parameter_targets=parameter_targets,
        trajectory_predictions=predictions.detach(),
        trajectory_targets=targets,
        trajectory_mask=mask,
    )
    assert set(metrics) == {"slot_utilization", "parameter_mae", "soh_mae"}
    assert metrics["slot_utilization"].shape == (2,)


def test_batterymformer_training_matrix_matches_frozen_adapter_config() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "batterymformer")
    assert entry["task_type"] == "soh_trajectory"
    assert entry["loss_names"] == [
        "trajectory",
        "parameter",
        "recovery",
        "memory_alignment",
        "memory_diversity",
    ]
    assert (
        entry["micro_batch_size"],
        entry["gradient_accumulation_steps"],
        entry["effective_batch_size"],
    ) == (128, 2, 256)
    assert entry["source_commit"] == BATTERY_MFORMER_UPSTREAM_COMMIT


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_batterymformer_configs_are_offline_and_blocked(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "batterymformer" / f"{stage}.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW"]
    assert payload["upstream_commit"] == BATTERY_MFORMER_UPSTREAM_COMMIT
    assert payload["runtime_dependencies"] == ["torch", "safetensors", "pyarrow"]
    assert payload["online_condition_embedding"] is False
    assert payload["optimization"]["effective_batch_size"] == 256


def test_batterymformer_recovery_mask_is_independent_of_embedding_width() -> None:
    losses = compute_batterymformer_losses(
        trajectory_predictions=torch.zeros((2, 5)),
        trajectory_targets=torch.ones((2, 5)),
        trajectory_mask=torch.tensor(
            [[True, True, False, False, False], [True, True, True, False, False]]
        ),
        parameter_predictions=torch.zeros((2, 2)),
        parameter_targets=torch.ones((2, 2)),
        recovered_trajectory=torch.zeros((2, 5)),
        recovery_targets=torch.ones((2, 5)),
        trajectory_embeddings=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        slot_attention=torch.eye(2),
        value_memory=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert torch.allclose(losses["recovery_loss"], torch.tensor(1.0))
