from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from quanxin_life.core import TrainingBlockedReason
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.magnet import (
    MAGNET_LICENSE_STATUS,
    MAGNET_UPSTREAM_COMMIT,
    MAGNetAdapter,
    MAGNetModel,
    compute_magnet_losses,
    magnet_qd_ed_metrics,
)


def test_magnet_binds_reviewed_upstream_and_data_gate() -> None:
    adapter = MAGNetAdapter()

    assert MAGNET_UPSTREAM_COMMIT == "aafb90c551d20748251a35fd51a34eae2539aaca"
    assert MAGNET_LICENSE_STATUS == "VERIFIED_LICENSE_PRESENT"
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.readiness(has_multi_condition_view=False) is (
        TrainingBlockedReason.BLOCKED_DATA_VIEW
    )


def test_magnet_pytorch_212_forward_is_deterministic() -> None:
    torch.manual_seed(17)
    model = MAGNetModel(d_model=4, layers=1)
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.05)

    output = model(torch.tensor([[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]]))

    assert output.shape == (1, 2, 2)
    assert torch.allclose(
        output,
        torch.tensor([[[0.0668, 0.0668], [0.0671, 0.0671]]]),
        atol=1e-4,
    )


def test_magnet_physics_and_meta_losses_follow_reviewed_equations() -> None:
    predictions = torch.tensor([[[1.0, 3.0], [0.5, 2.5]]], requires_grad=True)
    targets = torch.tensor([[[0.9, 3.1], [0.4, 2.4]]])
    soc_markers = torch.tensor([[[0.0], [50.0]]])
    mask = torch.ones_like(predictions, dtype=torch.bool)

    losses = compute_magnet_losses(
        predictions=predictions,
        targets=targets,
        observation_mask=mask,
        soc_markers=soc_markers,
        cutoff_voltage=2.5,
        meta_train_loss=torch.tensor(0.2),
        meta_test_loss=torch.tensor(0.3),
        meta_beta=2.0,
        proportion_weight=0.5,
        voltage_weight=0.25,
    )
    losses["total_loss"].backward()

    assert set(losses) == {
        "raw_mse",
        "soc_proportion_loss",
        "cutoff_voltage_loss",
        "meta_train_loss",
        "meta_test_loss",
        "total_loss",
    }
    assert torch.allclose(
        losses["total_loss"],
        losses["meta_train_loss"]
        + 2.0 * losses["meta_test_loss"]
        + 0.5 * losses["soc_proportion_loss"]
        + 0.25 * losses["cutoff_voltage_loss"],
    )
    assert torch.allclose(losses["raw_mse"], torch.tensor(0.01))
    assert torch.allclose(losses["soc_proportion_loss"], torch.zeros(()))
    assert torch.allclose(losses["cutoff_voltage_loss"], torch.zeros(()))
    assert torch.allclose(losses["total_loss"], torch.tensor(0.8))

    metrics = magnet_qd_ed_metrics(predictions.detach(), targets, mask)
    assert set(metrics) == {"qd_mae", "ed_mae"}
    assert torch.allclose(metrics["qd_mae"], torch.tensor(0.1))
    assert torch.allclose(metrics["ed_mae"], torch.tensor(0.1))


def test_magnet_missing_observations_fail_closed() -> None:
    predictions = torch.zeros((1, 2, 2))
    targets = torch.zeros_like(predictions)
    missing = torch.zeros_like(predictions, dtype=torch.bool)
    with pytest.raises(ValueError, match="observed Qd and Ed"):
        compute_magnet_losses(
            predictions=predictions,
            targets=targets,
            observation_mask=missing,
            soc_markers=torch.zeros((1, 2, 1)),
            cutoff_voltage=2.5,
            meta_train_loss=torch.zeros(()),
            meta_test_loss=torch.zeros(()),
            meta_beta=2.0,
            proportion_weight=0.5,
            voltage_weight=0.25,
        )
    with pytest.raises(ValueError, match="observed Qd and Ed"):
        magnet_qd_ed_metrics(predictions, targets, missing)


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_magnet_configs_freeze_compatibility_and_data_gate(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "magnet" / f"{stage}.json").read_text(encoding="utf-8")
    )

    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW"]
    assert payload["upstream_commit"] == MAGNET_UPSTREAM_COMMIT
    assert payload["runtime_dependencies"] == ["torch"]
    assert payload["compatibility_changes"] == "compatibility_changes.json"


def test_magnet_training_matrix_matches_frozen_adapter_config() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "magnet")
    assert entry["loss_names"] == [
        "raw_mse",
        "soc_proportion",
        "cutoff_voltage",
        "meta_train",
        "meta_test",
    ]
    assert (
        entry["micro_batch_size"],
        entry["gradient_accumulation_steps"],
        entry["effective_batch_size"],
    ) == (32, 1, 32)
    assert entry["selection_metric_name"] == "validation_qd_ed_mae"
    assert entry["source_commit"] == MAGNET_UPSTREAM_COMMIT


@pytest.mark.parametrize("field", ["meta_beta", "proportion_weight", "voltage_weight"])
def test_magnet_rejects_negative_loss_weights(field: str) -> None:
    kwargs = {
        "predictions": torch.ones((1, 2, 2)),
        "targets": torch.ones((1, 2, 2)),
        "observation_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "soc_markers": torch.ones((1, 2, 1)),
        "cutoff_voltage": 2.5,
        "meta_train_loss": torch.tensor(1.0),
        "meta_test_loss": torch.tensor(1.0),
        "meta_beta": 1.0,
        "proportion_weight": 1.0,
        "voltage_weight": 1.0,
    }
    kwargs[field] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        compute_magnet_losses(**kwargs)


@pytest.mark.parametrize("bad_loss", [torch.tensor(float("nan")), torch.ones(2)])
def test_magnet_rejects_nonfinite_or_nonscalar_meta_losses(bad_loss: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="meta losses"):
        compute_magnet_losses(
            predictions=torch.ones((1, 2, 2)),
            targets=torch.ones((1, 2, 2)),
            observation_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            soc_markers=torch.ones((1, 2, 1)),
            cutoff_voltage=2.5,
            meta_train_loss=bad_loss,
            meta_test_loss=torch.tensor(1.0),
            meta_beta=1.0,
            proportion_weight=1.0,
            voltage_weight=1.0,
        )


def test_magnet_rejects_a_fully_missing_qd_or_ed_channel() -> None:
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    mask[..., 1] = False
    with pytest.raises(ValueError, match="Qd and Ed"):
        magnet_qd_ed_metrics(torch.ones((1, 2, 2)), torch.ones((1, 2, 2)), mask)
