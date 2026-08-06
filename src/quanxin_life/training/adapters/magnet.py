"""PyTorch 2.12-compatible MAGNet primitives without upstream I/O."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn
from torch.utils.data import Dataset

from quanxin_life.core import SelectionMetricDirection, TrainingBlockedReason
from quanxin_life.data.dataset_bundle import ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.adapters.base import (
    EvalState,
    EvaluationResult,
    PredictionBatch,
    ResolvedTrainingConfig,
    TrainState,
)
from quanxin_life.training.engine import EpochMetrics

MAGNET_UPSTREAM_COMMIT = "aafb90c551d20748251a35fd51a34eae2539aaca"
MAGNET_LICENSE_STATUS = "VERIFIED_LICENSE_PRESENT"


class MAGNetModel(nn.Module):
    """Audited vLSTM entry reconstructed with the current PyTorch runtime."""

    def __init__(self, *, d_model: int = 12, layers: int = 2) -> None:
        super().__init__()
        if d_model < 1 or layers < 1:
            raise ValueError("MAGNet model dimensions must be positive")
        self.input_projection = nn.Linear(3, d_model)
        self.lstm = nn.LSTM(d_model, d_model, layers, bidirectional=True, batch_first=True)
        self.output_projection = nn.Linear(d_model * 2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != 3:
            raise ValueError("MAGNet input must be [batch, step, 3]")
        hidden, _ = self.lstm(self.input_projection(inputs))
        return self.output_projection.forward(hidden)


class MAGNetAdapter:
    adapter_version = "magnet-pytorch212-v1"
    selection_metric_name = "validation_qd_ed_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = MAGNET_UPSTREAM_COMMIT
    license_status = MAGNET_LICENSE_STATUS

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        if resolved_config.model_family != "magnet":
            raise ValueError("MAGNet adapter received a different model family")
        return MAGNetModel()

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        del manifest
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def train_epoch(self, state: TrainState) -> EpochMetrics:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def validate(self, state: EvalState) -> EvaluationResult:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def predict(self, state: EvalState) -> PredictionBatch:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def export_best(self, destination: Path) -> ArtifactManifest:
        del destination
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def readiness(self, *, has_multi_condition_view: bool) -> TrainingBlockedReason | None:
        if not has_multi_condition_view:
            return TrainingBlockedReason.BLOCKED_DATA_VIEW
        return None


def compute_magnet_losses(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
    soc_markers: torch.Tensor,
    cutoff_voltage: float,
    meta_train_loss: torch.Tensor,
    meta_test_loss: torch.Tensor,
    meta_beta: float,
    proportion_weight: float,
    voltage_weight: float,
) -> dict[str, torch.Tensor]:
    if predictions.shape != targets.shape or predictions.shape[-1] != 2:
        raise ValueError("MAGNet Qd/Ed predictions and targets must align")
    if observation_mask.shape != predictions.shape or observation_mask.dtype is not torch.bool:
        raise ValueError("MAGNet observation mask must align")
    if (
        not bool(observation_mask.any(dim=1).all().item())
        or not bool(observation_mask[..., 0].any(dim=1).all().item())
        or not bool(observation_mask[..., 1].any(dim=1).all().item())
    ):
        raise ValueError("MAGNet each sample requires observed Qd and Ed values")
    if soc_markers.shape != (*predictions.shape[:-1], 1):
        raise ValueError("MAGNet SOC markers must align")
    weights = (meta_beta, proportion_weight, voltage_weight, cutoff_voltage)
    if any(not math.isfinite(value) or value < 0 for value in weights[:-1]):
        raise ValueError("MAGNet loss settings must be finite and non-negative")
    if any(
        tensor.ndim != 0 or not bool(torch.isfinite(tensor).item())
        for tensor in (meta_train_loss, meta_test_loss)
    ):
        raise ValueError("MAGNet meta losses must be finite scalar tensors")
    if not bool(torch.isfinite(predictions).all().item()) or not bool(
        torch.isfinite(targets).all().item()
    ):
        raise ValueError("MAGNet prediction tensors must be finite")
    raw_mse = (predictions - targets).square()[observation_mask].mean()
    predicted_qd = predictions[..., 0]
    qd_max = predicted_qd.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    predicted_ratio = predicted_qd / qd_max
    queried_soc = 100.0 - soc_markers.squeeze(-1)
    soc_max = queried_soc.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    soc_ratio = queried_soc / soc_max
    soc_proportion_loss = functional.mse_loss(predicted_ratio, soc_ratio)
    predicted_voltage = predictions[..., 1]
    minimum_voltage = predicted_voltage.min(dim=1).values
    cutoff_voltage_loss = functional.mse_loss(
        minimum_voltage, torch.full_like(minimum_voltage, cutoff_voltage)
    )
    total_loss = (
        meta_train_loss
        + meta_beta * meta_test_loss
        + proportion_weight * soc_proportion_loss
        + voltage_weight * cutoff_voltage_loss
    )
    return {
        "raw_mse": raw_mse,
        "soc_proportion_loss": soc_proportion_loss,
        "cutoff_voltage_loss": cutoff_voltage_loss,
        "meta_train_loss": meta_train_loss,
        "meta_test_loss": meta_test_loss,
        "total_loss": total_loss,
    }


def magnet_qd_ed_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        predictions.shape != targets.shape
        or observation_mask.shape != predictions.shape
        or observation_mask.dtype is not torch.bool
    ):
        raise ValueError("MAGNet metric tensors must align")
    if (
        not bool(observation_mask.any(dim=1).all().item())
        or not bool(observation_mask[..., 0].any(dim=1).all().item())
        or not bool(observation_mask[..., 1].any(dim=1).all().item())
    ):
        raise ValueError("MAGNet each sample requires observed Qd and Ed values")
    absolute = torch.abs(predictions - targets)
    return {
        "qd_mae": absolute[..., 0][observation_mask[..., 0]].mean(),
        "ed_mae": absolute[..., 1][observation_mask[..., 1]].mean(),
    }


__all__ = [
    "MAGNET_LICENSE_STATUS",
    "MAGNET_UPSTREAM_COMMIT",
    "MAGNetAdapter",
    "MAGNetModel",
    "compute_magnet_losses",
    "magnet_qd_ed_metrics",
]
