"""Offline-safe BattGP primitives for field monitoring."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import Dataset

from quanxin_life.core import (
    SelectionMetricDirection,
    TrainingBlockedReason,
    TrainingTaskType,
)
from quanxin_life.data.dataset_bundle import ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.adapters.base import (
    EvalState,
    EvaluationResult,
    PredictionBatch,
    ResolvedTrainingConfig,
    TrainState,
    validate_upstream_artifact,
)
from quanxin_life.training.batching import calculate_effective_batch_size
from quanxin_life.training.engine import EpochMetrics

BATTGP_UPSTREAM_COMMIT = "6d5e1db3337f0de5f3a533acbc09eddccc178e9d"
BATTGP_LICENSE_STATUS = "VERIFIED_LICENSE_PRESENT"


class BattGPModel(nn.Module):
    """Small torch-only exact GP equivalent of BattGP's recursive state."""

    def __init__(self, *, noise_variance: float = 2.33e-6) -> None:
        super().__init__()
        if not math.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError("BattGP noise_variance must be finite and positive")
        self.noise_variance = float(noise_variance)
        self.register_buffer("train_x", torch.empty((0, 0), dtype=torch.float64))
        self.register_buffer("train_y", torch.empty((0,), dtype=torch.float64))
        self.register_buffer("inverse_covariance", torch.empty((0, 0), dtype=torch.float64))
        self.register_buffer("alpha", torch.empty((0,), dtype=torch.float64))

    @staticmethod
    def kernel(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2 or left.shape[1] != 4 or right.shape[1] != 4:
            raise ValueError("BattGP inputs must be [time, current, SOC, temperature]")
        delta = left[:, None, :] - right[None, :, :]
        lengthscales = left.new_tensor((12.11, 33.75, 45.14))
        rbf = torch.exp(-0.5 * (delta[..., 1:] / lengthscales).square().sum(dim=-1))
        # BattGP's temporal Wiener component is represented by the first feature.
        minimum_time = torch.minimum(left[:, None, 0], right[None, :, 0]).clamp_min(0.0)
        delta_time = (left[:, None, 0] - right[None, :, 0]).abs()
        wiener = minimum_time.square() * (minimum_time / 3.0 + delta_time / 2.0)
        return 0.0099 * rbf + 4.23e-13 * wiener

    def fit(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.shape[0]:
            raise ValueError("BattGP fit features and targets must align")
        if features.shape[1] != 4:
            raise ValueError("BattGP inputs must be [time, current, SOC, temperature]")
        if (
            features.shape[0] == 0
            or not torch.isfinite(features).all()
            or not torch.isfinite(targets).all()
        ):
            raise ValueError("BattGP fit requires non-empty finite observations")
        x = features.to(dtype=torch.float64)
        y = targets.to(dtype=torch.float64)
        covariance = self.kernel(x, x) + self.noise_variance * torch.eye(x.shape[0], dtype=x.dtype)
        inverse = torch.linalg.pinv(covariance)
        self.train_x = x.detach()
        self.train_y = y.detach()
        self.inverse_covariance = inverse.detach()
        self.alpha = (inverse @ y).detach()

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train_x.numel() == 0:
            raise RuntimeError("BattGP model is not fitted")
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError("BattGP inputs must be [time, current, SOC, temperature]")
        if not bool(torch.isfinite(features).all().item()):
            raise ValueError("BattGP prediction inputs must be finite")
        x = features.to(dtype=torch.float64)
        cross = self.kernel(x, self.train_x)
        mean = cross @ self.alpha
        prior = torch.diagonal(self.kernel(x, x))
        variance = prior - torch.sum((cross @ self.inverse_covariance) * cross, dim=1)
        if torch.any(variance < -1e-10):
            raise ValueError("BattGP posterior variance is materially negative")
        variance = torch.where(variance < 0, torch.zeros_like(variance), variance)
        return mean, variance


def compute_battgp_nll(
    model: BattGPModel, features: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    if model.train_x.numel() == 0:
        raise RuntimeError("BattGP model is not fitted")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("BattGP NLL inputs must align")
    covariance = model.kernel(features.to(torch.float64), features.to(torch.float64))
    covariance = covariance + model.noise_variance * torch.eye(
        features.shape[0], dtype=torch.float64
    )
    residual = targets.to(torch.float64)
    inverse = torch.linalg.pinv(covariance)
    sign, logdet = torch.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError("BattGP covariance is not positive definite")
    return cast(
        torch.Tensor,
        0.5
        * (
            residual @ inverse @ residual
            + logdet
            + features.shape[0] * math.log(2 * math.pi)
        ),
    )


def battgp_online_update(model: BattGPModel, features: torch.Tensor, targets: torch.Tensor) -> None:
    if model.train_x.numel() == 0:
        model.fit(features, targets)
        return
    model.fit(
        torch.cat((model.train_x, features), dim=0), torch.cat((model.train_y, targets), dim=0)
    )


def battgp_metrics(
    *, residuals: torch.Tensor, anomaly_labels: torch.Tensor | None
) -> dict[str, float | None | str]:
    if residuals.ndim != 1 or residuals.numel() == 0 or not torch.isfinite(residuals).all():
        raise ValueError("BattGP residuals must be finite rank-1 values")
    if anomaly_labels is None:
        return {
            "residual_mae": float(torch.abs(residuals).mean()),
            "auroc": None,
            "auprc": None,
            "fpr95": None,
            "anomaly_status": "REQUIRES_EXPLICIT_ANOMALY_LABELS",
        }
    if anomaly_labels.shape != residuals.shape or anomaly_labels.dtype is not torch.bool:
        raise ValueError("BattGP anomaly labels must be boolean and aligned")
    if bool(anomaly_labels.all()) or not bool(anomaly_labels.any()):
        return {
            "residual_mae": float(torch.abs(residuals).mean()),
            "auroc": None,
            "auprc": None,
            "fpr95": None,
            "anomaly_status": "INSUFFICIENT_ANOMALY_CLASSES",
        }
    scores = residuals.abs()
    positives = scores[anomaly_labels]
    negatives = scores[~anomaly_labels]
    auroc = float((positives[:, None] > negatives[None, :]).float().mean())
    order = torch.argsort(scores, descending=True)
    labels = anomaly_labels[order].to(torch.float64)
    precision = torch.cumsum(labels, 0) / torch.arange(1, labels.numel() + 1, dtype=torch.float64)
    auprc = float((precision * labels).sum() / labels.sum())
    positive_count = float(labels.sum())
    cumulative_true_positive = torch.cumsum(labels, 0)
    first_95 = torch.nonzero(cumulative_true_positive / positive_count >= 0.95, as_tuple=False)
    threshold_index = int(first_95[0].item()) if first_95.numel() else labels.numel() - 1
    threshold = scores[order][threshold_index]
    fpr95 = float((negatives >= threshold).float().mean())
    return {
        "residual_mae": float(torch.abs(residuals).mean()),
        "auroc": auroc,
        "auprc": auprc,
        "fpr95": fpr95,
        "anomaly_status": "COMPUTED_FROM_EXPLICIT_ANOMALY_LABELS",
    }


class BattGPAdapter:
    adapter_version = "battgp-offline-v1"
    selection_metric_name = "validation_nll"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    task_type = TrainingTaskType.FIELD_MONITORING
    upstream_commit = BATTGP_UPSTREAM_COMMIT
    license_status = BATTGP_LICENSE_STATUS

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        if resolved_config.task_type is not self.task_type:
            raise ValueError("BattGP adapter received an incompatible task type")
        return BattGPModel()

    def readiness(
        self, *, has_field_view: bool, gpytorch_available: bool = True
    ) -> TrainingBlockedReason | None:
        if not has_field_view:
            return TrainingBlockedReason.BLOCKED_DATA_VIEW
        if not gpytorch_available:
            return TrainingBlockedReason.BLOCKED_DEPENDENCY
        return None

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

    def run_manifest_fields(self, resolved_config: ResolvedTrainingConfig) -> dict[str, str | int]:
        effective = calculate_effective_batch_size(
            micro_batch_size=resolved_config.micro_batch_size,
            visible_gpu_count=1,
            gradient_accumulation_steps=resolved_config.gradient_accumulation_steps,
        )
        if effective != resolved_config.effective_batch_size:
            raise ValueError("BattGP effective batch differs from task identity")
        return {
            "adapter_version": self.adapter_version,
            "upstream_commit": self.upstream_commit,
            "visible_gpu_count": 1,
            "effective_batch_size": effective,
        }


def validate_battgp_artifacts(artifacts: Mapping[str, tuple[Path, str]]) -> dict[str, Any]:
    if set(artifacts) != {"parameters", "evidence"}:
        raise ValueError("BattGP requires parameters and evidence JSON artifacts")
    verified: dict[str, Any] = {}
    for name in sorted(artifacts):
        path, expected = artifacts[name]
        if path.suffix.lower() != ".json":
            raise ValueError("BattGP artifact format is unsafe; use JSON")
        verified[name] = validate_upstream_artifact(path, expected)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("BattGP JSON artifact cannot be parsed") from exc
        if not isinstance(payload, dict) or not payload:
            raise ValueError("BattGP JSON artifact must be a non-empty object")
    return verified


__all__ = [
    "BATTGP_LICENSE_STATUS",
    "BATTGP_UPSTREAM_COMMIT",
    "BattGPAdapter",
    "BattGPModel",
    "battgp_metrics",
    "battgp_online_update",
    "compute_battgp_nll",
    "validate_battgp_artifacts",
]
