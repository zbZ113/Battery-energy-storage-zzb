"""Offline-safe adapter primitives for the reviewed DITING upstream."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as functional
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
from quanxin_life.training.batching import calculate_effective_batch_size
from quanxin_life.training.engine import EpochMetrics

DITING_UPSTREAM_COMMIT = "b67f48373c591ca62c030fca257ee94910c974ce"
DITING_LICENSE_STATUS = "RESEARCH_ONLY_LICENSE_UNVERIFIED"


class DITINGAdapter:
    """Governed DITING adapter with explicit data and promotion blockers."""

    adapter_version = "diting-offline-v1"
    selection_metric_name = "validation_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = DITING_UPSTREAM_COMMIT
    license_status = DITING_LICENSE_STATUS
    promotion_blocker = TrainingBlockedReason.BLOCKED_LICENSE
    dependency_blocker = TrainingBlockedReason.BLOCKED_DEPENDENCY

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        if resolved_config.model_family != "diting_cptransformer":
            raise ValueError("DITING adapter received a different model family")
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}: "
            "the frozen upstream commit references a missing CPTransformer module"
        )

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        del manifest
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}: "
            "multi-domain HUST model View is unavailable"
        )

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
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_LICENSE}")

    def readiness(self, *, has_hust_view: bool = False) -> TrainingBlockedReason | None:
        if not has_hust_view:
            return TrainingBlockedReason.BLOCKED_DATA_VIEW
        return TrainingBlockedReason.BLOCKED_DEPENDENCY

    def run_manifest_fields(
        self, resolved_config: ResolvedTrainingConfig
    ) -> dict[str, str | int]:
        effective_batch_size = calculate_effective_batch_size(
            micro_batch_size=resolved_config.micro_batch_size,
            visible_gpu_count=1,
            gradient_accumulation_steps=resolved_config.gradient_accumulation_steps,
        )
        if effective_batch_size != resolved_config.effective_batch_size:
            raise ValueError("DITING single-card effective batch does not match task identity")
        return {
            "adapter_version": self.adapter_version,
            "upstream_commit": self.upstream_commit,
            "license_status": self.license_status,
            "visible_gpu_count": 1,
            "micro_batch_size": resolved_config.micro_batch_size,
            "gradient_accumulation_steps": resolved_config.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
        }


def gaussian_mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    kernel_multiplier: float = 2.0,
    kernel_count: int = 5,
) -> torch.Tensor:
    """Multi-kernel Gaussian MMD from DITING, generalized to unequal cohorts."""

    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] != target.shape[1]
        or source.shape[0] == 0
        or target.shape[0] == 0
    ):
        raise ValueError("DITING MMD inputs must be non-empty aligned feature matrices")
    if not math.isfinite(kernel_multiplier) or kernel_multiplier <= 0:
        raise ValueError("DITING kernel multiplier must be positive and finite")
    if isinstance(kernel_count, bool) or not isinstance(kernel_count, int) or kernel_count < 1:
        raise ValueError("DITING kernel count must be a positive integer")
    combined = torch.cat((source, target), dim=0)
    squared_distance = torch.cdist(combined, combined).square()
    sample_count = combined.shape[0]
    denominator = sample_count * sample_count - sample_count
    bandwidth = (squared_distance.detach().sum() / denominator).clamp_min(1e-12)
    bandwidth = bandwidth / (kernel_multiplier ** (kernel_count // 2))
    kernels = torch.zeros_like(squared_distance)
    for index in range(kernel_count):
        kernels = kernels + torch.exp(
            -squared_distance / (bandwidth * kernel_multiplier**index)
        )
    source_count = source.shape[0]
    source_source = kernels[:source_count, :source_count]
    target_target = kernels[source_count:, source_count:]
    source_target = kernels[:source_count, source_count:]
    target_source = kernels[source_count:, :source_count]
    return (
        source_source.mean()
        + target_target.mean()
        - source_target.mean()
        - target_source.mean()
    ).clamp_min(0.0)


def compute_diting_losses(
    *,
    source_predictions: torch.Tensor,
    source_targets: torch.Tensor,
    target_predictions: torch.Tensor,
    target_targets: torch.Tensor,
    source_embeddings: torch.Tensor,
    target_embeddings: torch.Tensor,
    target_cohort: Literal["adaptation", "zero_shot", "test"],
    domain_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute DITING's source + target + weighted MMD objective."""

    _validate_predictions(source_predictions, source_targets, "source")
    if target_cohort == "test":
        raise ValueError("DITING target test labels cannot enter supervised loss")
    if target_cohort == "zero_shot":
        if target_predictions.numel() != 0 or target_targets.numel() != 0:
            raise ValueError("DITING zero-shot target labels must be absent")
        target_loss = source_predictions.new_zeros(())
    else:
        _validate_predictions(target_predictions, target_targets, "target adaptation")
        target_loss = functional.mse_loss(target_predictions, target_targets)
    if not math.isfinite(domain_loss_weight) or domain_loss_weight < 0:
        raise ValueError("DITING domain loss weight must be finite and non-negative")
    source_loss = functional.mse_loss(source_predictions, source_targets)
    prediction_loss = source_loss + target_loss
    mmd_domain_loss = gaussian_mmd_loss(source_embeddings, target_embeddings)
    total_loss = prediction_loss + domain_loss_weight * mmd_domain_loss
    return {
        "source_loss": source_loss,
        "target_loss": target_loss,
        "prediction_loss": prediction_loss,
        "mmd_domain_loss": mmd_domain_loss,
        "total_loss": total_loss,
    }


def validate_diting_cell_cohorts(
    *,
    source_training_cells: Sequence[str],
    adaptation_cells: Mapping[int, Sequence[str]],
    test_cells: Mapping[int, Sequence[str]],
    early_stopping_cells: Sequence[str],
) -> None:
    """Enforce cell-level isolation for zero-, five- and ten-shot evaluation."""

    expected_shots = {0, 5, 10}
    if set(adaptation_cells) != expected_shots or set(test_cells) != expected_shots:
        raise ValueError("DITING requires independent zero-, five- and ten-shot cohorts")
    adaptation_sets: dict[int, set[str]] = {}
    test_sets: dict[int, set[str]] = {}
    for shot in sorted(expected_shots):
        adaptation = tuple(adaptation_cells[shot])
        test = tuple(test_cells[shot])
        if len(adaptation) != shot or len(set(adaptation)) != shot:
            raise ValueError(f"DITING {shot}-shot adaptation cells must be unique")
        if not test or len(set(test)) != len(test):
            raise ValueError(f"DITING {shot}-shot target test cells must be unique")
        adaptation_sets[shot] = set(adaptation)
        test_sets[shot] = set(test)

    all_adaptation = set().union(*adaptation_sets.values())
    all_test = set().union(*test_sets.values())
    if sum(map(len, adaptation_sets.values())) != len(all_adaptation):
        raise ValueError("DITING adaptation cohorts must be cell-disjoint")
    if sum(map(len, test_sets.values())) != len(all_test):
        raise ValueError("DITING target test cohorts must be cell-disjoint")
    if all_adaptation & all_test:
        raise ValueError("DITING adaptation and target test cells overlap")
    source_training = set(source_training_cells)
    if len(source_training) != len(tuple(source_training_cells)):
        raise ValueError("DITING source training cells must be unique")
    if source_training & all_test:
        raise ValueError("DITING target test labels cannot enter source training")
    if source_training & all_adaptation:
        raise ValueError("DITING source and target adaptation cells must be disjoint")
    early_stopping = set(early_stopping_cells)
    if len(early_stopping) != len(tuple(early_stopping_cells)):
        raise ValueError("DITING early-stopping cells must be unique")
    if early_stopping & all_test:
        raise ValueError("DITING target test labels cannot enter early stopping")
    if early_stopping & all_adaptation:
        raise ValueError("DITING early-stopping and adaptation cells must be disjoint")
    if early_stopping & source_training:
        raise ValueError("DITING source training and early-stopping cells must be disjoint")


def _validate_predictions(
    predictions: torch.Tensor, targets: torch.Tensor, cohort: str
) -> None:
    if predictions.shape != targets.shape or predictions.numel() == 0:
        raise ValueError(f"DITING {cohort} predictions and targets must align")
    if not bool(torch.isfinite(predictions).all().item()) or not bool(
        torch.isfinite(targets).all().item()
    ):
        raise ValueError(f"DITING {cohort} predictions and targets must be finite")


__all__ = [
    "DITING_LICENSE_STATUS",
    "DITING_UPSTREAM_COMMIT",
    "DITINGAdapter",
    "compute_diting_losses",
    "gaussian_mmd_loss",
    "validate_diting_cell_cohorts",
]
