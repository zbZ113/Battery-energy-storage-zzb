"""Offline-safe BatteryMFormer losses, artifacts, and governed adapter."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet  # type: ignore[import-untyped]
import torch
import torch.nn.functional as functional
from safetensors import SafetensorError, safe_open
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
    VerifiedUpstreamArtifact,
    validate_upstream_artifact,
)
from quanxin_life.training.batching import calculate_effective_batch_size
from quanxin_life.training.engine import EpochMetrics

BATTERY_MFORMER_UPSTREAM_COMMIT = "febe174032ad4861fa057b9af23f5bcee8a8fb77"
BATTERY_MFORMER_LICENSE_STATUS = "RESEARCH_ONLY_LICENSE_UNVERIFIED"

_REQUIRED_ARTIFACTS = frozenset({"condition_embeddings", "metadata", "history", "normalizer"})
_EXPECTED_SUFFIXES = {
    "condition_embeddings": ".safetensors",
    "metadata": ".json",
    "history": ".parquet",
    "normalizer": ".json",
}


class BatteryMFormerAdapter:
    adapter_version = "batterymformer-offline-v1"
    selection_metric_name = "validation_soh_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = BATTERY_MFORMER_UPSTREAM_COMMIT
    license_status = BATTERY_MFORMER_LICENSE_STATUS
    promotion_blocker = TrainingBlockedReason.BLOCKED_LICENSE

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        del resolved_config
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}: "
            "multi-domain trajectory View and condition embeddings are unavailable"
        )

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
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_LICENSE}")

    def readiness(self, available_artifacts: Set[str]) -> TrainingBlockedReason | None:
        if not available_artifacts >= _REQUIRED_ARTIFACTS:
            return TrainingBlockedReason.BLOCKED_DATA_VIEW
        return None

    def run_manifest_fields(self, resolved_config: ResolvedTrainingConfig) -> dict[str, str | int]:
        effective = calculate_effective_batch_size(
            micro_batch_size=resolved_config.micro_batch_size,
            visible_gpu_count=1,
            gradient_accumulation_steps=resolved_config.gradient_accumulation_steps,
        )
        if effective != resolved_config.effective_batch_size:
            raise ValueError("BatteryMFormer effective batch differs from task identity")
        return {
            "adapter_version": self.adapter_version,
            "upstream_commit": self.upstream_commit,
            "visible_gpu_count": 1,
            "effective_batch_size": effective,
        }


def validate_batterymformer_artifacts(
    artifacts: Mapping[str, tuple[Path, str]],
) -> dict[str, VerifiedUpstreamArtifact]:
    if set(artifacts) != _REQUIRED_ARTIFACTS:
        raise ValueError("BatteryMFormer requires the complete frozen artifact set")
    verified: dict[str, VerifiedUpstreamArtifact] = {}
    for name in sorted(artifacts):
        path, expected_sha256 = artifacts[name]
        if path.suffix.lower() != _EXPECTED_SUFFIXES[name]:
            raise ValueError("BatteryMFormer artifact format is unsafe or mismatched")
        verified[name] = validate_upstream_artifact(path, expected_sha256)
        try:
            if path.suffix.lower() == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise ValueError("BatteryMFormer safetensors artifact is empty")
            elif path.suffix.lower() == ".parquet":
                schema = parquet.read_schema(path)
                if not schema.names:
                    raise ValueError("BatteryMFormer history Parquet is empty")
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("BatteryMFormer JSON artifact must be an object")
        except (OSError, UnicodeError, json.JSONDecodeError, SafetensorError) as exc:
            raise ValueError("BatteryMFormer artifact cannot be safely parsed") from exc
    return verified


def compute_batterymformer_losses(
    *,
    trajectory_predictions: torch.Tensor,
    trajectory_targets: torch.Tensor,
    trajectory_mask: torch.Tensor,
    parameter_predictions: torch.Tensor,
    parameter_targets: torch.Tensor,
    trajectory_embeddings: torch.Tensor,
    slot_attention: torch.Tensor,
    value_memory: torch.Tensor,
    parameter_weight: float = 0.0,
    recovery_weight: float = 0.1,
    alignment_weight: float = 0.1,
    diversity_weight: float = 0.0,
    recovered_trajectory: torch.Tensor | None = None,
    recovery_targets: torch.Tensor | None = None,
    recovered_embeddings: torch.Tensor | None = None,
    target_embeddings: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if trajectory_predictions.shape != trajectory_targets.shape:
        raise ValueError("BatteryMFormer trajectory tensors must align")
    if (
        trajectory_mask.shape != trajectory_predictions.shape
        or trajectory_mask.dtype is not torch.bool
    ):
        raise ValueError("BatteryMFormer trajectory mask must be boolean and aligned")
    if not bool(trajectory_mask.any().item()):
        raise ValueError("BatteryMFormer trajectory mask cannot be empty")
    if parameter_predictions.shape != parameter_targets.shape:
        raise ValueError("BatteryMFormer parameter tensors must align")
    if recovered_trajectory is None:
        recovered_trajectory = recovered_embeddings
    if recovery_targets is None:
        recovery_targets = target_embeddings
    if (
        recovered_trajectory is None
        or recovery_targets is None
        or recovered_trajectory.shape != recovery_targets.shape
    ):
        raise ValueError("BatteryMFormer recovered embeddings must align")
    if recovered_trajectory.shape != trajectory_mask.shape:
        raise ValueError("BatteryMFormer recovery tensors must align with trajectory mask")
    if (
        slot_attention.ndim != 2
        or value_memory.ndim != 2
        or slot_attention.shape[1] != value_memory.shape[0]
    ):
        raise ValueError("BatteryMFormer memory attention and slots must align")
    weights = (parameter_weight, recovery_weight, alignment_weight, diversity_weight)
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("BatteryMFormer loss weights must be finite and non-negative")

    squared_error = (trajectory_predictions - trajectory_targets).square()
    trajectory_loss = squared_error[trajectory_mask].mean()
    parameter_loss = functional.mse_loss(parameter_predictions, parameter_targets)
    recovery_error = (recovered_trajectory - recovery_targets).square()
    recovery_loss = recovery_error[trajectory_mask].mean()
    retrieved = slot_attention @ value_memory
    if retrieved.shape != trajectory_embeddings.shape:
        raise ValueError("BatteryMFormer retrieved and trajectory embeddings must align")
    retrieved_direction = functional.normalize(retrieved.float(), dim=-1, eps=1e-8)
    target_direction = functional.normalize(trajectory_embeddings.float(), dim=-1, eps=1e-8)
    memory_alignment_loss = (1.0 - (retrieved_direction * target_direction).sum(dim=-1)).mean()
    normalized_memory = functional.normalize(value_memory, dim=-1)
    gram = normalized_memory @ normalized_memory.transpose(0, 1)
    identity = torch.eye(value_memory.shape[0], device=value_memory.device)
    memory_diversity_loss = functional.mse_loss(gram, identity)
    total_loss = (
        trajectory_loss
        + parameter_weight * parameter_loss
        + recovery_weight * recovery_loss
        + alignment_weight * memory_alignment_loss
        + diversity_weight * memory_diversity_loss
    )
    return {
        "trajectory_loss": trajectory_loss,
        "parameter_loss": parameter_loss,
        "recovery_loss": recovery_loss,
        "memory_alignment_loss": memory_alignment_loss,
        "memory_diversity_loss": memory_diversity_loss,
        "total_loss": total_loss,
    }


def batterymformer_memory_metrics(
    *,
    slot_attention: torch.Tensor,
    parameter_predictions: torch.Tensor,
    parameter_targets: torch.Tensor,
    trajectory_predictions: torch.Tensor,
    trajectory_targets: torch.Tensor,
    trajectory_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if slot_attention.ndim != 2:
        raise ValueError("BatteryMFormer slot attention must be a matrix")
    return {
        "slot_utilization": slot_attention.mean(dim=0),
        "parameter_mae": torch.abs(parameter_predictions - parameter_targets).mean(),
        "soh_mae": torch.abs(trajectory_predictions - trajectory_targets)[trajectory_mask].mean(),
    }


__all__ = [
    "BATTERY_MFORMER_LICENSE_STATUS",
    "BATTERY_MFORMER_UPSTREAM_COMMIT",
    "BatteryMFormerAdapter",
    "batterymformer_memory_metrics",
    "compute_batterymformer_losses",
    "validate_batterymformer_artifacts",
]
