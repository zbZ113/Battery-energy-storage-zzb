"""Evidence-only boundary for the unverified Smart Feature MATLAB scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from quanxin_life.core import SelectionMetricDirection, TrainingBlockedReason, TrainingTaskType
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

SMART_FEATURE_UPSTREAM_COMMIT = "dc6beea547960cf44d1721734dba93bdcab19a1f"
SMART_FEATURE_LICENSE_STATUS = "RESEARCH_ONLY_LICENSE_UNVERIFIED"


def validate_matlab_reference(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".json":
        raise ValueError("MATLAB reference must be a regular JSON file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("MATLAB reference cannot be parsed") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "matlab-feature-reference-v1"
    ):
        raise ValueError("MATLAB reference schema is invalid")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("MATLAB reference cases cannot be empty")
    for case in cases:
        if not isinstance(case, dict) or not case.get("input_sha256") or "features" not in case:
            raise ValueError("MATLAB reference case requires input_sha256 and features")
    return payload


class SmartFeatureAdapter:
    adapter_version = "smart-feature-evidence-v1"
    selection_metric_name = "feature_evidence_coverage"
    selection_metric_direction = SelectionMetricDirection.MAXIMIZE
    task_type = TrainingTaskType.PARTIAL_CHARGE_FEATURE
    upstream_commit = SMART_FEATURE_UPSTREAM_COMMIT
    license_status = SMART_FEATURE_LICENSE_STATUS
    promotion_blocker = TrainingBlockedReason.BLOCKED_LICENSE

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> object:
        if resolved_config.task_type is not self.task_type:
            raise ValueError("Smart Feature adapter received an incompatible task type")
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}: MATLAB reference is unverified"
        )

    def readiness(self, *, has_matlab_reference: bool) -> TrainingBlockedReason | None:
        return None if has_matlab_reference else TrainingBlockedReason.BLOCKED_DEPENDENCY

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        del manifest
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def train_epoch(self, state: TrainState) -> EpochMetrics:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}")

    def validate(self, state: EvalState) -> EvaluationResult:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}")

    def predict(self, state: EvalState) -> PredictionBatch:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}")

    def export_best(self, destination: Path) -> ArtifactManifest:
        del destination
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_LICENSE}")


__all__ = [
    "SMART_FEATURE_LICENSE_STATUS",
    "SMART_FEATURE_UPSTREAM_COMMIT",
    "SmartFeatureAdapter",
    "validate_matlab_reference",
]
