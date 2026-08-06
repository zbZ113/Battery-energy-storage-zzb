"""BLAST-Lite scenario adapter with an isolated CPU dependency boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
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
    validate_upstream_artifact,
)
from quanxin_life.training.engine import EpochMetrics

BLAST_UPSTREAM_COMMIT = "b093495b47dc40dd96dba865d91f553619501e94"
BLAST_LICENSE_STATUS = "VERIFIED_LICENSE_PRESENT"


def validate_blast_cpu_requirements(numpy_version: str) -> None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", numpy_version)
    if match is None:
        raise ValueError("BLAST requires a parseable NumPy version")
    if int(match.group(1)) >= 2:
        raise ValueError("BLAST CPU environment requires numpy<2")


def validate_blast_artifacts(artifacts: Mapping[str, tuple[Path, str]]) -> dict[str, Any]:
    if set(artifacts) != {"parameters", "residuals"}:
        raise ValueError("BLAST requires parameters and residuals artifacts")
    verified: dict[str, Any] = {}
    for name in sorted(artifacts):
        path, expected = artifacts[name]
        suffix = path.suffix.lower()
        expected_suffix = ".json" if name == "parameters" else ".parquet"
        if suffix != expected_suffix:
            raise ValueError("BLAST artifact format is unsafe or mismatched")
        verified[name] = validate_upstream_artifact(path, expected)
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("BLAST parameters JSON must be a non-empty object")
            else:
                import pyarrow.parquet as parquet  # type: ignore[import-untyped]

                if not parquet.read_schema(path).names:
                    raise ValueError("BLAST residuals Parquet is empty")
        except (OSError, UnicodeError, json.JSONDecodeError, ImportError) as exc:
            raise ValueError("BLAST artifact cannot be parsed") from exc
    return verified


class BLASTAdapter:
    adapter_version = "blast-lite-cpu-v1"
    selection_metric_name = "scenario_residual_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    task_type = TrainingTaskType.CONDITION_DEGRADATION
    upstream_commit = BLAST_UPSTREAM_COMMIT
    license_status = BLAST_LICENSE_STATUS

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> object:
        if resolved_config.task_type is not self.task_type:
            raise ValueError("BLAST adapter received an incompatible task type")
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}: BLAST requires isolated CPU environment"
        )

    def readiness(self, *, numpy_major: int) -> TrainingBlockedReason | None:
        return TrainingBlockedReason.BLOCKED_DEPENDENCY if numpy_major >= 2 else None

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
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DEPENDENCY}")


__all__ = [
    "BLAST_LICENSE_STATUS",
    "BLAST_UPSTREAM_COMMIT",
    "BLASTAdapter",
    "validate_blast_artifacts",
    "validate_blast_cpu_requirements",
]
