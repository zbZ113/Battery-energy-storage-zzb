"""Governed contracts shared by offline training adapters."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Protocol, TypeAlias, runtime_checkable

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch.utils.data import Dataset

from quanxin_life.core import SelectionMetricDirection, TrainingReadableSplit
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.dataset_bundle import ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.engine import EpochMetrics
from quanxin_life.training.matrix import TrainingMatrixEntry

ResolvedTrainingConfig: TypeAlias = TrainingMatrixEntry

_APPROVED_ARTIFACT_SUFFIXES = frozenset({".json", ".parquet", ".safetensors", ".ubj"})


class TrainState(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(gt=0)
    split: TrainingReadableSplit = TrainingReadableSplit.TRAIN

    @field_validator("split")
    @classmethod
    def split_is_train_only(cls, value: TrainingReadableSplit) -> TrainingReadableSplit:
        if value is not TrainingReadableSplit.TRAIN:
            raise ValueError("training state may read only the train split")
        return value


class EvalState(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: int = Field(ge=0)
    split: TrainingReadableSplit
    for_selection: bool = False

    @model_validator(mode="after")
    def selection_does_not_read_test(self) -> EvalState:
        if self.for_selection and self.split is TrainingReadableSplit.TEST:
            raise ValueError("selection evaluation cannot read the test split")
        return self


class EvaluationResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    loss: float = Field(ge=0, allow_inf_nan=False)
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def values_are_finite(self) -> EvaluationResult:
        if not math.isfinite(self.loss) or any(
            not math.isfinite(value) for value in self.metrics.values()
        ):
            raise ValueError("evaluation loss and metrics must be finite")
        return self


class PredictionBatch(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: TrainingReadableSplit
    entity_ids: tuple[str, ...] = Field(min_length=1)
    values: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def predictions_are_aligned_and_finite(self) -> PredictionBatch:
        if len(self.entity_ids) != len(self.values):
            raise ValueError("prediction entity_ids and values must align")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("prediction values must be finite")
        return self


class VerifiedUpstreamArtifact(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    size_bytes: int = Field(ge=0)
    sha256: Sha256


class SelectionMetricProvider(Protocol):
    selection_metric_name: str
    selection_metric_direction: SelectionMetricDirection


@runtime_checkable
class TrainingAdapter(SelectionMetricProvider, Protocol):
    adapter_version: str

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module: ...

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]: ...

    def train_epoch(self, state: TrainState) -> EpochMetrics: ...

    def validate(self, state: EvalState) -> EvaluationResult: ...

    def predict(self, state: EvalState) -> PredictionBatch: ...

    def export_best(self, destination: Path) -> ArtifactManifest: ...


def selection_metric_value(
    adapter: SelectionMetricProvider,
    result: EvaluationResult,
) -> float:
    name = adapter.selection_metric_name
    if not name or name not in result.metrics:
        raise ValueError(f"selection metric {name!r} is missing from evaluation result")
    value = result.metrics[name]
    if not math.isfinite(value):
        raise ValueError(f"selection metric {name!r} must be finite")
    return value


def selection_metric_improved(
    candidate: float,
    best: float | None,
    direction: SelectionMetricDirection,
) -> bool:
    if not math.isfinite(candidate):
        raise ValueError("selection metric must be finite")
    if best is None:
        return True
    if not math.isfinite(best):
        raise ValueError("best selection metric must be finite")
    if direction is SelectionMetricDirection.MINIMIZE:
        return candidate < best
    return candidate > best


def validate_upstream_artifact(
    path: Path,
    expected_sha256: str,
) -> VerifiedUpstreamArtifact:
    if (
        len(expected_sha256) != 64
        or expected_sha256.lower() != expected_sha256
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected SHA-256 must be a lowercase digest")
    if path.is_symlink() or not path.is_file():
        raise ValueError("upstream artifact must be a regular non-symlinked file")
    if path.suffix.lower() not in _APPROVED_ARTIFACT_SUFFIXES:
        raise ValueError("unsafe upstream artifact type; use an approved non-executable format")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("upstream artifact SHA-256 does not match the manifest")
    return VerifiedUpstreamArtifact(
        path=path.resolve(strict=True),
        size_bytes=path.stat().st_size,
        sha256=actual_sha256,
    )


__all__ = [
    "EvalState",
    "EvaluationResult",
    "PredictionBatch",
    "ResolvedTrainingConfig",
    "TrainState",
    "TrainingAdapter",
    "VerifiedUpstreamArtifact",
    "selection_metric_improved",
    "selection_metric_value",
    "validate_upstream_artifact",
]
