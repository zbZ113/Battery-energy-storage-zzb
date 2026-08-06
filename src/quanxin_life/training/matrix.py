"""Frozen, provenance-bound training task matrix.

The matrix is intentionally a small declarative layer.  It validates the
identity and gates for a run, but it never computes model or business metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import (
    SelectionMetricDirection,
    TrainingBlockedReason,
    TrainingMode,
    TrainingReadableSplit,
    TrainingTaskType,
)
from quanxin_life.core.schemas import ContractModel, Sha256


class TrainingMatrixEntry(ContractModel):
    """One immutable training identity and its readiness gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    mode: TrainingMode
    task_type: TrainingTaskType
    target_semantics: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    model_view_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    cutoff_cycle: int | None = Field(default=None, ge=0)
    prediction_horizon: int | None = Field(default=None, ge=1)
    seed: int = Field(ge=0)
    fold: int = Field(ge=0)
    optimizer: str = Field(min_length=1)
    loss_names: tuple[str, ...] = Field(min_length=1)
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    effective_batch_size: int = Field(gt=0)
    precision: Literal["fp32", "bf16", "fp16"]
    max_epochs: int = Field(gt=0)
    validation_interval: Literal[5] = 5
    early_stopping_patience: Literal[10] = 10
    selection_metric_name: str = Field(min_length=1)
    selection_metric_direction: SelectionMetricDirection
    required_artifacts: tuple[str, ...] = Field(min_length=1)
    readable_splits: tuple[TrainingReadableSplit, ...] = Field(min_length=1)
    enabled: bool = True
    blocked_reason: TrainingBlockedReason | None = None
    data_sha256: Sha256 | None
    model_view_sha256: Sha256 | None
    split_sha256: Sha256 | None
    config_sha256: Sha256 | None
    source_commit: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_gate_and_identity(self) -> TrainingMatrixEntry:
        if self.effective_batch_size != (
            self.micro_batch_size * self.gradient_accumulation_steps
        ):
            raise ValueError(
                "effective_batch_size must equal micro_batch_size * "
                "gradient_accumulation_steps"
            )

        if self.mode is TrainingMode.SELECT and TrainingReadableSplit.TEST in self.readable_splits:
            raise ValueError("select tasks cannot read the test split")

        if self.task_type is TrainingTaskType.CYCLE_LIFE and self.dataset_id.upper().startswith(
            "NAUMANN"
        ):
            raise ValueError(
                "Naumann condition-level data is not compatible with "
                "cycle_life supervision"
            )

        if self.enabled and self.blocked_reason is not None:
            raise ValueError("enabled entries cannot have a blocked_reason")
        if not self.enabled and self.blocked_reason is None:
            raise ValueError("disabled entries require a blocked_reason")
        hashes = (
            self.data_sha256,
            self.model_view_sha256,
            self.split_sha256,
            self.config_sha256,
        )
        if self.enabled and any(value is None for value in hashes):
            raise ValueError("enabled entries require all provenance hashes")
        return self


class TrainingTaskMatrix(ContractModel):
    """Versioned collection loaded from ``task_matrix_v1.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["task-matrix-v1"] = "task-matrix-v1"
    entries: tuple[TrainingMatrixEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def task_ids_are_unique(self) -> TrainingTaskMatrix:
        ids = [entry.task_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id values must be unique")
        return self


def load_training_matrix(path: str | Path | None = None) -> TrainingTaskMatrix:
    """Load and validate a repository JSON task matrix."""

    matrix_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "configs" / "training" / "task_matrix_v1.json"
    )
    with matrix_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TrainingTaskMatrix.model_validate(payload)


def plan_training_matrix(
    matrix: TrainingTaskMatrix,
    *,
    mode: TrainingMode | None = None,
    include_blocked: bool = True,
) -> tuple[TrainingMatrixEntry, ...]:
    """Return deterministic plan rows without opening a dataset or test split."""

    entries = matrix.entries
    if mode is not None:
        entries = tuple(entry for entry in entries if entry.mode == mode)
    if not include_blocked:
        entries = tuple(entry for entry in entries if entry.enabled)
    return entries


# Short name retained for callers that treat the file itself as the matrix.
TrainingMatrix = TrainingTaskMatrix


__all__ = [
    "TrainingMatrix",
    "TrainingMatrixEntry",
    "TrainingTaskMatrix",
    "load_training_matrix",
    "plan_training_matrix",
]
