"""Versioned metric definitions and immutable records for offline runs."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel


class MetricDefinition(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1)
    stage: Literal["train", "validation", "test", "calibration"]
    direction: Literal["minimize", "maximize"]
    unit: str = Field(min_length=1)
    aggregation: Literal["mean", "sum", "median", "last", "not_applicable"]
    required: bool = True
    display_name: str = Field(min_length=1)


class MetricRecord(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    model_family: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    task: str = Field(min_length=1)
    target: str = Field(min_length=1)
    split: Literal["train", "validation", "calibration", "test"]
    cutoff: int | None = Field(default=None, ge=0)
    seed: int = Field(ge=0)
    epoch: int = Field(ge=0)
    global_step: int = Field(ge=0)
    metric_name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    timestamp_utc: datetime

    @model_validator(mode="after")
    def valid_timestamp(self) -> MetricRecord:
        if self.timestamp_utc.tzinfo is None or self.timestamp_utc.utcoffset() is None:
            raise ValueError("timestamp_utc must include timezone")
        if not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        object.__setattr__(self, "timestamp_utc", self.timestamp_utc.astimezone(UTC))
        return self


__all__ = ["MetricDefinition", "MetricRecord"]
