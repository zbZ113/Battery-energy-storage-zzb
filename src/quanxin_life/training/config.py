"""Versioned, target-aware configuration for reproducible training suites."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.training.matrix import TrainingTaskMatrix

__all__ = [
    "CheckpointPolicy",
    "LoggingPolicy",
    "ModelTrainingConfig",
    "TrainingSuiteConfig",
    "TrainingTaskMatrix",
]


class ModelTrainingConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    max_epochs: int = Field(gt=0)
    validation_interval: int = Field(default=5, gt=0)
    early_stopping_patience: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    scheduler_patience: int = Field(default=3, gt=0)
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    minimum_learning_rate: float = Field(default=1e-6, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def learning_rate_bounds_are_consistent(self) -> ModelTrainingConfig:
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        return self


class CheckpointPolicy(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    save_every_epochs: int = Field(default=1, gt=0)
    keep_recent: int = Field(default=3, gt=0)
    save_best: bool = True
    resume_mode: Literal["auto", "never", "required"] = "auto"


class LoggingPolicy(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    console: bool = True
    jsonl: bool = True
    csv: bool = True
    local_mlflow: bool = True


class TrainingSuiteConfig(ContractModel):
    """One immutable dataset/model matrix and its provenance context."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["training-suite-v1"] = "training-suite-v1"
    dataset_id: str = Field(min_length=1)
    target: PredictionTarget
    cutoffs: tuple[int, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    models: tuple[ModelTrainingConfig, ...] = Field(min_length=1)
    physical_gpu_index: int = Field(ge=0)
    precision: Literal["fp32", "bf16"] = "fp32"
    batch_size: int = Field(gt=0)
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    checkpoint: CheckpointPolicy = Field(default_factory=CheckpointPolicy)
    logging: LoggingPolicy = Field(default_factory=LoggingPolicy)

    @model_validator(mode="after")
    def suite_is_unambiguous(self) -> TrainingSuiteConfig:
        if any(cutoff < 0 for cutoff in self.cutoffs) or any(
            current <= previous
            for previous, current in zip(self.cutoffs, self.cutoffs[1:], strict=False)
        ):
            raise ValueError("cutoffs must be unique, non-negative and increasing")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        model_names = [model.name for model in self.models]
        if len(set(model_names)) != len(model_names):
            raise ValueError("models must be unique")
        if (
            self.dataset_id == "MATR"
            and self.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        ):
            raise ValueError("MATR training suite must use the official cycle-life target")
        for model in self.models:
            if not math.isfinite(model.learning_rate):  # defensive for reconstructed models
                raise ValueError("model learning rates must be finite")
        return self

    @property
    def config_sha256(self) -> Sha256:
        payload = self.model_dump(mode="json")
        return sha256_canonical(payload)
