"""Validated MATR run configuration and deterministic model matrix expansion."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import PredictionTarget
from quanxin_life.core.schemas import ContractModel
from quanxin_life.training.config import TrainingSuiteConfig

_MATR_MODELS = {"dummy", "variance", "xgboost", "cpmlp", "hybrid"}


class MatrTrainingPaths(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_mat: str
    raw_manifest: str
    processed_root: str
    conversion_report: str
    supervision_root: str
    supervision_report: str
    split_manifest: str
    run_root: str

    @field_validator("*")
    @classmethod
    def paths_are_safe_and_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in value
        ):
            raise ValueError("training paths must be safe repository-relative POSIX paths")
        return value


class MatrRunConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-run-config-v1"] = "matr-run-config-v1"
    mode: Literal["smoke", "final"]
    suite: TrainingSuiteConfig
    paths: MatrTrainingPaths
    xgboost_early_stopping_rounds: int = Field(gt=0)

    @model_validator(mode="after")
    def matrix_is_the_approved_matr_suite(self) -> MatrRunConfig:
        if self.suite.dataset_id != "MATR":
            raise ValueError("MATR run config requires dataset_id MATR")
        if self.suite.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
            raise ValueError("MATR run config requires the official cycle-life target")
        names = {model.name for model in self.suite.models}
        if names != _MATR_MODELS:
            raise ValueError("MATR run config must contain the five approved models")
        xgboost = next(model for model in self.suite.models if model.name == "xgboost")
        if xgboost.early_stopping_patience != self.xgboost_early_stopping_rounds:
            raise ValueError("XGBoost early stopping settings must agree")
        if self.mode == "final" and self.suite.precision != "fp32":
            raise ValueError("formal MATR results require FP32")
        return self


class MatrThreeBatchTrainingPaths(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    three_batch_manifest: str
    split_manifest: str
    run_root: str

    @field_validator("*")
    @classmethod
    def paths_are_safe_and_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in value
        ):
            raise ValueError("training paths must be safe repository-relative POSIX paths")
        return value


class MatrThreeBatchRunConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-three-batch-run-config-v1"] = (
        "matr-three-batch-run-config-v1"
    )
    mode: Literal["smoke", "final"]
    suite: TrainingSuiteConfig
    paths: MatrThreeBatchTrainingPaths
    xgboost_early_stopping_rounds: int = Field(gt=0)

    @model_validator(mode="after")
    def matrix_is_the_approved_three_batch_suite(self) -> MatrThreeBatchRunConfig:
        if self.suite.dataset_id != "MATR":
            raise ValueError("MATR three-batch config requires dataset_id MATR")
        if self.suite.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
            raise ValueError("MATR three-batch config requires official cycle-life")
        if {model.name for model in self.suite.models} != _MATR_MODELS:
            raise ValueError("MATR three-batch config requires the five approved models")
        xgboost = next(model for model in self.suite.models if model.name == "xgboost")
        if xgboost.early_stopping_patience != self.xgboost_early_stopping_rounds:
            raise ValueError("XGBoost early stopping settings must agree")
        if self.mode == "final" and self.suite.precision != "fp32":
            raise ValueError("formal MATR results require FP32")
        return self


class TrainingRunKey(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["MATR"] = "MATR"
    model_name: str
    cutoff_cycle: int
    seed: int


def build_run_matrix(suite: TrainingSuiteConfig) -> tuple[TrainingRunKey, ...]:
    if suite.dataset_id != "MATR" or suite.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
        raise ValueError("run matrix requires the approved MATR suite")
    return tuple(
        TrainingRunKey(
            model_name=model.name,
            cutoff_cycle=cutoff,
            seed=seed,
        )
        for cutoff in suite.cutoffs
        for model in suite.models
        for seed in suite.seeds
    )
