"""Contracts for leakage-safe, immutable model views."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import DatasetBuildStatus, TrainingReadableSplit, TrainingTaskType
from quanxin_life.core.schemas import ContractModel, Sha256


class ModelViewRow(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1)
    split: TrainingReadableSplit
    features: dict[str, float | None] = Field(min_length=1)
    feature_mask: dict[str, bool] = Field(min_length=1)
    target: float | None = Field(default=None, allow_inf_nan=False)
    right_censored: bool = False
    evidence: str = Field(default="OBSERVED", min_length=1)

    @model_validator(mode="after")
    def masks_match_features(self) -> ModelViewRow:
        if set(self.features) != set(self.feature_mask):
            raise ValueError("feature_mask keys must match features")
        for name, value in self.features.items():
            if self.feature_mask[name] is (value is None):
                raise ValueError("feature mask must be false exactly when value is missing")
        if self.right_censored and self.target is not None:
            raise ValueError("right-censored rows cannot contain a point target")
        if self.target is None and (
            self.evidence == "OBSERVED"
            or (
                not self.right_censored
                and not self.evidence.startswith("NO_POINT_TARGET:")
            )
        ):
            raise ValueError("missing point target requires explicit target evidence")
        return self


class ModelViewConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["model-view-config-v1"] = "model-view-config-v1"
    view_id: str = Field(min_length=1)
    view_version: str = Field(min_length=1)
    task_type: TrainingTaskType
    target_semantics: str = Field(min_length=1)
    entity_key: Literal["cell_id", "system_id", "condition_id"]
    feature_names: tuple[str, ...] = Field(min_length=1)
    mask_semantics: tuple[str, ...] = Field(min_length=1)
    source_dataset_ids: tuple[str, ...] = Field(min_length=1)
    cutoff_cycle: int | None = Field(default=None, ge=0)


class NormalizerStatistics(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    means: dict[str, float]
    scales: dict[str, float]
    training_entity_ids_sha256: Sha256
    fitted_split: Literal["train"] = "train"


class ModelViewArtifact(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or "\\" in value
            or len(path.parts) != 1
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("model view artifact path must be a safe basename")
        if path.suffix not in {".json", ".safetensors"}:
            raise ValueError("model view artifact format is not allowed")
        return value


class ModelViewManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["model-view-manifest-v1", "model-view-manifest-v2"] = (
        "model-view-manifest-v1"
    )
    view_id: str = Field(min_length=1)
    view_version: str = Field(min_length=1)
    task_type: TrainingTaskType
    target_semantics: str = Field(min_length=1)
    mask_semantics: tuple[str, ...] = Field(min_length=1)
    cutoff_cycle: int | None = Field(default=None, ge=0)
    canonical_sha256: Sha256
    split_sha256: Sha256
    builder_version: str = Field(min_length=1)
    builder_code_sha256: Sha256
    config_sha256: Sha256
    normalization_sha256: Sha256
    training_entity_ids_sha256: Sha256
    row_count: int = Field(ge=0)
    entity_key: Literal["cell_id", "system_id", "condition_id"] | None = None
    source_dataset_ids: tuple[str, ...] = ()
    artifacts: tuple[ModelViewArtifact, ...] = ()

    @model_validator(mode="after")
    def v2_has_closed_artifact_context(self) -> ModelViewManifest:
        if self.schema_version == "model-view-manifest-v1":
            if self.artifacts:
                raise ValueError("v1 model view manifest cannot contain artifacts")
            return self
        if self.entity_key is None or not self.source_dataset_ids or not self.artifacts:
            raise ValueError("v2 model view manifest requires entity, source and artifacts")
        paths = [artifact.relative_path for artifact in self.artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("model view artifacts must be unique and sorted")
        return self


class ModelViewBuildResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DatasetBuildStatus
    output_sha256: Sha256
    manifest: ModelViewManifest


__all__ = [
    "ModelViewArtifact",
    "ModelViewBuildResult",
    "ModelViewConfig",
    "ModelViewManifest",
    "ModelViewRow",
    "NormalizerStatistics",
]
