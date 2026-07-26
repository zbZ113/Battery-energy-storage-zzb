"""Strict safetensors artifacts for the governed CPMLP and Hybrid models.

No class name from disk is imported dynamically.  Architecture and feature
JSON files are parsed into an explicit allow-list before a network is built.
Only then are hash-verified, finite float32 tensors loaded with strict keys and
shapes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.features.early_cycle_sequence import (
    VARIABLE_NAMES,
    EarlyCycleNormalizer,
)
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
    CyclePatchBatLiNet,
)
from quanxin_life.models.cpmlp import (
    CPMLPLifePredictor,
    _CPMLPNetwork,
    _CurveSchema,
)
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchLifeRegressor,
    EarlyCycleBatch,
)
from quanxin_life.models.hybrid_degradation import (
    HybridDegradationPredictor,
    _FittedContext,
    _HybridNetwork,
    normalise_prediction_cycle_positions,
)
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Inputs,
    HybridPatchV2Output,
    HybridPatchV2Predictor,
)


class DeepArtifactKind(StrEnum):
    CPMLP = "cpmlp-eol80"
    HYBRID = "hybrid-soh-trajectory"
    CYCLEPATCH_DIRECT = "cyclepatch-direct-official-cycle-life"
    CYCLEPATCH_BATLINET = "cyclepatch-batlinet-official-cycle-life"
    CURRENT_HYBRID = "current-hybrid-soh-trajectory"
    HYBRIDPATCH_V2 = "hybridpatch-v2-soh-trajectory"


class DeepArtifactFileRole(StrEnum):
    WEIGHTS = "weights"
    ARCHITECTURE = "architecture"
    FEATURE_CONFIG = "feature-config"
    REFERENCE_LIBRARY = "reference-library"
    INFERENCE_CONTEXT = "inference-context"
    REFERENCE_BATCH = "reference-batch"


class AdvancedOutputTarget(StrEnum):
    MATR_OFFICIAL_CYCLE_LIFE = "matr_official_cycle_life"
    SOH_TRAJECTORY = "soh_trajectory"


class DeepArtifactFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: DeepArtifactFileRole
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain inside the artifact root")
        return path.as_posix()


class DeepModelArtifactManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "deep-model-artifact-v1", "deep-model-artifact-v2"
    ] = "deep-model-artifact-v1"
    artifact_id: str
    artifact_kind: DeepArtifactKind
    output_target: AdvancedOutputTarget | None = None
    files: tuple[DeepArtifactFile, ...] = Field(min_length=3, max_length=6)
    created_at: datetime
    manifest_sha256: Sha256

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def file_roles_are_exact(self) -> DeepModelArtifactManifest:
        roles = [item.role for item in self.files]
        required = {
            DeepArtifactFileRole.WEIGHTS,
            DeepArtifactFileRole.ARCHITECTURE,
            DeepArtifactFileRole.FEATURE_CONFIG,
        }
        if self.artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
            required.add(DeepArtifactFileRole.REFERENCE_LIBRARY)
        if self.schema_version == "deep-model-artifact-v2":
            if self.artifact_kind in {DeepArtifactKind.CPMLP, DeepArtifactKind.HYBRID}:
                raise ValueError("deep artifact v2 is reserved for Advanced models")
            required.add(DeepArtifactFileRole.INFERENCE_CONTEXT)
            if self.artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
                required.add(DeepArtifactFileRole.REFERENCE_BATCH)
            expected_target = _output_target_for_kind(self.artifact_kind)
            if self.output_target is not expected_target:
                raise ValueError("artifact kind and output_target do not match")
        elif self.output_target is not None:
            raise ValueError("deep artifact v1 cannot declare output_target")
        if len(set(roles)) != len(roles) or set(roles) != required:
            raise ValueError("deep artifact files do not match the approved model kind")
        return self


class CPMLPArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cpmlp-architecture-v1"] = "cpmlp-architecture-v1"
    architecture: Literal["quanxin_cpmlp_eol80"] = "quanxin_cpmlp_eol80"
    curve_hidden_dim: int = Field(gt=0)
    aggregation_hidden_dim: int = Field(gt=0)


class CPMLPFeatureConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cpmlp-feature-context-v1"] = "cpmlp-feature-context-v1"
    dataset_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    cycle_indices: tuple[int, ...] = Field(min_length=1)
    voltage_grid_v: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def axes_are_valid(self) -> CPMLPFeatureConfig:
        if any(
            current <= previous
            for previous, current in zip(
                self.cycle_indices, self.cycle_indices[1:], strict=False
            )
        ):
            raise ValueError("cycle_indices must be strictly increasing")
        if any(not math.isfinite(value) for value in self.voltage_grid_v):
            raise ValueError("voltage_grid_v must be finite")
        return self


class HybridArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hybrid-architecture-v1"] = "hybrid-architecture-v1"
    architecture: Literal["quanxin_hybrid_soh_trajectory"] = (
        "quanxin_hybrid_soh_trajectory"
    )
    hidden_dim: int = Field(gt=0)


class CurrentHybridArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["current-hybrid-architecture-v1"] = (
        "current-hybrid-architecture-v1"
    )
    architecture: Literal["quanxin_current_hybrid_advanced"] = (
        "quanxin_current_hybrid_advanced"
    )
    input_dim: Literal[3] = 3
    hidden_dim: Literal[32, 64, 128]
    horizon: int = Field(ge=3)


class HybridFeatureConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hybrid-feature-context-v1"] = "hybrid-feature-context-v1"
    dataset_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    condition_feature_names: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def context_is_valid(self) -> HybridFeatureConfig:
        if len(set(self.condition_feature_names)) != len(self.condition_feature_names):
            raise ValueError("condition_feature_names must be unique")
        if any(not name.strip() for name in self.condition_feature_names):
            raise ValueError("condition_feature_names must not contain blank names")
        if any(
            current <= previous
            for previous, current in zip(
                self.prediction_cycles, self.prediction_cycles[1:], strict=False
            )
        ):
            raise ValueError("prediction_cycles must be strictly increasing")
        if any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
        return self


class CyclePatchArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cyclepatch-architecture-v1"] = (
        "cyclepatch-architecture-v1"
    )
    architecture: Literal["quanxin_cyclepatch_direct"] = (
        "quanxin_cyclepatch_direct"
    )
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float
    max_cycles: Literal[151] = 151
    condition_count: int = Field(gt=0)

    @model_validator(mode="after")
    def approved_search_space(self) -> CyclePatchArchitecture:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("CyclePatch dropout must be 0.05 or 0.10")
        if self.d_model % self.heads:
            raise ValueError("CyclePatch d_model must be divisible by heads")
        return self


class BatLiNetArchitecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["batlinet-architecture-v1"] = (
        "batlinet-architecture-v1"
    )
    architecture: Literal["quanxin_cyclepatch_batlinet"] = (
        "quanxin_cyclepatch_batlinet"
    )
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float
    max_cycles: Literal[151] = 151
    condition_count: int = Field(gt=0)
    lambda_pair: float
    lambda_rank: float
    fusion_alpha: float
    reference_count: Literal[16, 32, 64]

    @model_validator(mode="after")
    def approved_search_space(self) -> BatLiNetArchitecture:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("BatLiNet dropout must be 0.05 or 0.10")
        if self.lambda_pair not in {0.25, 0.5, 1.0}:
            raise ValueError("BatLiNet lambda_pair is outside the approved search")
        if self.lambda_rank not in {0.0, 0.1}:
            raise ValueError("BatLiNet lambda_rank is outside the approved search")
        if self.fusion_alpha not in {0.25, 0.5, 0.75}:
            raise ValueError("BatLiNet fusion_alpha is outside the approved search")
        if self.d_model % self.heads:
            raise ValueError("BatLiNet d_model must be divisible by heads")
        return self


class HybridPatchV2Architecture(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["hybridpatch-v2-architecture-v1"] = (
        "hybridpatch-v2-architecture-v1"
    )
    architecture: Literal["quanxin_hybridpatch_v2"] = "quanxin_hybridpatch_v2"
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float
    max_cycles: Literal[151] = 151
    condition_count: int = Field(gt=0)
    query_token_count: Literal[0, 8, 16]
    query_layers: Literal[1, 2, 3]
    decoder_hidden_dim: Literal[32, 64, 128]
    huber_delta: float = Field(gt=0.0)
    lambda_history: float
    lambda_smooth: float
    lambda_order: float
    lambda_residual: float
    max_prediction_cycle: Literal[500] = 500

    @model_validator(mode="after")
    def approved_search_space(self) -> HybridPatchV2Architecture:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("HybridPatch-v2 dropout must be 0.05 or 0.10")
        approved = {
            "lambda_history": {0.0, 0.1},
            "lambda_smooth": {0.0, 0.01},
            "lambda_order": {0.0, 0.05},
            "lambda_residual": {0.001, 0.01},
        }
        for name, values in approved.items():
            if getattr(self, name) not in values:
                raise ValueError(f"HybridPatch-v2 {name} is outside the approved search")
        if self.d_model % self.heads:
            raise ValueError("HybridPatch-v2 d_model must be divisible by heads")
        return self


class _AdvancedFeatureBase(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["MATR"] = "MATR"
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoff_cycle: Literal[20, 50, 100, 150]
    condition_names: tuple[str, ...] = Field(min_length=1)
    normalization_sha256: Sha256
    candidate_config_sha256: Sha256
    inference_context_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def condition_schema_is_valid(self) -> _AdvancedFeatureBase:
        if len(set(self.condition_names)) != len(self.condition_names) or any(
            not name.strip() for name in self.condition_names
        ):
            raise ValueError("condition_names must be unique and nonblank")
        return self


class EarlyCycleNormalizerArtifactConfig(ContractModel):
    """Complete train-only normalization statistics required for inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["early-cycle-normalizer-artifact-v1"] = (
        "early-cycle-normalizer-artifact-v1"
    )
    dataset_id: Literal["MATR"] = "MATR"
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoff_cycle: Literal[20, 50, 100, 150]
    cycle_indices: tuple[int, ...] = Field(min_length=1)
    sample_count: Literal[150]
    condition_names: tuple[str, ...] = Field(min_length=1)
    variable_means: tuple[float, ...]
    variable_stds: tuple[float, ...]
    condition_means: tuple[float, ...]
    condition_stds: tuple[float, ...]
    training_cell_ids_sha256: Sha256
    statistics_sha256: Sha256

    @model_validator(mode="after")
    def statistics_are_valid(self) -> EarlyCycleNormalizerArtifactConfig:
        self.to_runtime()
        return self

    @classmethod
    def from_runtime(
        cls,
        normalizer: EarlyCycleNormalizer,
    ) -> EarlyCycleNormalizerArtifactConfig:
        return cls.model_validate(
            {
                "dataset_id": normalizer.dataset_id,
                "data_version": normalizer.data_version,
                "feature_version": normalizer.feature_version,
                "cutoff_cycle": normalizer.cutoff_cycle,
                "cycle_indices": normalizer.cycle_indices,
                "sample_count": normalizer.sample_count,
                "condition_names": normalizer.condition_names,
                "variable_means": normalizer.variable_means,
                "variable_stds": normalizer.variable_stds,
                "condition_means": normalizer.condition_means,
                "condition_stds": normalizer.condition_stds,
                "training_cell_ids_sha256": normalizer.training_cell_ids_sha256,
                "statistics_sha256": normalizer.statistics_sha256,
            }
        )

    def to_runtime(self) -> EarlyCycleNormalizer:
        return EarlyCycleNormalizer(
            dataset_id=self.dataset_id,
            data_version=self.data_version,
            feature_version=self.feature_version,
            cutoff_cycle=self.cutoff_cycle,
            cycle_indices=self.cycle_indices,
            sample_count=self.sample_count,
            condition_names=self.condition_names,
            variable_means=self.variable_means,
            variable_stds=self.variable_stds,
            condition_means=self.condition_means,
            condition_stds=self.condition_stds,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
            statistics_sha256=self.statistics_sha256,
        )


class BatLiNetReferenceBatchMetadata(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["batlinet-reference-batch-v1"] = (
        "batlinet-reference-batch-v1"
    )
    dataset_id: Literal["MATR"] = "MATR"
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    normalization_statistics_sha256: Sha256
    cell_ids: tuple[str, ...] = Field(min_length=1)
    condition_names: tuple[str, ...] = Field(min_length=1)
    reference_library_sha256: Sha256
    tensor_file_sha256: Sha256


class AdvancedInferenceContext(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-inference-context-v1"] = (
        "advanced-inference-context-v1"
    )
    output_target: AdvancedOutputTarget
    normalizer: EarlyCycleNormalizerArtifactConfig
    reference_batch: BatLiNetReferenceBatchMetadata | None = None
    inference_context_sha256: Sha256

    @model_validator(mode="after")
    def context_hash_is_valid(self) -> AdvancedInferenceContext:
        payload = self.model_dump(
            mode="json",
            exclude={"inference_context_sha256"},
        )
        if sha256_canonical(payload) != self.inference_context_sha256:
            raise ValueError("inference_context_sha256 does not match its contents")
        return self


class CycleLifeTargetScalerConfig(ContractModel):
    """Complete, hash-bound output normalization for official cycle-life."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["matr-cycle-life-scaler-v1"] = (
        "matr-cycle-life-scaler-v1"
    )
    dataset_id: Literal["MATR"] = "MATR"
    target: PredictionTarget
    cutoff_cycle: Literal[20, 50, 100, 150]
    mean: float
    scale: float
    training_cell_ids_sha256: Sha256
    training_labels_sha256: Sha256
    context_sha256: Sha256

    @model_validator(mode="after")
    def context_is_valid(self) -> CycleLifeTargetScalerConfig:
        if self.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
            raise ValueError("target scaler must use MATR official cycle-life")
        if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("target scaler statistics must be finite with positive scale")
        payload = {
            "schema_version": "matr-cycle-life-target-scaler-v1",
            "dataset_id": self.dataset_id,
            "target": self.target.value,
            "cutoff_cycle": self.cutoff_cycle,
            "mean": self.mean,
            "scale": self.scale,
            "training_cell_ids_sha256": self.training_cell_ids_sha256,
            "training_labels_sha256": self.training_labels_sha256,
        }
        if sha256_canonical(payload) != self.context_sha256:
            raise ValueError("target scaler context_sha256 does not match its contents")
        return self

    @classmethod
    def from_runtime(cls, scaler: CycleLifeTargetScaler) -> CycleLifeTargetScalerConfig:
        return cls.model_validate(
            {
                "dataset_id": scaler.dataset_id,
                "target": scaler.target,
                "cutoff_cycle": scaler.cutoff_cycle,
                "mean": scaler.mean,
                "scale": scaler.scale,
                "training_cell_ids_sha256": scaler.training_cell_ids_sha256,
                "training_labels_sha256": scaler.training_labels_sha256,
                "context_sha256": scaler.context_sha256,
            }
        )

    def to_runtime(self) -> CycleLifeTargetScaler:
        return CycleLifeTargetScaler(
            dataset_id=self.dataset_id,
            target=self.target,
            cutoff_cycle=self.cutoff_cycle,
            mean=self.mean,
            scale=self.scale,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
            training_labels_sha256=self.training_labels_sha256,
            context_sha256=self.context_sha256,
        )


class CycleLifeReferenceLibraryConfig(ContractModel):
    """Complete train-only BatLiNet reference library stored with the model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["cycle-life-reference-library-artifact-v1"] = (
        "cycle-life-reference-library-artifact-v1"
    )
    cell_ids: tuple[str, ...] = Field(min_length=1)
    standardized_labels: tuple[float, ...] = Field(min_length=1)
    quantile_bins: tuple[int, ...] = Field(min_length=1)
    reference_count: Literal[16, 32, 64]
    seed: int
    scaler_context_sha256: Sha256
    training_cell_ids_sha256: Sha256
    training_labels_sha256: Sha256
    library_sha256: Sha256

    @model_validator(mode="after")
    def library_is_valid(self) -> CycleLifeReferenceLibraryConfig:
        if len(self.cell_ids) != self.reference_count or len(set(self.cell_ids)) != len(
            self.cell_ids
        ):
            raise ValueError("reference cell_ids must be unique and match reference_count")
        if not (
            len(self.standardized_labels)
            == len(self.quantile_bins)
            == self.reference_count
        ):
            raise ValueError("reference labels and quantile bins must align")
        if any(not cell_id.strip() for cell_id in self.cell_ids):
            raise ValueError("reference cell_ids must be nonblank")
        if any(not math.isfinite(value) for value in self.standardized_labels):
            raise ValueError("reference labels must be finite")
        if self.quantile_bins != tuple(range(self.reference_count)):
            raise ValueError("reference library must cover every quantile stratum")
        payload = {
            "schema_version": "cycle-life-reference-library-v1",
            "cell_ids": self.cell_ids,
            "standardized_labels": self.standardized_labels,
            "quantile_bins": self.quantile_bins,
            "reference_count": self.reference_count,
            "seed": self.seed,
            "scaler_context_sha256": self.scaler_context_sha256,
            "training_cell_ids_sha256": self.training_cell_ids_sha256,
            "training_labels_sha256": self.training_labels_sha256,
        }
        if sha256_canonical(payload) != self.library_sha256:
            raise ValueError("reference library_sha256 does not match its contents")
        return self

    @classmethod
    def from_runtime(
        cls,
        library: CycleLifeReferenceLibrary,
    ) -> CycleLifeReferenceLibraryConfig:
        return cls.model_validate(
            {
                "cell_ids": library.cell_ids,
                "standardized_labels": library.standardized_labels,
                "quantile_bins": library.quantile_bins,
                "reference_count": library.reference_count,
                "seed": library.seed,
                "scaler_context_sha256": library.scaler_context_sha256,
                "training_cell_ids_sha256": library.training_cell_ids_sha256,
                "training_labels_sha256": library.training_labels_sha256,
                "library_sha256": library.library_sha256,
            }
        )

    def to_runtime(self) -> CycleLifeReferenceLibrary:
        return CycleLifeReferenceLibrary(
            cell_ids=self.cell_ids,
            standardized_labels=self.standardized_labels,
            quantile_bins=self.quantile_bins,
            reference_count=self.reference_count,
            seed=self.seed,
            scaler_context_sha256=self.scaler_context_sha256,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
            training_labels_sha256=self.training_labels_sha256,
            library_sha256=self.library_sha256,
        )


class AdvancedFeatureConfig(_AdvancedFeatureBase):
    schema_version: Literal["advanced-feature-context-v1"] = (
        "advanced-feature-context-v1"
    )
    target_scaler: CycleLifeTargetScalerConfig


class BatLiNetFeatureConfig(_AdvancedFeatureBase):
    schema_version: Literal["batlinet-feature-context-v1"] = (
        "batlinet-feature-context-v1"
    )
    reference_library_sha256: Sha256
    target_scaler: CycleLifeTargetScalerConfig


class HybridPatchV2FeatureConfig(_AdvancedFeatureBase):
    schema_version: Literal["hybridpatch-v2-feature-context-v1"] = (
        "hybridpatch-v2-feature-context-v1"
    )
    max_prediction_cycle: Literal[500] = 500


class CurrentHybridFeatureConfig(_AdvancedFeatureBase):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["current-hybrid-feature-context-v1"] = (
        "current-hybrid-feature-context-v1"
    )
    prediction_cycles: tuple[int, ...] = Field(min_length=3)
    variable_names: tuple[str, ...]
    aggregation_version: Literal["masked-variable-mean-v1"] = (
        "masked-variable-mean-v1"
    )

    @model_validator(mode="after")
    def trajectory_context_is_valid(self) -> CurrentHybridFeatureConfig:
        if self.variable_names != VARIABLE_NAMES:
            raise ValueError(f"variable_names must be fixed to {VARIABLE_NAMES}")
        if any(
            current <= previous
            for previous, current in zip(
                self.prediction_cycles,
                self.prediction_cycles[1:],
                strict=False,
            )
        ):
            raise ValueError("prediction_cycles must be strictly increasing")
        if any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
        if self.prediction_cycles[-1] != 500:
            raise ValueError("prediction_cycles must end at real cycle 500")
        return self


def export_cpmlp_artifact(
    predictor: CPMLPLifePredictor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
) -> DeepModelArtifactManifest:
    if (
        predictor._network is None
        or predictor._dataset_id is None
        or predictor._curve_schema is None
    ):
        raise ValueError("CPMLP predictor must be fitted before export")
    _require_inference_network(predictor._network, _CPMLPNetwork)
    architecture = CPMLPArchitecture(
        curve_hidden_dim=predictor.curve_hidden_dim,
        aggregation_hidden_dim=predictor.aggregation_hidden_dim,
    )
    feature_config = CPMLPFeatureConfig(
        dataset_id=predictor._dataset_id,
        model_version=predictor.model_version,
        feature_version=predictor.feature_version,
        split_version=predictor.split_version,
        data_version=predictor.data_version,
        cutoff_cycle=predictor.cutoff_cycle,
        cycle_indices=predictor._curve_schema.cycle_indices,
        voltage_grid_v=predictor._curve_schema.voltage_grid_v,
    )
    return _export_deep_artifact(
        network=predictor._network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CPMLP,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature_config.model_dump(mode="json"),
        created_at=created_at,
    )


def export_hybrid_artifact(
    predictor: HybridDegradationPredictor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
) -> DeepModelArtifactManifest:
    if predictor._network is None or predictor._context is None:
        raise ValueError("Hybrid predictor must be fitted before export")
    _require_inference_network(predictor._network, _HybridNetwork)
    architecture = HybridArchitecture(hidden_dim=predictor.hidden_dim)
    feature_config = HybridFeatureConfig(
        dataset_id=predictor._context.dataset_id,
        model_version=predictor.model_version,
        feature_version=predictor.feature_version,
        split_version=predictor.split_version,
        data_version=predictor.data_version,
        cutoff_cycle=predictor.cutoff_cycle,
        prediction_cycles=predictor.prediction_cycles,
        condition_feature_names=predictor.condition_feature_names,
    )
    return _export_deep_artifact(
        network=predictor._network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.HYBRID,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature_config.model_dump(mode="json"),
        created_at=created_at,
    )


def load_cpmlp_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> CPMLPLifePredictor:
    if manifest.artifact_kind is not DeepArtifactKind.CPMLP:
        raise ValueError("artifact is not a CPMLP model")
    files = _verify_deep_artifact(artifact_root, manifest)
    architecture = CPMLPArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = CPMLPFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    network = _CPMLPNetwork(
        voltage_points=len(feature.voltage_grid_v),
        cutoff_cycle=feature.cutoff_cycle,
        curve_hidden_dim=architecture.curve_hidden_dim,
        aggregation_hidden_dim=architecture.aggregation_hidden_dim,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    predictor = CPMLPLifePredictor(
        model_version=feature.model_version,
        feature_version=feature.feature_version,
        split_version=feature.split_version,
        data_version=feature.data_version,
        cutoff_cycle=feature.cutoff_cycle,
        curve_hidden_dim=architecture.curve_hidden_dim,
        aggregation_hidden_dim=architecture.aggregation_hidden_dim,
        epochs=1,
    )
    predictor._network = network.eval()
    predictor._dataset_id = feature.dataset_id
    predictor._curve_schema = _CurveSchema(
        cycle_indices=feature.cycle_indices,
        voltage_grid_v=feature.voltage_grid_v,
    )
    return predictor


def load_hybrid_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> HybridDegradationPredictor:
    if manifest.artifact_kind is not DeepArtifactKind.HYBRID:
        raise ValueError("artifact is not a Hybrid model")
    files = _verify_deep_artifact(artifact_root, manifest)
    architecture = HybridArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = HybridFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    network = _HybridNetwork(
        input_dim=3 + len(feature.condition_feature_names),
        horizon=len(feature.prediction_cycles),
        hidden_dim=architecture.hidden_dim,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    predictor = HybridDegradationPredictor(
        model_version=feature.model_version,
        feature_version=feature.feature_version,
        split_version=feature.split_version,
        data_version=feature.data_version,
        cutoff_cycle=feature.cutoff_cycle,
        prediction_cycles=feature.prediction_cycles,
        condition_feature_names=feature.condition_feature_names,
        hidden_dim=architecture.hidden_dim,
        epochs=1,
    )
    predictor._network = network.eval()
    predictor._context = _FittedContext(
        dataset_id=feature.dataset_id,
        condition_feature_names=feature.condition_feature_names,
    )
    return predictor


class LoadedCyclePatchDirect(torch.nn.Module):
    """CyclePatch inference model bound to its approved feature context."""

    def __init__(
        self,
        network: CyclePatchLifeRegressor,
        feature: AdvancedFeatureConfig,
        normalizer: EarlyCycleNormalizerArtifactConfig | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.feature = feature
        self.target_scaler = feature.target_scaler.to_runtime()
        self.normalizer = None if normalizer is None else normalizer.to_runtime()

    def forward(self, batch: EarlyCycleBatch) -> Tensor:
        return self.predict_raw(batch)

    def predict_standardized(self, batch: EarlyCycleBatch) -> Tensor:
        _validate_advanced_batch(batch, self.feature)
        prediction: Tensor = self.network(batch)
        return prediction

    def predict_raw(self, batch: EarlyCycleBatch) -> Tensor:
        return self.target_scaler.inverse_transform(self.predict_standardized(batch))


class LoadedCyclePatchBatLiNet(torch.nn.Module):
    """BatLiNet inference model bound to normalization and reference hashes."""

    def __init__(
        self,
        network: CyclePatchBatLiNet,
        feature: BatLiNetFeatureConfig,
        reference_library: CycleLifeReferenceLibraryConfig,
        reference_batch: EarlyCycleBatch | None = None,
        normalizer: EarlyCycleNormalizerArtifactConfig | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.feature = feature
        self.target_scaler = feature.target_scaler.to_runtime()
        self.reference_library = reference_library.to_runtime()
        self.reference_batch = reference_batch
        self.normalizer = None if normalizer is None else normalizer.to_runtime()

    def encode(self, batch: EarlyCycleBatch) -> Tensor:
        _validate_advanced_batch(batch, self.feature)
        return self.network.encode(batch)

    def _validated_reference_labels(
        self,
        reference_batch: EarlyCycleBatch,
    ) -> Tensor:
        _validate_advanced_batch(reference_batch, self.feature)
        if reference_batch.cell_ids != self.reference_library.cell_ids:
            raise ValueError("reference batch cell_ids do not match the approved library")
        return torch.tensor(
            self.reference_library.standardized_labels,
            dtype=reference_batch.values.dtype,
            device=reference_batch.values.device,
        )

    def predict_standardized(
        self,
        target_batch: EarlyCycleBatch,
        reference_batch: EarlyCycleBatch | None = None,
    ) -> Tensor:
        reference_batch = reference_batch or self.reference_batch
        if reference_batch is None:
            raise ValueError("a self-contained or explicitly approved reference batch is required")
        _validate_advanced_batch(target_batch, self.feature)
        reference_labels = self._validated_reference_labels(reference_batch)
        target_embeddings = self.network.encode(target_batch)
        reference_embeddings = self.network.encode(reference_batch)
        return self.network.fuse_standardized(
            target_embeddings,
            reference_embeddings,
            reference_labels,
        )

    def predict_raw(
        self,
        target_batch: EarlyCycleBatch,
        reference_batch: EarlyCycleBatch | None = None,
    ) -> Tensor:
        return self.target_scaler.inverse_transform(
            self.predict_standardized(target_batch, reference_batch)
        )

    def fuse_standardized(
        self,
        targets: Tensor,
        references: Tensor,
        reference_labels: Tensor,
    ) -> Tensor:
        raise ValueError(
            "raw embedding fusion is disabled; use predict_standardized with "
            "the approved reference batch"
        )

    def fuse_raw(
        self,
        targets: Tensor,
        references: Tensor,
        reference_labels: Tensor,
    ) -> Tensor:
        raise ValueError(
            "raw embedding fusion is disabled; use predict_raw with the approved reference batch"
        )

class LoadedHybridPatchV2(torch.nn.Module):
    """HybridPatch-v2 inference model with a cycle-500 supervision boundary."""

    def __init__(
        self,
        network: HybridPatchV2Predictor,
        feature: HybridPatchV2FeatureConfig,
        normalizer: EarlyCycleNormalizerArtifactConfig | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.feature = feature
        self.normalizer = None if normalizer is None else normalizer.to_runtime()

    def forward(self, inputs: HybridPatchV2Inputs) -> HybridPatchV2Output:
        _validate_advanced_batch(inputs.early_batch, self.feature)
        if torch.any(inputs.prediction_cycles > self.feature.max_prediction_cycle):
            raise ValueError("HybridPatch-v2 prediction cycles cannot exceed 500")
        output: HybridPatchV2Output = self.network(inputs)
        return output


def export_cyclepatch_direct_artifact(
    network: CyclePatchLifeRegressor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
    dataset_id: str,
    data_version: str,
    split_version: str,
    feature_version: str,
    cutoff_cycle: int,
    condition_names: tuple[str, ...],
    normalization_sha256: str,
    candidate_config_sha256: str,
    target_scaler: CycleLifeTargetScaler,
    normalizer: EarlyCycleNormalizer | None = None,
    output_target: AdvancedOutputTarget | str | None = None,
) -> DeepModelArtifactManifest:
    _require_inference_network(network, CyclePatchLifeRegressor)
    config = network.encoder.config
    architecture = CyclePatchArchitecture.model_validate(
        {
            "d_model": config.d_model,
            "layers": config.layers,
            "heads": config.heads,
            "dropout": config.dropout,
            "max_cycles": config.max_cycles,
            "condition_count": network.encoder.condition_count,
        }
    )
    feature = AdvancedFeatureConfig.model_validate(
        {
            "dataset_id": dataset_id,
            "data_version": data_version,
            "split_version": split_version,
            "feature_version": feature_version,
            "cutoff_cycle": cutoff_cycle,
            "condition_names": condition_names,
            "normalization_sha256": normalization_sha256,
            "candidate_config_sha256": candidate_config_sha256,
            "target_scaler": CycleLifeTargetScalerConfig.from_runtime(target_scaler).model_dump(
                mode="json"
            ),
        }
    )
    _validate_scalar_scaler_context(feature.target_scaler, feature.cutoff_cycle)
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    normalized_target = _optional_output_target(
        DeepArtifactKind.CYCLEPATCH_DIRECT,
        output_target,
        normalizer,
    )
    return _export_deep_artifact(
        network=network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature.model_dump(mode="json"),
        created_at=created_at,
        normalizer=normalizer,
        output_target=normalized_target,
    )


def load_cyclepatch_direct_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    *,
    expected_normalization_sha256: str,
    expected_candidate_config_sha256: str,
    expected_target_scaler_context_sha256: str,
) -> LoadedCyclePatchDirect:
    files = _advanced_files(
        artifact_root, manifest, expected_kind=DeepArtifactKind.CYCLEPATCH_DIRECT
    )
    architecture = CyclePatchArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = AdvancedFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    _validate_expected_feature_hashes(
        feature,
        expected_normalization_sha256=expected_normalization_sha256,
        expected_candidate_config_sha256=expected_candidate_config_sha256,
    )
    _validate_target_scaler_context(
        feature.target_scaler, expected_target_scaler_context_sha256
    )
    normalizer = None
    if manifest.schema_version == "deep-model-artifact-v2":
        context, _reference_batch = _load_self_contained_inference_context(
            files,
            manifest,
            feature,
        )
        normalizer = context.normalizer
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    network = CyclePatchLifeRegressor(
        _cyclepatch_config_from(architecture),
        condition_count=architecture.condition_count,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    return LoadedCyclePatchDirect(network.eval(), feature, normalizer).eval()


def export_cyclepatch_batlinet_artifact(
    network: CyclePatchBatLiNet,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
    dataset_id: str,
    data_version: str,
    split_version: str,
    feature_version: str,
    cutoff_cycle: int,
    condition_names: tuple[str, ...],
    normalization_sha256: str,
    candidate_config_sha256: str,
    reference_library: CycleLifeReferenceLibrary,
    target_scaler: CycleLifeTargetScaler,
    reference_batch: EarlyCycleBatch | None = None,
    normalizer: EarlyCycleNormalizer | None = None,
    output_target: AdvancedOutputTarget | str | None = None,
) -> DeepModelArtifactManifest:
    _require_inference_network(network, CyclePatchBatLiNet)
    config = network.config
    encoder = config.encoder
    architecture = BatLiNetArchitecture.model_validate(
        {
            "d_model": encoder.d_model,
            "layers": encoder.layers,
            "heads": encoder.heads,
            "dropout": encoder.dropout,
            "max_cycles": encoder.max_cycles,
            "condition_count": network.encoder.condition_count,
            "lambda_pair": config.lambda_pair,
            "lambda_rank": config.lambda_rank,
            "fusion_alpha": config.fusion_alpha,
            "reference_count": config.reference_count,
        }
    )
    reference_config = CycleLifeReferenceLibraryConfig.from_runtime(reference_library)
    feature = BatLiNetFeatureConfig.model_validate(
        {
            "dataset_id": dataset_id,
            "data_version": data_version,
            "split_version": split_version,
            "feature_version": feature_version,
            "cutoff_cycle": cutoff_cycle,
            "condition_names": condition_names,
            "normalization_sha256": normalization_sha256,
            "candidate_config_sha256": candidate_config_sha256,
            "reference_library_sha256": reference_config.library_sha256,
            "target_scaler": CycleLifeTargetScalerConfig.from_runtime(target_scaler).model_dump(
                mode="json"
            ),
        }
    )
    _validate_scalar_scaler_context(feature.target_scaler, feature.cutoff_cycle)
    _validate_reference_scaler_context(
        reference_config,
        feature.target_scaler,
        architecture.reference_count,
    )
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    normalized_target = _optional_output_target(
        DeepArtifactKind.CYCLEPATCH_BATLINET,
        output_target,
        normalizer,
    )
    if normalized_target is not None and reference_batch is None:
        raise ValueError("BatLiNet artifact v2 requires a reference batch")
    return _export_deep_artifact(
        network=network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_BATLINET,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature.model_dump(mode="json"),
        reference_library=reference_config.model_dump(mode="json"),
        created_at=created_at,
        normalizer=normalizer,
        output_target=normalized_target,
        reference_batch=reference_batch,
        reference_library_sha256=reference_config.library_sha256,
    )


def load_cyclepatch_batlinet_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    *,
    expected_normalization_sha256: str,
    expected_candidate_config_sha256: str,
    expected_reference_library_sha256: str,
    expected_target_scaler_context_sha256: str,
) -> LoadedCyclePatchBatLiNet:
    files = _advanced_files(
        artifact_root, manifest, expected_kind=DeepArtifactKind.CYCLEPATCH_BATLINET
    )
    architecture = BatLiNetArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = BatLiNetFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    _validate_expected_feature_hashes(
        feature,
        expected_normalization_sha256=expected_normalization_sha256,
        expected_candidate_config_sha256=expected_candidate_config_sha256,
    )
    _validate_target_scaler_context(
        feature.target_scaler, expected_target_scaler_context_sha256
    )
    if feature.reference_library_sha256 != expected_reference_library_sha256:
        raise ValueError("reference_library_sha256 does not match the approved library")
    reference_library = CycleLifeReferenceLibraryConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.REFERENCE_LIBRARY])
    )
    if reference_library.library_sha256 != feature.reference_library_sha256:
        raise ValueError("reference library does not match feature context")
    _validate_reference_scaler_context(
        reference_library,
        feature.target_scaler,
        architecture.reference_count,
    )
    reference_batch = None
    normalizer = None
    if manifest.schema_version == "deep-model-artifact-v2":
        context, reference_batch = _load_self_contained_inference_context(
            files,
            manifest,
            feature,
            reference_library=reference_library,
        )
        normalizer = context.normalizer
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    network = CyclePatchBatLiNet(
        BatLiNetConfig(
            encoder=_cyclepatch_config_from(architecture),
            lambda_pair=architecture.lambda_pair,
            lambda_rank=architecture.lambda_rank,
            fusion_alpha=architecture.fusion_alpha,
            reference_count=architecture.reference_count,
        ),
        condition_count=architecture.condition_count,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    return LoadedCyclePatchBatLiNet(
        network.eval(),
        feature,
        reference_library,
        reference_batch,
        normalizer,
    ).eval()


def export_hybridpatch_v2_artifact(
    network: HybridPatchV2Predictor,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
    dataset_id: str,
    data_version: str,
    split_version: str,
    feature_version: str,
    cutoff_cycle: int,
    condition_names: tuple[str, ...],
    normalization_sha256: str,
    candidate_config_sha256: str,
    normalizer: EarlyCycleNormalizer | None = None,
    output_target: AdvancedOutputTarget | str | None = None,
) -> DeepModelArtifactManifest:
    _require_inference_network(network, HybridPatchV2Predictor)
    config = network.config
    encoder = config.cyclepatch
    architecture = HybridPatchV2Architecture.model_validate(
        {
            "d_model": encoder.d_model,
            "layers": encoder.layers,
            "heads": encoder.heads,
            "dropout": encoder.dropout,
            "max_cycles": encoder.max_cycles,
            "condition_count": network.cycle_encoder.condition_count,
            "query_token_count": config.query_token_count,
            "query_layers": config.query_layers,
            "decoder_hidden_dim": config.decoder_hidden_dim,
            "huber_delta": config.huber_delta,
            "lambda_history": config.lambda_history,
            "lambda_smooth": config.lambda_smooth,
            "lambda_order": config.lambda_order,
            "lambda_residual": config.lambda_residual,
            "max_prediction_cycle": config.max_prediction_cycle,
        }
    )
    feature = HybridPatchV2FeatureConfig.model_validate(
        {
            "dataset_id": dataset_id,
            "data_version": data_version,
            "split_version": split_version,
            "feature_version": feature_version,
            "cutoff_cycle": cutoff_cycle,
            "condition_names": condition_names,
            "normalization_sha256": normalization_sha256,
            "candidate_config_sha256": candidate_config_sha256,
            "max_prediction_cycle": config.max_prediction_cycle,
        }
    )
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    normalized_target = _optional_output_target(
        DeepArtifactKind.HYBRIDPATCH_V2,
        output_target,
        normalizer,
    )
    return _export_deep_artifact(
        network=network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.HYBRIDPATCH_V2,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature.model_dump(mode="json"),
        created_at=created_at,
        normalizer=normalizer,
        output_target=normalized_target,
    )


def load_hybridpatch_v2_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    *,
    expected_normalization_sha256: str,
    expected_candidate_config_sha256: str,
) -> LoadedHybridPatchV2:
    files = _advanced_files(
        artifact_root, manifest, expected_kind=DeepArtifactKind.HYBRIDPATCH_V2
    )
    architecture = HybridPatchV2Architecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = HybridPatchV2FeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    _validate_expected_feature_hashes(
        feature,
        expected_normalization_sha256=expected_normalization_sha256,
        expected_candidate_config_sha256=expected_candidate_config_sha256,
    )
    if architecture.max_prediction_cycle != feature.max_prediction_cycle:
        raise ValueError("HybridPatch-v2 prediction boundary does not match feature context")
    normalizer = None
    if manifest.schema_version == "deep-model-artifact-v2":
        context, _reference_batch = _load_self_contained_inference_context(
            files,
            manifest,
            feature,
        )
        normalizer = context.normalizer
    _validate_condition_count(architecture.condition_count, feature.condition_names)
    network = HybridPatchV2Predictor(
        HybridPatchV2Config(
            cyclepatch=_cyclepatch_config_from(architecture),
            query_token_count=architecture.query_token_count,
            query_layers=architecture.query_layers,
            decoder_hidden_dim=architecture.decoder_hidden_dim,
            huber_delta=architecture.huber_delta,
            lambda_history=architecture.lambda_history,
            lambda_smooth=architecture.lambda_smooth,
            lambda_order=architecture.lambda_order,
            lambda_residual=architecture.lambda_residual,
            max_prediction_cycle=architecture.max_prediction_cycle,
        ),
        condition_count=architecture.condition_count,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    return LoadedHybridPatchV2(network.eval(), feature, normalizer).eval()


class LoadedCurrentHybrid(torch.nn.Module):
    """Current Hybrid baseline bound to its advanced masked-mean feature contract."""

    def __init__(
        self,
        network: _HybridNetwork,
        feature: CurrentHybridFeatureConfig,
        normalizer: EarlyCycleNormalizerArtifactConfig | None = None,
    ) -> None:
        super().__init__()
        self.network = network
        self.feature = feature
        self.normalizer = None if normalizer is None else normalizer.to_runtime()
        cycle_positions = normalise_prediction_cycle_positions(
            prediction_cycles=feature.prediction_cycles,
            cutoff_cycle=feature.cutoff_cycle,
        )
        self.register_buffer(
            "cycle_positions",
            torch.tensor(cycle_positions, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, batch: EarlyCycleBatch, initial_soh: Tensor) -> Tensor:
        _validate_advanced_batch(batch, self.feature)
        if initial_soh.dtype != batch.values.dtype:
            raise ValueError("initial_soh dtype must match the inference batch")
        if initial_soh.device != batch.values.device:
            raise ValueError("initial_soh device must match the inference batch")
        if initial_soh.shape != (len(batch.cell_ids),):
            raise ValueError("initial_soh must align with inference batch cells")
        if not bool(
            (
                torch.isfinite(initial_soh)
                & (initial_soh > 0)
                & (initial_soh <= 1.5)
            )
            .all()
            .item()
        ):
            raise ValueError("initial_soh must be finite and in (0, 1.5]")
        mask = batch.sample_mask.unsqueeze(-1)
        clean = torch.where(mask, batch.values, torch.zeros_like(batch.values))
        count = mask.to(batch.values.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
        features = clean.sum(dim=(1, 2, 3)) / count
        output: Tensor = self.network(
            features,
            initial_soh,
            self.cycle_positions.to(
                device=batch.values.device,
                dtype=batch.values.dtype,
            ),
        )
        return output


def export_current_hybrid_artifact(
    network: _HybridNetwork,
    *,
    artifact_root: Path,
    artifact_id: str,
    created_at: datetime,
    dataset_id: str,
    data_version: str,
    split_version: str,
    feature_version: str,
    cutoff_cycle: int,
    condition_names: tuple[str, ...],
    prediction_cycles: tuple[int, ...],
    variable_names: tuple[str, ...],
    aggregation_version: str,
    normalization_sha256: str,
    candidate_config_sha256: str,
    normalizer: EarlyCycleNormalizer | None = None,
    output_target: AdvancedOutputTarget | str | None = None,
) -> DeepModelArtifactManifest:
    _require_inference_network(network, _HybridNetwork)
    input_layer = network.encoder[0]
    if not isinstance(input_layer, torch.nn.Linear):
        raise ValueError("Current Hybrid encoder input layer is not approved")
    architecture = CurrentHybridArchitecture.model_validate(
        {
            "input_dim": input_layer.in_features,
            "hidden_dim": input_layer.out_features,
            "horizon": network.horizon,
        }
    )
    feature = CurrentHybridFeatureConfig.model_validate(
        {
            "dataset_id": dataset_id,
            "data_version": data_version,
            "split_version": split_version,
            "feature_version": feature_version,
            "cutoff_cycle": cutoff_cycle,
            "condition_names": condition_names,
            "normalization_sha256": normalization_sha256,
            "candidate_config_sha256": candidate_config_sha256,
            "prediction_cycles": prediction_cycles,
            "variable_names": variable_names,
            "aggregation_version": aggregation_version,
        }
    )
    if architecture.input_dim != len(feature.variable_names):
        raise ValueError("variable_names do not match Current Hybrid input_dim")
    if architecture.horizon != len(feature.prediction_cycles):
        raise ValueError("prediction_cycles do not match Current Hybrid horizon")
    normalized_target = _optional_output_target(
        DeepArtifactKind.CURRENT_HYBRID,
        output_target,
        normalizer,
    )
    return _export_deep_artifact(
        network=network,
        artifact_root=artifact_root,
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CURRENT_HYBRID,
        architecture=architecture.model_dump(mode="json"),
        feature_config=feature.model_dump(mode="json"),
        created_at=created_at,
        normalizer=normalizer,
        output_target=normalized_target,
    )


def load_current_hybrid_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    *,
    expected_prediction_cycles: tuple[int, ...],
    expected_variable_names: tuple[str, ...],
    expected_aggregation_version: str,
    expected_normalization_sha256: str,
    expected_candidate_config_sha256: str,
) -> LoadedCurrentHybrid:
    files = _advanced_files(
        artifact_root,
        manifest,
        expected_kind=DeepArtifactKind.CURRENT_HYBRID,
    )
    architecture = CurrentHybridArchitecture.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.ARCHITECTURE])
    )
    feature = CurrentHybridFeatureConfig.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    )
    _validate_expected_feature_hashes(
        feature,
        expected_normalization_sha256=expected_normalization_sha256,
        expected_candidate_config_sha256=expected_candidate_config_sha256,
    )
    if feature.prediction_cycles != expected_prediction_cycles:
        raise ValueError("prediction_cycles do not match the approved trajectory axis")
    if feature.variable_names != expected_variable_names:
        raise ValueError("variable_names do not match the approved feature schema")
    if feature.aggregation_version != expected_aggregation_version:
        raise ValueError("aggregation_version does not match the approved feature transform")
    if architecture.input_dim != len(feature.variable_names):
        raise ValueError("variable_names do not match Current Hybrid input_dim")
    if architecture.horizon != len(feature.prediction_cycles):
        raise ValueError("prediction_cycles do not match Current Hybrid horizon")
    normalizer = None
    if manifest.schema_version == "deep-model-artifact-v2":
        context, _reference_batch = _load_self_contained_inference_context(
            files,
            manifest,
            feature,
        )
        normalizer = context.normalizer
    network = _HybridNetwork(
        input_dim=architecture.input_dim,
        horizon=architecture.horizon,
        hidden_dim=architecture.hidden_dim,
    )
    _load_verified_state(network, files[DeepArtifactFileRole.WEIGHTS])
    return LoadedCurrentHybrid(network.eval(), feature, normalizer).eval()


def _advanced_files(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    *,
    expected_kind: DeepArtifactKind,
) -> dict[DeepArtifactFileRole, Path]:
    if manifest.artifact_kind is not expected_kind:
        raise ValueError(
            f"artifact kind is not an approved {expected_kind.value} model"
        )
    return _verify_deep_artifact(artifact_root, manifest)


def _advanced_feature_from_files(
    files: dict[DeepArtifactFileRole, Path],
    kind: DeepArtifactKind,
) -> _AdvancedFeatureBase:
    payload = _read_strict_json(files[DeepArtifactFileRole.FEATURE_CONFIG])
    feature_types: dict[DeepArtifactKind, type[_AdvancedFeatureBase]] = {
        DeepArtifactKind.CYCLEPATCH_DIRECT: AdvancedFeatureConfig,
        DeepArtifactKind.CYCLEPATCH_BATLINET: BatLiNetFeatureConfig,
        DeepArtifactKind.CURRENT_HYBRID: CurrentHybridFeatureConfig,
        DeepArtifactKind.HYBRIDPATCH_V2: HybridPatchV2FeatureConfig,
    }
    feature_type = feature_types.get(kind)
    if feature_type is None:
        raise ValueError("artifact is not an Advanced deployment model")
    return feature_type.model_validate(payload)


def _load_self_contained_inference_context(
    files: dict[DeepArtifactFileRole, Path],
    manifest: DeepModelArtifactManifest,
    feature: _AdvancedFeatureBase,
    *,
    reference_library: CycleLifeReferenceLibraryConfig | None = None,
) -> tuple[AdvancedInferenceContext, EarlyCycleBatch | None]:
    if manifest.schema_version != "deep-model-artifact-v2":
        raise ValueError("self-contained Advanced runtime requires artifact v2")
    context = AdvancedInferenceContext.model_validate(
        _read_strict_json(files[DeepArtifactFileRole.INFERENCE_CONTEXT])
    )
    if (
        manifest.output_target is not context.output_target
        or context.output_target is not _output_target_for_kind(manifest.artifact_kind)
        or feature.inference_context_sha256 != context.inference_context_sha256
    ):
        raise ValueError("inference context does not match the artifact manifest")
    normalizer = context.normalizer
    if (
        normalizer.dataset_id != feature.dataset_id
        or normalizer.data_version != feature.data_version
        or normalizer.feature_version != feature.feature_version
        or normalizer.cutoff_cycle != feature.cutoff_cycle
        or normalizer.condition_names != feature.condition_names
        or normalizer.statistics_sha256 != feature.normalization_sha256
    ):
        raise ValueError("inference normalizer does not match feature context")
    if manifest.artifact_kind is not DeepArtifactKind.CYCLEPATCH_BATLINET:
        if context.reference_batch is not None:
            raise ValueError("reference batch is only valid for BatLiNet")
        return context, None
    if reference_library is None or context.reference_batch is None:
        raise ValueError("BatLiNet self-contained reference batch is missing")
    metadata = context.reference_batch
    reference_file = next(
        item
        for item in manifest.files
        if item.role is DeepArtifactFileRole.REFERENCE_BATCH
    )
    if (
        metadata.tensor_file_sha256 != reference_file.sha256
        or metadata.reference_library_sha256 != reference_library.library_sha256
        or metadata.cell_ids != reference_library.cell_ids
        or metadata.normalization_statistics_sha256
        != feature.normalization_sha256
    ):
        raise ValueError("reference batch metadata does not match BatLiNet context")
    reference_batch = _load_reference_batch(
        files[DeepArtifactFileRole.REFERENCE_BATCH],
        metadata,
    )
    return context, reference_batch


def _require_inference_network(
    network: torch.nn.Module,
    expected_type: type[torch.nn.Module],
) -> None:
    if not isinstance(network, expected_type):
        raise ValueError("advanced artifact network has an unexpected model type")
    if network.training:
        raise ValueError("advanced artifact export requires a trained network in eval mode")
    if not tuple(network.state_dict()):
        raise ValueError("advanced artifact network must contain trained tensor state")


def _cyclepatch_config_from(
    architecture: CyclePatchArchitecture
    | BatLiNetArchitecture
    | HybridPatchV2Architecture,
) -> CyclePatchConfig:
    return CyclePatchConfig(
        d_model=architecture.d_model,
        layers=architecture.layers,
        heads=architecture.heads,
        dropout=architecture.dropout,
        max_cycles=architecture.max_cycles,
    )


def _validate_condition_count(
    condition_count: int,
    condition_names: tuple[str, ...],
) -> None:
    if condition_count != len(condition_names):
        raise ValueError("condition_names do not match the approved architecture")


def _validate_expected_feature_hashes(
    feature: _AdvancedFeatureBase,
    *,
    expected_normalization_sha256: str,
    expected_candidate_config_sha256: str,
) -> None:
    if feature.normalization_sha256 != expected_normalization_sha256:
        raise ValueError("normalization_sha256 does not match the approved statistics")
    if feature.candidate_config_sha256 != expected_candidate_config_sha256:
        raise ValueError("candidate_config_sha256 does not match the approved candidate")


def _validate_scalar_scaler_context(
    scaler: CycleLifeTargetScalerConfig,
    cutoff_cycle: int,
) -> None:
    if scaler.cutoff_cycle != cutoff_cycle:
        raise ValueError("target scaler cutoff does not match artifact cutoff")


def _validate_target_scaler_context(
    scaler: CycleLifeTargetScalerConfig,
    expected_context_sha256: str,
) -> None:
    if scaler.context_sha256 != expected_context_sha256:
        raise ValueError("target_scaler_context_sha256 does not match the approved scaler")


def _validate_reference_scaler_context(
    library: CycleLifeReferenceLibraryConfig,
    scaler: CycleLifeTargetScalerConfig,
    expected_reference_count: int,
) -> None:
    if library.reference_count != expected_reference_count:
        raise ValueError("reference library count does not match model architecture")
    if library.scaler_context_sha256 != scaler.context_sha256:
        raise ValueError("reference library scaler context does not match target scaler")
    if library.training_cell_ids_sha256 != scaler.training_cell_ids_sha256:
        raise ValueError("reference library training cells do not match target scaler")
    if library.training_labels_sha256 != scaler.training_labels_sha256:
        raise ValueError("reference library training labels do not match target scaler")


def _output_target_for_kind(kind: DeepArtifactKind) -> AdvancedOutputTarget:
    if kind in {
        DeepArtifactKind.CYCLEPATCH_DIRECT,
        DeepArtifactKind.CYCLEPATCH_BATLINET,
    }:
        return AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE
    if kind in {
        DeepArtifactKind.CURRENT_HYBRID,
        DeepArtifactKind.HYBRIDPATCH_V2,
    }:
        return AdvancedOutputTarget.SOH_TRAJECTORY
    raise ValueError("artifact kind is not an Advanced deployment model")


def _optional_output_target(
    kind: DeepArtifactKind,
    output_target: AdvancedOutputTarget | str | None,
    normalizer: EarlyCycleNormalizer | None,
) -> AdvancedOutputTarget | None:
    if output_target is None and normalizer is None:
        return None
    if output_target is None or normalizer is None:
        raise ValueError("artifact v2 requires both normalizer and output_target")
    try:
        normalized = AdvancedOutputTarget(output_target)
    except ValueError as exc:
        raise ValueError("output_target is not supported") from exc
    if normalized is not _output_target_for_kind(kind):
        raise ValueError("artifact kind and output_target do not match")
    return normalized


def _validate_normalizer_context(
    normalizer: EarlyCycleNormalizer,
    *,
    dataset_id: str,
    data_version: str,
    feature_version: str,
    cutoff_cycle: int,
    condition_names: tuple[str, ...],
    normalization_sha256: str,
) -> EarlyCycleNormalizerArtifactConfig:
    config = EarlyCycleNormalizerArtifactConfig.from_runtime(normalizer)
    expected = (
        dataset_id,
        data_version,
        feature_version,
        cutoff_cycle,
        condition_names,
        normalization_sha256,
    )
    actual = (
        config.dataset_id,
        config.data_version,
        config.feature_version,
        config.cutoff_cycle,
        config.condition_names,
        config.statistics_sha256,
    )
    if actual != expected:
        raise ValueError("normalizer does not match the Advanced artifact context")
    return config


def _validate_advanced_batch(
    batch: EarlyCycleBatch,
    feature: _AdvancedFeatureBase,
) -> None:
    if batch.dataset_id != feature.dataset_id:
        raise ValueError("inference batch dataset_id does not match the artifact")
    if batch.data_version != feature.data_version:
        raise ValueError("inference batch data_version does not match the artifact")
    if batch.feature_version != feature.feature_version:
        raise ValueError("inference batch feature_version does not match the artifact")
    if batch.normalization_statistics_sha256 != feature.normalization_sha256:
        raise ValueError("inference batch normalization_sha256 does not match the artifact")
    if batch.condition_names != feature.condition_names:
        raise ValueError("inference batch condition_names do not match the artifact")
    if batch.cycle_mask.shape[1] != feature.cutoff_cycle + 1:
        raise ValueError("inference batch cutoff_cycle does not match the artifact")


def _export_deep_artifact(
    *,
    network: torch.nn.Module,
    artifact_root: Path,
    artifact_id: str,
    artifact_kind: DeepArtifactKind,
    architecture: dict[str, Any],
    feature_config: dict[str, Any],
    created_at: datetime,
    reference_library: dict[str, Any] | None = None,
    normalizer: EarlyCycleNormalizer | None = None,
    output_target: AdvancedOutputTarget | None = None,
    reference_batch: EarlyCycleBatch | None = None,
    reference_library_sha256: str | None = None,
) -> DeepModelArtifactManifest:
    try:
        normalized_id = str(UUID(artifact_id))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("artifact_id must be a UUID string") from exc
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("created_at must include a timezone")
    requires_reference = artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET
    if requires_reference != (reference_library is not None):
        raise ValueError("reference library presence does not match artifact kind")
    is_v2 = normalizer is not None or output_target is not None
    if is_v2 != (normalizer is not None and output_target is not None):
        raise ValueError("artifact v2 inference context is incomplete")
    if is_v2:
        assert normalizer is not None and output_target is not None
        _validate_normalizer_context(
            normalizer,
            dataset_id=str(feature_config.get("dataset_id", "")),
            data_version=str(feature_config.get("data_version", "")),
            feature_version=str(feature_config.get("feature_version", "")),
            cutoff_cycle=int(feature_config.get("cutoff_cycle", -1)),
            condition_names=tuple(feature_config.get("condition_names", ())),
            normalization_sha256=str(
                feature_config.get("normalization_sha256", "")
            ),
        )
        if requires_reference != (reference_batch is not None):
            raise ValueError("reference batch presence does not match artifact kind")
    elif reference_batch is not None:
        raise ValueError("reference batch requires artifact v2")
    root = _validated_root(artifact_root)
    target = root / normalized_id
    if target.exists() or target.is_symlink():
        raise ValueError("artifact target already exists")
    temporary = root / f".{normalized_id}.{os.getpid()}.tmp"
    if temporary.exists():
        raise ValueError("temporary artifact target already exists")
    temporary.mkdir()
    try:
        weights_path = temporary / "model.safetensors"
        architecture_path = temporary / "architecture.json"
        feature_path = temporary / "feature_config.json"
        reference_path = temporary / "reference_library.json"
        inference_context_path = temporary / "inference_context.json"
        reference_batch_path = temporary / "reference_batch.safetensors"
        _save_safetensors(network, weights_path)
        _write_json(architecture_path, architecture)
        registered_files = []
        if reference_library is not None:
            _write_json(reference_path, reference_library)
        reference_metadata = None
        if reference_batch is not None:
            if reference_library_sha256 is None:
                raise ValueError("reference batch requires reference library SHA-256")
            _save_reference_batch(reference_batch, reference_batch_path)
            reference_batch_sha256 = _sha256_file(reference_batch_path)
            reference_metadata = BatLiNetReferenceBatchMetadata.model_validate(
                {
                    "dataset_id": reference_batch.dataset_id,
                    "data_version": reference_batch.data_version,
                    "feature_version": reference_batch.feature_version,
                    "normalization_statistics_sha256": (
                        reference_batch.normalization_statistics_sha256
                    ),
                    "cell_ids": reference_batch.cell_ids,
                    "condition_names": reference_batch.condition_names,
                    "reference_library_sha256": reference_library_sha256,
                    "tensor_file_sha256": reference_batch_sha256,
                }
            )
        if is_v2:
            assert normalizer is not None and output_target is not None
            context_payload = {
                "schema_version": "advanced-inference-context-v1",
                "output_target": output_target.value,
                "normalizer": EarlyCycleNormalizerArtifactConfig.from_runtime(
                    normalizer
                ).model_dump(mode="json"),
                "reference_batch": (
                    None
                    if reference_metadata is None
                    else reference_metadata.model_dump(mode="json")
                ),
            }
            inference_context = AdvancedInferenceContext.model_validate(
                {
                    **context_payload,
                    "inference_context_sha256": sha256_canonical(context_payload),
                }
            )
            _write_json(
                inference_context_path,
                inference_context.model_dump(mode="json"),
            )
            feature_config = {
                **feature_config,
                "inference_context_sha256": (
                    inference_context.inference_context_sha256
                ),
            }
        _write_json(feature_path, feature_config)
        registered_files.extend(
            [
                (DeepArtifactFileRole.WEIGHTS, weights_path),
                (DeepArtifactFileRole.ARCHITECTURE, architecture_path),
                (DeepArtifactFileRole.FEATURE_CONFIG, feature_path),
            ]
        )
        if reference_library is not None:
            registered_files.append(
                (DeepArtifactFileRole.REFERENCE_LIBRARY, reference_path)
            )
        if is_v2:
            registered_files.append(
                (DeepArtifactFileRole.INFERENCE_CONTEXT, inference_context_path)
            )
        if reference_batch is not None:
            registered_files.append(
                (DeepArtifactFileRole.REFERENCE_BATCH, reference_batch_path)
            )
        files = tuple(
            DeepArtifactFile(
                role=role,
                relative_path=f"{normalized_id}/{path.name}",
                size_bytes=path.stat().st_size,
                sha256=_sha256_file(path),
            )
            for role, path in registered_files
        )
        manifest_payload = {
            "schema_version": (
                "deep-model-artifact-v2" if is_v2 else "deep-model-artifact-v1"
            ),
            "artifact_id": normalized_id,
            "artifact_kind": artifact_kind.value,
            "files": [item.model_dump(mode="json") for item in files],
            "created_at": created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        }
        if output_target is not None:
            manifest_payload["output_target"] = output_target.value
        manifest = DeepModelArtifactManifest.model_validate(
            {
                **manifest_payload,
                "manifest_sha256": sha256_canonical(manifest_payload),
            }
        )
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        temporary.replace(target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_deep_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> dict[DeepArtifactFileRole, Path]:
    root = _validated_root(artifact_root)
    payload = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    accepted_hashes = {sha256_canonical(payload)}
    if manifest.schema_version == "deep-model-artifact-v1":
        legacy_payload = {
            key: value for key, value in payload.items() if key != "output_target"
        }
        accepted_hashes.add(sha256_canonical(legacy_payload))
    if manifest.manifest_sha256 not in accepted_hashes:
        raise ValueError("manifest SHA-256 does not match its contents")
    artifact_directory = root / manifest.artifact_id
    if artifact_directory.is_symlink():
        raise ValueError("artifact directory must not be a symbolic link")
    resolved_directory = artifact_directory.resolve(strict=True)
    if not resolved_directory.is_relative_to(root) or not resolved_directory.is_dir():
        raise ValueError("artifact directory must remain inside the artifact root")
    manifest_path = resolved_directory / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("artifact manifest must not be a symbolic link")
    disk_manifest = DeepModelArtifactManifest.model_validate(_read_strict_json(manifest_path))
    if disk_manifest != manifest:
        raise ValueError("on-disk artifact manifest does not match the registered manifest")

    expected_names = {"manifest.json", *(Path(item.relative_path).name for item in manifest.files)}
    actual_names = {item.name for item in resolved_directory.iterdir()}
    if actual_names != expected_names:
        raise ValueError("artifact directory contains an unexpected file")

    verified: dict[DeepArtifactFileRole, Path] = {}
    for item in manifest.files:
        path = root / item.relative_path
        if path.is_symlink():
            raise ValueError("artifact file must not be a symbolic link")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(resolved_directory) or not resolved.is_file():
            raise ValueError("artifact file must remain inside its artifact directory")
        if resolved.stat().st_size != item.size_bytes:
            raise ValueError("artifact file size does not match the manifest")
        if _sha256_file(resolved) != item.sha256:
            raise ValueError("artifact file SHA-256 does not match the manifest")
        verified[item.role] = resolved
    return verified


def verify_deep_model_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> None:
    """Reverify every registered byte without constructing a model."""

    _verify_deep_artifact(artifact_root, manifest)


def verify_self_contained_advanced_artifact(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> AdvancedInferenceContext:
    """Verify that an Advanced artifact contains all preprocessing state."""

    if manifest.schema_version != "deep-model-artifact-v2":
        raise ValueError("self-contained Advanced runtime requires artifact v2")
    files = _verify_deep_artifact(artifact_root, manifest)
    feature = _advanced_feature_from_files(files, manifest.artifact_kind)
    reference_library = None
    if manifest.artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
        reference_library = CycleLifeReferenceLibraryConfig.model_validate(
            _read_strict_json(files[DeepArtifactFileRole.REFERENCE_LIBRARY])
        )
    context, _reference_batch = _load_self_contained_inference_context(
        files,
        manifest,
        feature,
        reference_library=reference_library,
    )
    return context


def _save_reference_batch(batch: EarlyCycleBatch, path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for reference batch export") from exc
    tensors = {
        "values": batch.values.detach().cpu().contiguous(),
        "cycle_indices": batch.cycle_indices.detach().cpu().contiguous(),
        "cycle_mask": batch.cycle_mask.detach().cpu().contiguous(),
        "sample_mask": batch.sample_mask.detach().cpu().contiguous(),
        "condition_values": batch.condition_values.detach().cpu().contiguous(),
        "condition_mask": batch.condition_mask.detach().cpu().contiguous(),
    }
    save_file(tensors, str(path))


def _load_reference_batch(
    path: Path,
    metadata: BatLiNetReferenceBatchMetadata,
) -> EarlyCycleBatch:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for reference batch loading") from exc
    tensors = load_file(str(path), device="cpu")
    expected = {
        "values",
        "cycle_indices",
        "cycle_mask",
        "sample_mask",
        "condition_values",
        "condition_mask",
    }
    if set(tensors) != expected:
        raise ValueError("reference batch safetensors keys are invalid")
    try:
        return EarlyCycleBatch(
            dataset_id=metadata.dataset_id,
            data_version=metadata.data_version,
            feature_version=metadata.feature_version,
            normalization_statistics_sha256=(
                metadata.normalization_statistics_sha256
            ),
            cell_ids=metadata.cell_ids,
            condition_names=metadata.condition_names,
            values=tensors["values"],
            cycle_indices=tensors["cycle_indices"],
            cycle_mask=tensors["cycle_mask"],
            sample_mask=tensors["sample_mask"],
            condition_values=tensors["condition_values"],
            condition_mask=tensors["condition_mask"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("reference batch tensors are invalid") from exc


def _save_safetensors(network: torch.nn.Module, path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for deep model export") from exc
    state: dict[str, Tensor] = {}
    for name, tensor in network.state_dict().items():
        value = tensor.detach().to(device="cpu").contiguous()
        if value.dtype is not torch.float32:
            raise ValueError("deep model weights must use float32")
        if not torch.isfinite(value).all():
            raise ValueError("deep model weights must be finite")
        state[name] = value
    save_file(state, str(path))


def _load_verified_state(network: torch.nn.Module, path: Path) -> None:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required for deep model loading") from exc
    state = load_file(str(path), device="cpu")
    expected = network.state_dict()
    if set(state) != set(expected):
        raise ValueError("safetensors keys do not exactly match the approved architecture")
    for name, value in state.items():
        if value.dtype is not torch.float32:
            raise ValueError("safetensors weights must use float32")
        if tuple(value.shape) != tuple(expected[name].shape):
            raise ValueError("safetensors weight shape does not match the approved architecture")
        if not torch.isfinite(value).all():
            raise ValueError("safetensors weights must be finite")
    try:
        network.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "safetensors state is incompatible with the approved architecture"
        ) from exc


def _validated_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("artifact root must exist") from exc
    if root.is_symlink() or not resolved.is_dir():
        raise ValueError("artifact root must be a regular directory")
    return resolved


def _read_strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact JSON must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
