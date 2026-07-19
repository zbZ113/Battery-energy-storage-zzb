"""Strict configuration and evidence contracts for advanced MATR model selection."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, TypeAdapter, field_validator, model_validator

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

AdvancedFamily = Literal[
    "cyclepatch_direct",
    "cyclepatch_batlinet",
    "current_hybrid",
    "hybridpatch_v2",
]


def _safe_posix_relative(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe repository-relative POSIX path")
    return value


def _candidate_id(value: str) -> str:
    _safe_posix_relative(value, "candidate_id")
    if len(PurePosixPath(value).parts) != 1 or re.fullmatch(
        r"[a-z0-9][a-z0-9._-]{0,95}", value
    ) is None:
        raise ValueError("candidate_id must be a single safe lowercase identifier")
    return value


class _CandidateIdentityBase(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    learning_rate: float = Field(ge=1e-6, le=1e-2)

    _candidate_id_is_safe = field_validator("candidate_id")(_candidate_id)

    @property
    def config_sha256(self) -> str:
        return sha256_canonical(self.model_dump(mode="json"))


class _RegularizedCandidateBase(_CandidateIdentityBase):
    weight_decay: float = Field(ge=0.0, le=0.2)


class CyclePatchDirectCandidate(_RegularizedCandidateBase):
    family: Literal["cyclepatch_direct"] = "cyclepatch_direct"
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float = Field(ge=0.05, le=0.1)
    huber_delta: float = Field(gt=0.0, le=5.0)

    @model_validator(mode="after")
    def attention_width_is_valid(self) -> CyclePatchDirectCandidate:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("dropout must be exactly 0.05 or 0.1")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self


class CyclePatchBatLiNetCandidate(_RegularizedCandidateBase):
    family: Literal["cyclepatch_batlinet"] = "cyclepatch_batlinet"
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float = Field(ge=0.05, le=0.1)
    lambda_pair: float = Field(ge=0.25, le=1.0)
    lambda_rank: float = Field(ge=0.0, le=0.1)
    reference_count: Literal[16, 32, 64]
    fusion_alpha: float = Field(ge=0.25, le=0.75)
    huber_delta: float = Field(gt=0.0, le=5.0)

    @model_validator(mode="after")
    def attention_width_is_valid(self) -> CyclePatchBatLiNetCandidate:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("dropout must be exactly 0.05 or 0.1")
        if self.lambda_pair not in {0.25, 0.5, 1.0}:
            raise ValueError("lambda_pair must be 0.25, 0.5 or 1.0")
        if self.lambda_rank not in {0.0, 0.1}:
            raise ValueError("lambda_rank must be 0 or 0.1")
        if self.fusion_alpha not in {0.25, 0.5, 0.75}:
            raise ValueError("fusion_alpha must be 0.25, 0.5 or 0.75")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self


class CurrentHybridCandidate(_CandidateIdentityBase):
    family: Literal["current_hybrid"] = "current_hybrid"
    hidden_dim: Literal[32, 64, 128]


class HybridPatchV2Candidate(_RegularizedCandidateBase):
    family: Literal["hybridpatch_v2"] = "hybridpatch_v2"
    d_model: Literal[128, 256]
    layers: Literal[2, 4]
    heads: Literal[4, 8]
    dropout: float = Field(ge=0.05, le=0.1)
    query_token_count: Literal[0, 8, 16]
    query_layers: Literal[1, 2, 3]
    decoder_hidden_dim: Literal[32, 64, 128]
    huber_delta: float = Field(gt=0.0, le=5.0)
    lambda_history: float = Field(ge=0.0, le=0.1)
    lambda_smooth: float = Field(ge=0.0, le=0.01)
    lambda_order: float = Field(ge=0.0, le=0.05)
    lambda_residual: float = Field(ge=0.001, le=0.01)

    @model_validator(mode="after")
    def attention_width_is_valid(self) -> HybridPatchV2Candidate:
        if self.dropout not in {0.05, 0.1}:
            raise ValueError("dropout must be exactly 0.05 or 0.1")
        if self.lambda_history not in {0.0, 0.1}:
            raise ValueError("lambda_history must be 0 or 0.1")
        if self.lambda_smooth not in {0.0, 0.01}:
            raise ValueError("lambda_smooth must be 0 or 0.01")
        if self.lambda_order not in {0.0, 0.05}:
            raise ValueError("lambda_order must be 0 or 0.05")
        if self.lambda_residual not in {0.001, 0.01}:
            raise ValueError("lambda_residual must be 0.001 or 0.01")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        return self


AdvancedCandidate = Annotated[
    CyclePatchDirectCandidate
    | CyclePatchBatLiNetCandidate
    | CurrentHybridCandidate
    | HybridPatchV2Candidate,
    Field(discriminator="family"),
]


class AdvancedSelectionPolicy(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage1_epochs: Literal[30] = 30
    stage2_epochs: Literal[90] = 90
    recheck_epochs: Literal[90] = 90
    keep_fraction: float = Field(default=0.5, ge=0.5, le=0.5)
    per_family_finalists: Literal[2] = 2
    initial_cutoff: Literal[100] = 100
    initial_seed: Literal[38] = 38
    recheck_seeds: tuple[Literal[38, 39, 40], ...] = (38, 39, 40)
    recheck_cutoffs: tuple[Literal[20, 50, 100, 150], ...] = (20, 50, 100, 150)

    @model_validator(mode="after")
    def protocol_is_exact(self) -> AdvancedSelectionPolicy:
        if self.keep_fraction != 0.5:
            raise ValueError("keep_fraction must be exactly 0.5")
        if self.recheck_seeds != (38, 39, 40):
            raise ValueError("recheck_seeds must be exactly 38, 39 and 40")
        if self.recheck_cutoffs != (20, 50, 100, 150):
            raise ValueError("recheck_cutoffs must be exactly 20, 50, 100 and 150")
        return self


class AdvancedTrainingPaths(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    three_batch_manifest: str
    split_manifest: str
    run_root: str

    @field_validator("*")
    @classmethod
    def paths_are_safe(cls, value: str) -> str:
        return _safe_posix_relative(value, "advanced training path")


class AdvancedMatrThreeBatchRunConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-matr-three-batch-run-v1"] = (
        "advanced-matr-three-batch-run-v1"
    )
    mode: Literal["smoke", "select", "final"]
    dataset_id: Literal["MATR"] = "MATR"
    target: Literal[PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE] = (
        PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    )
    physical_gpu_index: Literal[1] = 1
    precision: Literal["fp32"] = "fp32"
    seeds: tuple[int, ...]
    cutoffs: tuple[int, ...]
    max_epochs: int = Field(gt=0)
    paths: AdvancedTrainingPaths
    candidates: tuple[AdvancedCandidate, ...] = Field(min_length=4)
    selection_policy: AdvancedSelectionPolicy = Field(
        default_factory=AdvancedSelectionPolicy
    )
    selection_manifest_path: str | None = None
    expected_selection_sha256: Sha256 | None = None

    @property
    def config_sha256(self) -> str:
        return sha256_canonical(self.model_dump(mode="json"))

    @field_validator("selection_manifest_path")
    @classmethod
    def selection_path_is_safe(cls, value: str | None) -> str | None:
        if value is not None:
            return _safe_posix_relative(value, "selection_manifest_path")
        return value

    @model_validator(mode="after")
    def mode_is_exact(self) -> AdvancedMatrThreeBatchRunConfig:
        expected_axes = {
            "smoke": ((38,), (50,)),
            "select": ((38, 39, 40), (20, 50, 100, 150)),
            "final": ((38, 39, 40, 41, 42), (20, 50, 100, 150)),
        }
        expected_seeds, expected_cutoffs = expected_axes[self.mode]
        if self.seeds != expected_seeds or self.cutoffs != expected_cutoffs:
            raise ValueError(f"{self.mode} seeds and cutoffs must match the approved matrix")
        if self.mode == "smoke" and self.max_epochs != 10:
            raise ValueError("smoke max_epochs must be exactly 10")
        if self.mode == "select":
            if (
                self.selection_manifest_path is not None
                or self.expected_selection_sha256 is not None
            ):
                raise ValueError("selection runs cannot consume a selection manifest")
        elif self.mode == "final":
            if self.selection_manifest_path is None:
                raise ValueError("final run templates require a selection manifest path")
            families = [candidate.family for candidate in self.candidates]
            if len(self.candidates) != 4 or set(families) != {
                "cyclepatch_direct",
                "cyclepatch_batlinet",
                "current_hybrid",
                "hybridpatch_v2",
            }:
                raise ValueError("final runs require exactly one selected candidate per family")
        elif self.selection_manifest_path is not None or self.expected_selection_sha256 is not None:
            raise ValueError("smoke runs cannot consume a selection manifest")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique")
        return self


class AdvancedCandidateSearchConfig(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-candidate-search-v1"] = (
        "advanced-candidate-search-v1"
    )
    family: Literal["cyclepatch_direct", "cyclepatch_batlinet", "hybridpatch_v2"]
    candidates: tuple[AdvancedCandidate, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def candidates_match_family(self) -> AdvancedCandidateSearchConfig:
        if any(candidate.family != self.family for candidate in self.candidates):
            raise ValueError("every search candidate must match the declared family")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("search candidate_id values must be unique")
        return self


class AdvancedSelectionEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: AdvancedFamily
    candidate_id: str
    candidate_config_sha256: Sha256
    stage: Literal["selection_recheck"] = "selection_recheck"
    cutoff_cycle: int
    seed: Literal[38, 39, 40]
    validation_mae: float = Field(ge=0.0)

    _candidate_id_is_safe = field_validator("candidate_id")(_candidate_id)

    @field_validator("cutoff_cycle")
    @classmethod
    def cutoff_is_approved(cls, value: int) -> int:
        if value not in {20, 50, 100, 150}:
            raise ValueError("cutoff_cycle must be 20, 50, 100 or 150")
        return value

    @model_validator(mode="after")
    def metrics_are_finite(self) -> AdvancedSelectionEvidence:
        if not math.isfinite(self.validation_mae):
            raise ValueError("validation metrics must be finite")
        return self


class AdvancedSelectionManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-selection-manifest-v1"] = (
        "advanced-selection-manifest-v1"
    )
    input_bundle_sha256: Sha256
    data_sha256: Sha256
    split_sha256: Sha256
    feature_sha256: Sha256
    normalization_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    validation_cell_sha256: Sha256
    validation_results: tuple[AdvancedSelectionEvidence, ...] = Field(min_length=4)
    selected_candidates: tuple[AdvancedCandidate, ...] = Field(min_length=4, max_length=4)
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def manifest_is_bound_and_complete(self) -> AdvancedSelectionManifest:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if sha256_canonical(payload) != self.manifest_sha256:
            raise ValueError("manifest_sha256 does not match selection contents")
        selected = {candidate.family: candidate for candidate in self.selected_candidates}
        if set(selected) != {
            "cyclepatch_direct",
            "cyclepatch_batlinet",
            "current_hybrid",
            "hybridpatch_v2",
        }:
            raise ValueError("selection manifest requires one candidate per family")
        evidence_pairs = {
            (
                item.family,
                item.candidate_id,
                item.candidate_config_sha256,
                item.cutoff_cycle,
                item.seed,
            )
            for item in self.validation_results
        }
        if len(evidence_pairs) != len(self.validation_results):
            raise ValueError("selection recheck evidence keys must be unique")
        recheck_axes = {
            (cutoff, seed)
            for cutoff in (20, 50, 100, 150)
            for seed in (38, 39, 40)
        }
        for candidate in self.selected_candidates:
            observed_axes = {
                (cutoff, seed)
                for family, candidate_id, config_hash, cutoff, seed in evidence_pairs
                if family == candidate.family
                and candidate_id == candidate.candidate_id
                and config_hash == candidate.config_sha256
            }
            if observed_axes != recheck_axes:
                raise ValueError(
                    "every selected candidate requires a complete 12-run recheck grid"
                )
        return self


class AdvancedRunKey(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: AdvancedFamily
    candidate_id: str
    cutoff_cycle: int
    seed: int
    stage: Literal[
        "smoke", "selection_stage1", "selection_stage2", "selection_recheck", "final"
    ]
    max_epochs: int = Field(gt=0)
    candidate_config_sha256: Sha256

    _candidate_id_is_safe = field_validator("candidate_id")(_candidate_id)

    @field_validator("cutoff_cycle")
    @classmethod
    def cutoff_is_approved(cls, value: int) -> int:
        if value not in {20, 50, 100, 150}:
            raise ValueError("cutoff_cycle must be 20, 50, 100 or 150")
        return value


def build_advanced_run_matrix(
    config: AdvancedMatrThreeBatchRunConfig,
    *,
    selection_stage: Literal[
        "selection_stage1", "selection_stage2", "selection_recheck"
    ] = "selection_stage1",
    candidate_ids: tuple[str, ...] = (),
    repository_root: Path | None = None,
) -> tuple[AdvancedRunKey, ...]:
    config = AdvancedMatrThreeBatchRunConfig.model_validate(
        config.model_dump(mode="python")
    )
    if config.mode == "select":
        candidates = config.candidates
        axes: tuple[tuple[int, int], ...]
        max_epochs: int
        if selection_stage == "selection_stage1":
            if candidate_ids:
                raise ValueError("selection stage1 always evaluates every candidate")
            axes = ((config.selection_policy.initial_cutoff, config.selection_policy.initial_seed),)
            max_epochs = config.selection_policy.stage1_epochs
        else:
            if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
                raise ValueError("later selection stages require unique candidate_ids")
            by_id = {candidate.candidate_id: candidate for candidate in candidates}
            if any(candidate_id not in by_id for candidate_id in candidate_ids):
                raise ValueError("selection candidate_ids must belong to the configured search")
            candidates = tuple(by_id[candidate_id] for candidate_id in candidate_ids)
            if selection_stage == "selection_stage2":
                axes = (
                    (
                        config.selection_policy.initial_cutoff,
                        config.selection_policy.initial_seed,
                    ),
                )
                max_epochs = config.selection_policy.stage2_epochs
            else:
                axes = tuple(
                    (cutoff, seed)
                    for cutoff in config.selection_policy.recheck_cutoffs
                    for seed in config.selection_policy.recheck_seeds
                )
                max_epochs = config.selection_policy.recheck_epochs
        return tuple(
            AdvancedRunKey(
                family=candidate.family,
                candidate_id=candidate.candidate_id,
                cutoff_cycle=cutoff,
                seed=seed,
                stage=selection_stage,
                max_epochs=max_epochs,
                candidate_config_sha256=candidate.config_sha256,
            )
            for candidate in candidates
            for cutoff, seed in axes
        )
    if candidate_ids or selection_stage != "selection_stage1":
        raise ValueError("selection stage arguments are valid only in select mode")
    if config.mode == "final":
        _validate_final_selection_binding(config, repository_root)
    stage: Literal["smoke", "final"] = config.mode
    return tuple(
        AdvancedRunKey(
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            cutoff_cycle=cutoff,
            seed=seed,
            stage=stage,
            max_epochs=config.max_epochs,
            candidate_config_sha256=candidate.config_sha256,
        )
        for candidate in config.candidates
        for cutoff in config.cutoffs
        for seed in config.seeds
    )


def _validate_final_selection_binding(
    config: AdvancedMatrThreeBatchRunConfig,
    repository_root: Path | None,
) -> None:
    if config.expected_selection_sha256 is None:
        raise ValueError("final config is a template until selection SHA-256 is resolved")
    if repository_root is None or config.selection_manifest_path is None:
        raise ValueError("final matrix requires a repository root and selection manifest")
    root = repository_root.resolve(strict=True)
    path = (root / PurePosixPath(config.selection_manifest_path)).resolve(strict=True)
    if not path.is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError("selection manifest must be a regular file inside repository_root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != config.expected_selection_sha256:
        raise ValueError("selection manifest file SHA-256 does not match final config")
    manifest = AdvancedSelectionManifest.model_validate(_strict_json_object(payload))
    configured = {
        candidate.family: candidate.model_dump(mode="json")
        for candidate in config.candidates
    }
    selected = {
        candidate.family: candidate.model_dump(mode="json")
        for candidate in manifest.selected_candidates
    }
    if configured != selected:
        raise ValueError("final candidates do not exactly match the selection manifest")


def _strict_json_object(payload: bytes) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            payload,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("selection manifest JSON is invalid") from exc
    if not isinstance(parsed, dict):
        raise ValueError("selection manifest must contain a JSON object")
    return parsed


ADVANCED_CANDIDATE_ADAPTER: TypeAdapter[AdvancedCandidate] = TypeAdapter(
    AdvancedCandidate
)
