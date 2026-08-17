"""Server-reviewed default BLAST scenario profiles for proactive Feishu analysis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Literal, Self
from uuid import NAMESPACE_URL, uuid5

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.scenarios import (
    BlastRouteManifest,
    BlastRouteRejected,
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSupportStatus,
    VerifiedScenarioContext,
    assess_operation_scenario,
    load_packaged_blast_route_catalog,
)
from quanxin_life.tools.blast_scenarios import CompareOperationScenariosToolInput
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

from .workflow import FeishuAnalysisTask

_MAX_PROFILE_BYTES = 1_048_576
_SCHEMA_VERSION = "feishu-default-scenario-registry-v1"
_CREATED_BY_REFERENCE_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z"
)


class ReviewedDefaultScenarioProfile(ContractModel):
    """One immutable default scenario bound to an exact reviewed batch identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(min_length=1, max_length=200)
    profile_version: str = Field(min_length=1, max_length=100)
    review_status: Literal["APPROVED"]
    task_type: Literal["compare_operation_scenarios"]
    dataset_id: str = Field(min_length=1, max_length=200)
    cell_id: str = Field(min_length=1, max_length=200)
    data_version: str = Field(min_length=1, max_length=200)
    split_version: str = Field(min_length=1, max_length=200)
    feature_version: str = Field(min_length=1, max_length=200)
    allowed_cutoff_cycles: tuple[int, ...] = Field(min_length=1)
    allowed_source_sha256s: tuple[Sha256, ...] = Field(min_length=1)
    chemistry: str = Field(min_length=1, max_length=100)
    nominal_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    source_cell_format: Literal["cylindrical", "prismatic"]
    reference_cell_format: Literal["cylindrical", "prismatic"]
    route_id: str = Field(min_length=1, max_length=200)
    trusted_reference_use: bool
    reference_use_reason_code: str = Field(min_length=1, max_length=100)
    initial_state_policy: Literal["BOL_ONLY"]
    baseline: OperationScenario
    comparisons: tuple[OperationScenario, ...] = Field(min_length=1, max_length=8)

    @field_validator(
        "profile_id",
        "profile_version",
        "dataset_id",
        "cell_id",
        "data_version",
        "split_version",
        "feature_version",
        "chemistry",
        "route_id",
        "reference_use_reason_code",
    )
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("default scenario identifiers must be safe nonblank text")
        return normalized

    @model_validator(mode="after")
    def profile_is_supported_and_explicit(self) -> Self:
        if not _CREATED_BY_REFERENCE_PATTERN.fullmatch(self.created_by_reference):
            raise ValueError("default scenario created-by reference must be safe ASCII")
        if len(set(self.allowed_cutoff_cycles)) != len(self.allowed_cutoff_cycles):
            raise ValueError("allowed cutoff cycles must be unique")
        if any(value < 1 for value in self.allowed_cutoff_cycles):
            raise ValueError("allowed cutoff cycles must be positive")
        if len(set(self.allowed_source_sha256s)) != len(self.allowed_source_sha256s):
            raise ValueError("allowed source SHA-256 values must be unique")
        scenario_ids = [
            self.baseline.scenario_id,
            *(item.scenario_id for item in self.comparisons),
        ]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("default scenario IDs must be unique")
        catalog = load_packaged_blast_route_catalog()
        route = next(
            (item for item in catalog.routes if item.route_id == self.route_id),
            None,
        )
        if route is None or self.task_type not in route.task_types:
            raise ValueError("default scenario route is not registered for the task")
        if route.cell_format != self.reference_cell_format:
            raise ValueError("default scenario reference format does not match its route")
        mismatch = (
            self.source_cell_format != self.reference_cell_format
            or not math.isclose(
                self.nominal_capacity_ah,
                route.nominal_capacity_reference_ah,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        if mismatch and not self.trusted_reference_use:
            raise ValueError("default scenario reference use is not approved")
        try:
            catalog.authorize_reference_use(
                route_id=self.route_id,
                chemistry=self.chemistry,
                nominal_capacity_ah=self.nominal_capacity_ah,
                cell_format=self.source_cell_format,
                trusted_reference_use=self.trusted_reference_use,
            )
        except BlastRouteRejected as exc:
            raise ValueError("default scenario reference use is invalid") from exc
        for scenario in (self.baseline, *self.comparisons):
            if assess_operation_scenario(route, scenario).status is ScenarioSupportStatus.REJECTED:
                raise ValueError("default scenario is outside the route support boundary")
        return self

    @property
    def profile_sha256(self) -> str:
        return sha256_canonical(self.model_dump(mode="json"))

    @property
    def created_by_reference(self) -> str:
        return f"feishu-default-scenario:{self.profile_id}:{self.profile_version}"


@dataclass(frozen=True, slots=True)
class DefaultScenarioContextTemplate:
    task: FeishuAnalysisTask
    scenario_context_id: str
    verified_context: VerifiedScenarioContext
    analysis_input: CompareOperationScenariosToolInput
    created_by_reference: str
    profile_id: str
    profile_version: str
    profile_sha256: str


class ReviewedDefaultScenarioRegistry:
    """Resolve at most one approved profile for an exact trusted batch."""

    def __init__(
        self,
        profiles: tuple[ReviewedDefaultScenarioProfile, ...] = (),
        *,
        file_sha256: str | None = None,
    ) -> None:
        validated = tuple(
            ReviewedDefaultScenarioProfile.model_validate(
                item.model_dump(mode="json")
            )
            for item in profiles
        )
        profile_identities = tuple(
            (item.profile_id, item.profile_version) for item in validated
        )
        if len(profile_identities) != len(set(profile_identities)):
            raise ValueError("default scenario profile identity must be unique")
        identities = tuple(
            (
                item.dataset_id,
                item.cell_id,
                item.data_version,
                item.split_version,
                item.feature_version,
                tuple(sorted(item.allowed_cutoff_cycles)),
                tuple(sorted(item.allowed_source_sha256s)),
            )
            for item in validated
        )
        if len(identities) != len(set(identities)):
            raise ValueError("default scenario profile match identities must be unique")
        for first, second in combinations(validated, 2):
            same_fixed_identity = (
                first.dataset_id == second.dataset_id
                and first.cell_id == second.cell_id
                and first.data_version == second.data_version
                and first.split_version == second.split_version
                and first.feature_version == second.feature_version
                and first.chemistry.casefold() == second.chemistry.casefold()
                and math.isclose(
                    first.nominal_capacity_ah,
                    second.nominal_capacity_ah,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            )
            cutoff_overlap = set(first.allowed_cutoff_cycles).intersection(
                second.allowed_cutoff_cycles
            )
            sha_overlap = set(first.allowed_source_sha256s).intersection(
                second.allowed_source_sha256s
            )
            if same_fixed_identity and cutoff_overlap and sha_overlap:
                raise ValueError("default scenario profile match sets must not overlap")
        self._profiles = validated
        self._file_sha256 = file_sha256

    @property
    def profiles(self) -> tuple[ReviewedDefaultScenarioProfile, ...]:
        return self._profiles

    @property
    def file_sha256(self) -> str | None:
        return self._file_sha256

    def resolve(
        self,
        batch: VerifiedEarlyCycleBatch,
    ) -> ReviewedDefaultScenarioProfile | None:
        checked = VerifiedEarlyCycleBatch.model_validate(batch.model_dump(mode="json"))
        matches = tuple(
            profile
            for profile in self._profiles
            if (
                profile.dataset_id == checked.metadata.dataset_id
                and profile.cell_id == checked.metadata.cell_id
                and profile.data_version == checked.data_version
                and profile.split_version == checked.split_version
                and profile.feature_version == checked.feature_config.feature_version
                and profile.chemistry.casefold()
                == checked.metadata.chemistry.casefold()
                and math.isclose(
                    profile.nominal_capacity_ah,
                    checked.metadata.nominal_capacity_ah,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                and checked.feature_config.cutoff_cycle
                in profile.allowed_cutoff_cycles
                and checked.metadata.source_sha256
                in profile.allowed_source_sha256s
            )
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("default scenario profile match is not unique")
        return matches[0]

    def authorize_reference_use(
        self,
        *,
        task: FeishuAnalysisTask,
        data_batch_id: str,
        batch: VerifiedEarlyCycleBatch,
        route: BlastRouteManifest,
    ) -> bool:
        """Authorize Aily reference use only from the exact reviewed profile."""

        checked = VerifiedEarlyCycleBatch.model_validate(batch.model_dump(mode="json"))
        profile = self.resolve(checked)
        if (
            data_batch_id != checked.record_batch_id
            or profile is None
            or profile.task_type != task.value
            or profile.route_id != route.route_id
            or profile.reference_cell_format != route.cell_format
        ):
            raise ValueError("reviewed default scenario reference use is not authorized")
        return profile.trusted_reference_use

    def create_context_template(
        self,
        *,
        batch: VerifiedEarlyCycleBatch,
        profile: ReviewedDefaultScenarioProfile,
        source_job_id: str,
    ) -> DefaultScenarioContextTemplate:
        checked_batch = VerifiedEarlyCycleBatch.model_validate(
            batch.model_dump(mode="json")
        )
        checked_profile = self.resolve(checked_batch)
        if checked_profile is None or checked_profile != profile:
            raise ValueError("default scenario profile is not bound to the batch")
        context_id = str(
            uuid5(
                NAMESPACE_URL,
                sha256_canonical(
                    {
                        "schema_version": "feishu-default-scenario-context-v1",
                        "source_job_id": source_job_id,
                        "record_batch_id": checked_batch.record_batch_id,
                        "profile_sha256": checked_profile.profile_sha256,
                    }
                ),
            )
        )
        cell = ScenarioCellDescriptor(
            chemistry=checked_batch.metadata.chemistry,
            nominal_capacity_ah=checked_batch.metadata.nominal_capacity_ah,
            cell_format=checked_profile.source_cell_format,
        )
        profile_provenance = ProvenanceRecord(
            source_id=checked_profile.profile_id,
            source_kind=SourceKind.SIMULATED,
            uri=(
                "configuration://feishu/default-scenarios/"
                f"{checked_profile.profile_id}/{checked_profile.profile_version}"
            ),
            sha256=checked_profile.profile_sha256,
            description=(
                "Server-reviewed default BLAST reference-use scenario profile; "
                f"reason={checked_profile.reference_use_reason_code}."
            ),
            created_at=_profile_provenance_time(checked_batch),
        )
        verified_context = VerifiedScenarioContext(
            scenario_context_id=context_id,
            cell=cell,
            trusted_reference_use=checked_profile.trusted_reference_use,
            data_version=checked_batch.data_version,
            provenance=(*checked_batch.provenance, profile_provenance),
        )
        analysis_input = CompareOperationScenariosToolInput(
            run_id=context_id,
            scenario_context_id=context_id,
            route_id=checked_profile.route_id,
            cell=cell,
            baseline=checked_profile.baseline,
            comparisons=checked_profile.comparisons,
            current_state_reference=None,
        )
        return DefaultScenarioContextTemplate(
            task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
            scenario_context_id=context_id,
            verified_context=verified_context,
            analysis_input=analysis_input,
            created_by_reference=checked_profile.created_by_reference,
            profile_id=checked_profile.profile_id,
            profile_version=checked_profile.profile_version,
            profile_sha256=checked_profile.profile_sha256,
        )


def load_reviewed_default_scenario_registry(
    path: str | Path,
) -> ReviewedDefaultScenarioRegistry:
    profile_path = Path(path)
    if profile_path.is_symlink() or not profile_path.is_file():
        raise ValueError("default scenario registry must be a regular file")
    if profile_path.stat().st_size > _MAX_PROFILE_BYTES:
        raise ValueError("default scenario registry is too large")
    try:
        payload_bytes = profile_path.read_bytes()
        parsed = json.loads(
            payload_bytes.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        if not isinstance(parsed, dict) or parsed.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("default scenario registry schema is unsupported")
        unknown = set(parsed) - {"schema_version", "profiles"}
        if unknown:
            raise ValueError("default scenario registry has unknown fields")
        raw_profiles = parsed.get("profiles")
        if not isinstance(raw_profiles, list):
            raise ValueError("default scenario registry profiles are invalid")
        profiles = tuple(
            ReviewedDefaultScenarioProfile.model_validate(
                _normalize_profile_json(item),
                strict=True,
            )
            for item in raw_profiles
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("default scenario registry must be strict UTF-8 JSON") from exc
    return ReviewedDefaultScenarioRegistry(
        profiles,
        file_sha256=hashlib.sha256(payload_bytes).hexdigest(),
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _normalize_profile_json(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    for field_name in ("allowed_cutoff_cycles", "allowed_source_sha256s"):
        field_value = normalized.get(field_name)
        if isinstance(field_value, list):
            normalized[field_name] = tuple(field_value)
    normalized["baseline"] = _normalize_scenario_json(normalized.get("baseline"))
    comparisons = normalized.get("comparisons")
    if isinstance(comparisons, list):
        normalized["comparisons"] = tuple(
            _normalize_scenario_json(item) for item in comparisons
        )
    return normalized


def _normalize_scenario_json(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized = dict(value)
    segments = normalized.get("segments")
    if isinstance(segments, list):
        normalized["segments"] = tuple(segments)
    return normalized


def _profile_provenance_time(batch: VerifiedEarlyCycleBatch) -> datetime:
    value = max(item.created_at for item in batch.provenance)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("batch provenance timestamp must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "DefaultScenarioContextTemplate",
    "ReviewedDefaultScenarioProfile",
    "ReviewedDefaultScenarioRegistry",
    "load_reviewed_default_scenario_registry",
]
