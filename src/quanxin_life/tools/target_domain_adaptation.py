"""Ledger-bound CORAL adaptation from source-train to target-train evidence.

The public caller supplies only a server-owned adaptation cohort identifier.
The cohort resolver identifies cell-disjoint source/target *training* feature
results; this tool re-resolves every result from the audit ledger, validates
their standard early-cycle envelopes, and runs the deterministic CORAL adapter.
No target labels, target-test cells, raw feature rows, or adaptation
hyperparameters enter through an API, MCP client, Agent, or UI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.adaptation import CORALFeatureAdapter
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools.cycle_life_prediction import EarlyCycleFeatureEvidence
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

TARGET_DOMAIN_ADAPTATION_TOOL_VERSION = "target-domain-adaptation-tool-v1"
ADAPTATION_EVIDENCE_TYPE = "quanxin_life.target_domain_adaptation.v1"
CORAL_METHOD = "CORAL"
TARGET_DOMAIN_ADAPTATION_NOT_CALIBRATION_WARNING = (
    "TARGET_DOMAIN_ADAPTATION_DOES_NOT_ESTABLISH_CONFORMAL_COVERAGE"
)
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _TrustedAdaptationModel(ContractModel):
    """Strict trusted-store contract; never a public tool input."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class VerifiedAdaptationCohort(_TrustedAdaptationModel):
    """Server-owned source/target train result IDs and CORAL configuration."""

    adaptation_cohort_id: str = Field(min_length=1)
    source_feature_result_ids: tuple[str, ...] = Field(min_length=2)
    target_feature_result_ids: tuple[str, ...] = Field(min_length=2)
    source_split_manifest: SplitManifest
    target_split_manifest: SplitManifest
    adapter_version: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    covariance_regularization: float = Field(default=1e-4, gt=0, allow_inf_nan=False)

    @field_validator("adaptation_cohort_id", "adapter_version")
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("adaptation cohort identifiers must not be blank")
        return normalized

    @field_validator("source_feature_result_ids", "target_feature_result_ids")
    @classmethod
    def require_unique_uuid_result_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for result_id in value:
            try:
                UUID(result_id)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("adaptation feature result IDs must be UUID strings") from exc
        if len(set(value)) != len(value):
            raise ValueError("adaptation feature result IDs must be unique within each cohort")
        return value

    @field_validator("feature_names")
    @classmethod
    def require_unique_feature_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("feature_names must be non-blank and unique")
        return value

    @model_validator(mode="after")
    def require_disjoint_train_only_context(self) -> VerifiedAdaptationCohort:
        if set(self.source_feature_result_ids) & set(self.target_feature_result_ids):
            raise ValueError("source and target feature result IDs must be disjoint")
        if self.source_split_manifest.dataset_id == self.target_split_manifest.dataset_id:
            raise ValueError("source and target adaptation datasets must differ")
        if set(self.source_split_manifest.train) & set(self.target_split_manifest.train):
            raise ValueError("source and target adaptation train cell_id values must be disjoint")
        if len(self.source_feature_result_ids) != len(self.source_split_manifest.train):
            raise ValueError("source result IDs must exactly represent source split_manifest.train")
        if len(self.target_feature_result_ids) != len(self.target_split_manifest.train):
            raise ValueError("target result IDs must exactly represent target split_manifest.train")
        return self


class VerifiedAdaptationCohortResolver(Protocol):
    """Server-side lookup for an approved, cell-disjoint adaptation cohort."""

    def resolve_verified_adaptation_cohort(
        self, adaptation_cohort_id: str
    ) -> VerifiedAdaptationCohort: ...


class AdaptToTargetDomainToolInput(ContractModel):
    """Public input has no features, labels, model weights or split memberships."""

    adaptation_cohort_id: str = Field(min_length=1)

    @field_validator("adaptation_cohort_id")
    @classmethod
    def require_nonblank_cohort_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("adaptation_cohort_id must not be blank")
        return normalized


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_cohort(
    resolver: VerifiedAdaptationCohortResolver,
    adaptation_cohort_id: str,
) -> VerifiedAdaptationCohort:
    resolved = resolver.resolve_verified_adaptation_cohort(adaptation_cohort_id)
    try:
        cohort = VerifiedAdaptationCohort.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(f"trusted adaptation cohort violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted adaptation cohort does not satisfy its public contract") from exc
    if cohort.adaptation_cohort_id != adaptation_cohort_id:
        raise ValueError("trusted adaptation resolver returned a mismatched cohort identifier")
    return cohort


def _decode_early_feature_evidence(result: ToolResult) -> EarlyCycleFeatureEvidence:
    if result.tool_name != StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value:
        raise ValueError("adaptation result IDs must resolve to extract_early_cycle_features")
    if result.tool_version != EARLY_CYCLE_FEATURE_TOOL_VERSION:
        raise ValueError("adaptation feature result has an unsupported tool_version")
    if result.model_version != EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION:
        raise ValueError("adaptation feature result has an unsupported model_version")
    if result.values.get("artifact_type") != EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE:
        raise ValueError("adaptation feature result must contain the expected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError("adaptation feature result must contain values.artifact")
    try:
        evidence = EarlyCycleFeatureEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("adaptation feature artifact does not satisfy its contract") from exc
    for field_name in ("data_version", "feature_version"):
        if getattr(result, field_name) != getattr(evidence, field_name):
            raise ValueError(f"adaptation feature result {field_name} must match its artifact")
    if not any(record.source_kind is SourceKind.OBSERVED for record in result.provenance):
        raise ValueError("adaptation feature result requires OBSERVED provenance")
    return evidence


def _resolve_feature_results(
    result_ids: Sequence[str], *, audit_ledger: AuditLedger
) -> tuple[tuple[ToolResult, EarlyCycleFeatureEvidence], ...]:
    return tuple(
        (
            result := audit_ledger.resolve_registered_result(result_id),
            _decode_early_feature_evidence(result),
        )
        for result_id in result_ids
    )


def _feature_mapping(
    entries: Sequence[tuple[ToolResult, EarlyCycleFeatureEvidence]],
    *,
    expected_dataset_id: str,
    expected_cell_ids: tuple[str, ...],
    expected_feature_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, float]], str, str, str, int]:
    if not entries:
        raise ValueError("adaptation feature evidence must be non-empty")
    feature_mapping: dict[str, dict[str, float]] = {}
    feature_versions: set[str] = set()
    split_versions: set[str] = set()
    data_versions: set[str] = set()
    cutoff_cycles: set[int] = set()
    for _, evidence in entries:
        if evidence.dataset_id != expected_dataset_id:
            raise ValueError("adaptation feature dataset_id must match its split manifest")
        if evidence.cell_id in feature_mapping:
            raise ValueError("adaptation feature evidence must contain unique cell_id values")
        missing_features = set(expected_feature_names) - set(evidence.condition_features)
        if missing_features:
            raise ValueError("adaptation feature evidence is missing configured feature_names")
        feature_mapping[evidence.cell_id] = {
            feature_name: float(evidence.condition_features[feature_name])
            for feature_name in expected_feature_names
        }
        feature_versions.add(evidence.feature_version)
        split_versions.add(evidence.split_version)
        data_versions.add(evidence.data_version)
        cutoff_cycles.add(evidence.cutoff_cycle)
    if set(feature_mapping) != set(expected_cell_ids):
        raise ValueError("adaptation feature cell_ids must exactly match split_manifest.train")
    if (
        len(feature_versions) != 1
        or len(split_versions) != 1
        or len(data_versions) != 1
        or len(cutoff_cycles) != 1
    ):
        raise ValueError(
            "each adaptation domain must use one feature, split, data version and cutoff_cycle"
        )
    return (
        feature_mapping,
        next(iter(feature_versions)),
        next(iter(split_versions)),
        next(iter(data_versions)),
        next(iter(cutoff_cycles)),
    )


def _merge_provenance(results: Sequence[ToolResult]) -> list[ProvenanceRecord]:
    merged: list[ProvenanceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for result in results:
        for record in result.provenance:
            fingerprint = (record.source_id, record.uri, record.sha256)
            if fingerprint not in seen:
                seen.add(fingerprint)
                merged.append(record)
    return merged


def execute_adapt_to_target_domain_tool(
    input_value: AdaptToTargetDomainToolInput,
    *,
    audit_ledger: AuditLedger,
    resolver: VerifiedAdaptationCohortResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Align source-train features to target-train statistics with CORAL only."""

    validated_input = AdaptToTargetDomainToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    cohort = _resolve_verified_cohort(resolver, validated_input.adaptation_cohort_id)
    source_entries = _resolve_feature_results(
        cohort.source_feature_result_ids, audit_ledger=audit_ledger
    )
    target_entries = _resolve_feature_results(
        cohort.target_feature_result_ids, audit_ledger=audit_ledger
    )
    (
        source_features,
        source_feature_version,
        source_split_version,
        source_data_version,
        source_cutoff_cycle,
    ) = (
        _feature_mapping(
            source_entries,
            expected_dataset_id=cohort.source_split_manifest.dataset_id,
            expected_cell_ids=cohort.source_split_manifest.train,
            expected_feature_names=cohort.feature_names,
        )
    )
    (
        target_features,
        target_feature_version,
        target_split_version,
        target_data_version,
        target_cutoff_cycle,
    ) = (
        _feature_mapping(
            target_entries,
            expected_dataset_id=cohort.target_split_manifest.dataset_id,
            expected_cell_ids=cohort.target_split_manifest.train,
            expected_feature_names=cohort.feature_names,
        )
    )
    if source_feature_version != target_feature_version:
        raise ValueError("source and target adaptation feature_version must match")
    if source_cutoff_cycle != target_cutoff_cycle:
        raise ValueError("source and target adaptation cutoff_cycle must match")

    adapter = CORALFeatureAdapter(
        adapter_version=cohort.adapter_version,
        feature_version=source_feature_version,
        source_split_version=source_split_version,
        target_split_version=target_split_version,
        feature_names=cohort.feature_names,
        covariance_regularization=cohort.covariance_regularization,
    ).fit(
        source_features=source_features,
        source_split_manifest=cohort.source_split_manifest,
        target_adaptation_features=target_features,
        target_split_manifest=cohort.target_split_manifest,
    )
    adapted_source_features = adapter.transform_source_features(source_features)
    source_results = tuple(entry[0] for entry in source_entries)
    target_results = tuple(entry[0] for entry in target_entries)
    artifact = {
        "adaptation_cohort_id": cohort.adaptation_cohort_id,
        "method": CORAL_METHOD,
        "adapter_version": cohort.adapter_version,
        "source_dataset_id": cohort.source_split_manifest.dataset_id,
        "target_dataset_id": cohort.target_split_manifest.dataset_id,
        "source_feature_result_ids": list(cohort.source_feature_result_ids),
        "target_feature_result_ids": list(cohort.target_feature_result_ids),
        "source_split_manifest_hash": sha256_canonical(
            cohort.source_split_manifest.model_dump(mode="json")
        ),
        "target_split_manifest_hash": sha256_canonical(
            cohort.target_split_manifest.model_dump(mode="json")
        ),
        "feature_names": list(cohort.feature_names),
        "feature_version": source_feature_version,
        "cutoff_cycle": source_cutoff_cycle,
        "source_split_version": source_split_version,
        "target_split_version": target_split_version,
        "source_data_version": source_data_version,
        "target_data_version": target_data_version,
        "adapted_source_features": adapted_source_features,
        "adapted_source_features_sha256": sha256_canonical(adapted_source_features),
    }
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.ADAPT_TO_TARGET_DOMAIN.value,
        tool_version=TARGET_DOMAIN_ADAPTATION_TOOL_VERSION,
        model_version=cohort.adapter_version,
        data_version=f"{source_data_version}__to__{target_data_version}",
        feature_version=source_feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={"artifact_type": ADAPTATION_EVIDENCE_TYPE, "artifact": artifact},
        uncertainty=None,
        warnings=[TARGET_DOMAIN_ADAPTATION_NOT_CALIBRATION_WARNING],
        provenance=_merge_provenance((*source_results, *target_results)),
        created_at=_execution_timestamp(clock),
    )


def register_adapt_to_target_domain_tool(
    registry: ToolRegistry,
    *,
    audit_ledger: AuditLedger,
    resolver: VerifiedAdaptationCohortResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[AdaptToTargetDomainToolInput]:
    """Register CORAL only in a service context owning trusted cohort metadata."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.ADAPT_TO_TARGET_DOMAIN,
            tool_version=TARGET_DOMAIN_ADAPTATION_TOOL_VERSION,
            input_model=AdaptToTargetDomainToolInput,
            executor=lambda input_value: execute_adapt_to_target_domain_tool(
                input_value,
                audit_ledger=audit_ledger,
                resolver=resolver,
                clock=clock,
            ),
        )
    )
