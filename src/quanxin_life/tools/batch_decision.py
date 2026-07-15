"""Ledger-bound batch triage from registered interval and quality evidence.

The public tool is deliberately narrower than the pure decision function in
``quanxin_life.decision``.  Callers cannot submit an interval, threshold,
quality report, provenance, domain state or timestamp: those are resolved from
registered tool results, a trusted policy store, and the server clock.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    NormalizedConformalCalibration,
    NormalizedPredictionInterval,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import DataQualityIssue, DataQualityReport
from quanxin_life.decision import BatchDecisionPolicy, make_batch_decision
from quanxin_life.tools.conformal_calibration import (
    CONFORMAL_CALIBRATION_TOOL_VERSION,
    NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
)
from quanxin_life.tools.data_quality import (
    DATA_QUALITY_MODEL_VERSION,
    DATA_QUALITY_TOOL_VERSION,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

BATCH_DECISION_TOOL_VERSION = "batch-decision-tool-v2"
NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE = "quanxin_life.normalized_prediction_interval.v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _EvidenceModel(ContractModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class BatchDecisionToolInput(ContractModel):
    """Only registered result IDs and a trusted policy identifier are public."""

    prediction_interval_result_id: str
    calibration_result_id: str
    quality_result_id: str
    policy_id: str = Field(min_length=1)

    @field_validator(
        "prediction_interval_result_id", "calibration_result_id", "quality_result_id"
    )
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result IDs must be UUID strings") from exc
        return value

    @field_validator("policy_id")
    @classmethod
    def require_nonblank_policy_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy_id must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_distinct_upstream_result_ids(self) -> BatchDecisionToolInput:
        result_ids = (
            self.prediction_interval_result_id,
            self.calibration_result_id,
            self.quality_result_id,
        )
        if len(result_ids) != len(set(result_ids)):
            raise ValueError("upstream result IDs must be distinct")
        return self


class DecisionIntervalEvidence(_EvidenceModel):
    """A target-cell normalized interval issued by the Conformal tool chain."""

    prediction_interval: NormalizedPredictionInterval
    calibration_result_id: str
    calibration_domain_id: str = Field(min_length=1)
    target_domain_id: str = Field(min_length=1)
    target_domain_calibrated: bool

    @field_validator("calibration_result_id")
    @classmethod
    def require_calibration_result_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("calibration_result_id must be a UUID string") from exc
        return value

    @model_validator(mode="after")
    def require_calibrated_target_to_match_calibration_domain(self) -> DecisionIntervalEvidence:
        if self.target_domain_calibrated and self.target_domain_id != self.calibration_domain_id:
            raise ValueError(
                "target_domain_id must match calibration_domain_id when target domain is calibrated"
            )
        return self


class CalibrationEvidence(_EvidenceModel):
    """Subset of the standard calibration output needed for interval validation."""

    calibration_cohort_id: str = Field(min_length=1)
    calibration_domain_id: str = Field(min_length=1)
    calibration_scope: str = Field(min_length=1)
    calibration: NormalizedConformalCalibration
    calibration_cell_ids: tuple[str, ...] = Field(min_length=1)
    split_manifest_hash: Sha256
    source_manifest_hash: Sha256


class DataQualityEvidence(_EvidenceModel):
    """Strict decoder for deterministic data-quality tool output."""

    dataset_id: str = Field(min_length=1)
    blocked: bool
    quality_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    issue_count: int = Field(ge=0)
    issues: tuple[DataQualityIssue, ...]

    @model_validator(mode="after")
    def require_consistent_quality_summary(self) -> DataQualityEvidence:
        report = DataQualityReport(dataset_id=self.dataset_id, issues=self.issues)
        if self.blocked != report.blocked:
            raise ValueError("quality evidence blocked flag must match its issues")
        if self.issue_count != len(self.issues):
            raise ValueError("quality evidence issue_count must match its issues")
        if not math.isclose(self.quality_score, report.quality_score, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("quality evidence score must match deterministic issue scoring")
        return self

    def as_report(self) -> DataQualityReport:
        return DataQualityReport(dataset_id=self.dataset_id, issues=self.issues)


class VerifiedBatchDecisionPolicy(_EvidenceModel):
    """Human-approved policy resolved only by a trusted policy store."""

    policy_id: str = Field(min_length=1)
    policy: BatchDecisionPolicy
    policy_domain_id: str = Field(min_length=1)
    source_manifest_hash: Sha256
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("policy_id", "policy_domain_id")
    @classmethod
    def require_nonblank_identifiers(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("policy identifiers must not be blank")
        return normalized


class VerifiedBatchDecisionPolicyResolver(Protocol):
    """Server-side lookup for a versioned, human-approved decision policy."""

    def resolve_verified_batch_decision_policy(
        self, policy_id: str
    ) -> VerifiedBatchDecisionPolicy: ...


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _require_artifact(
    result: ToolResult,
    *,
    expected_tool_name: StandardToolName,
    expected_tool_version: str,
    expected_artifact_type: str,
    description: str,
) -> Mapping[str, object]:
    if result.tool_name != expected_tool_name.value:
        raise ValueError(f"{description} result must come from {expected_tool_name.value}")
    if result.tool_version != expected_tool_version:
        raise ValueError(f"{description} result has an unsupported tool_version")
    if result.values.get("artifact_type") != expected_artifact_type:
        raise ValueError(f"{description} result must contain the expected artifact_type")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{description} result must contain values.artifact")
    return artifact


def _decode_interval_evidence(result: ToolResult) -> DecisionIntervalEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        expected_tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        expected_artifact_type=NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
        description="prediction interval",
    )
    try:
        evidence = DecisionIntervalEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("prediction interval artifact does not satisfy its contract") from exc
    interval = evidence.prediction_interval
    for field_name in ("model_version", "data_version", "feature_version"):
        expected = getattr(interval.calibration, field_name)
        if getattr(result, field_name) != expected:
            raise ValueError(f"prediction interval result {field_name} must match its artifact")
    return evidence


def _decode_calibration_evidence(result: ToolResult) -> CalibrationEvidence:
    artifact = _require_artifact(
        result,
        expected_tool_name=StandardToolName.CALIBRATE_PREDICTION_INTERVAL,
        expected_tool_version=CONFORMAL_CALIBRATION_TOOL_VERSION,
        expected_artifact_type=NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
        description="conformal calibration",
    )
    try:
        evidence = CalibrationEvidence.model_validate(artifact)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("conformal calibration artifact does not satisfy its contract") from exc
    for field_name in ("model_version", "data_version", "feature_version"):
        expected = getattr(evidence.calibration, field_name)
        if getattr(result, field_name) != expected:
            raise ValueError(f"conformal calibration result {field_name} must match its artifact")
    return evidence


def _decode_quality_evidence(result: ToolResult) -> DataQualityEvidence:
    if result.tool_name != StandardToolName.VALIDATE_BATTERY_DATA.value:
        raise ValueError("quality result must come from validate_battery_data")
    if result.tool_version != DATA_QUALITY_TOOL_VERSION:
        raise ValueError("quality result has an unsupported tool_version")
    if result.model_version != DATA_QUALITY_MODEL_VERSION:
        raise ValueError("quality result has an unsupported model_version")
    if not any(
        record.source_kind in {SourceKind.OBSERVED, SourceKind.NEWLY_OBSERVED}
        for record in result.provenance
    ):
        raise ValueError("quality result requires OBSERVED or NEWLY_OBSERVED provenance")
    try:
        return DataQualityEvidence.model_validate(result.values)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("quality result values do not satisfy their contract") from exc


def _resolve_verified_policy(
    resolver: VerifiedBatchDecisionPolicyResolver, policy_id: str
) -> VerifiedBatchDecisionPolicy:
    resolved = resolver.resolve_verified_batch_decision_policy(policy_id)
    try:
        return VerifiedBatchDecisionPolicy.model_validate(resolved.model_dump(mode="json"))
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(f"trusted decision policy violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted decision policy does not satisfy its public contract") from exc


def _merge_provenance(*chains: Sequence[ProvenanceRecord]) -> list[ProvenanceRecord]:
    merged: list[ProvenanceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in chains:
        for record in chain:
            key = (record.source_id, record.uri, record.sha256)
            if key not in seen:
                seen.add(key)
                merged.append(record)
    return merged


def _validate_evidence_chain(
    *,
    interval_evidence: DecisionIntervalEvidence,
    calibration_evidence: CalibrationEvidence,
    quality_evidence: DataQualityEvidence,
    policy: VerifiedBatchDecisionPolicy,
) -> None:
    interval = interval_evidence.prediction_interval
    if interval_evidence.calibration_domain_id != calibration_evidence.calibration_domain_id:
        raise ValueError(
            "interval and calibration evidence must declare the same calibration domain"
        )
    if interval.calibration != calibration_evidence.calibration:
        raise ValueError("interval calibration must match registered calibration evidence")
    if quality_evidence.dataset_id != interval.dataset_id:
        raise ValueError("quality result dataset_id must match prediction interval dataset_id")
    if policy.policy_domain_id != interval.dataset_id:
        raise ValueError("approved policy domain must match prediction interval dataset_id")


def execute_batch_decision_tool(
    input_value: BatchDecisionToolInput,
    *,
    audit_ledger: AuditLedger,
    policy_resolver: VerifiedBatchDecisionPolicyResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Create a deterministic decision from ledger-registered upstream evidence."""

    validated_input = BatchDecisionToolInput.model_validate(input_value.model_dump(mode="json"))
    interval_result = audit_ledger.resolve_registered_result(
        validated_input.prediction_interval_result_id
    )
    calibration_result = audit_ledger.resolve_registered_result(
        validated_input.calibration_result_id
    )
    quality_result = audit_ledger.resolve_registered_result(validated_input.quality_result_id)
    interval_evidence = _decode_interval_evidence(interval_result)
    calibration_evidence = _decode_calibration_evidence(calibration_result)
    quality_evidence = _decode_quality_evidence(quality_result)
    if interval_evidence.calibration_result_id != calibration_result.result_id:
        raise ValueError("prediction interval must reference the registered calibration result")
    policy = _resolve_verified_policy(policy_resolver, validated_input.policy_id)
    if policy.policy_id != validated_input.policy_id:
        raise ValueError("trusted policy_id must match the requested policy_id")
    _validate_evidence_chain(
        interval_evidence=interval_evidence,
        calibration_evidence=calibration_evidence,
        quality_evidence=quality_evidence,
        policy=policy,
    )
    outcome = make_batch_decision(
        prediction_interval=interval_evidence.prediction_interval,
        quality_report=quality_evidence.as_report(),
        target_domain_calibrated=interval_evidence.target_domain_calibrated,
        policy=policy.policy,
        decided_at=_execution_timestamp(clock),
    )
    upstream_result_ids = (
        validated_input.prediction_interval_result_id,
        validated_input.calibration_result_id,
        validated_input.quality_result_id,
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.MAKE_BATCH_DECISION.value,
        tool_version=BATCH_DECISION_TOOL_VERSION,
        model_version=interval_evidence.prediction_interval.calibration.model_version,
        data_version=interval_evidence.prediction_interval.calibration.data_version,
        feature_version=interval_evidence.prediction_interval.calibration.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "cell_id": outcome.cell_id,
            "dataset_id": outcome.dataset_id,
            "decision": outcome.decision.value,
            "interval_lower_eol_cycle": outcome.interval_lower_eol_cycle,
            "interval_upper_eol_cycle": outcome.interval_upper_eol_cycle,
            "policy_id": policy.policy_id,
            "policy_version": outcome.policy_version,
            "policy_source_manifest_hash": policy.source_manifest_hash,
            "quality_report_blocked": outcome.quality_report_blocked,
            "reason_codes": list(outcome.reason_codes),
            "required_eol_cycle": outcome.required_eol_cycle,
            "target_domain_calibrated": outcome.target_domain_calibrated,
            "upstream_result_ids": list(upstream_result_ids),
        },
        uncertainty={
            "coverage_target": interval_evidence.prediction_interval.coverage_target,
            "lower_eol_cycle": outcome.interval_lower_eol_cycle,
            "upper_eol_cycle": outcome.interval_upper_eol_cycle,
        },
        warnings=list(outcome.reason_codes),
        provenance=_merge_provenance(
            interval_result.provenance,
            quality_result.provenance,
            policy.provenance,
        ),
        created_at=outcome.decided_at,
    )


def register_batch_decision_tool(
    registry: ToolRegistry,
    *,
    audit_ledger: AuditLedger,
    policy_resolver: VerifiedBatchDecisionPolicyResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[BatchDecisionToolInput]:
    """Register evidence-bound batch triage only for an explicit trusted context."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.MAKE_BATCH_DECISION,
            tool_version=BATCH_DECISION_TOOL_VERSION,
            input_model=BatchDecisionToolInput,
            executor=lambda input_value: execute_batch_decision_tool(
                input_value,
                audit_ledger=audit_ledger,
                policy_resolver=policy_resolver,
                clock=clock,
            ),
        )
    )
