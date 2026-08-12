"""Ledger-bound ``generate_audited_report`` tool with no caller-supplied numbers.

The public contract accepts only controlled report/claim keys and references to
registered ``ToolResult`` paths.  The executor resolves finite values inside
``AuditLedger`` and creates the internal rendering evidence itself.  This keeps
LLMs and untrusted clients outside the numerical-reporting boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import Field, field_validator, model_validator

from quanxin_life.audit import AuditLedger, NumericEvidence
from quanxin_life.core import EvidenceLevel, ProvenanceRecord, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.reporting import ReportClaim, build_audited_markdown_report
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import AUDITED_REPORT_TOOL_VERSION
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

REPORT_DATA_VERSION = "ledger-bound-toolresults-v1"
REPORT_FEATURE_VERSION = "audited-evidence-v1"
Clock = Callable[[], datetime]


class ReportKind(StrEnum):
    """Controlled report titles; callers cannot render arbitrary titles."""

    LIFETIME_DECISION = "lifetime_decision"
    STORAGE_LIFETIME_SCENARIO = "storage_lifetime_scenario"


class ReportClaimKind(StrEnum):
    """Controlled non-numeric narratives accepted by the formal report path."""

    LIFETIME_PREDICTION = "lifetime_prediction"
    DECISION_POLICY = "decision_policy"
    SCENARIO_PROJECTION = "scenario_projection"


_REPORT_TITLES: dict[ReportKind, str] = {
    ReportKind.LIFETIME_DECISION: "储能电芯寿命决策审计报告",
    ReportKind.STORAGE_LIFETIME_SCENARIO: "Storage lifetime scenario audit report",
}
_CLAIM_NARRATIVES: dict[ReportClaimKind, str] = {
    ReportClaimKind.LIFETIME_PREDICTION: "下列数值仅由已登记工具结果中的证据路径解析并渲染。",
}
_CLAIM_NARRATIVES[ReportClaimKind.DECISION_POLICY] = (
    "Decision policy thresholds are resolved from a registered, human-approved policy result."
)
_CLAIM_NARRATIVES[ReportClaimKind.SCENARIO_PROJECTION] = (
    "Values are deterministic outputs from a candidate BLAST reference scenario. "
    "They are not a confidence interval, product-specific validation, or a lifetime promise."
)
_CLAIM_EVIDENCE_LEVELS: dict[ReportClaimKind, EvidenceLevel] = {
    ReportClaimKind.LIFETIME_PREDICTION: EvidenceLevel.MODEL_INFERENCE,
    ReportClaimKind.DECISION_POLICY: EvidenceLevel.DOMAIN_KNOWLEDGE,
    ReportClaimKind.SCENARIO_PROJECTION: EvidenceLevel.PHYSICS_REFERENCE,
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


class NumericEvidenceReference(ContractModel):
    """A non-numeric pointer to one finite ledger value."""

    result_id: str
    json_path: str = Field(min_length=1)

    @field_validator("result_id")
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc
        return value

    @field_validator("json_path")
    @classmethod
    def require_tool_value_path(cls, value: str) -> str:
        segments = value.split(".")
        if segments[0] != "values" or any(not segment for segment in segments):
            raise ValueError("json_path must start with values and contain nonblank segments")
        return value


class AuditedReportClaimReference(ContractModel):
    """A controlled statement kind plus non-numeric evidence references."""

    claim_kind: ReportClaimKind
    numeric_evidence: tuple[NumericEvidenceReference, ...] = Field(min_length=1)


class GenerateAuditedReportToolInput(ContractModel):
    """Formal report input with no caller-rendered title, narrative or numbers."""

    report_kind: ReportKind
    claims: tuple[AuditedReportClaimReference, ...] = Field(min_length=1)
    upstream_result_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("upstream_result_ids")
    @classmethod
    def require_unique_uuid_result_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("upstream_result_ids must be unique")
        try:
            for result_id in value:
                UUID(result_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("upstream_result_ids must contain UUID strings") from exc
        return value

    @model_validator(mode="after")
    def require_declared_evidence_and_unique_claim_kinds(self) -> GenerateAuditedReportToolInput:
        claim_kinds = tuple(claim.claim_kind for claim in self.claims)
        if len(claim_kinds) != len(set(claim_kinds)):
            raise ValueError("claim kinds must be unique in one formal report")
        declared_ids = set(self.upstream_result_ids)
        referenced_ids: set[str] = set()
        for claim in self.claims:
            for evidence in claim.numeric_evidence:
                referenced_ids.add(evidence.result_id)
                if evidence.result_id not in declared_ids:
                    raise ValueError(
                        "every NumericEvidenceReference.result_id must be listed in "
                        "declared upstream_result_ids"
                    )
        if declared_ids != referenced_ids:
            raise ValueError(
                "declared upstream_result_ids must exactly match evidence result IDs"
            )
        return self


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_declared_results(
    *,
    input_value: GenerateAuditedReportToolInput,
    audit_ledger: AuditLedger,
) -> tuple[ToolResult, ...]:
    return tuple(
        audit_ledger.resolve_registered_result(result_id)
        for result_id in input_value.upstream_result_ids
    )


def _resolve_claims(
    *,
    claim_references: tuple[AuditedReportClaimReference, ...],
    audit_ledger: AuditLedger,
) -> tuple[ReportClaim, ...]:
    """Create internal numeric evidence only after resolving ledger values."""

    resolved_claims: list[ReportClaim] = []
    for claim_reference in claim_references:
        evidence = tuple(
            NumericEvidence(
                result_id=reference.result_id,
                json_path=reference.json_path,
                reported_value=audit_ledger.resolve_numeric_value(
                    reference.result_id,
                    reference.json_path,
                ),
                evidence_level=_CLAIM_EVIDENCE_LEVELS[claim_reference.claim_kind],
            )
            for reference in claim_reference.numeric_evidence
        )
        resolved_claims.append(
            ReportClaim(
                claim_id=claim_reference.claim_kind.value,
                narrative=_CLAIM_NARRATIVES[claim_reference.claim_kind],
                numeric_evidence=evidence,
            )
        )
    return tuple(resolved_claims)


def _derive_provenance(results: Sequence[ToolResult]) -> list[ProvenanceRecord]:
    seen: set[str] = set()
    derived: list[ProvenanceRecord] = []
    for result in results:
        for record in result.provenance:
            fingerprint = sha256_canonical(record.model_dump(mode="json"))
            if fingerprint not in seen:
                seen.add(fingerprint)
                derived.append(record)
    if not derived:
        raise ValueError("registered upstream ToolResults must contain provenance")
    return derived


def _upstream_context(results: Sequence[ToolResult]) -> list[dict[str, str]]:
    return [
        {
            "result_id": result.result_id,
            "tool_name": result.tool_name,
            "tool_version": result.tool_version,
            "model_version": result.model_version or "",
            "data_version": result.data_version or "",
            "feature_version": result.feature_version or "",
            "input_hash": result.input_hash,
        }
        for result in results
    ]


def execute_generate_audited_report_tool(
    input_value: GenerateAuditedReportToolInput,
    *,
    audit_ledger: AuditLedger,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Render a formal report from controlled templates and ledger-only values."""

    validated_input = GenerateAuditedReportToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    upstream_results = _resolve_declared_results(
        input_value=validated_input,
        audit_ledger=audit_ledger,
    )
    generated_at = _execution_timestamp(clock)
    report = build_audited_markdown_report(
        title=_REPORT_TITLES[validated_input.report_kind],
        claims=_resolve_claims(
            claim_references=validated_input.claims,
            audit_ledger=audit_ledger,
        ),
        ledger=audit_ledger,
        generated_at=generated_at,
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.GENERATE_AUDITED_REPORT.value,
        tool_version=AUDITED_REPORT_TOOL_VERSION,
        model_version=REPORTING_VERSION,
        data_version=REPORT_DATA_VERSION,
        feature_version=REPORT_FEATURE_VERSION,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "report_id": report.report_id,
            "report_kind": validated_input.report_kind.value,
            "rendering_version": report.reporting_version,
            "markdown": report.markdown,
            "claim_ids": [resolved_claim.claim.claim_id for resolved_claim in report.claims],
            "upstream_result_ids": list(validated_input.upstream_result_ids),
            "upstream_context": _upstream_context(upstream_results),
        },
        uncertainty=None,
        warnings=[],
        provenance=_derive_provenance(upstream_results),
        created_at=generated_at,
    )


def register_generate_audited_report_tool(
    registry: ToolRegistry,
    *,
    audit_ledger: AuditLedger,
    clock: Clock = _utc_now,
) -> RegisteredTool[GenerateAuditedReportToolInput]:
    """Register the ledger-bound formal report implementation."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.GENERATE_AUDITED_REPORT,
            tool_version=AUDITED_REPORT_TOOL_VERSION,
            input_model=GenerateAuditedReportToolInput,
            executor=lambda input_value: execute_generate_audited_report_tool(
                input_value,
                audit_ledger=audit_ledger,
                clock=clock,
            ),
        )
    )
