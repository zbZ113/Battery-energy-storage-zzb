"""Markdown report construction after numeric evidence has passed the firewall."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.audit import AuditLedger, NumericEvidence
from quanxin_life.core.schemas import ContractModel

REPORTING_VERSION = "audited-markdown-v1"


class _ReportModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ReportClaim(_ReportModel):
    """A narrative claim with every displayed number separately evidence-bound."""

    claim_id: str = Field(min_length=1)
    narrative: str = Field(min_length=1)
    numeric_evidence: tuple[NumericEvidence, ...] = Field(min_length=1)

    @field_validator("claim_id")
    @classmethod
    def require_nonblank_claim_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim_id must not be blank")
        return value


class ResolvedReportClaim(_ReportModel):
    claim: ReportClaim
    resolved_numeric_values: tuple[float, ...]


class AuditedMarkdownReport(_ReportModel):
    report_id: str
    reporting_version: str = REPORTING_VERSION
    title: str = Field(min_length=1)
    claims: tuple[ResolvedReportClaim, ...] = Field(min_length=1)
    markdown: str = Field(min_length=1)
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return value.astimezone(UTC)


def build_audited_markdown_report(
    *,
    title: str,
    claims: tuple[ReportClaim, ...],
    ledger: AuditLedger,
    generated_at: datetime,
) -> AuditedMarkdownReport:
    """Resolve every displayed number before building a deterministic Markdown view."""

    if not title.strip():
        raise ValueError("title must not be blank")
    if not claims:
        raise ValueError("at least one report claim is required")
    claim_ids = tuple(claim.claim_id for claim in claims)
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError("report claim_ids must be unique")

    resolved_claims = tuple(
        ResolvedReportClaim(
            claim=claim,
            resolved_numeric_values=tuple(
                ledger.verify_numeric_evidence(evidence) for evidence in claim.numeric_evidence
            ),
        )
        for claim in claims
    )
    report_id = str(uuid4())
    return AuditedMarkdownReport(
        report_id=report_id,
        title=title,
        claims=resolved_claims,
        markdown=_render_markdown(title=title, claims=resolved_claims, generated_at=generated_at),
        generated_at=generated_at,
    )


def _render_markdown(
    *,
    title: str,
    claims: tuple[ResolvedReportClaim, ...],
    generated_at: datetime,
) -> str:
    timestamp = _normalized_timestamp(generated_at)
    lines = [f"# {title}", "", f"生成时间 (UTC): {timestamp}"]
    for resolved_claim in claims:
        lines.extend(
            ("", f"## {resolved_claim.claim.claim_id}", "", resolved_claim.claim.narrative, "")
        )
        for evidence, value in zip(
            resolved_claim.claim.numeric_evidence,
            resolved_claim.resolved_numeric_values,
            strict=True,
        ):
            lines.append(
                "- "
                f"`{evidence.json_path}` = `{value!r}` "
                f"({evidence.evidence_level.value}; ToolResult: `{evidence.result_id}`)"
            )
    return "\n".join(lines) + "\n"


def _normalized_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    return value.astimezone(UTC).isoformat()
