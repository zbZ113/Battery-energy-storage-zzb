"""Contract tests for the ledger-bound ``generate_audited_report`` tool."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.tools.registry import StandardToolName, ToolRegistry


def _provenance(*, source_id: str = "audited-report-fixture") -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=SourceKind.PREDICTED,
        uri=f"test://audited-report/{source_id}",
        sha256=sha256_canonical({"source_id": source_id}),
        description="Registered provenance used only for audited report tool tests",
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def _result(*, value: float = 333.0) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version="life-tool-v1",
        model_version="hybrid-v1",
        data_version="synthetic-data-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"fixture": "audited-report"}),
        values={"lifetime": {"predicted_eol_cycle": value}},
        provenance=[_provenance()],
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


def _input(result: ToolResult):
    from quanxin_life.tools.audited_report import (
        AuditedReportClaimReference,
        GenerateAuditedReportToolInput,
        NumericEvidenceReference,
        ReportClaimKind,
        ReportKind,
    )

    return GenerateAuditedReportToolInput(
        report_kind=ReportKind.LIFETIME_DECISION,
        claims=(
            AuditedReportClaimReference(
                claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                numeric_evidence=(
                    NumericEvidenceReference(
                        result_id=result.result_id,
                        json_path="values.lifetime.predicted_eol_cycle",
                    ),
                ),
            ),
        ),
        upstream_result_ids=(result.result_id,),
    )


def test_registered_audited_report_tool_resolves_numeric_values_only_from_ledger() -> None:
    from quanxin_life.tools.audited_report import register_generate_audited_report_tool

    result = _result()
    executed_at = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)
    registry = ToolRegistry()
    register_generate_audited_report_tool(
        registry,
        audit_ledger=AuditLedger((result,)),
        clock=lambda: executed_at,
    )
    tool_input = _input(result)

    output = registry.execute(StandardToolName.GENERATE_AUDITED_REPORT, tool_input)

    assert output.input_hash == sha256_canonical(tool_input.model_dump(mode="json"))
    assert output.created_at == executed_at
    assert output.provenance == result.provenance
    assert output.values["report_kind"] == "lifetime_decision"
    assert output.values["claim_ids"] == ["lifetime_prediction"]
    assert "333.0" in output.values["markdown"]


def test_decision_policy_threshold_is_not_labeled_as_model_inference() -> None:
    from quanxin_life.tools.audited_report import (
        AuditedReportClaimReference,
        GenerateAuditedReportToolInput,
        NumericEvidenceReference,
        ReportClaimKind,
        ReportKind,
        execute_generate_audited_report_tool,
    )

    prediction = _result()
    policy = _result(value=444.0).model_copy(
        update={
            "tool_name": StandardToolName.MAKE_BATCH_DECISION.value,
            "values": {"required_eol_cycle": 444.0},
        }
    )
    tool_input = GenerateAuditedReportToolInput(
        report_kind=ReportKind.LIFETIME_DECISION,
        claims=(
            AuditedReportClaimReference(
                claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                numeric_evidence=(
                    NumericEvidenceReference(
                        result_id=prediction.result_id,
                        json_path="values.lifetime.predicted_eol_cycle",
                    ),
                ),
            ),
            AuditedReportClaimReference(
                claim_kind=ReportClaimKind.DECISION_POLICY,
                numeric_evidence=(
                    NumericEvidenceReference(
                        result_id=policy.result_id,
                        json_path="values.required_eol_cycle",
                    ),
                ),
            ),
        ),
        upstream_result_ids=(prediction.result_id, policy.result_id),
    )

    output = execute_generate_audited_report_tool(
        tool_input,
        audit_ledger=AuditLedger((prediction, policy)),
        clock=lambda: datetime(2026, 7, 14, 8, 30, tzinfo=UTC),
    )

    markdown = output.values["markdown"]
    assert output.values["claim_ids"] == ["lifetime_prediction", "decision_policy"]
    assert "values.required_eol_cycle" in markdown
    assert "DOMAIN_KNOWLEDGE" in markdown


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("title", "寿命约九百循环"),
        ("narrative", "预测寿命为" + "\uff19\uff10\uff05。"),
        ("reported_value", 333.0),
        ("evidence_level", "DATA_DIRECT"),
    ),
)
def test_audited_report_tool_rejects_caller_rendered_text_and_numeric_values(
    field_name: str,
    value: str | float,
) -> None:
    from quanxin_life.tools.audited_report import GenerateAuditedReportToolInput

    result = _result()
    payload = _input(result).model_dump(mode="python")
    if field_name == "title":
        payload["title"] = value
    else:
        claim_payload = payload["claims"][0]
        if field_name == "narrative":
            claim_payload["narrative"] = value
        elif field_name == "evidence_level":
            claim_payload["numeric_evidence"][0]["evidence_level"] = value
        else:
            claim_payload["numeric_evidence"][0]["reported_value"] = value

    with pytest.raises(ValueError, match="Extra inputs"):
        GenerateAuditedReportToolInput.model_validate(payload)


def test_audited_report_tool_rejects_unknown_or_undeclared_evidence_reference() -> None:
    from quanxin_life.tools.audited_report import (
        GenerateAuditedReportToolInput,
        NumericEvidenceReference,
    )

    result = _result()
    tool_input = _input(result)
    unknown_id = str(uuid4())
    unknown_reference = NumericEvidenceReference(
        result_id=unknown_id,
        json_path="values.lifetime.predicted_eol_cycle",
    )
    payload = tool_input.model_dump(mode="python")
    payload["claims"][0]["numeric_evidence"] = [unknown_reference.model_dump(mode="python")]
    with pytest.raises(ValueError, match="declared upstream_result_ids"):
        GenerateAuditedReportToolInput.model_validate(payload)

    payload["upstream_result_ids"] = [unknown_id]
    declared_unknown = GenerateAuditedReportToolInput.model_validate(payload)
    from quanxin_life.tools.audited_report import execute_generate_audited_report_tool

    with pytest.raises(ValueError, match="not registered"):
        execute_generate_audited_report_tool(
            declared_unknown,
            audit_ledger=AuditLedger((result,)),
        )


def test_audited_report_tool_rejects_declared_result_not_used_by_a_claim() -> None:
    from quanxin_life.tools.audited_report import GenerateAuditedReportToolInput

    result = _result()
    unrelated_result = _result(value=444.0)
    payload = _input(result).model_dump(mode="python")
    payload["upstream_result_ids"] = [result.result_id, unrelated_result.result_id]

    with pytest.raises(ValueError, match="exactly match"):
        GenerateAuditedReportToolInput.model_validate(payload)


def test_audited_report_tool_rejects_non_numeric_evidence_path() -> None:
    from quanxin_life.tools.audited_report import execute_generate_audited_report_tool

    result = _result()
    payload = _input(result).model_dump(mode="python")
    payload["claims"][0]["numeric_evidence"][0]["json_path"] = "values.lifetime"
    from quanxin_life.tools.audited_report import GenerateAuditedReportToolInput

    with pytest.raises(ValueError, match="not numeric"):
        execute_generate_audited_report_tool(
            GenerateAuditedReportToolInput.model_validate(payload),
            audit_ledger=AuditLedger((result,)),
        )


def test_audited_report_created_at_uses_execution_clock_not_caller_timestamp() -> None:
    from quanxin_life.tools.audited_report import execute_generate_audited_report_tool

    result = _result()
    executed_at = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)
    output = execute_generate_audited_report_tool(
        _input(result),
        audit_ledger=AuditLedger((result,)),
        clock=lambda: executed_at,
    )

    assert output.created_at == executed_at
