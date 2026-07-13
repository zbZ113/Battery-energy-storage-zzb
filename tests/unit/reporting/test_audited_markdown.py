from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger, NumericEvidence
from quanxin_life.core import (
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)


def _result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="life-tool-v1",
        model_version="xgboost-v1",
        data_version="matr-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"cell_id": "cell-a"}),
        values={"lifetime": {"predicted_eol_cycle": 333.0}},
        provenance=[
            ProvenanceRecord(
                source_id="reporting-test-source",
                source_kind=SourceKind.PREDICTED,
                uri="test://reporting/source",
                sha256=sha256_canonical({"reporting": "source"}),
                description="Reporting test provenance",
                created_at=datetime(2026, 7, 13, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _claim(result: ToolResult, *, value: float = 333.0):
    from quanxin_life.reporting.audited_markdown import ReportClaim

    return ReportClaim(
        claim_id="life-estimate",
        narrative="早期循环寿命模型已完成本次推理, 数值见下方可追溯证据。",
        numeric_evidence=(
            NumericEvidence(
                result_id=result.result_id,
                json_path="values.lifetime.predicted_eol_cycle",
                reported_value=value,
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
            ),
        ),
    )


def test_audited_markdown_resolves_numeric_claims_only_through_the_ledger() -> None:
    from quanxin_life.reporting.audited_markdown import build_audited_markdown_report

    result = _result()
    report = build_audited_markdown_report(
        title="泉芯智寿研发评估",
        claims=(_claim(result),),
        ledger=AuditLedger((result,)),
        generated_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert report.claims[0].resolved_numeric_values == (333.0,)
    assert result.result_id in report.markdown
    assert "333.0" in report.markdown
    assert "MODEL_INFERENCE" in report.markdown


def test_report_build_blocks_an_altered_numeric_claim_before_markdown_is_emitted() -> None:
    from quanxin_life.reporting.audited_markdown import build_audited_markdown_report

    result = _result()
    with pytest.raises(ValueError, match="does not match"):
        build_audited_markdown_report(
            title="泉芯智寿研发评估",
            claims=(_claim(result, value=334.0),),
            ledger=AuditLedger((result,)),
            generated_at=datetime(2026, 7, 13, tzinfo=UTC),
        )


def test_report_claim_requires_at_least_one_numeric_evidence_reference() -> None:
    from quanxin_life.reporting.audited_markdown import ReportClaim

    with pytest.raises(ValueError, match="numeric_evidence"):
        ReportClaim(
            claim_id="unsupported",
            narrative="没有来源的数值性结论不得进入正式报告。",
            numeric_evidence=(),
        )
