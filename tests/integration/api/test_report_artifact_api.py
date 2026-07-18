from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)

NOW = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)


def _report_result() -> ToolResult:
    report_id = str(uuid4())
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=AUDITED_REPORT_TOOL_NAME,
        tool_version=AUDITED_REPORT_TOOL_VERSION,
        model_version=REPORTING_VERSION,
        data_version="ledger-bound-toolresults-v1",
        feature_version="audited-evidence-v1",
        input_hash=sha256_canonical({"report": report_id}),
        values={
            "report_id": report_id,
            "rendering_version": REPORTING_VERSION,
            "markdown": "# Audited report\n\nEvidence remains ledger-bound.\n",
        },
        provenance=[
            ProvenanceRecord(
                source_id="report-api-source",
                source_kind=SourceKind.PREDICTED,
                uri="test://report-api/source",
                sha256=sha256_canonical({"source": "report-api"}),
                description="Reviewed source for report API test",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_report_artifact_endpoint_returns_hashed_json_and_markdown_downloads() -> None:
    result = _report_result()
    ledger = AuditLedger((result,))
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        report_exporter=AuditedReportArtifactExporter(ledger),
    )
    client = TestClient(app)

    json_response = client.get(f"/v1/reports/{result.result_id}/artifacts/json")
    markdown_response = client.get(
        f"/v1/reports/{result.result_id}/artifacts/markdown"
    )

    assert json_response.status_code == 200
    assert json.loads(json_response.content)["result_id"] == result.result_id
    assert json_response.headers["etag"].startswith('"sha256:')
    assert "attachment;" in json_response.headers["content-disposition"]
    assert markdown_response.status_code == 200
    assert markdown_response.text == result.values["markdown"]
    assert markdown_response.headers["content-type"].startswith("text/markdown")


def test_report_artifact_endpoint_rejects_unknown_format_or_result() -> None:
    result = _report_result()
    ledger = AuditLedger((result,))
    client = TestClient(
        create_fastapi_app(
            create_available_tool_invocation_service(),
            report_exporter=AuditedReportArtifactExporter(ledger),
        )
    )

    unsupported = client.get(
        f"/v1/reports/{result.result_id}/artifacts/executable"
    )
    missing = client.get(f"/v1/reports/{uuid4()}/artifacts/markdown")

    assert unsupported.status_code == 422
    assert missing.status_code == 404
