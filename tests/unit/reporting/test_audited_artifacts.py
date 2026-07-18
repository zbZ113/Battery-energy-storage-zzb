from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.reporting.audited_artifacts import (
    AuditedReportArtifactExporter,
    ReportArtifactFormat,
    ReviewedPdfFont,
)
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)

NOW = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)


def test_reporting_package_import_keeps_optional_renderers_and_ml_stacks_lazy() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import quanxin_life.reporting; "
                "blocked = {'docx', 'reportlab', 'scipy', 'torch'}; "
                "loaded = sorted(name for name in blocked if name in sys.modules); "
                "assert not loaded, loaded"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr or probe.stdout


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
            "markdown": (
                "# Audited battery report\n\n"
                "Generated at (UTC): 2026-07-18T08:00:00+00:00\n\n"
                "## lifetime_prediction\n\n"
                "Values below are resolved from registered ToolResult evidence.\n\n"
                "- `values.predicted_cycle` = `333.0` "
                f"(MODEL_INFERENCE; ToolResult: `{uuid4()}`)\n"
            ),
            "claim_ids": ["lifetime_prediction"],
            "upstream_result_ids": [str(uuid4())],
            "upstream_context": [],
        },
        uncertainty=None,
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="audited-artifact-source",
                source_kind=SourceKind.PREDICTED,
                uri="test://reporting/artifact",
                sha256=sha256_canonical({"source": "artifact"}),
                description="Reviewed source for report artifact tests",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_json_and_markdown_artifacts_are_ledger_bound_and_hashed() -> None:
    result = _report_result()
    exporter = AuditedReportArtifactExporter(AuditLedger((result,)))

    json_artifact = exporter.export(result.result_id, ReportArtifactFormat.JSON)
    markdown_artifact = exporter.export(
        result.result_id,
        ReportArtifactFormat.MARKDOWN,
    )

    assert json_artifact.media_type == "application/json"
    assert json.loads(json_artifact.payload)["result_id"] == result.result_id
    assert markdown_artifact.media_type == "text/markdown; charset=utf-8"
    assert markdown_artifact.payload.decode() == result.values["markdown"]
    assert json_artifact.sha256 == sha256(json_artifact.payload).hexdigest()
    assert markdown_artifact.sha256 == sha256(markdown_artifact.payload).hexdigest()


def test_exporter_rejects_unregistered_or_wrong_tool_results() -> None:
    result = _report_result()
    exporter = AuditedReportArtifactExporter(AuditLedger((result,)))

    with pytest.raises(ValueError, match="not registered"):
        exporter.export(str(uuid4()), ReportArtifactFormat.MARKDOWN)

    wrong = result.model_copy(
        update={
            "result_id": str(uuid4()),
            "tool_name": "validate_battery_data",
        }
    )
    with pytest.raises(ValueError, match="audited report"):
        AuditedReportArtifactExporter(AuditLedger((wrong,))).export(
            wrong.result_id,
            ReportArtifactFormat.MARKDOWN,
        )


def test_docx_artifact_contains_only_the_registered_markdown_content() -> None:
    docx = pytest.importorskip("docx")
    result = _report_result()
    artifact = AuditedReportArtifactExporter(AuditLedger((result,))).export(
        result.result_id,
        ReportArtifactFormat.DOCX,
    )

    document = docx.Document(BytesIO(artifact.payload))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    header_text = "\n".join(
        paragraph.text
        for section in document.sections
        for paragraph in section.header.paragraphs
    )
    assert artifact.media_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert "Audited battery report" in text
    assert "333.0" in text
    assert "\u6cc9\u82af\u667a\u5bff" in header_text
    assert document.core_properties.author == "\u6cc9\u82af\u667a\u5bff"
    assert result.result_id not in text


def test_pdf_artifact_requires_and_verifies_an_operator_owned_font() -> None:
    reportlab = pytest.importorskip("reportlab")
    pypdf = pytest.importorskip("pypdf")
    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    if not font_path.is_file():
        pytest.skip("ReportLab test font is unavailable")
    font_hash = sha256(font_path.read_bytes()).hexdigest()
    result = _report_result()
    exporter = AuditedReportArtifactExporter(
        AuditLedger((result,)),
        pdf_font=ReviewedPdfFont(path=font_path, sha256=font_hash),
    )

    artifact = exporter.export(result.result_id, ReportArtifactFormat.PDF)
    reader = pypdf.PdfReader(BytesIO(artifact.payload))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert artifact.media_type == "application/pdf"
    assert artifact.payload.startswith(b"%PDF-")
    assert "Audited battery report" in text
    assert "333.0" in text
    with pytest.raises(ValueError, match="font SHA-256"):
        AuditedReportArtifactExporter(
            AuditLedger((result,)),
            pdf_font=ReviewedPdfFont(path=font_path, sha256="0" * 64),
        ).export(result.result_id, ReportArtifactFormat.PDF)
