from __future__ import annotations

import csv
import json
import zipfile
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO, StringIO
from uuid import uuid4

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.reporting.audited_artifacts import ReviewedPdfFont
from quanxin_life.reporting.contracts import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    ADVANCED_CELL_REPORT_MODEL_VERSION,
    ADVANCED_CELL_REPORT_TOOL_VERSION,
)
from quanxin_life.reporting.project_artifacts import (
    PROJECT_REPORT_BUNDLE_VERSION,
    ProjectReportArtifactFormat,
    ProjectReportArtifactRenderer,
)

NOW = datetime(2026, 8, 3, 9, 30, tzinfo=UTC)


def _provenance() -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            source_id="reviewed-matr-source",
            source_kind=SourceKind.OBSERVED,
            uri="test://reviewed/matr",
            sha256=sha256_canonical({"source": "reviewed-matr"}),
            description="Reviewed source used only to construct ToolResult evidence",
            created_at=NOW,
        )
    ]


def _result(*, ordinal: int, tool_name: str = "validate_battery_data") -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version=f"tool-v{ordinal}",
        model_version=f"model-v{ordinal}",
        data_version="MATR-reviewed-v1",
        feature_version="features-v1",
        input_hash=sha256_canonical({"ordinal": ordinal}),
        values={"artifact_type": f"test.artifact.{ordinal}", "ordinal": ordinal},
        provenance=_provenance(),
        created_at=NOW,
    )


def _nine_results() -> tuple[ToolResult, ...]:
    upstream = [_result(ordinal=ordinal) for ordinal in range(1, 9)]
    soh_result_id = upstream[4].result_id
    report = ToolResult(
        result_id=str(uuid4()),
        tool_name="generate_audited_report",
        tool_version=ADVANCED_CELL_REPORT_TOOL_VERSION,
        model_version=ADVANCED_CELL_REPORT_MODEL_VERSION,
        data_version="MATR-reviewed-v1",
        feature_version="advanced-cell-report-evidence-v1",
        input_hash=sha256_canonical({"upstream": [item.result_id for item in upstream]}),
        values={
            "artifact_type": ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
            "artifact": {
                "dataset_id": "MATR",
                "cell_id": "b3c34",
                "cutoff_cycle": 20,
                "split_version": "matr-cell-disjoint-v1",
                "soh": {
                    "result_id": soh_result_id,
                    "prediction_cycles": [21, 100, 500],
                    "predicted_soh": [0.991, 0.941, 0.803],
                },
                "soh_conformal": {
                    "result_id": upstream[6].result_id,
                    "prediction_result_id": soh_result_id,
                    "prediction_cycles": [21, 100, 500],
                    "predicted_soh": [0.991, 0.941, 0.803],
                    "lower_soh": [0.970, 0.910, 0.770],
                    "upper_soh": [1.012, 0.972, 0.836],
                },
                "upstream_result_ids": [item.result_id for item in upstream[3:7]],
            },
            "markdown": "# Advanced single-cell audited report\n\n- Cell: `b3c34`\n",
        },
        provenance=_provenance(),
        created_at=NOW,
    )
    return (*upstream, report)


def _reviewed_test_font() -> ReviewedPdfFont:
    reportlab = pytest.importorskip("reportlab")
    font_path = (
        __import__("pathlib").Path(reportlab.__file__).resolve().parent
        / "fonts"
        / "Vera.ttf"
    )
    if not font_path.is_file():
        pytest.skip("ReportLab test font is unavailable")
    return ReviewedPdfFont(path=font_path, sha256=sha256(font_path.read_bytes()).hexdigest())


def test_renderer_exports_nine_toolresults_and_toolresult_derived_soh_csv() -> None:
    results = _nine_results()
    renderer = ProjectReportArtifactRenderer()

    collection = renderer.render(
        ProjectReportArtifactFormat.TOOL_RESULTS_JSON,
        results=results,
        generated_at=NOW,
    )
    csv_artifact = renderer.render(
        ProjectReportArtifactFormat.SOH_CSV,
        results=results,
        generated_at=NOW,
    )

    decoded = json.loads(collection.payload)
    assert decoded["schema_version"] == PROJECT_REPORT_BUNDLE_VERSION
    assert decoded["report_result_id"] == results[-1].result_id
    assert [item["result_id"] for item in decoded["tool_results"]] == [
        item.result_id for item in results
    ]
    assert len(decoded["tool_results"]) == 9
    rows = list(csv.DictReader(StringIO(csv_artifact.payload.decode("utf-8"))))
    assert rows == [
        {
            "cycle": "21",
            "predicted_soh": "0.991",
            "lower_soh": "0.97",
            "upper_soh": "1.012",
            "prediction_result_id": results[4].result_id,
            "conformal_result_id": results[6].result_id,
        },
        {
            "cycle": "100",
            "predicted_soh": "0.941",
            "lower_soh": "0.91",
            "upper_soh": "0.972",
            "prediction_result_id": results[4].result_id,
            "conformal_result_id": results[6].result_id,
        },
        {
            "cycle": "500",
            "predicted_soh": "0.803",
            "lower_soh": "0.77",
            "upper_soh": "0.836",
            "prediction_result_id": results[4].result_id,
            "conformal_result_id": results[6].result_id,
        },
    ]
    assert collection.sha256 == sha256(collection.payload).hexdigest()
    assert csv_artifact.sha256 == sha256(csv_artifact.payload).hexdigest()


def test_renderer_zip_has_fixed_allowlisted_files_and_verified_manifest_hashes() -> None:
    results = _nine_results()
    artifact = ProjectReportArtifactRenderer(pdf_font=_reviewed_test_font()).render(
        ProjectReportArtifactFormat.ZIP,
        results=results,
        generated_at=NOW,
    )

    with zipfile.ZipFile(BytesIO(artifact.payload)) as archive:
        names = archive.namelist()
        assert "manifest.json" in names
        assert "report/report.md" in names
        assert "report/report.json" in names
        assert "report/report.docx" in names
        assert "report/report.pdf" in names
        assert "derived/soh-trajectory.csv" in names
        assert "tool-results/tool-results.json" in names
        assert len([name for name in names if name.startswith("tool-results/step-")]) == 9
        assert not any("raw" in name.casefold() or "upload" in name.casefold() for name in names)
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["schema_version"] == PROJECT_REPORT_BUNDLE_VERSION
        assert manifest["generated_at"] == NOW.isoformat()
        listed = {item["path"]: item for item in manifest["files"]}
        assert set(listed) == set(names) - {"manifest.json"}
        for path, item in listed.items():
            payload = archive.read(path)
            assert item["sha256"] == sha256(payload).hexdigest()
            assert item["size_bytes"] == len(payload)


def test_renderer_rejects_incomplete_or_mismatched_toolresult_evidence() -> None:
    results = _nine_results()
    renderer = ProjectReportArtifactRenderer()

    with pytest.raises(ValueError, match="exactly nine"):
        renderer.render(
            ProjectReportArtifactFormat.SOH_CSV,
            results=results[:-1],
            generated_at=NOW,
        )

    report = results[-1]
    bad_values = json.loads(json.dumps(report.values))
    bad_values["artifact"]["soh_conformal"]["prediction_result_id"] = str(uuid4())
    with pytest.raises(ValueError, match="SOH prediction"):
        renderer.render(
            ProjectReportArtifactFormat.SOH_CSV,
            results=(*results[:-1], report.model_copy(update={"values": bad_values})),
            generated_at=NOW,
        )


def test_renderer_exports_one_toolresult_without_recomputing_its_values() -> None:
    result = _nine_results()[0]

    artifact = ProjectReportArtifactRenderer().render_tool_result_json(result)

    assert artifact.filename == f"tool-result-{result.result_id}.json"
    assert artifact.media_type == "application/json"
    assert json.loads(artifact.payload) == result.model_dump(mode="json")
    assert artifact.sha256 == sha256(artifact.payload).hexdigest()
