"""Deterministic project report artifacts derived only from verified ToolResults."""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from quanxin_life.core import ReportExportFormat, ToolResult
from quanxin_life.reporting.audited_artifacts import (
    ReviewedPdfFont,
    _render_docx,
    _render_pdf,
)
from quanxin_life.reporting.contracts import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    ADVANCED_CELL_REPORT_MODEL_VERSION,
    ADVANCED_CELL_REPORT_TOOL_VERSION,
    AUDITED_REPORT_TOOL_NAME,
)

PROJECT_REPORT_BUNDLE_VERSION = "project-report-bundle-v1"


ProjectReportArtifactFormat = ReportExportFormat


@dataclass(frozen=True, slots=True)
class ProjectReportArtifact:
    format: ReportExportFormat
    filename: str
    media_type: str
    payload: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedReport:
    results: tuple[ToolResult, ...]
    report: ToolResult
    markdown: str
    artifact: Mapping[str, Any]


class ProjectReportArtifactRenderer:
    """Render an allowlisted artifact set without reading source dataset files."""

    def __init__(self, *, pdf_font: ReviewedPdfFont | None = None) -> None:
        self._pdf_font = pdf_font

    def render_tool_result_json(self, result: ToolResult) -> ProjectReportArtifact:
        """Serialize one validated ToolResult without changing any business value."""

        validated = ToolResult.model_validate(result.model_dump(mode="json"))
        payload = _json_bytes(validated.model_dump(mode="json"))
        return ProjectReportArtifact(
            format=ReportExportFormat.JSON,
            filename=f"tool-result-{validated.result_id}.json",
            media_type="application/json",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )

    def render(
        self,
        format: ReportExportFormat,
        *,
        results: Sequence[ToolResult],
        generated_at: datetime,
    ) -> ProjectReportArtifact:
        validated = _validate_report(results, generated_at=generated_at)
        if format is ReportExportFormat.MARKDOWN:
            payload = validated.markdown.encode("utf-8")
            filename = "advanced-cell-report.md"
            media_type = "text/markdown; charset=utf-8"
        elif format is ReportExportFormat.JSON:
            payload = _json_bytes(validated.report.model_dump(mode="json"))
            filename = "advanced-cell-report.json"
            media_type = "application/json"
        elif format is ReportExportFormat.DOCX:
            payload = _render_docx(validated.markdown, result=validated.report)
            filename = "advanced-cell-report.docx"
            media_type = (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            )
        elif format is ReportExportFormat.PDF:
            if self._pdf_font is None:
                raise ValueError("PDF export requires a reviewed font")
            payload = _render_pdf(
                validated.markdown,
                font=self._pdf_font,
                source_result_id=validated.report.result_id,
            )
            filename = "advanced-cell-report.pdf"
            media_type = "application/pdf"
        elif format is ReportExportFormat.SOH_CSV:
            payload = _render_soh_csv(validated)
            filename = "soh-trajectory.csv"
            media_type = "text/csv; charset=utf-8"
        elif format is ReportExportFormat.TOOL_RESULTS_JSON:
            payload = _render_result_collection(validated)
            filename = "nine-tool-results.json"
            media_type = "application/json"
        elif format is ReportExportFormat.ZIP:
            payload = self._render_zip(validated, generated_at=generated_at)
            filename = "advanced-cell-audit-package.zip"
            media_type = "application/zip"
        else:  # pragma: no cover - exhaustive StrEnum branch
            raise ValueError("unsupported project report artifact format")
        return ProjectReportArtifact(
            format=format,
            filename=filename,
            media_type=media_type,
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )

    def _render_zip(
        self,
        validated: _ValidatedReport,
        *,
        generated_at: datetime,
    ) -> bytes:
        base_files: list[tuple[str, str, bytes, tuple[str, ...]]] = []
        for ordinal, result in enumerate(validated.results, start=1):
            base_files.append(
                (
                    f"tool-results/step-{ordinal:02d}-{result.result_id}.json",
                    "application/json",
                    _json_bytes(result.model_dump(mode="json")),
                    (result.result_id,),
                )
            )
        formats = (
            (ReportExportFormat.TOOL_RESULTS_JSON, "tool-results/tool-results.json"),
            (ReportExportFormat.SOH_CSV, "derived/soh-trajectory.csv"),
            (ReportExportFormat.MARKDOWN, "report/report.md"),
            (ReportExportFormat.JSON, "report/report.json"),
            (ReportExportFormat.DOCX, "report/report.docx"),
            (ReportExportFormat.PDF, "report/report.pdf"),
        )
        for artifact_format, path in formats:
            artifact = self.render(
                artifact_format,
                results=validated.results,
                generated_at=generated_at,
            )
            source_ids = (
                tuple(result.result_id for result in validated.results)
                if artifact_format
                in {
                    ReportExportFormat.TOOL_RESULTS_JSON,
                    ReportExportFormat.SOH_CSV,
                }
                else (validated.report.result_id,)
            )
            base_files.append((path, artifact.media_type, artifact.payload, source_ids))
        manifest = {
            "schema_version": PROJECT_REPORT_BUNDLE_VERSION,
            "generated_at": _utc(generated_at).isoformat(),
            "report_result_id": validated.report.result_id,
            "tool_result_count": len(validated.results),
            "files": [
                {
                    "path": path,
                    "media_type": media_type,
                    "size_bytes": len(payload),
                    "sha256": sha256(payload).hexdigest(),
                    "source_result_ids": list(source_ids),
                }
                for path, media_type, payload, source_ids in base_files
            ],
        }
        entries = [
            ("manifest.json", _json_bytes(manifest)),
            *((path, payload) for path, _media_type, payload, _source_ids in base_files),
        ]
        output = io.BytesIO()
        with zipfile.ZipFile(output, mode="w") as archive:
            for path, payload in entries:
                info = zipfile.ZipInfo(path, date_time=_zip_timestamp(generated_at))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload)
        return output.getvalue()


def _validate_report(
    results: Sequence[ToolResult],
    *,
    generated_at: datetime,
) -> _ValidatedReport:
    _utc(generated_at)
    validated = tuple(
        ToolResult.model_validate(result.model_dump(mode="json")) for result in results
    )
    if len(validated) != 9:
        raise ValueError("project report export requires exactly nine ToolResults")
    result_ids = {result.result_id for result in validated}
    if len(result_ids) != len(validated):
        raise ValueError("project report ToolResult identities must be unique")
    reports = tuple(
        result
        for result in validated
        if result.tool_name == AUDITED_REPORT_TOOL_NAME
    )
    if len(reports) != 1:
        raise ValueError("project report export requires one audited report ToolResult")
    report = reports[0]
    if (
        report.tool_version != ADVANCED_CELL_REPORT_TOOL_VERSION
        or report.model_version != ADVANCED_CELL_REPORT_MODEL_VERSION
        or report.values.get("artifact_type") != ADVANCED_CELL_REPORT_EVIDENCE_TYPE
    ):
        raise ValueError("project report ToolResult uses an unsupported contract")
    markdown = report.values.get("markdown")
    artifact = report.values.get("artifact")
    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("project report ToolResult contains no Markdown")
    if not isinstance(artifact, Mapping):
        raise ValueError("project report ToolResult contains no structured artifact")
    return _ValidatedReport(
        results=validated,
        report=report,
        markdown=markdown,
        artifact=artifact,
    )


def _render_result_collection(validated: _ValidatedReport) -> bytes:
    return _json_bytes(
        {
            "schema_version": PROJECT_REPORT_BUNDLE_VERSION,
            "report_result_id": validated.report.result_id,
            "tool_results": [
                result.model_dump(mode="json") for result in validated.results
            ],
        }
    )


def _render_soh_csv(validated: _ValidatedReport) -> bytes:
    soh = _mapping(validated.artifact.get("soh"), label="SOH prediction")
    conformal = _mapping(
        validated.artifact.get("soh_conformal"),
        label="SOH Conformal band",
    )
    prediction_id = _result_id(soh.get("result_id"), validated.results)
    conformal_id = _result_id(conformal.get("result_id"), validated.results)
    if conformal.get("prediction_result_id") != prediction_id:
        raise ValueError("SOH prediction and Conformal evidence identities do not match")
    cycles = _numeric_sequence(soh.get("prediction_cycles"), label="SOH cycles")
    predicted = _numeric_sequence(soh.get("predicted_soh"), label="predicted SOH")
    band_cycles = _numeric_sequence(
        conformal.get("prediction_cycles"), label="SOH band cycles"
    )
    band_predicted = _numeric_sequence(
        conformal.get("predicted_soh"), label="SOH band prediction"
    )
    lower = _numeric_sequence(conformal.get("lower_soh"), label="SOH lower band")
    upper = _numeric_sequence(conformal.get("upper_soh"), label="SOH upper band")
    if cycles != band_cycles or predicted != band_predicted:
        raise ValueError("SOH prediction and Conformal evidence axes do not match")
    if len({len(cycles), len(predicted), len(lower), len(upper)}) != 1:
        raise ValueError("SOH prediction and Conformal evidence lengths do not match")
    if any(
        not low <= point <= high
        for point, low, high in zip(predicted, lower, upper, strict=True)
    ):
        raise ValueError("SOH Conformal band does not contain its ToolResult prediction")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "cycle",
            "predicted_soh",
            "lower_soh",
            "upper_soh",
            "prediction_result_id",
            "conformal_result_id",
        )
    )
    for cycle, point, low, high in zip(
        cycles,
        predicted,
        lower,
        upper,
        strict=True,
    ):
        writer.writerow((cycle, point, low, high, prediction_id, conformal_id))
    return output.getvalue().encode("utf-8")


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing from the report ToolResult")
    return value


def _result_id(value: object, results: Sequence[ToolResult]) -> str:
    if not isinstance(value, str) or value not in {result.result_id for result in results}:
        raise ValueError("report artifact references an unavailable ToolResult")
    return value


def _numeric_sequence(value: object, *, label: str) -> tuple[int | float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty ToolResult sequence")
    values: list[int | float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain only numbers")
        if not math.isfinite(float(item)):
            raise ValueError(f"{label} must contain only finite numbers")
        values.append(item)
    return tuple(values)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("report generation time must include a timezone")
    return value.astimezone(UTC)


def _zip_timestamp(value: datetime) -> tuple[int, int, int, int, int, int]:
    timestamp = _utc(value)
    if timestamp.year < 1980:
        raise ValueError("ZIP generation time must be 1980 or later")
    return (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    )


__all__ = [
    "PROJECT_REPORT_BUNDLE_VERSION",
    "ProjectReportArtifact",
    "ProjectReportArtifactFormat",
    "ProjectReportArtifactRenderer",
]
