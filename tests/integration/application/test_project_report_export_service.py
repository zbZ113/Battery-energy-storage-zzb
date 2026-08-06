from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from quanxin_life.application.project_reports import (
    ProjectReportConflictError,
    ProjectReportExpiredError,
    ProjectReportExportService,
    ProjectReportNotFoundError,
    SqlProjectReportResultResolver,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentRunStatus,
    ProvenanceRecord,
    ReportExportFormat,
    ReportExportStatus,
    ReportStatus,
    SourceKind,
    ToolResult,
    UserRole,
    sha256_canonical,
)
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import (
    ProvenanceRecordRow,
    Report,
    ReportExport,
    ToolResultRecord,
)
from quanxin_life.reporting.audited_artifacts import ReviewedPdfFont
from quanxin_life.reporting.contracts import (
    ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
    ADVANCED_CELL_REPORT_MODEL_VERSION,
    ADVANCED_CELL_REPORT_TOOL_VERSION,
)
from quanxin_life.reporting.project_artifacts import ProjectReportArtifactRenderer

NOW = datetime(2026, 8, 3, 10, 0, tzinfo=UTC)
PROJECT_ID = str(uuid4())
RUN_ID = str(uuid4())


@dataclass(frozen=True)
class _Run:
    run_id: str
    project_id: str
    status: AgentRunStatus


@dataclass(frozen=True)
class _RunResult:
    ordinal: int
    result: ToolResult


class _RunReader:
    def __init__(self, results: tuple[ToolResult, ...]) -> None:
        self.results = results
        self.run = _Run(RUN_ID, PROJECT_ID, AgentRunStatus.COMPLETED)

    def get_run(self, principal: AuthPrincipal, run_id: str) -> _Run:
        if principal.user_id != "member-1" or run_id != self.run.run_id:
            raise ValueError("run not visible")
        return self.run

    def list_results(
        self,
        principal: AuthPrincipal,
        run_id: str,
    ) -> tuple[_RunResult, ...]:
        self.get_run(principal, run_id)
        return tuple(
            _RunResult(ordinal, result)
            for ordinal, result in enumerate(self.results, start=1)
        )


class _ResultResolver:
    def __init__(self, results: tuple[ToolResult, ...]) -> None:
        self.results = results

    def resolve(
        self,
        *,
        project_id: str,
        run_id: str,
        result_ids: tuple[str, ...],
    ) -> tuple[ToolResult, ...]:
        assert project_id == PROJECT_ID
        assert run_id == RUN_ID
        assert result_ids == tuple(result.result_id for result in self.results)
        return self.results


def _provenance() -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            source_id="reviewed-source",
            source_kind=SourceKind.OBSERVED,
            uri="test://reviewed/source",
            sha256=sha256_canonical({"source": "reviewed"}),
            description="Reviewed source",
            created_at=NOW,
        )
    ]


def _results() -> tuple[ToolResult, ...]:
    upstream = tuple(
        ToolResult(
            result_id=str(uuid4()),
            tool_name=f"tool_{ordinal}",
            tool_version="tool-v1",
            model_version="model-v1",
            data_version="MATR-reviewed-v1",
            feature_version="features-v1",
            input_hash=sha256_canonical({"ordinal": ordinal}),
            values={"ordinal": ordinal},
            provenance=_provenance(),
            created_at=NOW,
        )
        for ordinal in range(1, 9)
    )
    report = ToolResult(
        result_id=str(uuid4()),
        tool_name="generate_audited_report",
        tool_version=ADVANCED_CELL_REPORT_TOOL_VERSION,
        model_version=ADVANCED_CELL_REPORT_MODEL_VERSION,
        data_version="MATR-reviewed-v1",
        feature_version="advanced-cell-report-evidence-v1",
        input_hash=sha256_canonical({"results": [item.result_id for item in upstream]}),
        values={
            "artifact_type": ADVANCED_CELL_REPORT_EVIDENCE_TYPE,
            "artifact": {
                "dataset_id": "MATR",
                "cell_id": "b3c34",
                "cutoff_cycle": 20,
                "split_version": "matr-cell-disjoint-v1",
                "soh": {
                    "result_id": upstream[4].result_id,
                    "prediction_cycles": [21, 500],
                    "predicted_soh": [0.99, 0.8],
                },
                "soh_conformal": {
                    "result_id": upstream[6].result_id,
                    "prediction_result_id": upstream[4].result_id,
                    "prediction_cycles": [21, 500],
                    "predicted_soh": [0.99, 0.8],
                    "lower_soh": [0.97, 0.77],
                    "upper_soh": [1.01, 0.83],
                },
                "upstream_result_ids": [item.result_id for item in upstream[3:7]],
            },
            "markdown": "# Advanced report\n\n- Cell: `b3c34`\n",
        },
        provenance=_provenance(),
        created_at=NOW,
    )
    return (*upstream, report)


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        user_id="member-1",
        session_id="session-1",
        username="member@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _font() -> ReviewedPdfFont:
    reportlab = pytest.importorskip("reportlab")
    font_path = Path(reportlab.__file__).resolve().parent / "fonts" / "Vera.ttf"
    if not font_path.is_file():
        pytest.skip("ReportLab test font is unavailable")
    return ReviewedPdfFont(path=font_path, sha256=sha256(font_path.read_bytes()).hexdigest())


def test_service_persists_generates_and_reverifies_all_project_exports(tmp_path: Path) -> None:
    results = _results()
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'reports.sqlite3'}")
    Base.metadata.create_all(engine)
    service = ProjectReportExportService(
        create_session_factory(engine),
        run_reader=_RunReader(results),
        result_resolver=_ResultResolver(results),
        renderer=ProjectReportArtifactRenderer(pdf_font=_font()),
        artifact_root=tmp_path / "artifacts",
        export_ttl=timedelta(days=7),
    )

    pending = service.create_report(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
        now=NOW,
    )
    repeated = service.create_report(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
        now=NOW + timedelta(seconds=1),
    )

    assert pending == repeated
    assert pending.status is ReportStatus.PENDING
    assert {item.format for item in pending.exports} == set(ReportExportFormat)
    assert {item.status for item in pending.exports} == {ReportExportStatus.PENDING}

    ready = service.generate_report(pending.report_id, now=NOW + timedelta(minutes=1))
    assert ready.status is ReportStatus.READY
    assert {item.status for item in ready.exports} == {ReportExportStatus.READY}
    for item in ready.exports:
        assert item.filename
        assert item.media_type
        assert item.size_bytes and item.size_bytes > 0
        assert item.sha256 and len(item.sha256) == 64
        assert item.expires_at == NOW + timedelta(days=7, minutes=1)
        downloaded = service.download(
            _principal(),
            project_id=PROJECT_ID,
            run_id=RUN_ID,
            report_result_id=results[-1].result_id,
            format=item.format,
            now=NOW + timedelta(hours=1),
        )
        assert downloaded.payload
        assert len(downloaded.payload) == item.size_bytes
        assert sha256(downloaded.payload).hexdigest() == item.sha256
    single = service.download_tool_result(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
        result_id=results[0].result_id,
    )
    assert json.loads(single.payload) == results[0].model_dump(mode="json")
    assert single.sha256 == sha256(single.payload).hexdigest()


def test_service_rejects_wrong_project_or_incomplete_run(tmp_path: Path) -> None:
    results = _results()
    reader = _RunReader(results)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'reports.sqlite3'}")
    Base.metadata.create_all(engine)
    service = ProjectReportExportService(
        create_session_factory(engine),
        run_reader=reader,
        result_resolver=_ResultResolver(results),
        renderer=ProjectReportArtifactRenderer(pdf_font=_font()),
        artifact_root=tmp_path / "artifacts",
    )

    with pytest.raises(ProjectReportConflictError, match="project"):
        service.create_report(
            _principal(),
            project_id=str(uuid4()),
            run_id=RUN_ID,
            report_result_id=results[-1].result_id,
            now=NOW,
        )

    reader.run = _Run(RUN_ID, PROJECT_ID, AgentRunStatus.RUNNING)
    with pytest.raises(ProjectReportConflictError, match="completed"):
        service.create_report(
            _principal(),
            project_id=PROJECT_ID,
            run_id=RUN_ID,
            report_result_id=results[-1].result_id,
            now=NOW,
        )


def test_sql_result_resolver_fails_closed_without_the_exact_completed_run(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'resolver.sqlite3'}")
    Base.metadata.create_all(engine)
    resolver = SqlProjectReportResultResolver(create_session_factory(engine))

    with pytest.raises(ProjectReportNotFoundError, match="completed Agent run"):
        resolver.resolve(
            project_id=PROJECT_ID,
            run_id=RUN_ID,
            result_ids=tuple(str(uuid4()) for _ in range(9)),
        )


def test_sql_result_resolver_normalizes_duplicate_provenance_sources(
    tmp_path: Path,
) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'resolver-order.sqlite3'}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    result_id = str(uuid4())
    with session_factory.begin() as session:
        row = ToolResultRecord(
            id=result_id,
            run_id=None,
            agent_step_id=None,
            tool_name="ordered_provenance_tool",
            tool_version="tool-v1",
            model_version=None,
            data_version="data-v1",
            feature_version=None,
            input_hash=sha256_canonical({"input": "ordered"}),
            values_json={"verified": True},
            uncertainty_json=None,
            warnings_json=[],
            created_at=NOW,
        )
        session.add(row)
        session.add_all(
            (
                ProvenanceRecordRow(
                    id="00000000-0000-0000-0000-000000000001",
                    tool_result_id=result_id,
                    source_id="duplicate-source",
                    source_kind=SourceKind.PREDICTED.value,
                    uri="artifact://model/z",
                    sha256=sha256_canonical({"source": "predicted"}),
                    description="Predicted source",
                    created_at=NOW,
                ),
                ProvenanceRecordRow(
                    id="00000000-0000-0000-0000-000000000002",
                    tool_result_id=result_id,
                    source_id="duplicate-source",
                    source_kind=SourceKind.OBSERVED.value,
                    uri="dataset://observed/a",
                    sha256=sha256_canonical({"source": "observed"}),
                    description="Observed source",
                    created_at=NOW,
                ),
            )
        )

    with session_factory() as session:
        row = session.get(ToolResultRecord, result_id)
        assert row is not None
        rebuilt = SqlProjectReportResultResolver._tool_result(session, row)

    assert [item.source_kind for item in rebuilt.provenance] == [
        SourceKind.OBSERVED,
        SourceKind.PREDICTED,
    ]


def test_expired_download_is_persisted_and_fails_closed(tmp_path: Path) -> None:
    results = _results()
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'expiry.sqlite3'}")
    Base.metadata.create_all(engine)
    service = ProjectReportExportService(
        create_session_factory(engine),
        run_reader=_RunReader(results),
        result_resolver=_ResultResolver(results),
        renderer=ProjectReportArtifactRenderer(pdf_font=_font()),
        artifact_root=tmp_path / "artifacts",
        export_ttl=timedelta(minutes=5),
    )
    pending = service.create_report(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
        now=NOW,
    )
    service.generate_report(pending.report_id, now=NOW)

    with pytest.raises(ProjectReportExpiredError):
        service.download(
            _principal(),
            project_id=PROJECT_ID,
            run_id=RUN_ID,
            report_result_id=results[-1].result_id,
            format=ReportExportFormat.PDF,
            now=NOW + timedelta(minutes=5),
        )

    catalog = service.get_catalog(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
    )
    pdf = next(item for item in catalog.exports if item.format is ReportExportFormat.PDF)
    assert pdf.status is ReportExportStatus.EXPIRED


def test_duplicate_worker_delivery_observes_the_existing_running_claim(
    tmp_path: Path,
) -> None:
    results = _results()
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'duplicate.sqlite3'}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    service = ProjectReportExportService(
        session_factory,
        run_reader=_RunReader(results),
        result_resolver=_ResultResolver(results),
        renderer=ProjectReportArtifactRenderer(pdf_font=_font()),
        artifact_root=tmp_path / "artifacts",
    )
    pending = service.create_report(
        _principal(),
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=results[-1].result_id,
        now=NOW,
    )
    with session_factory.begin() as session:
        report = session.get(Report, pending.report_id)
        assert report is not None
        report.status = ReportStatus.RUNNING.value
        for export in session.query(ReportExport).filter_by(report_id=report.id):
            export.status = ReportExportStatus.RUNNING.value

    observed = service.generate_report(pending.report_id, now=NOW)

    assert observed.status is ReportStatus.RUNNING
    assert {item.status for item in observed.exports} == {ReportExportStatus.RUNNING}
