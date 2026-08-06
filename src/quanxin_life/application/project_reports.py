"""Project-authorized report persistence and verified artifact downloads."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentRunStatus,
    AgentStepStatus,
    ReportExportFormat,
    ReportExportStatus,
    ReportStatus,
    ToolResult,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentRun,
    AgentStep,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    Report,
    ReportExport,
    SessionRecord,
    ToolResultRecord,
)
from quanxin_life.reporting.project_artifacts import (
    PROJECT_REPORT_BUNDLE_VERSION,
    ProjectReportArtifact,
    ProjectReportArtifactRenderer,
)


class ProjectReportError(RuntimeError):
    """Base error for project report state and integrity failures."""


class ProjectReportNotFoundError(ProjectReportError):
    pass


class ProjectReportConflictError(ProjectReportError):
    pass


class ProjectReportGenerationError(ProjectReportError):
    pass


class ProjectReportExpiredError(ProjectReportError):
    pass


class _VisibleRun(Protocol):
    run_id: str
    project_id: str
    status: AgentRunStatus


class _VisibleRunResult(Protocol):
    ordinal: int
    result: ToolResult


class ProjectReportRunReader(Protocol):
    def get_run(self, principal: AuthPrincipal, run_id: str) -> _VisibleRun: ...

    def list_results(
        self,
        principal: AuthPrincipal,
        run_id: str,
    ) -> Sequence[_VisibleRunResult]: ...


class ProjectReportResultResolver(Protocol):
    def resolve(
        self,
        *,
        project_id: str,
        run_id: str,
        result_ids: tuple[str, ...],
    ) -> tuple[ToolResult, ...]: ...


class SqlProjectReportResultResolver:
    """Rebuild a completed nine-step result set and verify every SQL binding."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def resolve(
        self,
        *,
        project_id: str,
        run_id: str,
        result_ids: tuple[str, ...],
    ) -> tuple[ToolResult, ...]:
        if len(result_ids) != 9 or len(set(result_ids)) != 9:
            raise ProjectReportConflictError(
                "report requires nine unique ToolResult identities"
            )
        with session_scope(self._session_factory) as session:
            run = session.get(AgentRun, run_id)
            if (
                run is None
                or run.project_id != project_id
                or run.status != AgentRunStatus.COMPLETED.value
                or run.session_id is None
                or run.plan_hash is None
            ):
                raise ProjectReportNotFoundError(
                    "exact completed Agent run was not found"
                )
            steps = tuple(
                session.scalars(
                    select(AgentStep)
                    .where(AgentStep.run_id == run_id)
                    .order_by(AgentStep.ordinal)
                ).all()
            )
            if (
                len(steps) != 9
                or tuple(step.ordinal for step in steps) != tuple(range(1, 10))
                or any(
                    step.status != AgentStepStatus.COMPLETED.value for step in steps
                )
            ):
                raise ProjectReportConflictError(
                    "persisted Agent run does not contain nine completed steps"
                )
            rows = tuple(
                session.scalars(
                    select(ToolResultRecord).where(
                        ToolResultRecord.run_id == run_id,
                        ToolResultRecord.id.in_(result_ids),
                    )
                ).all()
            )
            rows_by_step = {
                row.agent_step_id: row
                for row in rows
                if row.agent_step_id is not None
            }
            if len(rows) != 9 or len(rows_by_step) != 9:
                raise ProjectReportConflictError(
                    "persisted Agent run does not contain nine bound ToolResults"
                )
            resolved: list[ToolResult] = []
            for expected_id, step in zip(result_ids, steps, strict=True):
                row = rows_by_step.get(step.id)
                binding = session.get(ProjectToolResultBindingRecord, expected_id)
                if (
                    row is None
                    or row.id != expected_id
                    or row.tool_name != step.tool_name
                    or binding is None
                    or binding.project_id != project_id
                    or binding.agent_run_id != run_id
                    or binding.agent_step_id != step.id
                    or binding.step_id != step.step_id
                    or binding.actor_user_id != run.created_by_user_id
                    or binding.actor_session_id != run.session_id
                    or binding.plan_hash != run.plan_hash
                    or row.agent_step_id != binding.agent_step_id
                    or step.execution_claim_sha256 != binding.claim_token_sha256
                    or step.resolved_input_hash != binding.input_hash
                    or step.execution_snapshot_sha256
                    != binding.execution_snapshot_sha256
                    or step.dependency_evidence_sha256
                    != binding.dependency_evidence_sha256
                ):
                    raise ProjectReportConflictError(
                        "project ToolResult binding integrity check failed"
                    )
                actor_session = session.get(SessionRecord, binding.actor_session_id)
                if (
                    actor_session is None
                    or actor_session.user_id != binding.actor_user_id
                ):
                    raise ProjectReportConflictError(
                        "project ToolResult actor binding is invalid"
                    )
                result = self._tool_result(session, row)
                binding.created_at = _database_utc(binding.created_at)
                try:
                    SqlProjectAuditLedger.verify_persisted_binding(binding, result)
                except RuntimeError as exc:
                    raise ProjectReportConflictError(
                        "project ToolResult hash binding is invalid"
                    ) from exc
                resolved.append(result)
            return tuple(resolved)

    @staticmethod
    def _tool_result(session: Session, row: ToolResultRecord) -> ToolResult:
        provenance_rows = tuple(
            session.scalars(
                select(ProvenanceRecordRow)
                .where(ProvenanceRecordRow.tool_result_id == row.id)
            ).all()
        )
        try:
            return SqlProjectAuditLedger.rebuild_persisted_result(
                row,
                provenance_rows,
            )
        except (TypeError, ValueError) as exc:
            raise ProjectReportConflictError(
                "persisted ToolResult does not satisfy its contract"
            ) from exc


class ReportExportRecord(ContractModel):
    export_id: str
    report_id: str
    format: ReportExportFormat
    status: ReportExportStatus
    filename: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=1)
    sha256: Sha256 | None = None
    created_at: datetime
    completed_at: datetime | None = None
    expires_at: datetime | None = None


class ProjectReportRecord(ContractModel):
    report_id: str
    project_id: str
    run_id: str
    report_result_id: str
    status: ReportStatus
    template_version: str
    created_at: datetime
    completed_at: datetime | None = None
    exports: tuple[ReportExportRecord, ...]


@dataclass(frozen=True, slots=True)
class DownloadedProjectReportArtifact:
    record: ReportExportRecord
    payload: bytes


@dataclass(frozen=True, slots=True)
class DownloadedToolResultArtifact:
    result_id: str
    filename: str
    media_type: str
    sha256: str
    payload: bytes


class ProjectReportExportService:
    """Create one report per completed run and serve only verified derivatives."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        run_reader: ProjectReportRunReader,
        result_resolver: ProjectReportResultResolver,
        renderer: ProjectReportArtifactRenderer,
        artifact_root: Path,
        export_ttl: timedelta = timedelta(days=7),
    ) -> None:
        if export_ttl <= timedelta(0):
            raise ValueError("report export TTL must be positive")
        root = artifact_root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir() or root.is_symlink():
            raise ValueError("report artifact root must be a real directory")
        self._session_factory = session_factory
        self._run_reader = run_reader
        self._result_resolver = result_resolver
        self._renderer = renderer
        self._artifact_root = root
        self._export_ttl = export_ttl

    def create_report(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        run_id: str,
        report_result_id: str,
        now: datetime,
    ) -> ProjectReportRecord:
        timestamp = _utc(now)
        run = self._run_reader.get_run(principal, run_id)
        if run.project_id != project_id:
            raise ProjectReportConflictError("Agent run does not belong to the project")
        if AgentRunStatus(run.status) is not AgentRunStatus.COMPLETED:
            raise ProjectReportConflictError("Agent run must be completed before export")
        catalog = tuple(
            sorted(
                self._run_reader.list_results(principal, run_id),
                key=lambda item: item.ordinal,
            )
        )
        if tuple(item.ordinal for item in catalog) != tuple(range(1, 10)):
            raise ProjectReportConflictError(
                "completed report requires exactly nine ordered results"
            )
        results = tuple(item.result for item in catalog)
        if results[-1].result_id != report_result_id:
            raise ProjectReportConflictError("report result must be the ninth Agent result")
        self._renderer.render(
            ReportExportFormat.TOOL_RESULTS_JSON,
            results=results,
            generated_at=timestamp,
        )
        result_ids = [result.result_id for result in results]
        with session_scope(self._session_factory) as session:
            existing = session.scalar(select(Report).where(Report.run_id == run_id))
            if existing is not None:
                if (
                    existing.project_id != project_id
                    or existing.result_ids_json != result_ids
                    or existing.template_version != PROJECT_REPORT_BUNDLE_VERSION
                ):
                    raise ProjectReportConflictError(
                        "persisted report does not match the completed Agent run"
                    )
                return self._record(session, existing)
            report = Report(
                id=str(uuid4()),
                project_id=project_id,
                run_id=run_id,
                status=ReportStatus.PENDING.value,
                template_version=PROJECT_REPORT_BUNDLE_VERSION,
                result_ids_json=result_ids,
                object_uri=None,
                sha256=None,
                failure_code=None,
                created_at=timestamp,
                updated_at=timestamp,
                completed_at=None,
            )
            session.add(report)
            session.flush((report,))
            for export_format in ReportExportFormat:
                session.add(
                    ReportExport(
                        id=str(uuid4()),
                        report_id=report.id,
                        export_format=export_format.value,
                        status=ReportExportStatus.PENDING.value,
                        object_uri=None,
                        sha256=None,
                        filename=None,
                        media_type=None,
                        size_bytes=None,
                        failure_code=None,
                        created_at=timestamp,
                        completed_at=None,
                        expires_at=None,
                    )
                )
            session.flush()
            return self._record(session, report)

    def generate_report(self, report_id: str, *, now: datetime) -> ProjectReportRecord:
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            report = session.scalar(
                select(Report).where(Report.id == report_id).with_for_update()
            )
            if report is None:
                raise ProjectReportNotFoundError("project report was not found")
            persisted_status = ReportStatus(report.status)
            if persisted_status in {ReportStatus.READY, ReportStatus.RUNNING}:
                return self._record(session, report)
            if persisted_status is not ReportStatus.PENDING:
                raise ProjectReportConflictError("project report is not pending")
            report.status = ReportStatus.RUNNING.value
            report.updated_at = timestamp
            exports = self._exports(session, report.id)
            for row in exports:
                row.status = ReportExportStatus.RUNNING.value
                row.failure_code = None
            project_id = report.project_id
            run_id = report.run_id
            result_ids = tuple(report.result_ids_json)
            if run_id is None:
                raise ProjectReportConflictError("project report has no Agent run")
        try:
            results = self._result_resolver.resolve(
                project_id=project_id,
                run_id=run_id,
                result_ids=result_ids,
            )
            artifacts = {
                export_format: self._renderer.render(
                    export_format,
                    results=results,
                    generated_at=timestamp,
                )
                for export_format in ReportExportFormat
            }
            stored = {
                export_format: self._store(report_id, artifact)
                for export_format, artifact in artifacts.items()
            }
        except Exception as exc:
            self._mark_failed(report_id, now=timestamp)
            raise ProjectReportGenerationError("project report generation failed") from exc
        expires_at = timestamp + self._export_ttl
        with session_scope(self._session_factory) as session:
            report = session.scalar(
                select(Report).where(Report.id == report_id).with_for_update()
            )
            if report is None or ReportStatus(report.status) is not ReportStatus.RUNNING:
                raise ProjectReportConflictError("project report generation claim was lost")
            rows = self._exports(session, report.id)
            rows_by_format = {ReportExportFormat(row.export_format): row for row in rows}
            for export_format, artifact in artifacts.items():
                row = rows_by_format[export_format]
                row.status = ReportExportStatus.READY.value
                row.object_uri = str(stored[export_format])
                row.sha256 = artifact.sha256
                row.filename = artifact.filename
                row.media_type = artifact.media_type
                row.size_bytes = len(artifact.payload)
                row.failure_code = None
                row.completed_at = timestamp
                row.expires_at = expires_at
            zip_row = rows_by_format[ReportExportFormat.ZIP]
            report.status = ReportStatus.READY.value
            report.object_uri = zip_row.object_uri
            report.sha256 = zip_row.sha256
            report.failure_code = None
            report.updated_at = timestamp
            report.completed_at = timestamp
            session.flush()
            return self._record(session, report)

    def get_catalog(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        run_id: str,
        report_result_id: str,
    ) -> ProjectReportRecord:
        self._authorize(principal, project_id=project_id, run_id=run_id)
        with session_scope(self._session_factory) as session:
            report = self._find_report(
                session,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
            )
            return self._record(session, report)

    def download(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        run_id: str,
        report_result_id: str,
        format: ReportExportFormat,
        now: datetime,
    ) -> DownloadedProjectReportArtifact:
        timestamp = _utc(now)
        self._authorize(principal, project_id=project_id, run_id=run_id)
        expired = False
        downloaded: DownloadedProjectReportArtifact | None = None
        with session_scope(self._session_factory) as session:
            report = self._find_report(
                session,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
            )
            row = session.scalar(
                select(ReportExport).where(
                    ReportExport.report_id == report.id,
                    ReportExport.export_format == format.value,
                )
            )
            if row is None:
                raise ProjectReportNotFoundError("report export was not found")
            if row.expires_at is not None and _utc(row.expires_at) <= timestamp:
                row.status = ReportExportStatus.EXPIRED.value
                expired = True
            elif (
                ReportExportStatus(row.status) is not ReportExportStatus.READY
                or row.object_uri is None
                or row.sha256 is None
                or row.filename is None
                or row.media_type is None
                or row.size_bytes is None
            ):
                raise ProjectReportConflictError("report export is not ready")
            else:
                path = Path(row.object_uri).resolve(strict=True)
                if not path.is_relative_to(self._artifact_root) or not path.is_file():
                    raise ProjectReportConflictError(
                        "report export storage boundary is invalid"
                    )
                payload = path.read_bytes()
                if (
                    len(payload) != row.size_bytes
                    or sha256(payload).hexdigest() != row.sha256
                ):
                    raise ProjectReportConflictError(
                        "report export integrity check failed"
                    )
                downloaded = DownloadedProjectReportArtifact(
                    record=self._export_record(row),
                    payload=payload,
                )
        if expired:
            raise ProjectReportExpiredError("report export has expired")
        if downloaded is None:  # pragma: no cover - guarded by branches above
            raise ProjectReportConflictError("report export is unavailable")
        return downloaded

    def download_tool_result(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        run_id: str,
        report_result_id: str,
        result_id: str,
    ) -> DownloadedToolResultArtifact:
        self._authorize(principal, project_id=project_id, run_id=run_id)
        with session_scope(self._session_factory) as session:
            report = self._find_report(
                session,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
            )
            persisted_ids = tuple(report.result_ids_json)
        catalog = tuple(
            sorted(
                self._run_reader.list_results(principal, run_id),
                key=lambda item: item.ordinal,
            )
        )
        results = tuple(item.result for item in catalog)
        if tuple(result.result_id for result in results) != persisted_ids:
            raise ProjectReportConflictError(
                "Agent ToolResult catalog no longer matches the report"
            )
        selected = next(
            (result for result in results if result.result_id == result_id),
            None,
        )
        if selected is None:
            raise ProjectReportNotFoundError("ToolResult export was not found")
        artifact = self._renderer.render_tool_result_json(selected)
        return DownloadedToolResultArtifact(
            result_id=selected.result_id,
            filename=artifact.filename,
            media_type=artifact.media_type,
            sha256=artifact.sha256,
            payload=artifact.payload,
        )

    def _authorize(self, principal: AuthPrincipal, *, project_id: str, run_id: str) -> None:
        run = self._run_reader.get_run(principal, run_id)
        if run.project_id != project_id:
            raise ProjectReportNotFoundError("project report was not found")

    def _find_report(
        self,
        session: Session,
        *,
        project_id: str,
        run_id: str,
        report_result_id: str,
    ) -> Report:
        row = session.scalar(
            select(Report).where(
                Report.project_id == project_id,
                Report.run_id == run_id,
            )
        )
        if row is None or not row.result_ids_json or row.result_ids_json[-1] != report_result_id:
            raise ProjectReportNotFoundError("project report was not found")
        return row

    def _store(self, report_id: str, artifact: ProjectReportArtifact) -> Path:
        directory = (self._artifact_root / "report-exports" / report_id).resolve()
        if not directory.is_relative_to(self._artifact_root):
            raise ValueError("report artifact path escaped its root")
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / artifact.filename
        temporary = directory / f".{artifact.filename}.{uuid4()}.tmp"
        temporary.write_bytes(artifact.payload)
        temporary.replace(destination)
        return destination.resolve(strict=True)

    def _mark_failed(self, report_id: str, *, now: datetime) -> None:
        with session_scope(self._session_factory) as session:
            report = session.get(Report, report_id)
            if report is None:
                return
            report.status = ReportStatus.FAILED.value
            report.failure_code = "REPORT_EXPORT_GENERATION_FAILED"
            report.updated_at = now
            report.completed_at = now
            for row in self._exports(session, report.id):
                row.status = ReportExportStatus.FAILED.value
                row.failure_code = "REPORT_EXPORT_GENERATION_FAILED"
                row.completed_at = now

    @staticmethod
    def _exports(session: Session, report_id: str) -> tuple[ReportExport, ...]:
        return tuple(
            session.scalars(
                select(ReportExport)
                .where(ReportExport.report_id == report_id)
                .order_by(ReportExport.export_format)
            ).all()
        )

    def _record(self, session: Session, report: Report) -> ProjectReportRecord:
        if report.run_id is None or not report.result_ids_json:
            raise ProjectReportConflictError("persisted project report is incomplete")
        return ProjectReportRecord(
            report_id=report.id,
            project_id=report.project_id,
            run_id=report.run_id,
            report_result_id=report.result_ids_json[-1],
            status=ReportStatus(report.status),
            template_version=report.template_version,
            created_at=_utc(report.created_at),
            completed_at=(
                _utc(report.completed_at) if report.completed_at is not None else None
            ),
            exports=tuple(
                self._export_record(row) for row in self._exports(session, report.id)
            ),
        )

    @staticmethod
    def _export_record(row: ReportExport) -> ReportExportRecord:
        return ReportExportRecord(
            export_id=row.id,
            report_id=row.report_id,
            format=ReportExportFormat(row.export_format),
            status=ReportExportStatus(row.status),
            filename=row.filename,
            media_type=row.media_type,
            size_bytes=row.size_bytes,
            sha256=row.sha256,
            created_at=_utc(row.created_at),
            completed_at=_utc(row.completed_at) if row.completed_at is not None else None,
            expires_at=_utc(row.expires_at) if row.expires_at is not None else None,
        )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("project report timestamp must include a timezone")
    return value.astimezone(UTC)


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "DownloadedProjectReportArtifact",
    "DownloadedToolResultArtifact",
    "ProjectReportConflictError",
    "ProjectReportError",
    "ProjectReportExpiredError",
    "ProjectReportExportService",
    "ProjectReportGenerationError",
    "ProjectReportNotFoundError",
    "ProjectReportRecord",
    "ReportExportRecord",
    "SqlProjectReportResultResolver",
]
