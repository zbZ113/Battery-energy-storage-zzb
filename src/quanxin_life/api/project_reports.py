"""FastAPI adapter for project-authorized asynchronous report exports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.project_reports import (
    ProjectReportConflictError,
    ProjectReportExpiredError,
    ProjectReportExportService,
    ProjectReportGenerationError,
    ProjectReportNotFoundError,
    ProjectReportRecord,
)
from quanxin_life.application.report_queue import ProjectReportQueue
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import ReportExportFormat, UserRole

_SAFE_FILENAME = re.compile(r"[A-Za-z0-9_.-]{1,255}\Z")


@dataclass(frozen=True, slots=True)
class ProjectReportHttpAdapter:
    router: APIRouter


def create_project_report_http_adapter(
    service: ProjectReportExportService,
    *,
    auth_adapter: AuthHttpAdapter,
    queue: ProjectReportQueue,
) -> ProjectReportHttpAdapter:
    """Build report create, status and download routes over project-visible runs."""

    router = APIRouter(
        prefix=(
            "/v1/projects/{project_id}/agent/runs/{run_id}/reports/"
            "{report_result_id}/exports"
        ),
        tags=["project-report-exports"],
    )
    operator = Depends(auth_adapter.require_roles({UserRole.ADMIN, UserRole.MEMBER}))
    ready_user = Depends(auth_adapter.require_ready_user)

    @router.post(
        "",
        response_model=ProjectReportRecord,
        status_code=202,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_exports(
        project_id: str,
        run_id: str,
        report_result_id: str,
        principal: AuthPrincipal = operator,
    ) -> Any:
        try:
            record = service.create_report(
                principal,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
                now=datetime.now(UTC),
            )
            queue.enqueue(report_id=record.report_id)
            return record
        except ProjectReportNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_report_not_found") from exc
        except ProjectReportConflictError as exc:
            raise HTTPException(status_code=409, detail="project_report_conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_project_report") from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="report_dispatch_unavailable") from exc

    @router.get("", response_model=ProjectReportRecord)
    def get_exports(
        project_id: str,
        run_id: str,
        report_result_id: str,
        principal: AuthPrincipal = ready_user,
    ) -> Any:
        try:
            return service.get_catalog(
                principal,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
            )
        except ProjectReportNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_report_not_found") from exc
        except ProjectReportConflictError as exc:
            raise HTTPException(status_code=409, detail="project_report_conflict") from exc

    @router.get("/tool-results/{result_id}.json")
    def download_tool_result(
        project_id: str,
        run_id: str,
        report_result_id: str,
        result_id: str,
        principal: AuthPrincipal = ready_user,
    ) -> Response:
        try:
            artifact = service.download_tool_result(
                principal,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
                result_id=result_id,
            )
        except ProjectReportNotFoundError as exc:
            raise HTTPException(status_code=404, detail="tool_result_not_found") from exc
        except ProjectReportConflictError as exc:
            raise HTTPException(status_code=409, detail="tool_result_unavailable") from exc
        if _SAFE_FILENAME.fullmatch(artifact.filename) is None:
            raise HTTPException(status_code=409, detail="tool_result_unavailable")
        return Response(
            content=artifact.payload,
            media_type=artifact.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{artifact.filename}"',
                "ETag": f'"sha256:{artifact.sha256}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @router.get("/{export_format}")
    def download_export(
        project_id: str,
        run_id: str,
        report_result_id: str,
        export_format: str,
        principal: AuthPrincipal = ready_user,
    ) -> Response:
        try:
            selected = ReportExportFormat(export_format)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="unsupported_report_format") from exc
        try:
            downloaded = service.download(
                principal,
                project_id=project_id,
                run_id=run_id,
                report_result_id=report_result_id,
                format=selected,
                now=datetime.now(UTC),
            )
        except ProjectReportNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_report_not_found") from exc
        except ProjectReportExpiredError as exc:
            raise HTTPException(status_code=410, detail="report_export_expired") from exc
        except ProjectReportConflictError as exc:
            raise HTTPException(status_code=409, detail="report_export_unavailable") from exc
        except ProjectReportGenerationError as exc:
            raise HTTPException(status_code=503, detail="report_export_unavailable") from exc
        record = downloaded.record
        if (
            record.filename is None
            or record.media_type is None
            or record.sha256 is None
            or _SAFE_FILENAME.fullmatch(record.filename) is None
        ):
            raise HTTPException(status_code=409, detail="report_export_unavailable")
        return Response(
            content=downloaded.payload,
            media_type=record.media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{record.filename}"',
                "ETag": f'"sha256:{record.sha256}"',
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    return ProjectReportHttpAdapter(router=router)


__all__ = ["ProjectReportHttpAdapter", "create_project_report_http_adapter"]
