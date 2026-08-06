from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.project_reports import create_project_report_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.application.project_reports import (
    DownloadedProjectReportArtifact,
    DownloadedToolResultArtifact,
    ProjectReportNotFoundError,
    ProjectReportRecord,
    ReportExportRecord,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    ReportExportFormat,
    ReportExportStatus,
    ReportStatus,
    UserRole,
)

NOW = datetime(2026, 8, 3, 10, 30, tzinfo=UTC)
PROJECT_ID = str(uuid4())
RUN_ID = str(uuid4())
REPORT_RESULT_ID = str(uuid4())
REPORT_ID = str(uuid4())


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        user_id="member-1",
        session_id="session-1",
        username="member@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _record(status: ReportExportStatus = ReportExportStatus.READY) -> ProjectReportRecord:
    expires = NOW + timedelta(days=7)
    exports = tuple(
        ReportExportRecord(
            export_id=str(uuid4()),
            report_id=REPORT_ID,
            format=format,
            status=status,
            filename=f"report.{format.value}",
            media_type="application/octet-stream",
            size_bytes=2 if status is ReportExportStatus.READY else None,
            sha256="a" * 64 if status is ReportExportStatus.READY else None,
            created_at=NOW,
            completed_at=NOW if status is ReportExportStatus.READY else None,
            expires_at=expires,
        )
        for format in ReportExportFormat
    )
    return ProjectReportRecord(
        report_id=REPORT_ID,
        project_id=PROJECT_ID,
        run_id=RUN_ID,
        report_result_id=REPORT_RESULT_ID,
        status=ReportStatus.READY,
        template_version="project-report-bundle-v1",
        created_at=NOW,
        completed_at=NOW,
        exports=exports,
    )


class _Auth:
    router = APIRouter()
    allowed_origins = ("https://app.example.test",)

    @staticmethod
    def require_ready_user() -> AuthPrincipal:
        return _principal()

    @staticmethod
    def require_roles(_roles: set[UserRole]) -> Any:
        return _Auth.require_ready_user

    @staticmethod
    def require_trusted_origin() -> None:
        return None


@dataclass
class _Queue:
    report_ids: list[str]

    def enqueue(self, *, report_id: str) -> object:
        self.report_ids.append(report_id)
        return object()


class _Service:
    def __init__(self) -> None:
        self.catalog = _record()
        self.downloaded = DownloadedProjectReportArtifact(
            record=self.catalog.exports[0],
            payload=b"ok",
        )

    def create_report(self, principal: AuthPrincipal, **kwargs: object) -> ProjectReportRecord:
        assert principal == _principal()
        assert kwargs["project_id"] == PROJECT_ID
        assert kwargs["run_id"] == RUN_ID
        assert kwargs["report_result_id"] == REPORT_RESULT_ID
        return self.catalog.model_copy(update={"status": ReportStatus.PENDING})

    def get_catalog(self, principal: AuthPrincipal, **kwargs: object) -> ProjectReportRecord:
        assert principal == _principal()
        return self.catalog

    def download(
        self,
        principal: AuthPrincipal,
        **kwargs: object,
    ) -> DownloadedProjectReportArtifact:
        assert principal == _principal()
        assert kwargs["format"] is ReportExportFormat.MARKDOWN
        return self.downloaded

    def download_tool_result(
        self,
        principal: AuthPrincipal,
        **kwargs: object,
    ) -> DownloadedToolResultArtifact:
        assert principal == _principal()
        result_id = str(kwargs["result_id"])
        return DownloadedToolResultArtifact(
            result_id=result_id,
            filename=f"tool-result-{result_id}.json",
            media_type="application/json",
            sha256="b" * 64,
            payload=b'{"result_id":"verified"}\n',
        )


def _client() -> tuple[TestClient, _Queue, _Service]:
    queue = _Queue([])
    service = _Service()
    app = create_fastapi_app(
        service=create_available_tool_invocation_service(),
        auth_adapter=_Auth(),
        project_report_adapter=create_project_report_http_adapter(
            service,
            auth_adapter=_Auth(),
            queue=queue,
        ),
    )
    return TestClient(app), queue, service


def test_project_report_api_creates_catalog_and_streams_verified_download() -> None:
    client, queue, _service = _client()
    base = f"/v1/projects/{PROJECT_ID}/agent/runs/{RUN_ID}/reports/{REPORT_RESULT_ID}/exports"

    created = client.post(base, headers={"Origin": "https://app.example.test"})
    assert created.status_code == 202
    assert created.json()["status"] == "PENDING"
    assert queue.report_ids == [REPORT_ID]

    catalog = client.get(base)
    assert catalog.status_code == 200
    assert catalog.json()["project_id"] == PROJECT_ID
    assert len(catalog.json()["exports"]) == len(ReportExportFormat)

    download = client.get(f"{base}/markdown")
    assert download.status_code == 200
    assert download.content == b"ok"
    assert download.headers["etag"].startswith('"sha256:')
    assert "attachment;" in download.headers["content-disposition"]
    assert download.headers["x-content-type-options"] == "nosniff"
    assert download.headers["cache-control"] == "private, no-store"

    result_id = str(uuid4())
    single = client.get(f"{base}/tool-results/{result_id}.json")
    assert single.status_code == 200
    assert single.json() == {"result_id": "verified"}
    assert single.headers["etag"] == f'"sha256:{"b" * 64}"'


def test_project_report_api_rejects_unknown_or_missing_exports() -> None:
    client, _queue, service = _client()
    base = f"/v1/projects/{PROJECT_ID}/agent/runs/{RUN_ID}/reports/{REPORT_RESULT_ID}/exports"

    assert client.get(f"{base}/exe").status_code == 422
    service.get_catalog = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        ProjectReportNotFoundError("missing")
    )
    assert client.get(base).status_code == 404
