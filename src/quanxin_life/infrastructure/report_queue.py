"""Celery adapter that dispatches report generation by identity only."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from quanxin_life.application.report_queue import ProjectReportDispatchReceipt

PROJECT_REPORT_TASK = "quanxin_life.project_reports.generate.v1"
PROJECT_REPORT_QUEUE = "report-exports"


class _AsyncResult(Protocol):
    id: str


class _CeleryApplication(Protocol):
    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult: ...


class CeleryProjectReportQueue:
    def __init__(self, *, app: _CeleryApplication) -> None:
        self._app = app

    def enqueue(self, *, report_id: str) -> ProjectReportDispatchReceipt:
        try:
            parsed = UUID(report_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("report_id must be a canonical UUID") from exc
        if str(parsed) != report_id:
            raise ValueError("report_id must be a canonical UUID")
        result = self._app.send_task(
            PROJECT_REPORT_TASK,
            kwargs={"report_id": report_id},
            queue=PROJECT_REPORT_QUEUE,
        )
        if not isinstance(result.id, str) or not result.id.strip():
            raise ValueError("Celery returned an invalid task identifier")
        return ProjectReportDispatchReceipt(report_id=report_id, task_id=result.id)


__all__ = [
    "PROJECT_REPORT_QUEUE",
    "PROJECT_REPORT_TASK",
    "CeleryProjectReportQueue",
]
