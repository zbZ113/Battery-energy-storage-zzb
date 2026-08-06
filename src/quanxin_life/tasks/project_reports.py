"""Celery task registration for identity-only project report generation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol

from quanxin_life.core import ReportStatus
from quanxin_life.infrastructure.report_queue import PROJECT_REPORT_TASK

TaskResult = dict[str, str]
TaskCallable = Callable[..., TaskResult]


class _GeneratedReport(Protocol):
    report_id: str
    status: ReportStatus


class ProjectReportGenerator(Protocol):
    def generate_report(self, report_id: str, *, now: datetime) -> _GeneratedReport: ...


class CeleryTaskApplication(Protocol):
    def task(self, **options: object) -> Callable[[TaskCallable], TaskCallable]: ...


def register_project_report_task(
    *,
    app: CeleryTaskApplication,
    worker: ProjectReportGenerator,
) -> TaskCallable:
    @app.task(
        name=PROJECT_REPORT_TASK,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def generate_project_report(*, report_id: str) -> TaskResult:
        record = worker.generate_report(report_id, now=datetime.now(UTC))
        return {"report_id": record.report_id, "status": record.status.value}

    return generate_project_report


__all__ = ["ProjectReportGenerator", "register_project_report_task"]
