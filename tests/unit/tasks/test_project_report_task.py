from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.core import ReportStatus
from quanxin_life.infrastructure.report_queue import PROJECT_REPORT_TASK
from quanxin_life.tasks.project_reports import register_project_report_task


@dataclass(frozen=True)
class _Record:
    report_id: str
    status: ReportStatus


class _Worker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_report(self, report_id: str, *, now: datetime) -> _Record:
        assert now.tzinfo is UTC
        self.calls.append(report_id)
        return _Record(report_id=report_id, status=ReportStatus.READY)


class _App:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}
        self.callback = None

    def task(self, **options: object):
        self.options = options

        def decorator(callback):
            self.callback = callback
            return callback

        return decorator


def test_report_task_resolves_everything_from_the_report_identity() -> None:
    app = _App()
    worker = _Worker()
    callback = register_project_report_task(app=app, worker=worker)
    report_id = str(uuid4())

    result = callback(report_id=report_id)

    assert app.options == {
        "name": PROJECT_REPORT_TASK,
        "acks_late": True,
        "reject_on_worker_lost": True,
        "ignore_result": True,
    }
    assert worker.calls == [report_id]
    assert result == {"report_id": report_id, "status": "READY"}
