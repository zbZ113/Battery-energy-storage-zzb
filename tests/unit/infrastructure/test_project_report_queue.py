from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from quanxin_life.infrastructure.report_queue import (
    PROJECT_REPORT_QUEUE,
    PROJECT_REPORT_TASK,
    CeleryProjectReportQueue,
)


@dataclass(frozen=True)
class _AsyncResult:
    id: str


class _App:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def send_task(self, name: str, **kwargs: object) -> _AsyncResult:
        self.calls.append({"name": name, **kwargs})
        return _AsyncResult("report-task-1")


def test_report_queue_dispatches_only_a_canonical_report_identity() -> None:
    app = _App()
    queue = CeleryProjectReportQueue(app=app)
    report_id = str(uuid4())

    receipt = queue.enqueue(report_id=report_id)

    assert receipt.report_id == report_id
    assert receipt.task_id == "report-task-1"
    assert app.calls == [
        {
            "name": PROJECT_REPORT_TASK,
            "kwargs": {"report_id": report_id},
            "queue": PROJECT_REPORT_QUEUE,
        }
    ]


@pytest.mark.parametrize("report_id", ["", "not-a-uuid", "A" * 64])
def test_report_queue_rejects_noncanonical_report_id(report_id: str) -> None:
    with pytest.raises(ValueError, match="canonical UUID"):
        CeleryProjectReportQueue(app=_App()).enqueue(report_id=report_id)
