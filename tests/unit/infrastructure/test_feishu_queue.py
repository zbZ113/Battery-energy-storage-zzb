from __future__ import annotations

from dataclasses import dataclass

from quanxin_life.infrastructure.celery_queue import AGENT_RUN_QUEUE
from quanxin_life.infrastructure.feishu_queue import (
    FEISHU_ANALYSIS_TASK,
    CeleryFeishuJobQueue,
)


@dataclass
class _AsyncResult:
    id: str


class _App:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult:
        self.calls.append({"name": name, "kwargs": kwargs, "queue": queue})
        return _AsyncResult(id="celery-feishu-task")


def test_feishu_queue_reuses_shared_celery_queue_and_sends_only_job_identity() -> None:
    app = _App()

    receipt = CeleryFeishuJobQueue(app=app).enqueue(
        job_id="3a3c972b-a23e-42c3-af76-e39038806f13"
    )

    assert app.calls == [
        {
            "name": FEISHU_ANALYSIS_TASK,
            "kwargs": {"job_id": "3a3c972b-a23e-42c3-af76-e39038806f13"},
            "queue": AGENT_RUN_QUEUE,
        }
    ]
    assert receipt.task_id == "celery-feishu-task"
