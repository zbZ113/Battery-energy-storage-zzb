"""Identity-only Feishu jobs dispatched through the existing Celery queue."""

from __future__ import annotations

from typing import Protocol

from quanxin_life.infrastructure.celery_queue import AGENT_RUN_QUEUE
from quanxin_life.integrations.feishu.jobs import (
    FeishuJobDispatchReceipt,
    validate_feishu_job_id,
)

FEISHU_ANALYSIS_TASK = "quanxin_life.feishu.analysis.execute.v1"


class CeleryAsyncResult(Protocol):
    id: str


class CeleryApplication(Protocol):
    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> CeleryAsyncResult: ...


class CeleryFeishuJobQueue:
    """Reuse the shared broker/worker queue and send only a persisted job UUID."""

    def __init__(self, *, app: CeleryApplication) -> None:
        self._app = app

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        normalized_id = validate_feishu_job_id(job_id)
        result = self._app.send_task(
            FEISHU_ANALYSIS_TASK,
            kwargs={"job_id": normalized_id},
            queue=AGENT_RUN_QUEUE,
        )
        task_id = result.id
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Celery returned an invalid task identifier")
        return FeishuJobDispatchReceipt(job_id=normalized_id, task_id=task_id)


__all__ = [
    "FEISHU_ANALYSIS_TASK",
    "CeleryApplication",
    "CeleryFeishuJobQueue",
]
