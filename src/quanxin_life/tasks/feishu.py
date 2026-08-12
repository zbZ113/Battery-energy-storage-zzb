"""Celery task registration for durable Feishu analysis jobs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from quanxin_life.infrastructure.feishu_queue import FEISHU_ANALYSIS_TASK
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuJobBusyError,
    FeishuJobRetryableError,
    validate_feishu_job_id,
)

TaskResult = dict[str, str]
TaskCallable = Callable[..., TaskResult]
FEISHU_TASK_RETRY_COUNTDOWN_SECONDS = 60
FEISHU_TASK_MAX_RETRIES = 61


class FeishuAnalysisJobExecutor(Protocol):
    def execute(self, *, job_id: str) -> FeishuAnalysisJobStatus: ...


class CeleryTaskApplication(Protocol):
    def task(self, **options: object) -> Callable[[TaskCallable], TaskCallable]: ...


class BoundCeleryTask(Protocol):
    def retry(
        self,
        *,
        exc: BaseException,
        countdown: int,
        max_retries: int,
    ) -> BaseException: ...


def register_feishu_analysis_task(
    *,
    app: CeleryTaskApplication,
    worker: FeishuAnalysisJobExecutor,
) -> TaskCallable:
    """Register one late-acknowledged identity-only task on the shared worker."""

    @app.task(
        name=FEISHU_ANALYSIS_TASK,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def execute_feishu_analysis(
        bound_task: BoundCeleryTask,
        *,
        job_id: str,
    ) -> TaskResult:
        normalized_id = validate_feishu_job_id(job_id)
        try:
            status = worker.execute(job_id=normalized_id)
        except (FeishuJobBusyError, FeishuJobRetryableError) as exc:
            raise bound_task.retry(
                exc=exc,
                countdown=FEISHU_TASK_RETRY_COUNTDOWN_SECONDS,
                max_retries=FEISHU_TASK_MAX_RETRIES,
            ) from exc
        return {"job_id": normalized_id, "status": status.value}

    return execute_feishu_analysis


__all__ = [
    "FEISHU_TASK_MAX_RETRIES",
    "FEISHU_TASK_RETRY_COUNTDOWN_SECONDS",
    "FeishuAnalysisJobExecutor",
    "register_feishu_analysis_task",
]
