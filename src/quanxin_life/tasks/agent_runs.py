"""Celery task registration for durable Agent run execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from quanxin_life.core import AgentRunState
from quanxin_life.infrastructure.celery_queue import AGENT_RUN_TASK

TaskResult = dict[str, str]
TaskCallable = Callable[..., TaskResult]


class AgentRunExecutor(Protocol):
    def execute(self, *, run_id: str, plan_hash: str) -> AgentRunState: ...


class CeleryTaskApplication(Protocol):
    def task(self, **options: object) -> Callable[[TaskCallable], TaskCallable]: ...


class BoundCeleryTask(Protocol):
    """Only the non-sensitive delivery identity is available from Celery."""


def register_agent_run_task(
    *,
    app: CeleryTaskApplication,
    worker: AgentRunExecutor,
) -> TaskCallable:
    """Register the versioned identity-only task without import-time startup."""

    @app.task(
        name=AGENT_RUN_TASK,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def execute_agent_run(
        bound_task: BoundCeleryTask,
        *,
        run_id: str,
        plan_hash: str,
    ) -> TaskResult:
        del bound_task
        state = worker.execute(run_id=run_id, plan_hash=plan_hash)
        return {"run_id": state.run_id, "status": state.status.value}

    return execute_agent_run


__all__ = ["AgentRunExecutor", "register_agent_run_task"]
