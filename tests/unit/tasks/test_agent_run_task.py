from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import pytest

from quanxin_life.core import AgentRunState, AgentRunStatus


@dataclass
class _Request:
    id: str


@dataclass
class _BoundTask:
    request: _Request


class _FakeCeleryApp:
    def __init__(self) -> None:
        self.options: dict[str, object] | None = None
        self.registered: Callable[..., object] | None = None

    def task(self, **options: object) -> Callable[[Callable[..., object]], Callable[..., object]]:
        self.options = options

        def decorator(function: Callable[..., object]) -> Callable[..., object]:
            self.registered = function
            return function

        return decorator


class _Worker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def execute(self, *, run_id: str, plan_hash: str) -> AgentRunState:
        self.calls.append((run_id, plan_hash))
        return AgentRunState(
            run_id=run_id,
            intent_id=str(uuid4()),
            plan_hash=plan_hash,
            status=AgentRunStatus.COMPLETED,
        )


class _RetrySignal(RuntimeError):
    pass


class _RetryingBoundTask:
    def __init__(self) -> None:
        self.request = _Request(id="delivery-busy")
        self.calls: list[tuple[type[BaseException], int, int]] = []

    def retry(
        self,
        *,
        exc: BaseException,
        countdown: int,
        max_retries: int,
    ) -> BaseException:
        self.calls.append((type(exc), countdown, max_retries))
        return _RetrySignal("retry requested")


class _BusyWorker:
    def execute(self, *, run_id: str, plan_hash: str) -> AgentRunState:
        del run_id, plan_hash
        from quanxin_life.application.agent_run_execution import (
            AgentRunExecutionBusyError,
        )

        raise AgentRunExecutionBusyError("active lease")


def test_register_agent_run_task_uses_versioned_name_and_identity_only_payload() -> None:
    from quanxin_life.infrastructure.celery_queue import AGENT_RUN_TASK
    from quanxin_life.tasks.agent_runs import register_agent_run_task

    app = _FakeCeleryApp()
    worker = _Worker()
    task = register_agent_run_task(app=app, worker=worker)
    run_id = str(uuid4())
    plan_hash = "a" * 64

    result = task(_BoundTask(request=_Request(id="delivery-1")), run_id=run_id, plan_hash=plan_hash)

    assert app.options == {
        "name": AGENT_RUN_TASK,
        "bind": True,
        "acks_late": True,
        "reject_on_worker_lost": True,
        "ignore_result": True,
    }
    assert worker.calls == [(run_id, plan_hash)]
    assert result == {"run_id": run_id, "status": AgentRunStatus.COMPLETED.value}
    assert "result_ids" not in result
    assert app.registered is task


def test_agent_run_task_retries_when_another_worker_owns_the_step_lease() -> None:
    from quanxin_life.application.agent_run_execution import AgentRunExecutionBusyError
    from quanxin_life.tasks.agent_runs import register_agent_run_task

    app = _FakeCeleryApp()
    task = register_agent_run_task(app=app, worker=_BusyWorker())
    bound = _RetryingBoundTask()

    with pytest.raises(_RetrySignal, match="retry requested"):
        task(bound, run_id=str(uuid4()), plan_hash="a" * 64)

    assert bound.calls == [(AgentRunExecutionBusyError, 5, 20)]
