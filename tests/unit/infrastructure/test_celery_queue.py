from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from quanxin_life.application.task_queue import AgentRunDispatchReceipt
from quanxin_life.infrastructure.celery_queue import (
    AGENT_RUN_QUEUE,
    AGENT_RUN_TASK,
    CeleryAgentRunQueue,
    CeleryQueueConfig,
    create_celery_app,
)


@dataclass
class _AsyncResult:
    id: str


class _FakeCeleryApp:
    def __init__(self) -> None:
        self.conf: dict[str, object] = {}
        self.calls: list[dict[str, object]] = []

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult:
        self.calls.append({"name": name, "kwargs": kwargs, "queue": queue})
        return _AsyncResult(id="task-123")


def test_config_redacts_redis_url_credentials() -> None:
    config = CeleryQueueConfig(
        broker_url="redis://worker:secret-password@redis:6379/0",
    )

    rendered = repr(config) + str(config)
    assert "secret-password" not in rendered
    assert "worker" not in rendered


@pytest.mark.parametrize(
    "broker_url",
    ["", "amqp://rabbitmq:5672", "http://redis:6379", "redis://redis:6379/not-a-db"],
)
def test_config_rejects_missing_or_non_redis_broker(broker_url: str) -> None:
    with pytest.raises(ValidationError):
        CeleryQueueConfig(broker_url=broker_url)


def test_factory_configures_json_only_utc_worker_without_connecting() -> None:
    created: dict[str, object] = {}
    fake_app = _FakeCeleryApp()

    def factory(name: str, *, broker: str) -> _FakeCeleryApp:
        created["name"] = name
        created["broker"] = broker
        return fake_app

    app = create_celery_app(
        CeleryQueueConfig(broker_url="redis://redis:6379/0"),
        app_factory=factory,
    )

    assert app is fake_app
    assert created == {"name": "quanxin-life", "broker": "redis://redis:6379/0"}
    assert fake_app.conf == {
        "accept_content": ["json"],
        "task_serializer": "json",
        "result_serializer": "json",
        "enable_utc": True,
        "timezone": "UTC",
        "task_ignore_result": True,
        "task_acks_late": True,
        "task_reject_on_worker_lost": True,
        "worker_prefetch_multiplier": 1,
        "task_default_queue": AGENT_RUN_QUEUE,
    }


def test_queue_sends_only_run_identity_and_plan_hash() -> None:
    app = _FakeCeleryApp()
    queue = CeleryAgentRunQueue(app=app)
    run_id = "5f42f320-32cf-4db5-8fd8-55d479a01b21"
    plan_hash = "a" * 64

    receipt = queue.enqueue(run_id=run_id, plan_hash=plan_hash)

    assert receipt == AgentRunDispatchReceipt(run_id=run_id, task_id="task-123")
    assert app.calls == [
        {
            "name": AGENT_RUN_TASK,
            "kwargs": {"run_id": run_id, "plan_hash": plan_hash},
            "queue": AGENT_RUN_QUEUE,
        }
    ]


@pytest.mark.parametrize(
    ("run_id", "plan_hash"),
    [
        ("not-a-uuid", "a" * 64),
        ("5f42f320-32cf-4db5-8fd8-55d479a01b21", "not-a-hash"),
        ("5F42F320-32CF-4DB5-8FD8-55D479A01B21", "a" * 64),
    ],
)
def test_queue_rejects_invalid_identifiers_before_dispatch(
    run_id: str,
    plan_hash: str,
) -> None:
    app = _FakeCeleryApp()
    queue = CeleryAgentRunQueue(app=app)

    with pytest.raises(ValueError):
        queue.enqueue(run_id=run_id, plan_hash=plan_hash)

    assert app.calls == []


def test_queue_rejects_blank_celery_task_id() -> None:
    app = _FakeCeleryApp()
    queue = CeleryAgentRunQueue(app=app)
    app.send_task = lambda *args, **kwargs: _AsyncResult(id="")  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="task identifier"):
        queue.enqueue(
            run_id="5f42f320-32cf-4db5-8fd8-55d479a01b21",
            plan_hash="a" * 64,
        )
