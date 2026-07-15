"""Celery/Redis adapter that dispatches identity-only Agent run messages."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from typing import Any, Protocol, cast
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from quanxin_life.application.task_queue import AgentRunDispatchReceipt

AGENT_RUN_TASK = "quanxin_life.agent_runs.execute.v1"
AGENT_RUN_QUEUE = "agent-runs"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CeleryQueueConfig(BaseModel):
    """Runtime broker configuration; the Redis URL may contain credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    broker_url: SecretStr

    @field_validator("broker_url")
    @classmethod
    def validate_broker_url(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value().strip()
        parsed = urlsplit(raw_value)
        if parsed.scheme not in {"redis", "rediss"} or not parsed.hostname:
            raise ValueError("Celery broker must be a valid Redis URL")
        if parsed.path not in {"", "/"}:
            database = parsed.path.removeprefix("/")
            if not database.isdigit():
                raise ValueError("Celery Redis broker database must be numeric")
        return SecretStr(raw_value)


class CeleryAsyncResult(Protocol):
    id: str


class CeleryConfiguration(Protocol):
    def update(self, values: dict[str, object]) -> None: ...


class CeleryApplication(Protocol):
    conf: CeleryConfiguration

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> CeleryAsyncResult: ...


CeleryAppFactory = Callable[..., CeleryApplication]


def create_celery_app(
    config: CeleryQueueConfig,
    *,
    app_factory: CeleryAppFactory | None = None,
) -> CeleryApplication:
    """Construct a JSON-only Celery app without opening a broker connection."""

    factory = app_factory
    if factory is None:
        try:
            celery_module = importlib.import_module("celery")
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Celery support requires installing the infrastructure dependency group"
            ) from exc
        factory = cast(CeleryAppFactory, cast(Any, celery_module).Celery)
    application = factory(
        "quanxin-life",
        broker=config.broker_url.get_secret_value(),
    )
    application.conf.update(
        {
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
    )
    return application


class CeleryAgentRunQueue:
    """Send only a run identity and immutable plan hash to Redis."""

    def __init__(self, *, app: CeleryApplication) -> None:
        self._app = app

    def enqueue(self, *, run_id: str, plan_hash: str) -> AgentRunDispatchReceipt:
        """Dispatch one versioned Agent run envelope."""

        try:
            parsed_run_id = UUID(run_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("run_id must be a canonical UUID") from exc
        if str(parsed_run_id) != run_id:
            raise ValueError("run_id must be a canonical lowercase UUID")
        if not isinstance(plan_hash, str) or not _SHA256.fullmatch(plan_hash):
            raise ValueError("plan_hash must be a lowercase SHA-256 digest")

        async_result = self._app.send_task(
            AGENT_RUN_TASK,
            kwargs={"run_id": run_id, "plan_hash": plan_hash},
            queue=AGENT_RUN_QUEUE,
        )
        task_id = async_result.id
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Celery returned an invalid task identifier")
        return AgentRunDispatchReceipt(run_id=run_id, task_id=task_id)


__all__ = [
    "AGENT_RUN_QUEUE",
    "AGENT_RUN_TASK",
    "CeleryAgentRunQueue",
    "CeleryApplication",
    "CeleryQueueConfig",
    "create_celery_app",
]
