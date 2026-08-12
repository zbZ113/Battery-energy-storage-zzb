from __future__ import annotations

import pytest

from quanxin_life.infrastructure.feishu_queue import FEISHU_ANALYSIS_TASK
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuJobBusyError,
    FeishuJobRetryableError,
)
from quanxin_life.tasks.feishu import (
    FEISHU_TASK_MAX_RETRIES,
    FEISHU_TASK_RETRY_COUNTDOWN_SECONDS,
    register_feishu_analysis_task,
)


class _App:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def task(self, **options: object):  # type: ignore[no-untyped-def]
        self.options = options
        return lambda function: function


class _BoundTask:
    def __init__(self) -> None:
        self.retry_call: dict[str, object] | None = None

    def retry(self, **kwargs: object) -> RuntimeError:
        self.retry_call = dict(kwargs)
        return RuntimeError("retry")


class _Worker:
    def __init__(self, *, busy: bool = False, retryable: bool = False) -> None:
        self.busy = busy
        self.retryable = retryable
        self.job_ids: list[str] = []

    def execute(self, *, job_id: str) -> FeishuAnalysisJobStatus:
        self.job_ids.append(job_id)
        if self.busy:
            raise FeishuJobBusyError("owned by another worker")
        if self.retryable:
            raise FeishuJobRetryableError("download retryable")
        return FeishuAnalysisJobStatus.SUCCEEDED


def test_feishu_task_is_identity_only_and_late_acknowledged() -> None:
    app = _App()
    worker = _Worker()
    task = register_feishu_analysis_task(app=app, worker=worker)

    result = task(
        _BoundTask(),
        job_id="3a3c972b-a23e-42c3-af76-e39038806f13",
    )

    assert app.options == {
        "name": FEISHU_ANALYSIS_TASK,
        "bind": True,
        "acks_late": True,
        "reject_on_worker_lost": True,
        "ignore_result": True,
    }
    assert worker.job_ids == ["3a3c972b-a23e-42c3-af76-e39038806f13"]
    assert result == {
        "job_id": "3a3c972b-a23e-42c3-af76-e39038806f13",
        "status": "SUCCEEDED",
    }


def test_feishu_task_retry_window_outlasts_the_longest_job_lease() -> None:
    assert FEISHU_TASK_RETRY_COUNTDOWN_SECONDS * FEISHU_TASK_MAX_RETRIES > 3600


def test_feishu_task_retries_live_lease_contention() -> None:
    app = _App()
    task = register_feishu_analysis_task(app=app, worker=_Worker(busy=True))
    bound = _BoundTask()

    try:
        task(bound, job_id="3a3c972b-a23e-42c3-af76-e39038806f13")
    except RuntimeError as exc:
        assert str(exc) == "retry"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("busy jobs must request a Celery retry")

    assert bound.retry_call is not None
    assert bound.retry_call["countdown"] == FEISHU_TASK_RETRY_COUNTDOWN_SECONDS
    assert bound.retry_call["max_retries"] == FEISHU_TASK_MAX_RETRIES


def test_feishu_task_retries_a_durably_recorded_transient_failure() -> None:
    app = _App()
    task = register_feishu_analysis_task(app=app, worker=_Worker(retryable=True))
    bound = _BoundTask()

    with pytest.raises(RuntimeError, match="retry"):
        task(bound, job_id="3a3c972b-a23e-42c3-af76-e39038806f13")

    assert bound.retry_call is not None
    assert bound.retry_call["countdown"] == FEISHU_TASK_RETRY_COUNTDOWN_SECONDS
