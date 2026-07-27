from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from quanxin_life.core import AdvancedCalibrationMaterializationStatus


class _FakeCeleryApp:
    def __init__(self) -> None:
        self.options: dict[str, object] | None = None
        self.registered: Callable[..., object] | None = None

    def task(
        self,
        **options: object,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        self.options = options

        def decorator(function: Callable[..., object]) -> Callable[..., object]:
            self.registered = function
            return function

        return decorator


@dataclass
class _Request:
    id: str


@dataclass
class _BoundTask:
    request: _Request


class _Worker:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def execute(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationStatus:
        self.calls.append(materialization_id)
        return AdvancedCalibrationMaterializationStatus.READY


class _RetrySignal(RuntimeError):
    pass


class _RetryingBoundTask:
    def __init__(self) -> None:
        self.request = _Request(id="calibration-delivery-busy")
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
    def execute(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationStatus:
        del materialization_id
        from quanxin_life.application.advanced_calibration_jobs import (
            AdvancedCalibrationMaterializationBusyError,
        )

        raise AdvancedCalibrationMaterializationBusyError("active claim")


def test_task_uses_versioned_name_and_identity_only_payload() -> None:
    from quanxin_life.infrastructure.calibration_queue import (
        ADVANCED_CALIBRATION_TASK,
    )
    from quanxin_life.tasks.advanced_calibration import (
        register_advanced_calibration_task,
    )

    app = _FakeCeleryApp()
    worker = _Worker()
    task = register_advanced_calibration_task(app=app, worker=worker)
    materialization_id = "fd11757b-b4d9-46e8-9633-fe26ffd70f13"

    result = task(
        _BoundTask(request=_Request(id="calibration-delivery-1")),
        materialization_id=materialization_id,
    )

    assert ADVANCED_CALIBRATION_TASK == (
        "quanxin_life.advanced_calibration.materialize.v1"
    )
    assert app.options == {
        "name": ADVANCED_CALIBRATION_TASK,
        "bind": True,
        "acks_late": True,
        "reject_on_worker_lost": True,
        "ignore_result": True,
    }
    assert worker.calls == [materialization_id]
    assert result == {
        "materialization_id": materialization_id,
        "status": AdvancedCalibrationMaterializationStatus.READY.value,
    }
    assert set(result) == {"materialization_id", "status"}
    assert all(
        forbidden not in result
        for forbidden in (
            "samples",
            "sample_ids",
            "results",
            "result_ids",
            "observed",
            "predicted",
        )
    )
    assert app.registered is task


def test_task_retries_with_a_bound_when_materialization_claim_is_busy() -> None:
    from quanxin_life.application.advanced_calibration_jobs import (
        AdvancedCalibrationMaterializationBusyError,
    )
    from quanxin_life.tasks.advanced_calibration import (
        register_advanced_calibration_task,
    )

    app = _FakeCeleryApp()
    task = register_advanced_calibration_task(app=app, worker=_BusyWorker())
    bound = _RetryingBoundTask()

    with pytest.raises(_RetrySignal, match="retry requested"):
        task(
            bound,
            materialization_id="fd11757b-b4d9-46e8-9633-fe26ffd70f13",
        )

    assert bound.calls == [
        (AdvancedCalibrationMaterializationBusyError, 5, 20),
    ]


@pytest.mark.parametrize(
    "materialization_id",
    [
        "not-a-uuid",
        "FD11757B-B4D9-46E8-9633-FE26FFD70F13",
    ],
)
def test_task_rejects_noncanonical_materialization_identity(
    materialization_id: str,
) -> None:
    from quanxin_life.tasks.advanced_calibration import (
        register_advanced_calibration_task,
    )

    app = _FakeCeleryApp()
    worker = _Worker()
    task = register_advanced_calibration_task(app=app, worker=worker)

    with pytest.raises(ValueError, match="canonical"):
        task(
            _BoundTask(request=_Request(id="invalid-calibration-delivery")),
            materialization_id=materialization_id,
        )

    assert worker.calls == []
