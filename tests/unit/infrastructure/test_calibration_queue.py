from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass
class _AsyncResult:
    id: str


class _FakeCeleryApp:
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
        return _AsyncResult(id="calibration-task-123")


def test_queue_dispatches_only_canonical_materialization_identity() -> None:
    from quanxin_life.application.advanced_calibration_jobs import (
        AdvancedCalibrationDispatchReceipt,
    )
    from quanxin_life.infrastructure.calibration_queue import (
        ADVANCED_CALIBRATION_QUEUE,
        ADVANCED_CALIBRATION_TASK,
        CeleryAdvancedCalibrationQueue,
    )

    app = _FakeCeleryApp()
    queue = CeleryAdvancedCalibrationQueue(app=app)
    materialization_id = "fd11757b-b4d9-46e8-9633-fe26ffd70f13"

    receipt = queue.enqueue(materialization_id=materialization_id)

    assert ADVANCED_CALIBRATION_TASK == (
        "quanxin_life.advanced_calibration.materialize.v1"
    )
    assert receipt == AdvancedCalibrationDispatchReceipt(
        materialization_id=materialization_id,
        task_id="calibration-task-123",
    )
    assert app.calls == [
        {
            "name": ADVANCED_CALIBRATION_TASK,
            "kwargs": {"materialization_id": materialization_id},
            "queue": ADVANCED_CALIBRATION_QUEUE,
        }
    ]
    assert set(app.calls[0]["kwargs"]) == {"materialization_id"}


@pytest.mark.parametrize(
    "materialization_id",
    [
        "not-a-uuid",
        "FD11757B-B4D9-46E8-9633-FE26FFD70F13",
        "{fd11757b-b4d9-46e8-9633-fe26ffd70f13}",
    ],
)
def test_queue_rejects_noncanonical_materialization_id_before_dispatch(
    materialization_id: str,
) -> None:
    from quanxin_life.infrastructure.calibration_queue import (
        CeleryAdvancedCalibrationQueue,
    )

    app = _FakeCeleryApp()
    queue = CeleryAdvancedCalibrationQueue(app=app)

    with pytest.raises(ValueError, match="canonical"):
        queue.enqueue(materialization_id=materialization_id)

    assert app.calls == []


def test_queue_rejects_blank_celery_task_id() -> None:
    from quanxin_life.infrastructure.calibration_queue import (
        CeleryAdvancedCalibrationQueue,
    )

    app = _FakeCeleryApp()
    app.send_task = lambda *args, **kwargs: _AsyncResult(id="")  # type: ignore[method-assign]
    queue = CeleryAdvancedCalibrationQueue(app=app)

    with pytest.raises(ValueError, match="task identifier"):
        queue.enqueue(
            materialization_id="fd11757b-b4d9-46e8-9633-fe26ffd70f13",
        )
