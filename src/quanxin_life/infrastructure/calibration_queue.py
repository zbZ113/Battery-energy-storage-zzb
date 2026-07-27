"""Celery adapter for identity-only Advanced calibration jobs."""

from __future__ import annotations

from typing import Protocol

from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationDispatchReceipt,
    validate_materialization_id,
)

ADVANCED_CALIBRATION_TASK = (
    "quanxin_life.advanced_calibration.materialize.v1"
)
ADVANCED_CALIBRATION_QUEUE = "advanced-calibration"


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


class CeleryAdvancedCalibrationQueue:
    """Dispatch only a persisted materialization identity through Redis."""

    def __init__(self, *, app: CeleryApplication) -> None:
        self._app = app

    def enqueue(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationDispatchReceipt:
        normalized_id = validate_materialization_id(materialization_id)
        async_result = self._app.send_task(
            ADVANCED_CALIBRATION_TASK,
            kwargs={"materialization_id": normalized_id},
            queue=ADVANCED_CALIBRATION_QUEUE,
        )
        task_id = async_result.id
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Celery returned an invalid task identifier")
        return AdvancedCalibrationDispatchReceipt(
            materialization_id=normalized_id,
            task_id=task_id,
        )


__all__ = [
    "ADVANCED_CALIBRATION_QUEUE",
    "ADVANCED_CALIBRATION_TASK",
    "CeleryAdvancedCalibrationQueue",
    "CeleryApplication",
]
