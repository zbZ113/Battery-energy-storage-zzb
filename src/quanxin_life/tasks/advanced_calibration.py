"""Celery task registration for Advanced calibration materialization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationMaterializationBusyError,
    validate_materialization_id,
)
from quanxin_life.core import AdvancedCalibrationMaterializationStatus
from quanxin_life.infrastructure.calibration_queue import (
    ADVANCED_CALIBRATION_TASK,
)

TaskResult = dict[str, str]
TaskCallable = Callable[..., TaskResult]


class AdvancedCalibrationMaterializationExecutor(Protocol):
    def execute(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationStatus: ...


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


def register_advanced_calibration_task(
    *,
    app: CeleryTaskApplication,
    worker: AdvancedCalibrationMaterializationExecutor,
) -> TaskCallable:
    """Register the versioned task without import-time worker assembly."""

    @app.task(
        name=ADVANCED_CALIBRATION_TASK,
        bind=True,
        acks_late=True,
        reject_on_worker_lost=True,
        ignore_result=True,
    )
    def execute_advanced_calibration(
        bound_task: BoundCeleryTask,
        *,
        materialization_id: str,
    ) -> TaskResult:
        normalized_id = validate_materialization_id(materialization_id)
        try:
            status = worker.execute(materialization_id=normalized_id)
        except AdvancedCalibrationMaterializationBusyError as exc:
            raise bound_task.retry(
                exc=exc,
                countdown=5,
                max_retries=20,
            ) from exc
        return {
            "materialization_id": normalized_id,
            "status": status.value,
        }

    return execute_advanced_calibration


__all__ = [
    "AdvancedCalibrationMaterializationExecutor",
    "register_advanced_calibration_task",
]
