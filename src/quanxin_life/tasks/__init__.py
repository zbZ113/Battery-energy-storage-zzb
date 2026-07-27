"""Explicitly registered background task adapters."""

from quanxin_life.tasks.advanced_calibration import (
    register_advanced_calibration_task,
)
from quanxin_life.tasks.agent_runs import register_agent_run_task

__all__ = [
    "register_advanced_calibration_task",
    "register_agent_run_task",
]
