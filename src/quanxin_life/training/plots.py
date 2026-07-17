"""Headless figures produced only from evaluated model outputs."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from quanxin_life.core import PredictionTarget
from quanxin_life.training.tasks import CycleLifeCurveBatch, HybridTrajectoryBatch


def write_cycle_life_evaluation_plot(
    destination: Path,
    *,
    batch: CycleLifeCurveBatch,
    predicted: np.ndarray,
    interval_radius_cycle: float,
) -> None:
    """Plot test predictions and calibrated intervals against official labels."""

    if batch.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
        raise ValueError("MATR cycle-life plot requires the explicit official target")
    values = np.asarray(predicted, dtype=np.float64)
    observed = batch.observed_cycles.detach().cpu().numpy().astype(np.float64)
    if values.shape != observed.shape or values.ndim != 1:
        raise ValueError("cycle-life plot predictions must match the test cohort")
    if not np.all(np.isfinite(values)) or not math.isfinite(interval_radius_cycle):
        raise ValueError("cycle-life plot inputs must be finite")
    if interval_radius_cycle < 0:
        raise ValueError("cycle-life interval radius cannot be negative")
    lower = np.maximum(float(batch.cutoff_cycle), values - interval_radius_cycle)
    upper = values + interval_radius_cycle

    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.errorbar(
        observed,
        values,
        yerr=np.vstack((values - lower, upper - values)),
        fmt="o",
        capsize=3,
        color="#1565c0",
        ecolor="#90caf9",
        label="Prediction with 90% conformal interval",
    )
    boundary_min = float(min(observed.min(), lower.min()))
    boundary_max = float(max(observed.max(), upper.max()))
    axis.plot(
        (boundary_min, boundary_max),
        (boundary_min, boundary_max),
        linestyle="--",
        color="#424242",
        label="Ideal",
    )
    axis.set_title("MATR official cycle-life: predicted vs observed")
    axis.set_xlabel("Observed official cycle-life (cycle)")
    axis.set_ylabel("Predicted official cycle-life (cycle)")
    axis.grid(alpha=0.2)
    axis.legend()
    _save_png(figure, destination, pyplot)


def write_hybrid_trajectory_plot(
    destination: Path,
    *,
    batch: HybridTrajectoryBatch,
    predicted: np.ndarray,
) -> None:
    """Plot mean test SOH trajectories through cycle 500 without extrapolation."""

    values = np.asarray(predicted, dtype=np.float64)
    observed = batch.target_soh.detach().cpu().numpy().astype(np.float64)
    if values.shape != observed.shape or values.ndim != 2:
        raise ValueError("Hybrid plot predictions must match the test trajectories")
    if not np.all(np.isfinite(values)):
        raise ValueError("Hybrid plot predictions must be finite")
    cycles = np.asarray(batch.prediction_cycles, dtype=np.int64)
    pyplot = _pyplot()
    figure, axis = pyplot.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    axis.plot(
        cycles,
        observed.mean(axis=0),
        color="#212121",
        linewidth=2,
        label="Observed mean SOH",
    )
    axis.plot(
        cycles,
        values.mean(axis=0),
        color="#ef6c00",
        linewidth=2,
        label="Predicted mean SOH",
    )
    axis.set_title("MATR real SOH trajectory through cycle 500")
    axis.set_xlabel("Cycle")
    axis.set_ylabel("SOH")
    axis.grid(alpha=0.2)
    axis.legend()
    _save_png(figure, destination, pyplot)


def _pyplot() -> Any:
    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg", force=True)
        pyplot = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:  # pragma: no cover - required by the A100 lock
        raise RuntimeError("training plots require matplotlib") from exc
    return pyplot


def _save_png(figure: Any, destination: Path, pyplot: Any) -> None:
    if destination.suffix.lower() != ".png":
        raise ValueError("training figures must use PNG")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.png")
    try:
        figure.savefig(temporary, dpi=160)
        temporary.replace(destination)
    finally:
        pyplot.close(figure)
        temporary.unlink(missing_ok=True)
