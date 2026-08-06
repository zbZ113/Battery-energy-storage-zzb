"""Finite, task-aware metric calculations over observed predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def regression_metrics(true: Sequence[float], pred: Sequence[float]) -> dict[str, float]:
    if len(true) != len(pred) or not true:
        raise ValueError("true and predicted values must be non-empty and aligned")
    actual = torch.tensor(true, dtype=torch.float64)
    estimate = torch.tensor(pred, dtype=torch.float64)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(estimate).all()):
        raise ValueError("metric inputs must be finite")
    error = estimate - actual
    mae = float(error.abs().mean())
    mse = float(error.square().mean())
    result = {
        "mae": mae,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": float(error.mean()),
        "median_absolute_error": float(error.abs().median()),
        "p75_absolute_error": float(torch.quantile(error.abs(), 0.75)),
        "p90_absolute_error": float(torch.quantile(error.abs(), 0.90)),
        "p95_absolute_error": float(torch.quantile(error.abs(), 0.95)),
    }
    if bool((actual == 0).any()):
        result["smape"] = float(
            (2 * error.abs() / (actual.abs() + estimate.abs()).clamp_min(1e-12)).mean()
        )
    else:
        result["mape"] = float((error.abs() / actual.abs()).mean())
    return result


__all__ = ["regression_metrics"]
