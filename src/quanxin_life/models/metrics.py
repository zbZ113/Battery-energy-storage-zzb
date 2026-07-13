"""Leakage-safe cell-level EOL80 regression metric calculation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean

from quanxin_life.core import LifePrediction, LifetimeMetrics, PredictionTarget

_CONTEXT_FIELDS = (
    "dataset_id",
    "cutoff_cycle",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)


def evaluate_eol80_predictions(predictions: Sequence[LifePrediction]) -> LifetimeMetrics:
    """Calculate regression metrics from one observed EOL80 label per cell.

    Metrics are only defined for a homogeneous prediction cohort.  This keeps
    comparisons from silently mixing models, feature sets, splits or source
    datasets, and rejects censored labels rather than treating them as events.
    """

    cohort = tuple(predictions)
    if not cohort:
        raise ValueError("at least one cell-level prediction is required for evaluation")

    reference = cohort[0]
    _validate_prediction(reference, seen_cell_ids=set())
    seen_cell_ids = {reference.cell_id}

    for prediction in cohort[1:]:
        _validate_prediction(prediction, seen_cell_ids=seen_cell_ids)
        for field in _CONTEXT_FIELDS:
            if getattr(prediction, field) != getattr(reference, field):
                raise ValueError(f"evaluation predictions must share {field}")

    observed = [_require_observed_eol80(prediction) for prediction in cohort]
    predicted = [prediction.predicted_eol_cycle for prediction in cohort]
    errors = [estimate - actual for estimate, actual in zip(predicted, observed, strict=True)]
    absolute_errors = [abs(error) for error in errors]

    mae = fmean(absolute_errors)
    rmse = math.sqrt(fmean(error * error for error in errors))
    mape = 100.0 * fmean(
        absolute_error / actual
        for absolute_error, actual in zip(absolute_errors, observed, strict=True)
    )
    r2, warnings = _calculate_r2(observed=observed, errors=errors)

    return LifetimeMetrics(
        target=PredictionTarget.EOL80_CYCLE,
        evaluated_cell_count=len(cohort),
        mae_cycle=mae,
        rmse_cycle=rmse,
        mape_percent=mape,
        r2=r2,
        warnings=warnings,
    )


def _validate_prediction(prediction: LifePrediction, *, seen_cell_ids: set[str]) -> None:
    if prediction.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 predictions can be evaluated")
    if prediction.cell_id in seen_cell_ids:
        raise ValueError(f"duplicate cell_id in evaluation cohort: {prediction.cell_id}")
    seen_cell_ids.add(prediction.cell_id)
    _require_observed_eol80(prediction)


def _require_observed_eol80(prediction: LifePrediction) -> int:
    if prediction.right_censored:
        raise ValueError("right-censored predictions cannot be evaluated")
    if prediction.observed_eol_cycle is None:
        raise ValueError("observed EOL80 cycle is required for evaluation")
    if prediction.observed_eol_cycle == 0:
        raise ValueError("observed EOL80 cycle must be positive for MAPE evaluation")
    return prediction.observed_eol_cycle


def _calculate_r2(
    *, observed: Sequence[int], errors: Sequence[float]
) -> tuple[float | None, list[str]]:
    observed_mean = fmean(observed)
    total_sum_of_squares = sum((actual - observed_mean) ** 2 for actual in observed)
    if total_sum_of_squares == 0:
        return None, ["R2_UNDEFINED_CONSTANT_OBSERVED_EOL80"]

    residual_sum_of_squares = sum(error * error for error in errors)
    return 1.0 - residual_sum_of_squares / total_sum_of_squares, []
