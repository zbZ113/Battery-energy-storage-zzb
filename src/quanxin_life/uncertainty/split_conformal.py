"""Cell-disjoint split conformal intervals for EOL80 lifetime estimates.

The functions in this module never fit a predictor and do not make lifetime
estimates.  They calibrate residuals from an explicit cell-level calibration
cohort, then expand existing :class:`LifePrediction` point estimates.  This
keeps interval construction separate from model fitting and prevents training
or test labels from silently entering calibration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean

from pydantic import ConfigDict, Field

from quanxin_life.core import (
    ConformalCalibration,
    LifePrediction,
    PredictionInterval,
    PredictionTarget,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import SplitManifest

_PREDICTION_CONTEXT_FIELDS = (
    "dataset_id",
    "cutoff_cycle",
    "target",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)
_CALIBRATION_VERSION_FIELDS = (
    "target",
    "feature_version",
    "split_version",
    "model_version",
    "data_version",
)


class IntervalCoverage(ContractModel):
    """Coverage statistics for one homogeneous observed cell-level cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    target: PredictionTarget = PredictionTarget.EOL80_CYCLE
    evaluated_cell_count: int = Field(gt=0)
    picp: float = Field(ge=0, le=1)
    mpiw: float = Field(ge=0)
    calibration: ConformalCalibration


def calibrate_split_conformal(
    calibration_predictions: Sequence[LifePrediction],
    *,
    split_manifest: SplitManifest,
    alpha: float = 0.10,
) -> ConformalCalibration:
    """Calibrate a finite-sample split conformal residual radius.

    ``calibration_predictions`` must contain exactly one observed, non-censored
    EOL80 label per cell.  Every cell must explicitly belong to the calibration
    split in ``split_manifest``; train, validation and test cells are rejected.
    For a calibration cohort of size ``n``, the returned residual radius uses
    rank ``ceil((n + 1) * (1 - alpha))`` of sorted absolute residuals, clamped
    to the largest available residual when the requested finite-sample rank
    exceeds ``n``.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    cohort = tuple(calibration_predictions)
    if not cohort:
        raise ValueError("at least one calibration-cell prediction is required")

    reference = cohort[0]
    seen_cell_ids: set[str] = set()
    residuals: list[float] = []
    for prediction in cohort:
        _validate_calibration_prediction(
            prediction,
            split_manifest=split_manifest,
            seen_cell_ids=seen_cell_ids,
            reference=reference,
        )
        assert prediction.observed_eol_cycle is not None
        residuals.append(abs(prediction.predicted_eol_cycle - prediction.observed_eol_cycle))

    residuals.sort()
    finite_sample_rank = math.ceil((len(residuals) + 1) * (1.0 - alpha))
    selected_index = min(finite_sample_rank, len(residuals)) - 1
    residual_quantile = residuals[selected_index]

    return ConformalCalibration(
        target=reference.target,
        alpha=alpha,
        residual_quantile_cycle=residual_quantile,
        calibration_cell_count=len(cohort),
        feature_version=reference.feature_version,
        split_version=reference.split_version,
        model_version=reference.model_version,
        data_version=reference.data_version,
    )


def make_prediction_interval(
    prediction: LifePrediction, calibration: ConformalCalibration
) -> PredictionInterval:
    """Expand one model-produced point EOL80 prediction using a calibration radius.

    No model-supplied uncertainty field is accepted.  The point estimate remains
    unchanged, the lower endpoint is truncated at ``prediction.cutoff_cycle``,
    and the point is always contained in the returned interval.
    """

    if prediction.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 point predictions support split conformal intervals")
    _require_calibration_versions(prediction, calibration)

    lower = max(
        float(prediction.cutoff_cycle),
        prediction.predicted_eol_cycle - calibration.residual_quantile_cycle,
    )
    upper = prediction.predicted_eol_cycle + calibration.residual_quantile_cycle
    return PredictionInterval(
        dataset_id=prediction.dataset_id,
        cell_id=prediction.cell_id,
        cutoff_cycle=prediction.cutoff_cycle,
        target=prediction.target,
        point_prediction_cycle=prediction.predicted_eol_cycle,
        lower_eol_cycle=lower,
        upper_eol_cycle=upper,
        calibration=calibration,
    )


def evaluate_interval_coverage(
    intervals: Sequence[PredictionInterval], observed_predictions: Sequence[LifePrediction]
) -> IntervalCoverage:
    """Evaluate PICP and MPIW from a homogeneous observed cell-level cohort.

    The interval and observed-prediction sequences must contain the same unique
    cells.  Right-censored labels and context mixing are rejected rather than
    treated as observed events.
    """

    interval_cohort = tuple(intervals)
    observation_cohort = tuple(observed_predictions)
    if not interval_cohort:
        raise ValueError("at least one prediction interval is required for coverage")
    if len(interval_cohort) != len(observation_cohort):
        raise ValueError("intervals and observed predictions must have the same cell count")

    reference = interval_cohort[0]
    interval_by_cell: dict[str, PredictionInterval] = {}
    for interval in interval_cohort:
        _validate_interval_context(interval, reference=reference, known_cells=interval_by_cell)
        interval_by_cell[interval.cell_id] = interval

    observation_cells: set[str] = set()
    covered: list[float] = []
    widths: list[float] = []
    for observation in observation_cohort:
        if observation.cell_id in observation_cells:
            raise ValueError(
                f"duplicate cell_id in observed prediction cohort: {observation.cell_id}"
            )
        observation_cells.add(observation.cell_id)
        matched_interval = interval_by_cell.get(observation.cell_id)
        if matched_interval is None:
            raise ValueError("interval and observed prediction cell_ids must exactly match")
        _validate_observation_context(observation, interval=matched_interval)
        assert observation.observed_eol_cycle is not None
        covered.append(
            float(
                matched_interval.lower_eol_cycle
                <= observation.observed_eol_cycle
                <= matched_interval.upper_eol_cycle
            )
        )
        widths.append(matched_interval.upper_eol_cycle - matched_interval.lower_eol_cycle)

    if observation_cells != set(interval_by_cell):
        raise ValueError("interval and observed prediction cell_ids must exactly match")
    return IntervalCoverage(
        target=reference.target,
        evaluated_cell_count=len(interval_cohort),
        picp=fmean(covered),
        mpiw=fmean(widths),
        calibration=reference.calibration,
    )


def _validate_calibration_prediction(
    prediction: LifePrediction,
    *,
    split_manifest: SplitManifest,
    seen_cell_ids: set[str],
    reference: LifePrediction,
) -> None:
    if prediction.cell_id in seen_cell_ids:
        raise ValueError(f"duplicate calibration cell_id: {prediction.cell_id}")
    seen_cell_ids.add(prediction.cell_id)
    if prediction.dataset_id != split_manifest.dataset_id:
        raise ValueError("calibration prediction dataset_id must match the split manifest")
    if prediction.cell_id not in split_manifest.calibration:
        raise ValueError(
            f"calibration prediction cell_id is outside the calibration split: {prediction.cell_id}"
        )
    if prediction.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 predictions can calibrate split conformal intervals")
    if prediction.right_censored:
        raise ValueError("right-censored predictions cannot calibrate split conformal intervals")
    if prediction.observed_eol_cycle is None:
        raise ValueError("calibration predictions require an observed EOL80 cycle")
    for field in _PREDICTION_CONTEXT_FIELDS:
        if getattr(prediction, field) != getattr(reference, field):
            raise ValueError(f"calibration predictions must share {field}")


def _require_calibration_versions(
    prediction: LifePrediction, calibration: ConformalCalibration
) -> None:
    for field in _CALIBRATION_VERSION_FIELDS:
        if getattr(prediction, field) != getattr(calibration, field):
            raise ValueError(f"prediction {field} must match the conformal calibration")


def _validate_interval_context(
    interval: PredictionInterval,
    *,
    reference: PredictionInterval,
    known_cells: dict[str, PredictionInterval],
) -> None:
    if interval.cell_id in known_cells:
        raise ValueError(f"duplicate cell_id in interval cohort: {interval.cell_id}")
    if interval.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 intervals can be evaluated")
    for field in ("dataset_id", "cutoff_cycle", "target", "calibration"):
        if getattr(interval, field) != getattr(reference, field):
            raise ValueError(f"interval cohort must share {field}")


def _validate_observation_context(
    observation: LifePrediction, *, interval: PredictionInterval
) -> None:
    if observation.target is not PredictionTarget.EOL80_CYCLE:
        raise ValueError("only EOL80 observed predictions can evaluate intervals")
    if observation.right_censored:
        raise ValueError("right-censored predictions cannot evaluate interval coverage")
    if observation.observed_eol_cycle is None:
        raise ValueError("observed EOL80 cycle is required for interval coverage")
    for field in ("dataset_id", "cell_id", "cutoff_cycle", "target"):
        if getattr(observation, field) != getattr(interval, field):
            raise ValueError(f"observed prediction {field} must match its interval")
    _require_calibration_versions(observation, interval.calibration)
