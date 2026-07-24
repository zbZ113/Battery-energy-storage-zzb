"""Cell-clustered metric closure for completed advanced RUL experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean

from pydantic import Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel

_DEFAULT_TOLERANCES = (5.0, 10.0, 15.0, 20.0)
_QUANTILES = (50.0, 75.0, 90.0, 95.0, 100.0)


class RulPredictionPoint(ContractModel):
    """One immutable cell prediction from one completed model run."""

    run_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    seed: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    batch_date: str = Field(min_length=1)
    true_cycle_life: float = Field(gt=0, allow_inf_nan=False)
    predicted_cycle_life: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("run_id", "family", "candidate_id", "cell_id", "batch_date")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("RUL prediction text fields must not be blank")
        return normalized


class MetricInterval(ContractModel):
    """Percentile bootstrap interval for one cell-clustered metric."""

    estimate: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)

    @model_validator(mode="after")
    def interval_is_ordered(self) -> MetricInterval:
        if self.lower > self.upper:
            raise ValueError("metric interval lower bound must not exceed upper bound")
        return self


class RulSummary(ContractModel):
    """Five-seed aggregate and cell-clustered uncertainty for one RUL cohort."""

    schema_version: str = "advanced-rul-metric-closure-v1"
    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    cell_count: int = Field(ge=1)
    seed_count: int = Field(ge=1)
    mae_cycle: float = Field(ge=0, allow_inf_nan=False)
    rmse_cycle: float = Field(ge=0, allow_inf_nan=False)
    mape_percent: float = Field(ge=0, allow_inf_nan=False)
    r2: float | None = Field(default=None, allow_inf_nan=False)
    accuracy_at_tolerance_percent: dict[str, float]
    absolute_error_quantiles_cycle: dict[str, float]
    bootstrap_confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_intervals: dict[str, MetricInterval]


class RulModelComparison(ContractModel):
    """Paired cell-level comparison between two models at one cutoff."""

    schema_version: str = "advanced-rul-model-comparison-v1"
    left_family: str = Field(min_length=1)
    left_candidate_id: str = Field(min_length=1)
    right_family: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    cell_count: int = Field(ge=1)
    seed_count: int = Field(ge=1)
    mean_absolute_error_delta_cycle: float = Field(allow_inf_nan=False)
    left_cell_win_rate_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    bootstrap_confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_interval: MetricInterval


class SohPredictionPoint(ContractModel):
    """One future-cycle SOH prediction from one completed trajectory run."""

    run_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    seed: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    batch_date: str = Field(min_length=1)
    cycle: int = Field(ge=1)
    true_soh: float = Field(ge=0, allow_inf_nan=False)
    predicted_soh: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("run_id", "family", "candidate_id", "cell_id", "batch_date")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("SOH prediction text fields must not be blank")
        return normalized

    @model_validator(mode="after")
    def prediction_is_after_cutoff(self) -> SohPredictionPoint:
        if self.cycle <= self.cutoff_cycle:
            raise ValueError("SOH model predictions must occur after cutoff_cycle")
        return self


class SohSummary(ContractModel):
    """Cell-clustered trajectory metrics for one model and cutoff."""

    schema_version: str = "advanced-soh-metric-closure-v1"
    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    cell_count: int = Field(ge=1)
    seed_count: int = Field(ge=1)
    forecast_point_count_per_seed: int = Field(ge=1)
    forecast_horizon_cycle_min: int = Field(ge=1)
    forecast_horizon_cycle_max: int = Field(ge=1)
    mae_soh: float = Field(ge=0, allow_inf_nan=False)
    rmse_soh: float = Field(ge=0, allow_inf_nan=False)
    mean_signed_error_soh: float = Field(allow_inf_nan=False)
    monotonic_violation_rate_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    accuracy_at_soh_point_tolerance_percent: dict[str, float]
    cell_mae_quantiles_soh: dict[str, float]
    bootstrap_confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_intervals: dict[str, MetricInterval]


class SohModelComparison(ContractModel):
    """Paired cell-trajectory MAE comparison for two SOH models."""

    schema_version: str = "advanced-soh-model-comparison-v1"
    left_family: str = Field(min_length=1)
    left_candidate_id: str = Field(min_length=1)
    right_family: str = Field(min_length=1)
    right_candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=1)
    cell_count: int = Field(ge=1)
    seed_count: int = Field(ge=1)
    comparison_point_count: int = Field(ge=1)
    left_only_point_count: int = Field(ge=0)
    right_only_point_count: int = Field(ge=0)
    mean_cell_mae_delta_soh: float = Field(allow_inf_nan=False)
    left_cell_win_rate_percent: float = Field(ge=0, le=100, allow_inf_nan=False)
    bootstrap_confidence_level: float = Field(gt=0, lt=1)
    bootstrap_resamples: int = Field(ge=1)
    bootstrap_seed: int = Field(ge=0)
    bootstrap_interval: MetricInterval


@dataclass(frozen=True, slots=True)
class _SohAggregate:
    point_count: int
    absolute_error_sum: float
    squared_error_sum: float
    signed_error_sum: float
    tolerance_hit_counts: tuple[int, ...]
    monotonic_violation_count: int
    monotonic_comparison_count: int


class _ValidatedSohCohort:
    def __init__(self, points: Sequence[SohPredictionPoint]) -> None:
        if not points:
            raise ValueError("at least one SOH prediction point is required")
        self.points = tuple(
            SohPredictionPoint.model_validate(point.model_dump(mode="json"))
            for point in points
        )
        reference = self.points[0]
        self.family = reference.family
        self.candidate_id = reference.candidate_id
        self.cutoff_cycle = reference.cutoff_cycle
        self.by_seed_cell: dict[tuple[int, str], tuple[SohPredictionPoint, ...]] = {}
        raw_by_seed_cell: dict[tuple[int, str], list[SohPredictionPoint]] = defaultdict(list)
        actual_by_cell_cycle: dict[tuple[str, int], float] = {}
        batch_by_cell: dict[str, str] = {}
        run_by_seed: dict[int, str] = {}
        coordinates_by_seed: dict[int, set[tuple[str, int]]] = defaultdict(set)
        for point in self.points:
            if point.family != self.family:
                raise ValueError("SOH prediction cohort must share family")
            if point.candidate_id != self.candidate_id:
                raise ValueError("SOH prediction cohort must share candidate_id")
            if point.cutoff_cycle != self.cutoff_cycle:
                raise ValueError("SOH prediction cohort must share cutoff_cycle")
            coordinate = (point.cell_id, point.cycle)
            if coordinate in coordinates_by_seed[point.seed]:
                raise ValueError("SOH prediction cohort contains duplicate seed/cell/cycle rows")
            coordinates_by_seed[point.seed].add(coordinate)
            previous_run = run_by_seed.setdefault(point.seed, point.run_id)
            if previous_run != point.run_id:
                raise ValueError("each SOH seed must map to exactly one run_id")
            previous_actual = actual_by_cell_cycle.setdefault(coordinate, point.true_soh)
            if not math.isclose(previous_actual, point.true_soh, rel_tol=0, abs_tol=1e-12):
                raise ValueError("each cell/cycle must share the same true SOH")
            previous_batch = batch_by_cell.setdefault(point.cell_id, point.batch_date)
            if previous_batch != point.batch_date:
                raise ValueError("each SOH cell_id must share batch_date")
            raw_by_seed_cell[(point.seed, point.cell_id)].append(point)
        self.seeds = tuple(sorted(coordinates_by_seed))
        self.cell_ids = tuple(sorted(batch_by_cell))
        expected_coordinates = coordinates_by_seed[self.seeds[0]]
        if any(value != expected_coordinates for value in coordinates_by_seed.values()):
            raise ValueError("all seeds must use the same cell/cycle trajectory")
        for key, values in raw_by_seed_cell.items():
            self.by_seed_cell[key] = tuple(sorted(values, key=lambda item: item.cycle))
        self.forecast_horizon_cycle_min = min(
            point.cycle - self.cutoff_cycle for point in self.points
        )
        self.forecast_horizon_cycle_max = max(
            point.cycle - self.cutoff_cycle for point in self.points
        )


class _ValidatedCohort:
    def __init__(self, points: Sequence[RulPredictionPoint]) -> None:
        if not points:
            raise ValueError("at least one RUL prediction point is required")
        self.points = tuple(
            RulPredictionPoint.model_validate(point.model_dump(mode="json"))
            for point in points
        )
        reference = self.points[0]
        self.family = reference.family
        self.candidate_id = reference.candidate_id
        self.cutoff_cycle = reference.cutoff_cycle
        self.by_seed_cell: dict[tuple[int, str], RulPredictionPoint] = {}
        self.true_by_cell: dict[str, float] = {}
        run_by_seed: dict[int, str] = {}
        cells_by_seed: dict[int, set[str]] = defaultdict(set)
        for point in self.points:
            if point.family != self.family:
                raise ValueError("RUL prediction cohort must share family")
            if point.candidate_id != self.candidate_id:
                raise ValueError("RUL prediction cohort must share candidate_id")
            if point.cutoff_cycle != self.cutoff_cycle:
                raise ValueError("RUL prediction cohort must share cutoff_cycle")
            key = (point.seed, point.cell_id)
            if key in self.by_seed_cell:
                raise ValueError("RUL prediction cohort contains duplicate seed/cell pairs")
            self.by_seed_cell[key] = point
            previous_run = run_by_seed.setdefault(point.seed, point.run_id)
            if previous_run != point.run_id:
                raise ValueError("each seed must map to exactly one run_id")
            previous_true = self.true_by_cell.setdefault(
                point.cell_id, point.true_cycle_life
            )
            if not math.isclose(
                previous_true,
                point.true_cycle_life,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError("each cell_id must share the same true cycle life")
            cells_by_seed[point.seed].add(point.cell_id)
        self.seeds = tuple(sorted(cells_by_seed))
        self.cell_ids = tuple(sorted(self.true_by_cell))
        expected_cells = set(self.cell_ids)
        if any(cells != expected_cells for cells in cells_by_seed.values()):
            raise ValueError("all seeds must use the same cell_id cohort")


def summarize_rul_predictions(
    points: Sequence[RulPredictionPoint],
    *,
    tolerance_percentages: Sequence[float] = _DEFAULT_TOLERANCES,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_712,
    confidence_level: float = 0.95,
) -> RulSummary:
    """Aggregate complete seed/cell cohorts and bootstrap over cell identities."""

    cohort = _ValidatedCohort(points)
    tolerances = _validated_tolerances(tolerance_percentages)
    _validate_bootstrap(bootstrap_resamples, bootstrap_seed, confidence_level)
    base = _cohort_metrics(cohort, cohort.cell_ids, tolerances)
    per_cell_mean_absolute_error = [
        fmean(
            abs(
                cohort.by_seed_cell[(seed, cell_id)].predicted_cycle_life
                - cohort.true_by_cell[cell_id]
            )
            for seed in cohort.seeds
        )
        for cell_id in cohort.cell_ids
    ]
    bootstrap_values = _bootstrap_metric_values(
        cohort,
        tolerances=tolerances,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    intervals = {
        name: _interval(
            estimate=value,
            samples=bootstrap_values[name],
            confidence_level=confidence_level,
        )
        for name, value in base.items()
        if value is not None and name in bootstrap_values
    }
    return RulSummary(
        family=cohort.family,
        candidate_id=cohort.candidate_id,
        cutoff_cycle=cohort.cutoff_cycle,
        cell_count=len(cohort.cell_ids),
        seed_count=len(cohort.seeds),
        mae_cycle=_required_metric(base, "mae_cycle"),
        rmse_cycle=_required_metric(base, "rmse_cycle"),
        mape_percent=_required_metric(base, "mape_percent"),
        r2=base["r2"],
        accuracy_at_tolerance_percent={
            _format_number(tolerance): _required_metric(
                base, _accuracy_metric_name(tolerance)
            )
            for tolerance in tolerances
        },
        absolute_error_quantiles_cycle={
            _format_number(quantile): _quantile(
                per_cell_mean_absolute_error, quantile / 100.0
            )
            for quantile in _QUANTILES
        },
        bootstrap_confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_intervals=intervals,
    )


def compare_rul_models(
    left_points: Sequence[RulPredictionPoint],
    right_points: Sequence[RulPredictionPoint],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_712,
    confidence_level: float = 0.95,
) -> RulModelComparison:
    """Compare models through paired mean absolute errors for the same cells."""

    left = _ValidatedCohort(left_points)
    right = _ValidatedCohort(right_points)
    _validate_bootstrap(bootstrap_resamples, bootstrap_seed, confidence_level)
    if left.cutoff_cycle != right.cutoff_cycle:
        raise ValueError("model comparison requires the same cutoff_cycle")
    if left.seeds != right.seeds:
        raise ValueError("model comparison requires the same seeds")
    if left.cell_ids != right.cell_ids:
        raise ValueError("model comparison requires the same cell_id cohort")
    for cell_id in left.cell_ids:
        if not math.isclose(
            left.true_by_cell[cell_id],
            right.true_by_cell[cell_id],
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("model comparison requires identical true cycle life")
    deltas = {
        cell_id: _cell_mean_absolute_error(left, cell_id)
        - _cell_mean_absolute_error(right, cell_id)
        for cell_id in left.cell_ids
    }
    estimate = fmean(deltas.values())
    left_wins = sum(value < 0 for value in deltas.values())
    rng = random.Random(bootstrap_seed)
    samples = [
        fmean(deltas[rng.choice(left.cell_ids)] for _ in left.cell_ids)
        for _ in range(bootstrap_resamples)
    ]
    return RulModelComparison(
        left_family=left.family,
        left_candidate_id=left.candidate_id,
        right_family=right.family,
        right_candidate_id=right.candidate_id,
        cutoff_cycle=left.cutoff_cycle,
        cell_count=len(left.cell_ids),
        seed_count=len(left.seeds),
        mean_absolute_error_delta_cycle=estimate,
        left_cell_win_rate_percent=100.0 * left_wins / len(left.cell_ids),
        bootstrap_confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_interval=_interval(
            estimate=estimate,
            samples=samples,
            confidence_level=confidence_level,
        ),
    )


def summarize_soh_predictions(
    points: Sequence[SohPredictionPoint],
    *,
    point_tolerances_percentage_points: Sequence[float] = (1.0, 2.0, 5.0),
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_712,
    confidence_level: float = 0.95,
    monotonic_tolerance: float = 1e-9,
) -> SohSummary:
    """Aggregate future SOH trajectories and bootstrap over complete cells."""

    cohort = _ValidatedSohCohort(points)
    tolerances = _validated_tolerances(point_tolerances_percentage_points)
    _validate_bootstrap(bootstrap_resamples, bootstrap_seed, confidence_level)
    if not math.isfinite(monotonic_tolerance) or monotonic_tolerance < 0:
        raise ValueError("monotonic_tolerance must be finite and non-negative")
    aggregates = _soh_aggregates(cohort, tolerances, monotonic_tolerance)
    base = _soh_cohort_metrics(cohort, cohort.cell_ids, aggregates, tolerances)
    cell_mae = [
        _soh_cell_mae(cohort, aggregates, cell_id) for cell_id in cohort.cell_ids
    ]
    rng = random.Random(bootstrap_seed)
    bootstrap_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap_resamples):
        sampled = tuple(rng.choice(cohort.cell_ids) for _ in cohort.cell_ids)
        metrics = _soh_cohort_metrics(cohort, sampled, aggregates, tolerances)
        for name, value in metrics.items():
            bootstrap_values[name].append(value)
    intervals = {
        name: _interval(
            estimate=value,
            samples=bootstrap_values[name],
            confidence_level=confidence_level,
        )
        for name, value in base.items()
    }
    points_per_seed = len(cohort.points) // len(cohort.seeds)
    return SohSummary(
        family=cohort.family,
        candidate_id=cohort.candidate_id,
        cutoff_cycle=cohort.cutoff_cycle,
        cell_count=len(cohort.cell_ids),
        seed_count=len(cohort.seeds),
        forecast_point_count_per_seed=points_per_seed,
        forecast_horizon_cycle_min=cohort.forecast_horizon_cycle_min,
        forecast_horizon_cycle_max=cohort.forecast_horizon_cycle_max,
        mae_soh=base["mae_soh"],
        rmse_soh=base["rmse_soh"],
        mean_signed_error_soh=base["mean_signed_error_soh"],
        monotonic_violation_rate_percent=base[
            "monotonic_violation_rate_percent"
        ],
        accuracy_at_soh_point_tolerance_percent={
            _format_number(tolerance): base[_soh_accuracy_metric_name(tolerance)]
            for tolerance in tolerances
        },
        cell_mae_quantiles_soh={
            _format_number(quantile): _quantile(cell_mae, quantile / 100.0)
            for quantile in _QUANTILES
        },
        bootstrap_confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_intervals=intervals,
    )


def compare_soh_models(
    left_points: Sequence[SohPredictionPoint],
    right_points: Sequence[SohPredictionPoint],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_712,
    confidence_level: float = 0.95,
) -> SohModelComparison:
    """Compare mean cell-trajectory MAE for paired SOH prediction cohorts."""

    left = _ValidatedSohCohort(left_points)
    right = _ValidatedSohCohort(right_points)
    _validate_bootstrap(bootstrap_resamples, bootstrap_seed, confidence_level)
    if left.cutoff_cycle != right.cutoff_cycle:
        raise ValueError("SOH model comparison requires the same cutoff_cycle")
    if left.seeds != right.seeds:
        raise ValueError("SOH model comparison requires the same seeds")
    if left.cell_ids != right.cell_ids:
        raise ValueError("SOH model comparison requires the same cell_id cohort")
    left_coordinates = {
        (point.seed, point.cell_id, point.cycle) for point in left.points
    }
    right_coordinates = {
        (point.seed, point.cell_id, point.cycle) for point in right.points
    }
    common_coordinates = left_coordinates & right_coordinates
    if not common_coordinates:
        raise ValueError("SOH model comparison has no common prediction coordinates")
    cell_error_deltas: dict[str, list[float]] = defaultdict(list)
    for seed in left.seeds:
        for cell_id in left.cell_ids:
            left_by_cycle = {
                point.cycle: point for point in left.by_seed_cell[(seed, cell_id)]
            }
            right_by_cycle = {
                point.cycle: point for point in right.by_seed_cell[(seed, cell_id)]
            }
            common_cycles = sorted(left_by_cycle.keys() & right_by_cycle.keys())
            if not common_cycles:
                raise ValueError("SOH model comparison has a cell without common cycles")
            left_errors: list[float] = []
            right_errors: list[float] = []
            for cycle in common_cycles:
                left_point = left_by_cycle[cycle]
                right_point = right_by_cycle[cycle]
                if not math.isclose(
                    left_point.true_soh,
                    right_point.true_soh,
                    rel_tol=0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "SOH model comparison requires identical true SOH on common cycles"
                    )
                left_errors.append(abs(left_point.predicted_soh - left_point.true_soh))
                right_errors.append(abs(right_point.predicted_soh - right_point.true_soh))
            cell_error_deltas[cell_id].append(
                fmean(left_errors) - fmean(right_errors)
            )
    deltas = {
        cell_id: fmean(cell_error_deltas[cell_id]) for cell_id in left.cell_ids
    }
    estimate = fmean(deltas.values())
    left_wins = sum(value < 0 for value in deltas.values())
    rng = random.Random(bootstrap_seed)
    samples = [
        fmean(deltas[rng.choice(left.cell_ids)] for _ in left.cell_ids)
        for _ in range(bootstrap_resamples)
    ]
    return SohModelComparison(
        left_family=left.family,
        left_candidate_id=left.candidate_id,
        right_family=right.family,
        right_candidate_id=right.candidate_id,
        cutoff_cycle=left.cutoff_cycle,
        cell_count=len(left.cell_ids),
        seed_count=len(left.seeds),
        comparison_point_count=len(common_coordinates),
        left_only_point_count=len(left_coordinates - right_coordinates),
        right_only_point_count=len(right_coordinates - left_coordinates),
        mean_cell_mae_delta_soh=estimate,
        left_cell_win_rate_percent=100.0 * left_wins / len(left.cell_ids),
        bootstrap_confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        bootstrap_interval=_interval(
            estimate=estimate,
            samples=samples,
            confidence_level=confidence_level,
        ),
    )


def _soh_aggregates(
    cohort: _ValidatedSohCohort,
    tolerances: Sequence[float],
    monotonic_tolerance: float,
) -> dict[tuple[int, str], _SohAggregate]:
    aggregates: dict[tuple[int, str], _SohAggregate] = {}
    for key, trajectory in cohort.by_seed_cell.items():
        errors = [point.predicted_soh - point.true_soh for point in trajectory]
        absolute_errors = [abs(error) for error in errors]
        predictions = [point.predicted_soh for point in trajectory]
        aggregates[key] = _SohAggregate(
            point_count=len(trajectory),
            absolute_error_sum=sum(absolute_errors),
            squared_error_sum=sum(error * error for error in errors),
            signed_error_sum=sum(errors),
            tolerance_hit_counts=tuple(
                sum(error * 100.0 <= tolerance + 1e-9 for error in absolute_errors)
                for tolerance in tolerances
            ),
            monotonic_violation_count=sum(
                current > previous + monotonic_tolerance
                for previous, current in pairwise(predictions)
            ),
            monotonic_comparison_count=max(0, len(predictions) - 1),
        )
    return aggregates


def _soh_cohort_metrics(
    cohort: _ValidatedSohCohort,
    sampled_cell_ids: Sequence[str],
    aggregates: dict[tuple[int, str], _SohAggregate],
    tolerances: Sequence[float],
) -> dict[str, float]:
    metrics_by_seed = [
        _soh_seed_metrics(cohort, seed, sampled_cell_ids, aggregates, tolerances)
        for seed in cohort.seeds
    ]
    return {
        name: fmean(item[name] for item in metrics_by_seed)
        for name in metrics_by_seed[0]
    }


def _soh_seed_metrics(
    cohort: _ValidatedSohCohort,
    seed: int,
    sampled_cell_ids: Sequence[str],
    aggregates: dict[tuple[int, str], _SohAggregate],
    tolerances: Sequence[float],
) -> dict[str, float]:
    selected = [aggregates[(seed, cell_id)] for cell_id in sampled_cell_ids]
    point_count = sum(item.point_count for item in selected)
    if point_count == 0:
        raise ValueError("SOH metric cohort has no forecast points")
    comparison_count = sum(item.monotonic_comparison_count for item in selected)
    result = {
        "mae_soh": sum(item.absolute_error_sum for item in selected) / point_count,
        "rmse_soh": math.sqrt(
            sum(item.squared_error_sum for item in selected) / point_count
        ),
        "mean_signed_error_soh": sum(item.signed_error_sum for item in selected)
        / point_count,
        "monotonic_violation_rate_percent": (
            100.0
            * sum(item.monotonic_violation_count for item in selected)
            / comparison_count
            if comparison_count
            else 0.0
        ),
    }
    result.update(
        {
            _soh_accuracy_metric_name(tolerance): 100.0
            * sum(item.tolerance_hit_counts[index] for item in selected)
            / point_count
            for index, tolerance in enumerate(tolerances)
        }
    )
    return result


def _soh_cell_mae(
    cohort: _ValidatedSohCohort,
    aggregates: dict[tuple[int, str], _SohAggregate],
    cell_id: str,
) -> float:
    selected = [aggregates[(seed, cell_id)] for seed in cohort.seeds]
    return sum(item.absolute_error_sum for item in selected) / sum(
        item.point_count for item in selected
    )


def _cohort_metrics(
    cohort: _ValidatedCohort,
    sampled_cell_ids: Sequence[str],
    tolerances: Sequence[float],
) -> dict[str, float | None]:
    metrics_by_seed = [
        _seed_metrics(cohort, seed, sampled_cell_ids, tolerances)
        for seed in cohort.seeds
    ]
    names = tuple(metrics_by_seed[0])
    aggregated: dict[str, float | None] = {}
    for name in names:
        values: list[float] = []
        for item in metrics_by_seed:
            value = item[name]
            if value is not None:
                values.append(value)
        aggregated[name] = fmean(values) if values else None
    return aggregated


def _seed_metrics(
    cohort: _ValidatedCohort,
    seed: int,
    sampled_cell_ids: Sequence[str],
    tolerances: Sequence[float],
) -> dict[str, float | None]:
    actual = [cohort.true_by_cell[cell_id] for cell_id in sampled_cell_ids]
    predicted = [
        cohort.by_seed_cell[(seed, cell_id)].predicted_cycle_life
        for cell_id in sampled_cell_ids
    ]
    errors = [estimate - observed for estimate, observed in zip(predicted, actual, strict=True)]
    absolute_errors = [abs(error) for error in errors]
    percentage_errors = [
        100.0 * error / observed
        for error, observed in zip(absolute_errors, actual, strict=True)
    ]
    mean_actual = fmean(actual)
    total_sum_of_squares = sum((value - mean_actual) ** 2 for value in actual)
    residual_sum_of_squares = sum(error * error for error in errors)
    metrics: dict[str, float | None] = {
        "mae_cycle": fmean(absolute_errors),
        "rmse_cycle": math.sqrt(fmean(error * error for error in errors)),
        "mape_percent": fmean(percentage_errors),
        "r2": (
            1.0 - residual_sum_of_squares / total_sum_of_squares
            if total_sum_of_squares > 0
            else None
        ),
    }
    metrics.update(
        {
            _accuracy_metric_name(tolerance): 100.0
            * sum(error <= tolerance + 1e-12 for error in percentage_errors)
            / len(percentage_errors)
            for tolerance in tolerances
        }
    )
    return metrics


def _bootstrap_metric_values(
    cohort: _ValidatedCohort,
    *,
    tolerances: Sequence[float],
    resamples: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(resamples):
        sampled = tuple(rng.choice(cohort.cell_ids) for _ in cohort.cell_ids)
        metrics = _cohort_metrics(cohort, sampled, tolerances)
        for name, value in metrics.items():
            if value is not None:
                values[name].append(value)
    return dict(values)


def _interval(
    *,
    estimate: float,
    samples: Sequence[float],
    confidence_level: float,
) -> MetricInterval:
    if not samples:
        raise ValueError("bootstrap interval requires at least one finite sample")
    alpha = (1.0 - confidence_level) / 2.0
    return MetricInterval(
        estimate=estimate,
        lower=_quantile(samples, alpha),
        upper=_quantile(samples, 1.0 - alpha),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    if probability < 0 or probability > 1:
        raise ValueError("quantile probability must be between zero and one")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _cell_mean_absolute_error(cohort: _ValidatedCohort, cell_id: str) -> float:
    actual = cohort.true_by_cell[cell_id]
    return fmean(
        abs(cohort.by_seed_cell[(seed, cell_id)].predicted_cycle_life - actual)
        for seed in cohort.seeds
    )


def _validated_tolerances(values: Iterable[float]) -> tuple[float, ...]:
    tolerances = tuple(float(value) for value in values)
    if not tolerances:
        raise ValueError("at least one accuracy tolerance is required")
    if any(not math.isfinite(value) or value <= 0 or value > 100 for value in tolerances):
        raise ValueError("accuracy tolerances must be finite percentages in (0, 100]")
    if len(tolerances) != len(set(tolerances)):
        raise ValueError("accuracy tolerances must be unique")
    return tuple(sorted(tolerances))


def _validate_bootstrap(resamples: int, seed: int, confidence_level: float) -> None:
    if isinstance(resamples, bool) or resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be finite and between zero and one")


def _accuracy_metric_name(tolerance: float) -> str:
    return f"accuracy_at_{_format_number(tolerance)}_percent"


def _soh_accuracy_metric_name(tolerance: float) -> str:
    return f"accuracy_at_{_format_number(tolerance)}_soh_percentage_points"


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else format(value, "g")


def _required_metric(metrics: dict[str, float | None], name: str) -> float:
    value = metrics[name]
    if value is None:
        raise ValueError(f"required RUL metric is undefined: {name}")
    return value


__all__ = [
    "MetricInterval",
    "RulModelComparison",
    "RulPredictionPoint",
    "RulSummary",
    "SohModelComparison",
    "SohPredictionPoint",
    "SohSummary",
    "compare_rul_models",
    "compare_soh_models",
    "summarize_rul_predictions",
    "summarize_soh_predictions",
]
