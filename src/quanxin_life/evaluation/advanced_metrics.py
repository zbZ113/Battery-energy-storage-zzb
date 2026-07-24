"""Cell-clustered metric closure for completed advanced RUL experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
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
    "compare_rul_models",
    "summarize_rul_predictions",
]
