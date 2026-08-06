"""Seed-level aggregation and paired evaluation without pseudo-replication."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import fmean, median, stdev

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256


class SeedMetric(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    metric_name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    cohort_sha256: Sha256


class FailedSeedRun(ContractModel):
    """Persistable row for ``failed_runs.parquet``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    failure_stage: str = Field(min_length=1)
    error_code: str = Field(min_length=1)
    last_epoch: int | None = Field(default=None, ge=0)
    context_sha256: Sha256


class MultiSeedAggregate(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    metric_name: str
    cohort_sha256: Sha256
    expected_seeds: tuple[int, ...]
    valid_seed_count: int = Field(ge=0)
    failed_seed_count: int = Field(ge=0)
    complete_seed_matrix: bool
    mean: float = Field(allow_inf_nan=False)
    sample_std: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    median: float = Field(allow_inf_nan=False)
    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    seed_metrics: tuple[SeedMetric, ...]
    failed_runs: tuple[FailedSeedRun, ...]


class PairedObservation(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = Field(ge=0)
    cell_id: str = Field(min_length=1)
    left_value: float = Field(allow_inf_nan=False)
    right_value: float = Field(allow_inf_nan=False)


class PairedBootstrapResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_count: int = Field(gt=0)
    seed_count: int = Field(gt=0)
    effect: float = Field(allow_inf_nan=False)
    lower: float = Field(allow_inf_nan=False)
    upper: float = Field(allow_inf_nan=False)
    win_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    confidence_level: float = Field(gt=0, lt=1)
    resamples: int = Field(gt=0)
    random_seed: int = Field(ge=0)
    conclusion: str

    @model_validator(mode="after")
    def bounds_are_ordered(self) -> PairedBootstrapResult:
        if self.lower > self.upper:
            raise ValueError("bootstrap lower bound cannot exceed upper bound")
        return self


def aggregate_seed_metrics(
    metrics: Sequence[SeedMetric],
    *,
    expected_seeds: Sequence[int],
    failed_runs: Sequence[FailedSeedRun] = (),
) -> MultiSeedAggregate:
    values = tuple(SeedMetric.model_validate(item) for item in metrics)
    failures = tuple(FailedSeedRun.model_validate(item) for item in failed_runs)
    expected = tuple(expected_seeds)
    if not values:
        raise ValueError("at least one valid seed metric is required")
    if not expected or len(expected) != len(set(expected)) or any(seed < 0 for seed in expected):
        raise ValueError("expected_seeds must be unique non-negative values")
    valid_seeds = [item.seed for item in values]
    failed_seeds = [item.seed for item in failures]
    if len(valid_seeds) != len(set(valid_seeds)):
        raise ValueError("each valid seed must have exactly one metric")
    if len(failed_seeds) != len(set(failed_seeds)):
        raise ValueError("each failed seed must have exactly one failure record")
    if set(valid_seeds) & set(failed_seeds):
        raise ValueError("a seed cannot be both valid and failed")
    if set(valid_seeds) | set(failed_seeds) != set(expected):
        raise ValueError("every expected seed requires valid or failed evidence")
    metric_names = {item.metric_name for item in values}
    if len(metric_names) != 1:
        raise ValueError("seed metrics must share metric_name")
    cohort_hashes = {item.cohort_sha256 for item in values}
    if len(cohort_hashes) != 1:
        raise ValueError("all valid seeds must use the same cell cohort")
    ordered_values = tuple(sorted(values, key=lambda item: item.seed))
    ordered_failures = tuple(sorted(failures, key=lambda item: item.seed))
    numbers = [item.value for item in ordered_values]
    return MultiSeedAggregate(
        metric_name=ordered_values[0].metric_name,
        cohort_sha256=ordered_values[0].cohort_sha256,
        expected_seeds=expected,
        valid_seed_count=len(ordered_values),
        failed_seed_count=len(ordered_failures),
        complete_seed_matrix=not ordered_failures,
        mean=fmean(numbers),
        sample_std=stdev(numbers) if len(numbers) > 1 else None,
        median=median(numbers),
        minimum=min(numbers),
        maximum=max(numbers),
        seed_metrics=ordered_values,
        failed_runs=ordered_failures,
    )


def paired_bootstrap(
    observations: Sequence[PairedObservation],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    random_seed: int = 20260712,
) -> PairedBootstrapResult:
    pairs = tuple(PairedObservation.model_validate(item) for item in observations)
    if not pairs:
        raise ValueError("at least one paired seed/cell observation is required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0 < confidence_level < 1 or not math.isfinite(confidence_level):
        raise ValueError("confidence_level must be finite and between zero and one")
    coordinates = [(item.seed, item.cell_id) for item in pairs]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError("paired observations contain duplicate seed/cell coordinates")
    cells_by_seed: dict[int, set[str]] = {}
    for item in pairs:
        cells_by_seed.setdefault(item.seed, set()).add(item.cell_id)
    reference_cells = next(iter(cells_by_seed.values()))
    if any(cells != reference_cells for cells in cells_by_seed.values()):
        raise ValueError("all seeds must contain the same paired cell cohort")
    deltas_by_cell: dict[str, list[float]] = {}
    for item in pairs:
        deltas_by_cell.setdefault(item.cell_id, []).append(item.left_value - item.right_value)
    cell_deltas = {
        cell_id: fmean(deltas) for cell_id, deltas in deltas_by_cell.items()
    }
    deltas = list(cell_deltas.values())
    rng = random.Random(random_seed)
    sampled_effects = sorted(
        fmean(cell_deltas[rng.choice(tuple(cell_deltas))] for _ in cell_deltas)
        for _ in range(resamples)
    )
    tail = (1 - confidence_level) / 2
    lower = _quantile(sampled_effects, tail)
    upper = _quantile(sampled_effects, 1 - tail)
    if lower <= 0 <= upper:
        conclusion = "NO_DEMONSTRATED_SIGNIFICANT_ADVANTAGE"
    elif upper < 0:
        conclusion = "LEFT_LOWER_ON_REGISTERED_PAIRED_METRIC"
    else:
        conclusion = "RIGHT_LOWER_ON_REGISTERED_PAIRED_METRIC"
    return PairedBootstrapResult(
        pair_count=len(pairs),
        seed_count=len(cells_by_seed),
        effect=fmean(deltas),
        lower=lower,
        upper=upper,
        win_rate=sum(delta < 0 for delta in deltas) / len(deltas),
        confidence_level=confidence_level,
        resamples=resamples,
        random_seed=random_seed,
        conclusion=conclusion,
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    position = probability * (len(values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return values[lower_index]
    fraction = position - lower_index
    return values[lower_index] * (1 - fraction) + values[upper_index] * fraction


__all__ = [
    "FailedSeedRun",
    "MultiSeedAggregate",
    "PairedBootstrapResult",
    "PairedObservation",
    "SeedMetric",
    "aggregate_seed_metrics",
    "paired_bootstrap",
]
