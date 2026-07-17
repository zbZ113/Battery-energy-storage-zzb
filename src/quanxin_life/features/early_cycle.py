"""Deterministic features computed only from records at or before a cutoff.

This module deliberately returns ``None`` for unavailable signals.  It never
invents a numeric replacement for missing curves, temperature, resistance, or
charge capacity.  It is a data/feature layer, not a prediction service; later
model tools must retain this feature version and the source-cycle provenance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import fmean

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from quanxin_life.data.leakage import assert_cycles_within_cutoff
from quanxin_life.data.schemas import CycleRecord, DataQualitySeverity
from quanxin_life.data.validation import validate_cycle_records
from quanxin_life.features.telemetry import exclude_exact_duplicate_telemetry

EARLY_CYCLE_FEATURE_VERSION = "early-cycle-v1"
"""Version of the feature definitions in this module."""

_SUPPORTED_CUTOFF_CYCLES = (20, 50, 100, 150)

EARLY_CYCLE_FEATURE_NAMES: tuple[str, ...] = (
    "capacity_first_ah",
    "capacity_last_ah",
    "capacity_delta_ah",
    "capacity_relative_change",
    "capacity_slope_ah_per_cycle",
    "coulombic_efficiency_mean",
    "coulombic_efficiency_last",
    "coulombic_efficiency_std",
    "cycle_duration_first_s",
    "cycle_duration_last_s",
    "cycle_duration_delta_s",
    "temperature_mean_c",
    "temperature_max_c",
    "temperature_std_c",
    "temperature_missing_fraction",
    "internal_resistance_first_ohm",
    "internal_resistance_last_ohm",
    "internal_resistance_delta_ohm",
    "internal_resistance_slope_ohm_per_cycle",
    "delta_q_mean_ah",
    "delta_q_variance_ah2",
    "delta_q_min_ah",
    "delta_q_max_ah",
    "delta_q_l2_ah",
)


class EarlyCycleFeatureConfig(BaseModel):
    """Versioned configuration for leakage-safe early-cycle features."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_cycle: int
    voltage_grid_step_v: float = Field(default=0.01, gt=0, le=0.1)
    min_curve_points: int = Field(default=3, ge=2)
    feature_version: str = Field(default=EARLY_CYCLE_FEATURE_VERSION, min_length=1)

    @model_validator(mode="after")
    def cutoff_is_supported(self) -> EarlyCycleFeatureConfig:
        if self.cutoff_cycle not in _SUPPORTED_CUTOFF_CYCLES:
            raise ValueError(
                f"cutoff_cycle must be one of {_SUPPORTED_CUTOFF_CYCLES}, got {self.cutoff_cycle}"
            )
        return self


class EarlyCycleFeatureSet(BaseModel):
    """Feature values and the exact source span used to calculate them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    source_cycles: tuple[int, ...]
    delta_q_cycles: tuple[int, int] | None = None
    values: dict[str, float | None]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def values_match_the_declared_schema(self) -> EarlyCycleFeatureSet:
        if tuple(self.values) != EARLY_CYCLE_FEATURE_NAMES:
            raise ValueError("feature value keys must match EARLY_CYCLE_FEATURE_NAMES in order")
        return self


def extract_early_cycle_features(
    records: Sequence[CycleRecord], *, config: EarlyCycleFeatureConfig
) -> EarlyCycleFeatureSet:
    """Extract features for one cell without observing any later cycle.

    The caller must pass a single-cell record sequence.  Passing a record after
    the requested cutoff is an error rather than a hint to silently drop it:
    this makes future-data leakage detectable at the module boundary.
    """

    assert_cycles_within_cutoff(
        (record.cycle_index for record in records), cutoff_cycle=config.cutoff_cycle
    )
    prepared_records, duplicate_count = exclude_exact_duplicate_telemetry(records)
    report = validate_cycle_records(prepared_records)
    fatal_issues = tuple(
        issue
        for issue in report.issues
        if issue.severity in {DataQualitySeverity.ERROR, DataQualitySeverity.BLOCKING}
    )
    if fatal_issues:
        codes = ", ".join(issue.code for issue in fatal_issues)
        raise ValueError(f"feature extraction rejected due to data quality: {codes}")

    valid_records = tuple(record for record in prepared_records if record.valid)
    if not valid_records:
        raise ValueError("feature extraction requires at least one valid record")

    dataset_ids = {record.dataset_id for record in valid_records}
    cell_ids = {record.cell_id for record in valid_records}
    if len(dataset_ids) != 1 or len(cell_ids) != 1:
        raise ValueError(
            "feature extraction requires records from exactly one dataset and one cell"
        )

    by_cycle = _group_by_cycle(valid_records)
    source_cycles = tuple(sorted(by_cycle))
    if len(source_cycles) < 2:
        raise ValueError("feature extraction requires at least two valid observed cycles")

    values: dict[str, float | None] = {name: None for name in EARLY_CYCLE_FEATURE_NAMES}
    warnings: list[str] = []
    if duplicate_count:
        warnings.append("EXACT_DUPLICATE_TELEMETRY_EXCLUDED")
    if len(valid_records) != len(prepared_records):
        warnings.append("INVALID_RECORDS_EXCLUDED")

    _populate_capacity_features(values, by_cycle, source_cycles, warnings)
    _populate_coulombic_efficiency_features(values, by_cycle, source_cycles, warnings)
    _populate_duration_features(values, by_cycle, source_cycles, warnings)
    _populate_temperature_features(values, valid_records, warnings)
    _populate_resistance_features(values, by_cycle, source_cycles, warnings)
    delta_q_cycles = _populate_delta_q_features(values, by_cycle, config, warnings)

    return EarlyCycleFeatureSet(
        dataset_id=next(iter(dataset_ids)),
        cell_id=next(iter(cell_ids)),
        cutoff_cycle=config.cutoff_cycle,
        feature_version=config.feature_version,
        source_cycles=source_cycles,
        delta_q_cycles=delta_q_cycles,
        values=values,
        warnings=tuple(sorted(set(warnings))),
    )


def _group_by_cycle(records: Iterable[CycleRecord]) -> dict[int, tuple[CycleRecord, ...]]:
    grouped: dict[int, list[CycleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cycle_index].append(record)
    return {
        cycle_index: tuple(sorted(cycle_records, key=lambda record: record.sample_index))
        for cycle_index, cycle_records in grouped.items()
    }


def _cycle_maximum(records: Sequence[CycleRecord], field_name: str) -> float | None:
    observations = [getattr(record, field_name) for record in records]
    values = [float(value) for value in observations if value is not None]
    return max(values) if values else None


def _cycle_duration_s(records: Sequence[CycleRecord]) -> float | None:
    if len(records) < 2:
        return None
    return float(records[-1].time_s - records[0].time_s)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    mean = fmean(values)
    return float(mean), float(np.std(values, ddof=0))


def _populate_capacity_features(
    values: dict[str, float | None],
    by_cycle: dict[int, tuple[CycleRecord, ...]],
    source_cycles: tuple[int, ...],
    warnings: list[str],
) -> None:
    capacity_by_cycle = {
        cycle: _cycle_maximum(by_cycle[cycle], "discharge_capacity_ah") for cycle in source_cycles
    }
    all_observed = [
        (cycle, capacity) for cycle, capacity in capacity_by_cycle.items() if capacity is not None
    ]
    observed = [
        (cycle, capacity)
        for cycle, capacity in all_observed
        if capacity is not None and capacity > 0
    ]
    if len(observed) != len(all_observed):
        warnings.append("NONPOSITIVE_CAPACITY_EXCLUDED_FROM_TREND")
    if len(observed) < 2:
        warnings.append("DISCHARGE_CAPACITY_UNAVAILABLE")
        return

    first_cycle, first_capacity = observed[0]
    last_cycle, last_capacity = observed[-1]
    assert first_capacity is not None and last_capacity is not None
    delta = last_capacity - first_capacity
    values["capacity_first_ah"] = first_capacity
    values["capacity_last_ah"] = last_capacity
    values["capacity_delta_ah"] = delta
    values["capacity_relative_change"] = delta / first_capacity
    values["capacity_slope_ah_per_cycle"] = delta / (last_cycle - first_cycle)


def _populate_coulombic_efficiency_features(
    values: dict[str, float | None],
    by_cycle: dict[int, tuple[CycleRecord, ...]],
    source_cycles: tuple[int, ...],
    warnings: list[str],
) -> None:
    efficiencies: list[float] = []
    for cycle in source_cycles:
        records = by_cycle[cycle]
        charge = _cycle_maximum(records, "charge_capacity_ah")
        discharge = _cycle_maximum(records, "discharge_capacity_ah")
        if charge is not None and charge > 0 and discharge is not None:
            efficiencies.append(discharge / charge)
    if not efficiencies:
        warnings.append("COULOMBIC_EFFICIENCY_UNAVAILABLE")
        return
    mean, std = _mean_std(efficiencies)
    values["coulombic_efficiency_mean"] = mean
    values["coulombic_efficiency_last"] = efficiencies[-1]
    values["coulombic_efficiency_std"] = std


def _populate_duration_features(
    values: dict[str, float | None],
    by_cycle: dict[int, tuple[CycleRecord, ...]],
    source_cycles: tuple[int, ...],
    warnings: list[str],
) -> None:
    durations = [(cycle, _cycle_duration_s(by_cycle[cycle])) for cycle in source_cycles]
    observed = [(cycle, duration) for cycle, duration in durations if duration is not None]
    if len(observed) < 2:
        warnings.append("CYCLE_DURATION_UNAVAILABLE")
        return
    _, first_duration = observed[0]
    _, last_duration = observed[-1]
    assert first_duration is not None and last_duration is not None
    values["cycle_duration_first_s"] = first_duration
    values["cycle_duration_last_s"] = last_duration
    values["cycle_duration_delta_s"] = last_duration - first_duration


def _populate_temperature_features(
    values: dict[str, float | None], records: Sequence[CycleRecord], warnings: list[str]
) -> None:
    observed = [
        float(record.temperature_c) for record in records if record.temperature_c is not None
    ]
    values["temperature_missing_fraction"] = 1.0 - len(observed) / len(records)
    if not observed:
        warnings.append("TEMPERATURE_UNAVAILABLE")
        return
    mean, std = _mean_std(observed)
    values["temperature_mean_c"] = mean
    values["temperature_max_c"] = max(observed)
    values["temperature_std_c"] = std


def _populate_resistance_features(
    values: dict[str, float | None],
    by_cycle: dict[int, tuple[CycleRecord, ...]],
    source_cycles: tuple[int, ...],
    warnings: list[str],
) -> None:
    observed = [
        (cycle, resistance)
        for cycle in source_cycles
        if (resistance := _cycle_maximum(by_cycle[cycle], "internal_resistance_ohm")) is not None
    ]
    if len(observed) < 2:
        warnings.append("INTERNAL_RESISTANCE_UNAVAILABLE")
        return
    first_cycle, first_resistance = observed[0]
    last_cycle, last_resistance = observed[-1]
    assert first_resistance is not None and last_resistance is not None
    delta = last_resistance - first_resistance
    values["internal_resistance_first_ohm"] = first_resistance
    values["internal_resistance_last_ohm"] = last_resistance
    values["internal_resistance_delta_ohm"] = delta
    values["internal_resistance_slope_ohm_per_cycle"] = delta / (last_cycle - first_cycle)


def _populate_delta_q_features(
    values: dict[str, float | None],
    by_cycle: dict[int, tuple[CycleRecord, ...]],
    config: EarlyCycleFeatureConfig,
    warnings: list[str],
) -> tuple[int, int] | None:
    selection = _select_delta_q_curves(by_cycle, config.min_curve_points)
    if selection is None:
        warnings.append("DELTA_Q_UNAVAILABLE")
        return None
    curves, used_non_diagnostic_fallback = selection
    if used_non_diagnostic_fallback:
        warnings.append("DELTA_Q_NON_DIAGNOSTIC_FALLBACK")
    (first_cycle, first_voltage, first_capacity), (last_cycle, last_voltage, last_capacity) = curves

    lower = max(float(first_voltage[0]), float(last_voltage[0]))
    upper = min(float(first_voltage[-1]), float(last_voltage[-1]))
    if upper - lower < config.voltage_grid_step_v:
        warnings.append("DELTA_Q_INSUFFICIENT_VOLTAGE_OVERLAP")
        warnings.append("DELTA_Q_UNAVAILABLE")
        return None

    point_count = int(np.floor((upper - lower) / config.voltage_grid_step_v + 1e-12)) + 1
    grid = lower + np.arange(point_count, dtype=np.float64) * config.voltage_grid_step_v
    if len(grid) < config.min_curve_points:
        warnings.append("DELTA_Q_UNAVAILABLE")
        return None

    first_interpolator = PchipInterpolator(first_voltage, first_capacity, extrapolate=False)
    last_interpolator = PchipInterpolator(last_voltage, last_capacity, extrapolate=False)
    delta_q = np.asarray(last_interpolator(grid) - first_interpolator(grid), dtype=np.float64)
    if not np.all(np.isfinite(delta_q)):
        warnings.append("DELTA_Q_UNAVAILABLE")
        return None

    values["delta_q_mean_ah"] = float(np.mean(delta_q))
    values["delta_q_variance_ah2"] = float(np.var(delta_q, ddof=0))
    values["delta_q_min_ah"] = float(np.min(delta_q))
    values["delta_q_max_ah"] = float(np.max(delta_q))
    values["delta_q_l2_ah"] = float(np.sqrt(np.mean(np.square(delta_q))))
    return first_cycle, last_cycle


def _select_delta_q_curves(
    by_cycle: dict[int, tuple[CycleRecord, ...]], min_curve_points: int
) -> (
    tuple[
        tuple[tuple[int, np.ndarray, np.ndarray], tuple[int, np.ndarray, np.ndarray]],
        bool,
    ]
    | None
):
    diagnostic_candidates = _curves_from_cycles(
        {
            cycle: records
            for cycle, records in by_cycle.items()
            if any(record.diagnostic for record in records)
        },
        min_curve_points,
    )
    use_non_diagnostic_fallback = len(diagnostic_candidates) < 2
    candidates = (
        _curves_from_cycles(by_cycle, min_curve_points)
        if use_non_diagnostic_fallback
        else diagnostic_candidates
    )
    if len(candidates) < 2:
        return None
    return (candidates[0], candidates[-1]), use_non_diagnostic_fallback


def _curves_from_cycles(
    by_cycle: dict[int, tuple[CycleRecord, ...]], min_curve_points: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    curves: list[tuple[int, np.ndarray, np.ndarray]] = []
    for cycle, records in sorted(by_cycle.items()):
        curve = _clean_voltage_capacity_curve(records, min_curve_points)
        if curve is not None:
            voltage, capacity = curve
            curves.append((cycle, voltage, capacity))
    return curves


def _clean_voltage_capacity_curve(
    records: Sequence[CycleRecord], min_curve_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = sorted(
        (
            (float(record.voltage_v), float(record.discharge_capacity_ah))
            for record in records
            if record.discharge_capacity_ah is not None
        ),
        key=lambda pair: pair[0],
    )
    if not pairs:
        return None

    grouped: dict[float, list[float]] = defaultdict(list)
    for voltage_value, capacity_value in pairs:
        grouped[voltage_value].append(capacity_value)
    voltage_grid = np.asarray(sorted(grouped), dtype=np.float64)
    capacity_grid = np.asarray(
        [fmean(grouped[float(item)]) for item in voltage_grid], dtype=np.float64
    )
    if len(voltage_grid) < min_curve_points or not np.all(np.diff(voltage_grid) > 0):
        return None
    return voltage_grid, capacity_grid
