"""Leakage-safe ΔQ(V) variance extraction for explicitly selected cycles.

This module is deliberately a feature primitive rather than a lifetime model.
It accepts only canonical :class:`CycleRecord` objects, rejects any record
beyond the requested early-cycle cutoff, and never replaces unavailable curves
with a fabricated numeric value.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from statistics import fmean

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from quanxin_life.data.leakage import assert_cycles_within_cutoff
from quanxin_life.data.schemas import CycleRecord, DataQualitySeverity
from quanxin_life.data.validation import validate_cycle_records

DELTA_Q_VARIANCE_FEATURE_VERSION = "delta-q-variance-v1"


class DeltaQVarianceConfig(BaseModel):
    """Versioned definition of the fixed voltage grid used for ΔQ(V)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_cycle: int = Field(ge=0)
    voltage_grid_step_v: float = Field(default=0.01, gt=0, le=0.1)
    min_curve_points: int = Field(default=3, ge=2)
    feature_version: str = Field(default=DELTA_Q_VARIANCE_FEATURE_VERSION, min_length=1)


class DeltaQVarianceFeature(BaseModel):
    """Traceable ΔQ(V) variance for one explicitly selected pair of cycles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    anchor_cycle: int = Field(ge=0)
    comparison_cycle: int = Field(ge=0)
    valid_grid_point_count: int = Field(ge=0)
    delta_q_variance_ah2: float | None = Field(default=None, ge=0)
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def ordered_cycles_are_explicit(self) -> DeltaQVarianceFeature:
        if self.anchor_cycle >= self.comparison_cycle:
            raise ValueError("anchor_cycle must be earlier than comparison_cycle")
        return self


def extract_delta_q_variance(
    records: Sequence[CycleRecord],
    *,
    config: DeltaQVarianceConfig,
    anchor_cycle: int,
    comparison_cycle: int,
) -> DeltaQVarianceFeature:
    """Calculate variance of comparison-minus-anchor ΔQ on a shared fixed grid.

    ``anchor_cycle`` and ``comparison_cycle`` are intentionally explicit.  The
    implementation never guesses a pair from array positions, preventing an
    ordering change from silently changing the scientific meaning of a feature.
    """

    assert_cycles_within_cutoff(
        (record.cycle_index for record in records), cutoff_cycle=config.cutoff_cycle
    )
    if not records:
        raise ValueError("delta-q variance extraction requires at least one record")
    if anchor_cycle < 0 or comparison_cycle < 0:
        raise ValueError("anchor_cycle and comparison_cycle must be non-negative")
    if anchor_cycle >= comparison_cycle:
        raise ValueError("anchor_cycle must be earlier than comparison_cycle")
    if comparison_cycle > config.cutoff_cycle:
        raise ValueError("explicit comparison_cycle must not exceed cutoff_cycle")

    report = validate_cycle_records(records)
    fatal_issues = tuple(
        issue
        for issue in report.issues
        if issue.severity in {DataQualitySeverity.ERROR, DataQualitySeverity.BLOCKING}
    )
    if fatal_issues:
        codes = ", ".join(issue.code for issue in fatal_issues)
        raise ValueError(f"delta-q variance extraction rejected due to data quality: {codes}")

    dataset_ids = {record.dataset_id for record in records}
    cell_ids = {record.cell_id for record in records}
    if len(dataset_ids) != 1 or len(cell_ids) != 1:
        raise ValueError("delta-q variance extraction requires exactly one dataset and one cell")

    warnings: list[str] = []
    valid_records = tuple(record for record in records if record.valid)
    if len(valid_records) != len(records):
        warnings.append("INVALID_RECORDS_EXCLUDED")

    anchor_curve = _curve_for_cycle(valid_records, anchor_cycle, config.min_curve_points)
    comparison_curve = _curve_for_cycle(valid_records, comparison_cycle, config.min_curve_points)
    if anchor_curve is None:
        warnings.append("ANCHOR_CURVE_UNAVAILABLE")
    if comparison_curve is None:
        warnings.append("COMPARISON_CURVE_UNAVAILABLE")
    if anchor_curve is None or comparison_curve is None:
        return _unavailable_result(
            dataset_id=next(iter(dataset_ids)),
            cell_id=next(iter(cell_ids)),
            config=config,
            anchor_cycle=anchor_cycle,
            comparison_cycle=comparison_cycle,
            warnings=warnings,
        )

    anchor_voltage, anchor_capacity = anchor_curve
    comparison_voltage, comparison_capacity = comparison_curve
    lower_bound = max(float(anchor_voltage[0]), float(comparison_voltage[0]))
    upper_bound = min(float(anchor_voltage[-1]), float(comparison_voltage[-1]))
    grid = _shared_fixed_grid(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        step_v=config.voltage_grid_step_v,
    )
    if len(grid) < config.min_curve_points:
        warnings.append("DELTA_Q_INSUFFICIENT_VOLTAGE_OVERLAP")
        return _unavailable_result(
            dataset_id=next(iter(dataset_ids)),
            cell_id=next(iter(cell_ids)),
            config=config,
            anchor_cycle=anchor_cycle,
            comparison_cycle=comparison_cycle,
            warnings=warnings,
        )

    anchor_interpolator = PchipInterpolator(anchor_voltage, anchor_capacity, extrapolate=False)
    comparison_interpolator = PchipInterpolator(
        comparison_voltage, comparison_capacity, extrapolate=False
    )
    delta_q = np.asarray(
        comparison_interpolator(grid) - anchor_interpolator(grid), dtype=np.float64
    )
    if not np.all(np.isfinite(delta_q)):
        warnings.append("DELTA_Q_NONFINITE_INTERPOLATION")
        return _unavailable_result(
            dataset_id=next(iter(dataset_ids)),
            cell_id=next(iter(cell_ids)),
            config=config,
            anchor_cycle=anchor_cycle,
            comparison_cycle=comparison_cycle,
            warnings=warnings,
        )

    return DeltaQVarianceFeature(
        dataset_id=next(iter(dataset_ids)),
        cell_id=next(iter(cell_ids)),
        feature_version=config.feature_version,
        cutoff_cycle=config.cutoff_cycle,
        anchor_cycle=anchor_cycle,
        comparison_cycle=comparison_cycle,
        valid_grid_point_count=len(grid),
        delta_q_variance_ah2=float(np.var(delta_q, ddof=0)),
        warnings=tuple(sorted(set(warnings))),
    )


def _unavailable_result(
    *,
    dataset_id: str,
    cell_id: str,
    config: DeltaQVarianceConfig,
    anchor_cycle: int,
    comparison_cycle: int,
    warnings: Sequence[str],
) -> DeltaQVarianceFeature:
    return DeltaQVarianceFeature(
        dataset_id=dataset_id,
        cell_id=cell_id,
        feature_version=config.feature_version,
        cutoff_cycle=config.cutoff_cycle,
        anchor_cycle=anchor_cycle,
        comparison_cycle=comparison_cycle,
        valid_grid_point_count=0,
        delta_q_variance_ah2=None,
        warnings=tuple(sorted({*warnings, "DELTA_Q_UNAVAILABLE"})),
    )


def _curve_for_cycle(
    records: Sequence[CycleRecord], cycle_index: int, min_curve_points: int
) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = sorted(
        (
            (float(record.voltage_v), float(record.discharge_capacity_ah))
            for record in records
            if record.cycle_index == cycle_index and record.discharge_capacity_ah is not None
        ),
        key=lambda pair: pair[0],
    )
    if not pairs:
        return None

    capacities_by_voltage: dict[float, list[float]] = defaultdict(list)
    for voltage_v, capacity_ah in pairs:
        capacities_by_voltage[voltage_v].append(capacity_ah)
    voltage = np.asarray(sorted(capacities_by_voltage), dtype=np.float64)
    capacity = np.asarray(
        [fmean(capacities_by_voltage[float(point)]) for point in voltage], dtype=np.float64
    )
    if len(voltage) < min_curve_points or not np.all(np.diff(voltage) > 0):
        return None
    return voltage, capacity


def _shared_fixed_grid(*, lower_bound: float, upper_bound: float, step_v: float) -> np.ndarray:
    if upper_bound < lower_bound:
        return np.asarray([], dtype=np.float64)
    point_count = int(np.floor((upper_bound - lower_bound) / step_v + 1e-12)) + 1
    return lower_bound + np.arange(point_count, dtype=np.float64) * step_v
