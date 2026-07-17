"""Build cutoff-safe, masked discharge-capacity curves for sequence models.

The output deliberately remains a Python/Pydantic data structure.  A later
PyTorch adapter may convert it to tensors only after preserving the cell-level
split, fixed cycle axis, voltage grid, and missing-curve mask defined here.
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
from quanxin_life.features.telemetry import exclude_exact_duplicate_telemetry

CURVE_TENSOR_FEATURE_VERSION = "discharge-curve-tensor-v1"
_SUPPORTED_CUTOFF_CYCLES = (20, 50, 100, 150)


class CurveTensorConfig(BaseModel):
    """Versioned interpolation and cycle-axis configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cutoff_cycle: int
    voltage_grid_step_v: float = Field(default=0.01, gt=0, le=0.1)
    voltage_min_v: float | None = Field(default=None, allow_inf_nan=False)
    voltage_max_v: float | None = Field(default=None, allow_inf_nan=False)
    min_curve_points: int = Field(default=3, ge=2)
    feature_version: str = Field(default=CURVE_TENSOR_FEATURE_VERSION, min_length=1)

    @model_validator(mode="after")
    def cutoff_is_supported(self) -> CurveTensorConfig:
        if self.cutoff_cycle not in _SUPPORTED_CUTOFF_CYCLES:
            raise ValueError(
                f"cutoff_cycle must be one of {_SUPPORTED_CUTOFF_CYCLES}, got {self.cutoff_cycle}"
            )
        if (self.voltage_min_v is None) != (self.voltage_max_v is None):
            raise ValueError("voltage_min_v and voltage_max_v must be provided together")
        if self.voltage_min_v is not None and self.voltage_max_v is not None:
            if self.voltage_max_v <= self.voltage_min_v:
                raise ValueError("voltage_max_v must exceed voltage_min_v")
            point_count = (
                int(
                    np.floor(
                        (self.voltage_max_v - self.voltage_min_v)
                        / self.voltage_grid_step_v
                        + 1e-12
                    )
                )
                + 1
            )
            if point_count < self.min_curve_points:
                raise ValueError("explicit voltage grid has fewer than min_curve_points")
        return self


class CurveTensor(BaseModel):
    """Fixed-cycle-axis, fixed-voltage-grid curve matrix with an explicit mask."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    feature_version: str = Field(min_length=1)
    cycle_indices: tuple[int, ...]
    voltage_grid_v: tuple[float, ...]
    values: tuple[tuple[float | None, ...], ...]
    observed_mask: tuple[bool, ...]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def values_align_with_axes_and_mask(self) -> CurveTensor:
        if len(self.cycle_indices) != len(self.values) or len(self.values) != len(
            self.observed_mask
        ):
            raise ValueError("cycle_indices, values, and observed_mask must have equal lengths")
        if self.cycle_indices != tuple(sorted(self.cycle_indices)):
            raise ValueError("cycle_indices must be sorted")
        if any(cycle > self.cutoff_cycle for cycle in self.cycle_indices):
            raise ValueError("cycle_indices cannot exceed cutoff_cycle")
        if any(
            current <= previous
            for previous, current in zip(self.voltage_grid_v, self.voltage_grid_v[1:], strict=False)
        ):
            raise ValueError("voltage_grid_v must be strictly increasing")
        for row, observed in zip(self.values, self.observed_mask, strict=True):
            if len(row) != len(self.voltage_grid_v):
                raise ValueError("every curve row must match voltage_grid_v length")
            if observed and any(value is None for value in row):
                raise ValueError("observed curve rows cannot contain missing grid values")
            if not observed and any(value is not None for value in row):
                raise ValueError("unobserved curve rows must retain None values")
        return self


def build_discharge_curve_tensor(
    records: Sequence[CycleRecord], *, config: CurveTensorConfig
) -> CurveTensor:
    """Interpolate valid discharge-capacity curves on a shared fixed voltage grid.

    The fixed cycle axis is ``0..cutoff_cycle``.  Missing or insufficient curves
    occupy a fully ``None`` row and receive ``observed_mask=False``; no missing
    capacity is replaced by zero.
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
        raise ValueError(f"curve tensor extraction rejected due to data quality: {codes}")

    valid_records = tuple(record for record in prepared_records if record.valid)
    if not valid_records:
        raise ValueError("curve tensor extraction requires at least one valid record")
    dataset_ids = {record.dataset_id for record in valid_records}
    cell_ids = {record.cell_id for record in valid_records}
    if len(dataset_ids) != 1 or len(cell_ids) != 1:
        raise ValueError("curve tensor extraction requires exactly one dataset and one cell")

    cycle_indices = tuple(range(config.cutoff_cycle + 1))
    by_cycle = _group_by_cycle(valid_records)
    warnings: list[str] = []
    if duplicate_count:
        warnings.append("EXACT_DUPLICATE_TELEMETRY_EXCLUDED")
    if len(valid_records) != len(prepared_records):
        warnings.append("INVALID_RECORDS_EXCLUDED")

    curves = {
        cycle_index: curve
        for cycle_index in cycle_indices
        if (curve := _clean_discharge_curve(by_cycle.get(cycle_index, ()), config.min_curve_points))
        is not None
    }
    for cycle_index in cycle_indices:
        if cycle_index not in curves:
            warnings.append(f"CURVE_UNAVAILABLE_CYCLE_{cycle_index}")

    voltage_grid: np.ndarray | None
    if config.voltage_min_v is not None and config.voltage_max_v is not None:
        voltage_grid = _bounded_voltage_grid(
            lower=config.voltage_min_v,
            upper=config.voltage_max_v,
            step=config.voltage_grid_step_v,
        )
    else:
        voltage_grid = _shared_voltage_grid(
            tuple(curves.values()),
            voltage_grid_step_v=config.voltage_grid_step_v,
            min_curve_points=config.min_curve_points,
        )
    if voltage_grid is None:
        warnings.append("CURVE_TENSOR_GRID_UNAVAILABLE")
        return CurveTensor(
            dataset_id=next(iter(dataset_ids)),
            cell_id=next(iter(cell_ids)),
            cutoff_cycle=config.cutoff_cycle,
            feature_version=config.feature_version,
            cycle_indices=cycle_indices,
            voltage_grid_v=(),
            values=tuple(() for _ in cycle_indices),
            observed_mask=tuple(False for _ in cycle_indices),
            warnings=tuple(sorted(set(warnings))),
        )

    rows: list[tuple[float | None, ...]] = []
    observed_mask: list[bool] = []
    for cycle_index in cycle_indices:
        curve = curves.get(cycle_index)
        if curve is None:
            rows.append(tuple(None for _ in voltage_grid))
            observed_mask.append(False)
            continue
        voltage, capacity = curve
        interpolation = PchipInterpolator(voltage, capacity, extrapolate=False)
        resampled = np.asarray(interpolation(voltage_grid), dtype=np.float64)
        if not np.all(np.isfinite(resampled)):
            warnings.append(f"CURVE_INTERPOLATION_UNAVAILABLE_CYCLE_{cycle_index}")
            rows.append(tuple(None for _ in voltage_grid))
            observed_mask.append(False)
            continue
        rows.append(tuple(float(value) for value in resampled))
        observed_mask.append(True)

    return CurveTensor(
        dataset_id=next(iter(dataset_ids)),
        cell_id=next(iter(cell_ids)),
        cutoff_cycle=config.cutoff_cycle,
        feature_version=config.feature_version,
        cycle_indices=cycle_indices,
        voltage_grid_v=tuple(float(value) for value in voltage_grid),
        values=tuple(rows),
        observed_mask=tuple(observed_mask),
        warnings=tuple(sorted(set(warnings))),
    )


def _group_by_cycle(records: Sequence[CycleRecord]) -> dict[int, tuple[CycleRecord, ...]]:
    grouped: dict[int, list[CycleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cycle_index].append(record)
    return {
        cycle_index: tuple(sorted(cycle_records, key=lambda record: record.sample_index))
        for cycle_index, cycle_records in grouped.items()
    }


def _clean_discharge_curve(
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
    capacity_by_voltage: dict[float, list[float]] = defaultdict(list)
    for voltage_v, capacity_ah in pairs:
        capacity_by_voltage[voltage_v].append(capacity_ah)
    voltage = np.asarray(sorted(capacity_by_voltage), dtype=np.float64)
    capacity = np.asarray(
        [fmean(capacity_by_voltage[float(point)]) for point in voltage], dtype=np.float64
    )
    if len(voltage) < min_curve_points or not np.all(np.diff(voltage) > 0):
        return None
    return voltage, capacity


def _shared_voltage_grid(
    curves: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    voltage_grid_step_v: float,
    min_curve_points: int,
) -> np.ndarray | None:
    if not curves:
        return None
    lower = max(float(voltage[0]) for voltage, _ in curves)
    upper = min(float(voltage[-1]) for voltage, _ in curves)
    if upper < lower:
        return None
    point_count = int(np.floor((upper - lower) / voltage_grid_step_v + 1e-12)) + 1
    if point_count < min_curve_points:
        return None
    return lower + np.arange(point_count, dtype=np.float64) * voltage_grid_step_v


def _bounded_voltage_grid(*, lower: float, upper: float, step: float) -> np.ndarray:
    point_count = int(np.floor((upper - lower) / step + 1e-12)) + 1
    return lower + np.arange(point_count, dtype=np.float64) * step
