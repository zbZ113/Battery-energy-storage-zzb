"""Build label-free fixed-grid early-cycle sequences from canonical records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]

from quanxin_life.data.schemas import (
    CycleRecord,
    DataQualitySeverity,
)
from quanxin_life.data.validation import validate_cycle_records
from quanxin_life.features.early_cycle_sequence import (
    SAMPLES_PER_PHASE,
    EarlyCycleSequence,
)

SUPPORTED_CUTOFFS = frozenset({20, 50, 100, 150})
CURRENT_ZERO_TOLERANCE_A = 1e-9
CONDITION_NAMES = (
    "mean_temperature_c",
    "mean_charge_current_a",
    "mean_discharge_current_a",
)


@dataclass(frozen=True)
class MultichannelCycleConfig:
    """Versioned, fixed-shape extraction settings without supervision fields."""

    cutoff_cycle: int
    feature_version: str
    samples_per_phase: int = SAMPLES_PER_PHASE
    min_phase_points: int = 3
    time_monotonic_tolerance_s: float = 1e-9
    capacity_monotonic_tolerance_ah: float = 1e-6
    max_phase_segments: int = 4

    def __post_init__(self) -> None:
        if self.cutoff_cycle not in SUPPORTED_CUTOFFS:
            raise ValueError(f"cutoff_cycle must be one of {sorted(SUPPORTED_CUTOFFS)}")
        if not self.feature_version.strip():
            raise ValueError("feature_version must be non-empty")
        if self.samples_per_phase != SAMPLES_PER_PHASE:
            raise ValueError(
                f"samples_per_phase must be fixed to {SAMPLES_PER_PHASE}"
            )
        if self.min_phase_points < 3:
            raise ValueError("min_phase_points must be at least 3")
        if (
            not isfinite(self.time_monotonic_tolerance_s)
            or self.time_monotonic_tolerance_s < 0
        ):
            raise ValueError(
                "time_monotonic_tolerance_s must be finite and non-negative"
            )
        if (
            not isfinite(self.capacity_monotonic_tolerance_ah)
            or self.capacity_monotonic_tolerance_ah < 0
        ):
            raise ValueError(
                "capacity_monotonic_tolerance_ah must be finite and non-negative"
            )
        if self.max_phase_segments < 1:
            raise ValueError("max_phase_segments must be at least 1")


def build_early_cycle_sequence(
    records: Sequence[CycleRecord],
    *,
    config: MultichannelCycleConfig,
    data_version: str,
) -> EarlyCycleSequence:
    """Convert cutoff-bounded observations without accepting any label input."""

    if not data_version.strip():
        raise ValueError("data_version must be non-empty")
    if any(record.cycle_index > config.cutoff_cycle for record in records):
        raise ValueError("input records must not exceed cutoff_cycle")

    valid_records = tuple(
        sorted(
            (record for record in records if record.valid),
            key=_record_sort_key,
        )
    )
    quality = validate_cycle_records(
        valid_records,
        time_monotonic_tolerance_s=config.time_monotonic_tolerance_s,
    )
    severe_issues = tuple(
        issue
        for issue in quality.issues
        if issue.severity in {DataQualitySeverity.ERROR, DataQualitySeverity.BLOCKING}
    )
    if severe_issues:
        codes = ", ".join(issue.code for issue in severe_issues)
        raise ValueError(f"cycle records failed quality validation: {codes}")

    dataset_ids = {record.dataset_id for record in valid_records}
    cell_ids = {record.cell_id for record in valid_records}
    if len(dataset_ids) != 1 or len(cell_ids) != 1:
        raise ValueError("records must describe a single dataset and single cell")

    cycle_count = config.cutoff_cycle + 1
    values = torch.full(
        (cycle_count, 2, config.samples_per_phase, 3),
        float("nan"),
        dtype=torch.float32,
    )
    sample_mask = torch.zeros(
        (cycle_count, 2, config.samples_per_phase), dtype=torch.bool
    )
    by_cycle: dict[int, list[CycleRecord]] = defaultdict(list)
    for record in valid_records:
        by_cycle[record.cycle_index].append(record)

    for cycle_index, cycle_records in by_cycle.items():
        charge = _resample_phase(
            cycle_records,
            charge=True,
            samples=config.samples_per_phase,
            min_points=config.min_phase_points,
            capacity_tolerance=config.capacity_monotonic_tolerance_ah,
            max_segments=config.max_phase_segments,
        )
        discharge = _resample_phase(
            cycle_records,
            charge=False,
            samples=config.samples_per_phase,
            min_points=config.min_phase_points,
            capacity_tolerance=config.capacity_monotonic_tolerance_ah,
            max_segments=config.max_phase_segments,
        )
        for phase_index, phase_values in enumerate((charge, discharge)):
            if phase_values is None:
                continue
            values[cycle_index, phase_index] = phase_values
            sample_mask[cycle_index, phase_index] = True

    cycle_mask = sample_mask.any(dim=(1, 2))
    if not bool(cycle_mask.any().item()):
        raise ValueError("cell does not contain any valid phase before cutoff_cycle")

    condition_values, condition_mask = _conditions(valid_records)
    first = valid_records[0]
    return EarlyCycleSequence(
        dataset_id=first.dataset_id,
        cell_id=first.cell_id,
        cutoff_cycle=config.cutoff_cycle,
        data_version=data_version,
        feature_version=config.feature_version,
        cycle_indices=tuple(range(cycle_count)),
        values=values,
        cycle_mask=cycle_mask,
        sample_mask=sample_mask,
        condition_names=CONDITION_NAMES,
        condition_values=condition_values,
        condition_mask=condition_mask,
    )


def _resample_phase(
    records: Sequence[CycleRecord],
    *,
    charge: bool,
    samples: int,
    min_points: int,
    capacity_tolerance: float,
    max_segments: int,
) -> torch.Tensor | None:
    raw_phase_records: list[tuple[float, CycleRecord]] = []
    for record in sorted(records, key=_phase_record_sort_key):
        if charge:
            if record.current_a <= CURRENT_ZERO_TOLERANCE_A:
                continue
            capacity = record.charge_capacity_ah
        else:
            if record.current_a >= -CURRENT_ZERO_TOLERANCE_A:
                continue
            capacity = record.discharge_capacity_ah
        if capacity is not None:
            raw_phase_records.append((capacity, record))

    segments: list[list[tuple[float, CycleRecord]]] = []
    for capacity, record in raw_phase_records:
        if not segments:
            segments.append([(capacity, record)])
            continue
        previous_capacity = segments[-1][-1][0]
        if capacity < previous_capacity - capacity_tolerance:
            segments.append([(capacity, record)])
        else:
            snapped_capacity = previous_capacity if capacity < previous_capacity else capacity
            segments[-1].append((snapped_capacity, record))
    if len(segments) > max_segments:
        raise ValueError(
            "TOO_MANY_PHASE_SEGMENTS: capacity reset count exceeds configured limit"
        )

    qualified = [
        segment
        for segment in segments
        if len({capacity for capacity, _ in segment}) >= min_points
        and segment[-1][0] > segment[0][0]
    ]
    if not qualified:
        return None
    phase_records = max(
        qualified,
        key=lambda segment: (
            segment[-1][0] - segment[0][0],
            len({capacity for capacity, _ in segment}),
            -segment[0][1].sample_index,
            -segment[0][1].time_s,
        ),
    )

    grouped: dict[float, list[CycleRecord]] = defaultdict(list)
    for capacity, record in phase_records:
        grouped[capacity].append(record)
    for capacity_records in grouped.values():
        capacity_records.sort(key=_phase_record_sort_key)

    capacities = np.asarray(sorted(grouped), dtype=np.float64)
    if len(capacities) < min_points or capacities[-1] <= capacities[0]:
        return None
    voltages = np.asarray(
        [_mean(item.voltage_v for item in grouped[capacity]) for capacity in capacities],
        dtype=np.float64,
    )
    currents = np.asarray(
        [_mean(item.current_a for item in grouped[capacity]) for capacity in capacities],
        dtype=np.float64,
    )
    progress = (capacities - capacities[0]) / (capacities[-1] - capacities[0])
    grid = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    resampled = np.column_stack(
        (
            PchipInterpolator(progress, voltages, extrapolate=False)(grid),
            PchipInterpolator(progress, currents, extrapolate=False)(grid),
            PchipInterpolator(progress, capacities, extrapolate=False)(grid),
        )
    )
    if not np.isfinite(resampled).all():
        return None
    return torch.from_numpy(resampled.astype(np.float32, copy=False))


def _conditions(records: Sequence[CycleRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    temperatures = [record.temperature_c for record in records if record.temperature_c is not None]
    charge_currents = [
        record.current_a
        for record in records
        if record.current_a > CURRENT_ZERO_TOLERANCE_A
    ]
    discharge_currents = [
        record.current_a
        for record in records
        if record.current_a < -CURRENT_ZERO_TOLERANCE_A
    ]
    groups: tuple[Sequence[float], ...] = (
        temperatures,
        charge_currents,
        discharge_currents,
    )
    mask = torch.tensor([bool(group) for group in groups], dtype=torch.bool)
    values = torch.tensor(
        [_mean(group) if group else float("nan") for group in groups],
        dtype=torch.float32,
    )
    return values, mask


def _mean(values: Iterable[float]) -> float:
    materialized: tuple[float, ...] = tuple(values)
    return fsum(materialized) / len(materialized)


def _phase_record_sort_key(record: CycleRecord) -> tuple[int, float, float, float]:
    return (
        record.sample_index,
        record.time_s,
        record.voltage_v,
        record.current_a,
    )


def _record_sort_key(
    record: CycleRecord,
) -> tuple[
    str,
    str,
    int,
    int,
    float,
    float,
    float,
    tuple[bool, float],
    tuple[bool, float],
    tuple[bool, float],
]:
    return (
        record.dataset_id,
        record.cell_id,
        record.cycle_index,
        record.sample_index,
        record.time_s,
        record.voltage_v,
        record.current_a,
        _optional_float_sort_key(record.temperature_c),
        _optional_float_sort_key(record.charge_capacity_ah),
        _optional_float_sort_key(record.discharge_capacity_ah),
    )


def _optional_float_sort_key(value: float | None) -> tuple[bool, float]:
    return (value is None, 0.0 if value is None else value)
