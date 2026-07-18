from collections import defaultdict
from collections.abc import Sequence
from itertools import pairwise
from math import isfinite

from quanxin_life.data.schemas import (
    CycleRecord,
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
)


def validate_cycle_records(
    records: Sequence[CycleRecord],
    *,
    time_monotonic_tolerance_s: float = 0.0,
) -> DataQualityReport:
    """Run deterministic structural checks before feature extraction or labeling."""
    if (
        not isfinite(time_monotonic_tolerance_s)
        or time_monotonic_tolerance_s < 0.0
    ):
        raise ValueError("time_monotonic_tolerance_s must be finite and non-negative")
    if not records:
        return DataQualityReport(
            dataset_id="unknown",
            issues=(
                DataQualityIssue(
                    code="EMPTY_CELL",
                    severity=DataQualitySeverity.BLOCKING,
                    message="cell has no sample records",
                ),
            ),
        )

    issues: list[DataQualityIssue] = []
    dataset_ids = {record.dataset_id for record in records}
    cell_ids = {record.cell_id for record in records}
    dataset_id = next(iter(dataset_ids)) if len(dataset_ids) == 1 else "mixed"

    if len(dataset_ids) != 1:
        issues.append(
            DataQualityIssue(
                code="MIXED_DATASET",
                severity=DataQualitySeverity.BLOCKING,
                message="records from multiple datasets cannot form one cell",
            )
        )
    if len(cell_ids) != 1:
        issues.append(
            DataQualityIssue(
                code="MIXED_CELL",
                severity=DataQualitySeverity.BLOCKING,
                message="records from multiple cells cannot be validated as one cell",
            )
        )

    seen: set[tuple[str, int, int]] = set()
    by_cycle: dict[int, list[CycleRecord]] = defaultdict(list)
    for record in records:
        key = (record.cell_id, record.cycle_index, record.sample_index)
        if key in seen:
            issues.append(
                DataQualityIssue(
                    code="DUPLICATE_SAMPLE",
                    severity=DataQualitySeverity.BLOCKING,
                    message="duplicate (cell, cycle, sample) key",
                    cell_id=record.cell_id,
                    cycle_index=record.cycle_index,
                )
            )
        seen.add(key)
        by_cycle[record.cycle_index].append(record)

    for cycle_index, cycle_records in sorted(by_cycle.items()):
        ordered = sorted(cycle_records, key=lambda record: record.sample_index)
        times = [record.time_s for record in ordered]
        adjacent_times = tuple(pairwise(times))
        if any(
            current <= previous
            if time_monotonic_tolerance_s == 0.0
            else current < previous - time_monotonic_tolerance_s
            for previous, current in adjacent_times
        ):
            issues.append(
                DataQualityIssue(
                    code="NON_MONOTONIC_TIME",
                    severity=DataQualitySeverity.ERROR,
                    message="elapsed time must increase within a cycle",
                    cell_id=ordered[0].cell_id,
                    cycle_index=cycle_index,
                )
            )
        elif time_monotonic_tolerance_s > 0.0 and any(
            current <= previous for previous, current in adjacent_times
        ):
            issues.append(
                DataQualityIssue(
                    code="TIME_WITHIN_NUMERIC_TOLERANCE",
                    severity=DataQualitySeverity.WARNING,
                    message=(
                        "elapsed time is non-increasing only within the approved "
                        "numeric tolerance"
                    ),
                    cell_id=ordered[0].cell_id,
                    cycle_index=cycle_index,
                )
            )

    cycles = sorted(by_cycle)
    if any(current - previous > 1 for previous, current in pairwise(cycles)):
        issues.append(
            DataQualityIssue(
                code="CYCLE_GAP",
                severity=DataQualitySeverity.WARNING,
                message="one or more cycle indices are missing",
                cell_id=records[0].cell_id,
            )
        )
    if any(record.temperature_c is None for record in records):
        issues.append(
            DataQualityIssue(
                code="MISSING_TEMPERATURE",
                severity=DataQualitySeverity.WARNING,
                message="temperature is unavailable for one or more samples",
                cell_id=records[0].cell_id,
            )
        )

    return DataQualityReport(dataset_id=dataset_id, issues=tuple(issues))
