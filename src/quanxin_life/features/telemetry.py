"""Shared, conservative sanitation for feature-only telemetry inputs."""

from __future__ import annotations

from collections.abc import Sequence

from quanxin_life.data.schemas import CycleRecord


def exclude_exact_duplicate_telemetry(
    records: Sequence[CycleRecord],
) -> tuple[tuple[CycleRecord, ...], int]:
    """Drop only identical measurements at the same cell, cycle and timestamp."""

    seen: set[tuple[object, ...]] = set()
    retained: list[CycleRecord] = []
    duplicate_count = 0
    for record in records:
        key = (
            record.dataset_id,
            record.cell_id,
            record.cycle_index,
            record.time_s,
            record.voltage_v,
            record.current_a,
            record.temperature_c,
            record.charge_capacity_ah,
            record.discharge_capacity_ah,
            record.internal_resistance_ohm,
            record.diagnostic,
            record.valid,
        )
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        retained.append(record)
    return tuple(retained), duplicate_count
