import math

import pytest

from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features.variance import (
    DeltaQVarianceConfig,
    extract_delta_q_variance,
)


def _record(
    *,
    cycle_index: int,
    sample_index: int,
    voltage_v: float,
    discharge_capacity_ah: float | None,
) -> CycleRecord:
    return CycleRecord(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cycle_index=cycle_index,
        sample_index=sample_index,
        time_s=float(sample_index + 1),
        voltage_v=voltage_v,
        current_a=-1.0,
        discharge_capacity_ah=discharge_capacity_ah,
    )


def _records(*, comparison_has_capacity: bool = True) -> tuple[CycleRecord, ...]:
    anchor = (
        _record(cycle_index=2, sample_index=0, voltage_v=3.6, discharge_capacity_ah=0.6),
        _record(cycle_index=2, sample_index=1, voltage_v=3.3, discharge_capacity_ah=0.3),
        _record(cycle_index=2, sample_index=2, voltage_v=3.0, discharge_capacity_ah=0.0),
    )
    comparison = (
        _record(
            cycle_index=5,
            sample_index=0,
            voltage_v=3.6,
            discharge_capacity_ah=0.72 if comparison_has_capacity else None,
        ),
        _record(
            cycle_index=5,
            sample_index=1,
            voltage_v=3.3,
            discharge_capacity_ah=0.33 if comparison_has_capacity else None,
        ),
        _record(
            cycle_index=5,
            sample_index=2,
            voltage_v=3.0,
            discharge_capacity_ah=0.0 if comparison_has_capacity else None,
        ),
    )
    return anchor + comparison


def test_uses_explicit_anchor_and_comparison_cycles_on_a_fixed_grid() -> None:
    result = extract_delta_q_variance(
        tuple(reversed(_records())),
        config=DeltaQVarianceConfig(cutoff_cycle=20, voltage_grid_step_v=0.1),
        anchor_cycle=2,
        comparison_cycle=5,
    )

    assert result.dataset_id == "MATR"
    assert result.cell_id == "MATR_b1c0"
    assert result.anchor_cycle == 2
    assert result.comparison_cycle == 5
    assert result.valid_grid_point_count == 7
    assert result.delta_q_variance_ah2 is not None
    assert math.isfinite(result.delta_q_variance_ah2)
    assert result.delta_q_variance_ah2 > 0


def test_rejects_any_future_record_before_selecting_explicit_cycles() -> None:
    future_record = _record(
        cycle_index=21,
        sample_index=0,
        voltage_v=3.6,
        discharge_capacity_ah=0.7,
    )

    with pytest.raises(ValueError, match="after cutoff 20"):
        extract_delta_q_variance(
            (*_records(), future_record),
            config=DeltaQVarianceConfig(cutoff_cycle=20),
            anchor_cycle=2,
            comparison_cycle=5,
        )


def test_returns_explicit_unavailable_result_without_fabricating_zero() -> None:
    result = extract_delta_q_variance(
        _records(comparison_has_capacity=False),
        config=DeltaQVarianceConfig(cutoff_cycle=20),
        anchor_cycle=2,
        comparison_cycle=5,
    )

    assert result.valid_grid_point_count == 0
    assert result.delta_q_variance_ah2 is None
    assert "DELTA_Q_UNAVAILABLE" in result.warnings


def test_deterministically_aggregates_duplicate_voltage_samples() -> None:
    duplicate = _record(
        cycle_index=2,
        sample_index=3,
        voltage_v=3.3,
        discharge_capacity_ah=0.5,
    )
    config = DeltaQVarianceConfig(cutoff_cycle=20, voltage_grid_step_v=0.1)

    forward = extract_delta_q_variance(
        (*_records(), duplicate), config=config, anchor_cycle=2, comparison_cycle=5
    )
    reverse = extract_delta_q_variance(
        tuple(reversed((*_records(), duplicate))),
        config=config,
        anchor_cycle=2,
        comparison_cycle=5,
    )

    assert forward.delta_q_variance_ah2 == pytest.approx(reverse.delta_q_variance_ah2)
    assert forward.valid_grid_point_count == reverse.valid_grid_point_count
