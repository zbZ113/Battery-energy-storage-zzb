import pytest

from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features.curve_tensor import CurveTensorConfig, build_discharge_curve_tensor


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


def _records(*, second_cycle_has_capacity: bool = True) -> tuple[CycleRecord, ...]:
    first_cycle = tuple(
        _record(
            cycle_index=1,
            sample_index=sample_index,
            voltage_v=voltage_v,
            discharge_capacity_ah=capacity,
        )
        for sample_index, (voltage_v, capacity) in enumerate(((3.6, 0.0), (3.3, 0.3), (3.0, 0.6)))
    )
    second_cycle = tuple(
        _record(
            cycle_index=2,
            sample_index=sample_index,
            voltage_v=voltage_v,
            discharge_capacity_ah=capacity if second_cycle_has_capacity else None,
        )
        for sample_index, (voltage_v, capacity) in enumerate(((3.6, 0.0), (3.3, 0.27), (3.0, 0.54)))
    )
    return first_cycle + second_cycle


def test_builds_a_fixed_voltage_grid_and_explicit_cycle_mask() -> None:
    result = build_discharge_curve_tensor(
        tuple(reversed(_records())),
        config=CurveTensorConfig(cutoff_cycle=20, voltage_grid_step_v=0.1),
    )

    assert result.dataset_id == "MATR"
    assert result.cell_id == "MATR_b1c0"
    assert result.cycle_indices == tuple(range(21))
    assert result.observed_mask[0] is False
    assert result.observed_mask[1] is True
    assert result.observed_mask[2] is True
    assert result.voltage_grid_v == pytest.approx((3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6))
    assert all(value is None for value in result.values[0])
    assert result.values[1][0] == pytest.approx(0.6)
    assert result.values[2][-1] == pytest.approx(0.0)


def test_rejects_any_record_after_the_declared_cutoff() -> None:
    future = _record(
        cycle_index=21,
        sample_index=0,
        voltage_v=3.6,
        discharge_capacity_ah=0.0,
    )

    with pytest.raises(ValueError, match="after cutoff 20"):
        build_discharge_curve_tensor(
            (*_records(), future),
            config=CurveTensorConfig(cutoff_cycle=20),
        )


def test_preserves_missing_curves_as_masked_none_rows_without_zero_fill() -> None:
    result = build_discharge_curve_tensor(
        _records(second_cycle_has_capacity=False),
        config=CurveTensorConfig(cutoff_cycle=20, voltage_grid_step_v=0.1),
    )

    assert result.observed_mask[1] is True
    assert result.observed_mask[2] is False
    assert all(value is None for value in result.values[2])
    assert "CURVE_UNAVAILABLE_CYCLE_2" in result.warnings


def test_explicit_voltage_bounds_produce_a_cohort_stable_grid() -> None:
    result = build_discharge_curve_tensor(
        _records(),
        config=CurveTensorConfig(
            cutoff_cycle=20,
            voltage_grid_step_v=0.1,
            voltage_min_v=3.1,
            voltage_max_v=3.5,
        ),
    )

    assert result.voltage_grid_v == pytest.approx((3.1, 3.2, 3.3, 3.4, 3.5))
    assert result.observed_mask[1:3] == (True, True)


def test_explicit_voltage_bounds_must_be_provided_together() -> None:
    with pytest.raises(ValueError, match="provided together"):
        CurveTensorConfig(cutoff_cycle=20, voltage_min_v=3.0)


def test_excludes_exact_duplicate_zero_information_telemetry_rows() -> None:
    duplicate_zero = CycleRecord(
        dataset_id="MATR",
        cell_id="MATR_b1c0",
        cycle_index=0,
        sample_index=0,
        time_s=0.0,
        voltage_v=0.0,
        current_a=0.0,
        charge_capacity_ah=0.0,
        discharge_capacity_ah=0.0,
    )
    repeated = duplicate_zero.model_copy(update={"sample_index": 1})

    result = build_discharge_curve_tensor(
        (duplicate_zero, repeated, *_records()),
        config=CurveTensorConfig(cutoff_cycle=20, voltage_grid_step_v=0.1),
    )

    assert "EXACT_DUPLICATE_TELEMETRY_EXCLUDED" in result.warnings
    assert result.observed_mask[1:3] == (True, True)


def test_same_time_with_different_telemetry_is_preserved_with_warning() -> None:
    first, second, *remaining = _records()
    conflicting = second.model_copy(update={"time_s": first.time_s})

    result = build_discharge_curve_tensor(
        (first, conflicting, *remaining),
        config=CurveTensorConfig(
            cutoff_cycle=20,
            voltage_grid_step_v=0.1,
            time_monotonic_tolerance_s=1e-9,
        ),
    )

    assert "TIME_WITHIN_NUMERIC_TOLERANCE" in result.warnings
    assert result.observed_mask[1:3] == (True, True)
