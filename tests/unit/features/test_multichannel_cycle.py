from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features.multichannel_cycle import (
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)


def _record(
    *,
    cycle: int,
    sample: int,
    current: float,
    capacity: float | None,
    voltage: float,
    temperature: float | None = 25.0,
    dataset_id: str = "MATR",
    cell_id: str = "b1c0",
    valid: bool = True,
) -> CycleRecord:
    return CycleRecord(
        dataset_id=dataset_id,
        cell_id=cell_id,
        cycle_index=cycle,
        sample_index=sample,
        time_s=float(sample),
        voltage_v=voltage,
        current_a=current,
        temperature_c=temperature,
        charge_capacity_ah=capacity if current > 1e-9 else None,
        discharge_capacity_ah=capacity if current < -1e-9 else None,
        valid=valid,
    )


def _phase_records(cycle: int = 1) -> tuple[CycleRecord, ...]:
    return (
        _record(cycle=cycle, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=cycle, sample=1, current=1.2, capacity=0.5, voltage=3.5),
        _record(cycle=cycle, sample=2, current=1.4, capacity=1.0, voltage=4.0),
        _record(cycle=cycle, sample=3, current=-0.8, capacity=0.0, voltage=4.1),
        _record(cycle=cycle, sample=4, current=-1.0, capacity=0.5, voltage=3.6),
        _record(cycle=cycle, sample=5, current=-1.2, capacity=1.0, voltage=3.1),
    )


def _config(cutoff: int = 20) -> MultichannelCycleConfig:
    return MultichannelCycleConfig(cutoff_cycle=cutoff, feature_version="multichannel-v1")


def test_config_fixes_supported_cutoffs_and_grid() -> None:
    for cutoff in (20, 50, 100, 150):
        assert _config(cutoff).cutoff_cycle == cutoff

    with pytest.raises(ValueError, match="cutoff"):
        _config(21)
    with pytest.raises(ValueError, match="150"):
        replace(_config(), samples_per_phase=149)
    with pytest.raises(ValueError, match="min_phase_points"):
        replace(_config(), min_phase_points=2)
    with pytest.raises(ValueError, match="capacity_monotonic_tolerance_ah"):
        replace(_config(), capacity_monotonic_tolerance_ah=-1.0)
    with pytest.raises(ValueError, match="relative_capacity_jitter_tolerance"):
        replace(_config(), relative_capacity_jitter_tolerance=-1.0)
    with pytest.raises(ValueError, match="max_phase_segments"):
        replace(_config(), max_phase_segments=0)


def test_builder_separates_phases_and_preserves_endpoints() -> None:
    sequence = build_early_cycle_sequence(
        _phase_records(), config=_config(), data_version="matr-v1"
    )

    assert sequence.values.shape == (21, 2, 150, 3)
    assert sequence.sample_mask[1].all()
    assert sequence.cycle_mask[1]
    assert sequence.values[1, 0, 0].tolist() == pytest.approx([3.0, 1.0, 0.0])
    assert sequence.values[1, 0, -1].tolist() == pytest.approx([4.0, 1.4, 1.0])
    assert sequence.values[1, 1, 0].tolist() == pytest.approx([4.1, -0.8, 0.0])
    assert sequence.values[1, 1, -1].tolist() == pytest.approx([3.1, -1.2, 1.0])
    assert sequence.normalization_version == "none"


def test_duplicate_capacity_points_are_averaged_before_interpolation() -> None:
    records = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=1.0, capacity=0.5, voltage=3.4),
        _record(cycle=1, sample=2, current=2.0, capacity=0.5, voltage=3.6),
        _record(cycle=1, sample=3, current=1.0, capacity=1.0, voltage=4.0),
    )

    sequence = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )

    midpoint = sequence.values[1, 0, 74:76].mean(dim=0)
    assert midpoint.tolist() == pytest.approx([3.5, 1.5, 0.5], abs=2e-3)


def test_invalid_phase_stays_nan_and_masked() -> None:
    charge_only = _phase_records()[:3]

    sequence = build_early_cycle_sequence(
        charge_only, config=_config(), data_version="matr-v1"
    )

    assert sequence.sample_mask[1, 0].all()
    assert not sequence.sample_mask[1, 1].any()
    assert torch.isnan(sequence.values[1, 1]).all()
    assert sequence.cycle_mask[1]
    assert not sequence.cycle_mask[0]


def test_builder_ignores_neutral_and_invalid_records_for_features() -> None:
    records = (
        *_phase_records()[:3],
        _record(cycle=1, sample=3, current=0.0, capacity=None, voltage=9.0),
        _record(
            cycle=1,
            sample=4,
            current=-1.0,
            capacity=0.0,
            voltage=9.0,
            temperature=99.0,
            valid=False,
        ),
    )

    sequence = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )

    assert sequence.condition_values.tolist() == pytest.approx(
        [25.0, 1.2, float("nan")], nan_ok=True
    )
    assert sequence.condition_mask.tolist() == [True, True, False]


def test_builder_rejects_future_cycles_and_mixed_identity() -> None:
    with pytest.raises(ValueError, match="cutoff"):
        build_early_cycle_sequence(
            _phase_records(cycle=21), config=_config(), data_version="matr-v1"
        )

    mixed = (
        *_phase_records(),
        _record(
            cycle=2,
            sample=0,
            current=1.0,
            capacity=0.0,
            voltage=3.0,
            cell_id="other",
        ),
    )
    with pytest.raises(ValueError, match=r"MIXED_CELL|single|multiple"):
        build_early_cycle_sequence(mixed, config=_config(), data_version="matr-v1")


def test_builder_rejects_structural_errors_from_existing_validator() -> None:
    records = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=1.0, capacity=0.5, voltage=3.5),
        _record(cycle=1, sample=2, current=1.0, capacity=1.0, voltage=4.0),
    )
    broken = (*records[:1], records[0], *records[1:])

    with pytest.raises(ValueError, match=r"quality|DUPLICATE_SAMPLE"):
        build_early_cycle_sequence(broken, config=_config(), data_version="matr-v1")


def test_builder_rejects_cell_without_any_valid_phase() -> None:
    too_short = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=1.0, capacity=1.0, voltage=4.0),
    )

    with pytest.raises(ValueError, match="valid phase"):
        build_early_cycle_sequence(too_short, config=_config(), data_version="matr-v1")


def test_input_order_does_not_change_values_or_hash() -> None:
    records = _phase_records()
    forward = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )
    reverse = build_early_cycle_sequence(
        tuple(reversed(records)), config=_config(), data_version="matr-v1"
    )

    assert torch.equal(forward.sample_mask, reverse.sample_mask)
    assert torch.allclose(forward.values, reverse.values, equal_nan=True)
    assert forward.input_hash == reverse.input_hash


def test_condition_values_use_only_valid_cutoff_observations() -> None:
    sequence = build_early_cycle_sequence(
        _phase_records(), config=_config(), data_version="matr-v1"
    )

    assert sequence.condition_names == (
        "mean_temperature_c",
        "mean_charge_current_a",
        "mean_discharge_current_a",
    )
    assert sequence.condition_values.tolist() == pytest.approx([25.0, 1.2, -1.0])
    assert sequence.condition_mask.tolist() == [True, True, True]


def test_capacity_reset_selects_qualified_segment_with_largest_span() -> None:
    records = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=1.0, capacity=0.4, voltage=3.4),
        _record(cycle=1, sample=2, current=1.0, capacity=0.8, voltage=3.8),
        _record(cycle=1, sample=3, current=1.0, capacity=0.1, voltage=3.1),
        _record(cycle=1, sample=4, current=1.0, capacity=0.6, voltage=3.6),
        _record(cycle=1, sample=5, current=1.0, capacity=1.1, voltage=4.1),
    )

    sequence = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )

    assert sequence.values[1, 0, (0, -1), 2].tolist() == pytest.approx([0.1, 1.1])
    assert sequence.values[1, 0, (0, -1), 0].tolist() == pytest.approx([3.1, 4.1])


def test_micro_capacity_regression_is_snapped_and_aggregated_stably() -> None:
    records = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=1.0, capacity=0.5, voltage=3.4),
        _record(
            cycle=1,
            sample=2,
            current=2.0,
            capacity=0.5 - 5e-7,
            voltage=3.6,
        ),
        _record(cycle=1, sample=3, current=1.0, capacity=1.0, voltage=4.0),
    )

    forward = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )
    reverse = build_early_cycle_sequence(
        tuple(reversed(records)), config=_config(), data_version="matr-v1"
    )

    midpoint = forward.values[1, 0, 74:76].mean(dim=0)
    assert midpoint.tolist() == pytest.approx([3.5, 1.5, 0.5], abs=2e-3)
    assert forward.input_hash == reverse.input_hash


def test_b3c37_scale_relative_capacity_jitter_remains_one_phase_segment() -> None:
    capacities = (
        0.0,
        0.5,
        1.03,
        1.03 - 2.8e-6,
        1.05,
        1.05 - 2.7e-5,
        1.06,
        1.06 - 1.2e-5,
        1.07,
        1.07 - 5.0e-6,
        1.08,
    )
    records = tuple(
        _record(
            cycle=16,
            sample=index,
            current=1.0,
            capacity=capacity,
            voltage=3.0 + 0.8 * capacity,
            cell_id="MATR_b3c37",
        )
        for index, capacity in enumerate(capacities)
    )

    sequence = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-three-batch-v1"
    )

    assert sequence.sample_mask[16, 0].all()
    assert sequence.values[16, 0, (0, -1), 2].tolist() == pytest.approx(
        [0.0, 1.08]
    )


def test_phase_with_more_than_four_capacity_segments_is_rejected() -> None:
    records = tuple(
        _record(
            cycle=1,
            sample=segment * 3 + point,
            current=1.0,
            capacity=point * 0.5,
            voltage=3.0 + point * 0.2,
        )
        for segment in range(5)
        for point in range(3)
    )

    with pytest.raises(ValueError, match="TOO_MANY_PHASE_SEGMENTS"):
        build_early_cycle_sequence(records, config=_config(), data_version="matr-v1")


def test_duplicate_capacity_aggregation_is_hash_stable_for_any_input_order() -> None:
    records = (
        _record(cycle=1, sample=0, current=1.0, capacity=0.0, voltage=3.0),
        _record(cycle=1, sample=1, current=0.1, capacity=0.5, voltage=3.4),
        _record(cycle=1, sample=2, current=0.2, capacity=0.5, voltage=3.6),
        _record(cycle=1, sample=3, current=0.3, capacity=1.0, voltage=4.0),
    )
    permutations = (
        records,
        tuple(reversed(records)),
        (records[2], records[0], records[3], records[1]),
    )

    sequences = tuple(
        build_early_cycle_sequence(items, config=_config(), data_version="matr-v1")
        for items in permutations
    )

    assert len({sequence.input_hash for sequence in sequences}) == 1
    assert all(
        torch.allclose(sequences[0].values, sequence.values, equal_nan=True)
        for sequence in sequences[1:]
    )


def test_realistic_records_select_capacity_column_by_current_sign() -> None:
    records = tuple(
        CycleRecord(
            dataset_id="MATR",
            cell_id="b1c0",
            cycle_index=1,
            sample_index=index,
            time_s=float(index),
            voltage_v=voltage,
            current_a=current,
            temperature_c=25.0,
            charge_capacity_ah=charge_capacity,
            discharge_capacity_ah=discharge_capacity,
        )
        for index, voltage, current, charge_capacity, discharge_capacity in (
            (0, 3.0, 1.0, 0.0, 9.0),
            (1, 3.5, 1.0, 0.5, 8.0),
            (2, 4.0, 1.0, 1.0, 7.0),
            (3, 9.0, 0.0, 5.0, 5.0),
            (4, 4.1, -1.0, 7.0, 0.0),
            (5, 3.6, -1.0, 8.0, 0.5),
            (6, 3.1, -1.0, 9.0, 1.0),
        )
    )

    sequence = build_early_cycle_sequence(
        records, config=_config(), data_version="matr-v1"
    )

    assert sequence.values[1, 0, (0, -1), 2].tolist() == pytest.approx([0.0, 1.0])
    assert sequence.values[1, 1, (0, -1), 2].tolist() == pytest.approx([0.0, 1.0])
    assert not torch.any(sequence.values[1, :, :, 0] == 9.0)


def test_builder_accepts_only_time_regression_within_configured_tolerance() -> None:
    base = _phase_records()[:3]
    within_tolerance = (
        base[0],
        base[1].model_copy(update={"time_s": 1.0}),
        base[2].model_copy(update={"time_s": 1.0 - 5e-10}),
    )

    sequence = build_early_cycle_sequence(
        within_tolerance, config=_config(), data_version="matr-v1"
    )

    assert sequence.cycle_mask[1]

    outside_tolerance = (
        base[0],
        base[1].model_copy(update={"time_s": 1.0}),
        base[2].model_copy(update={"time_s": 1.0 - 2e-9}),
    )
    with pytest.raises(ValueError, match="NON_MONOTONIC_TIME"):
        build_early_cycle_sequence(
            outside_tolerance, config=_config(), data_version="matr-v1"
        )


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_time_monotonic_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="time_monotonic_tolerance_s"):
        replace(_config(), time_monotonic_tolerance_s=tolerance)


@pytest.mark.parametrize("tolerance", [-1.0, float("nan"), float("inf")])
def test_config_rejects_invalid_relative_capacity_jitter_tolerance(
    tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="relative_capacity_jitter_tolerance"):
        replace(_config(), relative_capacity_jitter_tolerance=tolerance)
