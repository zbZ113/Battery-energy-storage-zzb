import pytest

from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features.early_cycle import (
    EARLY_CYCLE_FEATURE_NAMES,
    EarlyCycleFeatureConfig,
    extract_early_cycle_features,
)


def _records(
    *, include_future: bool = False, include_capacity: bool = True
) -> tuple[CycleRecord, ...]:
    records: list[CycleRecord] = []
    for cycle_index, capacity_scale in ((1, 1.0), (2, 0.9)):
        for sample_index, (time_s, voltage_v, capacity) in enumerate(
            ((0.0, 3.6, 0.0), (10.0, 3.3, 0.2), (20.0, 3.0, 0.4))
        ):
            records.append(
                CycleRecord(
                    dataset_id="MATR",
                    cell_id="MATR_b1c0",
                    cycle_index=cycle_index,
                    sample_index=sample_index,
                    time_s=time_s,
                    voltage_v=voltage_v,
                    current_a=-1.0,
                    temperature_c=25.0 + cycle_index,
                    charge_capacity_ah=capacity,
                    discharge_capacity_ah=capacity * capacity_scale if include_capacity else None,
                    internal_resistance_ohm=0.010 + cycle_index * 0.001,
                )
            )
    if include_future:
        records.append(
            CycleRecord(
                dataset_id="MATR",
                cell_id="MATR_b1c0",
                cycle_index=21,
                sample_index=0,
                time_s=0.0,
                voltage_v=3.6,
                current_a=-1.0,
            )
        )
    return tuple(records)


def test_extracts_traceable_features_from_cutoff_limited_records() -> None:
    result = extract_early_cycle_features(
        _records(),
        config=EarlyCycleFeatureConfig(cutoff_cycle=20),
    )

    assert result.dataset_id == "MATR"
    assert result.cell_id == "MATR_b1c0"
    assert result.source_cycles == (1, 2)
    assert result.values["capacity_first_ah"] == pytest.approx(0.4)
    assert result.values["capacity_last_ah"] == pytest.approx(0.36)
    assert result.values["capacity_relative_change"] == pytest.approx(-0.1)
    assert result.values["temperature_mean_c"] == pytest.approx(26.5)
    assert result.values["internal_resistance_last_ohm"] == pytest.approx(0.012)
    assert result.delta_q_cycles == (1, 2)
    assert result.values["delta_q_variance_ah2"] is not None
    assert set(result.values) == set(EARLY_CYCLE_FEATURE_NAMES)


def test_rejects_feature_input_after_the_configured_cutoff() -> None:
    with pytest.raises(ValueError, match="after cutoff 20"):
        extract_early_cycle_features(
            _records(include_future=True),
            config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        )


def test_feature_values_do_not_depend_on_input_record_order() -> None:
    config = EarlyCycleFeatureConfig(cutoff_cycle=20)

    in_cycle_order = extract_early_cycle_features(_records(), config=config)
    reversed_input = extract_early_cycle_features(tuple(reversed(_records())), config=config)

    assert reversed_input.source_cycles == in_cycle_order.source_cycles
    assert reversed_input.delta_q_cycles == in_cycle_order.delta_q_cycles
    assert reversed_input.values == pytest.approx(in_cycle_order.values)


def test_delta_q_grid_never_steps_past_the_shared_voltage_boundary() -> None:
    records = list(_records())
    records[3] = records[3].model_copy(update={"voltage_v": 3.596})
    records[5] = records[5].model_copy(update={"voltage_v": 3.01})

    result = extract_early_cycle_features(
        tuple(records), config=EarlyCycleFeatureConfig(cutoff_cycle=20)
    )

    assert result.delta_q_cycles == (1, 2)
    assert result.values["delta_q_variance_ah2"] is not None


def test_marks_missing_cycle_duration_as_an_explicit_degradation() -> None:
    records = tuple(
        record for record in _records() if not (record.cycle_index == 1 and record.sample_index)
    )

    result = extract_early_cycle_features(records, config=EarlyCycleFeatureConfig(cutoff_cycle=20))

    assert result.values["cycle_duration_first_s"] is None
    assert "CYCLE_DURATION_UNAVAILABLE" in result.warnings


def test_marks_when_delta_q_falls_back_to_non_diagnostic_cycles() -> None:
    result = extract_early_cycle_features(
        _records(), config=EarlyCycleFeatureConfig(cutoff_cycle=20)
    )

    assert "DELTA_Q_NON_DIAGNOSTIC_FALLBACK" in result.warnings


def test_marks_unavailable_delta_q_without_fabricating_numeric_values() -> None:
    result = extract_early_cycle_features(
        _records(include_capacity=False),
        config=EarlyCycleFeatureConfig(cutoff_cycle=20),
    )

    assert result.delta_q_cycles is None
    assert result.values["delta_q_variance_ah2"] is None
    assert "DELTA_Q_UNAVAILABLE" in result.warnings


def test_rejects_non_monotonic_time_even_when_cycle_records_are_otherwise_valid() -> None:
    records = list(_records())
    records[2] = records[2].model_copy(update={"time_s": 5.0})

    with pytest.raises(ValueError, match="NON_MONOTONIC_TIME"):
        extract_early_cycle_features(
            tuple(records),
            config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        )


def test_excludes_exact_duplicate_placeholder_rows_before_trend_features() -> None:
    placeholder = CycleRecord(
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
    repeated = placeholder.model_copy(update={"sample_index": 1})

    result = extract_early_cycle_features(
        (placeholder, repeated, *_records()),
        config=EarlyCycleFeatureConfig(cutoff_cycle=20),
    )

    assert "EXACT_DUPLICATE_TELEMETRY_EXCLUDED" in result.warnings
    assert "NONPOSITIVE_CAPACITY_EXCLUDED_FROM_TREND" in result.warnings
    assert result.values["capacity_first_ah"] == pytest.approx(0.4)
    assert result.values["capacity_last_ah"] == pytest.approx(0.36)
