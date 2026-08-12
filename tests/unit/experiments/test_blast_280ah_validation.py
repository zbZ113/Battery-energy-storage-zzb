from __future__ import annotations

import pytest

from quanxin_life.data.adapters.lfp_280ah_dod import (
    DodCapacityCellSeries,
    DodCapacityObservation,
)
from quanxin_life.experiments.blast_280ah_validation import (
    evaluate_lfp_280ah_capacity_series,
)


def _series(*, elapsed_hours_per_cycle: float = 5.0) -> DodCapacityCellSeries:
    observations = tuple(
        DodCapacityObservation(
            cycle_number=index,
            elapsed_days=(index - 1) * elapsed_hours_per_cycle / 24.0,
            discharge_capacity_ah=capacity,
            nominal_soh=capacity / 280.0,
            relative_capacity_ratio=capacity / 282.0,
            mean_temperature_c=25.0,
            sample_count=10,
        )
        for index, capacity in ((1, 282.0), (2, 281.0), (3, 280.0))
    )
    return DodCapacityCellSeries(
        dataset_id="LFP_280AH_DOD",
        vendor="CATL",
        cell_id="CATL-2763",
        nominal_capacity_ah=280.0,
        dod_fraction=1.0,
        c_rate=0.5,
        reference_capacity_ah=282.0,
        source_outer_file="data/raw/LFP_280AH_DOD/v3/CATL.zip",
        source_member="CATL/CATL-0.5C-100% Depth of Discharge/2763.zip",
        outer_archive_sha256="a" * 64,
        nested_archive_sha256="b" * 64,
        csv_members=("cell-0.5C-100%DOD-1.csv",),
        observations=observations,
    )


def test_280ah_reference_check_is_deterministic_and_observed_range_only() -> None:
    first = evaluate_lfp_280ah_capacity_series((_series(),), bundle_sha256="c" * 64)
    second = evaluate_lfp_280ah_capacity_series((_series(),), bundle_sha256="c" * 64)

    assert first == second
    assert len(first.predictions) == 3
    assert first.predictions[0].predicted_soh == 1.0
    assert first.predictions[-1].scheduled_efc == 2.0
    assert first.predictions[-1].elapsed_days == 10.0 / 24.0
    assert all(point.support_status == "SUPPORTED_BY_ROUTE_MANIFEST" for point in first.predictions)
    assert {metric.group_kind for metric in first.metrics} >= {"overall", "cell_id"}
    assert first.methodology["parameter_fitting_performed"] is False
    assert first.methodology["comparison_scope"] == "OBSERVED_CYCLE_RANGE_ONLY"
    assert "REFERENCE_250AH_MODEL_NOT_280AH_PRODUCT_MODEL" in first.warnings
    assert "NOT_15_TO_25_YEAR_VALIDATION" in first.warnings


def test_280ah_reference_check_uses_source_time_for_effective_rate() -> None:
    evaluation = evaluate_lfp_280ah_capacity_series(
        (_series(elapsed_hours_per_cycle=3.88),),
        bundle_sha256="c" * 64,
    )

    point = evaluation.predictions[-1]
    assert point.elapsed_days == 2 * 3.88 / 24
    assert point.charge_c_rate == point.discharge_c_rate
    assert point.charge_c_rate == pytest.approx(2 / 3.88)
    assert point.support_status == "SUPPORTED_BY_ROUTE_MANIFEST"
    assert evaluation.methodology["source_protocol_c_rate"] == 0.5
