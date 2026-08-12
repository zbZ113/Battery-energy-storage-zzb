from __future__ import annotations

import pytest

from quanxin_life.scenarios import (
    BlastRouteManifest,
    OperationScenario,
    ScenarioSegment,
    ScenarioSupportStatus,
    assess_operation_scenario,
    load_packaged_blast_route_catalog,
)


def _scenario(**overrides: float) -> OperationScenario:
    values = {
        "temperature_c": 25.0,
        "charge_c_rate": 0.5,
        "discharge_c_rate": 0.5,
        "soc_lower_bound": 0.075,
        "soc_upper_bound": 0.925,
        "dod": 0.85,
        "equivalent_full_cycles_per_year": 300.0,
        "rest_duration_hours": 1.0,
    }
    values.update(overrides)
    return OperationScenario(
        scenario_id="baseline",
        scenario_version="scenario-v1",
        horizon_years=25,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="years-1-25",
                start_year=0,
                end_year=25,
                **values,
            ),
        ),
    )


def _prismatic_route() -> BlastRouteManifest:
    catalog = load_packaged_blast_route_catalog()
    return next(route for route in catalog.routes if route.cell_format == "prismatic")


def test_support_assessment_accepts_manifest_bounded_scenario() -> None:
    assessment = assess_operation_scenario(_prismatic_route(), _scenario())

    assert assessment.status is ScenarioSupportStatus.SUPPORTED
    assert assessment.out_of_range_fields == ()
    assert assessment.warnings == ()


def test_support_assessment_warns_near_temperature_boundary() -> None:
    assessment = assess_operation_scenario(
        _prismatic_route(),
        _scenario(temperature_c=11.0),
    )

    assert assessment.status is ScenarioSupportStatus.NEAR_BOUNDARY
    assert assessment.out_of_range_fields == ()
    assert assessment.warnings == ("NEAR_EXPERIMENTAL_RANGE_BOUNDARY",)
    assert assessment.near_boundary_fields == ("segments.0.temperature_c",)


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"temperature_c": 46.0}, "segments.0.temperature_c"),
        ({"charge_c_rate": 0.7}, "segments.0.charge_c_rate"),
        ({"discharge_c_rate": 1.1}, "segments.0.discharge_c_rate"),
        ({"dod": 0.7, "soc_upper_bound": 0.775}, "segments.0.dod"),
    ],
)
def test_support_assessment_rejects_manifest_ood_fields(
    overrides: dict[str, float],
    expected_field: str,
) -> None:
    assessment = assess_operation_scenario(_prismatic_route(), _scenario(**overrides))

    assert assessment.status is ScenarioSupportStatus.REJECTED
    assert expected_field in assessment.out_of_range_fields
    assert "SCENARIO_OUTSIDE_EXPERIMENTAL_RANGE" in assessment.warnings


def test_support_assessment_rejects_infeasible_annual_schedule() -> None:
    assessment = assess_operation_scenario(
        _prismatic_route(),
        _scenario(
            charge_c_rate=0.1,
            discharge_c_rate=0.1,
            equivalent_full_cycles_per_year=400.0,
            rest_duration_hours=5.0,
        ),
    )

    assert assessment.status is ScenarioSupportStatus.REJECTED
    assert assessment.out_of_range_fields == (
        "segments.0.equivalent_full_cycles_per_year",
        "segments.0.rest_duration_hours",
    )
    assert "ANNUAL_SCHEDULE_EXCEEDS_NATURAL_TIME" in assessment.warnings
