"""Manifest-bound support assessment for BLAST operating scenarios."""

from __future__ import annotations

from enum import StrEnum

from quanxin_life.core.schemas import ContractModel
from quanxin_life.scenarios.contracts import OperationScenario, ScenarioSegment
from quanxin_life.scenarios.routes import BlastExperimentalRange, BlastRouteManifest

DAYS_PER_NATURAL_YEAR = 365.25
BOUNDARY_WARNING_FRACTION = 0.05


class ScenarioSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NEAR_BOUNDARY = "NEAR_BOUNDARY"
    REJECTED = "REJECTED"


class ScenarioSupportAssessment(ContractModel):
    status: ScenarioSupportStatus
    near_boundary_fields: tuple[str, ...] = ()
    out_of_range_fields: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _bounded_field(
    *,
    path: str,
    value: float,
    bounds: tuple[float, float],
    near: list[str],
    outside: list[str],
) -> None:
    lower, upper = bounds
    if value < lower or value > upper:
        outside.append(path)
        return
    margin = (upper - lower) * BOUNDARY_WARNING_FRACTION
    if value - lower <= margin or upper - value <= margin:
        near.append(path)


def _maximum_field(
    *,
    path: str,
    value: float,
    maximum: float,
    near: list[str],
    outside: list[str],
) -> None:
    if value > maximum:
        outside.append(path)
    elif value >= maximum * (1.0 - BOUNDARY_WARNING_FRACTION):
        near.append(path)


def _assess_segment(
    segment: ScenarioSegment,
    *,
    index: int,
    experimental_range: BlastExperimentalRange,
    near: list[str],
    outside: list[str],
    warnings: list[str],
) -> None:
    prefix = f"segments.{index}"
    _bounded_field(
        path=f"{prefix}.temperature_c",
        value=segment.temperature_c,
        bounds=experimental_range.cycling_temperature_c,
        near=near,
        outside=outside,
    )
    _bounded_field(
        path=f"{prefix}.dod",
        value=segment.dod,
        bounds=experimental_range.dod,
        near=near,
        outside=outside,
    )
    for field_name, value in (
        ("soc_lower_bound", segment.soc_lower_bound),
        ("soc_upper_bound", segment.soc_upper_bound),
    ):
        _bounded_field(
            path=f"{prefix}.{field_name}",
            value=value,
            bounds=experimental_range.soc,
            near=near,
            outside=outside,
        )
    _maximum_field(
        path=f"{prefix}.charge_c_rate",
        value=segment.charge_c_rate,
        maximum=experimental_range.max_rate_charge,
        near=near,
        outside=outside,
    )
    _maximum_field(
        path=f"{prefix}.discharge_c_rate",
        value=segment.discharge_c_rate,
        maximum=experimental_range.max_rate_discharge,
        near=near,
        outside=outside,
    )

    active_hours = segment.equivalent_full_cycles_per_year * (
        1.0 / segment.charge_c_rate + 1.0 / segment.discharge_c_rate
    )
    partial_cycles = segment.equivalent_full_cycles_per_year / segment.dod
    high_soc_rest_hours = partial_cycles * segment.rest_duration_hours
    if active_hours + high_soc_rest_hours > DAYS_PER_NATURAL_YEAR * 24.0:
        outside.extend(
            (
                f"{prefix}.equivalent_full_cycles_per_year",
                f"{prefix}.rest_duration_hours",
            )
        )
        warnings.append("ANNUAL_SCHEDULE_EXCEEDS_NATURAL_TIME")


def assess_operation_scenario(
    route: BlastRouteManifest,
    scenario: OperationScenario,
) -> ScenarioSupportAssessment:
    """Classify every scenario field against one reviewed route manifest."""

    near: list[str] = []
    outside: list[str] = []
    warnings: list[str] = []
    for index, segment in enumerate(scenario.segments):
        _assess_segment(
            segment,
            index=index,
            experimental_range=route.experimental_range,
            near=near,
            outside=outside,
            warnings=warnings,
        )

    if outside:
        warnings.insert(0, "SCENARIO_OUTSIDE_EXPERIMENTAL_RANGE")
        status = ScenarioSupportStatus.REJECTED
    elif near:
        warnings.append("NEAR_EXPERIMENTAL_RANGE_BOUNDARY")
        status = ScenarioSupportStatus.NEAR_BOUNDARY
    else:
        status = ScenarioSupportStatus.SUPPORTED
    return ScenarioSupportAssessment(
        status=status,
        near_boundary_fields=tuple(dict.fromkeys(near)),
        out_of_range_fields=tuple(dict.fromkeys(outside)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "BOUNDARY_WARNING_FRACTION",
    "DAYS_PER_NATURAL_YEAR",
    "ScenarioSupportAssessment",
    "ScenarioSupportStatus",
    "assess_operation_scenario",
]
