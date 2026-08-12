"""Deterministic, support-bounded BLAST-Lite scenario execution."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal, Protocol, cast
from uuid import UUID

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator

from quanxin_life._vendor.blast_lite import (
    Lfp_Gr_250AhPrismatic,
    Lfp_Gr_SonyMurata3Ah_Battery,
)
from quanxin_life.core import EvidenceLevel
from quanxin_life.core.schemas import ContractModel
from quanxin_life.scenarios.contracts import OperationScenario, ScenarioSegment
from quanxin_life.scenarios.routes import BlastExperimentalRange, BlastRouteManifest
from quanxin_life.scenarios.support import (
    DAYS_PER_NATURAL_YEAR,
    ScenarioSupportAssessment,
    ScenarioSupportStatus,
    assess_operation_scenario,
)

MONTHS_PER_YEAR = 12
TIME_RESOLUTION: Literal["monthly-calendar-block-v1"] = "monthly-calendar-block-v1"
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 24.0 * SECONDS_PER_HOUR
_EPSILON = 1e-9
FloatArray = NDArray[np.float64]


class BlastScenarioRejected(ValueError):
    """Raised when a route, state, schedule, or model output is unsupported."""


class ScenarioStateReference(ContractModel):
    """Audited current-state reference; v1 rejects it because BLAST is BOL-only."""

    result_id: str
    json_path: str = Field(min_length=1)

    @field_validator("result_id")
    @classmethod
    def result_id_is_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc
        return value

    @field_validator("json_path")
    @classmethod
    def json_path_targets_tool_values(cls, value: str) -> str:
        if not value.startswith("values.") or any(not part for part in value.split(".")):
            raise ValueError("json_path must start with values and contain nonblank parts")
        return value


class ScenarioEolOutcome(ContractModel):
    status: Literal["REACHED", "NOT_REACHED"]
    natural_year: float | None = Field(default=None, allow_inf_nan=False)
    equivalent_full_cycles: float | None = Field(default=None, allow_inf_nan=False)


class ScenarioProjection(ContractModel):
    scenario_id: str
    scenario_version: str
    route_id: str
    route_version: str
    model_class: str
    evidence_level: Literal[EvidenceLevel.PHYSICS_REFERENCE]
    support: ScenarioSupportAssessment
    experimental_range: BlastExperimentalRange
    time_resolution: Literal["monthly-calendar-block-v1"]
    annual_template_repetitions: dict[str, int]
    natural_years: tuple[float, ...]
    equivalent_full_cycles: tuple[float, ...]
    model_effective_full_cycles: tuple[float, ...]
    soh: tuple[float, ...]
    eol_threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    eol: ScenarioEolOutcome
    milestone_soh: dict[str, float]
    warnings: tuple[str, ...]


class BlastBatteryModel(Protocol):
    outputs: dict[str, FloatArray]
    stressors: dict[str, FloatArray]

    def update_battery_state(
        self,
        t_secs: FloatArray,
        soc: FloatArray,
        temperature_c: FloatArray,
    ) -> None: ...


ModelFactory = Callable[[], BlastBatteryModel]


def _sony_factory() -> BlastBatteryModel:
    return cast(BlastBatteryModel, Lfp_Gr_SonyMurata3Ah_Battery())


def _prismatic_factory() -> BlastBatteryModel:
    return cast(BlastBatteryModel, Lfp_Gr_250AhPrismatic())


_DEFAULT_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "Lfp_Gr_SonyMurata3Ah_Battery": _sony_factory,
    "Lfp_Gr_250AhPrismatic": _prismatic_factory,
}


def _finite_last(mapping: Mapping[str, FloatArray], key: str) -> float:
    values = mapping.get(key)
    if values is None or len(values) == 0:
        raise BlastScenarioRejected("MISSING_BLAST_OUTPUT")
    value = float(values[-1])
    if not math.isfinite(value):
        raise BlastScenarioRejected("NON_FINITE_BLAST_OUTPUT")
    return value


def _active_profile(
    segment: ScenarioSegment,
    *,
    scheduled_efc: float,
    start_time_s: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    times = [start_time_s]
    soc = [segment.soc_lower_bound]
    elapsed_s = start_time_s
    full_cycles = math.floor(scheduled_efc / segment.dod + _EPSILON)
    remainder = scheduled_efc - full_cycles * segment.dod
    if remainder < _EPSILON:
        remainder = 0.0

    for _ in range(full_cycles):
        elapsed_s += segment.dod / segment.charge_c_rate * SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(segment.soc_upper_bound)
        elapsed_s += segment.dod / segment.discharge_c_rate * SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(segment.soc_lower_bound)
    if remainder:
        elapsed_s += remainder / segment.charge_c_rate * SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(segment.soc_lower_bound + remainder)
        elapsed_s += remainder / segment.discharge_c_rate * SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(segment.soc_lower_bound)

    time_array = np.asarray(times, dtype=np.float64)
    soc_array = np.asarray(soc, dtype=np.float64)
    temperature = np.full(time_array.shape, segment.temperature_c, dtype=np.float64)
    return time_array, soc_array, temperature


def _rest_profile(
    *,
    start_time_s: float,
    duration_hours: float,
    soc: float,
    temperature_c: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    time_array = np.asarray(
        [start_time_s, start_time_s + duration_hours * SECONDS_PER_HOUR],
        dtype=np.float64,
    )
    soc_array = np.asarray([soc, soc], dtype=np.float64)
    temperature = np.full(2, temperature_c, dtype=np.float64)
    return time_array, soc_array, temperature


def _segment_for_year(scenario: OperationScenario, year_index: int) -> ScenarioSegment:
    return next(
        segment
        for segment in scenario.segments
        if segment.start_year <= year_index < segment.end_year
    )


def _first_eol_crossing(
    *,
    threshold: float,
    years: list[float],
    efc: list[float],
    soh: list[float],
) -> ScenarioEolOutcome:
    for index in range(1, len(soh)):
        previous = soh[index - 1]
        current = soh[index]
        if previous > threshold >= current:
            denominator = previous - current
            fraction = 1.0 if denominator <= 0 else (previous - threshold) / denominator
            return ScenarioEolOutcome(
                status="REACHED",
                natural_year=years[index - 1]
                + fraction * (years[index] - years[index - 1]),
                equivalent_full_cycles=efc[index - 1]
                + fraction * (efc[index] - efc[index - 1]),
            )
    return ScenarioEolOutcome(status="NOT_REACHED")


class BlastScenarioRunner:
    """Run one reviewed BLAST route from beginning of life only."""

    def __init__(
        self,
        *,
        model_factories: Mapping[str, ModelFactory] | None = None,
    ) -> None:
        self._model_factories = dict(_DEFAULT_MODEL_FACTORIES)
        if model_factories is not None:
            self._model_factories.update(model_factories)

    def run(
        self,
        *,
        route: BlastRouteManifest,
        scenario: OperationScenario,
        initial_state_reference: ScenarioStateReference | None = None,
    ) -> ScenarioProjection:
        if initial_state_reference is not None:
            raise BlastScenarioRejected("UNSUPPORTED_BLAST_STATE_INITIALIZATION")
        support = assess_operation_scenario(route, scenario)
        if support.status is ScenarioSupportStatus.REJECTED:
            raise BlastScenarioRejected("SCENARIO_OUTSIDE_SUPPORTED_RANGE")
        factory = self._model_factories.get(route.model_class)
        if factory is None:
            raise BlastScenarioRejected("BLAST_MODEL_NOT_PACKAGED")
        battery = factory()

        years = [0.0]
        scheduled_efc_axis = [0.0]
        model_efc_axis = [0.0]
        soh_axis = [_finite_last(battery.outputs, "q")]
        month_seconds = DAYS_PER_NATURAL_YEAR * SECONDS_PER_DAY / MONTHS_PER_YEAR
        current_time_s = 0.0
        scheduled_cumulative_efc = 0.0

        for month_index in range(scenario.horizon_years * MONTHS_PER_YEAR):
            year_index = month_index // MONTHS_PER_YEAR
            segment = _segment_for_year(scenario, year_index)
            month_efc = segment.equivalent_full_cycles_per_year / MONTHS_PER_YEAR
            active_time, active_soc, active_temperature = _active_profile(
                segment,
                scheduled_efc=month_efc,
                start_time_s=current_time_s,
            )
            if len(active_time) > 1:
                battery.update_battery_state(
                    active_time,
                    active_soc,
                    active_temperature,
                )
                current_time_s = float(active_time[-1])

            high_soc_rest_hours = (
                month_efc / segment.dod * segment.rest_duration_hours
            )
            if high_soc_rest_hours > _EPSILON:
                rest = _rest_profile(
                    start_time_s=current_time_s,
                    duration_hours=high_soc_rest_hours,
                    soc=segment.soc_upper_bound,
                    temperature_c=segment.temperature_c,
                )
                battery.update_battery_state(*rest)
                current_time_s = float(rest[0][-1])

            month_end_time_s = (month_index + 1) * month_seconds
            low_soc_rest_hours = (month_end_time_s - current_time_s) / SECONDS_PER_HOUR
            if low_soc_rest_hours < -_EPSILON:
                raise BlastScenarioRejected("ANNUAL_SCHEDULE_EXCEEDS_NATURAL_TIME")
            if low_soc_rest_hours > _EPSILON:
                rest = _rest_profile(
                    start_time_s=current_time_s,
                    duration_hours=low_soc_rest_hours,
                    soc=segment.soc_lower_bound,
                    temperature_c=segment.temperature_c,
                )
                battery.update_battery_state(*rest)
            current_time_s = month_end_time_s

            scheduled_cumulative_efc += month_efc
            q = _finite_last(battery.outputs, "q")
            model_efc = _finite_last(battery.stressors, "efc")
            if q < 0.0:
                raise BlastScenarioRejected("NON_PHYSICAL_BLAST_OUTPUT")
            years.append((month_index + 1) / MONTHS_PER_YEAR)
            scheduled_efc_axis.append(scheduled_cumulative_efc)
            model_efc_axis.append(model_efc)
            soh_axis.append(q)

        eol = _first_eol_crossing(
            threshold=scenario.eol_threshold,
            years=years,
            efc=scheduled_efc_axis,
            soh=soh_axis,
        )
        milestones = {
            str(year): soh_axis[year * MONTHS_PER_YEAR]
            for year in (15, 20, 25)
            if year <= scenario.horizon_years
        }
        warnings = tuple(
            dict.fromkeys(
                (
                    *support.warnings,
                    *route.warnings,
                    "MONTHLY_STRESSOR_AGGREGATION",
                    "HIGH_SOC_REST_AND_LOW_SOC_IDLE_ASSUMPTION",
                    "BLAST_COMBINED_C_RATE_STRESSOR",
                    "DETERMINISTIC_SCENARIO_NOT_CONFIDENCE_INTERVAL",
                )
            )
        )
        return ScenarioProjection(
            scenario_id=scenario.scenario_id,
            scenario_version=scenario.scenario_version,
            route_id=route.route_id,
            route_version=route.route_version,
            model_class=route.model_class,
            evidence_level=EvidenceLevel.PHYSICS_REFERENCE,
            support=support,
            experimental_range=route.experimental_range,
            time_resolution=TIME_RESOLUTION,
            annual_template_repetitions={
                segment.segment_id: segment.end_year - segment.start_year
                for segment in scenario.segments
            },
            natural_years=tuple(years),
            equivalent_full_cycles=tuple(scheduled_efc_axis),
            model_effective_full_cycles=tuple(model_efc_axis),
            soh=tuple(soh_axis),
            eol_threshold=scenario.eol_threshold,
            eol=eol,
            milestone_soh=milestones,
            warnings=warnings,
        )


__all__ = [
    "MONTHS_PER_YEAR",
    "TIME_RESOLUTION",
    "BlastScenarioRejected",
    "BlastScenarioRunner",
    "ScenarioEolOutcome",
    "ScenarioProjection",
    "ScenarioStateReference",
]
