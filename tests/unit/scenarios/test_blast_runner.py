from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from quanxin_life.scenarios import (
    BlastRouteManifest,
    BlastScenarioRejected,
    BlastScenarioRunner,
    OperationScenario,
    ScenarioSegment,
    ScenarioStateReference,
    load_packaged_blast_route_catalog,
)


def _segment(
    *,
    segment_id: str,
    start_year: int,
    end_year: int,
    annual_efc: float = 120.0,
    temperature_c: float = 25.0,
) -> ScenarioSegment:
    return ScenarioSegment(
        segment_id=segment_id,
        start_year=start_year,
        end_year=end_year,
        temperature_c=temperature_c,
        charge_c_rate=0.5,
        discharge_c_rate=0.5,
        soc_lower_bound=0.075,
        soc_upper_bound=0.925,
        dod=0.85,
        equivalent_full_cycles_per_year=annual_efc,
        rest_duration_hours=1.0,
    )


def _scenario(
    *,
    horizon_years: int = 1,
    eol_threshold: float = 0.8,
    segments: tuple[ScenarioSegment, ...] | None = None,
) -> OperationScenario:
    return OperationScenario(
        scenario_id="reference",
        scenario_version="reference-v1",
        horizon_years=horizon_years,
        eol_threshold=eol_threshold,
        segments=segments
        or (
            _segment(
                segment_id="all-years",
                start_year=0,
                end_year=horizon_years,
            ),
        ),
    )


def _route() -> BlastRouteManifest:
    catalog = load_packaged_blast_route_catalog()
    return next(route for route in catalog.routes if route.cell_format == "prismatic")


def test_runner_generates_monthly_natural_time_and_exact_annual_efc() -> None:
    projection = BlastScenarioRunner().run(route=_route(), scenario=_scenario())

    assert len(projection.natural_years) == 13
    assert projection.natural_years[0] == 0.0
    assert projection.natural_years[-1] == pytest.approx(1.0, abs=1e-12)
    assert projection.equivalent_full_cycles[0] == 0.0
    assert projection.equivalent_full_cycles[-1] == pytest.approx(120.0, abs=1e-9)
    assert len(projection.soh) == len(projection.natural_years)
    assert projection.time_resolution == "monthly-calendar-block-v1"
    assert projection.annual_template_repetitions == {"all-years": 1}


def test_runner_does_not_count_soc_residence_as_model_efc() -> None:
    projection = BlastScenarioRunner().run(
        route=_route(),
        scenario=_scenario(
            segments=(
                _segment(
                    segment_id="storage-only",
                    start_year=0,
                    end_year=1,
                    annual_efc=0.0,
                ),
            )
        ),
    )

    assert projection.equivalent_full_cycles[-1] == 0.0
    assert projection.model_effective_full_cycles[-1] == pytest.approx(0.0, abs=1e-12)


def test_runner_continues_one_model_state_across_staged_scenario() -> None:
    scenario = _scenario(
        horizon_years=2,
        segments=(
            _segment(segment_id="year-one", start_year=0, end_year=1, annual_efc=100.0),
            _segment(segment_id="year-two", start_year=1, end_year=2, annual_efc=200.0),
        ),
    )

    projection = BlastScenarioRunner().run(route=_route(), scenario=scenario)

    assert projection.equivalent_full_cycles[12] == pytest.approx(100.0, abs=1e-9)
    assert projection.equivalent_full_cycles[-1] == pytest.approx(300.0, abs=1e-9)
    assert projection.annual_template_repetitions == {"year-one": 1, "year-two": 1}
    assert all(
        later <= earlier
        for earlier, later in zip(
            projection.soh[:-1],
            projection.soh[1:],
            strict=True,
        )
    )


def test_runner_reports_first_eol_crossing_or_not_reached() -> None:
    reached = BlastScenarioRunner().run(
        route=_route(),
        scenario=_scenario(eol_threshold=0.9999),
    )
    not_reached = BlastScenarioRunner().run(
        route=_route(),
        scenario=_scenario(eol_threshold=0.5),
    )

    assert reached.eol.status == "REACHED"
    assert reached.eol_threshold == 0.9999
    assert reached.eol.natural_year is not None
    assert 0.0 < reached.eol.natural_year <= 1.0
    assert reached.eol.equivalent_full_cycles is not None
    assert not_reached.eol.status == "NOT_REACHED"
    assert not_reached.eol_threshold == 0.5
    assert not_reached.eol.natural_year is None
    assert not_reached.eol.equivalent_full_cycles is None


def test_runner_emits_only_milestones_present_in_horizon() -> None:
    projection = BlastScenarioRunner().run(
        route=_route(),
        scenario=_scenario(horizon_years=20),
    )

    assert set(projection.milestone_soh) == {"15", "20"}


def test_runner_rejects_non_bol_state_initialization() -> None:
    reference = ScenarioStateReference(
        result_id=str(uuid4()),
        json_path="values.artifact.current_soh",
    )

    with pytest.raises(
        BlastScenarioRejected,
        match="UNSUPPORTED_BLAST_STATE_INITIALIZATION",
    ):
        BlastScenarioRunner().run(
            route=_route(),
            scenario=_scenario(),
            initial_state_reference=reference,
        )


class _NonFiniteBattery:
    def __init__(self) -> None:
        self.outputs = {"q": np.array([1.0])}
        self.stressors = {"efc": np.array([0.0]), "t_days": np.array([0.0])}

    def update_battery_state(
        self,
        time_s: np.ndarray,
        soc: np.ndarray,
        temperature_c: np.ndarray,
    ) -> None:
        del time_s, soc, temperature_c
        self.outputs["q"] = np.append(self.outputs["q"], np.nan)
        self.stressors["efc"] = np.append(self.stressors["efc"], 1.0)
        self.stressors["t_days"] = np.append(self.stressors["t_days"], 1.0)


def test_runner_rejects_non_finite_model_output() -> None:
    runner = BlastScenarioRunner(
        model_factories={"Lfp_Gr_250AhPrismatic": _NonFiniteBattery}
    )

    with pytest.raises(BlastScenarioRejected, match="NON_FINITE_BLAST_OUTPUT"):
        runner.run(route=_route(), scenario=_scenario())


def test_runner_is_deterministic_for_identical_input() -> None:
    runner = BlastScenarioRunner()
    scenario = _scenario()

    first = runner.run(route=_route(), scenario=scenario)
    second = runner.run(route=_route(), scenario=scenario)

    assert first == second
