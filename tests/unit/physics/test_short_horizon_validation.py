from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from quanxin_life.physics.short_horizon import (
    UNCALIBRATED_PARAMETER_WARNING,
    PhysicsValidationRequest,
    PhysicsValidationStatus,
    validate_short_horizon,
)


def _request(**overrides: object) -> PhysicsValidationRequest:
    values: dict[str, object] = {
        "initial_soc": 0.55,
        "temperature_celsius": 25.0,
        "charge_c_rate": 0.5,
        "discharge_c_rate": 0.5,
        "lower_soc_bound": 0.2,
        "upper_soc_bound": 0.8,
        "repeat_count": 2,
    }
    values.update(overrides)
    return PhysicsValidationRequest(**values)


def test_returns_explicit_unavailable_result_without_pybamm() -> None:
    result = validate_short_horizon(_request(), pybamm_loader=lambda: None)

    assert result.status is PhysicsValidationStatus.UNAVAILABLE
    assert result.samples == ()
    assert result.model_name == "SPMe"
    assert result.parameter_set == "Prada2013"
    assert result.pybamm_version is None
    assert UNCALIBRATED_PARAMETER_WARNING in result.warnings
    assert "unavailable" in result.termination_reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"initial_soc": float("nan")},
        {"initial_soc": 1.1},
        {"lower_soc_bound": 0.8, "upper_soc_bound": 0.2},
        {"initial_soc": 0.1},
        {"charge_c_rate": 0.0},
        {"temperature_celsius": -80.0},
        {"repeat_count": 0},
        {"charge_c_rate": "0.5"},
    ],
)
def test_rejects_invalid_or_out_of_bound_operating_conditions(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _request(**overrides)


def test_result_contract_has_no_long_horizon_prediction_fields() -> None:
    field_names = {
        field_name.lower() for field_name in validate_short_horizon(
            _request(), pybamm_loader=lambda: None
        ).__class__.model_fields
    }

    assert not {"rul", "eol", "cycle_life", "lifetime", "long_term_degradation"} & field_names


class _FakeParameterValues:
    def __init__(self, _name: str) -> None:
        self.updated_values: dict[str, float] = {}

    def __getitem__(self, key: str) -> float:
        values = {
            "Lower voltage cut-off [V]": 2.5,
            "Upper voltage cut-off [V]": 4.2,
            "Nominal cell capacity [A.h]": 2.3,
        }
        return values[key]

    def update(self, values: dict[str, float], *, check_already_exists: bool) -> None:
        assert check_already_exists is False
        self.updated_values.update(values)


class _FakeExperiment:
    def __init__(self, steps: tuple[str, ...]) -> None:
        self.steps = steps


class _FakeSolution:
    termination = "final time"

    def __getitem__(self, key: str) -> Any:
        values = {
            "Time [h]": (0.0, 0.25),
            "Terminal voltage [V]": (3.2, 3.4),
            "Current [A]": (0.5, -0.5),
            "Discharge capacity [A.h]": (0.0, 0.1),
            "SoC": (0.2, 0.4),
        }
        return SimpleNamespace(entries=values[key])


class _FakeSolutionWithoutSoc(_FakeSolution):
    def __getitem__(self, key: str) -> Any:
        if key in {"SoC", "State of Charge", "State of Charge [-]"}:
            raise KeyError(key)
        return super().__getitem__(key)


class _FakeSimulation:
    def __init__(self, *args: object, fail_solve: bool = False, **kwargs: object) -> None:
        self.fail_solve = fail_solve
        self.args = args
        self.kwargs = kwargs

    def solve(self, *, initial_soc: float) -> _FakeSolution:
        if self.fail_solve:
            raise RuntimeError("fake solver failure")
        assert 0.0 <= initial_soc <= 1.0
        return _FakeSolution()


class _FakePyBaMM:
    __version__ = "fake-1.0"

    def __init__(self, *, fail_solve: bool = False) -> None:
        self.last_experiment: _FakeExperiment | None = None
        self.fail_solve = fail_solve
        self.lithium_ion = SimpleNamespace(SPMe=lambda: object())

    ParameterValues = _FakeParameterValues

    def Experiment(self, steps: tuple[str, ...]) -> _FakeExperiment:
        self.last_experiment = _FakeExperiment(steps)
        return self.last_experiment

    def Simulation(self, *args: object, **kwargs: object) -> _FakeSimulation:
        return _FakeSimulation(*args, fail_solve=self.fail_solve, **kwargs)


def test_loader_exception_returns_degraded_result_without_samples() -> None:
    def raise_loader_error() -> None:
        raise RuntimeError("missing native dependency")

    result = validate_short_horizon(_request(), pybamm_loader=raise_loader_error)

    assert result.status is PhysicsValidationStatus.UNAVAILABLE
    assert result.samples == ()
    assert result.termination_reason == "pybamm_load_failed:RuntimeError"
    assert result.configuration["experiment_steps"] == "not-generated: pybamm loader failed"
    assert result.configuration["lower_voltage_cutoff_volts"] == "unavailable"
    assert result.configuration["upper_voltage_cutoff_volts"] == "unavailable"


def test_initial_soc_at_lower_bound_skips_initial_discharge_step() -> None:
    fake_pybamm = _FakePyBaMM()

    result = validate_short_horizon(
        _request(initial_soc=0.2, lower_soc_bound=0.2, repeat_count=1),
        pybamm_loader=lambda: fake_pybamm,
    )

    assert result.status is PhysicsValidationStatus.COMPLETED
    assert fake_pybamm.last_experiment is not None
    assert fake_pybamm.last_experiment.steps == result.configuration["experiment_steps"]
    assert len(fake_pybamm.last_experiment.steps) == 2
    assert fake_pybamm.last_experiment.steps[0].startswith("Charge")
    assert all("initial" not in step.lower() for step in fake_pybamm.last_experiment.steps)


def test_solver_failure_returns_empty_samples_and_preserves_program_audit() -> None:
    fake_pybamm = _FakePyBaMM(fail_solve=True)

    result = validate_short_horizon(_request(), pybamm_loader=lambda: fake_pybamm)

    assert result.status is PhysicsValidationStatus.FAILED
    assert result.samples == ()
    assert result.termination_reason == "pybamm_execution_failed:RuntimeError"
    assert fake_pybamm.last_experiment is not None
    assert result.configuration["experiment_steps"] == fake_pybamm.last_experiment.steps
    assert result.configuration["lower_voltage_cutoff_volts"] == "2.5"
    assert result.configuration["upper_voltage_cutoff_volts"] == "4.2"


def test_completed_result_records_exact_programme_and_voltage_cutoffs() -> None:
    fake_pybamm = _FakePyBaMM()

    result = validate_short_horizon(_request(), pybamm_loader=lambda: fake_pybamm)

    assert result.status is PhysicsValidationStatus.COMPLETED
    assert fake_pybamm.last_experiment is not None
    assert result.configuration["experiment_steps"] == fake_pybamm.last_experiment.steps
    assert result.configuration["lower_voltage_cutoff_volts"] == "2.5"
    assert result.configuration["upper_voltage_cutoff_volts"] == "4.2"


def test_missing_solver_soc_uses_audited_coulomb_counting_fallback() -> None:
    class _SimulationWithoutSoc(_FakeSimulation):
        def solve(self, *, initial_soc: float) -> _FakeSolutionWithoutSoc:
            assert 0.0 <= initial_soc <= 1.0
            return _FakeSolutionWithoutSoc()

    class _PyBaMMWithoutSoc(_FakePyBaMM):
        def Simulation(self, *args: object, **kwargs: object) -> _SimulationWithoutSoc:
            return _SimulationWithoutSoc(*args, **kwargs)

    result = validate_short_horizon(
        _request(initial_soc=0.5, repeat_count=1),
        pybamm_loader=lambda: _PyBaMMWithoutSoc(),
    )

    assert result.status is PhysicsValidationStatus.COMPLETED
    assert result.configuration["soc_source"] == (
        "coulomb_counting_from_current_and_nominal_capacity"
    )
    assert result.configuration["nominal_capacity_ah"] == "2.3"
    assert result.samples[0].soc == pytest.approx(0.5)
    assert result.samples[-1].soc == pytest.approx(0.5)
