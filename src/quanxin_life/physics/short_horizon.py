"""Auditable, short-horizon PyBaMM operating-condition validation.

The implementation intentionally has no lifetime target or long-horizon
forecasting API.  PyBaMM is optional: importing this module must not import or
initialise it, and an unavailable dependency produces an explicit degraded
result rather than synthetic values.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from types import ModuleType
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNCALIBRATED_PARAMETER_WARNING = "参数未针对目标电芯标定，仅供短期趋势与边界核验"  # noqa: RUF001
PHYSICS_VALIDATOR_VERSION = "physics-short-horizon-v1"


class PhysicsValidationStatus(StrEnum):
    """Execution status for a condition-validation request."""

    COMPLETED = "COMPLETED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class _PhysicsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PhysicsValidationRequest(_PhysicsModel):
    """Explicit operating inputs for a limited SPMe experiment.

    The SOC window is translated into short constant-current durations.  It is
    not silently clamped or treated as a calibrated operating envelope.
    """

    initial_soc: float = Field(ge=0.0, le=1.0, strict=True)
    temperature_celsius: float = Field(ge=-30.0, le=70.0, strict=True)
    charge_c_rate: float = Field(gt=0.0, le=5.0, strict=True)
    discharge_c_rate: float = Field(gt=0.0, le=5.0, strict=True)
    lower_soc_bound: float = Field(ge=0.0, lt=1.0, strict=True)
    upper_soc_bound: float = Field(gt=0.0, le=1.0, strict=True)
    repeat_count: int = Field(ge=1, le=3, strict=True)

    @field_validator(
        "initial_soc",
        "temperature_celsius",
        "charge_c_rate",
        "discharge_c_rate",
        "lower_soc_bound",
        "upper_soc_bound",
    )
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("operating-condition values must be finite")
        return value

    @model_validator(mode="after")
    def validate_soc_window(self) -> PhysicsValidationRequest:
        if self.lower_soc_bound >= self.upper_soc_bound:
            raise ValueError("lower_soc_bound must be below upper_soc_bound")
        if not self.lower_soc_bound <= self.initial_soc <= self.upper_soc_bound:
            raise ValueError("initial_soc must lie inside the declared SOC bounds")
        return self


class PhysicsSample(_PhysicsModel):
    """One numerical sample returned directly by the physics solver."""

    time_hours: float = Field(ge=0.0, allow_inf_nan=False)
    voltage_volts: float = Field(allow_inf_nan=False)
    current_amperes: float = Field(allow_inf_nan=False)
    discharge_capacity_ah: float = Field(ge=0.0, allow_inf_nan=False)
    soc: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class PhysicsValidationResult(_PhysicsModel):
    """An auditable short-horizon physics result or explicit degraded outcome."""

    status: PhysicsValidationStatus
    request: PhysicsValidationRequest
    samples: tuple[PhysicsSample, ...] = ()
    termination_reason: str = Field(min_length=1)
    boundary_risks: tuple[str, ...] = ()
    model_name: Literal["SPMe"] = "SPMe"
    model_version: str = "pybamm.lithium_ion.SPMe"
    parameter_set: Literal["Prada2013"] = "Prada2013"
    parameter_set_version: str = "Prada2013"
    pybamm_version: str | None = None
    validator_version: str = PHYSICS_VALIDATOR_VERSION
    warnings: tuple[str, ...] = (UNCALIBRATED_PARAMETER_WARNING,)
    configuration: dict[str, str | tuple[str, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_execution_outcome(self) -> PhysicsValidationResult:
        if UNCALIBRATED_PARAMETER_WARNING not in self.warnings:
            raise ValueError("the uncalibrated-parameter warning is mandatory")
        if self.status is not PhysicsValidationStatus.COMPLETED and self.samples:
            raise ValueError("degraded outcomes must not contain simulated samples")
        if self.status is PhysicsValidationStatus.COMPLETED and not self.samples:
            raise ValueError("completed validation requires solver samples")
        return self


PyBaMMLoader = Callable[[], ModuleType | None]


@dataclass(frozen=True)
class _ExperimentProgram:
    """The exact short-horizon programme and cut-offs handed to PyBaMM."""

    steps: tuple[str, ...]
    lower_voltage_cutoff_volts: float
    upper_voltage_cutoff_volts: float


def validate_short_horizon(
    request: PhysicsValidationRequest,
    *,
    pybamm_loader: PyBaMMLoader | None = None,
) -> PhysicsValidationResult:
    """Validate one short operating programme with SPMe when PyBaMM is present.

    No numerical fallback is used.  Dependency absence and solver failures are
    explicit degraded outcomes carrying the same model and parameter-set audit
    metadata as a completed run.
    """

    loader = pybamm_loader or _load_pybamm
    try:
        pybamm_module = loader()
    except Exception as exc:  # Optional dependency import can fail beyond ModuleNotFoundError.
        return _degraded_result(
            request,
            status=PhysicsValidationStatus.UNAVAILABLE,
            termination_reason=f"pybamm_load_failed:{type(exc).__name__}",
            boundary_risks=("PYBAMM_LOAD_FAILED",),
            program_state="not-generated: pybamm loader failed",
        )
    if pybamm_module is None:
        return _degraded_result(
            request,
            status=PhysicsValidationStatus.UNAVAILABLE,
            termination_reason="pybamm_unavailable",
            boundary_risks=("PYBAMM_UNAVAILABLE",),
            program_state="not-generated: pybamm unavailable",
        )

    program: _ExperimentProgram | None = None
    try:
        program = _build_experiment_program(request, pybamm_module)
        parameter_values = pybamm_module.ParameterValues("Prada2013")
        parameter_values.update(
            {
                "Ambient temperature [K]": request.temperature_celsius + 273.15,
                "Initial temperature [K]": request.temperature_celsius + 273.15,
            },
            check_already_exists=False,
        )
        model = pybamm_module.lithium_ion.SPMe()
        experiment = pybamm_module.Experiment(program.steps)
        simulation = pybamm_module.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=experiment,
        )
        solution = simulation.solve(initial_soc=request.initial_soc)
        samples = _extract_samples(solution)
        termination_reason = _normalise_termination_reason(solution)
    except Exception as exc:  # External scientific dependency failures remain explicit.
        return _degraded_result(
            request,
            status=PhysicsValidationStatus.FAILED,
            termination_reason=f"pybamm_execution_failed:{type(exc).__name__}",
            boundary_risks=("PHYSICS_EXECUTION_FAILED",),
            pybamm_version=_pybamm_version(pybamm_module),
            program=program,
            program_state="not-generated: programme construction failed",
        )

    boundary_risks = _boundary_risks(termination_reason)
    return PhysicsValidationResult(
        status=PhysicsValidationStatus.COMPLETED,
        request=request,
        samples=samples,
        termination_reason=termination_reason,
        boundary_risks=boundary_risks,
        pybamm_version=_pybamm_version(pybamm_module),
        configuration=_configuration(request, program=program),
    )


def _load_pybamm() -> ModuleType | None:
    try:
        module = importlib.import_module("pybamm")
    except ModuleNotFoundError:
        return None
    if not isinstance(module, ModuleType):
        raise TypeError("pybamm import did not return a module")
    return module


def _build_experiment_program(
    request: PhysicsValidationRequest, pybamm_module: ModuleType
) -> _ExperimentProgram:
    parameter_values = pybamm_module.ParameterValues("Prada2013")
    lower_voltage = float(parameter_values["Lower voltage cut-off [V]"])
    upper_voltage = float(parameter_values["Upper voltage cut-off [V]"])

    initial_discharge_span = request.initial_soc - request.lower_soc_bound
    charge_window = _duration_hours(
        request.upper_soc_bound - request.lower_soc_bound, request.charge_c_rate
    )
    discharge_window = _duration_hours(
        request.upper_soc_bound - request.lower_soc_bound, request.discharge_c_rate
    )

    steps: list[str] = []
    if initial_discharge_span > 0.0:
        discharge_to_lower = _duration_hours(initial_discharge_span, request.discharge_c_rate)
        steps.append(
            _discharge_step(
                request.discharge_c_rate,
                discharge_to_lower,
                lower_voltage,
            )
        )
    for _ in range(request.repeat_count):
        steps.append(_charge_step(request.charge_c_rate, charge_window, upper_voltage))
        steps.append(
            _discharge_step(
                request.discharge_c_rate,
                discharge_window,
                lower_voltage,
            )
        )
    return _ExperimentProgram(
        steps=tuple(steps),
        lower_voltage_cutoff_volts=lower_voltage,
        upper_voltage_cutoff_volts=upper_voltage,
    )


def _duration_hours(soc_span: float, c_rate: float) -> float:
    duration = soc_span / c_rate
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("derived experiment duration must be finite and positive")
    return duration


def _charge_step(c_rate: float, duration_hours: float, upper_voltage: float) -> str:
    return f"Charge at {c_rate:.6g}C for {duration_hours:.6g} hours or until {upper_voltage:.6g} V"


def _discharge_step(c_rate: float, duration_hours: float, lower_voltage: float) -> str:
    return (
        f"Discharge at {c_rate:.6g}C for {duration_hours:.6g} hours or until {lower_voltage:.6g} V"
    )


def _extract_samples(solution: Any) -> tuple[PhysicsSample, ...]:
    time_hours = _read_solution_series(solution, ("Time [h]",))
    voltage_volts = _read_solution_series(solution, ("Terminal voltage [V]", "Voltage [V]"))
    current_amperes = _read_solution_series(solution, ("Current [A]",))
    discharge_capacity_ah = _read_solution_series(solution, ("Discharge capacity [A.h]",))
    soc = _read_solution_series(solution, ("SoC", "State of Charge", "State of Charge [-]"))

    sample_count = len(time_hours)
    if sample_count == 0:
        raise ValueError("PyBaMM returned no short-horizon samples")
    if any(
        len(series) != sample_count
        for series in (voltage_volts, current_amperes, discharge_capacity_ah, soc)
    ):
        raise ValueError("PyBaMM output series must have matching lengths")

    return tuple(
        PhysicsSample(
            time_hours=time_hours[index],
            voltage_volts=voltage_volts[index],
            current_amperes=current_amperes[index],
            discharge_capacity_ah=discharge_capacity_ah[index],
            soc=soc[index],
        )
        for index in range(sample_count)
    )


def _read_solution_series(solution: Any, variable_names: Sequence[str]) -> tuple[float, ...]:
    for variable_name in variable_names:
        try:
            variable = solution[variable_name]
        except (KeyError, TypeError):
            continue
        entries = getattr(variable, "entries", None)
        if entries is None:
            entries = getattr(variable, "data", None)
        if entries is None:
            continue
        values = tuple(float(value) for value in _as_iterable(entries))
        if values and all(math.isfinite(value) for value in values):
            return values
    names = ", ".join(variable_names)
    raise KeyError(f"PyBaMM solution did not expose a finite series for: {names}")


def _as_iterable(values: Any) -> Iterable[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError("PyBaMM series must not be text")
    try:
        iterator = cast(Iterator[object], iter(values))
    except TypeError as exc:
        raise TypeError("PyBaMM series must be iterable") from exc
    numeric_values: list[float] = []
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("PyBaMM series entries must be real numbers")
        numeric_values.append(float(value))
    return tuple(numeric_values)


def _normalise_termination_reason(solution: Any) -> str:
    termination = str(getattr(solution, "termination", "completed"))
    return termination.strip() or "completed"


def _boundary_risks(termination_reason: str) -> tuple[str, ...]:
    reason = termination_reason.lower()
    risks: list[str] = []
    if "voltage" in reason:
        risks.append("VOLTAGE_LIMIT_REACHED")
    if "event" in reason or "termination" in reason:
        risks.append("SOLVER_TERMINATION_EVENT")
    return tuple(risks)


def _pybamm_version(pybamm_module: ModuleType) -> str | None:
    version = getattr(pybamm_module, "__version__", None)
    return str(version) if version is not None else None


def _configuration(
    request: PhysicsValidationRequest,
    *,
    program: _ExperimentProgram | None = None,
    program_state: str | None = None,
) -> dict[str, str | tuple[str, ...]]:
    configuration: dict[str, str | tuple[str, ...]] = {
        "temperature_unit": "celsius",
        "programme": "constant-current short-horizon experiment",
        "soc_window_interpretation": "C-rate durations derived from declared SOC bounds",
        "repeat_count": str(request.repeat_count),
    }
    if program is not None:
        configuration.update(
            {
                "experiment_steps": program.steps,
                # These renderings deliberately match the values embedded in
                # the Experiment step strings, so audit consumers can compare
                # the recorded cut-offs without float-format ambiguity.
                "lower_voltage_cutoff_volts": f"{program.lower_voltage_cutoff_volts:.6g}",
                "upper_voltage_cutoff_volts": f"{program.upper_voltage_cutoff_volts:.6g}",
            }
        )
    else:
        configuration.update(
            {
                "experiment_steps": program_state or "not-generated: unavailable",
                "lower_voltage_cutoff_volts": "unavailable",
                "upper_voltage_cutoff_volts": "unavailable",
            }
        )
    return configuration


def _degraded_result(
    request: PhysicsValidationRequest,
    *,
    status: PhysicsValidationStatus,
    termination_reason: str,
    boundary_risks: tuple[str, ...],
    pybamm_version: str | None = None,
    program: _ExperimentProgram | None = None,
    program_state: str | None = None,
) -> PhysicsValidationResult:
    return PhysicsValidationResult(
        status=status,
        request=request,
        termination_reason=termination_reason,
        boundary_risks=boundary_risks,
        pybamm_version=pybamm_version,
        configuration=_configuration(request, program=program, program_state=program_state),
    )
