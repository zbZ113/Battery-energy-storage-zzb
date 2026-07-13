from __future__ import annotations

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
