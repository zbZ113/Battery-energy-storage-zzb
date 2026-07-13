from __future__ import annotations

from datetime import UTC, datetime

from quanxin_life.core import ProvenanceRecord, SourceKind
from quanxin_life.physics import (
    PhysicsValidationRequest,
    PhysicsValidationResult,
    PhysicsValidationStatus,
)
from quanxin_life.tools import ToolRegistry


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="physics-protocol-fixture",
            source_kind=SourceKind.OBSERVED,
            uri="file:///fixtures/operating-condition.json",
            sha256="d" * 64,
            description="Declared operating condition for short-horizon validation",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _request() -> PhysicsValidationRequest:
    return PhysicsValidationRequest(
        initial_soc=0.55,
        temperature_celsius=25.0,
        charge_c_rate=0.5,
        discharge_c_rate=0.5,
        lower_soc_bound=0.2,
        upper_soc_bound=0.8,
        repeat_count=1,
    )


def _unavailable_result(request: PhysicsValidationRequest) -> PhysicsValidationResult:
    return PhysicsValidationResult(
        status=PhysicsValidationStatus.UNAVAILABLE,
        request=request,
        termination_reason="pybamm_unavailable",
        boundary_risks=("PYBAMM_UNAVAILABLE",),
    )


def test_registered_physics_tool_returns_explicit_degraded_short_horizon_result() -> None:
    from quanxin_life.tools.physics_check import (
        CheckOperatingConditionToolInput,
        register_check_operating_condition_tool,
    )

    registry = ToolRegistry()
    register_check_operating_condition_tool(registry, validator=_unavailable_result)
    tool_input = CheckOperatingConditionToolInput(
        request=_request(),
        data_version="condition-fixture-v1",
        feature_version="operating-condition-v1",
        provenance=_provenance(),
        checked_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    result = registry.execute("check_operating_condition", tool_input)

    assert result.tool_name == "check_operating_condition"
    assert result.tool_version == "physics-check-tool-v1"
    assert result.model_version == "pybamm.lithium_ion.SPMe"
    assert result.values["status"] == "UNAVAILABLE"
    assert result.values["samples"] == []
    assert result.values["termination_reason"] == "pybamm_unavailable"
    assert "PYBAMM_UNAVAILABLE" in result.warnings
    assert "RUL" not in result.values
    assert "eol" not in {key.lower() for key in result.values}


def test_physics_tool_preserves_mandatory_uncalibrated_parameter_warning() -> None:
    from quanxin_life.tools.physics_check import (
        CheckOperatingConditionToolInput,
        execute_check_operating_condition_tool,
    )

    result = execute_check_operating_condition_tool(
        CheckOperatingConditionToolInput(
            request=_request(),
            data_version="condition-fixture-v1",
            feature_version="operating-condition-v1",
            provenance=_provenance(),
            checked_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
        validator=_unavailable_result,
    )

    assert result.values["warnings"]
    assert result.uncertainty == {"short_horizon_only": True}
