"""Real optional-dependency smoke test for the released PyBaMM solver."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytest.importorskip("pybamm")

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.physics.short_horizon import PhysicsValidationRequest
from quanxin_life.tools.physics_check import (
    CheckOperatingConditionToolInput,
    execute_check_operating_condition_tool,
)


def test_real_pybamm_spme_prada2013_completes_one_short_cycle() -> None:
    checked_at = datetime(2026, 7, 16, 12, tzinfo=UTC)
    result = execute_check_operating_condition_tool(
        CheckOperatingConditionToolInput(
            request=PhysicsValidationRequest(
                initial_soc=0.5,
                temperature_celsius=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                lower_soc_bound=0.2,
                upper_soc_bound=0.8,
                repeat_count=1,
            ),
            data_version="pybamm-real-smoke-v1",
            feature_version="operating-condition-v1",
            provenance=(
                ProvenanceRecord(
                    source_id="pybamm-real-smoke-input",
                    source_kind=SourceKind.SIMULATED,
                    uri="test://pybamm/real-smoke-input",
                    sha256=sha256_canonical({"fixture": "real-pybamm-smoke-v1"}),
                    description="Explicit operating inputs for a real PyBaMM smoke run",
                    created_at=checked_at,
                ),
            ),
            checked_at=checked_at,
        )
    )

    assert result.tool_name == "check_operating_condition"
    assert result.created_at == checked_at
    assert result.values["status"] == "COMPLETED"
    assert result.values["model_name"] == "SPMe"
    assert result.values["parameter_set"] == "Prada2013"
    assert result.values["pybamm_version"]
    samples = result.values["samples"]
    assert isinstance(samples, list)
    assert len(samples) > 2
    assert min(sample["soc"] for sample in samples) >= 0.0
    assert max(sample["soc"] for sample in samples) <= 1.0
    configuration = result.values["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["soc_source"] in {
        "pybamm_solution_variable",
        "coulomb_counting_from_current_and_nominal_capacity",
    }
    assert result.uncertainty == {"short_horizon_only": True}
    assert "rul" not in result.model_dump_json().lower()
    assert "lifetime" not in result.model_dump_json().lower()
