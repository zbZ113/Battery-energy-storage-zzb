"""Audited cycle-life to scenario-years conversion tool contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.tools.registry import StandardToolName, ToolExecutionError, ToolRegistry


def _cycle_life_result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version="cycle-life-prediction-tool-v1",
        model_version="xgboost-v1",
        data_version="hust-safe-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"upstream": "features"}),
        values={
            "artifact_type": "quanxin_life.predicted_cycle_life.v1",
            "artifact": {
                "life_prediction": {
                    "dataset_id": "hust-safe-v1",
                    "cell_id": "cell-001",
                    "cutoff_cycle": 100,
                    "target": "eol80_cycle",
                    "predicted_eol_cycle": 3652.5,
                    "observed_eol_cycle": None,
                    "right_censored": True,
                    "feature_version": "early-cycle-v1",
                    "split_version": "split-v1",
                    "model_version": "xgboost-v1",
                    "data_version": "hust-safe-v1",
                }
            },
        },
        provenance=[
            ProvenanceRecord(
                source_id="hust-safe-cell-001",
                source_kind=SourceKind.PREDICTED,
                uri="minio://datasets/hust-safe-v1/cell-001.parquet",
                sha256=sha256_canonical({"cell": "cell-001"}),
                description="Verified safe HUST conversion fixture",
                created_at=datetime(2026, 7, 16, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 16, 8, tzinfo=UTC),
    )


def test_scenario_lifetime_tool_converts_only_verified_cycle_life_result() -> None:
    from quanxin_life.tools.scenario_lifetime import (
        SCENARIO_LIFETIME_ARTIFACT_TYPE,
        ScenarioLifetimeToolInput,
        register_scenario_lifetime_tool,
    )

    upstream = _cycle_life_result()
    registry = ToolRegistry()
    register_scenario_lifetime_tool(
        registry,
        audit_ledger=AuditLedger((upstream,)),
        clock=lambda: datetime(2026, 7, 16, 9, tzinfo=UTC),
    )
    tool_input = ScenarioLifetimeToolInput(
        lifetime_result_id=upstream.result_id,
        operation_policy_version="one-efc-daily-v1",
        equivalent_cycles_per_day=1.0,
    )

    result = registry.execute(StandardToolName.CONVERT_SCENARIO_LIFETIME, tool_input)

    assert result.tool_name == StandardToolName.CONVERT_SCENARIO_LIFETIME.value
    assert result.tool_version == "scenario-lifetime-tool-v1"
    assert result.model_version == upstream.model_version
    assert result.data_version == upstream.data_version
    assert result.feature_version == upstream.feature_version
    assert result.provenance == upstream.provenance
    assert result.created_at == datetime(2026, 7, 16, 9, tzinfo=UTC)
    assert result.values["artifact_type"] == SCENARIO_LIFETIME_ARTIFACT_TYPE
    artifact = result.values["artifact"]
    assert artifact["source_lifetime_result_id"] == upstream.result_id
    assert artifact["scenario_years"] == pytest.approx(10.0)
    assert artifact["evidence_level"] == "MODEL_INFERENCE"
    assert artifact["limitations"]
    assert "SCENARIO_CONVERSION_NOT_OBSERVED_YEARS" in result.warnings


def test_scenario_lifetime_tool_rejects_non_prediction_tool_result() -> None:
    from quanxin_life.tools.scenario_lifetime import (
        ScenarioLifetimeToolInput,
        register_scenario_lifetime_tool,
    )

    upstream = _cycle_life_result().model_copy(
        update={"tool_name": StandardToolName.VALIDATE_BATTERY_DATA.value}
    )
    registry = ToolRegistry()
    register_scenario_lifetime_tool(registry, audit_ledger=AuditLedger((upstream,)))

    with pytest.raises(ToolExecutionError):
        registry.execute(
            StandardToolName.CONVERT_SCENARIO_LIFETIME,
            ScenarioLifetimeToolInput(
                lifetime_result_id=upstream.result_id,
                operation_policy_version="one-efc-daily-v1",
                equivalent_cycles_per_day=1.0,
            ),
        )


@pytest.mark.parametrize(
    "tampered_update",
    (
        {
            "values": {
                "artifact_type": "untrusted-cycle-life",
                "artifact": {},
            }
        },
        {"model_version": "tampered-model-v2"},
    ),
)
def test_scenario_lifetime_tool_rejects_tampered_prediction_evidence(
    tampered_update: dict[str, object],
) -> None:
    from quanxin_life.tools.scenario_lifetime import (
        ScenarioLifetimeToolInput,
        register_scenario_lifetime_tool,
    )

    upstream = _cycle_life_result().model_copy(update=tampered_update)
    registry = ToolRegistry()
    register_scenario_lifetime_tool(registry, audit_ledger=AuditLedger((upstream,)))

    with pytest.raises(ToolExecutionError):
        registry.execute(
            StandardToolName.CONVERT_SCENARIO_LIFETIME,
            ScenarioLifetimeToolInput(
                lifetime_result_id=upstream.result_id,
                operation_policy_version="one-efc-daily-v1",
                equivalent_cycles_per_day=1.0,
            ),
        )


def test_scenario_lifetime_input_rejects_unversioned_or_nonpositive_policy() -> None:
    from pydantic import ValidationError

    from quanxin_life.tools.scenario_lifetime import ScenarioLifetimeToolInput

    upstream = _cycle_life_result()
    with pytest.raises(ValidationError):
        ScenarioLifetimeToolInput(
            lifetime_result_id=upstream.result_id,
            operation_policy_version=" ",
            equivalent_cycles_per_day=0,
        )
