from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    Decision,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.tools.batch_decision import BATCH_DECISION_TOOL_VERSION
from quanxin_life.tools.registry import StandardToolName

NOW = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)


def _telemetry_payload(*, message_id: str, capacity_ah: float = 9.4) -> bytes:
    return json.dumps(
        {
            "message_id": message_id,
            "measurement_batch_id": "batch-sandbox-001",
            "dataset_id": "bms-sandbox-v1",
            "cell_id": "cell-001",
            "cycle": 50,
            "discharge_capacity_ah": capacity_ah,
            "reference_capacity_ah": 10.0,
            "measured_at": NOW.isoformat(),
            "reference_capacity_method": "reviewed_metadata_capacity",
            "data_version": "bms-sandbox-data-v1",
            "feature_version": "observed-soh-v1",
            "split_version": "bms-sandbox-split-v1",
            "schema_version": "bms-telemetry-v1",
        },
        separators=(",", ":"),
    ).encode()


def test_mqtt_sandbox_ingests_verified_measurement_without_accepting_soh() -> None:
    from quanxin_life.integrations.industrial import IndustrialBmsSandbox
    from quanxin_life.tools.observed_soh_ingestion import (
        NewlyObservedSOHIngestionInput,
        execute_ingest_newly_observed_soh_tool,
    )

    sandbox = IndustrialBmsSandbox()
    message_id = str(uuid4())
    receipt = sandbox.ingest_mqtt(
        "quanxin/v1/bms/batch-sandbox-001",
        _telemetry_payload(message_id=message_id),
    )
    batch = sandbox.resolve_verified_observation_batch("batch-sandbox-001")
    result = execute_ingest_newly_observed_soh_tool(
        NewlyObservedSOHIngestionInput(
            measurement_batch_id="batch-sandbox-001"
        ),
        resolver=sandbox,
        clock=lambda: NOW,
    )

    assert receipt.replayed is False
    assert receipt.transport == "MQTT_SANDBOX"
    assert batch.measurements[0].measurement_id == message_id
    assert batch.provenance[0].source_kind is SourceKind.NEWLY_OBSERVED
    assert result.values["artifact"]["observations"][0]["soh"] == pytest.approx(0.94)
    with pytest.raises(ValueError):
        sandbox.ingest_mqtt(
            "quanxin/v1/bms/batch-sandbox-001",
            _telemetry_payload(message_id=str(uuid4())).replace(
                b'"cycle":50',
                b'"cycle":50,"soh":0.99',
            ),
        )


def test_mqtt_sandbox_replays_identical_message_and_rejects_conflict() -> None:
    from quanxin_life.integrations.industrial import IndustrialBmsSandbox

    sandbox = IndustrialBmsSandbox()
    message_id = str(uuid4())
    payload = _telemetry_payload(message_id=message_id)

    first = sandbox.ingest_mqtt("quanxin/v1/bms/batch-sandbox-001", payload)
    replayed = sandbox.ingest_mqtt("quanxin/v1/bms/batch-sandbox-001", payload)

    assert first.record_hash == replayed.record_hash
    assert replayed.replayed is True
    with pytest.raises(ValueError, match="conflict"):
        sandbox.ingest_mqtt(
            "quanxin/v1/bms/batch-sandbox-001",
            _telemetry_payload(message_id=message_id, capacity_ah=9.3),
        )


def test_modbus_sandbox_uses_versioned_register_scaling() -> None:
    from quanxin_life.integrations.industrial import (
        IndustrialBmsSandbox,
        ModbusBmsSnapshot,
    )

    sandbox = IndustrialBmsSandbox()
    snapshot = ModbusBmsSnapshot(
        message_id=str(uuid4()),
        measurement_batch_id="batch-modbus-001",
        dataset_id="bms-sandbox-v1",
        cell_id="cell-modbus-001",
        registers=(0, 100, 0, 9_250, 0, 10_000, 26_745, 65_280),
        reference_capacity_method="reviewed_metadata_capacity",
        data_version="bms-sandbox-data-v1",
        feature_version="observed-soh-v1",
        split_version="bms-sandbox-split-v1",
        register_map_version="quanxin-modbus-bms-v1",
    )

    receipt = sandbox.ingest_modbus(snapshot)
    batch = sandbox.resolve_verified_observation_batch("batch-modbus-001")

    assert receipt.transport == "MODBUS_SANDBOX"
    assert batch.measurements[0].cycle == 100
    assert batch.measurements[0].discharge_capacity_ah == pytest.approx(9.25)
    assert batch.measurements[0].reference_capacity_ah == pytest.approx(10.0)


def _decision_result() -> ToolResult:
    provenance = ProvenanceRecord(
        source_id="ems-sandbox-source",
        source_kind=SourceKind.OBSERVED,
        uri="test://ems/source",
        sha256=sha256_canonical({"source": "ems"}),
        description="Reviewed source for EMS sandbox contract test",
        created_at=NOW,
    )
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.MAKE_BATCH_DECISION.value,
        tool_version=BATCH_DECISION_TOOL_VERSION,
        model_version="model-v1",
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash=sha256_canonical({"input": "ems"}),
        values={
            "cell_id": "cell-001",
            "dataset_id": "dataset-001",
            "decision": Decision.RECHECK.value,
            "interval_lower_eol_cycle": 300.0,
            "interval_upper_eol_cycle": 400.0,
            "policy_id": "policy-001",
            "policy_version": "policy-v1",
            "policy_source_manifest_hash": sha256_canonical({"policy": "v1"}),
            "quality_report_blocked": False,
            "reason_codes": ["INTERVAL_CROSSES_REQUIREMENT"],
            "required_eol_cycle": 350.0,
            "target_domain_calibrated": True,
            "upstream_result_ids": [str(uuid4()), str(uuid4()), str(uuid4())],
        },
        uncertainty={"coverage_target": 0.9, "lower_eol_cycle": 300.0, "upper_eol_cycle": 400.0},
        warnings=["INTERVAL_CROSSES_REQUIREMENT"],
        provenance=[provenance],
        created_at=NOW,
    )


def test_ems_sandbox_publishes_only_ledger_registered_decision_evidence() -> None:
    from quanxin_life.integrations.industrial import EmsDecisionSandboxPublisher

    decision_result = _decision_result()
    ledger = AuditLedger((decision_result,))
    message = EmsDecisionSandboxPublisher(ledger).build_message(
        decision_result.result_id
    )

    assert message.source_result_id == decision_result.result_id
    assert message.source_result_hash == sha256_canonical(
        decision_result.model_dump(mode="json")
    )
    assert message.decision is Decision.RECHECK
    assert message.reason_codes == ("INTERVAL_CROSSES_REQUIREMENT",)
    assert message.sandbox_only is True

    wrong_tool = decision_result.model_copy(
        update={
            "result_id": str(uuid4()),
            "tool_name": StandardToolName.VALIDATE_BATTERY_DATA.value,
        }
    )
    with pytest.raises(ValueError, match="batch decision"):
        EmsDecisionSandboxPublisher(AuditLedger((wrong_tool,))).build_message(
            wrong_tool.result_id
        )
