from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.testclient import TestClient

from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.api.industrial import create_industrial_sandbox_http_adapter
from quanxin_life.api.service import create_available_tool_invocation_service
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    Decision,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    sha256_canonical,
)
from quanxin_life.integrations.industrial import (
    EmsDecisionSandboxPublisher,
    IndustrialBmsSandbox,
)
from quanxin_life.tools.batch_decision import BATCH_DECISION_TOOL_VERSION
from quanxin_life.tools.registry import StandardToolName

ORIGIN = "https://app.example.test"
NOW = datetime(2026, 7, 18, 8, 0, tzinfo=UTC)


def _auth_adapter() -> AuthHttpAdapter:
    def require_trusted_origin(request: Request) -> None:
        if request.headers.get("origin") != ORIGIN:
            raise HTTPException(status_code=403, detail="origin_not_allowed")

    def require_ready_user() -> None:
        return None

    def require_roles(roles: set[UserRole]) -> object:
        assert roles
        return require_ready_user

    return AuthHttpAdapter(
        router=APIRouter(),
        allowed_origins=(ORIGIN,),
        get_principal=require_ready_user,
        require_trusted_origin=require_trusted_origin,
        require_ready_user=require_ready_user,
        require_roles=require_roles,
    )


def _telemetry(*, message_id: str | None = None) -> dict[str, object]:
    return {
        "message_id": message_id or str(uuid4()),
        "measurement_batch_id": "batch-http-001",
        "dataset_id": "bms-sandbox-v1",
        "cell_id": "cell-http-001",
        "cycle": 50,
        "discharge_capacity_ah": 9.4,
        "reference_capacity_ah": 10.0,
        "measured_at": NOW.isoformat(),
        "reference_capacity_method": "reviewed_metadata_capacity",
        "data_version": "bms-sandbox-data-v1",
        "feature_version": "observed-soh-v1",
        "split_version": "bms-sandbox-split-v1",
        "schema_version": "bms-telemetry-v1",
    }


def _decision_result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.MAKE_BATCH_DECISION.value,
        tool_version=BATCH_DECISION_TOOL_VERSION,
        model_version="model-v1",
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash=sha256_canonical({"input": "ems"}),
        values={
            "cell_id": "cell-http-001",
            "dataset_id": "dataset-http-001",
            "decision": Decision.RECHECK.value,
            "policy_version": "policy-v1",
            "reason_codes": ["INTERVAL_CROSSES_REQUIREMENT"],
        },
        uncertainty={},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="ems-http-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://ems-http/source",
                sha256=sha256_canonical({"source": "ems-http"}),
                description="Reviewed evidence for the EMS HTTP sandbox test",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _client() -> tuple[TestClient, ToolResult]:
    decision = _decision_result()
    auth = _auth_adapter()
    industrial = create_industrial_sandbox_http_adapter(
        IndustrialBmsSandbox(),
        EmsDecisionSandboxPublisher(AuditLedger((decision,))),
        auth_adapter=auth,
    )
    app = create_fastapi_app(
        create_available_tool_invocation_service(),
        auth_adapter=auth,
        industrial_adapter=industrial,
    )
    return TestClient(app, base_url="https://api.example.test"), decision


def test_rest_bms_sandbox_requires_trusted_origin_and_rejects_caller_soh() -> None:
    client, _ = _client()

    blocked = client.post(
        "/v1/integrations/industrial/sandbox/bms/rest",
        json=_telemetry(),
    )
    assert blocked.status_code == 403
    assert blocked.json()["detail"] == "origin_not_allowed"

    accepted = client.post(
        "/v1/integrations/industrial/sandbox/bms/rest",
        headers={"Origin": ORIGIN},
        json=_telemetry(),
    )
    assert accepted.status_code == 201
    assert accepted.json()["transport"] == "REST_SANDBOX"
    assert accepted.json()["replayed"] is False

    untrusted_number = _telemetry()
    untrusted_number["soh"] = 0.99
    rejected = client.post(
        "/v1/integrations/industrial/sandbox/bms/rest",
        headers={"Origin": ORIGIN},
        json=untrusted_number,
    )
    assert rejected.status_code == 422


def test_mqtt_and_modbus_http_sandboxes_preserve_protocol_versions() -> None:
    client, _ = _client()
    telemetry = _telemetry()
    mqtt_payload = json.dumps(telemetry, separators=(",", ":")).encode()

    mqtt = client.post(
        "/v1/integrations/industrial/sandbox/bms/mqtt",
        headers={"Origin": ORIGIN},
        json={
            "topic": "quanxin/v1/bms/batch-http-001",
            "payload_base64": base64.b64encode(mqtt_payload).decode(),
        },
    )
    assert mqtt.status_code == 201
    assert mqtt.json()["transport"] == "MQTT_SANDBOX"

    modbus = client.post(
        "/v1/integrations/industrial/sandbox/bms/modbus",
        headers={"Origin": ORIGIN},
        json={
            "message_id": str(uuid4()),
            "measurement_batch_id": "batch-modbus-http-001",
            "dataset_id": "bms-sandbox-v1",
            "cell_id": "cell-modbus-http-001",
            "registers": [0, 100, 0, 9250, 0, 10000, 26745, 65280],
            "reference_capacity_method": "reviewed_metadata_capacity",
            "data_version": "bms-sandbox-data-v1",
            "feature_version": "observed-soh-v1",
            "split_version": "bms-sandbox-split-v1",
            "register_map_version": "quanxin-modbus-bms-v1",
        },
    )
    assert modbus.status_code == 201
    assert modbus.json()["transport"] == "MODBUS_SANDBOX"


def test_ems_sandbox_reads_only_registered_batch_decision_evidence() -> None:
    client, decision = _client()

    response = client.get(
        f"/v1/integrations/industrial/sandbox/ems/decisions/{decision.result_id}"
    )

    assert response.status_code == 200
    assert response.json()["source_result_id"] == decision.result_id
    assert response.json()["decision"] == "RECHECK"
    assert response.json()["sandbox_only"] is True
    missing = client.get(
        f"/v1/integrations/industrial/sandbox/ems/decisions/{uuid4()}"
    )
    assert missing.status_code == 404
