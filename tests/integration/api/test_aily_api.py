from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from quanxin_life.api.aily import (
    AilyCompareScenarioContextRequest,
    AilyConnectorConfig,
    AilyCreateAnalysisTaskRequest,
    AilyScenarioContextState,
    create_aily_http_adapter,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    AgentRunState,
    AgentRunStatus,
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
)
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorization
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.reporting import AuditedReportArtifact, ReportArtifactFormat
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)
from quanxin_life.scenarios import OperationScenario, ScenarioSegment

NOW = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)


class _Gateway:
    def __init__(self, state: AgentRunState) -> None:
        self.state = state
        self.create_calls: list[AilyCreateAnalysisTaskRequest] = []

    def create_analysis_task(
        self, request: AilyCreateAnalysisTaskRequest
    ) -> AgentRunState:
        self.create_calls.append(request)
        return self.state

    def get_analysis_task(self, run_id: str) -> AgentRunState:
        if run_id != self.state.run_id:
            raise LookupError("unknown run")
        return self.state


class _ScenarioGateway:
    def __init__(self) -> None:
        self.create_calls: list[AilyCompareScenarioContextRequest] = []
        self.state = AilyScenarioContextState(
            scenario_context_id=str(uuid4()),
            task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
            data_batch_id="batch-safe",
            input_sha256="9" * 64,
            created_at=NOW,
        )

    def create_scenario_context(
        self,
        request: AilyCompareScenarioContextRequest,
    ) -> AilyScenarioContextState:
        self.create_calls.append(request)
        return self.state


class _Exporter:
    def __init__(self, report_result_id: str) -> None:
        self.report_result_id = report_result_id
        self.calls: list[tuple[str, ReportArtifactFormat]] = []

    def export(
        self, result_id: str, format: ReportArtifactFormat
    ) -> AuditedReportArtifact:
        self.calls.append((result_id, format))
        if result_id != self.report_result_id:
            raise ValueError("not an audited report")
        payload = b"# Audited report\n"
        return AuditedReportArtifact(
            source_result_id=result_id,
            format=format,
            filename="audited-report.md",
            media_type="text/markdown; charset=utf-8",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


class _Authorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        inactive = "MODEL_ROUTE_NOT_ACTIVATED" in result.warnings
        return AuditedResultAuthorization(
            allowed=not inactive,
            route_id="test-route",
            activation_status="NOT_ACTIVATED" if inactive else "ACTIVE",
            evidence_level=EvidenceLevel.DATA_DIRECT,
            supported_domain="test-only",
            rejection_reason="MODEL_ROUTE_NOT_ACTIVATED" if inactive else None,
        )


def _result(
    result_id: str, *, warnings: list[str] | None = None
) -> ToolResult:
    return ToolResult(
        result_id=result_id,
        tool_name="validate_battery_data",
        tool_version="data-quality-tool-v1",
        model_version=None,
        data_version="registered-batch-v1",
        feature_version="canonical-csv-v1",
        input_hash="1" * 64,
        values={"blocked": True},
        uncertainty=None,
        warnings=warnings or ["VALIDATION_BLOCKED"],
        provenance=[
            ProvenanceRecord(
                source_id="batch-safe",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:batch-safe",
                sha256="2" * 64,
                description="Registered canonical CSV batch",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _report_result(result_id: str) -> ToolResult:
    return ToolResult(
        result_id=result_id,
        tool_name=AUDITED_REPORT_TOOL_NAME,
        tool_version=AUDITED_REPORT_TOOL_VERSION,
        model_version=REPORTING_VERSION,
        data_version="registered-batch-v1",
        feature_version="canonical-csv-v1",
        input_hash="4" * 64,
        values={"report_kind": "audited_markdown", "markdown": "# Audited report\n"},
        uncertainty=None,
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="batch-safe",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:batch-safe",
                sha256="2" * 64,
                description="Registered canonical CSV batch",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _client(
    *, result_warnings: list[str] | None = None
) -> tuple[TestClient, _Gateway, _ScenarioGateway, ToolResult, str, _Exporter]:
    result = _result(str(uuid4()), warnings=result_warnings)
    report_result_id = str(uuid4())
    report_result = _report_result(report_result_id)
    state = AgentRunState(
        run_id=str(uuid4()),
        intent_id=str(uuid4()),
        plan_hash="3" * 64,
        status=AgentRunStatus.COMPLETED,
        completed_step_ids=("validate", "report"),
        result_ids=(result.result_id, report_result_id),
        updated_at=NOW,
    )
    gateway = _Gateway(state)
    scenario_gateway = _ScenarioGateway()
    exporter = _Exporter(report_result_id)
    adapter = create_aily_http_adapter(
        AilyConnectorConfig(api_key=SecretStr("connector-secret")),
        gateway=gateway,
        scenario_context_gateway=scenario_gateway,
        audit_ledger=AuditLedger((result, report_result)),
        report_exporter=exporter,
        result_authorizer=_Authorizer(),
    )
    app = FastAPI()
    app.include_router(adapter.router)
    return (
        TestClient(app),
        gateway,
        scenario_gateway,
        result,
        report_result_id,
        exporter,
    )


def _headers() -> dict[str, str]:
    return {"Authorization": "Bearer connector-secret"}


def test_aily_requires_the_configured_bearer_token() -> None:
    client, gateway, _, _, _, _ = _client()
    body = {
        "task_type": FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
        "data_batch_id": "batch-safe",
    }

    missing = client.post("/v1/aily/analysis-tasks", json=body)
    wrong = client.post(
        "/v1/aily/analysis-tasks",
        json=body,
        headers={"Authorization": "Bearer wrong"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert gateway.create_calls == []


def test_aily_creates_and_reads_an_existing_agent_run_state() -> None:
    client, gateway, _, _, _, _ = _client()

    created = client.post(
        "/v1/aily/analysis-tasks",
        json={
            "task_type": FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY.value,
            "data_batch_id": "batch-safe",
        },
        headers=_headers(),
    )
    fetched = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}", headers=_headers()
    )

    assert created.status_code == 202
    assert created.json() == gateway.state.model_dump(mode="json")
    assert fetched.json() == gateway.state.model_dump(mode="json")
    assert gateway.create_calls[0].data_batch_id == "batch-safe"


def test_aily_returns_only_a_run_bound_ledger_tool_result() -> None:
    client, gateway, _, result, _, _ = _client()

    response = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}/results/{result.result_id}",
        headers=_headers(),
    )
    unbound = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}/results/{uuid4()}",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json() == result.model_dump(mode="json")
    assert unbound.status_code == 404


def test_aily_does_not_return_values_for_an_inactive_model_result() -> None:
    client, gateway, _, result, _, _ = _client(
        result_warnings=["MODEL_ROUTE_NOT_ACTIVATED"]
    )

    response = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}/results/{result.result_id}",
        headers=_headers(),
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "aily_audited_result_rejected",
        "reason": "MODEL_ROUTE_NOT_ACTIVATED",
    }
    assert "values" not in response.json()


def test_aily_downloads_only_a_run_bound_audited_report_artifact() -> None:
    client, gateway, _, _, report_result_id, _ = _client()

    response = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}"
        f"/reports/{report_result_id}",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.content == b"# Audited report\n"
    assert response.headers["etag"] == f'"sha256:{sha256(response.content).hexdigest()}"'
    assert response.headers["x-tool-result-id"] == report_result_id


def test_aily_rejects_a_run_bound_non_report_result_before_export() -> None:
    client, gateway, _, result, _, exporter = _client()

    response = client.get(
        f"/v1/aily/analysis-tasks/{gateway.state.run_id}"
        f"/reports/{result.result_id}",
        headers=_headers(),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "aily_audited_report_not_found"
    assert exporter.calls == []


def _scenario(*, scenario_id: str) -> dict[str, object]:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=20,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="years-1-20",
                start_year=0,
                end_year=20,
                temperature_c=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=300.0,
                rest_duration_hours=1.0,
            ),
        ),
    ).model_dump(mode="json")


def test_aily_creates_reference_only_scenario_contexts_from_public_contracts() -> None:
    client, _, scenario_gateway, _, _, _ = _client()

    response = client.post(
        "/v1/aily/scenario-contexts",
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "data_batch_id": "batch-safe",
            "cell_format": "prismatic",
            "baseline": _scenario(scenario_id="baseline"),
            "comparisons": [_scenario(scenario_id="warmer")],
        },
        headers=_headers(),
    )

    assert response.status_code == 201
    assert response.json() == scenario_gateway.state.model_dump(mode="json")
    assert "route_id" not in response.json()
    assert "model_class" not in response.json()
    request = scenario_gateway.create_calls[0]
    assert request.baseline.scenario_id == "baseline"
    assert request.comparisons[0].segments[0].temperature_c == 25.0


def test_aily_rejects_caller_generated_scenario_outputs_before_the_gateway() -> None:
    client, _, scenario_gateway, _, _, _ = _client()

    response = client.post(
        "/v1/aily/scenario-contexts",
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "data_batch_id": "batch-safe",
            "cell_format": "prismatic",
            "baseline": _scenario(scenario_id="baseline"),
            "comparisons": [_scenario(scenario_id="warmer")],
            "final_soh": 0.9,
        },
        headers=_headers(),
    )

    assert response.status_code == 422
    assert scenario_gateway.create_calls == []


def test_aily_scenario_tasks_require_a_persisted_context_reference() -> None:
    client, gateway, scenario_gateway, _, _, _ = _client()

    accepted = client.post(
        "/v1/aily/analysis-tasks",
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "scenario_context_id": scenario_gateway.state.scenario_context_id,
        },
        headers=_headers(),
    )
    rejected = client.post(
        "/v1/aily/analysis-tasks",
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "data_batch_id": "batch-safe",
        },
        headers=_headers(),
    )

    assert accepted.status_code == 202
    assert rejected.status_code == 422
    assert gateway.create_calls[-1].scenario_context_id == (
        scenario_gateway.state.scenario_context_id
    )
