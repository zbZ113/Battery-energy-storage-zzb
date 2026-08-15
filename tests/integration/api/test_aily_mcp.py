from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any
from uuid import uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

pytest.importorskip("mcp")

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

import quanxin_life.api.aily_mcp as aily_mcp_module
from quanxin_life.api.aily import (
    AilyCreateAnalysisTaskRequest,
    AilyCreateScenarioContextRequest,
    AilyScenarioContextState,
)
from quanxin_life.api.aily_mcp import (
    AilyMcpConfig,
    create_aily_mcp_adapter,
)
from quanxin_life.core import (
    AgentRunState,
    AgentRunStatus,
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
)
from quanxin_life.integrations.feishu.cards import (
    AuditedResultAuthorization,
)
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
)
from quanxin_life.reporting import (
    AuditedReportArtifact,
    ReportArtifactFormat,
)

NOW = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)
ALLOWED_IP = "101.126.59.88"
ENDPOINT_TOKEN = "mcp_endpoint_token_0123456789abcdef"
BASE_TOOL_NAMES = {
    "quanxin_create_analysis_task",
    "quanxin_create_scenario_context",
    "quanxin_get_analysis_task",
    "quanxin_get_audited_report",
    "quanxin_get_audited_result",
}


class _Gateway:
    def __init__(self) -> None:
        self.create_calls: list[AilyCreateAnalysisTaskRequest] = []
        self.state = AgentRunState(
            run_id=str(uuid4()),
            intent_id=str(uuid4()),
            plan_hash="1" * 64,
            status=AgentRunStatus.PLANNING,
            updated_at=NOW,
        )

    def create_analysis_task(
        self,
        request: AilyCreateAnalysisTaskRequest,
    ) -> AgentRunState:
        self.create_calls.append(request)
        return self.state

    def get_analysis_task(self, run_id: str) -> AgentRunState:
        if run_id != self.state.run_id:
            raise LookupError("unknown run")
        return self.state


class _ScenarioGateway:
    def create_scenario_context(
        self,
        request: AilyCreateScenarioContextRequest,
    ) -> AilyScenarioContextState:
        return AilyScenarioContextState(
            scenario_context_id=str(uuid4()),
            task_type=request.task_type,
            data_batch_id=request.data_batch_id,
            input_sha256="2" * 64,
            created_at=NOW,
        )


class _Ledger:
    def resolve_registered_result(self, result_id: str) -> ToolResult:
        raise ValueError(f"unknown result: {result_id}")


def _result(result_id: str) -> ToolResult:
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
        warnings=["VALIDATION_BLOCKED"],
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


class _Exporter:
    def export(
        self,
        result_id: str,
        format: ReportArtifactFormat,
    ) -> AuditedReportArtifact:
        payload = b"# Audited report\n"
        return AuditedReportArtifact(
            source_result_id=result_id,
            format=format,
            filename="audited-report.md",
            media_type="text/markdown; charset=utf-8",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


class _ResultAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        return AuditedResultAuthorization(
            allowed=True,
            route_id="reviewed-route",
            activation_status="ACTIVE",
            evidence_level=EvidenceLevel.DATA_DIRECT,
            supported_domain="test-only",
        )


class _CallerAuthorizer:
    def __init__(self, *, deny_identity: bool = False) -> None:
        self.deny_identity = deny_identity
        self.identity_calls: list[str] = []
        self.calls: list[tuple[str, str]] = []

    def authorize_identity(self, *, aily_user_id: str) -> None:
        self.identity_calls.append(aily_user_id)
        if self.deny_identity:
            raise ValueError("unknown Aily identity")

    def authorize_run_reference(
        self,
        *,
        aily_user_id: str,
        run_id: str,
    ) -> None:
        self.calls.append((aily_user_id, run_id))


def _adapter(
    *,
    gateway: _Gateway | None = None,
    caller_authorizer: _CallerAuthorizer | None = None,
    allowed_hosts: tuple[str, ...] = ("testserver",),
) -> tuple[Any, _Gateway, _CallerAuthorizer]:
    resolved_gateway = gateway or _Gateway()
    resolved_caller_authorizer = caller_authorizer or _CallerAuthorizer()
    adapter = create_aily_mcp_adapter(
        AilyMcpConfig(
            endpoint_token=SecretStr(ENDPOINT_TOKEN),
            allowed_source_ips=(ALLOWED_IP,),
            allowed_hosts=allowed_hosts,
        ),
        gateway=resolved_gateway,
        scenario_context_gateway=_ScenarioGateway(),
        audit_ledger=_Ledger(),
        report_exporter=_Exporter(),
        result_authorizer=_ResultAuthorizer(),
        caller_authorizer=resolved_caller_authorizer,
    )
    return adapter, resolved_gateway, resolved_caller_authorizer


def _initialize_request() -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }


def test_aily_mcp_registers_only_the_reference_based_business_tools() -> None:
    adapter, _, _ = _adapter()

    tools = asyncio.run(adapter.server.list_tools())

    assert {tool.name for tool in tools} == BASE_TOOL_NAMES
    assert adapter.mount_path == f"/v1/aily/mcp/{ENDPOINT_TOKEN}"
    assert all("ToolRegistry" not in (tool.description or "") for tool in tools)
    assert ENDPOINT_TOKEN not in repr(adapter)


def test_aily_mcp_rejects_a_gateway_that_substitutes_another_run_identity() -> None:
    requested_run_id = str(uuid4())
    substituted = AgentRunState(
        run_id=str(uuid4()),
        intent_id=str(uuid4()),
        plan_hash="1" * 64,
        status=AgentRunStatus.PLANNING,
        updated_at=NOW,
    )

    class _SubstitutingGateway:
        def get_analysis_task(self, run_id: str) -> AgentRunState:
            assert run_id == requested_run_id
            return substituted

    with pytest.raises(ValueError, match="identity"):
        aily_mcp_module._resolve_run(_SubstitutingGateway(), requested_run_id)


def test_aily_mcp_rejects_a_ledger_that_substitutes_another_result_identity() -> None:
    requested_result_id = str(uuid4())
    substituted = _result(str(uuid4()))
    state = AgentRunState(
        run_id=str(uuid4()),
        intent_id=str(uuid4()),
        plan_hash="1" * 64,
        status=AgentRunStatus.COMPLETED,
        result_ids=(requested_result_id,),
        updated_at=NOW,
    )

    class _SubstitutingLedger:
        def resolve_registered_result(self, result_id: str) -> ToolResult:
            assert result_id == requested_result_id
            return substituted

    with pytest.raises(ValueError, match="identity"):
        aily_mcp_module._resolve_bound_result(
            state=state,
            result_id=requested_result_id,
            audit_ledger=_SubstitutingLedger(),
        )


def test_aily_mcp_bounds_the_final_call_tool_result_without_text_duplication() -> None:
    builder = getattr(aily_mcp_module, "_bounded_mcp_result", None)
    assert callable(builder)

    result = builder({"内容": "值" * 18_000}, 20_000)
    assert isinstance(result, CallToolResult)
    assert result.content == []
    rendered = json.dumps(
        result.model_dump(mode="json", by_alias=True, exclude_none=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(rendered) <= 20_000

    with pytest.raises(ValueError, match="上下文上限"):
        builder({"内容": "值" * 20_000}, 20_000)


@pytest.mark.parametrize(
    "headers",
    (
        {},
        {
            "x-quanxin-aily-source-ip": ALLOWED_IP,
            "x-aily-email": "engineer@example.com",
        },
        {
            "x-quanxin-aily-source-ip": "203.0.113.20",
            "x-aily-user": "aily-user-1",
            "x-aily-email": "engineer@example.com",
            "x-forwarded-for": ALLOWED_IP,
        },
    ),
)
def test_aily_mcp_rejects_untrusted_requests_before_protocol_handling(
    headers: dict[str, str],
) -> None:
    adapter, gateway, caller_authorizer = _adapter()
    app = FastAPI(lifespan=adapter.lifespan)
    app.mount(adapter.mount_path, adapter.asgi_app)

    with TestClient(app) as client:
        response = client.post(
            f"{adapter.mount_path}/",
            headers=headers,
            json=_initialize_request(),
        )

    assert response.status_code == 403
    assert gateway.create_calls == []
    assert caller_authorizer.calls == []


def test_aily_mcp_rejects_an_unmapped_user_before_protocol_handling() -> None:
    caller_authorizer = _CallerAuthorizer(deny_identity=True)
    adapter, gateway, _ = _adapter(caller_authorizer=caller_authorizer)
    app = FastAPI(lifespan=adapter.lifespan)
    app.mount(adapter.mount_path, adapter.asgi_app)

    with TestClient(app) as client:
        response = client.post(
            f"{adapter.mount_path}/",
            headers={
                "x-quanxin-aily-source-ip": ALLOWED_IP,
                "x-aily-user": "unknown-aily-user",
                "x-aily-email": "unknown@example.com",
            },
            json=_initialize_request(),
        )

    assert response.status_code == 403
    assert caller_authorizer.identity_calls == ["unknown-aily-user"]
    assert caller_authorizer.calls == []
    assert gateway.create_calls == []


def _reserve_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(server: uvicorn.Server) -> None:
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if server.started:
            return
        time.sleep(0.05)
    raise AssertionError("temporary Aily MCP server did not start")


async def _call_create_analysis_task(url: str) -> tuple[set[str], Any]:
    headers = {
        "x-quanxin-aily-source-ip": ALLOWED_IP,
        "x-aily-user": "aily-user-1",
        "x-aily-email": "engineer@example.com",
    }
    async with (
        httpx.AsyncClient(headers=headers, timeout=10.0) as client,
        streamable_http_client(url, http_client=client) as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        listed = await session.list_tools()
        called = await session.call_tool(
            "quanxin_create_analysis_task",
            arguments={
                "task_type": FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                "source_run_id": "root-run",
                "data_batch_id": "batch-safe",
            },
        )
        return {tool.name for tool in listed.tools}, called


def test_official_client_initializes_lists_and_calls_the_aily_mcp_surface() -> None:
    port = _reserve_tcp_port()
    adapter, gateway, caller_authorizer = _adapter(
        allowed_hosts=(f"127.0.0.1:{port}",)
    )
    app = FastAPI(lifespan=adapter.lifespan)
    app.mount(adapter.mount_path, adapter.asgi_app)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_for_server(server)
        listed, called = asyncio.run(
            _call_create_analysis_task(
                f"http://127.0.0.1:{port}{adapter.mount_path}/"
            )
        )
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    assert thread.is_alive() is False
    assert listed == BASE_TOOL_NAMES
    assert called.isError is False
    assert called.structuredContent["run_id"] == gateway.state.run_id
    assert called.content == []
    assert gateway.create_calls == [
        AilyCreateAnalysisTaskRequest(
            task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            source_run_id="root-run",
            data_batch_id="batch-safe",
        )
    ]
    assert caller_authorizer.calls == [("aily-user-1", "root-run")]
    assert caller_authorizer.identity_calls
