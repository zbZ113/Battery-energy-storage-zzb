from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry


class HostToolInput(ContractModel):
    request_label: str = Field(min_length=1)


def _result(validated_input: HostToolInput) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
        tool_version="1.0.0",
        model_version="mcp-host-test-model-v1",
        data_version="mcp-host-test-data-v1",
        feature_version="mcp-host-test-feature-v1",
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={"execution_status": "validated"},
        provenance=[
            ProvenanceRecord(
                source_id="mcp-host-test-input",
                source_kind=SourceKind.OBSERVED,
                uri="test://mcp-host/input",
                sha256=sha256_canonical({"fixture": "mcp-host"}),
                description="MCP host test input provenance",
                created_at=datetime(2026, 7, 15, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _service() -> ToolInvocationService:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="1.0.0",
            input_model=HostToolInput,
            executor=_result,
        )
    )
    return ToolInvocationService(registry=registry)


class _FakeFastMCP:
    def __init__(self, name: str, *, instructions: str | None = None) -> None:
        self.name = name
        self.instructions = instructions
        self.tools: dict[str, tuple[str, Any]] = {}
        self.run_calls: list[str] = []

    def tool(self, *, name: str, description: str):
        def register(handler: Any) -> Any:
            self.tools[name] = (description, handler)
            return handler

        return register

    def run(self, *, transport: str) -> None:
        self.run_calls.append(transport)


def _fake_sdk() -> Any:
    return SimpleNamespace(
        server=SimpleNamespace(fastmcp=SimpleNamespace(FastMCP=_FakeFastMCP))
    )


def test_factory_lazily_loads_sdk_registers_registry_tools_and_does_not_start() -> None:
    from quanxin_life.tools.mcp_host import McpHostConfig, create_mcp_host

    load_calls = 0

    def load_sdk() -> Any:
        nonlocal load_calls
        load_calls += 1
        return _fake_sdk()

    host = create_mcp_host(
        _service(),
        config=McpHostConfig(server_name="Quanxin MCP"),
        sdk_loader=load_sdk,
    )

    assert load_calls == 1
    assert host.server.name == "Quanxin MCP"
    assert tuple(host.server.tools) == ("validate_battery_data",)
    assert host.server.run_calls == []


def test_registered_handler_delegates_to_shared_tool_invocation_service() -> None:
    from quanxin_life.tools.mcp_host import create_mcp_host

    host = create_mcp_host(_service(), sdk_loader=_fake_sdk)
    _, handler = host.server.tools["validate_battery_data"]

    response = handler({"request_label": "batch-A"})

    assert response["tool_name"] == "validate_battery_data"
    assert response["input_hash"] == sha256_canonical({"request_label": "batch-A"})
    assert response["values"] == {"execution_status": "validated"}


@pytest.mark.parametrize(
    ("transport", "expected"),
    [("stdio", "stdio"), ("streamable-http", "streamable-http")],
)
def test_run_uses_only_the_configured_official_transport(
    transport: str,
    expected: str,
) -> None:
    from quanxin_life.tools.mcp_host import McpHostConfig, create_mcp_host

    host = create_mcp_host(
        _service(),
        config=McpHostConfig(transport=transport),
        sdk_loader=_fake_sdk,
    )

    host.run()

    assert host.server.run_calls == [expected]


def test_missing_sdk_is_an_explicit_unavailable_failure(monkeypatch) -> None:
    from quanxin_life.tools import mcp_host
    from quanxin_life.tools.mcp_adapter import McpSdkUnavailableError

    def unavailable() -> Any:
        raise McpSdkUnavailableError("optional MCP SDK is unavailable")

    monkeypatch.setattr(mcp_host, "load_optional_mcp_sdk", unavailable)

    with pytest.raises(McpSdkUnavailableError, match="optional MCP SDK"):
        mcp_host.create_mcp_host(_service())


def test_sdk_without_fastmcp_is_rejected_as_unavailable() -> None:
    from quanxin_life.tools.mcp_adapter import McpSdkUnavailableError
    from quanxin_life.tools.mcp_host import create_mcp_host

    with pytest.raises(McpSdkUnavailableError, match="FastMCP"):
        create_mcp_host(_service(), sdk_loader=lambda: SimpleNamespace())


def test_factory_requires_shared_service_boundary() -> None:
    from quanxin_life.tools.mcp_host import create_mcp_host

    with pytest.raises(TypeError, match="ToolInvocationService"):
        create_mcp_host(object(), sdk_loader=_fake_sdk)  # type: ignore[arg-type]


def test_config_rejects_unknown_transport_and_blank_server_name() -> None:
    from pydantic import ValidationError

    from quanxin_life.tools.mcp_host import McpHostConfig

    with pytest.raises(ValidationError):
        McpHostConfig(transport="sse")
    with pytest.raises(ValidationError):
        McpHostConfig(server_name="   ")
