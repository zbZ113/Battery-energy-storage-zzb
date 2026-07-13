from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel


class AdapterInput(ContractModel):
    request_label: str = Field(min_length=1)


def _provenance() -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="mcp-adapter-test-input",
        source_kind=SourceKind.OBSERVED,
        uri="test://mcp-adapter/input",
        sha256=sha256_canonical({"fixture": "mcp-adapter"}),
        description="MCP adapter test input provenance",
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _result(validated_input: AdapterInput) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="validate_battery_data",
        tool_version="1.0.0",
        model_version="mcp-adapter-test-model-v1",
        data_version="mcp-adapter-test-data-v1",
        feature_version="mcp-adapter-test-feature-v1",
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={"execution_status": "validated"},
        provenance=[_provenance()],
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _registry():
    from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="1.0.0",
            input_model=AdapterInput,
            executor=_result,
        )
    )
    return registry


def test_discovers_registered_tools_in_registry_order_without_mcp_sdk() -> None:
    from quanxin_life.tools.mcp_adapter import McpToolAdapter

    adapter = McpToolAdapter(_registry())

    tools = adapter.discover_tools()

    assert len(tools) == 1
    assert tools[0].tool_name == "validate_battery_data"
    assert tools[0].input_schema["properties"]["request_label"]["type"] == "string"


def test_validates_json_mapping_and_delegates_generic_call_to_registry() -> None:
    from quanxin_life.tools.mcp_adapter import McpToolAdapter

    adapter = McpToolAdapter(_registry())

    result = adapter.call_tool(
        {"tool_name": "validate_battery_data", "input": {"request_label": "batch-A"}}
    )

    assert result.tool_name == "validate_battery_data"
    assert result.input_hash == sha256_canonical({"request_label": "batch-A"})


def test_revalidates_constructed_request_and_rejects_non_json_mapping() -> None:
    from quanxin_life.tools.mcp_adapter import (
        McpRequestValidationError,
        McpToolAdapter,
        McpToolCallRequest,
    )

    adapter = McpToolAdapter(_registry())
    forged = McpToolCallRequest.model_construct(
        tool_name="validate_battery_data",
        input={"request_label": object()},
    )

    with pytest.raises(McpRequestValidationError):
        adapter.call_tool(forged)


def test_agent_call_delegates_to_strict_registry_boundary() -> None:
    from quanxin_life.tools import ToolAuthorizationError
    from quanxin_life.tools.mcp_adapter import McpToolAdapter

    adapter = McpToolAdapter(_registry())
    request = {"tool_name": "validate_battery_data", "input": {"request_label": "batch-A"}}

    with pytest.raises(ToolAuthorizationError, match="non-empty allowlist"):
        adapter.call_tool_for_agent(request)

    result = adapter.call_tool_for_agent(
        request,
        allowed_tool_names={"validate_battery_data"},
    )

    assert result.tool_name == "validate_battery_data"


def test_reports_explicit_degraded_error_when_optional_mcp_sdk_is_absent(monkeypatch) -> None:
    from quanxin_life.tools import mcp_adapter

    def missing_mcp_sdk(module_name: str):
        assert module_name == "mcp"
        raise ModuleNotFoundError("No module named 'mcp'", name="mcp")

    monkeypatch.setattr(mcp_adapter.importlib, "import_module", missing_mcp_sdk)

    with pytest.raises(mcp_adapter.McpSdkUnavailableError, match="optional MCP SDK"):
        mcp_adapter.load_optional_mcp_sdk()


def test_adapter_constructor_requires_tool_registry() -> None:
    from quanxin_life.tools.mcp_adapter import McpToolAdapter

    with pytest.raises(TypeError, match="ToolRegistry"):
        McpToolAdapter(object())  # type: ignore[arg-type]
