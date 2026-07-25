from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import (
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel


class _Input(ContractModel):
    batch_id: str = Field(min_length=1)


class _TrustingValidator:
    def revalidate(self, context: object) -> object:
        return context


def _result(value: _Input, *, tool_name: str, tool_version: str) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version=tool_version,
        model_version="project-boundary-model-v1",
        data_version="project-boundary-data-v1",
        feature_version="project-boundary-feature-v1",
        input_hash=sha256_canonical(value.model_dump(mode="json")),
        values={"validated_batch": value.batch_id},
        provenance=[
            ProvenanceRecord(
                source_id="project-boundary-fixture",
                source_kind=SourceKind.OBSERVED,
                uri="test://project-boundary/fixture",
                sha256=sha256_canonical({"fixture": "project-boundary"}),
                description="Project-scoped registry boundary fixture",
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _context(project_id: str = "project-a"):
    from quanxin_life.application.invocation_context import (
        ProjectInvocationSource,
        VerifiedProjectInvocationContext,
    )

    return VerifiedProjectInvocationContext(
        project_id=project_id,
        actor_user_id="user-a",
        actor_session_id="session-a",
        actor_role=UserRole.MEMBER,
        invocation_source=ProjectInvocationSource.HTTP,
        agent_run_id=None,
        _authorization_tag="0" * 64,
    )


def _registry(
    project_calls: list[tuple[_Input, object]] | None = None,
):
    from quanxin_life.tools import (
        StandardToolName,
        ToolDefinition,
        ToolExecutionScope,
        ToolRegistry,
    )

    registry = ToolRegistry(project_context_validator=_TrustingValidator())
    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="global-boundary-v1",
            input_model=_Input,
            execution_scope=ToolExecutionScope.GLOBAL,
            executor=lambda value: _result(
                value,
                tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
                tool_version="global-boundary-v1",
            ),
        )
    )

    def execute_in_project(value: _Input, context: object) -> ToolResult:
        if project_calls is not None:
            project_calls.append((value, context))
        return _result(
            value,
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            tool_version="project-boundary-v1",
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="project-boundary-v1",
            input_model=_Input,
            execution_scope=ToolExecutionScope.PROJECT,
            executor=None,
            project_executor=execute_in_project,
        )
    )
    return registry


def test_generic_registry_neither_discovers_nor_executes_project_tools() -> None:
    from quanxin_life.tools import ToolAuthorizationError

    registry = _registry()

    assert [schema.tool_name.value for schema in registry.list_schemas()] == [
        "validate_battery_data"
    ]
    assert registry.execute(
        "validate_battery_data", {"batch_id": "batch-a"}
    ).tool_name == "validate_battery_data"

    with pytest.raises(ToolAuthorizationError, match="project-scoped"):
        registry.execute("predict_cycle_life", {"batch_id": "batch-a"})


def test_project_registry_executes_only_project_tools_and_passes_exact_context() -> None:
    from quanxin_life.tools import ToolAuthorizationError

    calls: list[tuple[_Input, object]] = []
    registry = _registry(calls)
    context = _context()

    result = registry.execute_in_project(
        "predict_cycle_life",
        {"batch_id": "batch-a"},
        context=context,
    )

    assert result.tool_name == "predict_cycle_life"
    assert calls == [(_Input(batch_id="batch-a"), context)]
    with pytest.raises(ToolAuthorizationError, match="global-scoped"):
        registry.execute_in_project(
            "validate_battery_data",
            {"batch_id": "batch-a"},
            context=context,
        )


def test_mcp_adapter_neither_discovers_nor_calls_project_tools() -> None:
    from quanxin_life.tools import ToolAuthorizationError
    from quanxin_life.tools.mcp_adapter import McpToolAdapter

    adapter = McpToolAdapter(_registry())

    assert [schema.tool_name.value for schema in adapter.discover_tools()] == [
        "validate_battery_data"
    ]
    with pytest.raises(ToolAuthorizationError, match="project-scoped"):
        adapter.call_tool(
            {
                "tool_name": "predict_cycle_life",
                "input": {"batch_id": "batch-a"},
            }
        )


class _FakeFastMCP:
    def __init__(self, name: str, **_: object) -> None:
        self.name = name
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, description: str):
        del description

        def register(handler: Any) -> Any:
            self.tools[name] = handler
            return handler

        return register


def test_mcp_host_does_not_publish_project_tools() -> None:
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.tools.mcp_host import create_mcp_host

    sdk = SimpleNamespace(
        server=SimpleNamespace(fastmcp=SimpleNamespace(FastMCP=_FakeFastMCP))
    )
    host = create_mcp_host(
        ToolInvocationService(registry=_registry()),
        sdk_loader=lambda: sdk,
    )

    assert tuple(host.server.tools) == ("validate_battery_data",)
