"""Optional MCP transport host over the shared tool invocation service.

This module deliberately contains no battery-domain behavior.  It lazily loads
the optional MCP SDK, exposes the tools already assembled in the shared
``ToolRegistry``, and delegates every call to ``ToolInvocationService``.
Importing this module does not import or initialize the MCP SDK.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field, field_validator

from quanxin_life.core import ToolResult
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools.mcp_adapter import McpSdkUnavailableError, load_optional_mcp_sdk
from quanxin_life.tools.registry import StandardToolName, ToolRegistry, ToolSchema


class ToolInvocationServiceLike(Protocol):
    """Minimal transport service surface, kept independent of the API package."""

    registry: ToolRegistry

    def invoke(self, invocation: Any) -> ToolResult: ...


class McpTransport(StrEnum):
    """MCP transports supported by the official FastMCP host runner."""

    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable-http"


class McpHostConfig(ContractModel):
    """Dependency-free configuration for one optional MCP host."""

    server_name: str = Field(default="Quanxin Life Tools", min_length=1)
    transport: McpTransport = McpTransport.STDIO
    instructions: str = (
        "Battery engineering values are returned only by registered, audited domain tools."
    )

    @field_validator("server_name", "instructions")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("MCP host text fields must not be blank")
        return normalized


@dataclass(frozen=True, slots=True)
class McpHost:
    """A configured FastMCP server that is inert until ``run`` is called."""

    server: Any
    config: McpHostConfig

    def run(self) -> None:
        """Run the configured official transport without adding business logic."""

        self.server.run(transport=self.config.transport.value)


SdkLoader = Callable[[], object]


def create_mcp_host(
    service: ToolInvocationServiceLike,
    *,
    config: McpHostConfig | None = None,
    sdk_loader: SdkLoader = load_optional_mcp_sdk,
) -> McpHost:
    """Create, but do not start, a FastMCP host over one shared service.

    The SDK remains optional: construction explicitly fails with
    ``McpSdkUnavailableError`` when FastMCP cannot be loaded.  Each registered
    MCP handler accepts one JSON object named ``input_value`` and delegates its
    validation and execution to the existing Pydantic/registry boundary.
    """

    if not isinstance(getattr(service, "registry", None), ToolRegistry) or not callable(
        getattr(service, "invoke", None)
    ):
        raise TypeError("service must be a ToolInvocationService")

    resolved_config = config or McpHostConfig()
    sdk = sdk_loader()
    fastmcp_factory = _resolve_fastmcp_factory(sdk)
    server = fastmcp_factory(
        resolved_config.server_name,
        instructions=resolved_config.instructions,
    )

    for schema in service.registry.list_schemas():
        handler = _build_tool_handler(service, schema.tool_name)
        description = _tool_description(schema)
        server.tool(name=schema.tool_name.value, description=description)(handler)

    return McpHost(server=server, config=resolved_config)


def run_mcp_host(
    service: ToolInvocationServiceLike,
    *,
    config: McpHostConfig | None = None,
    sdk_loader: SdkLoader = load_optional_mcp_sdk,
) -> None:
    """Construct and run one stdio or Streamable HTTP MCP host."""

    create_mcp_host(service, config=config, sdk_loader=sdk_loader).run()


def _resolve_fastmcp_factory(sdk: object) -> Callable[..., Any]:
    fastmcp_factory = _nested_attribute(sdk, "server", "fastmcp", "FastMCP")
    if fastmcp_factory is None:
        try:
            fastmcp_module = importlib.import_module("mcp.server.fastmcp")
        except (ImportError, ModuleNotFoundError) as exc:
            raise McpSdkUnavailableError(
                "The optional MCP SDK does not provide the required FastMCP host"
            ) from exc
        fastmcp_factory = getattr(fastmcp_module, "FastMCP", None)

    if not callable(fastmcp_factory):
        raise McpSdkUnavailableError(
            "The optional MCP SDK does not provide the required FastMCP host"
        )
    return fastmcp_factory


def _nested_attribute(value: object, *names: str) -> object | None:
    current: object | None = value
    for name in names:
        if current is None:
            return None
        current = getattr(current, name, None)
    return current


def _build_tool_handler(
    service: ToolInvocationServiceLike,
    tool_name: StandardToolName,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    def invoke(input_value: Mapping[str, Any]) -> dict[str, Any]:
        from quanxin_life.api.service import ToolInvocation

        result = service.invoke(
            ToolInvocation(tool_name=tool_name, input_value=dict(input_value))
        )
        return result.model_dump(mode="json")

    invoke.__name__ = tool_name.value
    invoke.__qualname__ = tool_name.value
    invoke.__doc__ = f"Invoke the audited {tool_name.value} domain tool."
    return invoke


def _tool_description(schema: ToolSchema) -> str:
    serialized_schema = json.dumps(
        schema.input_schema,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        f"Audited domain tool version {schema.tool_version}. "
        f"Pass input_value matching this JSON Schema: {serialized_schema}"
    )
