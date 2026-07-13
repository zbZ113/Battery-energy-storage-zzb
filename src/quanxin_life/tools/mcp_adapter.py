"""Optional MCP-facing adapter that delegates exclusively to ``ToolRegistry``.

The adapter is intentionally transport-neutral: the core remains usable when
the optional MCP SDK is not installed, while a later host can lazily load that
SDK and expose these same typed discovery and call operations over MCP.
"""

from __future__ import annotations

import importlib
from collections.abc import Collection, Mapping
from types import ModuleType
from typing import Any

from pydantic import ValidationError, field_validator

from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools.registry import StandardToolName, ToolRegistry, ToolSchema


class McpAdapterError(RuntimeError):
    """Base class for explicit MCP adapter failures."""


class McpRequestValidationError(McpAdapterError):
    """Raised when an untrusted MCP tool request is not JSON-contract safe."""


class McpSdkUnavailableError(McpAdapterError):
    """Raised when a transport host requests the optional MCP SDK without it installed."""


class McpToolCallRequest(ContractModel):
    """Transport-neutral, JSON-safe request used by MCP and local callers."""

    tool_name: StandardToolName
    input: dict[str, Any]

    @field_validator("input")
    @classmethod
    def input_must_be_json_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            sha256_canonical(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("input must contain only JSON-compatible values") from exc
        return value


class McpToolAdapter:
    """A deterministic adapter over one shared in-process ``ToolRegistry``."""

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry")
        self._registry = registry

    def discover_tools(self) -> tuple[ToolSchema, ...]:
        """Return registry-provided tool schemas without loading an MCP SDK."""

        return self._registry.list_schemas()

    def call_tool(
        self, request: McpToolCallRequest | Mapping[str, Any]
    ) -> ToolResult:
        """Validate one JSON call and delegate generic execution to the registry."""

        validated_request = self._validate_request(request)
        return self._registry.execute(validated_request.tool_name, validated_request.input)

    def call_tool_for_agent(
        self,
        request: McpToolCallRequest | Mapping[str, Any],
        *,
        allowed_tool_names: Collection[StandardToolName | str] | None = None,
    ) -> ToolResult:
        """Validate one JSON call and delegate strict Agent execution to the registry."""

        validated_request = self._validate_request(request)
        return self._registry.execute_for_agent(
            validated_request.tool_name,
            validated_request.input,
            allowed_tool_names=allowed_tool_names,
        )

    @staticmethod
    def _validate_request(
        request: McpToolCallRequest | Mapping[str, Any],
    ) -> McpToolCallRequest:
        if isinstance(request, McpToolCallRequest):
            try:
                raw_request: Mapping[str, Any] = request.model_dump(mode="json")
            except (TypeError, ValueError) as exc:
                raise McpRequestValidationError(
                    "MCP request cannot be serialized as a JSON mapping"
                ) from exc
        elif isinstance(request, Mapping):
            raw_request = request
        else:
            raise McpRequestValidationError("MCP request must be a mapping or McpToolCallRequest")

        try:
            return McpToolCallRequest.model_validate(raw_request)
        except (TypeError, ValueError, ValidationError) as exc:
            message = "MCP request does not satisfy the call contract"
            raise McpRequestValidationError(message) from exc


def load_optional_mcp_sdk() -> ModuleType:
    """Load the MCP SDK only when a transport host explicitly requests it."""

    try:
        return importlib.import_module("mcp")
    except ModuleNotFoundError as exc:
        raise McpSdkUnavailableError(
            "The optional MCP SDK is unavailable; use the in-process McpToolAdapter "
            "or install the project's MCP extra before starting an MCP transport host"
        ) from exc
