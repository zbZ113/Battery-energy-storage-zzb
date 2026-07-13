"""Shared service boundary for transport layers that invoke domain tools.

HTTP, MCP and local user interfaces must delegate through this module rather
than implement their own tool validation or numeric behavior.  It does not
perform battery calculations and therefore cannot fabricate engineering values.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from pydantic import field_validator

from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, JsonMapping
from quanxin_life.tools import StandardToolName, ToolRegistry, create_available_tool_registry


class ToolInvocation(ContractModel):
    """Untrusted, JSON-only request for an already registered domain tool."""

    tool_name: StandardToolName
    input_value: JsonMapping

    @field_validator("input_value")
    @classmethod
    def require_json_compatible_input(cls, value: JsonMapping) -> JsonMapping:
        try:
            sha256_canonical(value)
        except TypeError as exc:
            raise ValueError("input_value must be JSON-compatible") from exc
        return value


@dataclass(frozen=True, slots=True)
class ToolInvocationService:
    """Single delegation point shared by API, MCP and interactive clients."""

    registry: ToolRegistry

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Execute an external invocation through the shared typed registry."""
        return self.registry.execute(invocation.tool_name, invocation.input_value)

    def invoke_for_agent(
        self,
        invocation: ToolInvocation,
        *,
        allowed_tool_names: Collection[StandardToolName | str] | None,
    ) -> ToolResult:
        """Execute an Agent invocation through the registry's strict allowlist path."""
        return self.registry.execute_for_agent(
            invocation.tool_name,
            invocation.input_value,
            allowed_tool_names=allowed_tool_names,
        )


def create_available_tool_invocation_service() -> ToolInvocationService:
    """Construct a service from the single shared implemented-tool assembly."""
    return ToolInvocationService(registry=create_available_tool_registry())
