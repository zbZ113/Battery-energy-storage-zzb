"""Shared service boundary for transport layers that invoke domain tools.

HTTP, MCP and local user interfaces must delegate through this module rather
than implement their own tool validation or numeric behavior.  It does not
perform battery calculations and therefore cannot fabricate engineering values.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import field_validator

from quanxin_life.audit import AuditLedger, AuditLedgerError, ProjectResultLedger
from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, JsonMapping
from quanxin_life.tools.bootstrap import create_available_tool_registry
from quanxin_life.tools.registry import (
    StandardToolName,
    ToolAuthorizationError,
    ToolRegistry,
)

if TYPE_CHECKING:
    from quanxin_life.application.agent_run_invocation import (
        AgentRunInvocationValidator,
        VerifiedAgentRunInvocationGrant,
    )
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )


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
    audit_ledger: AuditLedger | None = None
    project_audit_ledger: ProjectResultLedger | None = None
    agent_run_invocation_validator: AgentRunInvocationValidator | None = None

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """Execute an external invocation through the shared typed registry."""
        result = self.registry.execute(invocation.tool_name, invocation.input_value)
        return self._register_result(result)

    def invoke_for_agent(
        self,
        invocation: ToolInvocation,
        *,
        allowed_tool_names: Collection[StandardToolName | str] | None,
    ) -> ToolResult:
        """Execute an Agent invocation through the registry's strict allowlist path."""
        result = self.registry.execute_for_agent(
            invocation.tool_name,
            invocation.input_value,
            allowed_tool_names=allowed_tool_names,
        )
        return self._register_result(result)

    def invoke_in_project(
        self,
        invocation: ToolInvocation,
        *,
        context: VerifiedProjectInvocationContext,
    ) -> ToolResult:
        """Execute and audit one tool inside a server-verified project scope."""

        from quanxin_life.application.invocation_context import ProjectInvocationSource

        if context.invocation_source is ProjectInvocationSource.AGENT:
            raise ToolAuthorizationError(
                "AGENT project contexts require a verified per-step grant"
            )
        if self.project_audit_ledger is None:
            raise AuditLedgerError("project audit ledger is required before execution")
        result = self.registry.execute_in_project(
            invocation.tool_name,
            invocation.input_value,
            context=context,
        )
        return self.project_audit_ledger.register_result(context, result)

    def execute_in_project_unregistered(
        self,
        invocation: ToolInvocation,
        *,
        context: VerifiedProjectInvocationContext,
    ) -> ToolResult:
        """Execute a project tool without persisting it for an atomic caller."""

        from quanxin_life.application.invocation_context import ProjectInvocationSource

        if context.invocation_source is ProjectInvocationSource.AGENT:
            raise ToolAuthorizationError(
                "AGENT project contexts require a verified per-step grant"
            )
        if self.project_audit_ledger is None:
            raise AuditLedgerError("project audit ledger is required before execution")
        return self.registry.execute_in_project(
            invocation.tool_name,
            invocation.input_value,
            context=context,
        )

    def invoke_for_project_agent(
        self,
        invocation: ToolInvocation,
        *,
        grant: VerifiedAgentRunInvocationGrant,
    ) -> ToolResult:
        """Execute one PROJECT tool from a server-derived singleton step grant."""

        if self.agent_run_invocation_validator is None:
            raise ToolAuthorizationError(
                "project Agent invocation validator is required before execution"
            )
        if self.project_audit_ledger is None:
            raise AuditLedgerError("project audit ledger is required before execution")
        verified = self.agent_run_invocation_validator.revalidate(grant)
        from quanxin_life.application.invocation_context import ProjectInvocationSource

        if (
            verified.project_context.invocation_source
            is not ProjectInvocationSource.AGENT
            or verified.project_context.agent_run_id != verified.agent_run_id
            or verified.allowed_tool_names != frozenset({invocation.tool_name})
        ):
            raise ToolAuthorizationError(
                f"Tool '{invocation.tool_name.value}' is not permitted for this Agent step"
            )
        if (
            self.registry.canonical_input_hash(
                invocation.tool_name,
                invocation.input_value,
            )
            != verified.input_hash
        ):
            raise ToolAuthorizationError(
                "project Agent tool input does not match the frozen step grant"
            )
        result = self.registry.execute_in_project(
            invocation.tool_name,
            invocation.input_value,
            context=verified.project_context,
            allowed_tool_names=verified.allowed_tool_names,
        )
        verified_after_execution = self.agent_run_invocation_validator.revalidate(
            verified
        )
        if verified_after_execution != verified:
            raise ToolAuthorizationError(
                "project Agent invocation grant changed during execution"
            )
        return self.project_audit_ledger.commit_agent_step_result(
            grant=verified_after_execution,
            result=result,
            grant_validator=self.agent_run_invocation_validator,
        )

    def _register_result(self, result: ToolResult) -> ToolResult:
        if self.audit_ledger is None:
            return result
        return self.audit_ledger.register_result(result)


def create_available_tool_invocation_service() -> ToolInvocationService:
    """Construct a service from the single shared implemented-tool assembly."""
    return ToolInvocationService(registry=create_available_tool_registry())
