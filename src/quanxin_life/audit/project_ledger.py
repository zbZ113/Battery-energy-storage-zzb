"""Project-isolated audit bindings for trusted tool results."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Protocol

from quanxin_life.audit.numeric_firewall import AuditLedger
from quanxin_life.core import ToolResult, UserRole

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        ProjectInvocationSource,
        VerifiedProjectInvocationContext,
    )


class ProjectContextValidator(Protocol):
    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext: ...


@dataclass(frozen=True, slots=True)
class ProjectToolResultBinding:
    """Non-numeric ownership metadata for one audited result."""

    result_id: str
    project_id: str
    actor_user_id: str
    actor_session_id: str
    actor_role: UserRole
    invocation_source: ProjectInvocationSource
    agent_run_id: str | None
    tool_name: str
    input_hash: str


class ProjectAuditLedger:
    """Append-only ToolResult ledger that enforces exact project isolation."""

    def __init__(self, *, context_validator: ProjectContextValidator) -> None:
        if not callable(getattr(context_validator, "revalidate", None)):
            raise TypeError("context_validator must revalidate project contexts")
        self._context_validator = context_validator
        self._ledger = AuditLedger()
        self._bindings: dict[str, ProjectToolResultBinding] = {}
        self._lock = RLock()

    @property
    def context_validator(self) -> ProjectContextValidator:
        return self._context_validator

    def register_result(
        self,
        context: VerifiedProjectInvocationContext,
        result: ToolResult,
    ) -> ToolResult:
        """Register one result and its server-derived project/actor binding."""

        verified = self._require_context(context)
        binding = ProjectToolResultBinding(
            result_id=result.result_id,
            project_id=verified.project_id,
            actor_user_id=verified.actor_user_id,
            actor_session_id=verified.actor_session_id,
            actor_role=verified.actor_role,
            invocation_source=verified.invocation_source,
            agent_run_id=verified.agent_run_id,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
        )
        with self._lock:
            registered = self._ledger.register_result(result)
            self._bindings[registered.result_id] = binding
        return registered

    def resolve_registered_result(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ToolResult:
        """Resolve only when the result belongs to the exact context project."""

        verified = self._require_context(context)
        with self._lock:
            binding = self._bindings.get(result_id)
            if binding is None or binding.project_id != verified.project_id:
                raise ValueError("ToolResult is not registered for this project")
            return self._ledger.resolve_registered_result(result_id)

    def resolve_binding(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ProjectToolResultBinding:
        """Return immutable non-numeric audit metadata for the exact project."""

        verified = self._require_context(context)
        with self._lock:
            binding = self._bindings.get(result_id)
            if binding is None or binding.project_id != verified.project_id:
                raise ValueError("ToolResult binding is not registered for this project")
            return binding

    def _require_context(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        from quanxin_life.application.invocation_context import (
            VerifiedProjectInvocationContext,
        )

        if not isinstance(context, VerifiedProjectInvocationContext):
            raise TypeError("verified project invocation context is required")
        return self._context_validator.revalidate(context)


__all__ = ["ProjectAuditLedger", "ProjectToolResultBinding"]
