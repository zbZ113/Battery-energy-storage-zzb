from __future__ import annotations

from datetime import UTC, datetime
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


def _context(project_id: str):
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


def _result(value: _Input) -> ToolResult:
    from quanxin_life.tools import StandardToolName

    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version="project-service-v1",
        model_version="project-service-model-v1",
        data_version="project-service-data-v1",
        feature_version="project-service-feature-v1",
        input_hash=sha256_canonical(value.model_dump(mode="json")),
        values={"validated_batch": value.batch_id},
        provenance=[
            ProvenanceRecord(
                source_id="project-service-fixture",
                source_kind=SourceKind.OBSERVED,
                uri="test://project-service/fixture",
                sha256=sha256_canonical({"fixture": "project-service"}),
                description="Project-scoped service boundary fixture",
                created_at=datetime(2026, 7, 25, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 7, 25, tzinfo=UTC),
    )


def _registry(calls: list[tuple[_Input, object]] | None = None):
    from quanxin_life.tools import (
        StandardToolName,
        ToolDefinition,
        ToolExecutionScope,
        ToolRegistry,
    )

    def executor(value: _Input, context: object) -> ToolResult:
        if calls is not None:
            calls.append((value, context))
        return _result(value)

    registry = ToolRegistry(project_context_validator=_TrustingValidator())
    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="project-service-v1",
            input_model=_Input,
            execution_scope=ToolExecutionScope.PROJECT,
            executor=None,
            project_executor=executor,
        )
    )
    return registry


def _invocation():
    from quanxin_life.api.service import ToolInvocation
    from quanxin_life.tools import StandardToolName

    return ToolInvocation(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        input_value={"batch_id": "batch-a"},
    )


def test_generic_service_cannot_invoke_project_tool() -> None:
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.tools import ToolAuthorizationError

    service = ToolInvocationService(registry=_registry())

    with pytest.raises(ToolAuthorizationError, match="project-scoped"):
        service.invoke(_invocation())


def test_project_service_requires_project_audit_ledger_before_execution() -> None:
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.audit import AuditLedgerError

    calls: list[tuple[_Input, object]] = []
    service = ToolInvocationService(registry=_registry(calls))

    with pytest.raises(AuditLedgerError, match="project audit ledger"):
        service.invoke_in_project(_invocation(), context=_context("project-a"))

    assert calls == []


def test_project_service_passes_context_and_registers_project_bound_result() -> None:
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.audit import ProjectAuditLedger

    calls: list[tuple[_Input, object]] = []
    ledger = ProjectAuditLedger(context_validator=_TrustingValidator())
    context = _context("project-a")
    service = ToolInvocationService(
        registry=_registry(calls),
        project_audit_ledger=ledger,
    )

    result = service.invoke_in_project(_invocation(), context=context)

    assert calls == [(_Input(batch_id="batch-a"), context)]
    assert ledger.resolve_registered_result(context, result.result_id) == result


def test_project_result_cannot_be_resolved_from_another_project() -> None:
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.audit import ProjectAuditLedger

    ledger = ProjectAuditLedger(context_validator=_TrustingValidator())
    service = ToolInvocationService(
        registry=_registry(),
        project_audit_ledger=ledger,
    )
    result = service.invoke_in_project(
        _invocation(),
        context=_context("project-a"),
    )

    with pytest.raises(ValueError, match="project"):
        ledger.resolve_registered_result(_context("project-b"), result.result_id)
