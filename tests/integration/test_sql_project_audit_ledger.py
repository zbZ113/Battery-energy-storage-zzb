from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select

from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    VerifiedProjectInvocationContext,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    Project,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
    User,
)

NOW = datetime(2026, 7, 25, 16, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Context:
    session_factory: SessionFactory
    context_service: ProjectInvocationContextService
    invocation: VerifiedProjectInvocationContext
    other_project_invocation: VerifiedProjectInvocationContext
    session_id: str


def _context(tmp_path: Path) -> _Context:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'project-ledger.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    user_id = str(uuid4())
    session_id = str(uuid4())
    project_id = str(uuid4())
    other_project_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            User(
                id=user_id,
                username="project-ledger@example.test",
                credential_hash="reviewed-hash",
                must_change_credential=False,
                role=UserRole.MEMBER.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=session_id,
                user_id=user_id,
                token_hash="a" * 64,
                status=SessionStatus.ACTIVE.value,
                created_at=NOW,
                expires_at=NOW + timedelta(hours=1),
            )
        )
        for identifier in (project_id, other_project_id):
            session.add(
                Project(
                    id=identifier,
                    owner_user_id=user_id,
                    name=f"Project {identifier[:8]}",
                    status="ACTIVE",
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
    principal = AuthPrincipal(
        user_id=user_id,
        session_id=session_id,
        username="project-ledger@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )
    service = ProjectInvocationContextService(session_factory, clock=lambda: NOW)
    return _Context(
        session_factory=session_factory,
        context_service=service,
        invocation=service.resolve_http(principal, project_id),
        other_project_invocation=service.resolve_http(principal, other_project_id),
        session_id=session_id,
    )


def _result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="validate_battery_data",
        tool_version="persistent-project-ledger-v1",
        model_version=None,
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash=sha256_canonical({"record_batch_id": "binding-1"}),
        values={"validated_record_batch_id": "binding-1"},
        warnings=["fixture-warning"],
        provenance=[
            ProvenanceRecord(
                source_id="source-b",
                source_kind=SourceKind.OBSERVED,
                uri="test://project-ledger/source-b",
                sha256="b" * 64,
                description="Second source",
                created_at=NOW,
            ),
            ProvenanceRecord(
                source_id="source-a",
                source_kind=SourceKind.OBSERVED,
                uri="test://project-ledger/source-a",
                sha256="a" * 64,
                description="First source",
                created_at=NOW,
            ),
        ],
        created_at=NOW,
    )


def _ledger(context: _Context):
    from quanxin_life.audit import SqlProjectAuditLedger

    return SqlProjectAuditLedger(
        context.session_factory,
        context_validator=context.context_service,
        clock=lambda: NOW,
    )


def test_result_and_project_binding_survive_ledger_restart(tmp_path: Path) -> None:
    context = _context(tmp_path)
    registered = _ledger(context).register_result(context.invocation, _result())

    restored = _ledger(context)
    assert restored.resolve_registered_result(
        context.invocation, registered.result_id
    ) == registered
    binding = restored.resolve_binding(context.invocation, registered.result_id)
    assert binding.project_id == context.invocation.project_id
    assert binding.actor_user_id == context.invocation.actor_user_id
    assert binding.actor_session_id == context.invocation.actor_session_id
    assert binding.tool_name == registered.tool_name
    assert binding.input_hash == registered.input_hash

    with context.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ToolResultRecord)) == 1
        assert session.scalar(select(func.count()).select_from(ProvenanceRecordRow)) == 2
        assert (
            session.scalar(
                select(func.count()).select_from(ProjectToolResultBindingRecord)
            )
            == 1
        )


def test_tool_result_is_inserted_before_fk_dependent_project_binding(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    engine = context.session_factory.kw["bind"]
    insert_statements: list[str] = []

    def record_insert_order(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: object,
    ) -> None:
        if statement.lstrip().upper().startswith("INSERT"):
            insert_statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_insert_order)
    try:
        _ledger(context).register_result(context.invocation, _result())
    finally:
        event.remove(engine, "before_cursor_execute", record_insert_order)

    tool_result_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO tool_results" in statement
    )
    project_binding_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO project_tool_result_bindings" in statement
    )
    assert tool_result_position < project_binding_position


def test_result_is_hidden_from_another_live_project(tmp_path: Path) -> None:
    context = _context(tmp_path)
    ledger = _ledger(context)
    registered = ledger.register_result(context.invocation, _result())

    with pytest.raises(ValueError, match="project"):
        ledger.resolve_registered_result(
            context.other_project_invocation,
            registered.result_id,
        )


def test_revoked_session_invalidates_existing_context_before_resolution(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    ledger = _ledger(context)
    registered = ledger.register_result(context.invocation, _result())
    with context.session_factory.begin() as session:
        persisted = session.get(SessionRecord, context.session_id)
        assert persisted is not None
        persisted.status = SessionStatus.REVOKED.value
        persisted.revoked_at = NOW

    with pytest.raises(ProjectInvocationAccessError, match="no longer authorized"):
        ledger.resolve_registered_result(context.invocation, registered.result_id)


def test_tool_result_payload_tampering_fails_closed(tmp_path: Path) -> None:
    from quanxin_life.audit import AuditLedgerError

    context = _context(tmp_path)
    ledger = _ledger(context)
    registered = ledger.register_result(context.invocation, _result())
    with context.session_factory.begin() as session:
        row = session.get(ToolResultRecord, registered.result_id)
        assert row is not None
        row.values_json = {"validated_record_batch_id": "tampered"}

    with pytest.raises(AuditLedgerError, match="integrity"):
        ledger.resolve_registered_result(context.invocation, registered.result_id)


def test_tool_result_provenance_tampering_fails_closed(tmp_path: Path) -> None:
    from quanxin_life.audit import AuditLedgerError

    context = _context(tmp_path)
    ledger = _ledger(context)
    registered = ledger.register_result(context.invocation, _result())
    with context.session_factory.begin() as session:
        row = session.scalar(
            select(ProvenanceRecordRow).where(
                ProvenanceRecordRow.tool_result_id == registered.result_id,
                ProvenanceRecordRow.source_id == "source-a",
            )
        )
        assert row is not None
        row.sha256 = "c" * 64

    with pytest.raises(AuditLedgerError, match="integrity"):
        ledger.resolve_registered_result(context.invocation, registered.result_id)


def test_project_binding_tampering_fails_closed(tmp_path: Path) -> None:
    from quanxin_life.audit import AuditLedgerError

    context = _context(tmp_path)
    ledger = _ledger(context)
    registered = ledger.register_result(context.invocation, _result())
    with context.session_factory.begin() as session:
        row = session.get(ProjectToolResultBindingRecord, registered.result_id)
        assert row is not None
        row.actor_role = UserRole.ADMIN.value

    with pytest.raises(AuditLedgerError, match="integrity"):
        ledger.resolve_binding(context.invocation, registered.result_id)


def test_duplicate_result_id_is_rejected_without_partial_rows(tmp_path: Path) -> None:
    from quanxin_life.audit import DuplicateAuditResultError

    context = _context(tmp_path)
    ledger = _ledger(context)
    result = _result()
    ledger.register_result(context.invocation, result)

    with pytest.raises(DuplicateAuditResultError, match="duplicate ToolResult"):
        ledger.register_result(context.invocation, result)

    with context.session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ToolResultRecord)) == 1
        assert (
            session.scalar(
                select(func.count()).select_from(ProjectToolResultBindingRecord)
            )
            == 1
        )
