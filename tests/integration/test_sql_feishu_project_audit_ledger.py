from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import SqlProjectAuditLedger
from quanxin_life.core import (
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig
from quanxin_life.persistence.models import (
    FeishuBindingRow,
    Project,
    ProjectToolResultBindingRecord,
    User,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def test_feishu_project_result_survives_persistent_ledger_restart(
    tmp_path: Path,
) -> None:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'feishu-ledger.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    binding_id = str(uuid4())
    with sessions.begin() as session:
        session.add(
            User(
                id=user_id,
                username="feishu-ledger@example.test",
                credential_hash="not-used-by-feishu",
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Project(
                id=project_id,
                owner_user_id=user_id,
                name="Feishu ledger project",
                status=ProjectStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            FeishuBindingRow(
                id=binding_id,
                project_id=project_id,
                chat_id="oc-approved",
                bitable_app_token=None,
                bitable_table_id=None,
                user_open_id_map_json={user_id: "ou-approved"},
                binding_version="feishu-binding-v1",
                status="ACTIVE",
                created_at=NOW,
            )
        )

    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    context = context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="predict-cycle-life-v1",
        model_version="reviewed-model-v1",
        data_version="matr-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"record_batch_id": "batch-20"}),
        values={"prediction_result_id": "reviewed-result"},
        provenance=[
            ProvenanceRecord(
                source_id="batch-20",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch://batch-20",
                sha256="a" * 64,
                description="Reviewed frozen record batch",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )
    ledger = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )

    registered = ledger.register_result(context, result)
    restored = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )

    assert restored.resolve_registered_result(context, registered.result_id) == registered
    binding = restored.resolve_binding(context, registered.result_id)
    assert binding.actor_session_id is None
    assert binding.feishu_binding_id == binding_id
    with sessions() as session:
        row = session.scalar(
            select(ProjectToolResultBindingRecord).where(
                ProjectToolResultBindingRecord.result_id == registered.result_id
            )
        )
        assert row is not None
        assert row.binding_schema_version == "project-tool-result-binding-v3"
        assert row.invocation_source == "FEISHU"
        assert row.actor_session_id is None
        assert row.feishu_binding_id == binding_id
