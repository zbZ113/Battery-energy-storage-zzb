from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    ProjectInvocationSource,
)
from quanxin_life.core import ProjectStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import session_scope
from quanxin_life.persistence.models import FeishuBindingRow, Project, User

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


def _fixture() -> tuple[ProjectInvocationContextService, object, str, str, str]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    binding_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add(
            User(
                id=user_id,
                username="feishu-owner@example.test",
                credential_hash="not-used-by-feishu-binding",
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
                name="Feishu project",
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
    return (
        ProjectInvocationContextService(sessions, clock=lambda: NOW),
        sessions,
        user_id,
        project_id,
        binding_id,
    )


def test_active_feishu_binding_issues_a_revalidatable_project_context() -> None:
    service, _sessions, user_id, project_id, binding_id = _fixture()

    context = service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )

    assert context.project_id == project_id
    assert context.actor_user_id == user_id
    assert context.actor_session_id is None
    assert context.actor_role is UserRole.ADMIN
    assert context.invocation_source is ProjectInvocationSource.FEISHU
    assert context.feishu_binding_id == binding_id
    assert service.revalidate(context) == context


def test_feishu_context_stops_authorizing_after_binding_is_deactivated() -> None:
    service, sessions, _user_id, _project_id, binding_id = _fixture()
    context = service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    with session_scope(sessions) as session:
        binding = session.get(FeishuBindingRow, binding_id)
        assert binding is not None
        binding.status = "INACTIVE"

    with pytest.raises(ProjectInvocationAccessError, match="Feishu binding"):
        service.revalidate(context)


def test_feishu_context_stops_authorizing_after_identity_mapping_changes() -> None:
    service, sessions, user_id, _project_id, binding_id = _fixture()
    context = service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    with session_scope(sessions) as session:
        binding = session.get(FeishuBindingRow, binding_id)
        assert binding is not None
        binding.user_open_id_map_json = {user_id: "ou-reassigned"}

    with pytest.raises(ProjectInvocationAccessError, match="Feishu binding"):
        service.revalidate(context)


def test_feishu_resolution_rejects_ambiguous_active_identity_mapping() -> None:
    service, sessions, user_id, project_id, _binding_id = _fixture()
    with session_scope(sessions) as session:
        session.add(
            FeishuBindingRow(
                id=str(uuid4()),
                project_id=project_id,
                chat_id="oc-approved",
                bitable_app_token=None,
                bitable_table_id=None,
                user_open_id_map_json={user_id: "ou-approved"},
                binding_version="feishu-binding-v2",
                status="ACTIVE",
                created_at=NOW,
            )
        )

    with pytest.raises(ProjectInvocationAccessError, match="Feishu binding"):
        service.resolve_feishu(
            chat_id="oc-approved",
            sender_open_id="ou-approved",
        )
