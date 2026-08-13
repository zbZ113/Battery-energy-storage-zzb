from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    ProjectInvocationNotFoundError,
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import ProjectStatus, SessionStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import Project, SessionRecord, User, UserProjectRole

NOW = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)


def _principal(
    role: UserRole,
    *,
    user_id: str | None = None,
    must_change_password: bool = False,
) -> AuthPrincipal:
    resolved_user_id = user_id or str(uuid4())
    return AuthPrincipal(
        user_id=resolved_user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{resolved_user_id[:8]}@example.test",
        role=role,
        must_change_password=must_change_password,
    )


@dataclass(frozen=True)
class _Context:
    service: ProjectInvocationContextService
    session_factory: SessionFactory
    principals: dict[str, AuthPrincipal]
    active_project_id: str
    inactive_project_id: str


@pytest.fixture
def context(tmp_path: Path) -> _Context:
    engine = create_engine_from_config(
        DatabaseConfig(
            url=f"sqlite+pysqlite:///{tmp_path / 'project-invocation.sqlite3'}"
        )
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    member = _principal(UserRole.MEMBER)
    principals = {
        "admin": _principal(UserRole.ADMIN),
        "member": member,
        "outsider": _principal(UserRole.MEMBER),
        "judge": _principal(UserRole.JUDGE),
        "must_change": _principal(
            UserRole.MEMBER,
            user_id=member.user_id,
            must_change_password=True,
        ),
    }
    active_project_id = str(uuid4())
    inactive_project_id = str(uuid4())

    with session_factory.begin() as session:
        for principal in (
            principals["admin"],
            principals["member"],
            principals["outsider"],
            principals["judge"],
        ):
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only-not-used-for-authentication",
                    must_change_credential=False,
                    role=principal.role.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
            session.add(
                SessionRecord(
                    id=principal.session_id,
                    user_id=principal.user_id,
                    token_hash=principal.session_id.replace("-", "").ljust(64, "0")[:64],
                    status=SessionStatus.ACTIVE.value,
                    created_at=NOW,
                    expires_at=NOW + timedelta(hours=12),
                    revoked_at=None,
                )
            )
        session.add_all(
            (
                Project(
                    id=active_project_id,
                    owner_user_id=member.user_id,
                    name="active invocation project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Project(
                    id=inactive_project_id,
                    owner_user_id=member.user_id,
                    name="archived invocation project",
                    status=ProjectStatus.ARCHIVED.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
            )
        )
        session.add_all(
            (
                UserProjectRole(
                    id=str(uuid4()),
                    user_id=member.user_id,
                    project_id=active_project_id,
                    role=UserRole.MEMBER.value,
                    created_at=NOW,
                ),
                UserProjectRole(
                    id=str(uuid4()),
                    user_id=member.user_id,
                    project_id=inactive_project_id,
                    role=UserRole.MEMBER.value,
                    created_at=NOW,
                ),
                UserProjectRole(
                    id=str(uuid4()),
                    user_id=principals["judge"].user_id,
                    project_id=active_project_id,
                    role=UserRole.JUDGE.value,
                    created_at=NOW,
                ),
            )
        )

    return _Context(
        service=ProjectInvocationContextService(session_factory, clock=lambda: NOW),
        session_factory=session_factory,
        principals=principals,
        active_project_id=active_project_id,
        inactive_project_id=inactive_project_id,
    )


@pytest.mark.parametrize("principal_name", ["admin", "member"])
def test_http_operator_resolves_frozen_active_project_context(
    context: _Context,
    principal_name: str,
) -> None:
    principal = context.principals[principal_name]

    resolved = context.service.resolve_http(principal, context.active_project_id)

    assert isinstance(resolved, VerifiedProjectInvocationContext)
    assert tuple(
        field.name for field in fields(resolved) if not field.name.startswith("_")
    ) == (
        "project_id",
        "actor_user_id",
        "actor_session_id",
        "actor_role",
        "invocation_source",
        "agent_run_id",
        "feishu_binding_id",
    )
    assert resolved.project_id == context.active_project_id
    assert resolved.actor_user_id == principal.user_id
    assert resolved.actor_session_id == principal.session_id
    assert resolved.actor_role is principal.role
    assert resolved.invocation_source is ProjectInvocationSource.HTTP
    assert resolved.agent_run_id is None
    assert resolved.feishu_binding_id is None
    assert len(resolved._authorization_tag) == 64
    with pytest.raises(FrozenInstanceError):
        resolved.project_id = str(uuid4())  # type: ignore[misc]


@pytest.mark.parametrize(
    ("principal_name", "project_id_attribute"),
    (
        ("outsider", "active_project_id"),
        ("member", "missing"),
        ("member", "inactive_project_id"),
        ("admin", "inactive_project_id"),
    ),
)
def test_http_resolution_hides_invisible_missing_and_inactive_projects(
    context: _Context,
    principal_name: str,
    project_id_attribute: str,
) -> None:
    project_id = (
        str(uuid4())
        if project_id_attribute == "missing"
        else getattr(context, project_id_attribute)
    )

    with pytest.raises(ProjectInvocationNotFoundError):
        context.service.resolve_http(context.principals[principal_name], project_id)


def test_http_resolution_rejects_password_rotation_principal(
    context: _Context,
) -> None:
    with pytest.raises(ProjectInvocationAccessError):
        context.service.resolve_http(
            context.principals["must_change"],
            context.active_project_id,
        )


def test_http_resolution_rejects_non_operator_even_with_project_membership(
    context: _Context,
) -> None:
    with pytest.raises(ProjectInvocationAccessError):
        context.service.resolve_http(
            context.principals["judge"],
            context.active_project_id,
        )


def test_revalidation_rejects_forged_context(context: _Context) -> None:
    resolved = context.service.resolve_http(
        context.principals["member"],
        context.active_project_id,
    )
    forged = replace(resolved, _authorization_tag="0" * 64)

    with pytest.raises(ProjectInvocationAccessError):
        context.service.revalidate(forged)


def test_revalidation_rejects_project_archived_after_issue(context: _Context) -> None:
    resolved = context.service.resolve_http(
        context.principals["member"],
        context.active_project_id,
    )
    with context.session_factory.begin() as session:
        project = session.get(Project, context.active_project_id)
        assert project is not None
        project.status = ProjectStatus.ARCHIVED.value

    with pytest.raises(ProjectInvocationNotFoundError):
        context.service.revalidate(resolved)


def test_revalidation_rejects_membership_revoked_after_issue(context: _Context) -> None:
    member = context.principals["member"]
    resolved = context.service.resolve_http(member, context.active_project_id)
    with context.session_factory.begin() as session:
        project = session.get(Project, context.active_project_id)
        assert project is not None
        project.owner_user_id = context.principals["outsider"].user_id
        membership = (
            session.query(UserProjectRole)
            .filter_by(user_id=member.user_id, project_id=context.active_project_id)
            .one()
        )
        session.delete(membership)

    with pytest.raises(ProjectInvocationNotFoundError):
        context.service.revalidate(resolved)


def test_revalidation_rejects_disabled_user_and_revoked_session(
    context: _Context,
) -> None:
    member = context.principals["member"]
    resolved = context.service.resolve_http(member, context.active_project_id)
    with context.session_factory.begin() as session:
        user = session.get(User, member.user_id)
        assert user is not None
        user.status = UserStatus.DISABLED.value

    with pytest.raises(ProjectInvocationAccessError):
        context.service.revalidate(resolved)

    with context.session_factory.begin() as session:
        user = session.get(User, member.user_id)
        session_record = session.get(SessionRecord, member.session_id)
        assert user is not None and session_record is not None
        user.status = UserStatus.ACTIVE.value
        session_record.status = SessionStatus.REVOKED.value
        session_record.revoked_at = NOW

    with pytest.raises(ProjectInvocationAccessError):
        context.service.revalidate(resolved)
