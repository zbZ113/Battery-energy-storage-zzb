from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import event, func, select

from quanxin_life.application.projects import (
    ProjectAccessError,
    ProjectNotFoundError,
    ProjectService,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import ProjectStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import Project, User, UserProjectRole

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


@pytest.fixture
def project_context(
    tmp_path: Path,
) -> tuple[ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'projects.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals: dict[UserRole | str, AuthPrincipal] = {
        UserRole.ADMIN: _principal(str(uuid4()), UserRole.ADMIN),
        UserRole.MEMBER: _principal(str(uuid4()), UserRole.MEMBER),
        UserRole.JUDGE: _principal(str(uuid4()), UserRole.JUDGE),
        "outsider": _principal(str(uuid4()), UserRole.MEMBER),
    }
    with session_factory.begin() as session:
        for principal in principals.values():
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
    return ProjectService(session_factory), principals, session_factory


def test_member_creates_and_reads_only_an_owned_project(
    project_context: tuple[
        ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory
    ],
) -> None:
    service, principals, _ = project_context
    member = principals[UserRole.MEMBER]
    outsider = principals["outsider"]

    created = service.create_project(member, name="HUST寿命诊断", now=NOW)

    assert created.owner_user_id == member.user_id
    assert created.status is ProjectStatus.ACTIVE
    assert service.list_projects(member) == (created,)
    assert service.get_project(member, created.project_id) == created
    assert service.list_projects(outsider) == ()
    with pytest.raises(ProjectNotFoundError):
        service.get_project(outsider, created.project_id)


def test_admin_can_see_all_projects_but_judge_cannot_create(
    project_context: tuple[
        ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory
    ],
) -> None:
    service, principals, _ = project_context
    member = principals[UserRole.MEMBER]
    created = service.create_project(member, name="储能电芯项目", now=NOW)

    assert service.list_projects(principals[UserRole.ADMIN]) == (created,)
    assert service.list_projects(principals[UserRole.JUDGE]) == ()
    with pytest.raises(ProjectAccessError, match="role"):
        service.create_project(
            principals[UserRole.JUDGE], name="不允许创建", now=NOW
        )


def test_project_service_rejects_blank_names_and_naive_time(
    project_context: tuple[
        ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory
    ],
) -> None:
    service, principals, _ = project_context
    member = principals[UserRole.MEMBER]

    with pytest.raises(ValueError, match="name"):
        service.create_project(member, name="   ", now=NOW)
    with pytest.raises(ValueError, match="timezone"):
        service.create_project(
            member,
            name="valid",
            now=datetime(2026, 7, 15, 12, 0),
        )


def test_judge_requires_an_explicit_role_matching_membership(
    project_context: tuple[
        ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory
    ],
) -> None:
    service, principals, session_factory = project_context
    project = service.create_project(
        principals[UserRole.MEMBER], name="评委内置演示", now=NOW
    )
    judge = principals[UserRole.JUDGE]
    membership_id = str(uuid4())
    with session_factory.begin() as session:
        session.add(
            UserProjectRole(
                id=membership_id,
                user_id=judge.user_id,
                project_id=project.project_id,
                role=UserRole.MEMBER.value,
                created_at=NOW,
            )
        )

    assert service.list_projects(judge) == ()
    with session_factory.begin() as session:
        membership = session.get(UserProjectRole, membership_id)
        assert membership is not None
        membership.role = UserRole.JUDGE.value

    assert service.list_projects(judge) == (project,)


def test_project_creation_rolls_back_if_owner_membership_write_fails(
    project_context: tuple[
        ProjectService, dict[UserRole | str, AuthPrincipal], SessionFactory
    ],
) -> None:
    service, principals, session_factory = project_context

    def fail_membership_write(*_args: object) -> None:
        raise RuntimeError("simulated membership persistence failure")

    event.listen(UserProjectRole, "before_insert", fail_membership_write)
    try:
        with pytest.raises(RuntimeError, match="membership persistence"):
            service.create_project(
                principals[UserRole.MEMBER], name="必须回滚", now=NOW
            )
    finally:
        event.remove(UserProjectRole, "before_insert", fail_membership_write)

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(Project)) == 0
