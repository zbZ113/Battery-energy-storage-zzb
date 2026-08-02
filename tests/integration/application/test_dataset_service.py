from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.application.datasets import (
    DatasetAccessError,
    DatasetNotFoundError,
    DatasetService,
    DatasetStateError,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import DatasetStatus, ProjectStatus, UserRole, UserStatus
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import Dataset, Project, User, UserProjectRole

NOW = datetime(2026, 7, 16, 9, 0, tzinfo=UTC)


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        username=f"{user_id}@example.test",
        role=role,
        session_id=str(uuid4()),
        must_change_password=False,
    )


@pytest.fixture
def dataset_context(
    tmp_path: Path,
) -> tuple[
    DatasetService,
    ProjectService,
    dict[UserRole | str, AuthPrincipal],
    SessionFactory,
]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'datasets.sqlite3'}")
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
    project_service = ProjectService(session_factory)
    return (
        DatasetService(session_factory),
        project_service,
        principals,
        session_factory,
    )


def test_member_creates_reads_and_freezes_a_dataset_in_an_owned_project(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, _ = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="HUST lifespan", now=NOW)

    created = service.create_dataset(
        member,
        project_id=project.project_id,
        name="HUST safe parquet",
        data_version="hust-safe-v1",
        schema_version="canonical-cycle-v1",
        now=NOW,
    )

    assert created.project_id == project.project_id
    assert created.status is DatasetStatus.DRAFT
    assert created.frozen_at is None
    assert service.get_dataset(member, created.dataset_id) == created

    frozen = service.freeze_dataset(member, created.dataset_id, now=NOW)
    assert frozen.status is DatasetStatus.FROZEN
    assert frozen.frozen_at == NOW
    assert service.get_dataset(member, created.dataset_id) == frozen
    repeated = service.freeze_dataset(
        member,
        created.dataset_id,
        now=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
    )
    assert repeated == frozen
    assert repeated.frozen_at == NOW


def test_dataset_object_visibility_matches_the_project_role_membership(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, session_factory = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="visible project", now=NOW)
    created = service.create_dataset(
        member,
        project_id=project.project_id,
        name="dataset",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )

    assert service.get_dataset(principals[UserRole.ADMIN], created.dataset_id) == created
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset(principals["outsider"], created.dataset_id)
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset(principals[UserRole.JUDGE], created.dataset_id)

    judge = principals[UserRole.JUDGE]
    with session_factory.begin() as session:
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=judge.user_id,
                project_id=project.project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    assert service.get_dataset(judge, created.dataset_id) == created


def test_project_dataset_catalog_is_visible_and_newest_first(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, session_factory = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="catalog project", now=NOW)
    older = service.create_dataset(
        member,
        project_id=project.project_id,
        name="older dataset",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )
    newer = service.create_dataset(
        member,
        project_id=project.project_id,
        name="newer dataset",
        data_version="v2",
        schema_version="schema-v1",
        now=datetime(2026, 7, 16, 10, 0, tzinfo=UTC),
    )

    assert service.list_project_datasets(member, project.project_id) == (
        newer,
        older,
    )
    assert service.list_project_datasets(
        principals[UserRole.ADMIN], project.project_id
    ) == (newer, older)
    with pytest.raises(DatasetNotFoundError):
        service.list_project_datasets(principals["outsider"], project.project_id)

    judge = principals[UserRole.JUDGE]
    with session_factory.begin() as session:
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=judge.user_id,
                project_id=project.project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    assert service.list_project_datasets(judge, project.project_id) == (newer, older)


def test_judge_is_read_only_and_invisible_projects_do_not_leak_identifiers(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, session_factory = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="judge demo", now=NOW)
    judge = principals[UserRole.JUDGE]
    with session_factory.begin() as session:
        session.add(
            UserProjectRole(
                id=str(uuid4()),
                user_id=judge.user_id,
                project_id=project.project_id,
                role=UserRole.JUDGE.value,
                created_at=NOW,
            )
        )
    created = service.create_dataset(
        member,
        project_id=project.project_id,
        name="dataset",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )

    with pytest.raises(DatasetAccessError, match="role"):
        service.create_dataset(
            judge,
            project_id=project.project_id,
            name="forbidden",
            data_version="v2",
            schema_version="schema-v1",
            now=NOW,
        )
    with pytest.raises(DatasetAccessError, match="role"):
        service.freeze_dataset(judge, created.dataset_id, now=NOW)
    with pytest.raises(DatasetNotFoundError):
        service.create_dataset(
            principals["outsider"],
            project_id=project.project_id,
            name="hidden",
            data_version="v2",
            schema_version="schema-v1",
            now=NOW,
        )


def test_dataset_rejects_blank_versions_naive_time_and_unsupported_persisted_status(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, session_factory = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="validation", now=NOW)

    for field, values in (
        ("name", {"name": " "}),
        ("data_version", {"data_version": " "}),
        ("schema_version", {"schema_version": " "}),
    ):
        payload = {
            "name": "dataset",
            "data_version": "v1",
            "schema_version": "schema-v1",
            **values,
        }
        with pytest.raises(ValueError, match=field):
            service.create_dataset(
                member,
                project_id=project.project_id,
                now=NOW,
                **payload,
            )
    with pytest.raises(ValueError, match="timezone"):
        service.create_dataset(
            member,
            project_id=project.project_id,
            name="dataset",
            data_version="v1",
            schema_version="schema-v1",
            now=datetime(2026, 7, 16, 9, 0),
        )

    created = service.create_dataset(
        member,
        project_id=project.project_id,
        name="dataset",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )
    with session_factory.begin() as session:
        row = session.get(Dataset, created.dataset_id)
        assert row is not None
        row.status = "CORRUPTED"
    with pytest.raises(DatasetStateError, match="unsupported"):
        service.get_dataset(member, created.dataset_id)


def test_dataset_rejects_inconsistent_freeze_state_and_archived_project_mutations(
    dataset_context: tuple[
        DatasetService,
        ProjectService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, projects, principals, session_factory = dataset_context
    member = principals[UserRole.MEMBER]
    project = projects.create_project(member, name="lifecycle", now=NOW)
    created = service.create_dataset(
        member,
        project_id=project.project_id,
        name="dataset",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )

    with session_factory.begin() as session:
        dataset = session.get(Dataset, created.dataset_id)
        assert dataset is not None
        dataset.frozen_at = NOW
    with pytest.raises(DatasetStateError, match="freeze timestamp"):
        service.get_dataset(member, created.dataset_id)

    with session_factory.begin() as session:
        dataset = session.get(Dataset, created.dataset_id)
        project_row = session.get(Project, project.project_id)
        assert dataset is not None
        assert project_row is not None
        dataset.frozen_at = None
        project_row.status = ProjectStatus.ARCHIVED.value

    with pytest.raises(DatasetNotFoundError):
        service.freeze_dataset(member, created.dataset_id, now=NOW)
    with pytest.raises(DatasetNotFoundError):
        service.create_dataset(
            member,
            project_id=project.project_id,
            name="new version",
            data_version="v2",
            schema_version="schema-v1",
            now=NOW,
        )
