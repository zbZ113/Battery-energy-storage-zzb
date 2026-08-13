"""Project ownership service with fail-closed object-level authorization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from pydantic import Field, field_validator
from sqlalchemy import Select, or_, select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import ProjectStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import Project, UserProjectRole


class ProjectAccessError(RuntimeError):
    """Raised when a role cannot perform the requested project operation."""


class ProjectNotFoundError(RuntimeError):
    """Used for both absent and invisible projects to avoid identifier disclosure."""


class ProjectVisibilitySubject(Protocol):
    @property
    def user_id(self) -> str: ...

    @property
    def role(self) -> UserRole: ...


class ProjectRecord(ContractModel):
    project_id: str = Field(min_length=1)
    owner_user_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("project timestamps must include a timezone")
        return value.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("project timestamp must include a timezone")
    return value.astimezone(UTC)


def _project_record(project: Project) -> ProjectRecord:
    try:
        status = ProjectStatus(project.status)
    except ValueError as exc:
        raise RuntimeError("project has an unsupported persisted status") from exc
    return ProjectRecord(
        project_id=project.id,
        owner_user_id=project.owner_user_id,
        name=project.name,
        status=status,
        created_at=project.created_at,
        updated_at=project.updated_at,
    )


class ProjectService:
    """Create and read projects under global role and per-object membership rules."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def create_project(
        self,
        principal: AuthPrincipal,
        *,
        name: str,
        now: datetime,
    ) -> ProjectRecord:
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise ProjectAccessError("role is not allowed to create projects")
        normalized_name = name.strip() if isinstance(name, str) else ""
        if not normalized_name:
            raise ValueError("project name must not be blank")
        if len(normalized_name) > 200:
            raise ValueError("project name must contain at most 200 characters")
        timestamp = _utc(now)
        project = Project(
            id=str(uuid4()),
            owner_user_id=principal.user_id,
            name=normalized_name,
            status=ProjectStatus.ACTIVE.value,
            created_at=timestamp,
            updated_at=timestamp,
        )
        membership = UserProjectRole(
            id=str(uuid4()),
            user_id=principal.user_id,
            project_id=project.id,
            role=principal.role.value,
            created_at=timestamp,
        )
        with session_scope(self._session_factory) as session:
            session.add_all((project, membership))
            session.flush()
            result = _project_record(project)
        return result

    def list_projects(self, principal: AuthPrincipal) -> tuple[ProjectRecord, ...]:
        with session_scope(self._session_factory) as session:
            statement = self.visible_projects_statement(principal).order_by(
                Project.created_at, Project.id
            )
            projects = tuple(session.scalars(statement).all())
            return tuple(_project_record(project) for project in projects)

    def get_project(self, principal: AuthPrincipal, project_id: str) -> ProjectRecord:
        normalized_id = project_id.strip() if isinstance(project_id, str) else ""
        if not normalized_id:
            raise ProjectNotFoundError("project was not found")
        with session_scope(self._session_factory) as session:
            statement = self.visible_projects_statement(principal).where(
                Project.id == normalized_id
            )
            project = session.scalar(statement)
            if project is None:
                raise ProjectNotFoundError("project was not found")
            return _project_record(project)

    @staticmethod
    def visible_projects_statement(
        principal: ProjectVisibilitySubject,
    ) -> Select[tuple[Project]]:
        """Return the canonical object-visibility query for project-scoped services."""
        statement = select(Project)
        if principal.role is UserRole.ADMIN:
            return statement
        assigned_project_ids = select(UserProjectRole.project_id).where(
            UserProjectRole.user_id == principal.user_id,
            UserProjectRole.role == principal.role.value,
        )
        if principal.role is UserRole.JUDGE:
            return statement.where(Project.id.in_(assigned_project_ids))
        return statement.where(
            or_(
                Project.owner_user_id == principal.user_id,
                Project.id.in_(assigned_project_ids),
            )
        )


__all__ = [
    "ProjectAccessError",
    "ProjectNotFoundError",
    "ProjectRecord",
    "ProjectService",
    "ProjectVisibilitySubject",
]
