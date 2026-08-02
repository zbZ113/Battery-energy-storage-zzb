"""Versioned dataset registration with project-scoped authorization."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import Field, field_validator
from sqlalchemy import select

from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import DatasetStatus, ProjectStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import Dataset, Project


class DatasetAccessError(RuntimeError):
    """Raised when the authenticated role cannot mutate a dataset."""


class DatasetNotFoundError(RuntimeError):
    """Used for absent and invisible objects to avoid identifier disclosure."""


class DatasetStateError(RuntimeError):
    """Raised when persisted dataset state violates the public contract."""


class DatasetRecord(ContractModel):
    dataset_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=200)
    data_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    status: DatasetStatus
    manifest_uri: str | None = None
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    frozen_at: datetime | None = None

    @field_validator("created_at", "frozen_at")
    @classmethod
    def normalize_time(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("dataset timestamps must include a timezone")
        return value.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("dataset timestamp must include a timezone")
    return value.astimezone(UTC)


def _normalized_text(value: str, *, field: str, maximum: int) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"dataset {field} must not be blank")
    if len(normalized) > maximum:
        raise ValueError(f"dataset {field} must contain at most {maximum} characters")
    return normalized


def _dataset_record(dataset: Dataset) -> DatasetRecord:
    try:
        status = DatasetStatus(dataset.status)
    except ValueError as exc:
        raise DatasetStateError("dataset has an unsupported persisted status") from exc
    if status is DatasetStatus.DRAFT and dataset.frozen_at is not None:
        raise DatasetStateError("draft dataset must not have a freeze timestamp")
    if status is DatasetStatus.FROZEN and dataset.frozen_at is None:
        raise DatasetStateError("frozen dataset must have a freeze timestamp")
    return DatasetRecord(
        dataset_id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        data_version=dataset.data_version,
        schema_version=dataset.schema_version,
        status=status,
        manifest_uri=dataset.manifest_uri,
        manifest_sha256=dataset.manifest_sha256,
        created_at=dataset.created_at,
        frozen_at=dataset.frozen_at,
    )


class DatasetService:
    """Create, inspect and freeze dataset versions under project authorization."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def create_dataset(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        name: str,
        data_version: str,
        schema_version: str,
        now: datetime,
    ) -> DatasetRecord:
        self._require_operator(principal)
        normalized_project_id = self._normalized_identifier(project_id)
        timestamp = _utc(now)
        dataset = Dataset(
            id=str(uuid4()),
            project_id=normalized_project_id,
            name=_normalized_text(name, field="name", maximum=200),
            data_version=_normalized_text(
                data_version, field="data_version", maximum=100
            ),
            schema_version=_normalized_text(
                schema_version, field="schema_version", maximum=100
            ),
            status=DatasetStatus.DRAFT.value,
            manifest_uri=None,
            manifest_sha256=None,
            created_at=timestamp,
            frozen_at=None,
        )
        with session_scope(self._session_factory) as session:
            visible = ProjectService.visible_projects_statement(principal).where(
                Project.id == normalized_project_id,
                Project.status == ProjectStatus.ACTIVE.value,
            )
            if session.scalar(visible) is None:
                raise DatasetNotFoundError("project was not found")
            session.add(dataset)
            session.flush()
            return _dataset_record(dataset)

    def get_dataset(
        self, principal: AuthPrincipal, dataset_id: str
    ) -> DatasetRecord:
        normalized_id = self._normalized_identifier(dataset_id)
        with session_scope(self._session_factory) as session:
            visible_project_ids = ProjectService.visible_projects_statement(
                principal
            ).with_only_columns(Project.id)
            statement = select(Dataset).where(
                Dataset.id == normalized_id,
                Dataset.project_id.in_(visible_project_ids),
            )
            dataset = session.scalar(statement)
            if dataset is None:
                raise DatasetNotFoundError("dataset was not found")
            return _dataset_record(dataset)

    def list_project_datasets(
        self,
        principal: AuthPrincipal,
        project_id: str,
    ) -> tuple[DatasetRecord, ...]:
        normalized_project_id = self._normalized_identifier(project_id)
        with session_scope(self._session_factory) as session:
            visible_project = ProjectService.visible_projects_statement(principal).where(
                Project.id == normalized_project_id
            )
            if session.scalar(visible_project) is None:
                raise DatasetNotFoundError("project was not found")
            datasets = tuple(
                session.scalars(
                    select(Dataset)
                    .where(Dataset.project_id == normalized_project_id)
                    .order_by(Dataset.created_at.desc(), Dataset.id.desc())
                ).all()
            )
            return tuple(_dataset_record(dataset) for dataset in datasets)

    def freeze_dataset(
        self,
        principal: AuthPrincipal,
        dataset_id: str,
        *,
        now: datetime,
    ) -> DatasetRecord:
        self._require_operator(principal)
        normalized_id = self._normalized_identifier(dataset_id)
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            visible_project_ids = ProjectService.visible_projects_statement(
                principal
            ).where(Project.status == ProjectStatus.ACTIVE.value).with_only_columns(
                Project.id
            )
            statement = (
                select(Dataset)
                .where(
                    Dataset.id == normalized_id,
                    Dataset.project_id.in_(visible_project_ids),
                )
                .with_for_update()
            )
            dataset = session.scalar(statement)
            if dataset is None:
                raise DatasetNotFoundError("dataset was not found")
            current = _dataset_record(dataset)
            if current.status is DatasetStatus.FROZEN:
                return current
            if current.status is not DatasetStatus.DRAFT:
                raise DatasetStateError("dataset cannot be frozen from its current state")
            dataset.status = DatasetStatus.FROZEN.value
            dataset.frozen_at = timestamp
            session.flush()
            return _dataset_record(dataset)

    @staticmethod
    def _require_operator(principal: AuthPrincipal) -> None:
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise DatasetAccessError("role is not allowed to mutate datasets")

    @staticmethod
    def _normalized_identifier(value: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise DatasetNotFoundError("dataset or project was not found")
        return normalized


__all__ = [
    "DatasetAccessError",
    "DatasetNotFoundError",
    "DatasetRecord",
    "DatasetService",
    "DatasetStateError",
]
