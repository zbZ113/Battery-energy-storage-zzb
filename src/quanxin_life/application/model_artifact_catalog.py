"""Project-scoped metadata catalog for freshly verified safe model artifacts.

The catalog never accepts file paths, digests, status values, or model metadata
from HTTP callers.  Those values come from an injected verification source that
must recheck the governed artifact bytes before each registration attempt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, field_validator
from sqlalchemy import select

from quanxin_life.application.model_artifacts import ModelArtifactRegistry
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import ProjectStatus, UserRole
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    ModelArtifact,
    ModelManifest,
    Project,
)


class ModelArtifactCatalogAccessError(RuntimeError):
    """Raised when a role cannot mutate the governed artifact catalog."""


class ModelArtifactCatalogNotFoundError(RuntimeError):
    """Used for absent and invisible projects or artifacts."""


class ModelArtifactCatalogSourceError(RuntimeError):
    """Raised when the configured verification source cannot resolve an artifact."""


class ModelArtifactCatalogStateError(RuntimeError):
    """Raised when persisted metadata no longer matches verified source context."""


class VerifiedModelArtifactMetadata(ContractModel):
    """Allow-listed model context that cannot contain metrics or arbitrary fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_kind: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1, max_length=100)
    data_version: str = Field(min_length=1, max_length=100)
    feature_version: str = Field(min_length=1, max_length=100)
    split_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(ge=0)
    feature_names: tuple[str, ...] = Field(min_length=1)

    @field_validator("feature_names")
    @classmethod
    def feature_names_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("feature_names must not contain blank names")
        if len(value) != len(set(value)):
            raise ValueError("feature_names must be unique")
        return value


class VerifiedModelArtifactRegistration(ContractModel):
    """Freshly verified artifact identity supplied by a trusted local source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    artifact_format: str = Field(min_length=1, max_length=50)
    object_uri: str = Field(
        pattern=r"^verified-model-artifact://[0-9a-f-]{36}/.+$"
    )
    artifact_sha256: Sha256
    model_version: str = Field(min_length=1, max_length=100)
    manifest_uri: str = Field(
        pattern=r"^verified-model-artifact://[0-9a-f-]{36}/manifest$"
    )
    manifest_sha256: Sha256
    metadata: VerifiedModelArtifactMetadata

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc


class VerifiedModelArtifactCatalogSource(Protocol):
    """Read-only boundary that returns only freshly verified artifact metadata."""

    def resolve(
        self,
        artifact_id: str,
    ) -> VerifiedModelArtifactRegistration: ...


class ClassicModelArtifactCatalogSource:
    """Bridge the existing JSON/UBJ registry into catalog-safe metadata."""

    def __init__(self, registry: ModelArtifactRegistry) -> None:
        self._registry = registry

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration:
        verified = self._registry.resolve(artifact_id)
        manifest = verified.manifest
        return VerifiedModelArtifactRegistration(
            artifact_id=manifest.artifact_id,
            artifact_format=manifest.artifact_format.value,
            object_uri=(
                f"verified-model-artifact://{manifest.artifact_id}/"
                f"{manifest.relative_path}"
            ),
            artifact_sha256=manifest.sha256,
            model_version=manifest.model_version,
            manifest_uri=(
                f"verified-model-artifact://{manifest.artifact_id}/manifest"
            ),
            manifest_sha256=sha256_canonical(manifest.model_dump(mode="json")),
            metadata=VerifiedModelArtifactMetadata(
                artifact_kind=manifest.artifact_kind.value,
                dataset_id=manifest.dataset_id,
                data_version=manifest.data_version,
                feature_version=manifest.feature_version,
                split_version=manifest.split_version,
                schema_version=manifest.schema_version,
                cutoff_cycle=manifest.cutoff_cycle,
                feature_names=manifest.feature_names,
            ),
        )


class ModelArtifactCatalogRecord(ContractModel):
    """Project-visible artifact identity without local paths or model metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    project_id: str = Field(min_length=1)
    artifact_kind: str = Field(min_length=1, max_length=100)
    artifact_format: str = Field(min_length=1, max_length=50)
    object_uri: str = Field(min_length=1)
    sha256: Sha256
    status: Literal["VERIFIED"]
    model_manifest_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1, max_length=100)
    manifest_uri: str = Field(min_length=1)
    manifest_sha256: Sha256
    dataset_id: str = Field(min_length=1, max_length=100)
    data_version: str = Field(min_length=1, max_length=100)
    feature_version: str = Field(min_length=1, max_length=100)
    split_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(ge=0)
    feature_names: tuple[str, ...] = Field(min_length=1)
    registered_at: datetime

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("artifact_id must be a UUID string") from exc

    @field_validator("registered_at")
    @classmethod
    def registered_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("registered_at must include a timezone")
        return value.astimezone(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("registered_at must include a timezone")
    return value.astimezone(UTC)


def _identifier(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError("identifier must not be blank")
    return normalized


def _record(
    artifact: ModelArtifact,
    manifest: ModelManifest,
) -> ModelArtifactCatalogRecord:
    try:
        metadata = VerifiedModelArtifactMetadata.model_validate(manifest.metadata_json)
    except ValueError as exc:
        raise ModelArtifactCatalogStateError(
            "persisted model artifact metadata is invalid"
        ) from exc
    if artifact.status != "VERIFIED":
        raise ModelArtifactCatalogStateError(
            "persisted model artifact status is unsupported"
        )
    return ModelArtifactCatalogRecord(
        artifact_id=artifact.id,
        project_id=artifact.project_id,
        artifact_kind=metadata.artifact_kind,
        artifact_format=artifact.artifact_format,
        object_uri=artifact.object_uri,
        sha256=artifact.sha256,
        status="VERIFIED",
        model_manifest_id=manifest.id,
        model_version=manifest.model_version,
        manifest_uri=manifest.manifest_uri,
        manifest_sha256=manifest.manifest_sha256,
        dataset_id=metadata.dataset_id,
        data_version=metadata.data_version,
        feature_version=metadata.feature_version,
        split_version=metadata.split_version,
        schema_version=metadata.schema_version,
        cutoff_cycle=metadata.cutoff_cycle,
        feature_names=metadata.feature_names,
        registered_at=artifact.created_at,
    )


class ModelArtifactCatalogService:
    """Register verified identities and query them under project authorization."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        source: VerifiedModelArtifactCatalogSource,
    ) -> None:
        self._session_factory = session_factory
        self._source = source

    def register_artifact(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        artifact_id: str,
        registered_at: datetime,
    ) -> ModelArtifactCatalogRecord:
        if principal.role is not UserRole.ADMIN:
            raise ModelArtifactCatalogAccessError(
                "only administrators may register model artifacts"
            )
        normalized_project_id = _identifier(project_id)
        normalized_artifact_id = _identifier(artifact_id)
        timestamp = _utc(registered_at)
        source = self._load_source(normalized_artifact_id)

        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == normalized_project_id,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            )
            if project is None:
                raise ModelArtifactCatalogNotFoundError("project was not found")

            existing = session.get(ModelArtifact, source.artifact_id)
            same_digest = session.scalar(
                select(ModelArtifact).where(
                    ModelArtifact.sha256 == source.artifact_sha256
                )
            )
            if existing is None and same_digest is not None:
                raise ModelArtifactCatalogStateError(
                    "verified artifact digest is already bound to another identity"
                )
            if existing is not None:
                manifest = session.scalar(
                    select(ModelManifest).where(
                        ModelManifest.artifact_id == existing.id
                    )
                )
                if manifest is None:
                    raise ModelArtifactCatalogStateError(
                        "persisted model artifact manifest is missing"
                    )
                self._assert_persisted_context(
                    existing,
                    manifest,
                    project_id=normalized_project_id,
                    source=source,
                )
                return _record(existing, manifest)

            artifact = ModelArtifact(
                id=source.artifact_id,
                project_id=normalized_project_id,
                artifact_format=source.artifact_format,
                object_uri=source.object_uri,
                sha256=source.artifact_sha256,
                status="VERIFIED",
                created_at=timestamp,
            )
            manifest = ModelManifest(
                id=str(uuid4()),
                artifact_id=source.artifact_id,
                model_version=source.model_version,
                manifest_uri=source.manifest_uri,
                manifest_sha256=source.manifest_sha256,
                metadata_json=source.metadata.model_dump(mode="json"),
                created_at=timestamp,
            )
            session.add_all((artifact, manifest))
            session.flush()
            return _record(artifact, manifest)

    def list_artifacts(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str | None = None,
        artifact_kind: str | None = None,
        artifact_format: str | None = None,
        dataset_id: str | None = None,
        cutoff_cycle: int | None = None,
    ) -> tuple[ModelArtifactCatalogRecord, ...]:
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        statement = (
            select(ModelArtifact, ModelManifest)
            .join(ModelManifest, ModelManifest.artifact_id == ModelArtifact.id)
            .where(ModelArtifact.project_id.in_(visible_project_ids))
        )
        if project_id is not None:
            statement = statement.where(
                ModelArtifact.project_id == _identifier(project_id)
            )
        if artifact_format is not None:
            statement = statement.where(
                ModelArtifact.artifact_format == _identifier(artifact_format)
            )
        if cutoff_cycle is not None and cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        statement = statement.order_by(ModelArtifact.created_at, ModelArtifact.id)
        with session_scope(self._session_factory) as session:
            records = tuple(
                _record(artifact, manifest)
                for artifact, manifest in session.execute(statement)
            )
        normalized_kind = (
            _identifier(artifact_kind) if artifact_kind is not None else None
        )
        normalized_dataset = (
            _identifier(dataset_id) if dataset_id is not None else None
        )
        return tuple(
            record
            for record in records
            if (normalized_kind is None or record.artifact_kind == normalized_kind)
            and (
                normalized_dataset is None
                or record.dataset_id == normalized_dataset
            )
            and (cutoff_cycle is None or record.cutoff_cycle == cutoff_cycle)
        )

    def get_artifact(
        self,
        principal: AuthPrincipal,
        artifact_id: str,
    ) -> ModelArtifactCatalogRecord:
        normalized_id = _identifier(artifact_id)
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        with session_scope(self._session_factory) as session:
            row = session.execute(
                select(ModelArtifact, ModelManifest)
                .join(ModelManifest, ModelManifest.artifact_id == ModelArtifact.id)
                .where(
                    ModelArtifact.id == normalized_id,
                    ModelArtifact.project_id.in_(visible_project_ids),
                )
            ).one_or_none()
            if row is None:
                raise ModelArtifactCatalogNotFoundError(
                    "model artifact was not found"
                )
            return _record(row[0], row[1])

    def _load_source(
        self,
        artifact_id: str,
    ) -> VerifiedModelArtifactRegistration:
        try:
            source = self._source.resolve(artifact_id)
        except KeyError as exc:
            raise ModelArtifactCatalogSourceError(
                "verified model artifact source is unavailable"
            ) from exc
        if source.artifact_id != artifact_id:
            raise ModelArtifactCatalogSourceError(
                "verified model artifact source returned another identity"
            )
        return source

    @staticmethod
    def _assert_persisted_context(
        artifact: ModelArtifact,
        manifest: ModelManifest,
        *,
        project_id: str,
        source: VerifiedModelArtifactRegistration,
    ) -> None:
        expected_artifact = {
            "project_id": project_id,
            "artifact_format": source.artifact_format,
            "object_uri": source.object_uri,
            "sha256": source.artifact_sha256,
            "status": "VERIFIED",
        }
        expected_manifest = {
            "artifact_id": source.artifact_id,
            "model_version": source.model_version,
            "manifest_uri": source.manifest_uri,
            "manifest_sha256": source.manifest_sha256,
            "metadata_json": source.metadata.model_dump(mode="json"),
        }
        if any(
            getattr(artifact, name) != value
            for name, value in expected_artifact.items()
        ) or any(
            getattr(manifest, name) != value
            for name, value in expected_manifest.items()
        ):
            raise ModelArtifactCatalogStateError(
                "persisted model artifact context does not match verified source"
            )


__all__ = [
    "ClassicModelArtifactCatalogSource",
    "ModelArtifactCatalogAccessError",
    "ModelArtifactCatalogNotFoundError",
    "ModelArtifactCatalogRecord",
    "ModelArtifactCatalogService",
    "ModelArtifactCatalogSourceError",
    "ModelArtifactCatalogStateError",
    "VerifiedModelArtifactCatalogSource",
    "VerifiedModelArtifactMetadata",
    "VerifiedModelArtifactRegistration",
]
