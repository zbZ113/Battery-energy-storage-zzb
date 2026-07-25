"""Project-scoped metadata catalog for freshly verified safe model artifacts.

The catalog never accepts file paths, digests, status values, or model metadata
from HTTP callers.  Those values come from an injected verification source that
must recheck the governed artifact bytes before each registration attempt.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from quanxin_life.application.model_artifacts import ModelArtifactRegistry
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProjectStatus,
    UserRole,
)
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


class AdvancedModelRouteProvenance(ContractModel):
    """One inactive deployment route bound to a representative checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-model-route-provenance-v1"] = (
        "advanced-model-route-provenance-v1"
    )
    task: AdvancedModelTask
    role: AdvancedModelRouteRole
    disposition: Literal["CONDITIONAL"] = "CONDITIONAL"
    family: Literal[
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    ]
    candidate_id: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(gt=0)
    seed: int = Field(gt=0)
    best_epoch: int = Field(gt=0)
    run_id: str = Field(min_length=1, max_length=200)
    checkpoint_manifest_sha256: Sha256
    checkpoint_model_sha256: Sha256
    checkpoint_context_sha256: Sha256


class AdvancedModelArtifactProvenance(ContractModel):
    """Allow-listed deployment evidence without metrics or active-route state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-model-artifact-provenance-v1"] = (
        "advanced-model-artifact-provenance-v1"
    )
    lifecycle_status: Literal["REGISTERED_CANDIDATE"] = "REGISTERED_CANDIDATE"
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    deployment_bundle_manifest_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    final_output_sha256: Sha256
    final_config_sha256: Sha256
    promotion_manifest_sha256: Sha256
    promotion_decisions_sha256: Sha256
    promotion_source_evidence_sha256: Sha256
    selection_manifest_sha256: Sha256
    training_input_bundle_sha256: Sha256
    local_reconstructed_input_bundle_sha256: Sha256
    input_bundle_hashes_match: bool
    candidate_config_sha256: Sha256
    normalization_sha256: Sha256
    target_scaler_context_sha256: Sha256 | None = None
    reference_library_sha256: Sha256 | None = None
    routes: tuple[AdvancedModelRouteProvenance, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def comparison_and_routes_are_consistent(self) -> AdvancedModelArtifactProvenance:
        hashes_match = (
            self.training_input_bundle_sha256
            == self.local_reconstructed_input_bundle_sha256
        )
        if self.input_bundle_hashes_match != hashes_match:
            raise ValueError("input bundle comparison does not match its digests")
        route_keys = {(route.task, route.cutoff_cycle, route.role) for route in self.routes}
        if len(route_keys) != len(self.routes):
            raise ValueError("advanced artifact provenance contains duplicate routes")
        return self


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
    advanced_provenance: AdvancedModelArtifactProvenance | None = None

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
    advanced_provenance: AdvancedModelArtifactProvenance | None = None
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


class AdvancedModelArtifactCatalogBatchRecord(ContractModel):
    """One exact inactive Advanced candidate batch registered to a project."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-model-artifact-catalog-batch-v1"] = (
        "advanced-model-artifact-catalog-batch-v1"
    )
    project_id: str = Field(min_length=1)
    deployment_bundle_manifest_sha256: Sha256
    lifecycle_status: Literal["REGISTERED_CANDIDATE"] = "REGISTERED_CANDIDATE"
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    artifact_count: Literal[15] = 15
    artifacts: tuple[ModelArtifactCatalogRecord, ...] = Field(
        min_length=15,
        max_length=15,
    )

    @model_validator(mode="after")
    def artifacts_are_one_sorted_inactive_batch(
        self,
    ) -> AdvancedModelArtifactCatalogBatchRecord:
        artifact_ids = tuple(item.artifact_id for item in self.artifacts)
        if artifact_ids != tuple(sorted(artifact_ids)) or len(set(artifact_ids)) != 15:
            raise ValueError("Advanced catalog batch artifacts must be unique and sorted")
        for artifact in self.artifacts:
            provenance = artifact.advanced_provenance
            if artifact.project_id != self.project_id or provenance is None:
                raise ValueError("Advanced catalog batch project or provenance differs")
            if (
                provenance.deployment_bundle_manifest_sha256
                != self.deployment_bundle_manifest_sha256
                or provenance.lifecycle_status != self.lifecycle_status
                or provenance.activation_status != self.activation_status
            ):
                raise ValueError("Advanced catalog batch lifecycle context differs")
        return self


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
        advanced_provenance=metadata.advanced_provenance,
        registered_at=artifact.created_at,
    )


def _advanced_batch_record(
    *,
    project_id: str,
    records: list[ModelArtifactCatalogRecord],
) -> AdvancedModelArtifactCatalogBatchRecord:
    ordered = tuple(sorted(records, key=lambda item: item.artifact_id))
    if not ordered or ordered[0].advanced_provenance is None:
        raise ModelArtifactCatalogStateError(
            "persisted Advanced model artifact provenance is missing"
        )
    return AdvancedModelArtifactCatalogBatchRecord(
        project_id=project_id,
        deployment_bundle_manifest_sha256=(
            ordered[0].advanced_provenance.deployment_bundle_manifest_sha256
        ),
        artifacts=ordered,
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
        if source.metadata.advanced_provenance is not None:
            raise ModelArtifactCatalogStateError(
                "Advanced candidates must use atomic batch registration"
            )

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

    def register_advanced_candidates(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        registered_at: datetime,
    ) -> AdvancedModelArtifactCatalogBatchRecord:
        """Register the exact trusted 15-artifact bundle in one SQL transaction."""

        if principal.role is not UserRole.ADMIN:
            raise ModelArtifactCatalogAccessError(
                "only administrators may register model artifacts"
            )
        normalized_project_id = _identifier(project_id)
        timestamp = _utc(registered_at)
        sources = self._load_advanced_sources()
        source_ids = tuple(item.artifact_id for item in sources)
        source_digests = tuple(item.artifact_sha256 for item in sources)

        try:
            with session_scope(self._session_factory) as session:
                project = session.scalar(
                    ProjectService.visible_projects_statement(principal).where(
                        Project.id == normalized_project_id,
                        Project.status == ProjectStatus.ACTIVE.value,
                    )
                )
                if project is None:
                    raise ModelArtifactCatalogNotFoundError("project was not found")

                persisted = tuple(
                    session.scalars(
                        select(ModelArtifact).where(
                            (ModelArtifact.id.in_(source_ids))
                            | (ModelArtifact.sha256.in_(source_digests))
                        )
                    )
                )
                by_id = {item.id: item for item in persisted}
                by_digest = {item.sha256: item for item in persisted}
                existing_ids = tuple(
                    item.id for item in persisted if item.id in source_ids
                )
                manifests = {
                    item.artifact_id: item
                    for item in session.scalars(
                        select(ModelManifest).where(
                            ModelManifest.artifact_id.in_(existing_ids)
                        )
                    )
                }
                records: list[ModelArtifactCatalogRecord] = []
                for source in sources:
                    digest_owner = by_digest.get(source.artifact_sha256)
                    if digest_owner is not None and digest_owner.id != source.artifact_id:
                        raise ModelArtifactCatalogStateError(
                            "verified artifact digest is already bound to another identity"
                        )
                    existing = by_id.get(source.artifact_id)
                    if existing is not None:
                        manifest = manifests.get(existing.id)
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
                        records.append(_record(existing, manifest))
                        continue

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
                    records.append(_record(artifact, manifest))
                session.flush()
                return _advanced_batch_record(
                    project_id=normalized_project_id,
                    records=records,
                )
        except IntegrityError:
            try:
                return self._resolve_persisted_advanced_batch(
                    principal,
                    project_id=normalized_project_id,
                    sources=sources,
                )
            except (
                ModelArtifactCatalogNotFoundError,
                ModelArtifactCatalogStateError,
            ) as recovery_error:
                raise ModelArtifactCatalogStateError(
                    "concurrent Advanced artifact registration conflicted"
                ) from recovery_error

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

    def _load_advanced_sources(
        self,
    ) -> tuple[VerifiedModelArtifactRegistration, ...]:
        resolver = getattr(self._source, "resolve_all", None)
        if not callable(resolver):
            raise ModelArtifactCatalogSourceError(
                "verified Advanced model artifact batch source is unavailable"
            )
        try:
            raw = tuple(
                VerifiedModelArtifactRegistration.model_validate(item)
                for item in resolver()
            )
        except (AttributeError, KeyError, TypeError, ValidationError, ValueError) as exc:
            raise ModelArtifactCatalogSourceError(
                "verified Advanced model artifact batch source is unavailable"
            ) from exc
        sources = tuple(sorted(raw, key=lambda item: item.artifact_id))
        if len(sources) != 15:
            raise ModelArtifactCatalogSourceError(
                "verified Advanced model artifact source must contain exactly 15 artifacts"
            )
        ids = {item.artifact_id for item in sources}
        digests = {item.artifact_sha256 for item in sources}
        provenance = tuple(item.metadata.advanced_provenance for item in sources)
        if len(ids) != 15 or len(digests) != 15 or any(item is None for item in provenance):
            raise ModelArtifactCatalogSourceError(
                "verified Advanced model artifact source is duplicated or incomplete"
            )
        advanced = tuple(item for item in provenance if item is not None)
        bundle_hashes = {
            item.deployment_bundle_manifest_sha256 for item in advanced
        }
        route_keys = {
            (route.task, route.cutoff_cycle, route.role)
            for item in advanced
            for route in item.routes
        }
        route_count = sum(len(item.routes) for item in advanced)
        if (
            len(bundle_hashes) != 1
            or route_count != 15
            or len(route_keys) != 15
            or any(
                item.lifecycle_status != "REGISTERED_CANDIDATE"
                or item.activation_status != "NOT_ACTIVATED"
                for item in advanced
            )
        ):
            raise ModelArtifactCatalogSourceError(
                "verified Advanced model artifact lifecycle matrix is invalid"
            )
        return sources

    def _resolve_persisted_advanced_batch(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        sources: tuple[VerifiedModelArtifactRegistration, ...],
    ) -> AdvancedModelArtifactCatalogBatchRecord:
        source_ids = tuple(item.artifact_id for item in sources)
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == project_id,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            )
            if project is None:
                raise ModelArtifactCatalogNotFoundError("project was not found")
            artifacts = {
                item.id: item
                for item in session.scalars(
                    select(ModelArtifact).where(ModelArtifact.id.in_(source_ids))
                )
            }
            manifests = {
                item.artifact_id: item
                for item in session.scalars(
                    select(ModelManifest).where(
                        ModelManifest.artifact_id.in_(source_ids)
                    )
                )
            }
            if len(artifacts) != 15 or len(manifests) != 15:
                raise ModelArtifactCatalogStateError(
                    "concurrent Advanced artifact batch is incomplete"
                )
            records = []
            for source in sources:
                artifact = artifacts[source.artifact_id]
                manifest = manifests[source.artifact_id]
                self._assert_persisted_context(
                    artifact,
                    manifest,
                    project_id=project_id,
                    source=source,
                )
                records.append(_record(artifact, manifest))
            return _advanced_batch_record(project_id=project_id, records=records)

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
    "AdvancedModelArtifactCatalogBatchRecord",
    "AdvancedModelArtifactProvenance",
    "AdvancedModelRouteProvenance",
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
