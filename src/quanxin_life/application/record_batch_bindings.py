"""Project-scoped bindings over verified content-addressed canonical batches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from pydantic import Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    VerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import DatasetStatus, ProjectStatus, UserRole, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import Dataset, Project, RecordBatchBinding
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

BINDING_SCHEMA_VERSION = "record-batch-binding-v1"


class RecordBatchBindingAccessError(RuntimeError):
    """Raised when an actor cannot create project-owned batch bindings."""


class RecordBatchBindingNotFoundError(RuntimeError):
    """Hide absent and cross-project batch resources behind one error."""


class RecordBatchBindingStateError(RuntimeError):
    """Raised when dataset, binding or content integrity is not usable."""


class ProjectContextValidator(Protocol):
    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext: ...


class RecordBatchBindingRecord(ContractModel):
    """Public opaque binding record; internal content identifiers are excluded."""

    record_batch_id: str = Field(min_length=1)
    binding_schema_version: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_dataset_id: str = Field(min_length=1)
    dataset_schema_version: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(gt=0)
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def normalize_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("record batch binding timestamp must include a timezone")
        return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _DatasetSnapshot:
    project_id: str
    dataset_id: str
    data_version: str
    schema_version: str


@dataclass(frozen=True, slots=True)
class _BindingSnapshot:
    binding_schema_version: str
    content_batch_id: str
    project_id: str
    dataset_id: str
    source_manifest_sha256: str
    registration_sha256: str
    content_dataset_id: str
    dataset_schema_version: str
    cell_id: str
    cutoff_cycle: int
    data_version: str
    split_version: str
    feature_version: str


class RecordBatchBindingService:
    """Create opaque bindings and resolve them only inside a live project scope."""

    def __init__(
        self,
        session_factory: SessionFactory,
        batch_store: VerifiedEarlyCycleBatchStore,
        *,
        context_validator: ProjectContextValidator,
    ) -> None:
        if not callable(getattr(context_validator, "revalidate", None)):
            raise TypeError("context_validator must revalidate project contexts")
        self._session_factory = session_factory
        self._batch_store = batch_store
        self._context_validator = context_validator

    def register_canonical_csv(
        self,
        principal: AuthPrincipal,
        dataset_id: str,
        payload: bytes,
        registration: CanonicalCsvBatchRegistration,
        *,
        now: datetime,
    ) -> RecordBatchBindingRecord:
        """Register verified bytes, then atomically bind them to one DRAFT dataset."""

        self._require_operator(principal)
        normalized_dataset_id = self._identifier(dataset_id)
        timestamp = self._utc(now)
        validated_registration = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        initial_dataset = self._resolve_upload_dataset(
            principal,
            normalized_dataset_id,
        )
        if initial_dataset.data_version != validated_registration.data_version:
            raise ValueError("registration data_version must match the dataset version")

        content_batch_id = self._batch_store.register_canonical_csv(
            payload,
            registration=validated_registration,
        )
        try:
            content_batch = self._batch_store.resolve_verified_early_cycle_batch(
                content_batch_id
            )
        except (KeyError, ValueError) as exc:
            raise RecordBatchBindingStateError(
                "verified record batch content cannot be resolved after registration"
            ) from exc
        snapshot = self._binding_snapshot(
            dataset=initial_dataset,
            content_batch_id=content_batch_id,
            content_batch=content_batch,
            registration=validated_registration,
        )
        return self._persist_binding(
            principal,
            snapshot=snapshot,
            created_at=timestamp,
        )

    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        """Resolve one binding with live project, dataset and content revalidation."""

        verified_context = self._context_validator.revalidate(context)
        normalized_id = self._binding_identifier(record_batch_id)
        with session_scope(self._session_factory) as session:
            row = session.execute(
                select(RecordBatchBinding, Dataset, Project)
                .join(Dataset, Dataset.id == RecordBatchBinding.dataset_id)
                .join(Project, Project.id == RecordBatchBinding.project_id)
                .where(
                    RecordBatchBinding.id == normalized_id,
                    RecordBatchBinding.project_id == verified_context.project_id,
                    Dataset.project_id == RecordBatchBinding.project_id,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            ).one_or_none()
            if row is None:
                raise RecordBatchBindingNotFoundError("record batch binding was not found")
            binding, dataset, _project = row
            binding_snapshot = self._snapshot_from_model(binding)
            dataset_snapshot = _DatasetSnapshot(
                project_id=dataset.project_id,
                dataset_id=dataset.id,
                data_version=dataset.data_version,
                schema_version=dataset.schema_version,
            )
            try:
                dataset_status = DatasetStatus(dataset.status)
            except ValueError as exc:
                raise RecordBatchBindingStateError(
                    "record batch dataset has an invalid state"
                ) from exc
            if dataset_status is not DatasetStatus.FROZEN:
                raise RecordBatchBindingStateError(
                    "record batch resolution requires a FROZEN dataset"
                )

        if (
            dataset_snapshot.project_id != binding_snapshot.project_id
            or dataset_snapshot.dataset_id != binding_snapshot.dataset_id
            or dataset_snapshot.data_version != binding_snapshot.data_version
            or dataset_snapshot.schema_version
            != binding_snapshot.dataset_schema_version
        ):
            raise RecordBatchBindingStateError(
                "record batch dataset snapshot does not match the binding"
            )
        if binding_snapshot.binding_schema_version != BINDING_SCHEMA_VERSION:
            raise RecordBatchBindingStateError(
                "record batch binding schema version is unsupported"
            )

        try:
            content_batch = self._batch_store.resolve_verified_early_cycle_batch(
                binding_snapshot.content_batch_id
            )
        except (KeyError, ValueError) as exc:
            raise RecordBatchBindingStateError(
                "record batch content failed integrity verification"
            ) from exc
        self._verify_content_snapshot(binding_snapshot, content_batch)
        return VerifiedEarlyCycleBatch.model_validate(
            content_batch.model_copy(
                update={"record_batch_id": normalized_id}
            ).model_dump(mode="json")
        )

    def provision_frozen_binding(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        content_batch_id: str,
        registration: CanonicalCsvBatchRegistration,
        now: datetime,
    ) -> RecordBatchBindingRecord:
        """Create one immutable project binding for verified Feishu content."""

        verified_context = self._context_validator.revalidate(context)
        timestamp = self._utc(now)
        validated_registration = CanonicalCsvBatchRegistration.model_validate(
            registration.model_dump(mode="json")
        )
        try:
            content_batch = self._batch_store.resolve_verified_early_cycle_batch(
                content_batch_id
            )
        except (KeyError, ValueError) as exc:
            raise RecordBatchBindingStateError(
                "verified record batch content cannot be provisioned"
            ) from exc
        registration_sha256 = sha256_canonical(
            validated_registration.model_dump(mode="json")
        )
        dataset_name = (
            f"Feishu upload {validated_registration.metadata.cell_id} "
            f"{validated_registration.metadata.source_sha256[:12]}"
        )[:200]

        with session_scope(self._session_factory) as session:
            project = session.scalar(
                select(Project)
                .where(
                    Project.id == verified_context.project_id,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
                .with_for_update()
            )
            if project is None:
                raise RecordBatchBindingNotFoundError(
                    "record batch project scope was not found"
                )
            existing = tuple(
                session.scalars(
                    select(RecordBatchBinding)
                    .join(Dataset, Dataset.id == RecordBatchBinding.dataset_id)
                    .where(
                        RecordBatchBinding.project_id == verified_context.project_id,
                        RecordBatchBinding.content_batch_id == content_batch_id,
                        RecordBatchBinding.registration_sha256
                        == registration_sha256,
                        Dataset.project_id == verified_context.project_id,
                        Dataset.status == DatasetStatus.FROZEN.value,
                        Dataset.frozen_at.is_not(None),
                        Dataset.data_version == validated_registration.data_version,
                        Dataset.schema_version
                        == validated_registration.metadata.schema_version,
                        RecordBatchBinding.source_manifest_sha256
                        == content_batch.source_manifest_hash,
                        RecordBatchBinding.content_dataset_id
                        == validated_registration.metadata.dataset_id,
                        RecordBatchBinding.dataset_schema_version
                        == validated_registration.metadata.schema_version,
                        RecordBatchBinding.cell_id
                        == validated_registration.metadata.cell_id,
                        RecordBatchBinding.cutoff_cycle
                        == validated_registration.feature_config.cutoff_cycle,
                        RecordBatchBinding.data_version
                        == validated_registration.data_version,
                        RecordBatchBinding.split_version
                        == validated_registration.split_version,
                        RecordBatchBinding.feature_version
                        == validated_registration.feature_config.feature_version,
                    )
                    .order_by(RecordBatchBinding.id)
                ).all()
            )
            if len(existing) > 1:
                raise RecordBatchBindingStateError(
                    "verified content has ambiguous frozen project bindings"
                )
            if existing:
                self._verify_content_snapshot(
                    self._snapshot_from_model(existing[0]),
                    content_batch,
                )
                return self._record(existing[0])

            dataset = Dataset(
                id=str(uuid4()),
                project_id=verified_context.project_id,
                name=dataset_name,
                data_version=validated_registration.data_version,
                schema_version=validated_registration.metadata.schema_version,
                status=DatasetStatus.DRAFT.value,
                manifest_uri=None,
                manifest_sha256=None,
                created_at=timestamp,
                frozen_at=None,
            )
            session.add(dataset)
            session.flush()
            snapshot = self._binding_snapshot(
                dataset=_DatasetSnapshot(
                    project_id=dataset.project_id,
                    dataset_id=dataset.id,
                    data_version=dataset.data_version,
                    schema_version=dataset.schema_version,
                ),
                content_batch_id=content_batch_id,
                content_batch=content_batch,
                registration=validated_registration,
            )
            binding = RecordBatchBinding(
                id=str(uuid4()),
                **asdict(snapshot),
                created_by_user_id=verified_context.actor_user_id,
                created_at=timestamp,
            )
            session.add(binding)
            dataset.status = DatasetStatus.FROZEN.value
            dataset.frozen_at = timestamp
            session.flush()
            return self._record(binding)

    def list_dataset_batches(
        self,
        principal: AuthPrincipal,
        dataset_id: str,
    ) -> tuple[RecordBatchBindingRecord, ...]:
        normalized_dataset_id = self._identifier(dataset_id)
        with session_scope(self._session_factory) as session:
            visible_project_ids = ProjectService.visible_projects_statement(
                principal
            ).with_only_columns(Project.id)
            dataset = session.scalar(
                select(Dataset).where(
                    Dataset.id == normalized_dataset_id,
                    Dataset.project_id.in_(visible_project_ids),
                )
            )
            if dataset is None:
                raise RecordBatchBindingNotFoundError("dataset was not found")
            bindings = tuple(
                session.scalars(
                    select(RecordBatchBinding)
                    .where(RecordBatchBinding.dataset_id == normalized_dataset_id)
                    .order_by(
                        RecordBatchBinding.cell_id,
                        RecordBatchBinding.cutoff_cycle,
                        RecordBatchBinding.created_at,
                        RecordBatchBinding.id,
                    )
                ).all()
            )
            return tuple(self._record(binding) for binding in bindings)

    def _resolve_upload_dataset(
        self,
        principal: AuthPrincipal,
        dataset_id: str,
        *,
        lock: bool = False,
    ) -> _DatasetSnapshot:
        with session_scope(self._session_factory) as session:
            return self._resolve_upload_dataset_in_session(
                session,
                principal,
                dataset_id,
                lock=lock,
            )

    @staticmethod
    def _resolve_upload_dataset_in_session(
        session: Session,
        principal: AuthPrincipal,
        dataset_id: str,
        *,
        lock: bool,
    ) -> _DatasetSnapshot:
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).where(Project.status == ProjectStatus.ACTIVE.value).with_only_columns(
            Project.id
        )
        statement = select(Dataset).where(
            Dataset.id == dataset_id,
            Dataset.project_id.in_(visible_project_ids),
        )
        if lock:
            statement = statement.with_for_update()
        dataset = session.scalar(statement)
        if dataset is None:
            raise RecordBatchBindingNotFoundError(
                "record batch dataset scope was not found"
            )
        try:
            status = DatasetStatus(dataset.status)
        except ValueError as exc:
            raise RecordBatchBindingStateError(
                "record batch dataset has an invalid state"
            ) from exc
        if status is not DatasetStatus.DRAFT:
            raise RecordBatchBindingStateError(
                "record batch upload requires a DRAFT dataset"
            )
        return _DatasetSnapshot(
            project_id=dataset.project_id,
            dataset_id=dataset.id,
            data_version=dataset.data_version,
            schema_version=dataset.schema_version,
        )

    def _persist_binding(
        self,
        principal: AuthPrincipal,
        *,
        snapshot: _BindingSnapshot,
        created_at: datetime,
    ) -> RecordBatchBindingRecord:
        try:
            with session_scope(self._session_factory) as session:
                current_dataset = self._resolve_upload_dataset_in_session(
                    session,
                    principal,
                    snapshot.dataset_id,
                    lock=True,
                )
                if current_dataset != _DatasetSnapshot(
                    project_id=snapshot.project_id,
                    dataset_id=snapshot.dataset_id,
                    data_version=snapshot.data_version,
                    schema_version=snapshot.dataset_schema_version,
                ):
                    raise RecordBatchBindingStateError(
                        "record batch dataset snapshot changed during registration"
                    )
                existing = session.scalar(
                    select(RecordBatchBinding).where(
                        RecordBatchBinding.dataset_id == snapshot.dataset_id,
                        RecordBatchBinding.content_batch_id == snapshot.content_batch_id,
                    )
                )
                if existing is not None:
                    self._require_matching_binding(existing, snapshot)
                    return self._record(existing)
                binding = RecordBatchBinding(
                    id=str(uuid4()),
                    **asdict(snapshot),
                    created_by_user_id=principal.user_id,
                    created_at=created_at,
                )
                session.add(binding)
                session.flush()
                return self._record(binding)
        except IntegrityError:
            with session_scope(self._session_factory) as session:
                self._resolve_upload_dataset_in_session(
                    session,
                    principal,
                    snapshot.dataset_id,
                    lock=True,
                )
                existing = session.scalar(
                    select(RecordBatchBinding).where(
                        RecordBatchBinding.dataset_id == snapshot.dataset_id,
                        RecordBatchBinding.content_batch_id == snapshot.content_batch_id,
                    )
                )
                if existing is None:
                    raise RecordBatchBindingStateError(
                        "record batch binding conflicted during registration"
                    ) from None
                self._require_matching_binding(existing, snapshot)
                return self._record(existing)

    @staticmethod
    def _binding_snapshot(
        *,
        dataset: _DatasetSnapshot,
        content_batch_id: str,
        content_batch: VerifiedEarlyCycleBatch,
        registration: CanonicalCsvBatchRegistration,
    ) -> _BindingSnapshot:
        if content_batch.record_batch_id != content_batch_id:
            raise RecordBatchBindingStateError(
                "verified content store returned a mismatched batch identifier"
            )
        resolved_registration = CanonicalCsvBatchRegistration(
            metadata=content_batch.metadata,
            feature_config=content_batch.feature_config,
            data_version=content_batch.data_version,
            split_version=content_batch.split_version,
            provenance=content_batch.provenance,
        )
        requested_registration_sha256 = sha256_canonical(
            registration.model_dump(mode="json")
        )
        resolved_registration_sha256 = sha256_canonical(
            resolved_registration.model_dump(mode="json")
        )
        if requested_registration_sha256 != resolved_registration_sha256:
            raise RecordBatchBindingStateError(
                "verified content registration does not match the requested registration"
            )
        return _BindingSnapshot(
            binding_schema_version=BINDING_SCHEMA_VERSION,
            content_batch_id=content_batch_id,
            project_id=dataset.project_id,
            dataset_id=dataset.dataset_id,
            source_manifest_sha256=content_batch.source_manifest_hash,
            registration_sha256=resolved_registration_sha256,
            content_dataset_id=content_batch.metadata.dataset_id,
            dataset_schema_version=dataset.schema_version,
            cell_id=content_batch.metadata.cell_id,
            cutoff_cycle=content_batch.feature_config.cutoff_cycle,
            data_version=content_batch.data_version,
            split_version=content_batch.split_version,
            feature_version=content_batch.feature_config.feature_version,
        )

    @staticmethod
    def _snapshot_from_model(binding: RecordBatchBinding) -> _BindingSnapshot:
        return _BindingSnapshot(
            binding_schema_version=binding.binding_schema_version,
            content_batch_id=binding.content_batch_id,
            project_id=binding.project_id,
            dataset_id=binding.dataset_id,
            source_manifest_sha256=binding.source_manifest_sha256,
            registration_sha256=binding.registration_sha256,
            content_dataset_id=binding.content_dataset_id,
            dataset_schema_version=binding.dataset_schema_version,
            cell_id=binding.cell_id,
            cutoff_cycle=binding.cutoff_cycle,
            data_version=binding.data_version,
            split_version=binding.split_version,
            feature_version=binding.feature_version,
        )

    @classmethod
    def _verify_content_snapshot(
        cls,
        binding: _BindingSnapshot,
        content_batch: VerifiedEarlyCycleBatch,
    ) -> None:
        registration = CanonicalCsvBatchRegistration(
            metadata=content_batch.metadata,
            feature_config=content_batch.feature_config,
            data_version=content_batch.data_version,
            split_version=content_batch.split_version,
            provenance=content_batch.provenance,
        )
        actual = _BindingSnapshot(
            binding_schema_version=binding.binding_schema_version,
            content_batch_id=content_batch.record_batch_id,
            project_id=binding.project_id,
            dataset_id=binding.dataset_id,
            source_manifest_sha256=content_batch.source_manifest_hash,
            registration_sha256=sha256_canonical(registration.model_dump(mode="json")),
            content_dataset_id=content_batch.metadata.dataset_id,
            dataset_schema_version=binding.dataset_schema_version,
            cell_id=content_batch.metadata.cell_id,
            cutoff_cycle=content_batch.feature_config.cutoff_cycle,
            data_version=content_batch.data_version,
            split_version=content_batch.split_version,
            feature_version=content_batch.feature_config.feature_version,
        )
        if actual != binding:
            raise RecordBatchBindingStateError(
                "record batch content snapshot does not match the binding"
            )

    @classmethod
    def _require_matching_binding(
        cls,
        existing: RecordBatchBinding,
        expected: _BindingSnapshot,
    ) -> None:
        if cls._snapshot_from_model(existing) != expected:
            raise RecordBatchBindingStateError(
                "record batch binding conflicts with an existing registration"
            )

    @staticmethod
    def _record(binding: RecordBatchBinding) -> RecordBatchBindingRecord:
        if binding.binding_schema_version != BINDING_SCHEMA_VERSION:
            raise RecordBatchBindingStateError(
                "record batch binding schema version is invalid"
            )
        return RecordBatchBindingRecord(
            record_batch_id=binding.id,
            binding_schema_version=binding.binding_schema_version,
            project_id=binding.project_id,
            dataset_id=binding.dataset_id,
            source_manifest_sha256=binding.source_manifest_sha256,
            content_dataset_id=binding.content_dataset_id,
            dataset_schema_version=binding.dataset_schema_version,
            cell_id=binding.cell_id,
            cutoff_cycle=binding.cutoff_cycle,
            data_version=binding.data_version,
            split_version=binding.split_version,
            feature_version=binding.feature_version,
            created_at=binding.created_at,
        )

    @staticmethod
    def _require_operator(principal: AuthPrincipal) -> None:
        if not isinstance(principal, AuthPrincipal):
            raise RecordBatchBindingAccessError("authenticated principal is required")
        if principal.must_change_password:
            raise RecordBatchBindingAccessError("credential change is required")
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise RecordBatchBindingAccessError(
                "role is not allowed to register record batches"
            )

    @staticmethod
    def _identifier(value: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise RecordBatchBindingNotFoundError("record batch scope was not found")
        return normalized

    @classmethod
    def _binding_identifier(cls, value: str) -> str:
        normalized = cls._identifier(value)
        try:
            UUID(normalized)
        except (TypeError, ValueError, AttributeError) as exc:
            raise RecordBatchBindingNotFoundError(
                "record batch binding was not found"
            ) from exc
        return normalized

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("record batch timestamp must include a timezone")
        return value.astimezone(UTC)


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "RecordBatchBindingAccessError",
    "RecordBatchBindingNotFoundError",
    "RecordBatchBindingRecord",
    "RecordBatchBindingService",
    "RecordBatchBindingStateError",
]
