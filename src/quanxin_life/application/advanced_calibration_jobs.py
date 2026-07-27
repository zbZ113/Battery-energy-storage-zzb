"""Application contracts for durable Advanced calibration materialization jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.sql import Select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProjectStatus,
    UserRole,
    sha256_canonical,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    ModelRouteActivationStreamHead,
    Project,
    User,
)

if TYPE_CHECKING:
    from quanxin_life.application.advanced_calibration_materialization import (
        AdvancedCalibrationMaterializationRequest,
        AdvancedCalibrationSampleProducer,
    )
    from quanxin_life.application.invocation_context import (
        ProjectInvocationContextService,
        VerifiedProjectInvocationContext,
    )
    from quanxin_life.audit.project_ledger import (
        AtomicProjectResultMaterializer,
        MaterializedProjectResult,
    )

Clock = Callable[[], datetime]
IdFactory = Callable[[], str]
TokenFactory = Callable[[], str]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_uuid() -> str:
    return str(uuid4())


def _new_claim_token() -> str:
    return uuid4().hex + uuid4().hex


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationDispatchReceipt:
    """Identity-only acknowledgement returned after queue dispatch."""

    materialization_id: str
    task_id: str


class AdvancedCalibrationMaterializationError(RuntimeError):
    """Raised when a persisted calibration materialization cannot be executed."""


class AdvancedCalibrationMaterializationBusyError(
    AdvancedCalibrationMaterializationError
):
    """Raised so the queue retries after another live worker's claim."""


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationMaterializationPreparation:
    """Path-free runtime and source identity frozen before queue dispatch."""

    data_version: str
    split_version: str
    feature_version: str
    artifact_id: str
    artifact_manifest_sha256: str
    model_version: str
    normalization_statistics_sha256: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str
    source_registration_id: str
    source_identity_sha256: str


class AdvancedCalibrationPreparationResolver(Protocol):
    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
    ) -> AdvancedCalibrationMaterializationPreparation: ...


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationMaterializationRecord:
    """Non-sensitive materialization state exposed to project callers."""

    materialization_id: str
    project_id: str
    task: AdvancedModelTask
    cutoff_cycle: int
    route_role: AdvancedModelRouteRole
    status: AdvancedCalibrationMaterializationStatus
    data_version: str
    split_version: str
    feature_version: str
    artifact_id: str
    artifact_manifest_sha256: str
    normalization_statistics_sha256: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str
    source_registration_id: str
    source_identity_sha256: str
    sample_manifest_sha256: str | None
    sample_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    failure_code: str | None


@dataclass(frozen=True, slots=True)
class _AdvancedCalibrationWorkerClaim:
    materialization_id: str
    project_id: str
    task: AdvancedModelTask
    cutoff_cycle: int
    route_role: AdvancedModelRouteRole
    data_version: str
    split_version: str
    feature_version: str
    artifact_id: str
    artifact_manifest_sha256: str
    normalization_statistics_sha256: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str
    source_registration_id: str
    source_identity_sha256: str
    request_sha256: str
    created_by_user_id: str
    created_by_session_id: str
    claim_token: str
    claim_attempt: int
    claim_lease_expires_at: datetime


class AdvancedCalibrationMaterializationService:
    """Create and query server-owned calibration materialization identities."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        preparation_resolver: AdvancedCalibrationPreparationResolver,
        clock: Clock = _utc_now,
        id_factory: IdFactory = _new_uuid,
    ) -> None:
        self._session_factory = session_factory
        self._preparation_resolver = preparation_resolver
        self._clock = clock
        self._id_factory = id_factory

    def create(
        self,
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
        *,
        idempotency_key: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        self._require_context(context, project_id=request.project_id, admin=True)
        normalized_key = _nonblank(idempotency_key, "idempotency_key")
        request_sha256 = sha256_canonical(request.model_dump(mode="json"))
        idempotency_key_sha256 = sha256_canonical(
            {"idempotency_key": normalized_key}
        )
        existing = self._existing_idempotent_record(
            context,
            request=request,
            idempotency_key_sha256=idempotency_key_sha256,
            request_sha256=request_sha256,
        )
        if existing is not None:
            return existing
        preparation = _preparation(
            self._preparation_resolver.resolve(context, request)
        )
        _require_preparation(request, preparation)
        materialization_id = validate_materialization_id(self._id_factory())
        created_at = _utc(self._clock())

        try:
            with session_scope(self._session_factory) as session:
                project = session.scalar(
                    select(Project)
                    .where(Project.id == request.project_id)
                    .with_for_update()
                )
                head = session.scalar(
                    _head_statement(request).with_for_update()
                )
                self._require_active_scope(project, head, preparation)
                persisted_existing = session.scalar(
                    select(AdvancedCalibrationMaterialization)
                    .where(
                        AdvancedCalibrationMaterialization.project_id
                        == request.project_id,
                        AdvancedCalibrationMaterialization.idempotency_key_sha256
                        == idempotency_key_sha256,
                    )
                    .with_for_update()
                )
                if persisted_existing is not None:
                    return self._idempotent_record(
                        persisted_existing,
                        request_sha256=request_sha256,
                    )
                exact = session.scalar(
                    select(AdvancedCalibrationMaterialization)
                    .where(
                        AdvancedCalibrationMaterialization.project_id
                        == request.project_id,
                        AdvancedCalibrationMaterialization.task
                        == request.task.value,
                        AdvancedCalibrationMaterialization.cutoff_cycle
                        == request.cutoff_cycle,
                        AdvancedCalibrationMaterialization.route_role
                        == request.route_role.value,
                        AdvancedCalibrationMaterialization.decision_event_id
                        == preparation.decision_event_id,
                        AdvancedCalibrationMaterialization.source_registration_id
                        == preparation.source_registration_id,
                    )
                    .with_for_update()
                )
                if exact is not None:
                    return self._exact_record(
                        exact,
                        request_sha256=request_sha256,
                        preparation=preparation,
                    )
                row = AdvancedCalibrationMaterialization(
                    id=materialization_id,
                    project_id=request.project_id,
                    task=request.task.value,
                    cutoff_cycle=request.cutoff_cycle,
                    route_role=request.route_role.value,
                    status=AdvancedCalibrationMaterializationStatus.PENDING.value,
                    data_version=preparation.data_version,
                    split_version=preparation.split_version,
                    feature_version=preparation.feature_version,
                    artifact_id=preparation.artifact_id,
                    artifact_manifest_sha256=(
                        preparation.artifact_manifest_sha256
                    ),
                    normalization_statistics_sha256=(
                        preparation.normalization_statistics_sha256
                    ),
                    decision_event_id=preparation.decision_event_id,
                    ledger_sequence_number=preparation.ledger_sequence_number,
                    ledger_head_sha256=preparation.ledger_head_sha256,
                    source_registration_id=preparation.source_registration_id,
                    source_identity_sha256=preparation.source_identity_sha256,
                    sample_manifest_sha256=None,
                    sample_count=0,
                    idempotency_key_sha256=idempotency_key_sha256,
                    request_sha256=request_sha256,
                    created_by_user_id=context.actor_user_id,
                    created_by_session_id=context.actor_session_id,
                    created_by_role=context.actor_role.value,
                    created_at=created_at,
                    started_at=None,
                    completed_at=None,
                    failure_code=None,
                    claim_token_sha256=None,
                    claim_attempt=0,
                    claim_lease_expires_at=None,
                )
                session.add(row)
                session.flush()
                return _record(row)
        except AdvancedCalibrationMaterializationError:
            raise
        except IntegrityError as exc:
            recovered = self._recover_create(
                context,
                request=request,
                idempotency_key_sha256=idempotency_key_sha256,
                request_sha256=request_sha256,
                preparation=preparation,
            )
            if recovered is not None:
                return recovered
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration materialization conflicts with persisted state"
            ) from exc
        except SQLAlchemyError as exc:
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration materialization persistence failed"
            ) from exc

    def get(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        self._require_context(context, project_id=context.project_id, admin=False)
        normalized_id = validate_materialization_id(materialization_id)
        return self._refresh_record(
            context,
            materialization_id=normalized_id,
        )

    def list(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> tuple[AdvancedCalibrationMaterializationRecord, ...]:
        self._require_context(context, project_id=context.project_id, admin=False)
        with session_scope(self._session_factory) as session:
            materialization_ids = tuple(
                session.scalars(
                    select(AdvancedCalibrationMaterialization.id)
                    .where(
                        AdvancedCalibrationMaterialization.project_id
                        == context.project_id
                    )
                    .order_by(
                        AdvancedCalibrationMaterialization.created_at,
                        AdvancedCalibrationMaterialization.id,
                    )
                )
            )
        return tuple(
            self._refresh_record(
                context,
                materialization_id=materialization_id,
            )
            for materialization_id in materialization_ids
        )

    @staticmethod
    def _require_context(
        context: VerifiedProjectInvocationContext,
        *,
        project_id: str,
        admin: bool,
    ) -> None:
        if (
            context.project_id != project_id
            or context.actor_role
            not in ({UserRole.ADMIN} if admin else {UserRole.ADMIN, UserRole.MEMBER})
        ):
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration project context is not authorized"
            )

    @staticmethod
    def _require_active_scope(
        project: Project | None,
        head: ModelRouteActivationStreamHead | None,
        preparation: AdvancedCalibrationMaterializationPreparation,
    ) -> None:
        if project is None or project.status != ProjectStatus.ACTIVE.value:
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration project is inactive"
            )
        if head is None or not _head_matches(head, preparation):
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration active route changed during preparation"
            )

    @staticmethod
    def _idempotent_record(
        row: AdvancedCalibrationMaterialization,
        *,
        request_sha256: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        if row.request_sha256 != request_sha256:
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration idempotency key conflicts with another request"
            )
        return _record(row)

    @staticmethod
    def _exact_record(
        row: AdvancedCalibrationMaterialization,
        *,
        request_sha256: str,
        preparation: AdvancedCalibrationMaterializationPreparation,
    ) -> AdvancedCalibrationMaterializationRecord:
        if (
            row.request_sha256 != request_sha256
            or _row_preparation_identity(row) != _preparation_identity(preparation)
        ):
            raise AdvancedCalibrationMaterializationError(
                "Advanced calibration route conflicts with another source identity"
            )
        return _record(row)

    def _existing_idempotent_record(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        request: AdvancedCalibrationMaterializationRequest,
        idempotency_key_sha256: str,
        request_sha256: str,
    ) -> AdvancedCalibrationMaterializationRecord | None:
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                select(Project)
                .where(Project.id == request.project_id)
                .with_for_update()
            )
            head = session.scalar(_head_statement(request).with_for_update())
            row = session.scalar(
                select(AdvancedCalibrationMaterialization)
                .where(
                    AdvancedCalibrationMaterialization.project_id
                    == context.project_id,
                    AdvancedCalibrationMaterialization.idempotency_key_sha256
                    == idempotency_key_sha256,
                )
                .with_for_update()
            )
            if row is None:
                return None
            if project is None or project.status != ProjectStatus.ACTIVE.value:
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration project is inactive"
                )
            if (
                row.status
                == AdvancedCalibrationMaterializationStatus.READY.value
                and (head is None or not _head_matches(head, row))
            ):
                row.status = AdvancedCalibrationMaterializationStatus.STALE.value
                row.failure_code = "ACTIVE_ROUTE_CHANGED"
                row.completed_at = _utc(self._clock())
                session.flush()
            return self._idempotent_record(
                row,
                request_sha256=request_sha256,
            )

    def _refresh_record(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationRecord:
        locator = self._record_locator(
            context,
            materialization_id=materialization_id,
        )
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                select(Project)
                .where(Project.id == locator.project_id)
                .with_for_update()
            )
            head = session.scalar(
                _head_statement(locator).with_for_update()
            )
            row = session.scalar(
                select(AdvancedCalibrationMaterialization)
                .where(
                    AdvancedCalibrationMaterialization.id == materialization_id,
                    AdvancedCalibrationMaterialization.project_id
                    == context.project_id,
                )
                .with_for_update()
            )
            if (
                project is None
                or project.status != ProjectStatus.ACTIVE.value
                or row is None
                or _row_coordinates(row) != _row_coordinates(locator)
            ):
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration materialization was not found"
                )
            if (
                row.status
                == AdvancedCalibrationMaterializationStatus.READY.value
                and (head is None or not _head_matches(head, row))
            ):
                row.status = AdvancedCalibrationMaterializationStatus.STALE.value
                row.failure_code = "ACTIVE_ROUTE_CHANGED"
                row.completed_at = _utc(self._clock())
                session.flush()
            return _record(row)

    def _record_locator(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterialization:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(AdvancedCalibrationMaterialization).where(
                    AdvancedCalibrationMaterialization.id == materialization_id,
                    AdvancedCalibrationMaterialization.project_id
                    == context.project_id,
                )
            )
            if row is None:
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration materialization was not found"
                )
            session.expunge(row)
            return row

    def _recover_create(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        request: AdvancedCalibrationMaterializationRequest,
        idempotency_key_sha256: str,
        request_sha256: str,
        preparation: AdvancedCalibrationMaterializationPreparation,
    ) -> AdvancedCalibrationMaterializationRecord | None:
        try:
            with session_scope(self._session_factory) as session:
                project = session.scalar(
                    select(Project)
                    .where(Project.id == request.project_id)
                    .with_for_update()
                )
                head = session.scalar(
                    _head_statement(request).with_for_update()
                )
                self._require_active_scope(project, head, preparation)
                row = session.scalar(
                    select(AdvancedCalibrationMaterialization)
                    .where(
                        AdvancedCalibrationMaterialization.project_id
                        == context.project_id,
                        AdvancedCalibrationMaterialization.idempotency_key_sha256
                        == idempotency_key_sha256,
                    )
                    .with_for_update()
                )
                if row is None:
                    return None
                return self._idempotent_record(
                    row,
                    request_sha256=request_sha256,
                )
        except SQLAlchemyError:
            return None


class AdvancedCalibrationMaterializationWorker:
    """Claim and execute one persisted Advanced calibration materialization."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_service: ProjectInvocationContextService,
        producer: AdvancedCalibrationSampleProducer,
        materializer: AtomicProjectResultMaterializer,
        clock: Clock = _utc_now,
        token_factory: TokenFactory = _new_claim_token,
        lease_ttl: timedelta = timedelta(seconds=90),
    ) -> None:
        if lease_ttl <= timedelta(0):
            raise ValueError("Advanced calibration lease_ttl must be positive")
        self._session_factory = session_factory
        self._context_service = context_service
        self._producer = producer
        self._materializer = materializer
        self._clock = clock
        self._token_factory = token_factory
        self._lease_ttl = lease_ttl

    def execute(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationStatus:
        normalized_id = validate_materialization_id(materialization_id)
        claimed = self._claim(normalized_id)
        if isinstance(claimed, AdvancedCalibrationMaterializationStatus):
            return claimed
        try:
            context = self._reconstruct_context(claimed)
            request = _materialization_request(claimed)
            produced_at = _utc(self._clock())
            samples = self._producer.produce(
                context=context,
                materialization_id=claimed.materialization_id,
                request=request,
                created_at=produced_at,
            )
            model_version = _sample_model_version(
                samples,
                data_version=claimed.data_version,
                feature_version=claimed.feature_version,
            )
            self._commit(claimed, samples=samples, model_version=model_version)
            return AdvancedCalibrationMaterializationStatus.READY
        except Exception as exc:
            try:
                self._mark_failed(claimed, failure_code=type(exc).__name__)
            except AdvancedCalibrationMaterializationError as fence_error:
                raise fence_error from None
            return AdvancedCalibrationMaterializationStatus.FAILED

    def _claim(
        self,
        materialization_id: str,
    ) -> (
        _AdvancedCalibrationWorkerClaim
        | AdvancedCalibrationMaterializationStatus
    ):
        locator = self._locator(materialization_id)
        now = _utc(self._clock())
        claim_token = _nonblank(self._token_factory(), "claim token")
        claim_token_sha256 = sha256_canonical({"claim_token": claim_token})
        claim_lease_expires_at = now + self._lease_ttl
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                select(Project)
                .where(Project.id == locator.project_id)
                .with_for_update()
            )
            head = session.scalar(
                _head_statement(locator).with_for_update()
            )
            row = session.scalar(
                select(AdvancedCalibrationMaterialization)
                .where(AdvancedCalibrationMaterialization.id == materialization_id)
                .with_for_update()
            )
            if (
                project is None
                or project.status != ProjectStatus.ACTIVE.value
                or row is None
                or _row_coordinates(row) != _row_coordinates(locator)
            ):
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration materialization was not found"
                )
            status = _status(row.status)
            if status is AdvancedCalibrationMaterializationStatus.STALE:
                return status
            if status is AdvancedCalibrationMaterializationStatus.READY:
                if head is not None and _head_matches(head, row):
                    return status
                row.status = AdvancedCalibrationMaterializationStatus.STALE.value
                row.failure_code = "ACTIVE_ROUTE_CHANGED"
                row.completed_at = now
                return AdvancedCalibrationMaterializationStatus.STALE
            if (
                status is AdvancedCalibrationMaterializationStatus.RUNNING
                and _lease_is_active(row.claim_lease_expires_at, now)
            ):
                raise AdvancedCalibrationMaterializationBusyError(
                    "Advanced calibration materialization has an active running claim"
                )
            if head is None or not _head_matches(head, row):
                row.status = AdvancedCalibrationMaterializationStatus.FAILED.value
                row.started_at = row.started_at or now
                row.completed_at = now
                row.failure_code = "ACTIVE_ROUTE_CHANGED"
                row.claim_token_sha256 = None
                row.claim_attempt += 1
                row.claim_lease_expires_at = None
                return AdvancedCalibrationMaterializationStatus.FAILED
            if status not in {
                AdvancedCalibrationMaterializationStatus.PENDING,
                AdvancedCalibrationMaterializationStatus.RUNNING,
                AdvancedCalibrationMaterializationStatus.FAILED,
            }:
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration materialization cannot be claimed"
                )
            row.status = AdvancedCalibrationMaterializationStatus.RUNNING.value
            row.started_at = now
            row.completed_at = None
            row.failure_code = None
            row.claim_token_sha256 = claim_token_sha256
            row.claim_attempt += 1
            row.claim_lease_expires_at = claim_lease_expires_at
            session.flush()
            return _claim_record(
                row,
                claim_token=claim_token,
                claim_lease_expires_at=claim_lease_expires_at,
            )

    def _locator(
        self,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterialization:
        with session_scope(self._session_factory) as session:
            row = session.get(AdvancedCalibrationMaterialization, materialization_id)
            if row is None:
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration materialization was not found"
                )
            session.expunge(row)
            return row

    def _reconstruct_context(
        self,
        claim: _AdvancedCalibrationWorkerClaim,
    ) -> VerifiedProjectInvocationContext:
        with session_scope(self._session_factory) as session:
            user = session.get(User, claim.created_by_user_id)
            if user is None or user.role != UserRole.ADMIN.value:
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration ADMIN identity is no longer valid"
                )
            principal = AuthPrincipal(
                user_id=user.id,
                session_id=claim.created_by_session_id,
                username=user.username,
                role=UserRole.ADMIN,
                must_change_password=user.must_change_credential,
            )
        return self._context_service.resolve_http(principal, claim.project_id)

    def _commit(
        self,
        claim: _AdvancedCalibrationWorkerClaim,
        *,
        samples: tuple[MaterializedProjectResult, ...],
        model_version: str,
    ) -> None:
        from quanxin_life.audit.project_ledger import ProjectMaterializationCommit

        self._materializer.commit_advanced_calibration_materialization(
            ProjectMaterializationCommit(
                materialization_id=claim.materialization_id,
                project_id=claim.project_id,
                task=claim.task,
                cutoff_cycle=claim.cutoff_cycle,
                route_role=claim.route_role,
                data_version=claim.data_version,
                split_version=claim.split_version,
                feature_version=claim.feature_version,
                artifact_id=claim.artifact_id,
                artifact_manifest_sha256=claim.artifact_manifest_sha256,
                model_version=model_version,
                normalization_statistics_sha256=(
                    claim.normalization_statistics_sha256
                ),
                decision_event_id=claim.decision_event_id,
                ledger_sequence_number=claim.ledger_sequence_number,
                ledger_head_sha256=claim.ledger_head_sha256,
                source_registration_id=claim.source_registration_id,
                source_identity_sha256=claim.source_identity_sha256,
                request_sha256=claim.request_sha256,
                claim_token=claim.claim_token,
                claim_attempt=claim.claim_attempt,
                claim_lease_expires_at=claim.claim_lease_expires_at,
                samples=samples,
            )
        )

    def _mark_failed(
        self,
        claim: _AdvancedCalibrationWorkerClaim,
        *,
        failure_code: str,
    ) -> None:
        now = _utc(self._clock())
        safe_code = _failure_code(failure_code)
        expected_claim_sha256 = sha256_canonical(
            {"claim_token": claim.claim_token}
        )
        locator = self._locator(claim.materialization_id)
        with session_scope(self._session_factory) as session:
            session.scalar(
                select(Project)
                .where(Project.id == locator.project_id)
                .with_for_update()
            )
            session.scalar(_head_statement(locator).with_for_update())
            row = session.scalar(
                select(AdvancedCalibrationMaterialization)
                .where(
                    AdvancedCalibrationMaterialization.id
                    == claim.materialization_id
                )
                .with_for_update()
            )
            if (
                row is None
                or row.status
                != AdvancedCalibrationMaterializationStatus.RUNNING.value
                or row.claim_token_sha256 != expected_claim_sha256
                or row.claim_attempt != claim.claim_attempt
                or row.claim_lease_expires_at is None
                or _database_utc(row.claim_lease_expires_at)
                != claim.claim_lease_expires_at
                or not _lease_is_active(row.claim_lease_expires_at, now)
            ):
                raise AdvancedCalibrationMaterializationError(
                    "Advanced calibration failure claim fence is stale"
                )
            row.status = AdvancedCalibrationMaterializationStatus.FAILED.value
            row.completed_at = now
            row.failure_code = safe_code
            row.claim_token_sha256 = None
            row.claim_lease_expires_at = None


def validate_materialization_id(materialization_id: str) -> str:
    """Require the canonical lowercase UUID stored by the business database."""

    try:
        parsed = UUID(materialization_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "materialization_id must be a canonical UUID"
        ) from exc
    if str(parsed) != materialization_id:
        raise ValueError(
            "materialization_id must be a canonical lowercase UUID"
        )
    return materialization_id


def _preparation(value: object) -> AdvancedCalibrationMaterializationPreparation:
    try:
        source = cast(Any, value)
        return AdvancedCalibrationMaterializationPreparation(
            data_version=source.data_version,
            split_version=source.split_version,
            feature_version=source.feature_version,
            artifact_id=source.artifact_id,
            artifact_manifest_sha256=source.artifact_manifest_sha256,
            model_version=source.model_version,
            normalization_statistics_sha256=(
                source.normalization_statistics_sha256
            ),
            decision_event_id=source.decision_event_id,
            ledger_sequence_number=source.ledger_sequence_number,
            ledger_head_sha256=source.ledger_head_sha256,
            source_registration_id=source.source_registration_id,
            source_identity_sha256=source.source_identity_sha256,
        )
    except (AttributeError, TypeError) as exc:
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration preparation identity is invalid"
        ) from exc


def _require_preparation(
    request: AdvancedCalibrationMaterializationRequest,
    preparation: AdvancedCalibrationMaterializationPreparation,
) -> None:
    string_fields = (
        preparation.data_version,
        preparation.split_version,
        preparation.feature_version,
        preparation.model_version,
        preparation.source_registration_id,
    )
    sha_fields = (
        preparation.artifact_manifest_sha256,
        preparation.normalization_statistics_sha256,
        preparation.ledger_head_sha256,
        preparation.source_identity_sha256,
    )
    if (
        any(not isinstance(value, str) or not value.strip() for value in string_fields)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in sha_fields
        )
        or preparation.ledger_sequence_number <= 0
        or preparation.source_registration_id != request.source_registration_id
    ):
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration preparation identity is invalid"
        )
    validate_materialization_id(preparation.artifact_id)
    validate_materialization_id(preparation.decision_event_id)


def _head_statement(
    value: AdvancedCalibrationMaterializationRequest
    | AdvancedCalibrationMaterialization,
) -> Select[tuple[ModelRouteActivationStreamHead]]:
    return select(ModelRouteActivationStreamHead).where(
        ModelRouteActivationStreamHead.project_id == value.project_id,
        ModelRouteActivationStreamHead.task
        == (
            value.task.value
            if isinstance(value.task, AdvancedModelTask)
            else value.task
        ),
        ModelRouteActivationStreamHead.cutoff_cycle == value.cutoff_cycle,
        ModelRouteActivationStreamHead.route_role
        == (
            value.route_role.value
            if isinstance(value.route_role, AdvancedModelRouteRole)
            else value.route_role
        ),
    )


def _head_matches(
    head: ModelRouteActivationStreamHead,
    value: AdvancedCalibrationMaterializationPreparation
    | AdvancedCalibrationMaterialization,
) -> bool:
    return (
        head.head_event_id == value.decision_event_id
        and head.head_sequence == value.ledger_sequence_number
        and head.head_event_sha256 == value.ledger_head_sha256
    )


def _row_preparation_identity(
    row: AdvancedCalibrationMaterialization,
) -> tuple[object, ...]:
    return (
        row.data_version,
        row.split_version,
        row.feature_version,
        row.artifact_id,
        row.artifact_manifest_sha256,
        row.normalization_statistics_sha256,
        row.decision_event_id,
        row.ledger_sequence_number,
        row.ledger_head_sha256,
        row.source_registration_id,
        row.source_identity_sha256,
    )


def _preparation_identity(
    preparation: AdvancedCalibrationMaterializationPreparation,
) -> tuple[object, ...]:
    return (
        preparation.data_version,
        preparation.split_version,
        preparation.feature_version,
        preparation.artifact_id,
        preparation.artifact_manifest_sha256,
        preparation.normalization_statistics_sha256,
        preparation.decision_event_id,
        preparation.ledger_sequence_number,
        preparation.ledger_head_sha256,
        preparation.source_registration_id,
        preparation.source_identity_sha256,
    )


def _record(
    row: AdvancedCalibrationMaterialization,
) -> AdvancedCalibrationMaterializationRecord:
    return AdvancedCalibrationMaterializationRecord(
        materialization_id=row.id,
        project_id=row.project_id,
        task=AdvancedModelTask(row.task),
        cutoff_cycle=row.cutoff_cycle,
        route_role=AdvancedModelRouteRole(row.route_role),
        status=_status(row.status),
        data_version=row.data_version,
        split_version=row.split_version,
        feature_version=row.feature_version,
        artifact_id=row.artifact_id,
        artifact_manifest_sha256=row.artifact_manifest_sha256,
        normalization_statistics_sha256=row.normalization_statistics_sha256,
        decision_event_id=row.decision_event_id,
        ledger_sequence_number=row.ledger_sequence_number,
        ledger_head_sha256=row.ledger_head_sha256,
        source_registration_id=row.source_registration_id,
        source_identity_sha256=row.source_identity_sha256,
        sample_manifest_sha256=row.sample_manifest_sha256,
        sample_count=row.sample_count,
        created_at=_database_utc(row.created_at),
        started_at=(
            _database_utc(row.started_at)
            if row.started_at is not None
            else None
        ),
        completed_at=(
            _database_utc(row.completed_at)
            if row.completed_at is not None
            else None
        ),
        failure_code=row.failure_code,
    )


def _claim_record(
    row: AdvancedCalibrationMaterialization,
    *,
    claim_token: str,
    claim_lease_expires_at: datetime,
) -> _AdvancedCalibrationWorkerClaim:
    return _AdvancedCalibrationWorkerClaim(
        materialization_id=row.id,
        project_id=row.project_id,
        task=AdvancedModelTask(row.task),
        cutoff_cycle=row.cutoff_cycle,
        route_role=AdvancedModelRouteRole(row.route_role),
        data_version=row.data_version,
        split_version=row.split_version,
        feature_version=row.feature_version,
        artifact_id=row.artifact_id,
        artifact_manifest_sha256=row.artifact_manifest_sha256,
        normalization_statistics_sha256=row.normalization_statistics_sha256,
        decision_event_id=row.decision_event_id,
        ledger_sequence_number=row.ledger_sequence_number,
        ledger_head_sha256=row.ledger_head_sha256,
        source_registration_id=row.source_registration_id,
        source_identity_sha256=row.source_identity_sha256,
        request_sha256=row.request_sha256,
        created_by_user_id=row.created_by_user_id,
        created_by_session_id=row.created_by_session_id,
        claim_token=claim_token,
        claim_attempt=row.claim_attempt,
        claim_lease_expires_at=claim_lease_expires_at,
    )


def _materialization_request(
    claim: _AdvancedCalibrationWorkerClaim,
) -> AdvancedCalibrationMaterializationRequest:
    from quanxin_life.application.advanced_calibration_materialization import (
        AdvancedCalibrationMaterializationRequest,
    )

    request = AdvancedCalibrationMaterializationRequest(
        project_id=claim.project_id,
        task=claim.task,
        cutoff_cycle=claim.cutoff_cycle,
        route_role=claim.route_role,
        source_registration_id=claim.source_registration_id,
    )
    if sha256_canonical(request.model_dump(mode="json")) != claim.request_sha256:
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration persisted request identity changed"
        )
    return request


def _sample_model_version(
    samples: tuple[MaterializedProjectResult, ...],
    *,
    data_version: str,
    feature_version: str,
) -> str:
    if not samples:
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration producer returned no samples"
        )
    identities = {
        (
            sample.result.model_version,
            sample.result.data_version,
            sample.result.feature_version,
        )
        for sample in samples
    }
    if len(identities) != 1:
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration samples do not share one runtime identity"
        )
    model_version, sample_data_version, sample_feature_version = identities.pop()
    if (
        not model_version
        or sample_data_version != data_version
        or sample_feature_version != feature_version
    ):
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration sample runtime identity changed"
        )
    return model_version


def _row_coordinates(
    row: AdvancedCalibrationMaterialization,
) -> tuple[object, ...]:
    return (
        row.project_id,
        row.task,
        row.cutoff_cycle,
        row.route_role,
    )


def _status(value: str) -> AdvancedCalibrationMaterializationStatus:
    try:
        return AdvancedCalibrationMaterializationStatus(value)
    except ValueError as exc:
        raise AdvancedCalibrationMaterializationError(
            "Advanced calibration materialization status is invalid"
        ) from exc


def _lease_is_active(expires_at: datetime | None, now: datetime) -> bool:
    return (
        expires_at is not None
        and _database_utc(expires_at) > now
    )


def _failure_code(value: str) -> str:
    normalized = _nonblank(value, "failure code")
    return normalized[:100]


def _nonblank(value: str, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def _utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("Advanced calibration clock must return an aware datetime")
    return value.astimezone(UTC)


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "AdvancedCalibrationDispatchReceipt",
    "AdvancedCalibrationMaterializationBusyError",
    "AdvancedCalibrationMaterializationError",
    "AdvancedCalibrationMaterializationPreparation",
    "AdvancedCalibrationMaterializationRecord",
    "AdvancedCalibrationMaterializationService",
    "AdvancedCalibrationMaterializationWorker",
    "AdvancedCalibrationPreparationResolver",
    "validate_materialization_id",
]
