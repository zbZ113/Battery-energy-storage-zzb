"""Append-only manual activation and rollback decisions for Advanced routes."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quanxin_life.application.model_artifact_catalog import (
    AdvancedModelRouteProvenance,
    VerifiedModelArtifactCatalogSource,
    VerifiedModelArtifactMetadata,
    VerifiedModelArtifactRegistration,
)
from quanxin_life.application.projects import ProjectService, ProjectVisibilitySubject
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ModelRouteDecisionType,
    ProjectStatus,
    UserRole,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    ModelArtifact,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
)

GENESIS_EVENT_SHA256 = "0" * 64


class ModelRouteActivationAccessError(RuntimeError):
    """Raised when a principal cannot make a model route decision."""


class ModelRouteActivationNotFoundError(RuntimeError):
    """Used for absent or invisible projects, candidates and events."""


class ModelRouteActivationStateError(RuntimeError):
    """Raised when trusted source, persisted state or ledger history conflicts."""


class ModelRouteArtifactSource(Protocol):
    """Freshly verify one governed candidate before every state-changing decision."""

    def resolve(self, artifact_id: str) -> VerifiedModelArtifactRegistration: ...


class ModelRouteActivationEventRecord(ContractModel):
    """One immutable route state transition and its full provenance snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "model-route-activation-event-v1"
    event_id: str
    project_id: str = Field(min_length=1)
    sequence_number: int = Field(gt=0)
    task: AdvancedModelTask
    cutoff_cycle: int = Field(gt=0)
    role: AdvancedModelRouteRole
    decision_type: ModelRouteDecisionType
    artifact_id: str
    artifact_sha256: Sha256
    manifest_sha256: Sha256
    deployment_bundle_manifest_sha256: Sha256
    route_provenance_sha256: Sha256
    rollback_target_event_id: str | None = None
    previous_event_sha256: Sha256
    event_sha256: Sha256
    actor_user_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=2000)
    idempotency_key_sha256: Sha256
    request_sha256: Sha256
    created_at: datetime

    @field_validator("event_id", "artifact_id")
    @classmethod
    def identifiers_are_uuid_strings(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("event and artifact identifiers must be UUID strings") from exc

    @field_validator("rollback_target_event_id")
    @classmethod
    def rollback_target_is_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("rollback target must be a UUID string") from exc

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def decision_route_and_hash_are_consistent(
        self,
    ) -> ModelRouteActivationEventRecord:
        _validate_route(self.task, self.role)
        is_rollback = self.decision_type is ModelRouteDecisionType.ROLLBACK
        if is_rollback != (self.rollback_target_event_id is not None):
            raise ValueError("rollback target does not match decision type")
        unsigned = self.model_dump(mode="json", exclude={"event_sha256"})
        if self.event_sha256 != sha256_canonical(unsigned):
            raise ValueError("activation event SHA-256 does not match")
        return self


class VerifiedActiveModelRoute(ContractModel):
    """Freshly verified active candidate derived from one ledger stream head."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["verified-active-model-route-v1"] = (
        "verified-active-model-route-v1"
    )
    project_id: str = Field(min_length=1)
    task: AdvancedModelTask
    cutoff_cycle: int = Field(gt=0)
    role: AdvancedModelRouteRole
    artifact: VerifiedModelArtifactRegistration
    route_provenance: AdvancedModelRouteProvenance
    route_provenance_sha256: Sha256
    decision_event_id: str
    decision_type: ModelRouteDecisionType
    rollback_target_event_id: str | None = None
    ledger_sequence_number: int = Field(gt=0)
    ledger_head_sha256: Sha256

    @field_validator("decision_event_id", "rollback_target_event_id")
    @classmethod
    def decision_identifiers_are_uuid_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("decision event identifiers must be UUID strings") from exc

    @model_validator(mode="after")
    def route_and_candidate_are_consistent(self) -> VerifiedActiveModelRoute:
        _validate_route(self.task, self.role)
        route = self.route_provenance
        if (route.task, route.cutoff_cycle, route.role) != (
            self.task,
            self.cutoff_cycle,
            self.role,
        ):
            raise ValueError("resolved route provenance does not match route coordinates")
        provenance = self.artifact.metadata.advanced_provenance
        if provenance is None or route not in provenance.routes:
            raise ValueError("resolved artifact does not contain route provenance")
        if self.route_provenance_sha256 != sha256_canonical(
            route.model_dump(mode="json")
        ):
            raise ValueError("resolved route provenance SHA-256 does not match")
        is_rollback = self.decision_type is ModelRouteDecisionType.ROLLBACK
        if is_rollback != (self.rollback_target_event_id is not None):
            raise ValueError("resolved rollback target does not match decision type")
        return self


class ModelRouteActivationService:
    """Append decisions and resolve routes without a materialized artifact projection."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        source: VerifiedModelArtifactCatalogSource | ModelRouteArtifactSource,
    ) -> None:
        self._session_factory = session_factory
        self._source = source

    def activate(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
        artifact_id: str,
        reason: str,
        idempotency_key: str,
        expected_previous_event_sha256: str,
        occurred_at: datetime,
    ) -> ModelRouteActivationEventRecord:
        self._require_admin(principal)
        route = _route(task, cutoff_cycle, role)
        normalized_project = _identifier(project_id, "project_id")
        normalized_artifact = _uuid(artifact_id, "artifact_id")
        normalized_reason = _reason(reason)
        key_hash = _idempotency_hash(idempotency_key)
        expected = _sha256(expected_previous_event_sha256, "expected previous event")
        timestamp = _utc(occurred_at)
        request_sha = _request_sha(
            decision=ModelRouteDecisionType.ACTIVATE,
            actor_user_id=principal.user_id,
            project_id=normalized_project,
            route=route,
            artifact_id=normalized_artifact,
            rollback_target_event_id=None,
            reason=normalized_reason,
            idempotency_key_sha256=key_hash,
            expected_previous_event_sha256=expected,
        )
        try:
            with session_scope(self._session_factory) as session:
                self._active_project(session, principal, normalized_project)
                existing = self._idempotent_event(
                    session,
                    project_id=normalized_project,
                    idempotency_key_sha256=key_hash,
                    request_sha256=request_sha,
                )
                if existing is not None:
                    self._verified_project_ledger(session, normalized_project)
                    self._verified_candidate(
                        session,
                        project_id=normalized_project,
                        artifact_id=existing.artifact_id,
                        route=route,
                    )
                    return existing
                history = self._verified_route_history(
                    session,
                    project_id=normalized_project,
                    route=route,
                    for_update=True,
                )
                _require_expected_head(history, expected)
                source, persisted = self._verified_candidate(
                    session,
                    project_id=normalized_project,
                    artifact_id=normalized_artifact,
                    route=route,
                )
                if history and history[-1].artifact_id == normalized_artifact:
                    raise ModelRouteActivationStateError(
                        "candidate is already the latest route decision"
                    )
                event = _event(
                    project_id=normalized_project,
                    route=route,
                    decision=ModelRouteDecisionType.ACTIVATE,
                    source=source,
                    persisted=persisted,
                    rollback_target_event_id=None,
                    history=history,
                    actor_user_id=principal.user_id,
                    reason=normalized_reason,
                    idempotency_key_sha256=key_hash,
                    request_sha256=request_sha,
                    created_at=timestamp,
                )
                session.add(_event_row(event))
                session.flush()
                self._advance_stream_head(session, event, history)
                return event
        except IntegrityError as exc:
            return self._recover_idempotent_conflict(
                principal,
                normalized_project,
                route,
                key_hash,
                request_sha,
                exc,
            )

    def rollback(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
        target_activation_event_id: str,
        reason: str,
        idempotency_key: str,
        expected_previous_event_sha256: str,
        occurred_at: datetime,
    ) -> ModelRouteActivationEventRecord:
        self._require_admin(principal)
        route = _route(task, cutoff_cycle, role)
        normalized_project = _identifier(project_id, "project_id")
        target_id = _uuid(target_activation_event_id, "target_activation_event_id")
        normalized_reason = _reason(reason)
        key_hash = _idempotency_hash(idempotency_key)
        expected = _sha256(expected_previous_event_sha256, "expected previous event")
        timestamp = _utc(occurred_at)
        request_sha = _request_sha(
            decision=ModelRouteDecisionType.ROLLBACK,
            actor_user_id=principal.user_id,
            project_id=normalized_project,
            route=route,
            artifact_id=None,
            rollback_target_event_id=target_id,
            reason=normalized_reason,
            idempotency_key_sha256=key_hash,
            expected_previous_event_sha256=expected,
        )
        try:
            with session_scope(self._session_factory) as session:
                self._active_project(session, principal, normalized_project)
                existing = self._idempotent_event(
                    session,
                    project_id=normalized_project,
                    idempotency_key_sha256=key_hash,
                    request_sha256=request_sha,
                )
                if existing is not None:
                    self._verified_project_ledger(session, normalized_project)
                    self._verified_candidate(
                        session,
                        project_id=normalized_project,
                        artifact_id=existing.artifact_id,
                        route=route,
                    )
                    return existing
                history = self._verified_route_history(
                    session,
                    project_id=normalized_project,
                    route=route,
                    for_update=True,
                )
                _require_expected_head(history, expected)
                target = next(
                    (item for item in history if item.event_id == target_id),
                    None,
                )
                if (
                    target is None
                    or target.decision_type is not ModelRouteDecisionType.ACTIVATE
                    or target is history[-1]
                    or target.artifact_id == history[-1].artifact_id
                ):
                    raise ModelRouteActivationStateError(
                        "rollback target must be an earlier effective ACTIVATE event"
                    )
                source, persisted = self._verified_candidate(
                    session,
                    project_id=normalized_project,
                    artifact_id=target.artifact_id,
                    route=route,
                )
                event = _event(
                    project_id=normalized_project,
                    route=route,
                    decision=ModelRouteDecisionType.ROLLBACK,
                    source=source,
                    persisted=persisted,
                    rollback_target_event_id=target.event_id,
                    history=history,
                    actor_user_id=principal.user_id,
                    reason=normalized_reason,
                    idempotency_key_sha256=key_hash,
                    request_sha256=request_sha,
                    created_at=timestamp,
                )
                session.add(_event_row(event))
                session.flush()
                self._advance_stream_head(session, event, history)
                return event
        except IntegrityError as exc:
            return self._recover_idempotent_conflict(
                principal,
                normalized_project,
                route,
                key_hash,
                request_sha,
                exc,
            )

    def list_events(
        self,
        principal: ProjectVisibilitySubject,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> tuple[ModelRouteActivationEventRecord, ...]:
        route = _route(task, cutoff_cycle, role)
        normalized_project = _identifier(project_id, "project_id")
        with session_scope(self._session_factory) as session:
            if session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == normalized_project
                )
            ) is None:
                raise ModelRouteActivationNotFoundError("project was not found")
            self._verified_project_ledger(session, normalized_project)
            return self._verified_route_history(
                session,
                project_id=normalized_project,
                route=route,
            )

    def list_verified_active_model_routes(
        self,
        principal: ProjectVisibilitySubject,
        *,
        project_id: str,
    ) -> tuple[VerifiedActiveModelRoute, ...]:
        """List one path-free, freshly verified head for every active route."""

        normalized_project = _identifier(project_id, "project_id")
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == normalized_project,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            )
            if project is None:
                raise ModelRouteActivationNotFoundError(
                    "project was not found"
                )
            snapshot = _latest_route_snapshot(
                self._verified_project_ledger(session, normalized_project)
            )
        routes = tuple(
            self.resolve_verified_active_model_route(
                principal,
                project_id=normalized_project,
                task=task,
                cutoff_cycle=cutoff_cycle,
                role=role,
            )
            for task, cutoff_cycle, role, _, _, _ in snapshot
        )
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == normalized_project,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            )
            if project is None or _latest_route_snapshot(
                self._verified_project_ledger(session, normalized_project)
            ) != snapshot:
                raise ModelRouteActivationStateError(
                    "active model route collection changed during resolution"
                )
        return routes

    def resolve_verified_active_model_route(
        self,
        principal: ProjectVisibilitySubject,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedActiveModelRoute:
        """Resolve one exact active route without loading model tensors."""

        route = _route(task, cutoff_cycle, role)
        normalized_project = _identifier(project_id, "project_id")
        with session_scope(self._session_factory) as session:
            project = session.scalar(
                ProjectService.visible_projects_statement(principal).where(
                    Project.id == normalized_project,
                    Project.status == ProjectStatus.ACTIVE.value,
                )
            )
            if project is None:
                raise ModelRouteActivationNotFoundError("project was not found")
            ledger = self._verified_project_ledger(session, normalized_project)
            history = tuple(
                event
                for event in ledger
                if (event.task, event.cutoff_cycle, event.role) == route
            )
            if not history:
                raise ModelRouteActivationNotFoundError(
                    "active model route was not found"
                )
            decision = history[-1]
            source, _ = self._verified_candidate(
                session,
                project_id=normalized_project,
                artifact_id=decision.artifact_id,
                route=route,
            )
            provenance = source.metadata.advanced_provenance
            if provenance is None:  # pragma: no cover - guarded by candidate validation
                raise ModelRouteActivationStateError(
                    "resolved Advanced candidate provenance is missing"
                )
            matching = next(
                item
                for item in provenance.routes
                if (item.task, item.cutoff_cycle, item.role) == route
            )
            expected_snapshot = (
                source.artifact_id,
                source.artifact_sha256,
                source.manifest_sha256,
                provenance.deployment_bundle_manifest_sha256,
                sha256_canonical(matching.model_dump(mode="json")),
            )
            if _candidate_snapshot(decision) != expected_snapshot:
                raise ModelRouteActivationStateError(
                    "active route event snapshot differs from verified source"
                )
            resolved = VerifiedActiveModelRoute(
                project_id=normalized_project,
                task=route[0],
                cutoff_cycle=route[1],
                role=route[2],
                artifact=source,
                route_provenance=matching,
                route_provenance_sha256=decision.route_provenance_sha256,
                decision_event_id=decision.event_id,
                decision_type=decision.decision_type,
                rollback_target_event_id=decision.rollback_target_event_id,
                ledger_sequence_number=decision.sequence_number,
                ledger_head_sha256=decision.event_sha256,
            )

        with session_scope(self._session_factory) as session:
            self._active_project(session, principal, normalized_project)
            ledger = self._verified_project_ledger(session, normalized_project)
            latest = next(
                (
                    event
                    for event in reversed(ledger)
                    if (event.task, event.cutoff_cycle, event.role) == route
                ),
                None,
            )
            if (
                latest is None
                or latest.event_id != resolved.decision_event_id
                or latest.event_sha256 != resolved.ledger_head_sha256
                or latest.sequence_number != resolved.ledger_sequence_number
            ):
                raise ModelRouteActivationStateError(
                    "active model route changed during resolution"
                )
            artifact = session.get(ModelArtifact, resolved.artifact.artifact_id)
            manifest = session.scalar(
                select(ModelManifest).where(
                    ModelManifest.artifact_id == resolved.artifact.artifact_id
                )
            )
            if (
                artifact is None
                or manifest is None
                or artifact.project_id != normalized_project
            ):
                raise ModelRouteActivationStateError(
                    "active model route candidate changed during resolution"
                )
            _assert_persisted_candidate(artifact, manifest, resolved.artifact)
            return resolved

    def _verified_candidate(
        self,
        session: Session,
        *,
        project_id: str,
        artifact_id: str,
        route: tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
    ) -> tuple[VerifiedModelArtifactRegistration, tuple[ModelArtifact, ModelManifest]]:
        try:
            source = VerifiedModelArtifactRegistration.model_validate(
                self._source.resolve(artifact_id)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRouteActivationStateError(
                "verified Advanced candidate source is unavailable"
            ) from exc
        provenance = source.metadata.advanced_provenance
        if provenance is None:
            raise ModelRouteActivationStateError("candidate is not an Advanced artifact")
        matching_routes = tuple(
            item
            for item in provenance.routes
            if (AdvancedModelTask(item.task), item.cutoff_cycle, AdvancedModelRouteRole(item.role))
            == route
        )
        if len(matching_routes) != 1:
            raise ModelRouteActivationStateError(
                "candidate provenance does not declare the requested route"
            )
        row = session.get(ModelArtifact, artifact_id)
        manifest = session.scalar(
            select(ModelManifest).where(ModelManifest.artifact_id == artifact_id)
        )
        if row is None or manifest is None or row.project_id != project_id:
            raise ModelRouteActivationNotFoundError("registered candidate was not found")
        _assert_persisted_candidate(row, manifest, source)
        return source, (row, manifest)

    def _verified_project_ledger(
        self,
        session: Session,
        project_id: str,
    ) -> tuple[ModelRouteActivationEventRecord, ...]:
        rows = tuple(
            session.scalars(
                select(ModelRouteActivationEvent)
                .where(ModelRouteActivationEvent.project_id == project_id)
                .order_by(
                    ModelRouteActivationEvent.task,
                    ModelRouteActivationEvent.cutoff_cycle,
                    ModelRouteActivationEvent.route_role,
                    ModelRouteActivationEvent.stream_sequence,
                )
            )
        )
        heads = {
            (
                AdvancedModelTask(row.task),
                row.cutoff_cycle,
                AdvancedModelRouteRole(row.route_role),
            ): row
            for row in session.scalars(
                select(ModelRouteActivationStreamHead).where(
                    ModelRouteActivationStreamHead.project_id == project_id
                )
            )
        }
        by_route: dict[
            tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
            list[ModelRouteActivationEventRecord],
        ] = {}
        for row in rows:
            record = _record(row)
            key = (record.task, record.cutoff_cycle, record.role)
            by_route.setdefault(key, []).append(record)
        verified: list[ModelRouteActivationEventRecord] = []
        for history in by_route.values():
            _verify_chain(history)
            route = (history[0].task, history[0].cutoff_cycle, history[0].role)
            head = heads.pop(route, None)
            if head is None or not _head_matches(head, history[-1]):
                raise ModelRouteActivationStateError(
                    "model route activation stream head is invalid"
                )
            verified.extend(history)
        if heads:
            raise ModelRouteActivationStateError(
                "model route activation stream head has no event history"
            )
        return tuple(verified)

    @staticmethod
    def _advance_stream_head(
        session: Session,
        event: ModelRouteActivationEventRecord,
        history: Sequence[ModelRouteActivationEventRecord],
    ) -> None:
        head = session.get(
            ModelRouteActivationStreamHead,
            {
                "project_id": event.project_id,
                "task": event.task.value,
                "cutoff_cycle": event.cutoff_cycle,
                "route_role": event.role.value,
            },
        )
        if not history:
            if head is not None:
                raise ModelRouteActivationStateError(
                    "model route stream head exists without history"
                )
            session.add(
                ModelRouteActivationStreamHead(
                    project_id=event.project_id,
                    task=event.task.value,
                    cutoff_cycle=event.cutoff_cycle,
                    route_role=event.role.value,
                    head_event_id=event.event_id,
                    head_sequence=event.sequence_number,
                    head_event_sha256=event.event_sha256,
                    updated_at=event.created_at,
                )
            )
            session.flush()
            return
        if head is None or not _head_matches(head, history[-1]):
            raise ModelRouteActivationStateError(
                "model route stream head does not match prior history"
            )
        head.head_event_id = event.event_id
        head.head_sequence = event.sequence_number
        head.head_event_sha256 = event.event_sha256
        head.updated_at = event.created_at
        session.flush()

    def _verified_route_history(
        self,
        session: Session,
        *,
        project_id: str,
        route: tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
        for_update: bool = False,
    ) -> tuple[ModelRouteActivationEventRecord, ...]:
        task, cutoff, role = route
        statement = (
            select(ModelRouteActivationEvent)
            .where(
                ModelRouteActivationEvent.project_id == project_id,
                ModelRouteActivationEvent.task == task.value,
                ModelRouteActivationEvent.cutoff_cycle == cutoff,
                ModelRouteActivationEvent.route_role == role.value,
            )
            .order_by(ModelRouteActivationEvent.stream_sequence)
        )
        if for_update:
            statement = statement.with_for_update()
        history = tuple(_record(row) for row in session.scalars(statement))
        _verify_chain(history)
        return history

    @staticmethod
    def _idempotent_event(
        session: Session,
        *,
        project_id: str,
        idempotency_key_sha256: str,
        request_sha256: str,
    ) -> ModelRouteActivationEventRecord | None:
        row = session.scalar(
            select(ModelRouteActivationEvent).where(
                ModelRouteActivationEvent.project_id == project_id,
                ModelRouteActivationEvent.idempotency_key_sha256
                == idempotency_key_sha256,
            )
        )
        if row is None:
            return None
        record = _record(row)
        if record.request_sha256 != request_sha256:
            raise ModelRouteActivationStateError(
                "idempotency key is already bound to another request"
            )
        return record

    def _recover_idempotent_conflict(
        self,
        principal: AuthPrincipal,
        project_id: str,
        route: tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
        idempotency_key_sha256: str,
        request_sha256: str,
        conflict: IntegrityError,
    ) -> ModelRouteActivationEventRecord:
        with session_scope(self._session_factory) as session:
            self._active_project(session, principal, project_id)
            existing = self._idempotent_event(
                session,
                project_id=project_id,
                idempotency_key_sha256=idempotency_key_sha256,
                request_sha256=request_sha256,
            )
            if existing is None:
                raise ModelRouteActivationStateError(
                    "concurrent route decision conflicted with the ledger head"
                ) from conflict
            self._verified_project_ledger(session, project_id)
            self._verified_candidate(
                session,
                project_id=project_id,
                artifact_id=existing.artifact_id,
                route=route,
            )
            return existing

    @staticmethod
    def _active_project(
        session: Session,
        principal: ProjectVisibilitySubject,
        project_id: str,
    ) -> Project:
        project = session.scalar(
            ProjectService.visible_projects_statement(principal)
            .where(
                Project.id == project_id,
                Project.status == ProjectStatus.ACTIVE.value,
            )
            .with_for_update()
        )
        if project is None:
            raise ModelRouteActivationNotFoundError("project was not found")
        return project

    @staticmethod
    def _require_admin(principal: AuthPrincipal) -> None:
        if principal.role is not UserRole.ADMIN:
            raise ModelRouteActivationAccessError(
                "only administrators may activate or roll back model routes"
            )


def _event(
    *,
    project_id: str,
    route: tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
    decision: ModelRouteDecisionType,
    source: VerifiedModelArtifactRegistration,
    persisted: tuple[ModelArtifact, ModelManifest],
    rollback_target_event_id: str | None,
    history: Sequence[ModelRouteActivationEventRecord],
    actor_user_id: str,
    reason: str,
    idempotency_key_sha256: str,
    request_sha256: str,
    created_at: datetime,
) -> ModelRouteActivationEventRecord:
    del persisted
    provenance = source.metadata.advanced_provenance
    if provenance is None:  # pragma: no cover - guarded before construction
        raise ModelRouteActivationStateError("Advanced provenance is missing")
    matching = next(
        item
        for item in provenance.routes
        if (AdvancedModelTask(item.task), item.cutoff_cycle, AdvancedModelRouteRole(item.role))
        == route
    )
    task, cutoff, role = route
    payload = {
        "schema_version": "model-route-activation-event-v1",
        "event_id": str(uuid4()),
        "project_id": project_id,
        "sequence_number": len(history) + 1,
        "task": task.value,
        "cutoff_cycle": cutoff,
        "role": role.value,
        "decision_type": decision.value,
        "artifact_id": source.artifact_id,
        "artifact_sha256": source.artifact_sha256,
        "manifest_sha256": source.manifest_sha256,
        "deployment_bundle_manifest_sha256": (
            provenance.deployment_bundle_manifest_sha256
        ),
        "route_provenance_sha256": sha256_canonical(matching.model_dump(mode="json")),
        "rollback_target_event_id": rollback_target_event_id,
        "previous_event_sha256": (
            history[-1].event_sha256 if history else GENESIS_EVENT_SHA256
        ),
        "actor_user_id": actor_user_id,
        "reason": reason,
        "idempotency_key_sha256": idempotency_key_sha256,
        "request_sha256": request_sha256,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
    }
    return ModelRouteActivationEventRecord.model_validate(
        {**payload, "event_sha256": sha256_canonical(payload)}
    )


def _event_row(event: ModelRouteActivationEventRecord) -> ModelRouteActivationEvent:
    return ModelRouteActivationEvent(
        id=event.event_id,
        project_id=event.project_id,
        stream_sequence=event.sequence_number,
        task=event.task.value,
        cutoff_cycle=event.cutoff_cycle,
        route_role=event.role.value,
        decision_type=event.decision_type.value,
        artifact_id=event.artifact_id,
        artifact_sha256=event.artifact_sha256,
        manifest_sha256=event.manifest_sha256,
        deployment_bundle_manifest_sha256=event.deployment_bundle_manifest_sha256,
        route_provenance_sha256=event.route_provenance_sha256,
        rollback_target_event_id=event.rollback_target_event_id,
        previous_event_sha256=event.previous_event_sha256,
        event_sha256=event.event_sha256,
        actor_user_id=event.actor_user_id,
        reason=event.reason,
        idempotency_key_sha256=event.idempotency_key_sha256,
        request_sha256=event.request_sha256,
        created_at=event.created_at,
    )


def _record(row: ModelRouteActivationEvent) -> ModelRouteActivationEventRecord:
    try:
        return ModelRouteActivationEventRecord(
            event_id=row.id,
            project_id=row.project_id,
            sequence_number=row.stream_sequence,
            task=AdvancedModelTask(row.task),
            cutoff_cycle=row.cutoff_cycle,
            role=AdvancedModelRouteRole(row.route_role),
            decision_type=ModelRouteDecisionType(row.decision_type),
            artifact_id=row.artifact_id,
            artifact_sha256=row.artifact_sha256,
            manifest_sha256=row.manifest_sha256,
            deployment_bundle_manifest_sha256=row.deployment_bundle_manifest_sha256,
            route_provenance_sha256=row.route_provenance_sha256,
            rollback_target_event_id=row.rollback_target_event_id,
            previous_event_sha256=row.previous_event_sha256,
            event_sha256=row.event_sha256,
            actor_user_id=row.actor_user_id,
            reason=row.reason,
            idempotency_key_sha256=row.idempotency_key_sha256,
            request_sha256=row.request_sha256,
            created_at=row.created_at,
        )
    except ValueError as exc:
        raise ModelRouteActivationStateError(
            "persisted model route activation ledger is invalid"
        ) from exc


def _verify_chain(history: Sequence[ModelRouteActivationEventRecord]) -> None:
    previous = GENESIS_EVENT_SHA256
    prior_by_id: dict[str, ModelRouteActivationEventRecord] = {}
    effective_artifact_id: str | None = None
    for sequence, event in enumerate(history, start=1):
        if event.sequence_number != sequence or event.previous_event_sha256 != previous:
            raise ModelRouteActivationStateError(
                "model route activation ledger hash chain is invalid"
            )
        if event.decision_type is ModelRouteDecisionType.ACTIVATE:
            if event.artifact_id == effective_artifact_id:
                raise ModelRouteActivationStateError(
                    "model route activation ledger contains a duplicate activation"
                )
        else:
            target = prior_by_id.get(event.rollback_target_event_id or "")
            if (
                target is None
                or target.decision_type is not ModelRouteDecisionType.ACTIVATE
                or target.artifact_id == effective_artifact_id
                or _candidate_snapshot(event) != _candidate_snapshot(target)
            ):
                raise ModelRouteActivationStateError(
                    "model route activation ledger contains an invalid rollback"
                )
        effective_artifact_id = event.artifact_id
        prior_by_id[event.event_id] = event
        previous = event.event_sha256


def _candidate_snapshot(event: ModelRouteActivationEventRecord) -> tuple[str, ...]:
    return (
        event.artifact_id,
        event.artifact_sha256,
        event.manifest_sha256,
        event.deployment_bundle_manifest_sha256,
        event.route_provenance_sha256,
    )


def _latest_route_snapshot(
    history: Sequence[ModelRouteActivationEventRecord],
) -> tuple[
    tuple[
        AdvancedModelTask,
        int,
        AdvancedModelRouteRole,
        str,
        int,
        str,
    ],
    ...,
]:
    latest: dict[
        tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
        ModelRouteActivationEventRecord,
    ] = {}
    for event in history:
        latest[(event.task, event.cutoff_cycle, event.role)] = event
    return tuple(
        (
            task,
            cutoff_cycle,
            role,
            event.event_id,
            event.sequence_number,
            event.event_sha256,
        )
        for (task, cutoff_cycle, role), event in sorted(
            latest.items(),
            key=lambda item: (
                item[0][0].value,
                item[0][1],
                item[0][2].value,
            ),
        )
    )


def _head_matches(
    head: ModelRouteActivationStreamHead,
    event: ModelRouteActivationEventRecord,
) -> bool:
    return (
        head.head_event_id == event.event_id
        and head.head_sequence == event.sequence_number
        and head.head_event_sha256 == event.event_sha256
    )


def _assert_persisted_candidate(
    artifact: ModelArtifact,
    manifest: ModelManifest,
    source: VerifiedModelArtifactRegistration,
) -> None:
    try:
        metadata = VerifiedModelArtifactMetadata.model_validate(manifest.metadata_json)
    except ValueError as exc:
        raise ModelRouteActivationStateError(
            "persisted candidate metadata is invalid"
        ) from exc
    expected_artifact = (
        source.artifact_format,
        source.object_uri,
        source.artifact_sha256,
        "VERIFIED",
    )
    actual_artifact = (
        artifact.artifact_format,
        artifact.object_uri,
        artifact.sha256,
        artifact.status,
    )
    expected_manifest = (
        source.model_version,
        source.manifest_uri,
        source.manifest_sha256,
        source.metadata,
    )
    actual_manifest = (
        manifest.model_version,
        manifest.manifest_uri,
        manifest.manifest_sha256,
        metadata,
    )
    if actual_artifact != expected_artifact or actual_manifest != expected_manifest:
        raise ModelRouteActivationStateError(
            "persisted candidate context differs from verified source"
        )


def _request_sha(
    *,
    decision: ModelRouteDecisionType,
    actor_user_id: str,
    project_id: str,
    route: tuple[AdvancedModelTask, int, AdvancedModelRouteRole],
    artifact_id: str | None,
    rollback_target_event_id: str | None,
    reason: str,
    idempotency_key_sha256: str,
    expected_previous_event_sha256: str,
) -> str:
    task, cutoff, role = route
    return sha256_canonical(
        {
            "schema_version": "model-route-decision-request-v1",
            "decision_type": decision.value,
            "actor_user_id": actor_user_id,
            "project_id": project_id,
            "task": task.value,
            "cutoff_cycle": cutoff,
            "role": role.value,
            "artifact_id": artifact_id,
            "rollback_target_event_id": rollback_target_event_id,
            "reason": reason,
            "idempotency_key_sha256": idempotency_key_sha256,
            "expected_previous_event_sha256": expected_previous_event_sha256,
        }
    )


def _require_expected_head(
    history: Sequence[ModelRouteActivationEventRecord],
    expected: str,
) -> None:
    actual = history[-1].event_sha256 if history else GENESIS_EVENT_SHA256
    if actual != expected:
        raise ModelRouteActivationStateError("model route ledger head is stale")


def _route(
    task: AdvancedModelTask,
    cutoff_cycle: int,
    role: AdvancedModelRouteRole,
) -> tuple[AdvancedModelTask, int, AdvancedModelRouteRole]:
    task = AdvancedModelTask(task)
    role = AdvancedModelRouteRole(role)
    if cutoff_cycle <= 0:
        raise ValueError("cutoff_cycle must be positive")
    _validate_route(task, role)
    return task, cutoff_cycle, role


def _validate_route(task: AdvancedModelTask, role: AdvancedModelRouteRole) -> None:
    approved = {
        AdvancedModelTask.RUL: {
            AdvancedModelRouteRole.DEFAULT,
            AdvancedModelRouteRole.POINT_ACCURACY,
            AdvancedModelRouteRole.COVERAGE,
        },
        AdvancedModelTask.SOH: {
            AdvancedModelRouteRole.MEAN_ACCURACY,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
        },
    }
    if role not in approved[task]:
        raise ValueError("route role does not match its Advanced task")


def _identifier(value: str, label: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"{label} must not be blank")
    return normalized


def _uuid(value: str, label: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID string") from exc


def _reason(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 2000:
        raise ValueError("manual decision reason must contain 1 to 2000 characters")
    return normalized


def _idempotency_hash(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if len(normalized) < 8 or len(normalized) > 200:
        raise ValueError("idempotency key must contain 8 to 200 characters")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sha256(value: str, label: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must include a timezone")
    return value.astimezone(UTC)


__all__ = [
    "GENESIS_EVENT_SHA256",
    "ModelRouteActivationAccessError",
    "ModelRouteActivationEventRecord",
    "ModelRouteActivationNotFoundError",
    "ModelRouteActivationService",
    "ModelRouteActivationStateError",
    "VerifiedActiveModelRoute",
]
