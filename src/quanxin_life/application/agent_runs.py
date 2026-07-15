"""Persistent, idempotent control plane for planned Agent runs."""

from __future__ import annotations

from collections.abc import Collection
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol
from uuid import uuid4

from pydantic import Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from quanxin_life.agents.supervisor import (
    SupervisorPlanningRequest,
    SupervisorPlanningResult,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.task_queue import AgentRunQueue
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentDispatchStatus,
    AgentEventType,
    AgentPlan,
    AgentPlanningMode,
    AgentRunStatus,
    DatasetStatus,
    ProjectStatus,
    UserRole,
    sha256_canonical,
)
from quanxin_life.core.product import AgentIntent
from quanxin_life.core.schemas import ContractModel, JsonMapping, _json_mapping
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentEvent,
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    Dataset,
    Project,
)
from quanxin_life.tools import StandardToolName


class AgentRunAccessError(RuntimeError):
    """Raised when a principal cannot mutate Agent runs."""


class AgentRunNotFoundError(RuntimeError):
    """Used for absent and invisible runs/projects to prevent ID disclosure."""


class AgentRunConflictError(RuntimeError):
    """Raised when one idempotency key is reused for another request."""


class AgentRunDispatchError(RuntimeError):
    """Raised after a failed queue dispatch was durably recorded."""


class AgentRunStateError(RuntimeError):
    """Raised when a requested transition is unsafe for current state."""


class AgentPlanner(Protocol):
    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: Collection[StandardToolName],
    ) -> SupervisorPlanningResult: ...


class AgentEventRecord(ContractModel):
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sequence: int = Field(ge=1)
    event_type: AgentEventType
    payload: JsonMapping
    created_at: datetime

    @field_validator("payload")
    @classmethod
    def payload_is_json_safe(cls, value: JsonMapping) -> JsonMapping:
        _json_mapping(value)
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class AgentRunRecord(ContractModel):
    run_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    created_by_user_id: str = Field(min_length=1)
    status: AgentRunStatus
    planning_mode: AgentPlanningMode
    intent: AgentIntent
    plan: AgentPlan
    dispatch_status: AgentDispatchStatus
    dispatch_task_id: str | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

    @field_validator("created_at", "updated_at", "completed_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime | None) -> datetime | None:
        return _utc(value) if value is not None else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Agent run timestamps must include a timezone")
    return value.astimezone(UTC)


def _idempotency_digest(value: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if len(normalized) < 8 or len(normalized) > 200:
        raise ValueError("Idempotency-Key must contain between 8 and 200 characters")
    return sha256(normalized.encode("utf-8")).hexdigest()


def _request_digest(request: SupervisorPlanningRequest) -> str:
    return sha256_canonical(request.model_dump(mode="json"))


def _event_record(event: AgentEvent) -> AgentEventRecord:
    try:
        event_type = AgentEventType(event.event_type)
    except ValueError as exc:
        raise AgentRunStateError("Agent event has an unsupported persisted type") from exc
    return AgentEventRecord(
        event_id=event.id,
        run_id=event.run_id,
        sequence=event.sequence,
        event_type=event_type,
        payload=event.payload_json,
        created_at=event.created_at,
    )


class AgentRunService:
    """Persist plans, timeline events and durable queue dispatch state."""

    def __init__(self, session_factory: SessionFactory, *, planner: AgentPlanner) -> None:
        self._session_factory = session_factory
        self._planner = planner

    def create_run(
        self,
        principal: AuthPrincipal,
        *,
        request: SupervisorPlanningRequest,
        idempotency_key: str,
        available_tools: Collection[StandardToolName],
        now: datetime,
    ) -> AgentRunRecord:
        self._require_operator(principal)
        timestamp = _utc(now)
        validated = SupervisorPlanningRequest.model_validate(
            request.model_dump(mode="json")
        )
        key_hash = _idempotency_digest(idempotency_key)
        request_hash = _request_digest(validated)
        existing = self._find_idempotent(principal.user_id, key_hash)
        if existing is not None:
            return self._resolve_idempotent(existing, request_hash=request_hash)

        self._validate_run_scope(principal, validated)
        planning = self._planner.plan(validated, available_tools=available_tools)
        if planning.intent.project_id != validated.project_id:
            raise AgentRunStateError("planner changed the authorized project")
        if tuple(planning.intent.dataset_ids) != tuple(validated.dataset_ids):
            raise AgentRunStateError("planner changed the authorized dataset scope")

        run_id = str(uuid4())
        try:
            with session_scope(self._session_factory) as session:
                self._validate_run_scope_in_session(session, principal, validated)
                run = AgentRun(
                    id=run_id,
                    project_id=validated.project_id,
                    created_by_user_id=principal.user_id,
                    session_id=principal.session_id,
                    status=AgentRunStatus.RUNNING.value,
                    planning_mode=planning.plan.planning_mode.value,
                    intent_json=planning.intent.model_dump(mode="json"),
                    plan_json=planning.plan.model_dump(mode="json"),
                    plan_hash=planning.plan.plan_hash,
                    execution_plan_hash=None,
                    idempotency_key_hash=key_hash,
                    request_hash=request_hash,
                    created_at=timestamp,
                    updated_at=timestamp,
                    completed_at=None,
                )
                session.add(run)
                for ordinal, step in enumerate(planning.plan.steps, start=1):
                    session.add(
                        AgentStep(
                            id=str(uuid4()),
                            run_id=run_id,
                            step_id=step.step_id,
                            ordinal=ordinal,
                            role=step.role.value,
                            tool_name=step.tool_name,
                            status="PENDING",
                            input_refs_json=step.input_references,
                            depends_on_json=list(step.depends_on),
                            failure_policy=step.failure_policy.value,
                            requires_human_approval=step.requires_approval,
                            started_at=None,
                            completed_at=None,
                        )
                    )
                dispatch = AgentRunDispatch(
                    id=str(uuid4()),
                    run_id=run_id,
                    plan_hash=planning.plan.plan_hash,
                    status=AgentDispatchStatus.PENDING.value,
                    task_id=None,
                    attempts=0,
                    last_error_code=None,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                session.add(dispatch)
                self._append_event(
                    session,
                    run_id=run_id,
                    event_type=AgentEventType.RUN_CREATED,
                    payload={
                        "status": AgentRunStatus.RUNNING.value,
                        "planning_mode": planning.plan.planning_mode.value,
                        "plan_hash": planning.plan.plan_hash,
                        "warning_codes": list(planning.warnings),
                    },
                    now=timestamp,
                    sequence=1,
                )
                session.flush()
                return self._record(run, dispatch)
        except IntegrityError:
            existing = self._find_idempotent(principal.user_id, key_hash)
            if existing is None:
                raise
            return self._resolve_idempotent(existing, request_hash=request_hash)

    def get_run(self, principal: AuthPrincipal, run_id: str) -> AgentRunRecord:
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id)
            dispatch = self._dispatch_for_run(session, run.id)
            return self._record(run, dispatch)

    def list_events(
        self,
        principal: AuthPrincipal,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> tuple[AgentEventRecord, ...]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id)
            events = session.scalars(
                select(AgentEvent)
                .where(
                    AgentEvent.run_id == run.id,
                    AgentEvent.sequence > after_sequence,
                )
                .order_by(AgentEvent.sequence)
            ).all()
            return tuple(_event_record(event) for event in events)

    def dispatch_pending(
        self,
        run_id: str,
        *,
        queue: AgentRunQueue,
        now: datetime,
    ) -> AgentRunRecord:
        timestamp = _utc(now)
        failure: Exception | None = None
        record: AgentRunRecord | None = None
        with session_scope(self._session_factory) as session:
            run = session.scalar(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise AgentRunNotFoundError("Agent run was not found")
            if run.status == AgentRunStatus.CANCELLED.value:
                raise AgentRunStateError("cancelled Agent run cannot be dispatched")
            dispatch = session.scalar(
                select(AgentRunDispatch)
                .where(AgentRunDispatch.run_id == run.id)
                .with_for_update()
            )
            if dispatch is None:
                raise AgentRunStateError("Agent run has no durable dispatch record")
            if dispatch.status == AgentDispatchStatus.DISPATCHED.value:
                return self._record(run, dispatch)
            dispatch.attempts += 1
            dispatch.updated_at = timestamp
            try:
                receipt = queue.enqueue(run_id=run.id, plan_hash=dispatch.plan_hash)
            except Exception as exc:
                failure = exc
                dispatch.last_error_code = type(exc).__name__
                self._append_next_event(
                    session,
                    run_id=run.id,
                    event_type=AgentEventType.DISPATCH_FAILED,
                    payload={"error_code": type(exc).__name__},
                    now=timestamp,
                )
            else:
                dispatch.status = AgentDispatchStatus.DISPATCHED.value
                dispatch.task_id = receipt.task_id
                dispatch.last_error_code = None
                self._append_next_event(
                    session,
                    run_id=run.id,
                    event_type=AgentEventType.RUN_DISPATCHED,
                    payload={"task_id": receipt.task_id},
                    now=timestamp,
                )
            session.flush()
            record = self._record(run, dispatch)
        if failure is not None:
            raise AgentRunDispatchError("Agent run dispatch failed") from failure
        if record is None:  # pragma: no cover - defensive transaction invariant
            raise AgentRunStateError("Agent run dispatch produced no record")
        return record

    def cancel_run(
        self,
        principal: AuthPrincipal,
        run_id: str,
        *,
        now: datetime,
    ) -> AgentRunRecord:
        self._require_operator(principal)
        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id, for_update=True)
            dispatch = self._dispatch_for_run(session, run.id)
            try:
                status = AgentRunStatus(run.status)
            except ValueError as exc:
                raise AgentRunStateError("Agent run has an unsupported status") from exc
            if status is AgentRunStatus.CANCELLED:
                return self._record(run, dispatch)
            if status in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED}:
                raise AgentRunStateError("terminal Agent run cannot be cancelled")
            run.status = AgentRunStatus.CANCELLED.value
            run.completed_at = timestamp
            run.updated_at = timestamp
            self._append_next_event(
                session,
                run_id=run.id,
                event_type=AgentEventType.RUN_CANCELLED,
                payload={"status": AgentRunStatus.CANCELLED.value},
                now=timestamp,
            )
            session.flush()
            return self._record(run, dispatch)

    def _validate_run_scope(
        self,
        principal: AuthPrincipal,
        request: SupervisorPlanningRequest,
    ) -> None:
        with session_scope(self._session_factory) as session:
            self._validate_run_scope_in_session(session, principal, request)

    @staticmethod
    def _validate_run_scope_in_session(
        session: Session,
        principal: AuthPrincipal,
        request: SupervisorPlanningRequest,
    ) -> None:
        visible_project = session.scalar(
            ProjectService.visible_projects_statement(principal).where(
                Project.id == request.project_id,
                Project.status == ProjectStatus.ACTIVE.value,
            )
        )
        if visible_project is None:
            raise AgentRunNotFoundError("project was not found")
        if not request.dataset_ids:
            return
        datasets = tuple(
            session.scalars(
                select(Dataset).where(
                    Dataset.project_id == request.project_id,
                    Dataset.id.in_(request.dataset_ids),
                )
            ).all()
        )
        if len(datasets) != len(request.dataset_ids):
            raise AgentRunNotFoundError("dataset was not found")
        if any(dataset.status != DatasetStatus.FROZEN.value for dataset in datasets):
            raise AgentRunStateError("Agent runs require frozen datasets")

    def _find_idempotent(
        self, user_id: str, idempotency_key_hash: str
    ) -> tuple[AgentRun, AgentRunDispatch] | None:
        with session_scope(self._session_factory) as session:
            run = session.scalar(
                select(AgentRun).where(
                    AgentRun.created_by_user_id == user_id,
                    AgentRun.idempotency_key_hash == idempotency_key_hash,
                )
            )
            if run is None:
                return None
            dispatch = self._dispatch_for_run(session, run.id)
            session.expunge(run)
            session.expunge(dispatch)
            return run, dispatch

    def _resolve_idempotent(
        self,
        existing: tuple[AgentRun, AgentRunDispatch],
        *,
        request_hash: str,
    ) -> AgentRunRecord:
        run, dispatch = existing
        if run.request_hash != request_hash:
            raise AgentRunConflictError(
                "idempotency key was already used for a different request"
            )
        return self._record(run, dispatch)

    @staticmethod
    def _record(run: AgentRun, dispatch: AgentRunDispatch) -> AgentRunRecord:
        try:
            status = AgentRunStatus(run.status)
            planning_mode = AgentPlanningMode(run.planning_mode)
            dispatch_status = AgentDispatchStatus(dispatch.status)
            intent = AgentIntent.model_validate(run.intent_json)
            plan = AgentPlan.model_validate(run.plan_json)
        except ValueError as exc:
            raise AgentRunStateError("Agent run has invalid persisted state") from exc
        if run.plan_hash != plan.plan_hash:
            raise AgentRunStateError("Agent run plan hash does not match the stored plan")
        if planning_mode is not plan.planning_mode:
            raise AgentRunStateError("Agent run planning mode does not match the stored plan")
        return AgentRunRecord(
            run_id=run.id,
            project_id=run.project_id,
            created_by_user_id=run.created_by_user_id,
            status=status,
            planning_mode=planning_mode,
            intent=intent,
            plan=plan,
            dispatch_status=dispatch_status,
            dispatch_task_id=dispatch.task_id,
            created_at=run.created_at,
            updated_at=run.updated_at,
            completed_at=run.completed_at,
        )

    @staticmethod
    def _dispatch_for_run(session: Session, run_id: str) -> AgentRunDispatch:
        dispatch = session.scalar(
            select(AgentRunDispatch).where(AgentRunDispatch.run_id == run_id)
        )
        if dispatch is None:
            raise AgentRunStateError("Agent run has no durable dispatch record")
        return dispatch

    @staticmethod
    def _visible_run(
        session: Session,
        principal: AuthPrincipal,
        run_id: str,
        *,
        for_update: bool = False,
    ) -> AgentRun:
        normalized_id = run_id.strip() if isinstance(run_id, str) else ""
        if not normalized_id:
            raise AgentRunNotFoundError("Agent run was not found")
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        statement = select(AgentRun).where(
            AgentRun.id == normalized_id,
            AgentRun.project_id.in_(visible_project_ids),
        )
        if for_update:
            statement = statement.with_for_update()
        run = session.scalar(statement)
        if run is None:
            raise AgentRunNotFoundError("Agent run was not found")
        return run

    @staticmethod
    def _append_event(
        session: Session,
        *,
        run_id: str,
        event_type: AgentEventType,
        payload: JsonMapping,
        now: datetime,
        sequence: int,
    ) -> None:
        session.add(
            AgentEvent(
                id=str(uuid4()),
                run_id=run_id,
                sequence=sequence,
                event_type=event_type.value,
                payload_json=_json_mapping(payload),
                created_at=now,
            )
        )

    @classmethod
    def _append_next_event(
        cls,
        session: Session,
        *,
        run_id: str,
        event_type: AgentEventType,
        payload: JsonMapping,
        now: datetime,
    ) -> None:
        latest = session.scalar(
            select(AgentEvent.sequence)
            .where(AgentEvent.run_id == run_id)
            .order_by(AgentEvent.sequence.desc())
            .limit(1)
        )
        cls._append_event(
            session,
            run_id=run_id,
            event_type=event_type,
            payload=payload,
            now=now,
            sequence=(latest or 0) + 1,
        )

    @staticmethod
    def _require_operator(principal: AuthPrincipal) -> None:
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise AgentRunAccessError("role is not allowed to mutate Agent runs")


__all__ = [
    "AgentEventRecord",
    "AgentPlanner",
    "AgentRunAccessError",
    "AgentRunConflictError",
    "AgentRunDispatchError",
    "AgentRunNotFoundError",
    "AgentRunRecord",
    "AgentRunService",
    "AgentRunStateError",
]
