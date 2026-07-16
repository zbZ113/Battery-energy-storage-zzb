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
    AgentStepStatus,
    ApprovalKind,
    ApprovalRequest,
    ApprovalStatus,
    DatasetStatus,
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
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
    ApprovalAction,
    ApprovalRequestRow,
    Dataset,
    Project,
    ProvenanceRecordRow,
    ToolResultRecord,
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


def _database_utc(value: datetime) -> datetime:
    """Interpret timezone-naive SQLite values as UTC; production stores aware UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
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
                            status=AgentStepStatus.PENDING.value,
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

    def get_result(
        self,
        principal: AuthPrincipal,
        run_id: str,
        result_id: str,
    ) -> ToolResult:
        """Read one persisted result only through a project-visible Agent run."""
        normalized_result_id = result_id.strip() if isinstance(result_id, str) else ""
        if not normalized_result_id:
            raise AgentRunNotFoundError("Agent result was not found")
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id)
            row = session.scalar(
                select(ToolResultRecord)
                .join(AgentStep, ToolResultRecord.agent_step_id == AgentStep.id)
                .where(
                    ToolResultRecord.id == normalized_result_id,
                    ToolResultRecord.run_id == run.id,
                    AgentStep.run_id == run.id,
                    AgentStep.status == AgentStepStatus.COMPLETED.value,
                    ToolResultRecord.tool_name == AgentStep.tool_name,
                )
            )
            if row is None:
                raise AgentRunNotFoundError("Agent result was not found")
            provenance_rows = tuple(
                session.scalars(
                    select(ProvenanceRecordRow)
                    .where(ProvenanceRecordRow.tool_result_id == row.id)
                    .order_by(ProvenanceRecordRow.source_id, ProvenanceRecordRow.id)
                ).all()
            )
            try:
                return ToolResult(
                    result_id=row.id,
                    tool_name=row.tool_name,
                    tool_version=row.tool_version,
                    model_version=row.model_version,
                    data_version=row.data_version,
                    feature_version=row.feature_version,
                    input_hash=row.input_hash,
                    values=row.values_json,
                    uncertainty=row.uncertainty_json,
                    warnings=row.warnings_json,
                    provenance=[
                        ProvenanceRecord(
                            source_id=item.source_id,
                            source_kind=SourceKind(item.source_kind),
                            uri=item.uri,
                            sha256=item.sha256,
                            description=item.description,
                            created_at=_database_utc(item.created_at),
                        )
                        for item in provenance_rows
                    ],
                    created_at=_database_utc(row.created_at),
                )
            except ValueError as exc:
                raise AgentRunStateError("Agent result has invalid persisted state") from exc

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

    def request_approval(
        self,
        run_id: str,
        *,
        step_id: str,
        approval_kind: ApprovalKind,
        impact_scope: str,
        now: datetime,
        expires_at: datetime,
    ) -> ApprovalRequest:
        """Persist a worker-requested human gate and pause the run atomically."""

        timestamp = _utc(now)
        with session_scope(self._session_factory) as session:
            run = session.scalar(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )
            if run is None:
                raise AgentRunNotFoundError("Agent run was not found")
            if run.plan_hash is None:
                raise AgentRunStateError("Agent run has no approved plan hash")
            request = ApprovalRequest(
                approval_id=str(uuid4()),
                run_id=run_id,
                approval_kind=approval_kind,
                source_plan_hash=run.plan_hash,
                step_id=step_id,
                impact_scope=impact_scope,
                status=ApprovalStatus.PENDING,
                created_at=timestamp,
                expires_at=expires_at,
            )
            status = self._run_status(run)
            if status not in {AgentRunStatus.RUNNING, AgentRunStatus.AWAITING_APPROVAL}:
                raise AgentRunStateError("Agent run cannot request approval in its current state")
            step = session.scalar(
                select(AgentStep).where(
                    AgentStep.run_id == run.id,
                    AgentStep.step_id == request.step_id,
                )
            )
            if step is None or not step.requires_human_approval:
                raise AgentRunStateError("Agent step is not approved for a human gate")
            existing = session.scalar(
                select(ApprovalRequestRow).where(
                    ApprovalRequestRow.run_id == run.id,
                    ApprovalRequestRow.step_id == request.step_id,
                )
            )
            if existing is not None:
                persisted = self._approval_record(existing)
                if (
                    persisted.approval_kind is request.approval_kind
                    and persisted.impact_scope == request.impact_scope
                    and persisted.expires_at == request.expires_at
                ):
                    return persisted
                raise AgentRunConflictError("approval gate already exists for this step")
            row = ApprovalRequestRow(
                id=request.approval_id,
                run_id=run.id,
                approval_kind=request.approval_kind.value,
                source_plan_hash=request.source_plan_hash,
                step_id=request.step_id,
                impact_scope=request.impact_scope,
                status=request.status.value,
                created_at=request.created_at,
                expires_at=request.expires_at,
            )
            session.add(row)
            run.status = AgentRunStatus.AWAITING_APPROVAL.value
            run.updated_at = timestamp
            step.status = AgentStepStatus.AWAITING_APPROVAL.value
            self._append_next_event(
                session,
                run_id=run.id,
                event_type=AgentEventType.APPROVAL_REQUESTED,
                payload={
                    "approval_id": request.approval_id,
                    "approval_kind": request.approval_kind.value,
                    "step_id": request.step_id,
                    "expires_at": request.expires_at.isoformat(),
                },
                now=timestamp,
            )
            session.flush()
            return self._approval_record(row)

    def get_approval(
        self,
        principal: AuthPrincipal,
        run_id: str,
        approval_id: str,
    ) -> ApprovalRequest:
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id)
            row = session.scalar(
                select(ApprovalRequestRow).where(
                    ApprovalRequestRow.id == approval_id,
                    ApprovalRequestRow.run_id == run.id,
                )
            )
            if row is None:
                raise AgentRunNotFoundError("approval request was not found")
            return self._approval_record(row)

    def approve_run(
        self,
        principal: AuthPrincipal,
        run_id: str,
        *,
        approval_id: str,
        reason: str | None,
        now: datetime,
    ) -> AgentRunRecord:
        return self._act_on_approval(
            principal,
            run_id,
            approval_id=approval_id,
            reason=reason,
            target=ApprovalStatus.APPROVED,
            now=now,
        )

    def reject_run(
        self,
        principal: AuthPrincipal,
        run_id: str,
        *,
        approval_id: str,
        reason: str | None,
        now: datetime,
    ) -> AgentRunRecord:
        return self._act_on_approval(
            principal,
            run_id,
            approval_id=approval_id,
            reason=reason,
            target=ApprovalStatus.REJECTED,
            now=now,
        )

    def _act_on_approval(
        self,
        principal: AuthPrincipal,
        run_id: str,
        *,
        approval_id: str,
        reason: str | None,
        target: ApprovalStatus,
        now: datetime,
    ) -> AgentRunRecord:
        self._require_operator(principal)
        if target not in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}:
            raise ValueError("approval target must be APPROVED or REJECTED")
        timestamp = _utc(now)
        normalized_reason = self._normalized_reason(reason)
        expired = False
        record: AgentRunRecord | None = None
        with session_scope(self._session_factory) as session:
            run = self._visible_run(session, principal, run_id, for_update=True)
            dispatch = self._dispatch_for_run(session, run.id)
            row = session.scalar(
                select(ApprovalRequestRow)
                .where(
                    ApprovalRequestRow.id == approval_id,
                    ApprovalRequestRow.run_id == run.id,
                )
                .with_for_update()
            )
            if row is None:
                raise AgentRunNotFoundError("approval request was not found")
            persisted = self._approval_record(row)
            if persisted.status is target:
                return self._record(run, dispatch)
            if persisted.status is not ApprovalStatus.PENDING:
                raise AgentRunStateError("approval request is already closed")
            if persisted.source_plan_hash != run.plan_hash:
                raise AgentRunStateError("approval request plan hash is stale")
            if timestamp >= persisted.expires_at:
                row.status = ApprovalStatus.EXPIRED.value
                self._append_next_event(
                    session,
                    run_id=run.id,
                    event_type=AgentEventType.APPROVAL_EXPIRED,
                    payload={
                        "approval_id": row.id,
                        "step_id": row.step_id,
                    },
                    now=timestamp,
                )
                expired = True
            else:
                row.status = target.value
                session.add(
                    ApprovalAction(
                        id=str(uuid4()),
                        approval_request_id=row.id,
                        actor_user_id=principal.user_id,
                        action=target.value,
                        reason=normalized_reason,
                        acted_at=timestamp,
                    )
                )
                if target is ApprovalStatus.APPROVED:
                    run.status = AgentRunStatus.RUNNING.value
                    step = session.scalar(
                        select(AgentStep).where(
                            AgentStep.run_id == run.id,
                            AgentStep.step_id == row.step_id,
                        )
                    )
                    if step is None:
                        raise AgentRunStateError("approval request Agent step was not found")
                    step.status = AgentStepStatus.PENDING.value
                    dispatch.status = AgentDispatchStatus.PENDING.value
                    dispatch.task_id = None
                    dispatch.updated_at = timestamp
                    event_type = AgentEventType.APPROVAL_APPROVED
                else:
                    run.status = AgentRunStatus.CANCELLED.value
                    run.completed_at = timestamp
                    step = session.scalar(
                        select(AgentStep).where(
                            AgentStep.run_id == run.id,
                            AgentStep.step_id == row.step_id,
                        )
                    )
                    if step is None:
                        raise AgentRunStateError("approval request Agent step was not found")
                    step.status = AgentStepStatus.CANCELLED.value
                    step.completed_at = timestamp
                    event_type = AgentEventType.APPROVAL_REJECTED
                    self._cancel_other_pending_approvals(
                        session, run_id=run.id, exclude_approval_id=row.id
                    )
                run.updated_at = timestamp
                self._append_next_event(
                    session,
                    run_id=run.id,
                    event_type=event_type,
                    payload={
                        "approval_id": row.id,
                        "step_id": row.step_id,
                    },
                    now=timestamp,
                )
            session.flush()
            record = self._record(run, dispatch)
        if expired:
            raise AgentRunStateError("approval request has expired")
        if record is None:  # pragma: no cover - defensive transaction invariant
            raise AgentRunStateError("approval action produced no Agent run record")
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
            self._cancel_other_pending_approvals(session, run_id=run.id)
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
    def _approval_record(row: ApprovalRequestRow) -> ApprovalRequest:
        try:
            approval_kind = ApprovalKind(row.approval_kind)
            status = ApprovalStatus(row.status)
        except ValueError as exc:
            raise AgentRunStateError("approval request has invalid persisted state") from exc
        return ApprovalRequest(
            approval_id=row.id,
            run_id=row.run_id,
            approval_kind=approval_kind,
            source_plan_hash=row.source_plan_hash,
            step_id=row.step_id,
            impact_scope=row.impact_scope,
            status=status,
            created_at=row.created_at,
            expires_at=row.expires_at,
        )

    @staticmethod
    def _cancel_other_pending_approvals(
        session: Session,
        *,
        run_id: str,
        exclude_approval_id: str | None = None,
    ) -> None:
        statement = select(ApprovalRequestRow).where(
            ApprovalRequestRow.run_id == run_id,
            ApprovalRequestRow.status == ApprovalStatus.PENDING.value,
        )
        if exclude_approval_id is not None:
            statement = statement.where(ApprovalRequestRow.id != exclude_approval_id)
        for row in session.scalars(statement).all():
            row.status = ApprovalStatus.CANCELLED.value

    @staticmethod
    def _normalized_reason(reason: str | None) -> str | None:
        if reason is None:
            return None
        normalized = reason.strip()
        if not normalized:
            return None
        if len(normalized) > 2_000:
            raise ValueError("approval reason must contain at most 2000 characters")
        return normalized

    @staticmethod
    def _run_status(run: AgentRun) -> AgentRunStatus:
        try:
            return AgentRunStatus(run.status)
        except ValueError as exc:
            raise AgentRunStateError("Agent run has an unsupported status") from exc

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
