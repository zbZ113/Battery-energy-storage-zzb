"""Idempotent, step-checkpointed execution for durable Agent runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from quanxin_life.agents.execution_adapter import (
    AgentExecutionContextResolver,
    AgentExecutionReferenceError,
    compile_agent_step,
)
from quanxin_life.agents.orchestrator import WorkflowStatus, run_constrained_workflow
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.audit import AuditLedgerError
from quanxin_life.core import (
    AgentEventType,
    AgentFailurePolicy,
    AgentIntent,
    AgentPlan,
    AgentRunState,
    AgentRunStatus,
    AgentStepStatus,
    ApprovalKind,
    ApprovalStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.core.schemas import _json_mapping
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentEvent,
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    ApprovalRequestRow,
    ProvenanceRecordRow,
    ToolResultRecord,
)
from quanxin_life.tools import StandardToolName

Clock = Callable[[], datetime]
_TERMINAL_STATUSES = frozenset(
    {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED}
)


class AgentRunExecutionError(RuntimeError):
    """Raised when a queued run cannot be executed from trusted persisted state."""


class AgentRunExecutionBusyError(AgentRunExecutionError):
    """Raised so the queue retries after another live worker's lease."""


class AgentRunExecutionWorker:
    """Execute one Agent plan with a durable checkpoint after every tool result."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        run_service: AgentRunService,
        tool_service: ToolInvocationService,
        context_resolver: AgentExecutionContextResolver,
        clock: Clock,
        approval_ttl: timedelta = timedelta(hours=1),
        lease_ttl: timedelta = timedelta(minutes=30),
    ) -> None:
        if tool_service.audit_ledger is None:
            raise TypeError("Agent run execution requires a shared audit ledger")
        if approval_ttl <= timedelta(0):
            raise ValueError("approval_ttl must be positive")
        if lease_ttl <= timedelta(0):
            raise ValueError("lease_ttl must be positive")
        self._session_factory = session_factory
        self._run_service = run_service
        self._tool_service = tool_service
        self._context_resolver = context_resolver
        self._clock = clock
        self._approval_ttl = approval_ttl
        self._lease_ttl = lease_ttl

    def execute(self, *, run_id: str, plan_hash: str) -> AgentRunState:
        """Execute or safely resume one queue message until it pauses or finishes."""

        _validate_identity(run_id, plan_hash)
        while True:
            state = self.advance_once(run_id=run_id, plan_hash=plan_hash)
            if state.status is not AgentRunStatus.RUNNING:
                return state

    def advance_once(self, *, run_id: str, plan_hash: str) -> AgentRunState:
        """Advance at most one professional Agent tool attempt and checkpoint it."""

        _validate_identity(run_id, plan_hash)
        intent, plan, step_rows, status = self._load_run(run_id, plan_hash)
        if status in _TERMINAL_STATUSES or status is AgentRunStatus.AWAITING_APPROVAL:
            return self._state(run_id)

        completed_results = self._restore_completed_results(
            run_id,
            intent,
            plan,
            step_rows,
        )
        next_row = next(
            (
                row
                for row in step_rows
                if row.status != AgentStepStatus.COMPLETED.value
            ),
            None,
        )
        if next_row is None:
            self._complete_run(run_id, plan_hash)
            return self._state(run_id)
        if next_row.status == AgentStepStatus.RUNNING.value and _lease_is_active(
            next_row.lease_expires_at,
            self._now(),
        ):
            raise AgentRunExecutionBusyError("Agent step has an active execution lease")
        if next_row.status == AgentStepStatus.FAILED.value:
            self._fail_run(
                run_id,
                next_row.id,
                None,
                "PERSISTED_STEP_FAILED",
            )
            return self._state(run_id)

        if next_row.requires_human_approval and not self._is_step_approved(
            run_id, next_row.step_id, plan_hash
        ):
            self._ensure_approval(run_id, next_row)
            return self._state(run_id)

        claim_token = self._claim_step(run_id, plan_hash, next_row.id)
        if claim_token is None:
            return self._state(run_id)

        try:
            compiled = compile_agent_step(
                run_id=run_id,
                intent=intent,
                plan=plan,
                step_id=next_row.step_id,
                context_resolver=self._context_resolver,
                completed_results=completed_results,
            )
            outcome = run_constrained_workflow(
                service=self._tool_service,
                request_id=run_id,
                steps=(compiled,),
                approved_step_ids=(compiled.step_id,),
            )
        except (AgentExecutionReferenceError, AuditLedgerError, TypeError, ValueError) as exc:
            self._fail_run(
                run_id,
                next_row.id,
                claim_token,
                type(exc).__name__,
            )
            return self._state(run_id)

        if outcome.status is not WorkflowStatus.COMPLETED or len(outcome.tool_results) != 1:
            self._handle_step_failure(
                run_id,
                next_row.id,
                claim_token,
                outcome.failure_code or "CONSTRAINED_STEP_FAILED",
            )
            return self._state(run_id)
        self._persist_step_result(
            run_id=run_id,
            plan_hash=plan_hash,
            step_row_id=next_row.id,
            claim_token=claim_token,
            result=outcome.tool_results[0],
        )
        _, _, refreshed_steps, _ = self._load_run(run_id, plan_hash)
        if all(
            step.status == AgentStepStatus.COMPLETED.value
            for step in refreshed_steps
        ):
            self._complete_run(run_id, plan_hash)
        return self._state(run_id)

    def _load_run(
        self,
        run_id: str,
        plan_hash: str,
    ) -> tuple[AgentIntent, AgentPlan, tuple[AgentStep, ...], AgentRunStatus]:
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            if run is None:
                raise AgentRunExecutionError("Agent run was not found")
            dispatch = session.scalar(
                select(AgentRunDispatch).where(AgentRunDispatch.run_id == run_id)
            )
            if dispatch is None:
                raise AgentRunExecutionError("Agent run has no durable dispatch")
            try:
                intent = AgentIntent.model_validate(run.intent_json)
                plan = AgentPlan.model_validate(run.plan_json)
                status = AgentRunStatus(run.status)
            except ValueError as exc:
                raise AgentRunExecutionError("Agent run has invalid persisted state") from exc
            hashes = (run.plan_hash, dispatch.plan_hash, plan.plan_hash)
            if any(value != plan_hash for value in hashes):
                raise AgentRunExecutionError("Agent run plan hash does not match the queue message")
            if run.execution_plan_hash not in {None, plan_hash}:
                raise AgentRunExecutionError("Agent run execution plan hash is stale")
            rows = tuple(
                session.scalars(
                    select(AgentStep)
                    .where(AgentStep.run_id == run_id)
                    .order_by(AgentStep.ordinal)
                ).all()
            )
            self._validate_step_rows(plan, rows)
            for row in rows:
                session.expunge(row)
            return intent, plan, rows, status

    @staticmethod
    def _validate_step_rows(plan: AgentPlan, rows: tuple[AgentStep, ...]) -> None:
        if len(rows) != len(plan.steps):
            raise AgentRunExecutionError("persisted Agent steps do not match the plan")
        for ordinal, (planned, row) in enumerate(zip(plan.steps, rows, strict=True), start=1):
            expected = (
                ordinal,
                planned.step_id,
                planned.role.value,
                planned.tool_name,
                planned.input_references,
                list(planned.depends_on),
                planned.failure_policy.value,
                planned.requires_approval,
            )
            actual = (
                row.ordinal,
                row.step_id,
                row.role,
                row.tool_name,
                row.input_refs_json,
                row.depends_on_json,
                row.failure_policy,
                row.requires_human_approval,
            )
            if actual != expected:
                raise AgentRunExecutionError("persisted Agent step differs from the plan")

    def _restore_completed_results(
        self,
        run_id: str,
        intent: AgentIntent,
        plan: AgentPlan,
        step_rows: tuple[AgentStep, ...],
    ) -> dict[str, ToolResult]:
        completed: dict[str, ToolResult] = {}
        with session_scope(self._session_factory) as session:
            for step in step_rows:
                records = tuple(
                    session.scalars(
                        select(ToolResultRecord).where(
                            ToolResultRecord.run_id == run_id,
                            ToolResultRecord.agent_step_id == step.id,
                        )
                    ).all()
                )
                if step.status == AgentStepStatus.COMPLETED.value and len(records) != 1:
                    raise AgentRunExecutionError(
                        "completed Agent step must have exactly one persisted ToolResult"
                    )
                if step.status != AgentStepStatus.COMPLETED.value and records:
                    raise AgentRunExecutionError(
                        "unfinished Agent step cannot own a persisted ToolResult"
                    )
                if not records:
                    continue
                result = self._rehydrate_result(session, records[0])
                try:
                    compiled = compile_agent_step(
                        run_id=run_id,
                        intent=intent,
                        plan=plan,
                        step_id=step.step_id,
                        context_resolver=self._context_resolver,
                        completed_results=completed,
                    )
                except AgentExecutionReferenceError as exc:
                    raise AgentRunExecutionError(
                        "persisted ToolResult input references cannot be reconstructed"
                    ) from exc
                if result.input_hash != sha256_canonical(compiled.input_value):
                    raise AgentRunExecutionError(
                        "persisted ToolResult input hash does not match trusted references"
                    )
                ledger = self._tool_service.audit_ledger
                if ledger is None:  # pragma: no cover - constructor invariant
                    raise AgentRunExecutionError("Agent audit ledger is unavailable")
                registered = ledger.ensure_result(result)
                completed[step.step_id] = registered
        return completed

    @staticmethod
    def _rehydrate_result(session: Session, row: ToolResultRecord) -> ToolResult:
        provenance_rows = tuple(
            session.scalars(
                select(ProvenanceRecordRow)
                .where(ProvenanceRecordRow.tool_result_id == row.id)
                .order_by(ProvenanceRecordRow.source_id, ProvenanceRecordRow.id)
            ).all()
        )
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

    def _claim_step(
        self,
        run_id: str,
        plan_hash: str,
        step_row_id: str,
    ) -> str | None:
        now = self._now()
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            step = session.scalar(
                select(AgentStep).where(AgentStep.id == step_row_id).with_for_update()
            )
            if run is None or step is None:
                raise AgentRunExecutionError("Agent run step was not found")
            if run.plan_hash != plan_hash:
                raise AgentRunExecutionError("Agent run plan hash changed before execution")
            status = AgentRunStatus(run.status)
            if status in _TERMINAL_STATUSES or status is AgentRunStatus.AWAITING_APPROVAL:
                return None
            if step.status == AgentStepStatus.COMPLETED.value:
                return None
            if step.status == AgentStepStatus.RUNNING.value and _lease_is_active(
                step.lease_expires_at, now
            ):
                raise AgentRunExecutionBusyError(
                    "Agent step has an active execution lease"
                )
            if step.status not in {
                AgentStepStatus.PENDING.value,
                AgentStepStatus.RUNNING.value,
            }:
                raise AgentRunExecutionError("Agent run step cannot be claimed")
            claim_token = uuid4().hex + uuid4().hex
            run.execution_plan_hash = plan_hash
            run.updated_at = now
            step.status = AgentStepStatus.RUNNING.value
            step.attempts += 1
            step.claim_token = claim_token
            step.lease_expires_at = now + self._lease_ttl
            step.last_error_code = None
            step.started_at = now
            self._append_event(
                session,
                run_id=run_id,
                event_type=AgentEventType.STEP_STARTED,
                payload={
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "attempt": step.attempts,
                },
                now=now,
            )
            return claim_token

    def _persist_step_result(
        self,
        *,
        run_id: str,
        plan_hash: str,
        step_row_id: str,
        claim_token: str,
        result: ToolResult,
    ) -> None:
        now = self._now()
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            step = session.scalar(
                select(AgentStep).where(AgentStep.id == step_row_id).with_for_update()
            )
            if run is None or step is None:
                raise AgentRunExecutionError("Agent run step was not found")
            if run.plan_hash != plan_hash:
                raise AgentRunExecutionError("Agent run plan hash changed during execution")
            if (
                step.status != AgentStepStatus.RUNNING.value
                or step.claim_token != claim_token
            ):
                raise AgentRunExecutionError("Agent step execution claim is stale")
            existing = session.scalar(
                select(ToolResultRecord).where(ToolResultRecord.agent_step_id == step.id)
            )
            if existing is not None:
                if existing.id != result.result_id:
                    raise AgentRunExecutionError("Agent step already owns another ToolResult")
                return
            session.add(
                ToolResultRecord(
                    id=result.result_id,
                    run_id=run_id,
                    agent_step_id=step.id,
                    tool_name=result.tool_name,
                    tool_version=result.tool_version,
                    model_version=result.model_version,
                    data_version=result.data_version,
                    feature_version=result.feature_version,
                    input_hash=result.input_hash,
                    values_json=_json_mapping(result.values),
                    uncertainty_json=_json_mapping(result.uncertainty),
                    warnings_json=list(result.warnings),
                    created_at=result.created_at,
                )
            )
            for item in result.provenance:
                session.add(
                    ProvenanceRecordRow(
                        id=str(uuid4()),
                        tool_result_id=result.result_id,
                        source_id=item.source_id,
                        source_kind=item.source_kind.value,
                        uri=item.uri,
                        sha256=item.sha256,
                        description=item.description,
                        created_at=item.created_at,
                    )
                )
            step.status = AgentStepStatus.COMPLETED.value
            step.claim_token = None
            step.lease_expires_at = None
            step.last_error_code = None
            step.completed_at = now
            run.updated_at = now
            self._append_event(
                session,
                run_id=run_id,
                event_type=AgentEventType.STEP_COMPLETED,
                payload={
                    "step_id": step.step_id,
                    "tool_name": step.tool_name,
                    "result_id": result.result_id,
                },
                now=now,
            )

    def _is_step_approved(self, run_id: str, step_id: str, plan_hash: str) -> bool:
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(ApprovalRequestRow).where(
                    ApprovalRequestRow.run_id == run_id,
                    ApprovalRequestRow.step_id == step_id,
                )
            )
            if row is None:
                return False
            if row.source_plan_hash != plan_hash:
                raise AgentRunExecutionError("Agent approval references a stale plan")
            if row.status == ApprovalStatus.APPROVED.value:
                return True
            if row.status == ApprovalStatus.PENDING.value:
                return False
            raise AgentRunExecutionError("Agent approval is closed and cannot resume")

    def _ensure_approval(self, run_id: str, step: AgentStep) -> None:
        now = self._now()
        self._run_service.request_approval(
            run_id,
            step_id=step.step_id,
            approval_kind=_approval_kind(step.tool_name),
            impact_scope=f"Execute approved Agent step '{step.step_id}'",
            now=now,
            expires_at=now + self._approval_ttl,
        )

    def _complete_run(self, run_id: str, plan_hash: str) -> None:
        now = self._now()
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            if run is None:
                raise AgentRunExecutionError("Agent run was not found")
            if run.plan_hash != plan_hash:
                raise AgentRunExecutionError("Agent run plan hash changed before completion")
            status = AgentRunStatus(run.status)
            if status in _TERMINAL_STATUSES:
                return
            unfinished = session.scalar(
                select(AgentStep.id).where(
                    AgentStep.run_id == run_id,
                    AgentStep.status != AgentStepStatus.COMPLETED.value,
                )
            )
            if unfinished is not None:
                raise AgentRunExecutionError("Agent run cannot complete with unfinished steps")
            run.status = AgentRunStatus.COMPLETED.value
            run.completed_at = now
            run.updated_at = now
            self._append_event(
                session,
                run_id=run_id,
                event_type=AgentEventType.RUN_COMPLETED,
                payload={"status": AgentRunStatus.COMPLETED.value},
                now=now,
            )

    def _handle_step_failure(
        self,
        run_id: str,
        step_row_id: str,
        claim_token: str,
        failure_code: str,
    ) -> bool:
        """Apply the persisted failure policy and report whether to retry now."""

        now = self._now()
        safe_code = failure_code[:100]
        terminal_code = safe_code
        with session_scope(self._session_factory) as session:
            run = session.scalar(
                select(AgentRun).where(AgentRun.id == run_id).with_for_update()
            )
            step = session.scalar(
                select(AgentStep).where(AgentStep.id == step_row_id).with_for_update()
            )
            if run is None or step is None:
                raise AgentRunExecutionError("Agent run step was not found")
            if AgentRunStatus(run.status) in _TERMINAL_STATUSES:
                return False
            if (
                step.status != AgentStepStatus.RUNNING.value
                or step.claim_token != claim_token
            ):
                raise AgentRunExecutionError("Agent step failure claim is stale")
            try:
                policy = AgentFailurePolicy(step.failure_policy)
            except ValueError as exc:
                raise AgentRunExecutionError(
                    "Agent step has an unsupported failure policy"
                ) from exc
            if policy is AgentFailurePolicy.RETRY_ONCE and step.attempts < 2:
                step.status = AgentStepStatus.PENDING.value
                step.claim_token = None
                step.lease_expires_at = None
                step.last_error_code = safe_code
                step.completed_at = None
                run.updated_at = now
                self._append_event(
                    session,
                    run_id=run_id,
                    event_type=AgentEventType.STEP_FAILED,
                    payload={
                        "step_id": step.step_id,
                        "failure_code": safe_code,
                        "will_retry": True,
                    },
                    now=now,
                )
                return True
            if policy is AgentFailurePolicy.REPLAN:
                terminal_code = f"REPLAN_UNAVAILABLE:{safe_code}"[:100]

        self._fail_run(run_id, step_row_id, claim_token, terminal_code)
        return False

    def _fail_run(
        self,
        run_id: str,
        step_row_id: str,
        claim_token: str | None,
        failure_code: str,
    ) -> None:
        now = self._now()
        safe_code = failure_code[:100]
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            step = session.scalar(
                select(AgentStep).where(AgentStep.id == step_row_id).with_for_update()
            )
            if run is None or step is None:
                raise AgentRunExecutionError("Agent run step was not found")
            if AgentRunStatus(run.status) in _TERMINAL_STATUSES:
                return
            if claim_token is not None and step.claim_token != claim_token:
                raise AgentRunExecutionError("Agent step failure claim is stale")
            step.status = AgentStepStatus.FAILED.value
            step.claim_token = None
            step.lease_expires_at = None
            step.last_error_code = safe_code
            step.completed_at = now
            run.status = AgentRunStatus.FAILED.value
            run.completed_at = now
            run.updated_at = now
            self._append_event(
                session,
                run_id=run_id,
                event_type=AgentEventType.STEP_FAILED,
                payload={
                    "step_id": step.step_id,
                    "failure_code": safe_code,
                    "will_retry": False,
                },
                now=now,
            )
            self._append_event(
                session,
                run_id=run_id,
                event_type=AgentEventType.RUN_FAILED,
                payload={"step_id": step.step_id, "failure_code": safe_code},
                now=now,
            )

    def _state(self, run_id: str) -> AgentRunState:
        with session_scope(self._session_factory) as session:
            run = session.scalar(select(AgentRun).where(AgentRun.id == run_id))
            if run is None or run.plan_hash is None:
                raise AgentRunExecutionError("Agent run was not found")
            intent = AgentIntent.model_validate(run.intent_json)
            steps = tuple(
                session.scalars(
                    select(AgentStep)
                    .where(AgentStep.run_id == run_id)
                    .order_by(AgentStep.ordinal)
                ).all()
            )
            results_by_step = {
                row.agent_step_id: row.id
                for row in session.scalars(
                    select(ToolResultRecord).where(ToolResultRecord.run_id == run_id)
                ).all()
                if row.agent_step_id is not None
            }
            pending = tuple(
                row.step_id
                for row in session.scalars(
                    select(ApprovalRequestRow).where(
                        ApprovalRequestRow.run_id == run_id,
                        ApprovalRequestRow.status == ApprovalStatus.PENDING.value,
                    )
                ).all()
            )
            return AgentRunState(
                run_id=run.id,
                intent_id=intent.intent_id,
                plan_hash=run.plan_hash,
                status=AgentRunStatus(run.status),
                completed_step_ids=tuple(
                    step.step_id
                    for step in steps
                    if step.status == AgentStepStatus.COMPLETED.value
                ),
                pending_approval_step_ids=pending,
                result_ids=tuple(
                    results_by_step[step.id] for step in steps if step.id in results_by_step
                ),
                updated_at=_database_utc(run.updated_at),
            )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Agent execution clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _append_event(
        session: Session,
        *,
        run_id: str,
        event_type: AgentEventType,
        payload: Mapping[str, object],
        now: datetime,
    ) -> None:
        latest = session.scalar(
            select(AgentEvent.sequence)
            .where(AgentEvent.run_id == run_id)
            .order_by(AgentEvent.sequence.desc())
            .limit(1)
        )
        session.add(
            AgentEvent(
                id=str(uuid4()),
                run_id=run_id,
                sequence=(latest or 0) + 1,
                event_type=event_type.value,
                payload_json=_json_mapping(dict(payload)),
                created_at=now,
            )
        )
        # Make the assigned sequence visible to a second event appended in the
        # same transaction. The run row is already locked by every caller.
        session.flush()


def _validate_identity(run_id: str, plan_hash: str) -> None:
    try:
        parsed = UUID(run_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise AgentRunExecutionError("run_id must be a UUID") from exc
    if str(parsed) != run_id:
        raise AgentRunExecutionError("run_id must be canonical")
    if len(plan_hash) != 64 or any(character not in "0123456789abcdef" for character in plan_hash):
        raise AgentRunExecutionError("plan hash must be a lowercase SHA-256 digest")


def _approval_kind(tool_name: str) -> ApprovalKind:
    if tool_name == StandardToolName.MAKE_BATCH_DECISION.value:
        return ApprovalKind.FORMAL_DECISION
    if tool_name == StandardToolName.RECOMMEND_NEXT_EXPERIMENT.value:
        return ApprovalKind.EXPERIMENT_ACTION
    return ApprovalKind.EXTERNAL_WRITE


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _lease_is_active(expires_at: datetime | None, now: datetime) -> bool:
    return expires_at is not None and _database_utc(expires_at) > now


__all__ = [
    "AgentRunExecutionBusyError",
    "AgentRunExecutionError",
    "AgentRunExecutionWorker",
]
