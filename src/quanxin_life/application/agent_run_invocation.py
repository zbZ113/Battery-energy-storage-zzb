"""Server-signed per-step grants derived only from persistent AgentRun state."""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    AgentDispatchStatus,
    AgentIntent,
    AgentPlan,
    AgentRunStatus,
    AgentStepStatus,
    canonical_json_bytes,
    sha256_canonical,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    ApprovalAction,
    ApprovalRequestRow,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    ToolResultRecord,
)
from quanxin_life.tools import StandardToolName

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class AgentRunInvocationAccessError(RuntimeError):
    """Raised when persistent run state cannot authorize one exact tool step."""


@dataclass(frozen=True, slots=True)
class VerifiedAgentRunInvocationGrant:
    """Immutable singleton allowlist bound to one live Agent step claim."""

    agent_run_id: str
    agent_step_row_id: str
    step_id: str
    plan_hash: str
    input_hash: str
    claim_attempt: int
    claim_lease_expires_at: datetime
    step_spec_sha256: str
    dependency_evidence_sha256: str
    execution_snapshot_sha256: str
    approval_required: bool
    approval_request_id: str | None
    approval_action_id: str | None
    approval_evidence_sha256: str
    allowed_tool_names: frozenset[StandardToolName]
    project_context: VerifiedProjectInvocationContext
    _claim_token_sha256: str = field(repr=False)
    _authorization_tag: str = field(repr=False)

    def __post_init__(self) -> None:
        for value in (
            self.agent_run_id,
            self.agent_step_row_id,
            self.step_id,
            self.plan_hash,
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Agent run invocation grant identifiers must not be blank")
        if len(self.allowed_tool_names) != 1 or not all(
            isinstance(item, StandardToolName) for item in self.allowed_tool_names
        ):
            raise ValueError("Agent run invocation grant requires one allowed tool")
        if (
            not isinstance(self.project_context, VerifiedProjectInvocationContext)
            or self.project_context.invocation_source is not ProjectInvocationSource.AGENT
            or self.project_context.agent_run_id != self.agent_run_id
        ):
            raise ValueError("Agent run invocation grant requires an AGENT project context")
        if self.claim_attempt < 1:
            raise ValueError("Agent run invocation claim attempt must be positive")
        if (
            self.claim_lease_expires_at.tzinfo is None
            or self.claim_lease_expires_at.utcoffset() is None
        ):
            raise ValueError("Agent run invocation lease must include a timezone")
        if self.approval_required != bool(
            self.approval_request_id and self.approval_action_id
        ):
            raise ValueError("Agent run invocation approval evidence is incomplete")
        for digest in (
            self.input_hash,
            self.step_spec_sha256,
            self.dependency_evidence_sha256,
            self.execution_snapshot_sha256,
            self.approval_evidence_sha256,
            self._claim_token_sha256,
            self._authorization_tag,
        ):
            if (
                len(digest) != 64
                or digest.casefold() != digest
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Agent run invocation grant hashes must be SHA-256")


class AgentRunInvocationValidator(Protocol):
    def revalidate(
        self,
        grant: VerifiedAgentRunInvocationGrant,
    ) -> VerifiedAgentRunInvocationGrant: ...

    def revalidate_in_session(
        self,
        session: Session,
        grant: VerifiedAgentRunInvocationGrant,
    ) -> VerifiedAgentRunInvocationGrant: ...


@dataclass(frozen=True, slots=True)
class _PersistedStepSnapshot:
    agent_run_id: str
    agent_step_row_id: str
    step_id: str
    plan_hash: str
    tool_name: StandardToolName
    input_hash: str
    claim_attempt: int
    claim_lease_expires_at: datetime
    step_spec_sha256: str
    dependency_evidence_sha256: str
    execution_snapshot_sha256: str
    approval_required: bool
    approval_request_id: str | None
    approval_action_id: str | None
    approval_evidence_sha256: str
    claim_token_sha256: str
    project_id: str
    actor_user_id: str
    actor_session_id: str


class PersistentAgentRunInvocationResolver:
    """Issue and revalidate grants without accepting caller-owned authorization."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_service: ProjectInvocationContextService,
        clock: Clock = _utc_now,
    ) -> None:
        if not isinstance(context_service, ProjectInvocationContextService):
            raise TypeError("context_service must be a ProjectInvocationContextService")
        self._session_factory = session_factory
        self._context_service = context_service
        self._clock = clock
        self._signing_key = secrets.token_bytes(32)

    @property
    def context_service(self) -> ProjectInvocationContextService:
        return self._context_service

    def resolve(
        self,
        run_id: str,
        step_id: str,
        claim_token: str,
    ) -> VerifiedAgentRunInvocationGrant:
        """Issue one singleton grant from a current persistent step claim."""

        normalized_claim = self._identifier(claim_token, "claim token")
        snapshot = self._snapshot(
            run_id,
            step_id,
            expected_claim_token_sha256=self._claim_digest(normalized_claim),
        )
        context = self._context_service.resolve_agent_run(snapshot.agent_run_id)
        self._require_context_matches_snapshot(context, snapshot)
        unsigned = VerifiedAgentRunInvocationGrant(
            agent_run_id=snapshot.agent_run_id,
            agent_step_row_id=snapshot.agent_step_row_id,
            step_id=snapshot.step_id,
            plan_hash=snapshot.plan_hash,
            input_hash=snapshot.input_hash,
            claim_attempt=snapshot.claim_attempt,
            claim_lease_expires_at=snapshot.claim_lease_expires_at,
            step_spec_sha256=snapshot.step_spec_sha256,
            dependency_evidence_sha256=snapshot.dependency_evidence_sha256,
            execution_snapshot_sha256=snapshot.execution_snapshot_sha256,
            approval_required=snapshot.approval_required,
            approval_request_id=snapshot.approval_request_id,
            approval_action_id=snapshot.approval_action_id,
            approval_evidence_sha256=snapshot.approval_evidence_sha256,
            allowed_tool_names=frozenset({snapshot.tool_name}),
            project_context=context,
            _claim_token_sha256=snapshot.claim_token_sha256,
            _authorization_tag="0" * 64,
        )
        return replace(unsigned, _authorization_tag=self._sign(unsigned))

    def revalidate(
        self,
        grant: VerifiedAgentRunInvocationGrant,
    ) -> VerifiedAgentRunInvocationGrant:
        """Rebuild the exact grant from live database state before every use."""

        if not isinstance(grant, VerifiedAgentRunInvocationGrant):
            raise AgentRunInvocationAccessError(
                "verified Agent run invocation grant is required"
            )
        if not hmac.compare_digest(grant._authorization_tag, self._sign(grant)):
            raise AgentRunInvocationAccessError(
                "Agent run invocation grant is not trusted"
            )
        context = self._context_service.revalidate(grant.project_context)
        snapshot = self._snapshot(
            grant.agent_run_id,
            grant.step_id,
            expected_claim_token_sha256=grant._claim_token_sha256,
        )
        self._require_context_matches_snapshot(context, snapshot)
        expected = (
            snapshot.agent_step_row_id,
            snapshot.plan_hash,
            snapshot.input_hash,
            snapshot.claim_attempt,
            snapshot.claim_lease_expires_at,
            snapshot.step_spec_sha256,
            snapshot.dependency_evidence_sha256,
            snapshot.execution_snapshot_sha256,
            snapshot.approval_required,
            snapshot.approval_request_id,
            snapshot.approval_action_id,
            snapshot.approval_evidence_sha256,
            frozenset({snapshot.tool_name}),
        )
        actual = (
            grant.agent_step_row_id,
            grant.plan_hash,
            grant.input_hash,
            grant.claim_attempt,
            grant.claim_lease_expires_at,
            grant.step_spec_sha256,
            grant.dependency_evidence_sha256,
            grant.execution_snapshot_sha256,
            grant.approval_required,
            grant.approval_request_id,
            grant.approval_action_id,
            grant.approval_evidence_sha256,
            grant.allowed_tool_names,
        )
        if actual != expected:
            raise AgentRunInvocationAccessError(
                "Agent run invocation grant no longer matches persisted state"
            )
        return grant

    def revalidate_in_session(
        self,
        session: Session,
        grant: VerifiedAgentRunInvocationGrant,
    ) -> VerifiedAgentRunInvocationGrant:
        """Recompute exact step evidence inside the caller's commit transaction."""

        if not isinstance(grant, VerifiedAgentRunInvocationGrant):
            raise AgentRunInvocationAccessError(
                "verified Agent run invocation grant is required"
            )
        if not hmac.compare_digest(grant._authorization_tag, self._sign(grant)):
            raise AgentRunInvocationAccessError(
                "Agent run invocation grant is not trusted"
            )
        run = session.get(AgentRun, grant.agent_run_id)
        dispatch = session.scalar(
            select(AgentRunDispatch).where(
                AgentRunDispatch.run_id == grant.agent_run_id
            )
        )
        rows = tuple(
            session.scalars(
                select(AgentStep)
                .where(AgentStep.run_id == grant.agent_run_id)
                .order_by(AgentStep.ordinal)
            ).all()
        )
        if run is None or dispatch is None:
            raise AgentRunInvocationAccessError(
                "persisted Agent run invocation was not found"
            )
        try:
            plan = AgentPlan.model_validate(run.plan_json)
        except ValueError as exc:
            raise AgentRunInvocationAccessError(
                "persisted Agent run invocation is invalid"
            ) from exc
        self._validate_step_rows(plan, rows)
        row = next(
            (item for item in rows if item.id == grant.agent_step_row_id),
            None,
        )
        if row is not None and (
            row.resolved_input_json is None
            or not row.resolved_input_hash
            or sha256_canonical(row.resolved_input_json)
            != row.resolved_input_hash
            or row.resolved_input_hash != grant.input_hash
        ):
            raise AgentRunInvocationAccessError(
                "persisted Agent step frozen input has changed"
            )
        if (
            row is None
            or row.step_id != grant.step_id
            or run.plan_hash != grant.plan_hash
            or dispatch.plan_hash != grant.plan_hash
            or row.status != AgentStepStatus.RUNNING.value
            or row.attempts != grant.claim_attempt
            or not row.claim_token
            or self._claim_digest(row.claim_token) != grant._claim_token_sha256
            or row.execution_claim_sha256 != grant._claim_token_sha256
            or row.lease_expires_at is None
            or self._database_utc(row.lease_expires_at)
            != grant.claim_lease_expires_at
        ):
            raise AgentRunInvocationAccessError(
                "persisted Agent step claim is not active"
            )
        step_spec_sha256 = self._step_spec_sha256(row)
        dependency_evidence_sha256 = self._dependency_evidence_sha256(
            session,
            row=row,
            rows=rows,
        )
        approval_request_id, approval_action_id, approval_evidence_sha256 = (
            self._approval_evidence(
                session,
                row=row,
                plan_hash=grant.plan_hash,
            )
        )
        execution_snapshot_sha256 = sha256_canonical(
            {
                "schema_version": "agent-step-execution-snapshot-v1",
                "agent_run_id": run.id,
                "agent_step_row_id": row.id,
                "step_id": row.step_id,
                "plan_hash": grant.plan_hash,
                "tool_name": row.tool_name,
                "input_hash": row.resolved_input_hash,
                "step_spec_sha256": step_spec_sha256,
                "dependency_evidence_sha256": dependency_evidence_sha256,
            }
        )
        expected = (
            step_spec_sha256,
            dependency_evidence_sha256,
            execution_snapshot_sha256,
            row.requires_human_approval,
            approval_request_id,
            approval_action_id,
            approval_evidence_sha256,
        )
        actual = (
            grant.step_spec_sha256,
            grant.dependency_evidence_sha256,
            grant.execution_snapshot_sha256,
            grant.approval_required,
            grant.approval_request_id,
            grant.approval_action_id,
            grant.approval_evidence_sha256,
        )
        if (
            actual != expected
            or row.execution_snapshot_sha256 != execution_snapshot_sha256
            or row.dependency_evidence_sha256 != dependency_evidence_sha256
        ):
            raise AgentRunInvocationAccessError(
                "persisted Agent step evidence changed before commit"
            )
        return grant

    def _snapshot(
        self,
        run_id: str,
        step_id: str,
        *,
        expected_claim_token_sha256: str,
    ) -> _PersistedStepSnapshot:
        normalized_run_id = self._identifier(run_id, "run_id")
        normalized_step_id = self._identifier(step_id, "step_id")
        now = self._current_time()
        try:
            with session_scope(self._session_factory) as session:
                run = session.get(AgentRun, normalized_run_id)
                dispatch = session.scalar(
                    select(AgentRunDispatch).where(
                        AgentRunDispatch.run_id == normalized_run_id
                    )
                )
                rows = tuple(
                    session.scalars(
                        select(AgentStep)
                        .where(AgentStep.run_id == normalized_run_id)
                        .order_by(AgentStep.ordinal)
                    ).all()
                )
                if run is None or dispatch is None or run.session_id is None:
                    raise AgentRunInvocationAccessError(
                        "persisted Agent run invocation was not found"
                    )
                try:
                    intent = AgentIntent.model_validate(run.intent_json)
                    plan = AgentPlan.model_validate(run.plan_json)
                    run_status = AgentRunStatus(run.status)
                    dispatch_status = AgentDispatchStatus(dispatch.status)
                except ValueError as exc:
                    raise AgentRunInvocationAccessError(
                        "persisted Agent run invocation is invalid"
                    ) from exc
                if (
                    run_status is not AgentRunStatus.RUNNING
                    or dispatch_status is not AgentDispatchStatus.DISPATCHED
                    or run.project_id != intent.project_id
                    or plan.intent_id != intent.intent_id
                    or run.planning_mode != plan.planning_mode.value
                    or not run.plan_hash
                    or run.plan_hash != plan.plan_hash
                    or dispatch.plan_hash != plan.plan_hash
                    or run.execution_plan_hash not in {None, plan.plan_hash}
                ):
                    raise AgentRunInvocationAccessError(
                        "persisted Agent run invocation context has drifted"
                    )
                self._validate_step_rows(plan, rows)
                row = next(
                    (item for item in rows if item.step_id == normalized_step_id),
                    None,
                )
                if (
                    row is None
                    or row.status != AgentStepStatus.RUNNING.value
                    or not row.claim_token
                    or row.lease_expires_at is None
                    or row.lease_expires_at <= now
                    or self._claim_digest(row.claim_token)
                    != expected_claim_token_sha256
                    or row.execution_claim_sha256
                    != expected_claim_token_sha256
                    or row.resolved_input_json is None
                    or not row.resolved_input_hash
                    or sha256_canonical(row.resolved_input_json)
                    != row.resolved_input_hash
                ):
                    raise AgentRunInvocationAccessError(
                        "persisted Agent step claim is not active"
                    )
                try:
                    tool_name = StandardToolName(row.tool_name)
                except ValueError as exc:
                    raise AgentRunInvocationAccessError(
                        "persisted Agent step tool is invalid"
                    ) from exc
                step_spec_sha256 = self._step_spec_sha256(row)
                dependency_evidence_sha256 = self._dependency_evidence_sha256(
                    session,
                    row=row,
                    rows=rows,
                )
                (
                    approval_request_id,
                    approval_action_id,
                    approval_evidence_sha256,
                ) = self._approval_evidence(
                    session,
                    row=row,
                    plan_hash=plan.plan_hash,
                )
                execution_snapshot_sha256 = sha256_canonical(
                    {
                        "schema_version": "agent-step-execution-snapshot-v1",
                        "agent_run_id": run.id,
                        "agent_step_row_id": row.id,
                        "step_id": row.step_id,
                        "plan_hash": plan.plan_hash,
                        "tool_name": tool_name.value,
                        "input_hash": row.resolved_input_hash,
                        "step_spec_sha256": step_spec_sha256,
                        "dependency_evidence_sha256": dependency_evidence_sha256,
                    }
                )
                if row.execution_snapshot_sha256 not in {
                    None,
                    execution_snapshot_sha256,
                } or row.dependency_evidence_sha256 not in {
                    None,
                    dependency_evidence_sha256,
                }:
                    raise AgentRunInvocationAccessError(
                        "persisted Agent step execution snapshot has drifted"
                    )
                snapshot = _PersistedStepSnapshot(
                    agent_run_id=run.id,
                    agent_step_row_id=row.id,
                    step_id=row.step_id,
                    plan_hash=plan.plan_hash,
                    tool_name=tool_name,
                    input_hash=row.resolved_input_hash,
                    claim_attempt=row.attempts,
                    claim_lease_expires_at=self._database_utc(
                        row.lease_expires_at
                    ),
                    step_spec_sha256=step_spec_sha256,
                    dependency_evidence_sha256=dependency_evidence_sha256,
                    execution_snapshot_sha256=execution_snapshot_sha256,
                    approval_required=row.requires_human_approval,
                    approval_request_id=approval_request_id,
                    approval_action_id=approval_action_id,
                    approval_evidence_sha256=approval_evidence_sha256,
                    claim_token_sha256=expected_claim_token_sha256,
                    project_id=run.project_id,
                    actor_user_id=run.created_by_user_id,
                    actor_session_id=run.session_id,
                )
        except AgentRunInvocationAccessError:
            raise
        except SQLAlchemyError as exc:
            raise AgentRunInvocationAccessError(
                "persisted Agent run invocation could not be verified"
            ) from exc
        return snapshot

    @staticmethod
    def _step_spec_sha256(row: AgentStep) -> str:
        return sha256_canonical(
            {
                "agent_step_row_id": row.id,
                "agent_run_id": row.run_id,
                "step_id": row.step_id,
                "ordinal": row.ordinal,
                "role": row.role,
                "tool_name": row.tool_name,
                "input_references": row.input_refs_json,
                "depends_on": row.depends_on_json,
                "failure_policy": row.failure_policy,
                "requires_approval": row.requires_human_approval,
            }
        )

    @staticmethod
    def _dependency_evidence_sha256(
        session: object,
        *,
        row: AgentStep,
        rows: tuple[AgentStep, ...],
    ) -> str:
        from sqlalchemy.orm import Session

        if not isinstance(session, Session):  # pragma: no cover - internal invariant
            raise TypeError("dependency evidence requires a database session")
        by_step_id = {item.step_id: item for item in rows}
        evidence: list[dict[str, object]] = []
        for dependency_id in row.depends_on_json:
            dependency = by_step_id.get(dependency_id)
            if (
                dependency is None
                or dependency.ordinal >= row.ordinal
                or dependency.status != AgentStepStatus.COMPLETED.value
            ):
                raise AgentRunInvocationAccessError(
                    "persisted Agent step dependencies are not completed"
                )
            results = tuple(
                session.scalars(
                    select(ToolResultRecord).where(
                        ToolResultRecord.run_id == row.run_id,
                        ToolResultRecord.agent_step_id == dependency.id,
                    )
                ).all()
            )
            if len(results) != 1:
                raise AgentRunInvocationAccessError(
                    "persisted Agent step dependency evidence is incomplete"
                )
            result = results[0]
            provenance = tuple(
                session.scalars(
                    select(ProvenanceRecordRow)
                    .where(ProvenanceRecordRow.tool_result_id == result.id)
                    .order_by(ProvenanceRecordRow.source_id, ProvenanceRecordRow.id)
                ).all()
            )
            persisted_result_sha256 = sha256_canonical(
                {
                    "result_id": result.id,
                    "run_id": result.run_id,
                    "agent_step_id": result.agent_step_id,
                    "tool_name": result.tool_name,
                    "tool_version": result.tool_version,
                    "model_version": result.model_version,
                    "data_version": result.data_version,
                    "feature_version": result.feature_version,
                    "input_hash": result.input_hash,
                    "values": result.values_json,
                    "uncertainty": result.uncertainty_json,
                    "warnings": result.warnings_json,
                    "created_at": PersistentAgentRunInvocationResolver._database_utc(
                        result.created_at
                    ).isoformat(),
                    "provenance": [
                        {
                            "source_id": item.source_id,
                            "source_kind": item.source_kind,
                            "uri": item.uri,
                            "sha256": item.sha256,
                            "description": item.description,
                            "created_at": (
                                PersistentAgentRunInvocationResolver._database_utc(
                                    item.created_at
                                ).isoformat()
                            ),
                        }
                        for item in provenance
                    ],
                }
            )
            binding = session.get(ProjectToolResultBindingRecord, result.id)
            if binding is not None and (
                binding.agent_run_id != row.run_id
                or binding.agent_step_id != dependency.id
                or binding.step_id != dependency.step_id
            ):
                raise AgentRunInvocationAccessError(
                    "persisted Agent step dependency binding is invalid"
                )
            evidence.append(
                {
                    "agent_step_row_id": dependency.id,
                    "step_id": dependency.step_id,
                    "ordinal": dependency.ordinal,
                    "result_id": result.id,
                    "tool_name": result.tool_name,
                    "input_hash": result.input_hash,
                    "persisted_result_sha256": persisted_result_sha256,
                    "project_result_sha256": (
                        binding.result_sha256 if binding is not None else None
                    ),
                    "project_binding_sha256": (
                        binding.binding_sha256 if binding is not None else None
                    ),
                }
            )
        return sha256_canonical(
            {
                "schema_version": "agent-step-dependency-evidence-v1",
                "dependencies": evidence,
            }
        )

    def _approval_evidence(
        self,
        session: object,
        *,
        row: AgentStep,
        plan_hash: str,
    ) -> tuple[str | None, str | None, str]:
        from sqlalchemy.orm import Session

        if not isinstance(session, Session):  # pragma: no cover - internal invariant
            raise TypeError("approval evidence requires a database session")
        if not row.requires_human_approval:
            return (
                None,
                None,
                sha256_canonical(
                    {
                        "schema_version": "agent-step-approval-evidence-v1",
                        "status": "NOT_REQUIRED",
                    }
                ),
            )
        request = session.scalar(
            select(ApprovalRequestRow).where(
                ApprovalRequestRow.run_id == row.run_id,
                ApprovalRequestRow.step_id == row.step_id,
            )
        )
        if (
            request is None
            or request.status != "APPROVED"
            or request.source_plan_hash != plan_hash
            or request.agent_step_id != row.id
            or request.execution_snapshot_sha256
            != row.execution_snapshot_sha256
        ):
            raise AgentRunInvocationAccessError(
                "persisted Agent step approval evidence is invalid"
            )
        actions = tuple(
            session.scalars(
                select(ApprovalAction).where(
                    ApprovalAction.approval_request_id == request.id
                )
            ).all()
        )
        if (
            len(actions) != 1
            or actions[0].action != "APPROVED"
            or self._database_utc(actions[0].acted_at)
            >= self._database_utc(request.expires_at)
        ):
            raise AgentRunInvocationAccessError(
                "persisted Agent step approval action is invalid"
            )
        action = actions[0]
        evidence_sha256 = sha256_canonical(
            {
                "schema_version": "agent-step-approval-evidence-v1",
                "approval_request_id": request.id,
                "approval_kind": request.approval_kind,
                "source_plan_hash": request.source_plan_hash,
                "agent_step_row_id": row.id,
                "step_id": request.step_id,
                "status": request.status,
                "expires_at": self._database_utc(request.expires_at).isoformat(),
                "approval_action_id": action.id,
                "action": action.action,
                "actor_user_id": action.actor_user_id,
                "acted_at": self._database_utc(action.acted_at).isoformat(),
                "reason": action.reason,
            }
        )
        return request.id, action.id, evidence_sha256

    @staticmethod
    def _validate_step_rows(plan: AgentPlan, rows: tuple[AgentStep, ...]) -> None:
        from quanxin_life.agents.orchestrator import ROLE_TOOL_ALLOWLIST

        if len(rows) != len(plan.steps):
            raise AgentRunInvocationAccessError(
                "persisted Agent steps do not match the plan"
            )
        for ordinal, (planned, row) in enumerate(
            zip(plan.steps, rows, strict=True), start=1
        ):
            try:
                tool_name = StandardToolName(planned.tool_name)
            except ValueError as exc:
                raise AgentRunInvocationAccessError(
                    "persisted Agent plan tool is invalid"
                ) from exc
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
            if actual != expected or tool_name not in ROLE_TOOL_ALLOWLIST[planned.role]:
                raise AgentRunInvocationAccessError(
                    "persisted Agent step differs from the secured plan"
                )

    @staticmethod
    def _require_context_matches_snapshot(
        context: VerifiedProjectInvocationContext,
        snapshot: _PersistedStepSnapshot,
    ) -> None:
        if (
            context.project_id != snapshot.project_id
            or context.actor_user_id != snapshot.actor_user_id
            or context.actor_session_id != snapshot.actor_session_id
            or context.agent_run_id != snapshot.agent_run_id
            or context.invocation_source is not ProjectInvocationSource.AGENT
        ):
            raise AgentRunInvocationAccessError(
                "Agent project context does not match persisted run state"
            )

    def _current_time(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Agent run invocation clock must return an aware datetime")
        return value.astimezone(UTC)

    def _sign(self, grant: VerifiedAgentRunInvocationGrant) -> str:
        payload = canonical_json_bytes(
            {
                "agent_run_id": grant.agent_run_id,
                "agent_step_row_id": grant.agent_step_row_id,
                "step_id": grant.step_id,
                "plan_hash": grant.plan_hash,
                "input_hash": grant.input_hash,
                "claim_attempt": grant.claim_attempt,
                "claim_lease_expires_at": grant.claim_lease_expires_at.isoformat(),
                "step_spec_sha256": grant.step_spec_sha256,
                "dependency_evidence_sha256": grant.dependency_evidence_sha256,
                "execution_snapshot_sha256": grant.execution_snapshot_sha256,
                "approval_required": grant.approval_required,
                "approval_request_id": grant.approval_request_id,
                "approval_action_id": grant.approval_action_id,
                "approval_evidence_sha256": grant.approval_evidence_sha256,
                "allowed_tool_names": sorted(
                    item.value for item in grant.allowed_tool_names
                ),
                "project_context_authorization_tag": (
                    grant.project_context._authorization_tag
                ),
                "claim_token_sha256": grant._claim_token_sha256,
            }
        )
        return hmac.digest(self._signing_key, payload, "sha256").hex()

    @staticmethod
    def _claim_digest(value: str) -> str:
        return sha256_canonical({"claim_token": value})

    @staticmethod
    def _database_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @staticmethod
    def _identifier(value: str, field_name: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise AgentRunInvocationAccessError(f"{field_name} must not be blank")
        return normalized


__all__ = [
    "AgentRunInvocationAccessError",
    "AgentRunInvocationValidator",
    "PersistentAgentRunInvocationResolver",
    "VerifiedAgentRunInvocationGrant",
]
