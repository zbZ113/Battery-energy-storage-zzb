"""Transactional project bindings for persisted, integrity-checked ToolResults."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from quanxin_life.audit.numeric_firewall import (
    AuditLedgerError,
    DuplicateAuditResultError,
    InvalidAuditResultError,
)
from quanxin_life.audit.project_ledger import (
    ProjectContextValidator,
    ProjectToolResultBinding,
)
from quanxin_life.core import (
    AgentDispatchStatus,
    AgentRunStatus,
    AgentStepStatus,
    ProjectStatus,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
)
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentEvent,
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    ApprovalAction,
    ApprovalRequestRow,
    Project,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
    User,
    UserProjectRole,
)

if TYPE_CHECKING:
    from quanxin_life.application.agent_run_invocation import (
        AgentRunInvocationValidator,
        VerifiedAgentRunInvocationGrant,
    )
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )

PROJECT_RESULT_BINDING_SCHEMA_VERSION = "project-tool-result-binding-v1"
AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION = "project-tool-result-binding-v2"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqlProjectAuditLedger:
    """Persist ToolResult bytes and project ownership in one SQL transaction."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_validator: ProjectContextValidator,
        clock: Clock = _utc_now,
    ) -> None:
        if not callable(getattr(context_validator, "revalidate", None)):
            raise TypeError("context_validator must revalidate project contexts")
        self._session_factory = session_factory
        self._context_validator = context_validator
        self._clock = clock

    @property
    def context_validator(self) -> ProjectContextValidator:
        return self._context_validator

    def register_result(
        self,
        context: VerifiedProjectInvocationContext,
        result: ToolResult,
    ) -> ToolResult:
        """Atomically append one normalized result, provenance and project binding."""

        verified = self._require_context(context)
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        if verified.invocation_source is ProjectInvocationSource.AGENT:
            raise AuditLedgerError(
                "AGENT project results require an exact Agent step commit"
            )
        normalized = self._normalized_result(result)
        created_at = self._utc(self._clock())
        result_sha256 = sha256_canonical(normalized.model_dump(mode="json"))
        binding_values = self._binding_values(
            verified,
            normalized,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        try:
            with session_scope(self._session_factory) as session:
                if (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                ):
                    raise DuplicateAuditResultError(
                        f"duplicate ToolResult result_id: {normalized.result_id}"
                    )
                session.add(
                    ToolResultRecord(
                        id=normalized.result_id,
                        run_id=verified.agent_run_id,
                        agent_step_id=None,
                        tool_name=normalized.tool_name,
                        tool_version=normalized.tool_version,
                        model_version=normalized.model_version,
                        data_version=normalized.data_version,
                        feature_version=normalized.feature_version,
                        input_hash=normalized.input_hash,
                        values_json=dict(normalized.values),
                        uncertainty_json=(
                            dict(normalized.uncertainty)
                            if normalized.uncertainty is not None
                            else None
                        ),
                        warnings_json=list(normalized.warnings),
                        created_at=normalized.created_at,
                    )
                )
                for item in normalized.provenance:
                    session.add(
                        ProvenanceRecordRow(
                            id=str(uuid4()),
                            tool_result_id=normalized.result_id,
                            source_id=item.source_id,
                            source_kind=item.source_kind.value,
                            uri=item.uri,
                            sha256=item.sha256,
                            description=item.description,
                            created_at=item.created_at,
                        )
                    )
                session.add(ProjectToolResultBindingRecord(**binding_values))
                session.flush()
        except DuplicateAuditResultError:
            raise
        except IntegrityError as exc:
            with session_scope(self._session_factory) as session:
                duplicate_exists = (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                )
            if duplicate_exists:
                raise DuplicateAuditResultError(
                    f"duplicate ToolResult result_id: {normalized.result_id}"
                ) from exc
            raise AuditLedgerError(
                "project ToolResult persistence integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            raise AuditLedgerError("project ToolResult persistence failed") from exc
        return ToolResult.model_validate(normalized.model_dump(mode="json"))

    @classmethod
    def _require_live_agent_scope(
        cls,
        session: Session,
        *,
        grant: VerifiedAgentRunInvocationGrant,
        run: AgentRun | None,
        dispatch: AgentRunDispatch | None,
        step: AgentStep | None,
        now: datetime,
    ) -> None:
        context = grant.project_context
        actor = session.get(User, context.actor_user_id)
        actor_session = session.get(SessionRecord, context.actor_session_id)
        project = session.get(Project, context.project_id)
        membership = session.scalar(
            select(UserProjectRole).where(
                UserProjectRole.user_id == context.actor_user_id,
                UserProjectRole.project_id == context.project_id,
            )
        )
        visible = project is not None and (
            project.owner_user_id == context.actor_user_id or membership is not None
        )
        if (
            actor is None
            or actor.status != UserStatus.ACTIVE.value
            or actor.must_change_credential
            or actor.role != context.actor_role.value
            or actor_session is None
            or actor_session.user_id != context.actor_user_id
            or actor_session.status != SessionStatus.ACTIVE.value
            or cls._database_utc(actor_session.expires_at) <= now
            or project is None
            or project.status != ProjectStatus.ACTIVE.value
            or not visible
            or run is None
            or run.status != AgentRunStatus.RUNNING.value
            or run.project_id != context.project_id
            or run.created_by_user_id != context.actor_user_id
            or run.session_id != context.actor_session_id
            or run.plan_hash != grant.plan_hash
            or dispatch is None
            or dispatch.status != AgentDispatchStatus.DISPATCHED.value
            or dispatch.plan_hash != grant.plan_hash
            or step is None
            or step.run_id != grant.agent_run_id
            or step.step_id != grant.step_id
            or step.tool_name
            != next(iter(grant.allowed_tool_names)).value
            or step.status != AgentStepStatus.RUNNING.value
            or step.attempts != grant.claim_attempt
            or not step.claim_token
            or sha256_canonical({"claim_token": step.claim_token})
            != grant._claim_token_sha256
            or step.execution_claim_sha256 != grant._claim_token_sha256
            or step.lease_expires_at is None
            or cls._database_utc(step.lease_expires_at) <= now
            or cls._database_utc(step.lease_expires_at)
            != grant.claim_lease_expires_at
            or step.resolved_input_hash != grant.input_hash
            or step.execution_snapshot_sha256
            not in {None, grant.execution_snapshot_sha256}
            or step.dependency_evidence_sha256
            not in {None, grant.dependency_evidence_sha256}
        ):
            raise AuditLedgerError("Agent step execution claim is stale")
        if grant.approval_required:
            request = session.get(ApprovalRequestRow, grant.approval_request_id)
            action = session.get(ApprovalAction, grant.approval_action_id)
            if (
                request is None
                or action is None
                or request.run_id != grant.agent_run_id
                or request.step_id != grant.step_id
                or request.source_plan_hash != grant.plan_hash
                or request.status != "APPROVED"
                or request.agent_step_id != grant.agent_step_row_id
                or request.execution_snapshot_sha256
                != grant.execution_snapshot_sha256
                or action.approval_request_id != request.id
                or action.action != "APPROVED"
                or cls._database_utc(action.acted_at)
                >= cls._database_utc(request.expires_at)
            ):
                raise AuditLedgerError("Agent step approval evidence is stale")

    @staticmethod
    def _database_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def commit_agent_step_result(
        self,
        *,
        grant: VerifiedAgentRunInvocationGrant,
        result: ToolResult,
        grant_validator: AgentRunInvocationValidator,
    ) -> ToolResult:
        """Atomically fence, bind and complete one exact Agent step result."""

        verified = grant_validator.revalidate(grant)
        normalized = self._normalized_result(result)
        now = self._utc(self._clock())
        if (
            normalized.tool_name
            != next(iter(verified.allowed_tool_names)).value
            or normalized.input_hash != verified.input_hash
        ):
            raise AuditLedgerError(
                "Agent ToolResult does not match the frozen step invocation"
            )
        try:
            with session_scope(self._session_factory) as session:
                run = session.scalar(
                    select(AgentRun)
                    .where(AgentRun.id == verified.agent_run_id)
                    .with_for_update()
                )
                dispatch = session.scalar(
                    select(AgentRunDispatch)
                    .where(AgentRunDispatch.run_id == verified.agent_run_id)
                    .with_for_update()
                )
                step = session.scalar(
                    select(AgentStep)
                    .where(AgentStep.id == verified.agent_step_row_id)
                    .with_for_update()
                )
                self._require_live_agent_scope(
                    session,
                    grant=verified,
                    run=run,
                    dispatch=dispatch,
                    step=step,
                    now=now,
                )
                try:
                    grant_validator.revalidate_in_session(session, verified)
                except RuntimeError as exc:
                    raise AuditLedgerError(
                        "Agent step frozen input, approval, or dependency evidence is stale"
                    ) from exc
                if (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                ):
                    raise DuplicateAuditResultError(
                        f"duplicate ToolResult result_id: {normalized.result_id}"
                    )
                assert run is not None and step is not None
                session.add(
                    ToolResultRecord(
                        id=normalized.result_id,
                        run_id=verified.agent_run_id,
                        agent_step_id=verified.agent_step_row_id,
                        tool_name=normalized.tool_name,
                        tool_version=normalized.tool_version,
                        model_version=normalized.model_version,
                        data_version=normalized.data_version,
                        feature_version=normalized.feature_version,
                        input_hash=normalized.input_hash,
                        values_json=dict(normalized.values),
                        uncertainty_json=(
                            dict(normalized.uncertainty)
                            if normalized.uncertainty is not None
                            else None
                        ),
                        warnings_json=list(normalized.warnings),
                        created_at=normalized.created_at,
                    )
                )
                for item in normalized.provenance:
                    session.add(
                        ProvenanceRecordRow(
                            id=str(uuid4()),
                            tool_result_id=normalized.result_id,
                            source_id=item.source_id,
                            source_kind=item.source_kind.value,
                            uri=item.uri,
                            sha256=item.sha256,
                            description=item.description,
                            created_at=item.created_at,
                        )
                    )
                result_sha256 = sha256_canonical(
                    normalized.model_dump(mode="json")
                )
                binding_values = self._agent_binding_values(
                    verified,
                    normalized,
                    result_sha256=result_sha256,
                    created_at=now,
                )
                session.add(ProjectToolResultBindingRecord(**binding_values))
                if step.execution_snapshot_sha256 is None:
                    step.execution_snapshot_sha256 = (
                        verified.execution_snapshot_sha256
                    )
                if step.dependency_evidence_sha256 is None:
                    step.dependency_evidence_sha256 = (
                        verified.dependency_evidence_sha256
                    )
                step.status = AgentStepStatus.COMPLETED.value
                step.claim_token = None
                step.lease_expires_at = None
                step.last_error_code = None
                step.completed_at = now
                run.updated_at = now
                sequence = session.scalar(
                    select(func.max(AgentEvent.sequence)).where(
                        AgentEvent.run_id == verified.agent_run_id
                    )
                )
                session.add(
                    AgentEvent(
                        id=str(uuid4()),
                        run_id=verified.agent_run_id,
                        sequence=(sequence or 0) + 1,
                        event_type="STEP_COMPLETED",
                        payload_json={
                            "step_id": verified.step_id,
                            "tool_name": normalized.tool_name,
                            "result_id": normalized.result_id,
                        },
                        created_at=now,
                    )
                )
                session.flush()
        except (DuplicateAuditResultError, AuditLedgerError):
            raise
        except IntegrityError as exc:
            raise AuditLedgerError(
                "Agent step result persistence integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            raise AuditLedgerError("Agent step result persistence failed") from exc
        return ToolResult.model_validate(normalized.model_dump(mode="json"))

    def resolve_registered_result(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ToolResult:
        """Resolve one detached result only inside its live project scope."""

        verified = self._require_context(context)
        result, _binding = self._resolve_rows(verified, result_id)
        return result

    def resolve_binding(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ProjectToolResultBinding:
        """Resolve immutable non-numeric ownership metadata after full verification."""

        verified = self._require_context(context)
        _result, binding = self._resolve_rows(verified, result_id)
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        try:
            actor_role = UserRole(binding.actor_role)
            invocation_source = ProjectInvocationSource(binding.invocation_source)
        except ValueError as exc:  # pragma: no cover - checked in _resolve_rows
            raise AuditLedgerError("project ToolResult binding integrity check failed") from exc
        return ProjectToolResultBinding(
            result_id=binding.result_id,
            project_id=binding.project_id,
            actor_user_id=binding.actor_user_id,
            actor_session_id=binding.actor_session_id,
            actor_role=actor_role,
            invocation_source=invocation_source,
            agent_run_id=binding.agent_run_id,
            tool_name=binding.tool_name,
            input_hash=binding.input_hash,
            agent_step_id=binding.agent_step_id,
            step_id=binding.step_id,
            plan_hash=binding.plan_hash,
            claim_attempt=binding.claim_attempt,
            execution_snapshot_sha256=binding.execution_snapshot_sha256,
            approval_request_id=binding.approval_request_id,
        )

    def _resolve_rows(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> tuple[ToolResult, ProjectToolResultBindingRecord]:
        normalized_id = self._result_id(result_id)
        try:
            with session_scope(self._session_factory) as session:
                binding = session.scalar(
                    select(ProjectToolResultBindingRecord).where(
                        ProjectToolResultBindingRecord.result_id == normalized_id,
                        ProjectToolResultBindingRecord.project_id == context.project_id,
                    )
                )
                result_row = session.get(ToolResultRecord, normalized_id)
                if binding is None or result_row is None:
                    raise ValueError("ToolResult is not registered for this project")
                if result_row.run_id != binding.agent_run_id:
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                provenance_rows = tuple(
                    session.scalars(
                        select(ProvenanceRecordRow).where(
                            ProvenanceRecordRow.tool_result_id == normalized_id
                        )
                    ).all()
                )
                actor_session = session.get(SessionRecord, binding.actor_session_id)
                if (
                    actor_session is None
                    or actor_session.user_id != binding.actor_user_id
                ):
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                if binding.invocation_source == "AGENT":
                    agent_run = session.get(AgentRun, binding.agent_run_id)
                    agent_step = session.get(AgentStep, binding.agent_step_id)
                    if (
                        agent_run is None
                        or agent_run.project_id != binding.project_id
                        or agent_run.created_by_user_id != binding.actor_user_id
                        or agent_run.session_id != binding.actor_session_id
                        or agent_step is None
                        or agent_step.run_id != binding.agent_run_id
                        or agent_step.step_id != binding.step_id
                        or result_row.agent_step_id != binding.agent_step_id
                        or agent_step.execution_claim_sha256
                        != binding.claim_token_sha256
                        or agent_step.resolved_input_hash != binding.input_hash
                        or agent_step.execution_snapshot_sha256
                        != binding.execution_snapshot_sha256
                        or agent_step.dependency_evidence_sha256
                        != binding.dependency_evidence_sha256
                    ):
                        raise AuditLedgerError(
                            "project ToolResult binding integrity check failed"
                        )
                elif result_row.agent_step_id is not None:
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                result = self._result_from_rows(result_row, provenance_rows)
                self._verify_binding(binding, result)
                detached_binding = ProjectToolResultBindingRecord(
                    **{
                        column.name: getattr(binding, column.name)
                        for column in ProjectToolResultBindingRecord.__table__.columns
                    }
                )
        except (ValueError, AuditLedgerError):
            raise
        except (SQLAlchemyError, TypeError) as exc:
            raise AuditLedgerError(
                "project ToolResult persistence could not be verified"
            ) from exc
        return result, detached_binding

    @classmethod
    def _verify_binding(
        cls,
        binding: ProjectToolResultBindingRecord,
        result: ToolResult,
    ) -> None:
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        try:
            actor_role = UserRole(binding.actor_role)
            invocation_source = ProjectInvocationSource(binding.invocation_source)
        except ValueError as exc:
            raise AuditLedgerError(
                "project ToolResult binding integrity check failed"
            ) from exc
        if (
            binding.tool_name != result.tool_name
            or binding.input_hash != result.input_hash
            or binding.result_sha256
            != sha256_canonical(result.model_dump(mode="json"))
        ):
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        if binding.binding_schema_version == PROJECT_RESULT_BINDING_SCHEMA_VERSION:
            if (
                invocation_source is not ProjectInvocationSource.HTTP
                or binding.agent_run_id is not None
                or any(
                    value is not None
                    for value in (
                        binding.agent_step_id,
                        binding.step_id,
                        binding.plan_hash,
                        binding.claim_token_sha256,
                        binding.claim_attempt,
                        binding.claim_lease_expires_at,
                        binding.execution_snapshot_sha256,
                        binding.dependency_evidence_sha256,
                        binding.approval_required,
                        binding.approval_request_id,
                        binding.approval_action_id,
                        binding.approval_evidence_sha256,
                    )
                )
            ):
                raise AuditLedgerError(
                    "project ToolResult binding integrity check failed"
                )
            payload = cls._binding_hash_payload(
                result_id=binding.result_id,
                project_id=binding.project_id,
                actor_user_id=binding.actor_user_id,
                actor_session_id=binding.actor_session_id,
                actor_role=actor_role.value,
                invocation_source=invocation_source.value,
                agent_run_id=binding.agent_run_id,
                tool_name=binding.tool_name,
                input_hash=binding.input_hash,
                result_sha256=binding.result_sha256,
                created_at=binding.created_at,
            )
        elif (
            binding.binding_schema_version
            == AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION
        ):
            if (
                invocation_source is not ProjectInvocationSource.AGENT
                or not binding.agent_run_id
                or not binding.agent_step_id
                or not binding.step_id
                or not binding.plan_hash
                or not binding.claim_token_sha256
                or binding.claim_attempt is None
                or binding.claim_lease_expires_at is None
                or not binding.execution_snapshot_sha256
                or not binding.dependency_evidence_sha256
                or binding.approval_required is None
                or not binding.approval_evidence_sha256
            ):
                raise AuditLedgerError(
                    "project ToolResult binding integrity check failed"
                )
            payload = cls._agent_binding_hash_payload(
                result_id=binding.result_id,
                project_id=binding.project_id,
                actor_user_id=binding.actor_user_id,
                actor_session_id=binding.actor_session_id,
                actor_role=actor_role.value,
                agent_run_id=binding.agent_run_id,
                agent_step_id=binding.agent_step_id,
                step_id=binding.step_id,
                plan_hash=binding.plan_hash,
                claim_token_sha256=binding.claim_token_sha256,
                claim_attempt=binding.claim_attempt,
                claim_lease_expires_at=binding.claim_lease_expires_at,
                execution_snapshot_sha256=binding.execution_snapshot_sha256,
                dependency_evidence_sha256=binding.dependency_evidence_sha256,
                approval_required=binding.approval_required,
                approval_request_id=binding.approval_request_id,
                approval_action_id=binding.approval_action_id,
                approval_evidence_sha256=binding.approval_evidence_sha256,
                tool_name=binding.tool_name,
                input_hash=binding.input_hash,
                result_sha256=binding.result_sha256,
                created_at=binding.created_at,
            )
        else:
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        expected_hash = sha256_canonical(payload)
        if expected_hash != binding.binding_sha256:
            raise AuditLedgerError("project ToolResult binding integrity check failed")

    @classmethod
    def _agent_binding_values(
        cls,
        grant: VerifiedAgentRunInvocationGrant,
        result: ToolResult,
        *,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        context = grant.project_context
        payload = cls._agent_binding_hash_payload(
            result_id=result.result_id,
            project_id=context.project_id,
            actor_user_id=context.actor_user_id,
            actor_session_id=context.actor_session_id,
            actor_role=context.actor_role.value,
            agent_run_id=grant.agent_run_id,
            agent_step_id=grant.agent_step_row_id,
            step_id=grant.step_id,
            plan_hash=grant.plan_hash,
            claim_token_sha256=grant._claim_token_sha256,
            claim_attempt=grant.claim_attempt,
            claim_lease_expires_at=grant.claim_lease_expires_at,
            execution_snapshot_sha256=grant.execution_snapshot_sha256,
            dependency_evidence_sha256=grant.dependency_evidence_sha256,
            approval_required=grant.approval_required,
            approval_request_id=grant.approval_request_id,
            approval_action_id=grant.approval_action_id,
            approval_evidence_sha256=grant.approval_evidence_sha256,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        return payload | {
            "binding_sha256": sha256_canonical(payload),
            "claim_lease_expires_at": grant.claim_lease_expires_at,
            "created_at": created_at,
        }

    @staticmethod
    def _agent_binding_hash_payload(
        *,
        result_id: str,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
        actor_role: str,
        agent_run_id: str,
        agent_step_id: str,
        step_id: str,
        plan_hash: str,
        claim_token_sha256: str,
        claim_attempt: int,
        claim_lease_expires_at: datetime,
        execution_snapshot_sha256: str,
        dependency_evidence_sha256: str,
        approval_required: bool,
        approval_request_id: str | None,
        approval_action_id: str | None,
        approval_evidence_sha256: str,
        tool_name: str,
        input_hash: str,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        return {
            "result_id": result_id,
            "binding_schema_version": AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION,
            "project_id": project_id,
            "actor_user_id": actor_user_id,
            "actor_session_id": actor_session_id,
            "actor_role": actor_role,
            "invocation_source": "AGENT",
            "agent_run_id": agent_run_id,
            "agent_step_id": agent_step_id,
            "step_id": step_id,
            "plan_hash": plan_hash,
            "claim_token_sha256": claim_token_sha256,
            "claim_attempt": claim_attempt,
            "claim_lease_expires_at": claim_lease_expires_at.astimezone(UTC).isoformat(),
            "execution_snapshot_sha256": execution_snapshot_sha256,
            "dependency_evidence_sha256": dependency_evidence_sha256,
            "approval_required": approval_required,
            "approval_request_id": approval_request_id,
            "approval_action_id": approval_action_id,
            "approval_evidence_sha256": approval_evidence_sha256,
            "tool_name": tool_name,
            "input_hash": input_hash,
            "result_sha256": result_sha256,
            "created_at": created_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def _binding_values(
        cls,
        context: VerifiedProjectInvocationContext,
        result: ToolResult,
        *,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        payload = cls._binding_hash_payload(
            result_id=result.result_id,
            project_id=context.project_id,
            actor_user_id=context.actor_user_id,
            actor_session_id=context.actor_session_id,
            actor_role=context.actor_role.value,
            invocation_source=context.invocation_source.value,
            agent_run_id=context.agent_run_id,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        return payload | {
            "binding_sha256": sha256_canonical(payload),
            "created_at": created_at,
        }

    @staticmethod
    def _binding_hash_payload(
        *,
        result_id: str,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
        actor_role: str,
        invocation_source: str,
        agent_run_id: str | None,
        tool_name: str,
        input_hash: str,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        return {
            "result_id": result_id,
            "binding_schema_version": PROJECT_RESULT_BINDING_SCHEMA_VERSION,
            "project_id": project_id,
            "actor_user_id": actor_user_id,
            "actor_session_id": actor_session_id,
            "actor_role": actor_role,
            "invocation_source": invocation_source,
            "agent_run_id": agent_run_id,
            "tool_name": tool_name,
            "input_hash": input_hash,
            "result_sha256": result_sha256,
            "created_at": created_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def _result_from_rows(
        cls,
        row: ToolResultRecord,
        provenance_rows: tuple[ProvenanceRecordRow, ...],
    ) -> ToolResult:
        try:
            result = ToolResult(
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
                        created_at=cls._utc(item.created_at),
                    )
                    for item in provenance_rows
                ],
                created_at=cls._utc(row.created_at),
            )
        except (TypeError, ValueError) as exc:
            raise AuditLedgerError("project ToolResult integrity check failed") from exc
        return cls._normalized_result(result)

    @staticmethod
    def _normalized_result(result: ToolResult) -> ToolResult:
        try:
            validated = ToolResult.model_validate(result.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidAuditResultError(
                "ToolResult does not satisfy the public audit contract"
            ) from exc
        payload = validated.model_dump(mode="json")
        payload["provenance"] = sorted(
            payload["provenance"],
            key=lambda item: (
                item["source_id"],
                item["source_kind"],
                item["uri"],
                item["sha256"],
                item["description"],
                item["created_at"],
            ),
        )
        return ToolResult.model_validate(payload)

    def _require_context(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        from quanxin_life.application.invocation_context import (
            VerifiedProjectInvocationContext,
        )

        if not isinstance(context, VerifiedProjectInvocationContext):
            raise TypeError("verified project invocation context is required")
        return self._context_validator.revalidate(context)

    @staticmethod
    def _result_id(value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc
        return value

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise AuditLedgerError("project ToolResult timestamp must include a timezone")
        return value.astimezone(UTC)


__all__ = ["PROJECT_RESULT_BINDING_SCHEMA_VERSION", "SqlProjectAuditLedger"]
