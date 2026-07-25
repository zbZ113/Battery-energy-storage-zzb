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
from quanxin_life.persistence.models import AgentRun, AgentRunDispatch, AgentStep
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
        for digest in (self._claim_token_sha256, self._authorization_tag):
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


@dataclass(frozen=True, slots=True)
class _PersistedStepSnapshot:
    agent_run_id: str
    agent_step_row_id: str
    step_id: str
    plan_hash: str
    tool_name: StandardToolName
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
            frozenset({snapshot.tool_name}),
        )
        actual = (
            grant.agent_step_row_id,
            grant.plan_hash,
            grant.allowed_tool_names,
        )
        if actual != expected:
            raise AgentRunInvocationAccessError(
                "Agent run invocation grant no longer matches persisted state"
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
                snapshot = _PersistedStepSnapshot(
                    agent_run_id=run.id,
                    agent_step_row_id=row.id,
                    step_id=row.step_id,
                    plan_hash=plan.plan_hash,
                    tool_name=tool_name,
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
