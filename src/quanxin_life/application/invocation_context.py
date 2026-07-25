"""Server-derived project scope for trusted tool invocations."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import select

from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentRunStatus,
    ProjectStatus,
    SessionStatus,
    UserRole,
    UserStatus,
    canonical_json_bytes,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AgentRun,
    Project,
    SessionRecord,
    User,
    UserProjectRole,
)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectInvocationSource(StrEnum):
    """Trusted server entry points that may create a project context."""

    HTTP = "HTTP"
    AGENT = "AGENT"


class ProjectInvocationAccessError(RuntimeError):
    """Raised when an authenticated identity cannot invoke project tools."""


class ProjectInvocationNotFoundError(RuntimeError):
    """Hide absent, invisible and inactive projects behind one error."""


@dataclass(frozen=True, slots=True)
class VerifiedProjectInvocationContext:
    """Immutable project and actor scope created only by trusted services."""

    project_id: str
    actor_user_id: str
    actor_session_id: str
    actor_role: UserRole
    invocation_source: ProjectInvocationSource
    agent_run_id: str | None = None
    _authorization_tag: str = field(repr=False, compare=True, default="")

    def __post_init__(self) -> None:
        for field_name in (
            "project_id",
            "actor_user_id",
            "actor_session_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if not isinstance(self.actor_role, UserRole):
            raise TypeError("actor_role must be a UserRole")
        if not isinstance(self.invocation_source, ProjectInvocationSource):
            raise TypeError("invocation_source must be a ProjectInvocationSource")
        if self.invocation_source is ProjectInvocationSource.HTTP:
            if self.agent_run_id is not None:
                raise ValueError("HTTP invocation context cannot contain agent_run_id")
        elif not isinstance(self.agent_run_id, str) or not self.agent_run_id.strip():
            raise ValueError("Agent invocation context requires a nonblank agent_run_id")
        if (
            len(self._authorization_tag) != 64
            or self._authorization_tag.casefold() != self._authorization_tag
            or any(character not in "0123456789abcdef" for character in self._authorization_tag)
        ):
            raise ValueError("authorization tag must be a lowercase SHA-256 digest")


class ProjectInvocationContextService:
    """Resolve live HTTP identities into ACTIVE, visible project scopes."""

    def __init__(self, session_factory: SessionFactory, *, clock: Clock = _utc_now) -> None:
        self._session_factory = session_factory
        self._clock = clock
        self._signing_key = secrets.token_bytes(32)

    def resolve_http(
        self,
        principal: AuthPrincipal,
        project_id: str,
    ) -> VerifiedProjectInvocationContext:
        if not isinstance(principal, AuthPrincipal):
            raise ProjectInvocationAccessError("authenticated principal is required")
        if principal.must_change_password:
            raise ProjectInvocationAccessError("credential change is required")
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise ProjectInvocationAccessError("role is not allowed for project tools")
        normalized_project_id = project_id.strip() if isinstance(project_id, str) else ""
        if not normalized_project_id:
            raise ProjectInvocationNotFoundError("project scope was not found")

        self._authorize_live_scope(
            project_id=normalized_project_id,
            actor_user_id=principal.user_id,
            actor_session_id=principal.session_id,
            actor_role=principal.role,
        )
        unsigned = VerifiedProjectInvocationContext(
            project_id=normalized_project_id,
            actor_user_id=principal.user_id,
            actor_session_id=principal.session_id,
            actor_role=principal.role,
            invocation_source=ProjectInvocationSource.HTTP,
            _authorization_tag="0" * 64,
        )
        return replace(unsigned, _authorization_tag=self._sign(unsigned))

    def resolve_agent_run(self, run_id: str) -> VerifiedProjectInvocationContext:
        """Issue an AGENT context only from one live persisted run identity."""

        normalized_run_id = run_id.strip() if isinstance(run_id, str) else ""
        if not normalized_run_id:
            raise ProjectInvocationAccessError("persisted Agent run is required")
        with session_scope(self._session_factory) as session:
            run = session.get(AgentRun, normalized_run_id)
            if (
                run is None
                or run.session_id is None
                or not run.plan_hash
                or run.status != AgentRunStatus.RUNNING.value
            ):
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                )
            user = session.get(User, run.created_by_user_id)
            if user is None:
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                )
            try:
                actor_role = UserRole(user.role)
            except ValueError as exc:
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                ) from exc
            if actor_role not in {UserRole.ADMIN, UserRole.MEMBER}:
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                )
            project_id = run.project_id
            actor_user_id = run.created_by_user_id
            actor_session_id = run.session_id

        self._authorize_live_scope(
            project_id=project_id,
            actor_user_id=actor_user_id,
            actor_session_id=actor_session_id,
            actor_role=actor_role,
        )
        self._authorize_agent_run_binding(
            run_id=normalized_run_id,
            project_id=project_id,
            actor_user_id=actor_user_id,
            actor_session_id=actor_session_id,
        )
        unsigned = VerifiedProjectInvocationContext(
            project_id=project_id,
            actor_user_id=actor_user_id,
            actor_session_id=actor_session_id,
            actor_role=actor_role,
            invocation_source=ProjectInvocationSource.AGENT,
            agent_run_id=normalized_run_id,
            _authorization_tag="0" * 64,
        )
        return replace(unsigned, _authorization_tag=self._sign(unsigned))

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        """Verify issuer integrity and live actor/project authorization."""

        if not isinstance(context, VerifiedProjectInvocationContext):
            raise ProjectInvocationAccessError("verified project context is required")
        if not hmac.compare_digest(context._authorization_tag, self._sign(context)):
            raise ProjectInvocationAccessError("project invocation context is not trusted")
        self._authorize_live_scope(
            project_id=context.project_id,
            actor_user_id=context.actor_user_id,
            actor_session_id=context.actor_session_id,
            actor_role=context.actor_role,
        )
        if context.invocation_source is ProjectInvocationSource.AGENT:
            if context.agent_run_id is None:  # pragma: no cover - dataclass invariant
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                )
            self._authorize_agent_run_binding(
                run_id=context.agent_run_id,
                project_id=context.project_id,
                actor_user_id=context.actor_user_id,
                actor_session_id=context.actor_session_id,
            )
        return context

    def _authorize_agent_run_binding(
        self,
        *,
        run_id: str,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
    ) -> None:
        with session_scope(self._session_factory) as session:
            run = session.get(AgentRun, run_id)
            if (
                run is None
                or run.project_id != project_id
                or run.created_by_user_id != actor_user_id
                or run.session_id != actor_session_id
                or run.status != AgentRunStatus.RUNNING.value
                or not run.plan_hash
            ):
                raise ProjectInvocationAccessError(
                    "persisted Agent run is no longer authorized"
                )

    def _authorize_live_scope(
        self,
        *,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
        actor_role: UserRole,
    ) -> None:
        now = self._current_time()
        with session_scope(self._session_factory) as session:
            user = session.get(User, actor_user_id)
            session_record = session.get(SessionRecord, actor_session_id)
            if (
                user is None
                or user.status != UserStatus.ACTIVE.value
                or user.must_change_credential
                or user.role != actor_role.value
                or session_record is None
                or session_record.user_id != actor_user_id
                or session_record.status != SessionStatus.ACTIVE.value
                or session_record.revoked_at is not None
                or session_record.expires_at <= now
            ):
                raise ProjectInvocationAccessError("project actor is no longer authorized")

            project = session.get(Project, project_id)
            if project is None or project.status != ProjectStatus.ACTIVE.value:
                raise ProjectInvocationNotFoundError("project scope was not found")
            if actor_role is UserRole.ADMIN or project.owner_user_id == actor_user_id:
                return
            membership = session.scalar(
                select(UserProjectRole.id).where(
                    UserProjectRole.user_id == actor_user_id,
                    UserProjectRole.project_id == project_id,
                    UserProjectRole.role == actor_role.value,
                )
            )
            if membership is None:
                raise ProjectInvocationNotFoundError("project scope was not found")

    def _current_time(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("project invocation clock must return an aware datetime")
        return value.astimezone(UTC)

    def _sign(self, context: VerifiedProjectInvocationContext) -> str:
        payload = canonical_json_bytes(
            {
                "project_id": context.project_id,
                "actor_user_id": context.actor_user_id,
                "actor_session_id": context.actor_session_id,
                "actor_role": context.actor_role.value,
                "invocation_source": context.invocation_source.value,
                "agent_run_id": context.agent_run_id,
            }
        )
        return hmac.new(self._signing_key, payload, hashlib.sha256).hexdigest()


__all__ = [
    "ProjectInvocationAccessError",
    "ProjectInvocationContextService",
    "ProjectInvocationNotFoundError",
    "ProjectInvocationSource",
    "VerifiedProjectInvocationContext",
]
