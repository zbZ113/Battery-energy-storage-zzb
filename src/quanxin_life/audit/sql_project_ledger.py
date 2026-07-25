"""Transactional project bindings for persisted, integrity-checked ToolResults."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from quanxin_life.audit.numeric_firewall import (
    AuditLedgerError,
    DuplicateAuditResultError,
    InvalidAuditResultError,
)
from quanxin_life.audit.project_ledger import (
    ProjectContextValidator,
    ProjectToolResultBinding,
)
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, UserRole
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
)

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )

PROJECT_RESULT_BINDING_SCHEMA_VERSION = "project-tool-result-binding-v1"
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
                if (
                    result_row.run_id != binding.agent_run_id
                    or result_row.agent_step_id is not None
                ):
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
            binding.binding_schema_version
            != PROJECT_RESULT_BINDING_SCHEMA_VERSION
            or binding.tool_name != result.tool_name
            or binding.input_hash != result.input_hash
            or binding.result_sha256
            != sha256_canonical(result.model_dump(mode="json"))
        ):
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        if (
            invocation_source is ProjectInvocationSource.HTTP
            and binding.agent_run_id is not None
        ) or (
            invocation_source is ProjectInvocationSource.AGENT
            and not binding.agent_run_id
        ):
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        expected_hash = sha256_canonical(
            cls._binding_hash_payload(
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
        )
        if expected_hash != binding.binding_sha256:
            raise AuditLedgerError("project ToolResult binding integrity check failed")

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
