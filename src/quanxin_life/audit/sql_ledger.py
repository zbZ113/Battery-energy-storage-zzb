"""Transactional global persistence for integrity-checked ``ToolResult`` values."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from quanxin_life.audit.numeric_firewall import (
    AuditLedger,
    AuditLedgerError,
    DuplicateAuditResultError,
    InvalidAuditResultError,
)
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    GlobalToolResultBindingRecord,
    ProvenanceRecordRow,
    ToolResultRecord,
)

GLOBAL_RESULT_BINDING_SCHEMA_VERSION = "global-tool-result-binding-v1"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqlAuditLedger(AuditLedger):
    """Persist the shared non-project audit boundary across API and worker processes."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        clock: Clock = _utc_now,
    ) -> None:
        super().__init__()
        self._session_factory = session_factory
        self._clock = clock

    def register_result(self, result: ToolResult) -> ToolResult:
        normalized = self._normalized_result(result)
        result_sha256 = sha256_canonical(normalized.model_dump(mode="json"))
        created_at = self._utc(self._clock())
        binding_sha256 = self._binding_sha256(
            result_id=normalized.result_id,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        try:
            with session_scope(self._session_factory) as session:
                if (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        GlobalToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                ):
                    raise DuplicateAuditResultError(
                        f"duplicate ToolResult result_id: {normalized.result_id}"
                    )
                row = ToolResultRecord(
                    id=normalized.result_id,
                    run_id=None,
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
                session.add(row)
                session.flush((row,))
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
                session.add(
                    GlobalToolResultBindingRecord(
                        result_id=normalized.result_id,
                        binding_schema_version=GLOBAL_RESULT_BINDING_SCHEMA_VERSION,
                        result_sha256=result_sha256,
                        binding_sha256=binding_sha256,
                        created_at=created_at,
                    )
                )
                session.flush()
        except DuplicateAuditResultError:
            raise
        except IntegrityError as exc:
            if self._result_exists(normalized.result_id):
                raise DuplicateAuditResultError(
                    f"duplicate ToolResult result_id: {normalized.result_id}"
                ) from exc
            raise AuditLedgerError(
                "global ToolResult persistence integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            raise AuditLedgerError("global ToolResult persistence failed") from exc
        return self._detached(normalized)

    def ensure_result(self, result: ToolResult) -> ToolResult:
        normalized = self._normalized_result(result)
        try:
            existing = self.resolve_registered_result(normalized.result_id)
        except ValueError:
            try:
                return self.register_result(normalized)
            except DuplicateAuditResultError:
                existing = self.resolve_registered_result(normalized.result_id)
        if existing != normalized:
            raise DuplicateAuditResultError(
                f"conflicting ToolResult content for result_id: {normalized.result_id}"
            )
        return self._detached(existing)

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        checked_id = self._result_id(result_id)
        try:
            with session_scope(self._session_factory) as session:
                row = session.get(ToolResultRecord, checked_id)
                binding = session.get(GlobalToolResultBindingRecord, checked_id)
                if row is None or binding is None:
                    raise ValueError(
                        "referenced ToolResult is not registered in the global audit ledger"
                    )
                provenance = tuple(
                    session.scalars(
                        select(ProvenanceRecordRow).where(
                            ProvenanceRecordRow.tool_result_id == checked_id
                        )
                    ).all()
                )
                result = self._result_from_rows(row, provenance)
                self._verify_binding(binding, result)
        except (ValueError, AuditLedgerError):
            raise
        except SQLAlchemyError as exc:
            raise AuditLedgerError("global ToolResult resolution failed") from exc
        return self._detached(result)

    def _verify_binding(
        self,
        binding: GlobalToolResultBindingRecord,
        result: ToolResult,
    ) -> None:
        if binding.binding_schema_version != GLOBAL_RESULT_BINDING_SCHEMA_VERSION:
            raise AuditLedgerError("global ToolResult binding integrity check failed")
        result_sha256 = sha256_canonical(result.model_dump(mode="json"))
        created_at = self._utc(binding.created_at)
        if (
            binding.result_sha256 != result_sha256
            or binding.binding_sha256
            != self._binding_sha256(
                result_id=result.result_id,
                result_sha256=result_sha256,
                created_at=created_at,
            )
        ):
            raise AuditLedgerError("global ToolResult integrity check failed")

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
            raise AuditLedgerError("global ToolResult integrity check failed") from exc
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

    @staticmethod
    def _binding_sha256(
        *,
        result_id: str,
        result_sha256: str,
        created_at: datetime,
    ) -> str:
        return sha256_canonical(
            {
                "binding_schema_version": GLOBAL_RESULT_BINDING_SCHEMA_VERSION,
                "result_id": result_id,
                "result_sha256": result_sha256,
                "created_at": created_at.astimezone(UTC).isoformat(),
            }
        )

    def _result_exists(self, result_id: str) -> bool:
        with session_scope(self._session_factory) as session:
            return session.get(ToolResultRecord, result_id) is not None

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
            raise AuditLedgerError("global ToolResult timestamp must include a timezone")
        return value.astimezone(UTC)

    @staticmethod
    def _detached(result: ToolResult) -> ToolResult:
        return ToolResult.model_validate(result.model_dump(mode="json"))


__all__ = ["GLOBAL_RESULT_BINDING_SCHEMA_VERSION", "SqlAuditLedger"]
