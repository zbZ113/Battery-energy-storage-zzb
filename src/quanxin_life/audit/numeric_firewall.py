"""Resolve report numbers only from revalidated ``ToolResult`` values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from threading import RLock
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from quanxin_life.core import EvidenceLevel, ToolResult
from quanxin_life.core.schemas import ContractModel


class AuditLedgerError(ValueError):
    """Base class for failures inside the trusted append-only audit boundary."""


class DuplicateAuditResultError(AuditLedgerError):
    """Raised when one immutable result identifier is registered more than once."""


class InvalidAuditResultError(AuditLedgerError):
    """Raised when an object cannot satisfy the public ToolResult contract."""


class NumericEvidence(ContractModel):
    """One exact numeric assertion linked to a path in a registered tool output."""

    result_id: str
    json_path: str = Field(min_length=1)
    reported_value: float
    evidence_level: EvidenceLevel

    @field_validator("result_id")
    @classmethod
    def require_uuid_result_id(cls, value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc
        return value

    @field_validator("json_path")
    @classmethod
    def require_tool_value_path(cls, value: str) -> str:
        segments = value.split(".")
        if segments[0] != "values" or any(not segment for segment in segments):
            raise ValueError("json_path must start with values and contain nonblank segments")
        return value

    @field_validator("reported_value", mode="before")
    @classmethod
    def require_finite_numeric_value(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError("reported_value must be a finite numeric value")
        return value


class AuditLedger:
    """Append-only index of revalidated tool results for dependent tools and reports."""

    def __init__(self, tool_results: tuple[ToolResult, ...] = ()) -> None:
        self._results: dict[str, ToolResult] = {}
        self._lock = RLock()
        for result in tool_results:
            self.register_result(result)

    def register_result(self, result: ToolResult) -> ToolResult:
        """Append one detached result and reject duplicate or invalid evidence."""

        validated = self._validated_result(result)
        with self._lock:
            if validated.result_id in self._results:
                raise DuplicateAuditResultError(
                    f"duplicate ToolResult result_id: {validated.result_id}"
                )
            self._results[validated.result_id] = validated
        return ToolResult.model_validate(validated.model_dump(mode="json"))

    def ensure_result(self, result: ToolResult) -> ToolResult:
        """Idempotently restore identical evidence and reject ID/content conflicts."""

        validated = self._validated_result(result)
        with self._lock:
            existing = self._results.get(validated.result_id)
            if existing is None:
                self._results[validated.result_id] = validated
            elif existing != validated:
                raise DuplicateAuditResultError(
                    f"conflicting ToolResult content for result_id: {validated.result_id}"
                )
        return ToolResult.model_validate(validated.model_dump(mode="json"))

    @staticmethod
    def _validated_result(result: ToolResult) -> ToolResult:
        try:
            return ToolResult.model_validate(result.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidAuditResultError(
                "ToolResult does not satisfy the public audit contract"
            ) from exc

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        """Return a revalidated, detached result only when it is ledger-registered.

        Dependent domain tools use this method to bind an upstream ``result_id``
        to its audited context before consuming any numerical payload.  A fresh
        Pydantic reconstruction prevents a caller from mutating the ledger's
        internal evidence after resolution.
        """

        try:
            UUID(result_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc

        with self._lock:
            result = self._results.get(result_id)
        if result is None:
            raise ValueError("referenced ToolResult is not registered in the audit ledger")
        return ToolResult.model_validate(result.model_dump(mode="json"))

    def verify_numeric_evidence(self, evidence: NumericEvidence) -> float:
        """Resolve one evidence path and reject altered or nonnumeric report values."""

        numeric_actual = self.resolve_numeric_value(evidence.result_id, evidence.json_path)
        if numeric_actual != evidence.reported_value:
            raise ValueError("reported numeric evidence does not match the ToolResult value")
        return numeric_actual

    def resolve_numeric_value(self, result_id: str, json_path: str) -> float:
        """Resolve one finite numeric field from a ledger-registered result.

        Public report-tool inputs use only a result ID and JSON path.  This
        method keeps the actual business number inside the trusted ledger
        boundary until the renderer creates its internal ``NumericEvidence``.
        """

        result = self.resolve_registered_result(result_id)
        actual = _resolve_mapping_path(result.model_dump(mode="json"), json_path)
        if isinstance(actual, bool) or not isinstance(actual, Real):
            raise ValueError("referenced ToolResult path is not numeric")
        numeric_actual = float(actual)
        if not math.isfinite(numeric_actual):
            raise ValueError("referenced ToolResult numeric value must be finite")
        return numeric_actual


def _resolve_mapping_path(mapping: Mapping[str, Any], json_path: str) -> Any:
    value: Any = mapping
    for segment in json_path.split("."):
        if isinstance(value, Mapping) and segment in value:
            value = value[segment]
            continue
        if (
            isinstance(value, list)
            and segment.isascii()
            and segment.isdecimal()
            and (segment == "0" or not segment.startswith("0"))
        ):
            index = int(segment)
            if index < len(value):
                value = value[index]
                continue
        raise ValueError(f"ToolResult path does not exist: {json_path}")
    return value
