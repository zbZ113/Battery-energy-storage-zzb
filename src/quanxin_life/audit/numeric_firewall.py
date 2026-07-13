"""Resolve report numbers only from revalidated ``ToolResult`` values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from quanxin_life.core import EvidenceLevel, ToolResult
from quanxin_life.core.schemas import ContractModel


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
    """Immutable-in-practice index of revalidated tool results for report checks."""

    def __init__(self, tool_results: tuple[ToolResult, ...]) -> None:
        if not tool_results:
            raise ValueError("AuditLedger requires at least one ToolResult")
        indexed: dict[str, ToolResult] = {}
        for result in tool_results:
            try:
                validated = ToolResult.model_validate(result.model_dump(mode="json"))
            except (TypeError, ValueError) as exc:
                raise ValueError("ToolResult does not satisfy the public audit contract") from exc
            if validated.result_id in indexed:
                raise ValueError(f"duplicate ToolResult result_id: {validated.result_id}")
            indexed[validated.result_id] = validated
        self._results = indexed

    def verify_numeric_evidence(self, evidence: NumericEvidence) -> float:
        """Resolve one evidence path and reject altered or nonnumeric report values."""

        result = self._results.get(evidence.result_id)
        if result is None:
            raise ValueError("referenced ToolResult is not registered in the audit ledger")
        actual = _resolve_mapping_path(result.model_dump(mode="json"), evidence.json_path)
        if isinstance(actual, bool) or not isinstance(actual, Real):
            raise ValueError("referenced ToolResult path is not numeric")
        numeric_actual = float(actual)
        if not math.isfinite(numeric_actual):
            raise ValueError("referenced ToolResult numeric value must be finite")
        if numeric_actual != evidence.reported_value:
            raise ValueError("reported numeric evidence does not match the ToolResult value")
        return numeric_actual


def _resolve_mapping_path(mapping: Mapping[str, Any], json_path: str) -> Any:
    value: Any = mapping
    for segment in json_path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"ToolResult path does not exist: {json_path}")
        value = value[segment]
    return value
