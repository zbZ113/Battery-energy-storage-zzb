"""Project-bound engineering recommendations from authorized ToolResult evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.audit.project_ledger import (
    BoundProjectResultResolver,
    ProjectResultLedger,
)
from quanxin_life.core import Decision, ProvenanceRecord, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

ENGINEERING_RECOMMENDATION_TOOL_VERSION = "engineering-recommendation-tool-v1"
ENGINEERING_RECOMMENDATION_MODEL_VERSION = "engineering-recommendation-rule-engine-v1"
ENGINEERING_RECOMMENDATION_FEATURE_VERSION = "engineering-recommendation-evidence-v1"

_Recommendation = Literal["ADOPTABLE", "RECHECK_REQUIRED", "UNRESOLVED"]
_Comparator = Literal["LT", "LTE", "EQ", "NE", "GTE", "GT"]
_RECOMMENDATION_BY_DECISION: Mapping[Decision, _Recommendation] = {
    Decision.ADMIT: "ADOPTABLE",
    Decision.RECHECK: "RECHECK_REQUIRED",
}
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _RulesModel(ContractModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )


class EngineeringRecommendationToolInput(ContractModel):
    """Public input contains only unique result references and one ruleset ID."""

    result_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    ruleset_id: str = Field(min_length=1, max_length=200)

    @field_validator("result_ids")
    @classmethod
    def require_unique_uuid_result_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for result_id in value:
            try:
                normalized.append(str(UUID(result_id)))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError("result_ids must contain only UUID strings") from exc
        if len(normalized) != len(set(normalized)):
            raise ValueError("result_ids must be unique")
        return tuple(sorted(normalized))

    @field_validator("ruleset_id")
    @classmethod
    def require_nonblank_ruleset_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ruleset_id must not be blank")
        return normalized


class EngineeringRecommendationRule(_RulesModel):
    """One server-owned all-pass comparison over an exact ToolResult selector."""

    rule_id: str = Field(min_length=1, max_length=200)
    result_tool_name: StandardToolName
    result_tool_version: str = Field(min_length=1, max_length=200)
    value_path: str = Field(min_length=1, max_length=500)
    comparator: _Comparator
    threshold: float = Field(allow_inf_nan=False)
    recheck_reason_code: str = Field(min_length=1, max_length=200)

    @field_validator(
        "rule_id",
        "result_tool_version",
        "recheck_reason_code",
    )
    @classmethod
    def require_nonblank_rule_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("rule text must not be blank")
        return normalized

    @field_validator("value_path")
    @classmethod
    def require_values_path(cls, value: str) -> str:
        normalized = value.strip()
        segments = normalized.split(".")
        if segments[0] != "values" or len(segments) < 2 or any(not segment for segment in segments):
            raise ValueError("value_path must be a nonblank values.* path")
        return normalized


class VerifiedEngineeringRecommendationRuleset(_RulesModel):
    """Versioned rule evidence returned only by a trusted server-side resolver."""

    ruleset_id: str = Field(min_length=1, max_length=200)
    ruleset_version: str = Field(min_length=1, max_length=200)
    ruleset_manifest_sha256: Sha256
    unresolved_reason_code: str = Field(min_length=1, max_length=200)
    rules: tuple[EngineeringRecommendationRule, ...] = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator("ruleset_id", "ruleset_version", "unresolved_reason_code")
    @classmethod
    def require_nonblank_ruleset_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("ruleset text must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_unique_rule_ids(self) -> VerifiedEngineeringRecommendationRuleset:
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("ruleset rule_id values must be unique")
        return self


class AuthorizedResultResolver(Protocol):
    """Authorization-scoped result lookup, normally bound to one project ledger."""

    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class VerifiedEngineeringRecommendationRulesetResolver(Protocol):
    """Trusted lookup for one reviewed, versioned engineering ruleset."""

    def resolve_verified_engineering_recommendation_ruleset(
        self,
        ruleset_id: str,
    ) -> VerifiedEngineeringRecommendationRuleset: ...


def execute_engineering_recommendation_tool(
    input_value: EngineeringRecommendationToolInput,
    *,
    result_resolver: AuthorizedResultResolver,
    ruleset_resolver: VerifiedEngineeringRecommendationRulesetResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Evaluate server-owned comparisons over authorized result references only."""

    validated_input = EngineeringRecommendationToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    ruleset = _resolve_verified_ruleset(ruleset_resolver, validated_input.ruleset_id)
    if ruleset.ruleset_id != validated_input.ruleset_id:
        raise ValueError("trusted ruleset_id must match the requested ruleset_id")

    resolved_results = tuple(
        _resolve_authorized_result(result_resolver, result_id)
        for result_id in validated_input.result_ids
    )
    results_by_selector: dict[tuple[str, str], list[ToolResult]] = {}
    for result in resolved_results:
        selector = (result.tool_name, result.tool_version)
        results_by_selector.setdefault(selector, []).append(result)

    allowed_selectors = {
        (rule.result_tool_name.value, rule.result_tool_version) for rule in ruleset.rules
    }
    resolution_issues = [
        f"UNEXPECTED_RESULT_SELECTOR:{result.result_id}"
        for result in resolved_results
        if (result.tool_name, result.tool_version) not in allowed_selectors
    ]
    threshold_evidence: list[dict[str, object]] = []
    evaluated_value_paths: list[str] = []
    recheck_reason_codes: list[str] = []

    for rule in ruleset.rules:
        selector = (rule.result_tool_name.value, rule.result_tool_version)
        matches = results_by_selector.get(selector, [])
        if not matches:
            resolution_issues.append(f"MISSING_RESULT_SELECTOR:{rule.rule_id}")
            continue
        if len(matches) != 1:
            resolution_issues.append(f"AMBIGUOUS_RESULT_SELECTOR:{rule.rule_id}")
            continue
        result = matches[0]
        try:
            actual_value = _resolve_value_path(result.values, rule.value_path)
        except ValueError:
            resolution_issues.append(f"RULE_PATH_UNAVAILABLE:{rule.rule_id}")
            continue
        if isinstance(actual_value, bool) or not isinstance(actual_value, int | float):
            resolution_issues.append(f"RULE_VALUE_NOT_NUMERIC:{rule.rule_id}")
            continue
        if isinstance(actual_value, float) and not math.isfinite(actual_value):
            resolution_issues.append(f"RULE_VALUE_NOT_FINITE:{rule.rule_id}")
            continue

        comparison_passed = _compare(
            actual_value,
            comparator=rule.comparator,
            threshold=rule.threshold,
        )
        evaluated_value_paths.append(rule.value_path)
        threshold_evidence.append(
            {
                "rule_id": rule.rule_id,
                "result_id": result.result_id,
                "result_tool_name": result.tool_name,
                "result_tool_version": result.tool_version,
                "value_path": rule.value_path,
                "comparator": rule.comparator,
                "threshold": rule.threshold,
                "actual_value": actual_value,
                "comparison_passed": comparison_passed,
                "recheck_reason_code": rule.recheck_reason_code,
            }
        )
        if not comparison_passed and rule.recheck_reason_code not in recheck_reason_codes:
            recheck_reason_codes.append(rule.recheck_reason_code)

    recommendation: _Recommendation
    reason_codes: list[str]
    if resolution_issues:
        recommendation = "UNRESOLVED"
        reason_codes = [ruleset.unresolved_reason_code]
    elif recheck_reason_codes:
        recommendation = _RECOMMENDATION_BY_DECISION[Decision.RECHECK]
        reason_codes = recheck_reason_codes
    else:
        recommendation = _RECOMMENDATION_BY_DECISION[Decision.ADMIT]
        reason_codes = []

    created_at = _execution_timestamp(clock)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.MAKE_ENGINEERING_RECOMMENDATION.value,
        tool_version=ENGINEERING_RECOMMENDATION_TOOL_VERSION,
        model_version=ENGINEERING_RECOMMENDATION_MODEL_VERSION,
        data_version=ruleset.ruleset_version,
        feature_version=ENGINEERING_RECOMMENDATION_FEATURE_VERSION,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={
            "recommendation": recommendation,
            "ruleset_id": ruleset.ruleset_id,
            "ruleset_version": ruleset.ruleset_version,
            "ruleset_manifest_sha256": ruleset.ruleset_manifest_sha256,
            "authorized_upstream_result_ids": list(validated_input.result_ids),
            "authorized_result_selectors": _authorized_selectors(ruleset.rules),
            "evaluated_value_paths": evaluated_value_paths,
            "threshold_evidence": threshold_evidence,
            "reason_codes": reason_codes,
            "resolution_issues": resolution_issues,
        },
        uncertainty=None,
        warnings=[*reason_codes, *resolution_issues],
        provenance=_merge_provenance(
            *(result.provenance for result in resolved_results),
            ruleset.provenance,
        ),
        created_at=created_at,
    )


def register_engineering_recommendation_tool(
    registry: ToolRegistry,
    *,
    project_audit_ledger: ProjectResultLedger,
    ruleset_resolver: VerifiedEngineeringRecommendationRulesetResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[EngineeringRecommendationToolInput]:
    """Register recommendation evaluation behind the existing PROJECT boundary."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.MAKE_ENGINEERING_RECOMMENDATION,
            tool_version=ENGINEERING_RECOMMENDATION_TOOL_VERSION,
            input_model=EngineeringRecommendationToolInput,
            executor=None,
            execution_scope=ToolExecutionScope.PROJECT,
            project_executor=lambda input_value, context: (
                execute_engineering_recommendation_tool(
                    input_value,
                    result_resolver=BoundProjectResultResolver(
                        project_audit_ledger,
                        context,
                    ),
                    ruleset_resolver=ruleset_resolver,
                    clock=clock,
                )
            ),
        )
    )


register_project_engineering_recommendation_tool = register_engineering_recommendation_tool


def _resolve_verified_ruleset(
    resolver: VerifiedEngineeringRecommendationRulesetResolver,
    ruleset_id: str,
) -> VerifiedEngineeringRecommendationRuleset:
    resolved = resolver.resolve_verified_engineering_recommendation_ruleset(ruleset_id)
    try:
        return VerifiedEngineeringRecommendationRuleset.model_validate(
            resolved.model_dump(mode="json")
        )
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(f"trusted engineering ruleset violates its contract: {message}") from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted engineering ruleset does not satisfy its contract") from exc


def _resolve_authorized_result(
    resolver: AuthorizedResultResolver,
    result_id: str,
) -> ToolResult:
    result = resolver.resolve_registered_result(result_id)
    if not isinstance(result, ToolResult):
        raise ValueError("authorized result resolver must return a ToolResult")
    try:
        validated = ToolResult.model_validate(result.model_dump(mode="json"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "authorized result does not satisfy the public ToolResult contract"
        ) from exc
    if validated.result_id != result_id:
        raise ValueError("authorized result identity does not match the requested result_id")
    return validated


def _resolve_value_path(values: Mapping[str, object], value_path: str) -> object:
    value: object = values
    for segment in value_path.split(".")[1:]:
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
        raise ValueError(f"ToolResult path does not exist: {value_path}")
    return value


def _compare(
    actual_value: int | float,
    *,
    comparator: _Comparator,
    threshold: float,
) -> bool:
    if comparator == "LT":
        return actual_value < threshold
    if comparator == "LTE":
        return actual_value <= threshold
    if comparator == "EQ":
        return actual_value == threshold
    if comparator == "NE":
        return actual_value != threshold
    if comparator == "GTE":
        return actual_value >= threshold
    return actual_value > threshold


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _authorized_selectors(
    rules: Sequence[EngineeringRecommendationRule],
) -> list[dict[str, str]]:
    selectors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for rule in rules:
        key = (rule.result_tool_name.value, rule.result_tool_version)
        if key in seen:
            continue
        seen.add(key)
        selectors.append(
            {
                "result_tool_name": key[0],
                "result_tool_version": key[1],
            }
        )
    return selectors


def _merge_provenance(*chains: Sequence[ProvenanceRecord]) -> list[ProvenanceRecord]:
    merged: list[ProvenanceRecord] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in chains:
        for record in chain:
            key = (record.source_id, record.uri, record.sha256)
            if key in seen:
                continue
            seen.add(key)
            merged.append(record)
    return merged


__all__ = [
    "ENGINEERING_RECOMMENDATION_FEATURE_VERSION",
    "ENGINEERING_RECOMMENDATION_MODEL_VERSION",
    "ENGINEERING_RECOMMENDATION_TOOL_VERSION",
    "AuthorizedResultResolver",
    "EngineeringRecommendationRule",
    "EngineeringRecommendationToolInput",
    "VerifiedEngineeringRecommendationRuleset",
    "VerifiedEngineeringRecommendationRulesetResolver",
    "execute_engineering_recommendation_tool",
    "register_engineering_recommendation_tool",
    "register_project_engineering_recommendation_tool",
]
