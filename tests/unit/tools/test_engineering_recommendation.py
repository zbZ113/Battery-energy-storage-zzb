from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.tools.engineering_recommendation import (
    ENGINEERING_RECOMMENDATION_TOOL_VERSION,
    EngineeringRecommendationRule,
    EngineeringRecommendationToolInput,
    VerifiedEngineeringRecommendationRuleset,
    execute_engineering_recommendation_tool,
    register_engineering_recommendation_tool,
)
from quanxin_life.tools.registry import (
    StandardToolName,
    ToolExecutionScope,
    ToolRegistry,
)

_PLUS_EIGHT = timezone(timedelta(hours=8))


def _provenance(source_id: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=SourceKind.PREDICTED,
        uri=f"test://engineering-recommendation/{source_id}",
        sha256=sha256_canonical({"source_id": source_id}),
        description=f"Verified fixture provenance for {source_id}",
        created_at=datetime(2026, 8, 13, tzinfo=UTC),
    )


def _result(
    *,
    tool_name: StandardToolName,
    tool_version: str,
    values: dict[str, object],
    source_id: str,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version=tool_version,
        model_version=f"{source_id}-model-v1",
        data_version=f"{source_id}-data-v1",
        feature_version=f"{source_id}-feature-v1",
        input_hash=sha256_canonical({"source_id": source_id}),
        values=values,
        provenance=[_provenance(source_id)],
        created_at=datetime(2026, 8, 13, 1, tzinfo=UTC),
    )


def _rule(
    *,
    rule_id: str,
    tool_name: StandardToolName,
    tool_version: str,
    value_path: str,
    comparator: Literal["LT", "LTE", "EQ", "NE", "GTE", "GT"],
    threshold: float,
    recheck_reason_code: str,
) -> EngineeringRecommendationRule:
    return EngineeringRecommendationRule(
        rule_id=rule_id,
        result_tool_name=tool_name,
        result_tool_version=tool_version,
        value_path=value_path,
        comparator=comparator,
        threshold=threshold,
        recheck_reason_code=recheck_reason_code,
    )


def _ruleset(
    *rules: EngineeringRecommendationRule,
) -> VerifiedEngineeringRecommendationRuleset:
    return VerifiedEngineeringRecommendationRuleset(
        ruleset_id="engineering-release-gate",
        ruleset_version="engineering-release-gate-v3",
        ruleset_manifest_sha256=sha256_canonical(
            {"ruleset": "engineering-release-gate-v3"}
        ),
        unresolved_reason_code="RULESET_EVIDENCE_UNRESOLVED",
        rules=rules,
        provenance=(_provenance("engineering-ruleset-v3"),),
    )


class _ResultResolver:
    def __init__(
        self,
        *results: ToolResult,
        denied_result_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._results = {result.result_id: result for result in results}
        self._denied_result_ids = denied_result_ids
        self.calls: list[str] = []

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        self.calls.append(result_id)
        if result_id in self._denied_result_ids:
            raise PermissionError("ToolResult is not authorized for this project")
        try:
            return self._results[result_id]
        except KeyError as exc:
            raise ValueError("ToolResult is not registered for this project") from exc


class _RulesetResolver:
    def __init__(self, ruleset: VerifiedEngineeringRecommendationRuleset) -> None:
        self.ruleset = ruleset
        self.calls: list[str] = []

    def resolve_verified_engineering_recommendation_ruleset(
        self,
        ruleset_id: str,
    ) -> VerifiedEngineeringRecommendationRuleset:
        self.calls.append(ruleset_id)
        return self.ruleset


def _input(*results: ToolResult) -> EngineeringRecommendationToolInput:
    return EngineeringRecommendationToolInput(
        result_ids=tuple(result.result_id for result in results),
        ruleset_id="engineering-release-gate",
    )


def test_emits_adoptable_only_from_complete_authorized_rule_evidence() -> None:
    cycle = _result(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        tool_version="cycle-life-v7",
        values={"interval": {"lower_cycle": 820.0}},
        source_id="cycle-result",
    )
    quality = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.98},
        source_id="quality-result",
    )
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-lower-cycle",
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="cycle-life-v7",
            value_path="values.interval.lower_cycle",
            comparator="GTE",
            threshold=800.0,
            recheck_reason_code="LOWER_CYCLE_BELOW_RELEASE_GATE",
        ),
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        ),
    )
    resolver = _ResultResolver(cycle, quality)
    ruleset_resolver = _RulesetResolver(ruleset)

    result = execute_engineering_recommendation_tool(
        _input(quality, cycle),
        result_resolver=resolver,
        ruleset_resolver=ruleset_resolver,
        clock=lambda: datetime(2026, 8, 14, 9, tzinfo=_PLUS_EIGHT),
    )

    assert result.tool_name == StandardToolName.MAKE_ENGINEERING_RECOMMENDATION.value
    assert result.tool_version == ENGINEERING_RECOMMENDATION_TOOL_VERSION
    assert result.values["recommendation"] == "ADOPTABLE"
    assert result.values["reason_codes"] == []
    assert result.values["ruleset_id"] == ruleset.ruleset_id
    assert result.values["ruleset_version"] == ruleset.ruleset_version
    assert result.values["ruleset_manifest_sha256"] == ruleset.ruleset_manifest_sha256
    assert result.values["authorized_upstream_result_ids"] == sorted(
        [cycle.result_id, quality.result_id]
    )
    assert result.values["evaluated_value_paths"] == [
        "values.interval.lower_cycle",
        "values.quality_score",
    ]
    assert result.values["threshold_evidence"] == [
        {
            "rule_id": "minimum-lower-cycle",
            "result_id": cycle.result_id,
            "result_tool_name": StandardToolName.PREDICT_CYCLE_LIFE.value,
            "result_tool_version": "cycle-life-v7",
            "value_path": "values.interval.lower_cycle",
            "comparator": "GTE",
            "threshold": 800.0,
            "actual_value": 820.0,
            "comparison_passed": True,
            "recheck_reason_code": "LOWER_CYCLE_BELOW_RELEASE_GATE",
        },
        {
            "rule_id": "minimum-quality-score",
            "result_id": quality.result_id,
            "result_tool_name": StandardToolName.VALIDATE_BATTERY_DATA.value,
            "result_tool_version": "data-quality-v4",
            "value_path": "values.quality_score",
            "comparator": "GTE",
            "threshold": 0.95,
            "actual_value": 0.98,
            "comparison_passed": True,
            "recheck_reason_code": "QUALITY_SCORE_BELOW_RELEASE_GATE",
        },
    ]
    assert result.values["resolution_issues"] == []
    assert result.created_at == datetime(2026, 8, 14, 1, tzinfo=UTC)
    assert resolver.calls == sorted([cycle.result_id, quality.result_id])
    assert ruleset_resolver.calls == [ruleset.ruleset_id]
    assert [record.source_id for record in result.provenance] == [
        *[
            resolver._results[result_id].provenance[0].source_id
            for result_id in sorted([cycle.result_id, quality.result_id])
        ],
        "engineering-ruleset-v3",
    ]


def test_emits_recheck_required_from_a_complete_failed_comparison() -> None:
    cycle = _result(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        tool_version="cycle-life-v7",
        values={"interval": {"lower_cycle": 780.0}},
        source_id="cycle-result",
    )
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-lower-cycle",
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="cycle-life-v7",
            value_path="values.interval.lower_cycle",
            comparator="GTE",
            threshold=800.0,
            recheck_reason_code="LOWER_CYCLE_BELOW_RELEASE_GATE",
        )
    )

    result = execute_engineering_recommendation_tool(
        _input(cycle),
        result_resolver=_ResultResolver(cycle),
        ruleset_resolver=_RulesetResolver(ruleset),
    )

    assert result.values["recommendation"] == "RECHECK_REQUIRED"
    assert result.values["reason_codes"] == ["LOWER_CYCLE_BELOW_RELEASE_GATE"]
    assert result.values["threshold_evidence"][0]["actual_value"] == 780.0
    assert result.values["threshold_evidence"][0]["comparison_passed"] is False


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"quality_score": "not-numeric"},
        {"quality_score": float("nan")},
    ],
)
def test_emits_unresolved_for_missing_type_conflicting_or_nonfinite_evidence(
    values: dict[str, object],
) -> None:
    quality = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.98},
        source_id="quality-result",
    ).model_copy(update={"values": values})
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        )
    )

    result = execute_engineering_recommendation_tool(
        _input(quality),
        result_resolver=_ResultResolver(quality),
        ruleset_resolver=_RulesetResolver(ruleset),
    )

    assert result.values["recommendation"] == "UNRESOLVED"
    assert result.values["reason_codes"] == ["RULESET_EVIDENCE_UNRESOLVED"]
    assert result.values["resolution_issues"]


def test_emits_unresolved_when_one_rule_selector_matches_multiple_results() -> None:
    first = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.98},
        source_id="quality-result-a",
    )
    second = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.97},
        source_id="quality-result-b",
    )
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        )
    )

    result = execute_engineering_recommendation_tool(
        _input(first, second),
        result_resolver=_ResultResolver(first, second),
        ruleset_resolver=_RulesetResolver(ruleset),
    )

    assert result.values["recommendation"] == "UNRESOLVED"
    assert result.values["threshold_evidence"] == []
    assert result.values["resolution_issues"] == [
        "AMBIGUOUS_RESULT_SELECTOR:minimum-quality-score"
    ]


def test_preserves_and_evaluates_a_large_finite_integer_from_tool_evidence() -> None:
    large_actual = 10**4000
    quality = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": large_actual},
        source_id="quality-result",
    )
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        )
    )

    result = execute_engineering_recommendation_tool(
        _input(quality),
        result_resolver=_ResultResolver(quality),
        ruleset_resolver=_RulesetResolver(ruleset),
    )

    assert result.values["recommendation"] == "ADOPTABLE"
    assert result.values["threshold_evidence"][0]["actual_value"] == large_actual


def test_rejects_unauthorized_results_instead_of_downgrading_them() -> None:
    cycle = _result(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        tool_version="cycle-life-v7",
        values={"interval": {"lower_cycle": 820.0}},
        source_id="cycle-result",
    )
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-lower-cycle",
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="cycle-life-v7",
            value_path="values.interval.lower_cycle",
            comparator="GTE",
            threshold=800.0,
            recheck_reason_code="LOWER_CYCLE_BELOW_RELEASE_GATE",
        )
    )

    with pytest.raises(PermissionError, match="not authorized"):
        execute_engineering_recommendation_tool(
            _input(cycle),
            result_resolver=_ResultResolver(
                cycle,
                denied_result_ids=frozenset({cycle.result_id}),
            ),
            ruleset_resolver=_RulesetResolver(ruleset),
        )


def test_rejects_an_authorized_resolver_that_returns_an_invalid_tool_result() -> None:
    quality = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.98},
        source_id="quality-result",
    )
    forged_payload = dict(quality.__dict__)
    forged_payload["input_hash"] = "not-a-sha256"
    forged_payload["provenance"] = []
    forged = ToolResult.model_construct(**forged_payload)
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        )
    )

    with pytest.raises(ValueError, match="public ToolResult contract"):
        execute_engineering_recommendation_tool(
            _input(quality),
            result_resolver=_ResultResolver(forged),
            ruleset_resolver=_RulesetResolver(ruleset),
        )


def test_public_input_rejects_duplicate_ids_and_caller_owned_rule_fields() -> None:
    result_id = str(uuid4())

    with pytest.raises(ValidationError, match="unique"):
        EngineeringRecommendationToolInput(
            result_ids=(result_id, result_id),
            ruleset_id="engineering-release-gate",
        )

    with pytest.raises(ValidationError, match="extra_forbidden"):
        EngineeringRecommendationToolInput.model_validate(
            {
                "result_ids": [result_id],
                "ruleset_id": "engineering-release-gate",
                "value_path": "values.forged",
                "threshold": 1,
                "recommendation": "ADOPTABLE",
                "created_at": "2026-08-14T00:00:00Z",
                "provenance": [],
            }
        )


def test_ruleset_identity_substitution_rejects_and_registration_is_project_only() -> None:
    ruleset = _ruleset(
        _rule(
            rule_id="minimum-quality-score",
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="data-quality-v4",
            value_path="values.quality_score",
            comparator="GTE",
            threshold=0.95,
            recheck_reason_code="QUALITY_SCORE_BELOW_RELEASE_GATE",
        )
    )
    substituted = ruleset.model_copy(update={"ruleset_id": "other-ruleset"})
    quality = _result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        tool_version="data-quality-v4",
        values={"quality_score": 0.98},
        source_id="quality-result",
    )

    with pytest.raises(ValueError, match="ruleset_id"):
        execute_engineering_recommendation_tool(
            _input(quality),
            result_resolver=_ResultResolver(quality),
            ruleset_resolver=_RulesetResolver(substituted),
        )

    registry = ToolRegistry()
    registered = register_engineering_recommendation_tool(
        registry,
        project_audit_ledger=object(),  # type: ignore[arg-type]
        ruleset_resolver=_RulesetResolver(ruleset),
    )

    assert registered.tool_name is StandardToolName.MAKE_ENGINEERING_RECOMMENDATION
    assert registered.definition.execution_scope is ToolExecutionScope.PROJECT
    assert registry.list_schemas() == ()
    project_schemas = registry.list_schemas(execution_scope=ToolExecutionScope.PROJECT)
    assert len(project_schemas) == 1
    assert set(project_schemas[0].input_schema["properties"]) == {
        "result_ids",
        "ruleset_id",
    }
    assert project_schemas[0].input_schema["additionalProperties"] is False
