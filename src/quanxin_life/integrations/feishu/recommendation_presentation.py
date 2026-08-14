"""Controlled Chinese presentation for audited engineering recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from quanxin_life.core import ToolResult
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.engineering_recommendation import (
    ENGINEERING_RECOMMENDATION_TOOL_VERSION,
)

_LABELS = {
    "ADOPTABLE": "建议采用",
    "RECHECK_REQUIRED": "建议复检",
    "UNRESOLVED": "暂无法判断",
}
_REASONS = {
    "ADOPTABLE": "全部受审规则均通过",
    "RECHECK_REQUIRED": "至少一项受审规则未通过, 建议复检",
    "UNRESOLVED": "证据缺失或不一致, 暂无法判断",
}


@dataclass(frozen=True, slots=True)
class EngineeringRecommendationPresentation:
    outcome: str
    label: str
    reason: str
    ruleset_version: str


def engineering_recommendation_presentation(
    result: ToolResult,
) -> EngineeringRecommendationPresentation:
    checked = ToolResult.model_validate(result.model_dump(mode="json"))
    if (
        checked.tool_name != StandardToolName.MAKE_ENGINEERING_RECOMMENDATION.value
        or checked.tool_version != ENGINEERING_RECOMMENDATION_TOOL_VERSION
    ):
        raise ValueError("engineering recommendation result version is unsupported")
    outcome = checked.values.get("recommendation")
    if not isinstance(outcome, str) or outcome not in _LABELS:
        raise ValueError("engineering recommendation outcome is invalid")
    ruleset_version = _safe_text(
        checked.values.get("ruleset_version"),
        field_name="ruleset_version",
    )
    reason_codes = _safe_text_list(
        checked.values.get("reason_codes"),
        field_name="reason_codes",
    )
    resolution_issues = _safe_text_list(
        checked.values.get("resolution_issues"),
        field_name="resolution_issues",
    )
    if checked.warnings != [*reason_codes, *resolution_issues]:
        raise ValueError("engineering recommendation warnings are inconsistent")
    if outcome == "ADOPTABLE" and (reason_codes or resolution_issues):
        raise ValueError("adoptable recommendation cannot carry rejection reasons")
    if outcome == "RECHECK_REQUIRED" and (
        not reason_codes or resolution_issues
    ):
        raise ValueError("recheck recommendation reason evidence is invalid")
    if outcome == "UNRESOLVED" and (
        not reason_codes or not resolution_issues
    ):
        raise ValueError("unresolved recommendation evidence is invalid")
    return EngineeringRecommendationPresentation(
        outcome=outcome,
        label=_LABELS[outcome],
        reason=_REASONS[outcome],
        ruleset_version=ruleset_version,
    )


def _safe_text(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 200
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"engineering recommendation {field_name} is invalid")
    return normalized


def _safe_text_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"engineering recommendation {field_name} is invalid")
    return tuple(_safe_text(item, field_name=field_name) for item in value)


__all__ = [
    "EngineeringRecommendationPresentation",
    "engineering_recommendation_presentation",
]
