"""Deterministic Markdown rendering for audited engineering recommendations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from quanxin_life.core import ToolResult

_RECOMMENDATION_LABELS = {
    "ADOPTABLE": "建议采用",
    "RECHECK_REQUIRED": "建议复检",
    "UNRESOLVED": "暂无法判断",
}


def render_engineering_recommendation_markdown(result: ToolResult) -> str:
    """Render only values already present in one validated recommendation result."""

    checked = ToolResult.model_validate(result.model_dump(mode="json"))
    recommendation = checked.values.get("recommendation")
    if not isinstance(recommendation, str):
        raise ValueError("recommendation result has no controlled outcome")
    label = _RECOMMENDATION_LABELS.get(recommendation)
    if label is None:
        raise ValueError("recommendation result has an invalid controlled outcome")
    ruleset_id = _markdown_text(checked.values.get("ruleset_id"))
    ruleset_version = _markdown_text(checked.values.get("ruleset_version"))
    reason_codes = _text_list(checked.values.get("reason_codes"), "reason_codes")
    resolution_issues = _text_list(
        checked.values.get("resolution_issues"),
        "resolution_issues",
    )
    threshold_evidence = checked.values.get("threshold_evidence")
    if not isinstance(threshold_evidence, list):
        raise ValueError("recommendation threshold evidence is invalid")
    lines = [
        "# 工程综合建议审计报告",
        "",
        f"- 综合建议: {label}",
        f"- 规则集: {ruleset_id}",
        f"- 规则集版本: {ruleset_version}",
        f"- 推荐结果 ID: {checked.result_id}",
        "",
        "## 原因与缺口",
        "",
    ]
    messages = [*reason_codes, *resolution_issues]
    lines.extend(f"- {_markdown_text(item)}" for item in messages)
    if not messages:
        lines.append("- 已登记证据满足全部受审规则。")
    lines.extend(("", "## 受审规则证据", ""))
    if not threshold_evidence:
        lines.append("当前没有可用于阈值核验的完整数值证据。")
        return "\n".join(lines) + "\n"
    lines.extend(
        (
            "| 规则 | 来源结果 | JSON 路径 | 比较符 | 审核阈值 | 实际值 | 通过 |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        )
    )
    for item in threshold_evidence:
        if not isinstance(item, Mapping):
            raise ValueError("recommendation threshold evidence entry is invalid")
        threshold = _finite_number(item.get("threshold"), field_name="threshold")
        actual = _finite_number(item.get("actual_value"), field_name="actual_value")
        passed = item.get("comparison_passed")
        if not isinstance(passed, bool):
            raise ValueError("recommendation comparison outcome is invalid")
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_text(item.get("rule_id")),
                    _markdown_text(item.get("result_id")),
                    _markdown_text(item.get("value_path")),
                    _markdown_text(item.get("comparator")),
                    json.dumps(threshold, ensure_ascii=True, allow_nan=False),
                    json.dumps(actual, ensure_ascii=True, allow_nan=False),
                    "是" if passed else "否",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("recommendation report text evidence is invalid")
    normalized = value.strip().replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    if not normalized or any(ord(character) < 32 for character in normalized):
        raise ValueError("recommendation report text evidence is unsafe")
    return normalized


def _text_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"recommendation {field_name} is invalid")
    return tuple(_markdown_text(item) for item in value)


def _finite_number(value: object, *, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"recommendation {field_name} is not numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"recommendation {field_name} is not finite")
    return value


__all__ = ["render_engineering_recommendation_markdown"]
