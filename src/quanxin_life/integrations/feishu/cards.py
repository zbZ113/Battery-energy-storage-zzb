"""Deterministic Feishu cards that only carry traceable object references."""

from __future__ import annotations

import re
from typing import Any

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


def build_run_reference_card(
    *,
    run_id: str,
    result_id: str | None = None,
) -> dict[str, Any]:
    """Build a status card without copying numerical results into Feishu."""

    checked_run_id = _identifier(run_id, field_name="run_id")
    fields: list[dict[str, Any]] = [
        {
            "is_short": False,
            "text": {"tag": "lark_md", "content": f"**run_id**\n{checked_run_id}"},
        }
    ]
    if result_id is not None:
        checked_result_id = _identifier(result_id, field_name="result_id")
        fields.append(
            {
                "is_short": False,
                "text": {
                    "tag": "lark_md",
                    "content": f"**result_id**\n{checked_result_id}",
                },
            }
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "blue",
            "title": {"tag": "plain_text", "content": "泉芯智寿任务状态"},
        },
        "elements": [{"tag": "div", "fields": fields}],
    }


def _identifier(value: str, *, field_name: str) -> str:
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must not be blank")
    if _SAFE_REFERENCE.fullmatch(checked) is None:
        raise ValueError(f"{field_name} must be a safe machine identifier")
    return checked
