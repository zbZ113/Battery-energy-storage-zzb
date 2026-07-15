from __future__ import annotations

import importlib

import pytest

from workbench.presenters import build_lifetime_section
from workbench.streamlit_app import (
    NAVIGATION_PAGES,
    StreamlitDependencyUnavailable,
    load_streamlit,
    section_rows,
)


def test_missing_streamlit_reports_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_: str) -> object:
        raise ModuleNotFoundError("No module named 'streamlit'")

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(
        StreamlitDependencyUnavailable,
        match=r"quanxin-life\[streamlit\]",
    ):
        load_streamlit()


def test_workbench_exposes_clear_navigation_sections() -> None:
    assert NAVIGATION_PAGES == (
        "服务与数据",
        "工作流执行",
        "结果展示",
        "审计报告",
    )


def test_section_rows_preserve_backend_values_and_mark_missing_values() -> None:
    section = build_lifetime_section(
        {
            "values": {
                "artifact": {
                    "life_prediction": {"predicted_eol_cycle": 812.5},
                    "derived_rul_cycle": 712.5,
                }
            }
        },
        {"values": {"artifact": {"prediction_interval": {}}}},
    )

    rows = section_rows(section)

    assert rows[0]["值"] == 812.5
    assert rows[1]["值"] == 712.5
    assert rows[2]["值"] == "后端未提供"
    assert rows[0]["JSON 路径"] == (
        "values.artifact.life_prediction.predicted_eol_cycle"
    )
