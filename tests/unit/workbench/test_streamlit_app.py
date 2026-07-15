from __future__ import annotations

import importlib

import pytest

from workbench.streamlit_app import (
    StreamlitDependencyUnavailable,
    load_streamlit,
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

