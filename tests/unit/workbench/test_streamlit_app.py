from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest

from workbench.client import AuthPrincipal, AuthSession, WorkbenchConfig
from workbench.presenters import build_lifetime_section
from workbench.streamlit_app import (
    NAVIGATION_PAGES,
    StreamlitDependencyUnavailable,
    get_session_client,
    load_streamlit,
    render_workbench,
    section_rows,
)

CONFIG = WorkbenchConfig.from_mapping(
    {
        "QUANXIN_ENVIRONMENT": "development",
        "QUANXIN_API_BASE_URL": "http://localhost:8000",
        "QUANXIN_TRUSTED_ORIGIN": "http://localhost:8501",
    }
)


class RerunRequested(RuntimeError):
    pass


@dataclass
class FakeClient:
    principal: AuthPrincipal | None = None
    logout_calls: int = 0

    def login(self, *, username: str, password: str) -> AuthSession:
        assert username == "member@example.test"
        assert password == "temporary-secret"
        return AuthSession(
            user_id="user-1",
            username=username,
            role="MEMBER",
            must_change_password=True,
            expires_at="2026-07-17T00:00:00Z",
        )

    def change_password(self, *, current_password: str, new_password: str) -> AuthSession:
        assert current_password == "temporary-secret"
        assert new_password == "replacement-secret"
        return AuthSession(
            user_id="user-1",
            username="member@example.test",
            role="MEMBER",
            must_change_password=False,
            expires_at="2026-07-17T01:00:00Z",
        )

    def logout(self) -> None:
        self.logout_calls += 1

    def close(self) -> None:
        return None


class FakeStreamlit:
    def __init__(self, *, inputs: dict[str, str] | None = None, buttons: set[str] | None = None):
        self.session_state: dict[str, object] = {}
        self.inputs = inputs or {}
        self.buttons = buttons or set()
        self.text_input_labels: list[str] = []
        self.headers: list[str] = []
        self.sidebar = self

    def set_page_config(self, **_: object) -> None:
        return None

    def title(self, _: str) -> None:
        return None

    def caption(self, _: str) -> None:
        return None

    def header(self, label: str) -> None:
        self.headers.append(label)

    def subheader(self, _: str) -> None:
        return None

    def write(self, _: object) -> None:
        return None

    def text_input(self, label: str, **_: object) -> str:
        self.text_input_labels.append(label)
        return self.inputs.get(label, "")

    def button(self, label: str, **_: object) -> bool:
        return label in self.buttons

    def success(self, _: str) -> None:
        return None

    def error(self, _: str) -> None:
        return None

    def rerun(self) -> None:
        raise RerunRequested

    def radio(self, _: str, values: tuple[str, ...]) -> str:
        return values[0]


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


def test_each_streamlit_session_owns_one_distinct_cookie_client() -> None:
    state_a: dict[str, object] = {}
    state_b: dict[str, object] = {}
    created: list[object] = []

    def factory(_: WorkbenchConfig) -> Any:
        client = object()
        created.append(client)
        return client

    first = get_session_client(state_a, CONFIG, factory)
    repeated = get_session_client(state_a, CONFIG, factory)
    second_session = get_session_client(state_b, CONFIG, factory)

    assert first is repeated
    assert second_session is not first
    assert created == [first, second_session]


def test_logged_out_page_has_no_editable_api_url_and_logs_in() -> None:
    st = FakeStreamlit(
        inputs={"用户名": "member@example.test", "密码": "temporary-secret"},
        buttons={"登录"},
    )
    client = FakeClient()

    with pytest.raises(RerunRequested):
        render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert st.text_input_labels == ["用户名", "密码"]
    assert all("API" not in label and "地址" not in label for label in st.text_input_labels)
    principal = st.session_state["workbench_auth_principal"]
    assert isinstance(principal, AuthPrincipal)
    assert principal.must_change_password is True


def test_password_change_blocks_workbench_until_completed() -> None:
    st = FakeStreamlit(
        inputs={
            "当前密码": "temporary-secret",
            "新密码": "replacement-secret",
            "确认新密码": "replacement-secret",
        },
        buttons={"修改密码"},
    )
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=True,
    )

    with pytest.raises(RerunRequested):
        render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert st.headers == ["首次登录: 请修改临时密码"]
    principal = st.session_state["workbench_auth_principal"]
    assert isinstance(principal, AuthPrincipal)
    assert principal.must_change_password is False


def test_logout_clears_streamlit_auth_and_client_state() -> None:
    st = FakeStreamlit(buttons={"退出登录"})
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )
    st.session_state["workbench_api_client"] = client

    with pytest.raises(RerunRequested):
        render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert client.logout_calls == 1
    assert "workbench_auth_principal" not in st.session_state
    assert "workbench_api_client" not in st.session_state
