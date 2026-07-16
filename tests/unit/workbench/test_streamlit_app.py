from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest

from workbench.client import (
    AgentRunSnapshot,
    AuthPrincipal,
    AuthSession,
    WorkbenchConfig,
    WorkbenchError,
)
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
RUN_ID = "22222222-2222-4222-8222-222222222222"
RESULT_ID = "44444444-4444-4444-8444-444444444444"
APPROVAL_ID = "55555555-5555-4555-8555-555555555555"


class RerunRequested(RuntimeError):
    pass


@dataclass
class FakeClient:
    principal: AuthPrincipal | None = None
    logout_calls: int = 0
    created_runs: list[dict[str, object]] | None = None
    refreshed_runs: list[str] | None = None
    approval_actions: list[tuple[str, str, str, str | None]] | None = None
    cancelled_runs: list[str] | None = None
    scoped_result_reads: list[tuple[str, str]] | None = None
    create_failures_remaining: int = 0

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

    def create_agent_run(self, **kwargs: object) -> AgentRunSnapshot:
        if self.created_runs is None:
            self.created_runs = []
        self.created_runs.append(dict(kwargs))
        if self.create_failures_remaining > 0:
            self.create_failures_remaining -= 1
            raise WorkbenchError("temporary create failure")
        return _agent_run_snapshot()

    def get_agent_run(self, run_id: str) -> AgentRunSnapshot:
        assert run_id == RUN_ID
        if self.refreshed_runs is None:
            self.refreshed_runs = []
        self.refreshed_runs.append(run_id)
        return _agent_run_snapshot(status="AWAITING_APPROVAL")

    def approve_agent_run(
        self, run_id: str, *, approval_id: str, reason: str | None
    ) -> AgentRunSnapshot:
        assert run_id == RUN_ID
        assert approval_id == APPROVAL_ID
        self._record_approval("approve", run_id, approval_id, reason)
        return _agent_run_snapshot(status="RUNNING")

    def reject_agent_run(
        self, run_id: str, *, approval_id: str, reason: str | None
    ) -> AgentRunSnapshot:
        assert run_id == RUN_ID
        assert approval_id == APPROVAL_ID
        self._record_approval("reject", run_id, approval_id, reason)
        return _agent_run_snapshot(status="CANCELLED")

    def _record_approval(
        self, action: str, run_id: str, approval_id: str, reason: str | None
    ) -> None:
        if self.approval_actions is None:
            self.approval_actions = []
        self.approval_actions.append((action, run_id, approval_id, reason))

    def cancel_agent_run(self, run_id: str) -> AgentRunSnapshot:
        assert run_id == RUN_ID
        if self.cancelled_runs is None:
            self.cancelled_runs = []
        self.cancelled_runs.append(run_id)
        return _agent_run_snapshot(status="CANCELLED")

    def get_agent_run_result(self, run_id: str, result_id: str) -> dict[str, object]:
        assert run_id == RUN_ID
        assert result_id == RESULT_ID
        if self.scoped_result_reads is None:
            self.scoped_result_reads = []
        self.scoped_result_reads.append((run_id, result_id))
        return {
            "result_id": result_id,
            "tool_name": "predict_cycle_life",
            "values": {"server_signed": True},
        }

    def get_tool_result(self, result_id: str) -> dict[str, object]:
        raise AssertionError(f"global result endpoint must not be used: {result_id}")


def _agent_run_snapshot(*, status: str = "RUNNING") -> AgentRunSnapshot:
    return AgentRunSnapshot(
        run_id=RUN_ID,
        project_id="project-1",
        created_by_user_id="user-1",
        status=status,
        planning_mode="FIXED_FALLBACK",
        intent={"goal": "分析电芯寿命风险"},
        plan={"steps": [{"step_id": "validate", "tool_name": "validate_battery_data"}]},
        dispatch_status="DISPATCHED",
        dispatch_task_id="task-1",
        created_at="2026-07-16T00:00:00Z",
        updated_at="2026-07-16T00:00:01Z",
        completed_at=None,
    )


class FakeStreamlit:
    def __init__(
        self,
        *,
        inputs: dict[str, str] | None = None,
        text_areas: dict[str, str] | None = None,
        buttons: set[str] | None = None,
        selected_page: str | None = None,
    ):
        self.session_state: dict[str, object] = {}
        self.inputs = inputs or {}
        self.text_areas = text_areas or {}
        self.buttons = buttons or set()
        self.selected_page = selected_page
        self.text_input_labels: list[str] = []
        self.headers: list[str] = []
        self.json_payloads: list[object] = []
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

    def text_area(self, label: str, **_: object) -> str:
        return self.text_areas.get(label, "")

    def button(self, label: str, **_: object) -> bool:
        return label in self.buttons

    def success(self, _: str) -> None:
        return None

    def info(self, _: str) -> None:
        return None

    def json(self, value: object) -> None:
        self.json_payloads.append(value)

    def dataframe(self, *_: object, **__: object) -> None:
        return None

    def selectbox(self, _: str, values: list[str]) -> str:
        return values[0]

    def error(self, _: str) -> None:
        return None

    def rerun(self) -> None:
        raise RerunRequested

    def radio(self, _: str, values: tuple[str, ...]) -> str:
        return self.selected_page or values[0]


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
        "Agent 协同",
        "工作流执行",
        "结果展示",
        "审计报告",
    )


def test_agent_page_creates_run_and_keeps_run_id_in_browser_session() -> None:
    st = FakeStreamlit(
        inputs={"项目 ID": "project-1"},
        text_areas={
            "自然语言目标": "分析电芯寿命风险",
            "数据集 IDs (逗号或换行分隔)": "dataset-1, dataset-2",
            "需要的输出 (逗号或换行分隔)": "cycle_life\nreport",
        },
        buttons={"创建 Agent 任务"},
        selected_page="Agent 协同",
    )
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )

    render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert client.created_runs is not None
    assert client.created_runs[0]["project_id"] == "project-1"
    assert client.created_runs[0]["dataset_ids"] == ("dataset-1", "dataset-2")
    assert client.created_runs[0]["requested_outputs"] == ("cycle_life", "report")
    assert isinstance(client.created_runs[0]["idempotency_key"], str)
    assert st.session_state["workbench_agent_run_id"] == RUN_ID
    assert "workbench_agent_idempotency_key" not in st.session_state
    assert st.json_payloads[-1]["run_id"] == RUN_ID


def test_agent_page_refreshes_controls_and_reads_results_only_with_run_scope() -> None:
    st = FakeStreamlit(
        inputs={
            "审批 ID (来自任务事件)": APPROVAL_ID,
            "审批说明 (可选)": "证据已核对",
            "已完成 Result ID (来自任务事件)": RESULT_ID,
        },
        buttons={"刷新任务状态", "批准", "读取运行结果"},
        selected_page="Agent 协同",
    )
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )
    st.session_state["workbench_agent_run_id"] = RUN_ID

    render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert client.refreshed_runs == [RUN_ID]
    assert client.approval_actions == [
        ("approve", RUN_ID, APPROVAL_ID, "证据已核对")
    ]
    assert client.scoped_result_reads == [(RUN_ID, RESULT_ID)]
    assert st.session_state["workbench_agent_result_cache"] == {
        "run_id": RUN_ID,
        "results": {
            RESULT_ID: {
                "result_id": RESULT_ID,
                "tool_name": "predict_cycle_life",
                "values": {"server_signed": True},
            }
        },
    }
    assert st.json_payloads[-1]["result_id"] == RESULT_ID


def test_result_page_displays_only_results_previously_read_through_run_scope() -> None:
    st = FakeStreamlit(selected_page="结果展示")
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )
    st.session_state["workbench_agent_run_id"] = RUN_ID
    st.session_state["workbench_agent_result_cache"] = {
        "run_id": RUN_ID,
        "results": {
            RESULT_ID: {
                "result_id": RESULT_ID,
                "tool_name": "predict_cycle_life",
                "values": {"server_signed": True},
            }
        },
    }

    render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert client.scoped_result_reads is None
    assert st.json_payloads[-1]["result_id"] == RESULT_ID


def test_result_cache_is_not_displayed_after_agent_run_scope_changes() -> None:
    st = FakeStreamlit(selected_page="结果展示")
    client = FakeClient()
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )
    st.session_state["workbench_agent_run_id"] = (
        "33333333-3333-4333-8333-333333333333"
    )
    st.session_state["workbench_agent_result_cache"] = {
        "run_id": RUN_ID,
        "results": {
            RESULT_ID: {
                "result_id": RESULT_ID,
                "tool_name": "predict_cycle_life",
                "values": {"server_signed": True},
            }
        },
    }

    render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert st.json_payloads == []


def test_agent_create_idempotency_key_is_bound_to_normalized_request() -> None:
    st = FakeStreamlit(
        inputs={"项目 ID": " project-1 "},
        text_areas={
            "自然语言目标": " 分析电芯寿命风险 ",
            "数据集 IDs (逗号或换行分隔)": "dataset-1",
            "需要的输出 (逗号或换行分隔)": "cycle_life",
        },
        buttons={"创建 Agent 任务"},
        selected_page="Agent 协同",
    )
    client = FakeClient(create_failures_remaining=2)
    st.session_state["workbench_auth_principal"] = AuthPrincipal(
        user_id="user-1",
        username="member@example.test",
        role="MEMBER",
        must_change_password=False,
    )

    render_workbench(st, config=CONFIG, client_factory=lambda _: client)
    render_workbench(st, config=CONFIG, client_factory=lambda _: client)
    assert client.created_runs is not None
    first_key = client.created_runs[0]["idempotency_key"]
    assert client.created_runs[1]["idempotency_key"] == first_key

    st.inputs["项目 ID"] = "project-2"
    render_workbench(st, config=CONFIG, client_factory=lambda _: client)

    assert client.created_runs[2]["idempotency_key"] != first_key
    assert "workbench_agent_idempotency_key" not in st.session_state


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
