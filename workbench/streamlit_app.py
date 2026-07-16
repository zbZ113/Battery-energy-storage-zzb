"""Streamlit entry point for the HTTP-only research workbench."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any, cast

from workbench.client import ApiClient, AuthPrincipal, WorkbenchConfig, WorkbenchError
from workbench.presenters import (
    EvidenceSection,
    build_audit_section,
    build_decision_section,
    build_lifetime_section,
    build_quality_section,
)

NAVIGATION_PAGES = ("服务与数据", "工作流执行", "结果展示", "审计报告")
_WORKFLOW_SESSION_KEY = "signed_lifetime_workflow"
_CLIENT_SESSION_KEY = "workbench_api_client"
_PRINCIPAL_SESSION_KEY = "workbench_auth_principal"


class StreamlitDependencyUnavailable(RuntimeError):
    """Raised when the optional research-workbench dependency is absent."""


def load_streamlit() -> Any:
    """Load Streamlit lazily so importing the HTTP client remains dependency-free."""
    try:
        return importlib.import_module("streamlit")
    except ModuleNotFoundError as exc:
        message = (
            "Streamlit workbench support requires installing the "
            "'quanxin-life[streamlit]' optional dependency"
        )
        raise StreamlitDependencyUnavailable(message) from exc


def section_rows(section: EvidenceSection) -> list[dict[str, object]]:
    """Convert presentation fields to rows without transforming backend values."""
    return [
        {
            "字段": field.label,
            "值": field.display_value(),
            "JSON 路径": field.json_path,
        }
        for field in section.fields
    ]


def get_session_client(
    session_state: MutableMapping[str, object],
    config: WorkbenchConfig,
    client_factory: Callable[[WorkbenchConfig], ApiClient] = ApiClient.from_config,
) -> ApiClient:
    """Return one cookie-owning client for exactly one Streamlit browser session."""
    existing = session_state.get(_CLIENT_SESSION_KEY)
    if existing is not None:
        return cast(ApiClient, existing)
    client = client_factory(config)
    session_state[_CLIENT_SESSION_KEY] = client
    return client


def render_workbench(
    st: Any,
    *,
    config: WorkbenchConfig | None = None,
    client_factory: Callable[[WorkbenchConfig], ApiClient] = ApiClient.from_config,
) -> None:
    """Render server-issued identifiers and evidence without numeric logic."""
    st.set_page_config(page_title="泉芯智寿科研工作台", layout="wide")
    st.title("泉芯智寿科研工作台")
    st.caption("薄客户端: 所有工程数值、决策与审计报告均由 FastAPI 服务签发。")

    try:
        runtime_config = config or WorkbenchConfig.from_environment()
    except ValueError as exc:
        st.error(f"工作台配置无效: {exc}")
        return
    client = get_session_client(st.session_state, runtime_config, client_factory)
    principal = st.session_state.get(_PRINCIPAL_SESSION_KEY)
    if not isinstance(principal, AuthPrincipal):
        _render_login(st, client)
        return
    if principal.must_change_password:
        _render_password_change(st, client)
        return

    if st.sidebar.button("退出登录"):
        _show_action(st, client.logout)
        _clear_auth_session(st.session_state)
        st.rerun()
        return
    page = st.sidebar.radio("导航", NAVIGATION_PAGES)

    if page == "服务与数据":
        _render_service_and_data(st, client)
    elif page == "工作流执行":
        _render_workflow(st, client)
    elif page == "结果展示":
        _render_signed_results(st, client)
    else:
        _render_audit(st, client)


def _render_login(st: Any, client: ApiClient) -> None:
    st.header("登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if not st.button("登录", type="primary"):
        return

    def submit() -> None:
        session = client.login(username=username, password=password)
        st.session_state[_PRINCIPAL_SESSION_KEY] = AuthPrincipal(
            user_id=session.user_id,
            username=session.username,
            role=session.role,
            must_change_password=session.must_change_password,
        )
        st.rerun()

    _show_action(st, submit)


def _render_password_change(st: Any, client: ApiClient) -> None:
    st.header("首次登录: 请修改临时密码")
    current_password = st.text_input("当前密码", type="password")
    new_password = st.text_input("新密码", type="password")
    confirmation = st.text_input("确认新密码", type="password")
    if not st.button("修改密码", type="primary"):
        return
    if new_password != confirmation:
        st.error("两次输入的新密码不一致")
        return

    def submit() -> None:
        session = client.change_password(
            current_password=current_password,
            new_password=new_password,
        )
        st.session_state[_PRINCIPAL_SESSION_KEY] = AuthPrincipal(
            user_id=session.user_id,
            username=session.username,
            role=session.role,
            must_change_password=session.must_change_password,
        )
        st.rerun()

    _show_action(st, submit)


def _clear_auth_session(session_state: MutableMapping[str, object]) -> None:
    session_state.pop(_PRINCIPAL_SESSION_KEY, None)
    session_state.pop(_WORKFLOW_SESSION_KEY, None)
    client = session_state.pop(_CLIENT_SESSION_KEY, None)
    close = getattr(client, "close", None)
    if callable(close):
        close()


def _render_service_and_data(st: Any, client: ApiClient) -> None:
    st.header("服务连接与可信数据")
    st.write("仅连接 FastAPI, 不在工作台进程加载模型或数据处理代码。")

    if st.button("检查服务健康状态"):
        _show_action(st, lambda: st.json(client.health().display_payload()))
    if st.button("发现可用工具"):
        _show_action(st, lambda: st.json(list(client.list_tools())))

    st.subheader("注册可信 Canonical CSV 批次")
    csv_file = st.file_uploader("Canonical CSV", type=("csv",))
    registration_text = st.text_area(
        "CanonicalCsvBatchRegistration JSON",
        placeholder="粘贴经过审核的 metadata、feature_config、版本与 provenance",
    )
    if st.button("注册 CSV 批次"):

        def register_csv() -> None:
            if csv_file is None:
                raise ValueError("请选择 Canonical CSV 文件")
            try:
                registration = json.loads(registration_text)
            except json.JSONDecodeError as exc:
                raise ValueError("注册信息必须是有效 JSON") from exc
            if not isinstance(registration, dict):
                raise ValueError("注册信息 JSON 顶层必须是对象")
            record_batch_id = client.register_canonical_csv(
                csv_file.getvalue(),
                registration,
            )
            st.success("可信批次已注册")
            st.code(record_batch_id)

        _show_action(st, register_csv)


def _render_workflow(st: Any, client: ApiClient) -> None:
    st.header("寿命决策工作流")
    st.write("这里只提交服务端签发或审核的 ID, 不接受手工工程数值。")
    record_batch_id = st.text_input("可信批次 ID", placeholder="record_batch_id")
    calibration_cohort_id = st.text_input(
        "校准队列 ID", placeholder="calibration_cohort_id"
    )
    policy_id = st.text_input("决策策略 ID", placeholder="policy_id")

    if st.button("提交寿命决策工作流", type="primary"):

        def run_and_store() -> None:
            result = client.run_lifetime_workflow(
                record_batch_id=record_batch_id,
                calibration_cohort_id=calibration_cohort_id,
                policy_id=policy_id,
            )
            payload = result.display_payload()
            st.session_state[_WORKFLOW_SESSION_KEY] = payload
            st.json(payload)
            if result.report_result_id is None:
                st.info("当前终态没有审计报告; 请检查质量阻断状态和服务端警告。")

        _show_action(st, run_and_store)
    else:
        stored = _stored_workflow(st)
        if stored is not None:
            st.subheader("最近一次服务端工作流结果")
            st.json(dict(stored))


def _render_signed_results(st: Any, client: ApiClient) -> None:
    st.header("已签发结果")
    workflow = _stored_workflow(st)
    if workflow is None:
        st.info("尚无工作流结果, 请先在“工作流执行”区提交服务端 ID。")
        return
    st.subheader("工作流结果 ID")
    st.json(dict(workflow))

    quality = _fetch_optional_result(client, workflow.get("quality_result_id"))
    prediction = _fetch_optional_result(client, workflow.get("prediction_result_id"))
    interval = _fetch_optional_result(client, workflow.get("interval_result_id"))
    decision = _fetch_optional_result(client, workflow.get("decision_result_id"))

    _render_evidence_section(st, build_quality_section(quality))
    _render_evidence_section(st, build_lifetime_section(prediction, interval))
    _render_evidence_section(st, build_decision_section(decision))


def _render_audit(st: Any, client: ApiClient) -> None:
    st.header("审计证据与报告")
    workflow = _stored_workflow(st)
    if workflow is None:
        st.info("尚无工作流结果, 无法定位已签发 ToolResult。")
        return

    result_ids = [
        value
        for key, value in workflow.items()
        if key.endswith("_result_id") and isinstance(value, str) and value
    ]
    if result_ids:
        selected_id = st.selectbox("选择 ToolResult", result_ids)

        def show_evidence() -> None:
            payload = client.get_tool_result(selected_id)
            _render_evidence_section(st, build_audit_section(payload))

        _show_action(st, show_evidence)
    else:
        st.info("工作流响应未提供可读取的 ToolResult ID。")

    report_id = workflow.get("report_result_id")
    if not isinstance(report_id, str) or not report_id:
        st.info("后端未提供审计报告 ID。")
        return

    def show_report() -> None:
        report = client.get_audited_markdown(report_id)
        st.subheader("审计 Markdown")
        st.markdown(report.markdown)
        st.download_button(
            "下载审计 Markdown",
            data=report.download_bytes(),
            file_name=f"audited-report-{report.result_id}.md",
            mime="text/markdown; charset=utf-8",
        )

    _show_action(st, show_report)


def _render_evidence_section(st: Any, section: EvidenceSection) -> None:
    st.subheader(section.title)
    st.dataframe(section_rows(section), use_container_width=True, hide_index=True)


def _fetch_optional_result(client: ApiClient, result_id: object) -> Mapping[str, object]:
    if not isinstance(result_id, str) or not result_id:
        return {}
    return client.get_tool_result(result_id)


def _stored_workflow(st: Any) -> Mapping[str, object] | None:
    value = st.session_state.get(_WORKFLOW_SESSION_KEY)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _show_action(st: Any, action: Callable[[], None]) -> None:
    try:
        action()
    except (WorkbenchError, ValueError) as exc:
        st.error(str(exc))


def main() -> None:
    render_workbench(load_streamlit())


if __name__ == "__main__":
    main()
