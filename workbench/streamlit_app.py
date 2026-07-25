"""Streamlit entry point for the HTTP-only research workbench."""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable, Mapping, MutableMapping
from hashlib import sha256
from typing import Any, cast
from uuid import uuid4

from workbench.client import ApiClient, AuthPrincipal, WorkbenchConfig, WorkbenchError
from workbench.presenters import (
    EvidenceSection,
    build_audit_section,
)

NAVIGATION_PAGES = (
    "服务与数据",
    "Agent 协同",
    "工作流执行",
    "结果展示",
    "审计报告",
)
_WORKFLOW_SESSION_KEY = "signed_lifetime_workflow"
_CLIENT_SESSION_KEY = "workbench_api_client"
_PRINCIPAL_SESSION_KEY = "workbench_auth_principal"
_AGENT_RUN_ID_SESSION_KEY = "workbench_agent_run_id"
_AGENT_RUN_SESSION_KEY = "workbench_agent_run"
_AGENT_IDEMPOTENCY_SESSION_KEY = "workbench_agent_idempotency_key"
_AGENT_RESULT_CACHE_SESSION_KEY = "workbench_agent_result_cache"


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
    elif page == "Agent 协同":
        _render_agent_collaboration(st, client)
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
    session_state.pop(_AGENT_RUN_ID_SESSION_KEY, None)
    session_state.pop(_AGENT_RUN_SESSION_KEY, None)
    session_state.pop(_AGENT_IDEMPOTENCY_SESSION_KEY, None)
    session_state.pop(_AGENT_RESULT_CACHE_SESSION_KEY, None)
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
    dataset_id = st.text_input("目标数据集 ID", placeholder="dataset_id")
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
                dataset_id=dataset_id,
                payload=csv_file.getvalue(),
                registration=registration,
            )
            st.success("可信批次已注册")
            st.code(record_batch_id)

        _show_action(st, register_csv)


def _render_agent_collaboration(st: Any, client: ApiClient) -> None:
    st.header("Agent 协同")
    st.write(
        "科研台只提交自然语言目标和服务端 ID; 规划、工具调用与工程数值均由后端签发。"
    )
    st.info(
        "科研台采用手动刷新, 正式实时事件时间线请使用 Next.js 门户。"
    )

    project_id = st.text_input("项目 ID", placeholder="project_id")
    user_goal = st.text_area(
        "自然语言目标",
        placeholder="例如: 分析该电芯的早期寿命风险并生成可审计报告",
    )
    dataset_text = st.text_area(
        "数据集 IDs (逗号或换行分隔)",
        placeholder="dataset_id, 可填写多个",
    )
    outputs_text = st.text_area(
        "需要的输出 (逗号或换行分隔)",
        value="cycle_life",
    )
    if st.button("创建 Agent 任务", type="primary"):

        def create_run() -> None:
            normalized_project_id = project_id.strip()
            normalized_goal = user_goal.strip()
            dataset_ids = _split_identifiers(dataset_text)
            requested_outputs = _split_identifiers(
                outputs_text,
                require_nonempty=True,
            )
            request_payload: dict[str, object] = {
                "project_id": normalized_project_id,
                "user_goal": normalized_goal,
                "dataset_ids": list(dataset_ids),
                "requested_outputs": list(requested_outputs),
            }
            key = _idempotency_key_for_request(st.session_state, request_payload)
            run = client.create_agent_run(
                project_id=normalized_project_id,
                user_goal=normalized_goal,
                dataset_ids=dataset_ids,
                requested_outputs=requested_outputs,
                idempotency_key=key,
            )
            st.session_state[_AGENT_RUN_ID_SESSION_KEY] = run.run_id
            st.session_state[_AGENT_RUN_SESSION_KEY] = run.display_payload()
            st.session_state.pop(_AGENT_RESULT_CACHE_SESSION_KEY, None)
            st.session_state.pop(_AGENT_IDEMPOTENCY_SESSION_KEY, None)
            st.success("Agent 任务已创建")
            st.json(run.display_payload())

        _show_action(st, create_run)

    run_id = st.session_state.get(_AGENT_RUN_ID_SESSION_KEY)
    if not isinstance(run_id, str) or not run_id:
        st.info("当前浏览器会话尚未创建 Agent 任务。")
        return

    st.subheader("当前任务")
    st.write(f"run_id: {run_id}")
    if st.button("刷新任务状态"):

        def refresh_run() -> None:
            run = client.get_agent_run(run_id)
            st.session_state[_AGENT_RUN_SESSION_KEY] = run.display_payload()
            st.json(run.display_payload())

        _show_action(st, refresh_run)

    stored_run = _stored_agent_run(st)
    if stored_run is not None:
        st.json(dict(stored_run))

    st.subheader("人工审批")
    st.write("审批 ID 来自 Next.js 实时任务事件; 科研台不会猜测审批对象。")
    approval_id = st.text_input("审批 ID (来自任务事件)")
    approval_reason = st.text_input("审批说明 (可选)")
    if st.button("批准"):

        def approve() -> None:
            run = client.approve_agent_run(
                run_id,
                approval_id=approval_id,
                reason=approval_reason or None,
            )
            st.session_state[_AGENT_RUN_SESSION_KEY] = run.display_payload()
            st.success("审批已批准")
            st.json(run.display_payload())

        _show_action(st, approve)
    if st.button("拒绝"):

        def reject() -> None:
            run = client.reject_agent_run(
                run_id,
                approval_id=approval_id,
                reason=approval_reason or None,
            )
            st.session_state[_AGENT_RUN_SESSION_KEY] = run.display_payload()
            st.success("审批已拒绝")
            st.json(run.display_payload())

        _show_action(st, reject)
    if st.button("取消任务"):

        def cancel() -> None:
            run = client.cancel_agent_run(run_id)
            st.session_state[_AGENT_RUN_SESSION_KEY] = run.display_payload()
            st.success("取消请求已提交")
            st.json(run.display_payload())

        _show_action(st, cancel)

    st.subheader("运行范围内结果")
    st.write(
        "仅能读取属于当前 run_id 且步骤已完成的 ToolResult; Result ID 来自任务事件。"
    )
    result_id = st.text_input("已完成 Result ID (来自任务事件)")
    if st.button("读取运行结果"):

        def read_scoped_result() -> None:
            result = client.get_agent_run_result(run_id, result_id)
            cache = _agent_result_cache(st.session_state, run_id)
            cache[result_id.strip()] = dict(result)
            st.session_state[_AGENT_RESULT_CACHE_SESSION_KEY] = {
                "run_id": run_id,
                "results": cache,
            }
            st.json(result)

        _show_action(st, read_scoped_result)


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
    del client  # Results are displayed only after a scoped read on the Agent page.
    run_id = _current_agent_run_id(st.session_state)
    cache = _agent_result_cache(st.session_state, run_id) if run_id else {}
    if not cache:
        st.info("请先在“Agent 协同”区按 run_id + result_id 读取已完成结果。")
        return
    selected_id = st.selectbox("选择已验证 ToolResult", list(cache))
    st.json(dict(cache[selected_id]))


def _render_audit(st: Any, client: ApiClient) -> None:
    st.header("审计证据与报告")
    del client  # Audit rendering consumes only the scoped per-session cache.
    run_id = _current_agent_run_id(st.session_state)
    cache = _agent_result_cache(st.session_state, run_id) if run_id else {}
    if not cache:
        st.info("请先在“Agent 协同”区读取运行范围内的 ToolResult。")
        return
    selected_id = st.selectbox("选择 ToolResult", list(cache))
    _render_evidence_section(st, build_audit_section(cache[selected_id]))

    st.info(
        "科研台不调用全局管理员报告接口; run 范围报告导出后端尚未提供。"
    )


def _render_evidence_section(st: Any, section: EvidenceSection) -> None:
    st.subheader(section.title)
    st.dataframe(section_rows(section), use_container_width=True, hide_index=True)


def _stored_workflow(st: Any) -> Mapping[str, object] | None:
    value = st.session_state.get(_WORKFLOW_SESSION_KEY)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _stored_agent_run(st: Any) -> Mapping[str, object] | None:
    value = st.session_state.get(_AGENT_RUN_SESSION_KEY)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        return None
    return value


def _agent_result_cache(
    session_state: MutableMapping[str, object],
    run_id: str,
) -> dict[str, Mapping[str, object]]:
    value = session_state.get(_AGENT_RESULT_CACHE_SESSION_KEY)
    if not isinstance(value, dict) or value.get("run_id") != run_id:
        return {}
    results = value.get("results")
    if not isinstance(results, dict):
        return {}
    cache: dict[str, Mapping[str, object]] = {}
    for result_id, result in results.items():
        if (
            isinstance(result_id, str)
            and result_id
            and isinstance(result, dict)
            and result.get("result_id") == result_id
        ):
            cache[result_id] = cast(dict[str, object], result)
    return cache


def _current_agent_run_id(session_state: MutableMapping[str, object]) -> str | None:
    value = session_state.get(_AGENT_RUN_ID_SESSION_KEY)
    return value if isinstance(value, str) and value else None


def _idempotency_key_for_request(
    session_state: MutableMapping[str, object],
    request_payload: Mapping[str, object],
) -> str:
    canonical = json.dumps(
        dict(request_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = sha256(canonical.encode("utf-8")).hexdigest()
    existing = session_state.get(_AGENT_IDEMPOTENCY_SESSION_KEY)
    if isinstance(existing, dict):
        stored_fingerprint = existing.get("fingerprint")
        stored_key = existing.get("key")
        if stored_fingerprint == fingerprint and isinstance(stored_key, str) and stored_key:
            return stored_key
    key = f"streamlit-{uuid4()}"
    session_state[_AGENT_IDEMPOTENCY_SESSION_KEY] = {
        "fingerprint": fingerprint,
        "key": key,
    }
    return key


def _split_identifiers(
    value: str,
    *,
    require_nonempty: bool = False,
) -> tuple[str, ...]:
    identifiers = tuple(item.strip() for item in re.split(r"[,\r\n]+", value) if item.strip())
    if require_nonempty and not identifiers:
        raise ValueError("需要的输出不能为空")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("ID 或输出名称不能重复")
    return identifiers


def _show_action(st: Any, action: Callable[[], None]) -> None:
    try:
        action()
    except (WorkbenchError, ValueError) as exc:
        st.error(str(exc))


def main() -> None:
    render_workbench(load_streamlit())


if __name__ == "__main__":
    main()
