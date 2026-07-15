"""Streamlit entry point for the HTTP-only research workbench."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from typing import Any

from workbench.client import ApiClient, WorkbenchError


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


def render_workbench(
    st: Any,
    *,
    client_factory: Callable[[str], ApiClient] = ApiClient,
) -> None:
    """Render identifiers and server-issued evidence without numeric logic."""
    st.set_page_config(page_title="泉芯智寿科研工作台", layout="wide")
    st.title("泉芯智寿科研工作台")
    st.caption("薄客户端: 所有工程数值、决策与审计报告均由 FastAPI 服务签发。")

    base_url = st.text_input("FastAPI 地址", value="http://127.0.0.1:8000")
    client = client_factory(base_url)

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

    st.subheader("按服务端 ID 执行寿命决策工作流")
    record_batch_id = st.text_input("可信批次 ID", placeholder="record_batch_id")
    calibration_cohort_id = st.text_input(
        "校准队列 ID", placeholder="calibration_cohort_id"
    )
    policy_id = st.text_input("决策策略 ID", placeholder="policy_id")

    if st.button("提交寿命决策工作流", type="primary"):

        def run_and_render() -> None:
            result = client.run_lifetime_workflow(
                record_batch_id=record_batch_id,
                calibration_cohort_id=calibration_cohort_id,
                policy_id=policy_id,
            )
            st.json(result.display_payload())
            if result.report_result_id is None:
                st.info("当前终态没有审计报告; 请检查质量阻断状态和服务端警告。")
                return
            report = client.get_audited_markdown(result.report_result_id)
            st.subheader("审计 Markdown")
            st.markdown(report.markdown)
            st.download_button(
                "下载审计 Markdown",
                data=report.download_bytes(),
                file_name=f"audited-report-{report.result_id}.md",
                mime="text/markdown; charset=utf-8",
            )

        _show_action(st, run_and_render)


def _show_action(st: Any, action: Callable[[], None]) -> None:
    try:
        action()
    except (WorkbenchError, ValueError) as exc:
        st.error(str(exc))


def main() -> None:
    render_workbench(load_streamlit())


if __name__ == "__main__":
    main()
