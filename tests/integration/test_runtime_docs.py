from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_physics_extra_pins_the_verified_pybamm_release() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["optional-dependencies"]["physics"] == [
        "pybamm==25.12.2"
    ]


def test_mcp_extra_pins_the_verified_sdk_release() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["optional-dependencies"]["mcp"] == ["mcp==1.28.1"]


def test_readme_is_user_facing_and_only_documents_real_entry_points() -> None:
    readme = _read("README.md")

    for section in (
        "## 当前状态",
        "## 已验证结果",
        "## 核心技术路线",
        "## 成熟度边界",
        "## 快速开始",
        "## 质量门禁",
        "## 文档导航",
    ):
        assert section in readme

    for marker in (
        "Implemented",
        "Validated",
        "Planned",
        "受约束专业智能体工作流",
        "CyclePatch Direct",
        "CyclePatch-BatLiNet",
        "HybridPatch-v2",
        "matr-three-batch",
        "scripts/a100/train_dataset.sh",
        "frontend/package.json",
        "pnpm@10.28.1",
        "POST /v1/integrations/feishu/events",
        "src/quanxin_life/integrations/feishu",
        "ToolResult",
    ):
        assert marker in readme

    assert "deploy/foundation_api.py" in readme
    assert "workbench/streamlit_app.py" in readme
    assert "src/quanxin_life/tools/mcp_host.py" in readme
    assert "docs/runtime-setup.md" in readme
    assert ".[mcp]" in readme
    assert "FileSystemVerifiedEarlyCycleBatchStore" in readme
    assert "JsonlAuditLedger" in readme
    assert "A100 Smoke验收与正式五种子训练" not in readme
    assert "不等于生产部署" in readme


def test_public_evidence_docs_have_single_responsibilities_and_are_linked() -> None:
    readme = _read("README.md")
    expected = {
        "docs/status.md": ("# 项目状态", "Implemented", "Validated", "Planned"),
        "docs/benchmark.md": ("# Advanced Benchmark", "MAE", "PICP", "五种子"),
        "docs/model-card-advanced.md": (
            "# Advanced 模型卡",
            "CyclePatch Direct",
            "HybridPatch-v2",
            "拒绝",
        ),
        "docs/limitations.md": (
            "# 已知限制",
            "右删失",
            "HUST",
            "不等于生产部署",
        ),
        "docs/reproducibility.md": (
            "# 可复现性",
            "cell_id",
            "SHA-256",
            "Python 3.11",
        ),
        "ARCHITECTURE.md": (
            "# 架构说明",
            "ToolResult",
            "AgentStep",
            "fenced claim",
        ),
    }

    for relative_path, markers in expected.items():
        assert relative_path in readme
        document = _read(relative_path)
        for marker in markers:
            assert marker in document


def test_runtime_guide_covers_windows_linux_and_operator_owned_inputs() -> None:
    guide = _read("docs/runtime-setup.md")

    for required in (
        "## Windows 10 本机运行",
        "## Linux 服务器运行",
        "## Docker Compose 基础 API",
        "## 完整竞赛应用装配",
        "## 外部需提供项",
        "LLM Provider",
        "企业数据",
        "决策策略",
        "飞书",
        "MCP",
    ):
        assert required in guide

    assert "uvicorn deploy.foundation_api:app" in guide
    assert "streamlit run workbench/streamlit_app.py" in guide
    assert "create_competition_fastapi_app" in guide
    assert "Windows 10 足够" in guide
    assert "FileSystemVerifiedEarlyCycleBatchStore" in guide
    assert "JsonlAuditLedger" in guide
    assert "MATR" in guide and "接口" in guide
    assert "HUST" in guide and "pickle" in guide


def test_runtime_guide_documents_both_real_mcp_transports() -> None:
    guide = _read("docs/runtime-setup.md")

    assert "python scripts/run_mcp_host.py --transport stdio" in guide
    assert (
        "python scripts/run_mcp_host.py --transport streamable-http "
        "--host 127.0.0.1 --port 8001"
    ) in guide
    assert "http://127.0.0.1:8001/mcp" in guide


def test_runtime_guide_documents_reviewed_hybrid_knowledge_retrieval() -> None:
    guide = _read("docs/runtime-setup.md")

    assert "POST /v1/knowledge/documents/{document_id}/embeddings" in guide
    assert "DatabaseHybridEvidenceBackend" in guide
    assert "HYBRID_RETRIEVAL_UNAVAILABLE_INCOMPLETE_EMBEDDINGS" in guide
    assert "pgvector + BM25 + reranker" in guide


def test_runtime_guide_documents_industrial_protocol_sandboxes() -> None:
    guide = _read("docs/runtime-setup.md")

    for required in (
        "/v1/integrations/industrial/sandbox/bms/rest",
        "/v1/integrations/industrial/sandbox/bms/mqtt",
        "/v1/integrations/industrial/sandbox/bms/modbus",
        "/v1/integrations/industrial/sandbox/ems/decisions/{result_id}",
        "quanxin/v1/bms/{measurement_batch_id}",
        "quanxin-modbus-bms-v1",
        "IndustrialBmsSandbox",
        "EmsDecisionSandboxPublisher",
    ):
        assert required in guide

    assert "协议沙箱" in guide
    assert "不代表生产 BMS/EMS 已接入" in guide


def test_runtime_guide_documents_ledger_bound_report_artifacts() -> None:
    guide = _read("docs/runtime-setup.md")

    for required in (
        'python -m pip install -e ".[reporting]"',
        "/v1/reports/{result_id}/artifacts/{artifact_format}",
        "AuditedReportArtifactExporter",
        "ReviewedPdfFont",
        "JSON",
        "Markdown",
        "PDF",
        "DOCX",
        "SHA-256",
    ):
        assert required in guide

    assert "只能消费审计账本中已登记的" in guide


def test_runtime_guide_documents_a100_suite_result_import() -> None:
    guide = _read("docs/runtime-setup.md")

    for required in (
        "scripts/import_a100_suite_run.py",
        "configs/training/matr_three_batch_smoke.json",
        "configs/training/matr_three_batch_final.json",
        "--expected-source-commit",
        "--transfer-archive",
        "--transfer-sha256",
        "A100SuiteRunImporter",
        "completed-run-v2",
        "aggregate_metrics.json",
        "metrics_test.csv",
    ):
        assert required in guide

    assert "不加载模型权重" in guide
    assert "重复导入" in guide


def test_env_template_contains_no_secret_values() -> None:
    template = _read(".env.example")

    for name in (
        "QUANXIN_DATA_ROOT",
        "QUANXIN_ARTIFACT_ROOT",
        "QUANXIN_POLICY_ROOT",
        "QUANXIN_LLM_PROVIDER",
        "QUANXIN_LLM_API_KEY",
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
    ):
        assert re.search(rf"(?m)^{name}=\s*$", template)

    assert "sk-" not in template
    sensitive_lines = re.findall(
        r"(?mi)^[A-Z0-9_]*(?:SECRET|TOKEN|API_KEY)[A-Z0-9_]*=(.*)$",
        template,
    )
    assert sensitive_lines
    assert all(not value.strip() for value in sensitive_lines)


def test_compose_starts_only_the_real_foundation_api_entrypoint() -> None:
    compose = _read("deploy/compose.yaml")
    dockerfile = _read("deploy/Dockerfile")
    entrypoint = _read("deploy/foundation_api.py")

    assert "foundation-api:" in compose
    assert "deploy.foundation_api:app" in compose
    assert "8000:8000" in compose
    assert "/health" in compose
    assert "streamlit" not in compose.lower()
    assert "mcp" not in compose.lower()
    assert "COPY pyproject.toml README.md" in dockerfile
    assert 'pip install --no-cache-dir -e ".[api]"' in dockerfile
    assert "create_available_tool_invocation_service" in entrypoint
    assert "create_fastapi_app" in entrypoint
    assert "app =" in entrypoint


def test_local_pre_commit_gate_matches_the_project_quality_commands() -> None:
    config = _read(".pre-commit-config.yaml")
    pyproject = _read("pyproject.toml")

    assert "repo: local" in config
    assert "python -m ruff check ." in config
    assert "python -m mypy" in config
    assert "python -m pytest -q" in config
    assert "python -m compileall -q src workbench deploy" in config
    assert '"pre-commit>=' in pyproject
