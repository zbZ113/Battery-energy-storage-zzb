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


def test_scenarios_extra_contains_every_runtime_and_plot_dependency() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))

    assert pyproject["project"]["optional-dependencies"]["scenarios"] == [
        "matplotlib>=3.9,<4",
        "numpy>=2.1,<3",
        "pandas>=2.2,<3",
        "scipy>=1.14,<2",
    ]


def test_feishu_aily_docs_cover_the_audited_blast_scenario_flow() -> None:
    openapi = _read("docs/integrations/aily-connector-openapi.yaml")
    prompt = _read("docs/integrations/aily-system-prompt.md")
    guide = _read("docs/integrations/feishu-aily-model-tools.md")
    readme = _read("README.md")

    for marker in (
        "/v1/aily/scenario-contexts",
        "CreateScenarioContextRequest",
        "OperationScenario",
        "ScenarioSegment",
        "scenario_context_id",
        "project_storage_lifetime",
    ):
        assert marker in openapi

    for marker in (
        "compare_operation_scenarios",
        "project_storage_lifetime",
        "scenario_context_id",
        "PHYSICS_REFERENCE",
        "15/20/25",
    ):
        assert marker in prompt

    for marker in (
        "REGISTERED_CANDIDATE",
        "PHYSICS_REFERENCE",
        "Lfp_Gr_250AhPrismatic",
        "scenario_context_id",
        "durable job",
        "Fake Feishu scenario E2E",
        "deploy.competition_api:app",
        "0022",
        "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION",
        "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS",
        "feishu_app_secret",
        "aily_connector_api_key",
        "断网",
    ):
        assert marker in guide

    assert ".[scenarios]" in readme
    assert "project_storage_lifetime" in readme
    assert "REGISTERED_CANDIDATE" in readme
    assert "deploy.competition_api:app" in readme
    assert "0022" in readme
    assert "Fake Aily scenario E2E" in readme


def test_feishu_csv_registration_docs_require_exact_observed_sha_binding() -> None:
    documents = {
        "README.md": _read("README.md"),
        "docs/integrations/feishu-aily-model-tools.md": _read(
            "docs/integrations/feishu-aily-model-tools.md"
        ),
        "docs/deployment/competition-ecs.md": _read(
            "docs/deployment/competition-ecs.md"
        ),
        "docs/deployment/security-and-secrets.md": _read(
            "docs/deployment/security-and-secrets.md"
        ),
    }
    common_markers = (
        "feishu-csv-registrations.example.json",
        "feishu-csv-registrations.json",
        "payload_sha256",
        "metadata.source_sha256",
        "OBSERVED",
    )

    for path, document in documents.items():
        for marker in common_markers:
            assert marker in document, f"{path} must document {marker}"

    guide = documents["docs/integrations/feishu-aily-model-tools.md"]
    assert "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE" in guide
    assert "unknown payload" in guide

    ecs = documents["docs/deployment/competition-ecs.md"]
    assert "${CONFIG_ROOT}/feishu-csv-registrations.json" in ecs
    assert "sha256sum" in ecs


def test_public_docs_bind_the_latest_blast_evidence_and_field_limits() -> None:
    readme = _read("README.md")
    guide = _read("docs/integrations/feishu-aily-model-tools.md")

    for document in (readme, guide):
        for marker in (
            "blast_naumann_v1/validation-v2",
            "blast_scenarios_v1/scenario-v3",
            "19",
            "leave-one-condition",
            "parameter_refit=false",
            "LFP_FIELD_160AH",
            "no_health_target",
            "LOCAL_TIME_TIMEZONE_UNRESOLVED",
        ):
            assert marker in document

    assert "不是独立训练 holdout" in readme
    assert "不能生成 SOH 精度" in guide


def test_aily_openapi_matches_the_production_scenario_only_facade() -> None:
    openapi = _read("docs/integrations/aily-connector-openapi.yaml")
    scenario_context_path = openapi.split(
        "  /v1/aily/scenario-contexts:\n",
        maxsplit=1,
    )[1].split("  /v1/aily/analysis-tasks:\n", maxsplit=1)[0]
    analysis_task_path = openapi.split(
        "  /v1/aily/analysis-tasks:\n",
        maxsplit=1,
    )[1].split("  /v1/aily/analysis-tasks/{run_id}:\n", maxsplit=1)[0]
    create_request_schema = openapi.split(
        "    CreateAnalysisTaskRequest:\n",
        maxsplit=1,
    )[1].split("    CreateScenarioContextRequest:\n", maxsplit=1)[0]

    assert "        '201':" in scenario_context_path
    assert "        '200':" not in scenario_context_path
    assert "        '202':" in analysis_task_path
    assert "ScenarioAnalysisTaskRequest" in create_request_schema
    assert "BatchAnalysisTaskRequest" not in create_request_schema
    for unsupported in (
        "predict_cycle_life",
        "predict_soh_trajectory",
        "ingest_observed_soh",
        "update_trajectory",
    ):
        assert unsupported not in create_request_schema


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


def test_readme_links_the_public_documentation_spine() -> None:
    readme = _read("README.md")
    expected = {
        "docs/architecture/system-overview.md": (
            "# 系统总体设计",
            "ToolResult",
            "个人比赛",
        ),
        "ARCHITECTURE.md": ("# 架构说明", "信任边界", "competition.compose.yaml"),
        "docs/architecture/module-design.md": (
            "# 模块设计",
            "src/quanxin_life/core",
            "依赖",
        ),
        "docs/api/README.md": ("# API 文档", "OpenAPI", "Idempotency-Key"),
        "docs/development/code-walkthrough.md": (
            "# 代码走读",
            "Next.js",
            "Audit Ledger",
        ),
        "docs/algorithms/README.md": (
            "# 算法原理",
            "CyclePatch",
            "HybridPatch-v2",
            "Conformal",
        ),
        "docs/adr/README.md": ("# Architecture Decision Records", "ADR-0001"),
        "docs/deployment/competition-ecs.md": (
            "# 竞赛 ECS 单机部署",
            "competition.compose.yaml",
            "Not deployed",
        ),
        "docs/deployment/operations-runbook.md": (
            "# 竞赛部署运维 Runbook",
            "回滚",
            "备份",
        ),
        "docs/deployment/security-and-secrets.md": (
            "# 部署安全与 Secrets",
            "secret file",
            "TLS",
        ),
    }

    for relative_path, markers in expected.items():
        assert relative_path in readme
        document = _read(relative_path)
        for marker in markers:
            assert marker in document

    for state in ("Implemented", "Validated", "Deployed", "Planned"):
        assert state in readme

    assert "docs/assets/benchmark/advanced-final-20260723/" in readme


def test_pull_request_template_requires_documentation_impact_review() -> None:
    template = _read(".github/PULL_REQUEST_TEMPLATE.md")

    for marker in (
        "README",
        "docs/status.md",
        "Implemented",
        "Validated",
        "Published",
        "Deployed",
        "Demonstrated",
        "Planned",
        "ADR",
        "Markdown",
        "secrets",
    ):
        assert marker in template


def test_public_evidence_bytes_have_stable_git_attributes() -> None:
    attributes = _read(".gitattributes")

    assert (
        "docs/source-data/advanced-final-20260723/*.csv text eol=lf" in attributes
    )
    assert (
        "docs/assets/benchmark/advanced-final-20260723/manifest.json text eol=lf"
        in attributes
    )
    assert (
        "docs/assets/benchmark/advanced-final-20260723/*.png binary" in attributes
    )


def test_readme_links_the_published_competition_release_record() -> None:
    readme = _read("README.md")
    release = _read("docs/deployment/releases/2026.07.28-1.md")

    assert "docs/deployment/releases/2026.07.28-1.md" in readme
    for marker in (
        "2026.07.28-1",
        "912f8ae05a0c3a08120bf7dcf4634273986a3a53",
        "quanxin-backend@sha256:6832cc958e0699d1e37a9057e128c2168076d7b47cca911f2c5d4e378fc1532d",
        "quanxin-frontend@sha256:25dc676f4e96db57eceb7512468c4800afc0f6bf2401cc6bf51a936365030cde",
        "quanxin-postgres@sha256:87f0ea0960ba8e710fc2202594fde52fc9f53384a23dd937f924741d87070697",
        "quanxin-redis@sha256:5fd679dbad4f101639b55d1c49a2aedfc15a2e7ddc43a83b783671b8d9541134",
        "quanxin-nginx@sha256:ebee5f752c662a314744c6e1e6ba57f10f15a98f5c797872e0505a1dbfa56720",
    ):
        assert marker in release

    assert "https://47.98.37.232" in release


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
