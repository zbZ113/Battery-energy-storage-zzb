from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import (
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.integrations.feishu.analysis_bitable import (
    AnalysisBitableProjectionError,
    build_audited_analysis_bitable_fields,
)
from tests.unit.integrations.feishu.test_audited_cards import (
    _cycle_life_result,
    _soh_result,
)
from tests.unit.integrations.feishu.test_scenario_plot import (
    _result as _scenario_result,
)


def _recommendation_result() -> ToolResult:
    upstream_result_id = str(uuid4())
    ruleset_id = "reviewed-release-gate"
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="make_engineering_recommendation",
        tool_version="engineering-recommendation-tool-v1",
        model_version="engineering-recommendation-rule-engine-v1",
        data_version="reviewed-release-gate-v1",
        feature_version="engineering-recommendation-evidence-v1",
        input_hash=sha256_canonical(
            {
                "result_ids": [upstream_result_id],
                "ruleset_id": ruleset_id,
            }
        ),
        values={
            "recommendation": "RECHECK_REQUIRED",
            "ruleset_id": ruleset_id,
            "ruleset_version": "reviewed-release-gate-v1",
            "ruleset_manifest_sha256": "b" * 64,
            "authorized_upstream_result_ids": [upstream_result_id],
            "authorized_result_selectors": [
                {
                    "result_tool_name": "predict_cycle_life",
                    "result_tool_version": "rul-v1",
                }
            ],
            "evaluated_value_paths": ["values.score"],
            "threshold_evidence": [
                {
                    "rule_id": "reviewed-rule",
                    "result_id": upstream_result_id,
                    "result_tool_name": "predict_cycle_life",
                    "result_tool_version": "rul-v1",
                    "value_path": "values.score",
                    "comparator": "GTE",
                    "threshold": 900.0,
                    "actual_value": 850.0,
                    "comparison_passed": False,
                    "recheck_reason_code": "RUL_BELOW_REVIEWED_GATE",
                }
            ],
            "reason_codes": ["RUL_BELOW_REVIEWED_GATE"],
            "resolution_issues": [],
        },
        warnings=["RUL_BELOW_REVIEWED_GATE"],
        provenance=[
            ProvenanceRecord(
                source_id="reviewed-release-gate",
                source_kind=SourceKind.OBSERVED,
                uri="configuration://reviewed-release-gate",
                sha256="c" * 64,
                description="Reviewed release gate.",
                created_at=datetime(2026, 8, 14, tzinfo=UTC),
            )
        ],
        created_at=datetime(2026, 8, 14, tzinfo=UTC),
    )


def test_projects_cycle_life_business_fields_without_natural_year_claims() -> None:
    fields = build_audited_analysis_bitable_fields(_cycle_life_result())

    assert fields == {
        "analysis_summary": "已完成个体早期循环寿命预测",
        "applicability": "结果为循环次数预测。不表示自然年寿命",
        "cell_name": "MATR_b3c34",
        "chemistry": "LFP/graphite",
        "nominal_capacity_ah": "1.1 Ah",
        "data_source": "MATR",
        "test_specification": "公开 MATR 小容量 LFP 循环测试",
        "observed_cycle_count": "50",
        "predicted_total_cycles": "1074.5",
        "predicted_remaining_cycles": "1024.5",
    }
    assert "year" not in repr(fields).lower()
    assert all(not isinstance(value, list | dict) for value in fields.values())


def test_projects_finite_soh_boundary_without_extrapolating_the_curve() -> None:
    fields = build_audited_analysis_bitable_fields(_soh_result())

    assert fields == {
        "analysis_summary": "已完成有限循环 SOH 轨迹预测",
        "applicability": "仅覆盖至第 500 个循环。不表示自然年寿命",
        "cell_name": "MATR_b3c34",
        "chemistry": "LFP/graphite",
        "nominal_capacity_ah": "1.1 Ah",
        "data_source": "MATR",
        "test_specification": "公开 MATR 小容量 LFP 循环测试",
        "observed_cycle_count": "50",
        "soh_horizon_cycle": "500",
        "soh_horizon_value": "0.87",
    }
    assert "predicted_soh" not in fields
    assert all(not isinstance(value, list | dict) for value in fields.values())


def test_projects_reference_scenario_identity_without_raw_trajectory_arrays() -> None:
    fields = build_audited_analysis_bitable_fields(_scenario_result())

    assert fields == {
        "analysis_summary": "已完成参考工况退化对比",
        "applicability": "结果为物理参考工况。不是目标电芯个体寿命结论",
        "scenario_id": "baseline",
        "scenario_version": "baseline-v1",
    }
    assert all(not isinstance(value, list | dict) for value in fields.values())


def test_rejects_inconsistent_soh_horizon_instead_of_selecting_a_value() -> None:
    result = _soh_result()
    artifact = dict(result.values["artifact"])
    artifact["horizon_end_cycle"] = 499
    damaged = result.model_copy(
        update={"values": {**result.values, "artifact": artifact}}
    )

    with pytest.raises(AnalysisBitableProjectionError, match="horizon"):
        build_audited_analysis_bitable_fields(damaged)


def test_projects_controlled_chinese_recommendation_without_thresholds() -> None:
    fields = build_audited_analysis_bitable_fields(_recommendation_result())

    assert fields == {
        "analysis_summary": "已生成工程综合建议",
        "applicability": "建议由受审规则集生成。阈值与证据路径见详细报告",
        "recommendation": "建议复检",
        "recommendation_reason": "至少一项受审规则未通过。建议复检",
        "recommendation_ruleset_version": "reviewed-release-gate-v1",
    }
    rendered = repr(fields)
    assert "900" not in rendered
    assert "850" not in rendered
    assert "RUL_BELOW_REVIEWED_GATE" not in rendered
