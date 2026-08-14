from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.cards import (
    AuditedCardBuilder,
    AuditedCardError,
    AuditedResultAuthorization,
    FeishuCardStatus,
    build_status_card,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    SCENARIO_FEATURE_VERSION,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    execute_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 7, tzinfo=UTC)


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="uploaded-csv",
            source_kind=SourceKind.OBSERVED,
            uri="feishu://message/om_source/resource/file_source",
            sha256="a" * 64,
            description="Verified Feishu CSV attachment",
            created_at=NOW,
        ),
    )


class _Authorizer:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        del result
        return AuditedResultAuthorization(
            allowed=self.allowed,
            route_id="quality-validation",
            activation_status="ACTIVE" if self.allowed else "NOT_ACTIVATED",
            evidence_level=EvidenceLevel.DATA_DIRECT,
            supported_domain="validated-upload",
            rejection_reason=None if self.allowed else "MODEL_ROUTE_NOT_ACTIVATED",
        )


class _BindingVerifier:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str]] = []

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool:
        self.calls.append((run_id, result_id))
        return self.allowed


class _ScenarioAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "compare_operation_scenarios"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            activation_status="REGISTERED_CANDIDATE",
            evidence_level=EvidenceLevel.PHYSICS_REFERENCE,
            supported_domain="manifest-bounded-reference-scenario",
        )


class _ModelAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "predict_cycle_life"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="matr-rul-active-route",
            activation_status="ACTIVE",
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            supported_domain="activated project-bound MATR official cycle-life route",
        )


class _SohAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "predict_soh_trajectory"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="matr-soh-active-route",
            activation_status="ACTIVE",
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            supported_domain="activated project-bound MATR finite SOH route",
        )


def _builder(
    ledger: AuditLedger,
    *,
    authorizer: _Authorizer | None = None,
    binding_verifier: _BindingVerifier | None = None,
) -> AuditedCardBuilder:
    return AuditedCardBuilder(
        ledger,
        authorizer=authorizer or _Authorizer(),
        binding_verifier=binding_verifier or _BindingVerifier(),
    )


def _validation_result() -> ToolResult:
    return execute_validate_battery_data_tool(
        ValidateBatteryDataToolInput(
            records=(),
            data_version="uploaded-data-v1",
            feature_version="raw-cycle-v1",
            provenance=_provenance(),
            validated_at=NOW,
        )
    )


def _cell_metadata() -> dict[str, object]:
    return {
        "dataset_id": "MATR",
        "cell_id": "MATR_b3c34",
        "raw_cell_id": "b3c34",
        "chemistry": "LFP/graphite",
        "nominal_capacity_ah": 1.1,
        "reference_capacity_ah": 1.0,
        "eol_threshold": 0.8,
        "protocol_id": "MATR-standard",
        "protocol_description": "公开 MATR 小容量 LFP 循环测试",
        "source_uri": "trusted-store://matr/b3c34",
        "source_sha256": "d" * 64,
        "schema_version": "cell-metadata-v1",
        "adapter_version": "matr-adapter-v1",
    }


def _cycle_life_result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="advanced-rul-prediction-tool-v1",
        model_version="matr-cyclepatch-direct-cutoff-50-seed-39",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="b" * 64,
        values={
            "artifact_type": "quanxin_life.advanced_rul_prediction.v2",
            "artifact": {
                "cell_metadata": _cell_metadata(),
                "dataset_id": "MATR",
                "cell_id": "MATR_b3c34",
                "cutoff_cycle": 50,
                "cycle_life_prediction": {"predicted_cycle": 1074.4998779296875},
                "derived_remaining_cycles": 1024.4998779296875,
            },
        },
        uncertainty=None,
        warnings=[],
        provenance=list(_provenance()),
        created_at=NOW,
    )


def _soh_result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_soh_trajectory",
        tool_version="advanced-soh-prediction-tool-v1",
        model_version="matr-hybridpatch-cutoff-50-seed-38",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="c" * 64,
        values={
            "artifact_type": "quanxin_life.advanced_soh_trajectory.v2",
            "artifact": {
                "cell_metadata": _cell_metadata(),
                "dataset_id": "MATR",
                "cell_id": "MATR_b3c34",
                "cutoff_cycle": 50,
                "prediction_cycles": [51, 500],
                "predicted_soh": [0.99, 0.87],
                "horizon_end_cycle": 500,
            },
        },
        uncertainty={
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        },
        warnings=[],
        provenance=list(_provenance()),
        created_at=NOW,
    )


def test_audited_card_reads_allowlisted_values_from_registered_tool_result() -> None:
    result = _validation_result()
    ledger = AuditLedger((result,))

    card = _builder(ledger).build_result_card(
        run_id="run-safe",
        result_id=result.result_id,
    )
    rendered = json.dumps(card, ensure_ascii=False)

    assert "数据质量检查" in rendered
    assert "是否阻断" in rendered
    assert "质量评分" in rendered
    assert "**质量评分**\\n0" in rendered
    assert result.result_id not in rendered
    assert "values." not in rendered
    assert result.model_version not in rendered
    assert result.data_version not in rendered
    assert result.feature_version not in rendered


def test_cycle_life_card_is_a_chinese_engineering_analysis_sheet() -> None:
    result = _cycle_life_result()
    card = AuditedCardBuilder(
        AuditLedger((result,)),
        authorizer=_ModelAuthorizer(),
        binding_verifier=_BindingVerifier(),
    ).build_result_card(run_id="run-safe", result_id=result.result_id)

    rendered = json.dumps(card, ensure_ascii=False)

    assert "MATR_b3c34 | 早期寿命评估" in rendered
    assert "MATR_b3c34" in rendered
    assert "化学体系" in rendered
    assert "LFP/graphite" in rendered
    assert "标称容量" in rendered
    assert "1.1 Ah" in rendered
    assert "数据来源" in rendered
    assert "**数据来源**\\nMATR" in rendered
    assert "trusted-store://matr/b3c34" not in rendered
    assert "测试规格" in rendered
    assert "公开 MATR 小容量 LFP 循环测试" in rendered
    assert "已观测循环" in rendered
    assert "50" in rendered
    assert "预测总循环寿命" in rendered
    assert "1,074.5" in rendered
    assert "剩余循环" in rendered
    assert "1,024.5" in rendered
    assert "已激活" in rendered
    assert "模型推理" in rendered
    assert "年份需基于明确工况单独推演" in rendered
    assert "run-safe" not in rendered
    assert result.result_id not in rendered
    assert "values.artifact" not in rendered
    assert result.model_version not in rendered
    assert result.data_version not in rendered
    assert result.feature_version not in rendered
    assert "15 年" not in rendered
    assert "20 年" not in rendered
    assert "25 年" not in rendered


def test_soh_card_explains_the_finite_cycle_boundary_without_confidence_claims() -> None:
    result = _soh_result()
    card = AuditedCardBuilder(
        AuditLedger((result,)),
        authorizer=_SohAuthorizer(),  # type: ignore[arg-type]
        binding_verifier=_BindingVerifier(),
    ).build_result_card(run_id="run-safe", result_id=result.result_id)

    rendered = json.dumps(card, ensure_ascii=False)

    assert "MATR_b3c34 | 有限时域 SOH" in rendered
    assert "MATR_b3c34" in rendered
    assert "LFP/graphite" in rendered
    assert "1.1 Ah" in rendered
    assert "已观测循环" in rendered
    assert "50" in rendered
    assert "预测边界循环" in rendered
    assert "500" in rendered
    assert "边界周期 SOH" in rendered
    assert "0.87" in rendered
    assert "仅覆盖至第 500 个循环" in rendered
    assert "15 年" not in rendered
    assert "可信度" not in rendered
    assert result.result_id not in rendered
    assert result.model_version not in rendered


def test_missing_cell_metadata_is_explicitly_unavailable_instead_of_guessed() -> None:
    result = _cycle_life_result().model_copy(
        update={
            "values": {
                **_cycle_life_result().values,
                "artifact_type": "quanxin_life.advanced_rul_prediction.v1",
                "artifact": {
                    key: value
                    for key, value in _cycle_life_result().values["artifact"].items()
                    if key != "cell_metadata"
                },
            }
        }
    )

    card = AuditedCardBuilder(
        AuditLedger((result,)),
        authorizer=_ModelAuthorizer(),
        binding_verifier=_BindingVerifier(),
    ).build_result_card(run_id="run-safe", result_id=result.result_id)
    rendered = json.dumps(card, ensure_ascii=False)

    assert "化学体系" in rendered
    assert "标称容量" in rendered
    assert "数据来源" in rendered
    assert "测试规格" in rendered
    assert rendered.count("未登记") >= 4
    assert "1.1 Ah" not in rendered


def test_v2_result_without_cell_metadata_is_rejected() -> None:
    result = _cycle_life_result()
    artifact = result.values["artifact"]
    damaged = result.model_copy(
        update={
            "values": {
                **result.values,
                "artifact": {
                    key: value
                    for key, value in artifact.items()
                    if key != "cell_metadata"
                },
            }
        }
    )

    with pytest.raises(AuditedCardError, match="metadata"):
        AuditedCardBuilder(
            AuditLedger((damaged,)),
            authorizer=_ModelAuthorizer(),
            binding_verifier=_BindingVerifier(),
        ).build_result_card(run_id="run-safe", result_id=damaged.result_id)


def test_cell_metadata_dataset_identity_mismatch_is_rejected() -> None:
    result = _cycle_life_result()
    artifact = result.values["artifact"]
    mismatched = result.model_copy(
        update={
            "values": {
                **result.values,
                "artifact": {
                    **artifact,
                    "cell_metadata": {
                        **artifact["cell_metadata"],
                        "dataset_id": "OTHER",
                    },
                },
            }
        }
    )

    with pytest.raises(AuditedCardError, match="identity"):
        AuditedCardBuilder(
            AuditLedger((mismatched,)),
            authorizer=_ModelAuthorizer(),
            binding_verifier=_BindingVerifier(),
        ).build_result_card(run_id="run-safe", result_id=mismatched.result_id)


def test_audited_card_public_api_rejects_caller_supplied_numeric_value() -> None:
    result = _validation_result()
    builder = _builder(AuditLedger((result,)))

    with pytest.raises(TypeError):
        builder.build_result_card(  # type: ignore[call-arg]
            run_id="run-safe",
            result_id=result.result_id,
            value=object(),
        )


def test_missing_or_unsupported_result_is_rejected() -> None:
    with pytest.raises(AuditedCardError, match="registered"):
        _builder(AuditLedger()).build_result_card(
            run_id="run-safe",
            result_id=str(uuid4()),
        )

    unsupported = _validation_result().model_copy(
        update={"result_id": str(uuid4()), "tool_version": "unknown-version"}
    )
    with pytest.raises(AuditedCardError, match="allowlisted"):
        _builder(
            AuditLedger((unsupported,)), authorizer=_Authorizer()
        ).build_result_card(run_id="run-safe", result_id=unsupported.result_id)


def test_inactive_model_result_returns_rejection_card_before_reading_values() -> None:
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_cycle_life",
        tool_version="advanced-cycle-life-prediction-tool-v1",
        model_version="candidate-route-v1",
        data_version="registered-data-v1",
        feature_version="registered-feature-v1",
        input_hash="b" * 64,
        values={"untrusted_numeric_payload": object().__class__.__name__},
        uncertainty=None,
        warnings=[],
        provenance=list(_provenance()),
        created_at=NOW,
    )

    card = _builder(
        AuditLedger((result,)), authorizer=_Authorizer(allowed=False)
    ).build_result_card(run_id="run-safe", result_id=result.result_id)
    rendered = json.dumps(card, ensure_ascii=False)

    assert "模型路线尚未激活" in rendered
    assert "MODEL_ROUTE_NOT_ACTIVATED" not in rendered
    assert "untrusted_numeric_payload" not in rendered


def test_scenario_card_reads_only_fixed_scalar_summaries_and_embeds_curve() -> None:
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="compare_operation_scenarios",
        tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        model_version="blast-lite-route-v1",
        data_version="scenario-data-v1",
        feature_version=SCENARIO_FEATURE_VERSION,
        input_hash="b" * 64,
        values={
            "artifact": {
                "status": "COMPLETED",
                "baseline": {
                    "scenario_id": "baseline",
                    "scenario_version": "baseline-v1",
                    "final_natural_year": 25.0,
                    "final_equivalent_full_cycles": 7500.0,
                    "final_soh": 0.88,
                    "milestone_soh": {"15": 0.93, "20": 0.90, "25": 0.88},
                    "eol_threshold": 0.8,
                    "eol": {
                        "status": "NOT_REACHED",
                        "natural_year": None,
                        "equivalent_full_cycles": None,
                    },
                    "support": {"status": "SUPPORTED"},
                    "operating_segments": [
                        {
                            "segment_id": "all-years",
                            "start_year": 0,
                            "end_year": 25,
                            "temperature_c": 25.0,
                            "charge_c_rate": 0.5,
                            "discharge_c_rate": 0.5,
                            "soc_lower_bound": 0.1,
                            "soc_upper_bound": 0.9,
                            "dod": 0.8,
                            "equivalent_full_cycles_per_year": 300.0,
                            "rest_duration_hours": 1.0,
                        }
                    ],
                    "natural_years": [0.0, 25.0],
                    "soh": [1.0, 0.88],
                },
                "comparisons": [
                    {
                        "scenario_id": "warmer",
                        "scenario_version": "warmer-v1",
                        "final_natural_year": 25.0,
                        "final_equivalent_full_cycles": 7500.0,
                        "final_soh": 0.81,
                        "milestone_soh": {"15": 0.86, "20": 0.82, "25": 0.81},
                        "eol_threshold": 0.8,
                        "eol": {
                            "status": "REACHED",
                            "natural_year": 19.2,
                            "equivalent_full_cycles": 5760.0,
                        },
                        "support": {"status": "NEAR_BOUNDARY"},
                        "operating_segments": [
                            {
                                "segment_id": "all-years",
                                "start_year": 0,
                                "end_year": 25,
                                "temperature_c": 35.0,
                                "charge_c_rate": 0.5,
                                "discharge_c_rate": 1.0,
                                "soc_lower_bound": 0.1,
                                "soc_upper_bound": 0.9,
                                "dod": 0.8,
                                "equivalent_full_cycles_per_year": 300.0,
                                "rest_duration_hours": 1.0,
                            }
                        ],
                        "natural_years": [0.0, 25.0],
                        "soh": [1.0, 0.81],
                    }
                ],
            }
        },
        warnings=["CANDIDATE_ROUTE_RESEARCH_USE_ONLY"],
        provenance=list(_provenance()),
        created_at=NOW,
    )
    card = _builder(
        AuditLedger((result,)),
        authorizer=_ScenarioAuthorizer(),  # type: ignore[arg-type]
    ).build_result_card(
        run_id="run-safe",
        result_id=result.result_id,
        image_key="img-scenario-safe",
    )
    rendered = json.dumps(card, ensure_ascii=False)

    assert "储能工况年份推演" in rendered
    assert "基准工况" in rendered
    assert "对比工况 1" in rendered
    assert "推演终点" in rendered
    assert "25 年" in rendered
    assert "15 年 SOH" in rendered
    assert "0.93" in rendered
    assert "20 年 SOH" in rendered
    assert "0.9" in rendered
    assert "25 年 SOH" in rendered
    assert "0.88" in rendered
    assert "首次达到阈值年份" in rendered
    assert "19.2 年" in rendered
    assert "物理参考情景" in rendered
    assert "支持范围内" in rendered
    assert "接近支持边界" in rendered
    assert "25°C" in rendered
    assert "充电 0.5C / 放电 0.5C" in rendered
    assert "SOC 10%-90%" in rendered
    assert "DoD 80%" in rendered
    assert "300 EFC/年" in rendered
    assert "静置 1 小时" in rendered
    assert "35°C" in rendered
    assert "放电 1C" in rendered
    assert "不是 MATR 电芯的自然年换算" in rendered
    assert "values.artifact" not in rendered
    assert "PHYSICS_REFERENCE" not in rendered
    assert "CANDIDATE_ROUTE_RESEARCH_USE_ONLY" not in rendered
    assert "img-scenario-safe" in rendered
    assert "natural_years" not in rendered
    assert "run-safe" not in rendered
    assert result.result_id not in rendered


def test_unbound_result_is_rejected_before_ledger_resolution() -> None:
    result = _validation_result()
    verifier = _BindingVerifier(allowed=False)

    with pytest.raises(AuditedCardError, match="not bound"):
        _builder(
            AuditLedger((result,)), binding_verifier=verifier
        ).build_result_card(run_id="run-safe", result_id=result.result_id)

    assert verifier.calls == [("run-safe", result.result_id)]


@pytest.mark.parametrize(
    ("status", "expected_title"),
    (
        (FeishuCardStatus.RECEIVED, "文件已接收"),
        (FeishuCardStatus.DATA_CHECK, "正在检查数据"),
        (FeishuCardStatus.QUEUED, "任务已排队"),
        (FeishuCardStatus.RUNNING, "分析进行中"),
        (FeishuCardStatus.SUCCESS, "分析已完成"),
        (FeishuCardStatus.REJECTED, "分析未执行"),
        (FeishuCardStatus.DEGRADED, "结果已降级"),
        (FeishuCardStatus.REPORT_READY, "分析报告已生成"),
    ),
)
def test_status_cards_are_chinese_and_hide_machine_references(
    status: FeishuCardStatus,
    expected_title: str,
) -> None:
    rendered = json.dumps(
        build_status_card(status=status, run_id="run-safe", result_id=None),
        ensure_ascii=False,
    )

    assert expected_title in rendered
    assert "run-safe" not in rendered
    assert "SOH" not in rendered
    assert "RUL" not in rendered
    assert "置信区间" not in rendered


def test_rejection_status_card_translates_reviewed_reason_code() -> None:
    rendered = json.dumps(
        build_status_card(
            status=FeishuCardStatus.REJECTED,
            run_id="run-safe",
            reason_code="MODEL_ROUTE_NOT_ACTIVATED",
            task_type="predict_cycle_life",
        ),
        ensure_ascii=False,
    )

    assert "循环寿命预测" in rendered
    assert "模型路线尚未激活" in rendered
    assert "MODEL_ROUTE_NOT_ACTIVATED" not in rendered
    assert "predict_cycle_life" not in rendered
    assert "run-safe" not in rendered
