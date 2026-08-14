"""Deterministic Feishu cards built from traceable audit-ledger references."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any, Protocol

from quanxin_life.core import EvidenceLevel, ToolResult
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
    SCENARIO_FEATURE_VERSION,
)
from quanxin_life.tools.cell_metadata_evidence import (
    CellMetadataEvidenceIdentityError,
    validate_versioned_cell_metadata_evidence,
)
from quanxin_life.tools.data_quality import DATA_QUALITY_TOOL_VERSION

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class AuditedCardError(ValueError):
    """Raised when a card cannot prove its display evidence."""


class FeishuCardStatus(StrEnum):
    RECEIVED = "RECEIVED"
    DATA_CHECK = "DATA_CHECK"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    DEGRADED = "DEGRADED"
    REPORT_READY = "REPORT_READY"


@dataclass(frozen=True, slots=True)
class AuditedResultAuthorization:
    allowed: bool
    route_id: str
    activation_status: str
    evidence_level: EvidenceLevel
    supported_domain: str
    rejection_reason: str | None = None


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class RunResultBindingVerifier(Protocol):
    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool: ...


class AuditedResultAuthorizer(Protocol):
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization: ...


@dataclass(frozen=True, slots=True)
class _CardFieldPolicy:
    label: str
    json_path: str


_CARD_POLICIES: dict[tuple[str, str], tuple[_CardFieldPolicy, ...]] = {
    (
        "validate_battery_data",
        DATA_QUALITY_TOOL_VERSION,
    ): (
        _CardFieldPolicy("是否阻断", "values.blocked"),
        _CardFieldPolicy("质量评分", "values.quality_score"),
        _CardFieldPolicy("问题数量", "values.issue_count"),
    ),
    (
        "predict_soh_trajectory",
        "advanced-soh-prediction-tool-v1",
    ): (
        _CardFieldPolicy("预测边界循环", "values.artifact.horizon_end_cycle"),
    ),
}


_STATUS_PRESENTATION = {
    FeishuCardStatus.RECEIVED: (
        "文件已接收",
        "blue",
        "文件已进入受控分析流程。",
    ),
    FeishuCardStatus.DATA_CHECK: (
        "正在检查数据",
        "blue",
        "正在核对文件格式、字段、来源和数据质量。",
    ),
    FeishuCardStatus.QUEUED: (
        "任务已排队",
        "blue",
        "数据检查已完成, 任务正在等待计算资源。",
    ),
    FeishuCardStatus.RUNNING: (
        "分析进行中",
        "blue",
        "正在执行受审计的模型或情景工具。",
    ),
    FeishuCardStatus.SUCCESS: (
        "分析已完成",
        "green",
        "结果已经登记, 可继续查看分析报告。",
    ),
    FeishuCardStatus.REJECTED: (
        "分析未执行",
        "red",
        "当前输入或项目状态未满足受控分析条件。",
    ),
    FeishuCardStatus.DEGRADED: (
        "结果已降级",
        "orange",
        "结果仍可供参考, 但存在需要人工复核的边界。",
    ),
    FeishuCardStatus.REPORT_READY: (
        "分析报告已生成",
        "green",
        "受审计报告已生成并完成完整性校验。",
    ),
}
_SIMPLE_CARD_PRESENTATION = {
    ("validate_battery_data", DATA_QUALITY_TOOL_VERSION): ("数据质量检查", "blue"),
    ("predict_soh_trajectory", "advanced-soh-prediction-tool-v1"): (
        "有限时域 SOH 分析",
        "green",
    ),
}
_SCENARIO_RESULT_VERSIONS = {
    "compare_operation_scenarios": COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
    "project_storage_lifetime": PROJECT_STORAGE_LIFETIME_TOOL_VERSION,
}
_TASK_PRESENTATION = {
    "predict_cycle_life": "循环寿命预测",
    "predict_soh_trajectory": "有限时域 SOH 预测",
    "compare_operation_scenarios": "工况对比推演",
    "project_storage_lifetime": "储能寿命情景推演",
    "generate_audited_report": "分析报告",
}
_REASON_PRESENTATION = {
    "MODEL_ROUTE_NOT_ACTIVATED": "模型路线尚未激活",
    "PROJECT_RECORD_BATCH_NOT_FROZEN": "项目数据批次尚未冻结",
    "DATA_REGISTRATION_REJECTED": "数据登记未通过",
    "ANALYSIS_TASK_NOT_SUPPORTED": "当前分析类型暂不支持",
    "SCENARIO_OUTSIDE_SUPPORTED_RANGE": "工况超出参考模型支持范围",
    "SCENARIO_PARAMETERS_REQUIRED": "需要补充温度、倍率、SOC、DoD等工况参数",
    "ROUTE_NOT_ACTIVATED": "参考情景路线尚未启用",
    "UNSUPPORTED_BLAST_STATE_INITIALIZATION": "当前参考模型不能从该状态继续推演",
}
_ACTIVATION_PRESENTATION = {
    "ACTIVE": "已激活",
    "REGISTERED_CANDIDATE": "候选参考路线",
    "NOT_ACTIVATED": "未激活",
}
_EVIDENCE_PRESENTATION = {
    EvidenceLevel.DATA_DIRECT.value: "直接数据",
    EvidenceLevel.MODEL_INFERENCE.value: "模型推理",
    EvidenceLevel.PHYSICS_REFERENCE.value: "物理参考情景",
}
_SUPPORT_PRESENTATION = {
    "SUPPORTED": "支持范围内",
    "NEAR_BOUNDARY": "接近支持边界",
    "REJECTED": "超出支持范围",
}
_EOL_PRESENTATION = {
    "REACHED": "已达到",
    "NOT_REACHED": "未达到",
}
_WARNING_PRESENTATION = {
    "EMPTY_CELL": "未检测到可用于分析的有效记录",
    "SMALL_CALIBRATION_COHORT": "校准样本量较小, 结果需要谨慎解释",
    "CANDIDATE_ROUTE_RESEARCH_USE_ONLY": "当前情景仅用于研究参考",
    "REFERENCE_MODEL_NOT_CELL_SPECIFIC": "参考模型并非针对当前电芯定制",
    "LONG_HORIZON_IS_SCENARIO_NOT_EXPERIMENTAL_VALIDATION": (
        "长期结果是工况情景, 不是目标电芯的长期实验验证"
    ),
}


def build_status_card(
    *,
    status: FeishuCardStatus,
    run_id: str,
    result_id: str | None = None,
    reason_code: str | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    checked_run_id = _identifier(run_id, field_name="run_id")
    del checked_run_id
    title, template, description = _STATUS_PRESENTATION[status]
    fields: list[dict[str, Any]] = []
    if task_type is not None:
        checked_task = _identifier(task_type, field_name="task_type")
        fields.append(
            _display_field(
                "分析项目",
                _TASK_PRESENTATION.get(checked_task, "分析任务"),
                is_short=False,
            )
        )
    if result_id is not None:
        _identifier(result_id, field_name="result_id")
    if reason_code is not None:
        checked_reason = _identifier(reason_code, field_name="reason_code")
        fields.append(
            _display_field(
                "说明",
                _REASON_PRESENTATION.get(
                    checked_reason,
                    "任务未满足执行条件, 请查看审计记录。",
                ),
                is_short=False,
            )
        )
    elements: list[dict[str, Any]] = [_paragraph(description)]
    if fields:
        elements.append({"tag": "div", "fields": fields})
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "elements": elements,
    }


class AuditedCardBuilder:
    """Resolve and render only fixed paths from a registered, authorized result."""

    def __init__(
        self,
        resolver: RegisteredResultResolver,
        *,
        authorizer: AuditedResultAuthorizer,
        binding_verifier: RunResultBindingVerifier,
    ) -> None:
        if not callable(getattr(resolver, "resolve_registered_result", None)):
            raise TypeError("resolver must resolve registered ToolResults")
        if not callable(getattr(authorizer, "authorize", None)):
            raise TypeError("authorizer must validate result display eligibility")
        if not callable(
            getattr(binding_verifier, "is_result_bound_to_run", None)
        ):
            raise TypeError("binding_verifier must validate run-result bindings")
        self._resolver = resolver
        self._authorizer = authorizer
        self._binding_verifier = binding_verifier

    def build_result_card(
        self,
        *,
        run_id: str,
        result_id: str,
        image_key: str | None = None,
    ) -> dict[str, Any]:
        checked_run_id = _identifier(run_id, field_name="run_id")
        checked_result_id = _identifier(result_id, field_name="result_id")
        try:
            bound = self._binding_verifier.is_result_bound_to_run(
                run_id=checked_run_id,
                result_id=checked_result_id,
            )
        except Exception as exc:
            raise AuditedCardError(
                "ToolResult run binding could not be verified"
            ) from exc
        if bound is not True:
            raise AuditedCardError("ToolResult is not bound to the requested run")
        try:
            result = ToolResult.model_validate(
                self._resolver.resolve_registered_result(
                    checked_result_id
                ).model_dump(mode="json")
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AuditedCardError("ToolResult is not registered for card display") from exc
        authorization = self._authorizer.authorize(result)
        if not authorization.allowed:
            reason = authorization.rejection_reason or "RESULT_DISPLAY_NOT_AUTHORIZED"
            return build_status_card(
                status=FeishuCardStatus.REJECTED,
                run_id=checked_run_id,
                result_id=result.result_id,
                reason_code=reason,
                task_type=result.tool_name,
            )
        scenario_version = _SCENARIO_RESULT_VERSIONS.get(result.tool_name)
        if scenario_version == result.tool_version:
            return _build_scenario_result_card(
                run_id=checked_run_id,
                result=result,
                authorization=authorization,
                image_key=image_key,
            )
        if (
            result.tool_name == "predict_cycle_life"
            and result.tool_version == "advanced-rul-prediction-tool-v1"
        ):
            return _build_cycle_life_result_card(
                result=result,
                authorization=authorization,
            )
        if (
            result.tool_name == "predict_soh_trajectory"
            and result.tool_version == "advanced-soh-prediction-tool-v1"
        ):
            return _build_soh_result_card(
                result=result,
                authorization=authorization,
                image_key=image_key,
            )
        if image_key is not None:
            raise AuditedCardError("images are only supported for plotted result cards")
        policy = _CARD_POLICIES.get((result.tool_name, result.tool_version))
        if policy is None:
            raise AuditedCardError("ToolResult version has no allowlisted card paths")
        result_mapping = result.model_dump(mode="json")
        value_fields = [
            _display_field(
                item.label,
                _format_scalar(_resolve_scalar(result_mapping, item.json_path)),
                is_short=True,
            )
            for item in policy
        ]
        title, template = _SIMPLE_CARD_PRESENTATION[(result.tool_name, result.tool_version)]
        elements: list[dict[str, Any]] = [{"tag": "div", "fields": value_fields}]
        elements.extend(_warning_elements(result.warnings))
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": template,
                "title": {"tag": "plain_text", "content": title},
            },
            "elements": elements,
        }


def build_run_reference_card(
    *,
    run_id: str,
    result_id: str | None = None,
) -> dict[str, Any]:
    """Build a status card without copying numerical results into Feishu."""

    return build_status_card(
        status=FeishuCardStatus.RUNNING,
        run_id=run_id,
        result_id=result_id,
    )


def _display_field(
    label: str,
    value: str,
    *,
    is_short: bool,
) -> dict[str, Any]:
    return {
        "is_short": is_short,
        "text": {
            "tag": "lark_md",
            "content": f"**{label}**\n{_escape_lark_markdown(value)}",
        },
    }


def _paragraph(content: str) -> dict[str, Any]:
    return {
        "text": {
            "tag": "lark_md",
            "content": content,
        },
        "tag": "div",
    }


def _build_cycle_life_result_card(
    *,
    result: ToolResult,
    authorization: AuditedResultAuthorization,
) -> dict[str, Any]:
    mapping = result.model_dump(mode="json")
    cell_id = _resolve_scalar(mapping, "values.artifact.cell_id")
    if not isinstance(cell_id, str):
        raise AuditedCardError("cycle-life cell identity is invalid")
    fields = [
        _display_field("电芯", cell_id, is_short=True),
        *_cell_metadata_fields(
            result,
            expected_cell_id=cell_id,
            legacy_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
        ),
        _display_field(
            "已观测循环",
            _format_count(_resolve_scalar(mapping, "values.artifact.cutoff_cycle")),
            is_short=True,
        ),
        _display_field(
            "预测总循环寿命",
            _format_count(
                _resolve_scalar(
                    mapping,
                    "values.artifact.cycle_life_prediction.predicted_cycle",
                )
            ),
            is_short=True,
        ),
        _display_field(
            "剩余循环",
            _format_count(
                _resolve_scalar(mapping, "values.artifact.derived_remaining_cycles")
            ),
            is_short=True,
        ),
    ]
    activation = _ACTIVATION_PRESENTATION.get(
        authorization.activation_status,
        "需人工复核",
    )
    evidence = _EVIDENCE_PRESENTATION.get(
        authorization.evidence_level.value,
        "受审计证据",
    )
    elements: list[dict[str, Any]] = [
        {"tag": "div", "fields": fields},
        _paragraph(
            "**结果说明**\n"
            f"模型路线: {activation}  证据类型: {evidence}\n"
            "该结果为循环寿命预测; 年份需基于明确工况单独推演。"
        ),
        _paragraph(
            "**适用边界**\n"
            "当前结果用于公开 MATR 数据的研究与比赛演示, "
            "不等同于目标工业电芯的现场寿命验证。"
        ),
    ]
    elements.extend(_warning_elements(result.warnings))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {
                "tag": "plain_text",
                "content": _bounded_text(
                    f"{cell_id} | 早期寿命评估",
                    field_name="card title",
                ),
            },
        },
        "elements": elements,
    }


def _build_soh_result_card(
    *,
    result: ToolResult,
    authorization: AuditedResultAuthorization,
    image_key: str | None,
) -> dict[str, Any]:
    mapping = result.model_dump(mode="json")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AuditedCardError("SOH artifact is invalid")
    cycles = artifact.get("prediction_cycles")
    predicted_soh = artifact.get("predicted_soh")
    if (
        not isinstance(cycles, list)
        or not isinstance(predicted_soh, list)
        or not cycles
        or len(cycles) != len(predicted_soh)
        or cycles[-1] != artifact.get("horizon_end_cycle")
    ):
        raise AuditedCardError("SOH trajectory is invalid")
    last_index = len(predicted_soh) - 1
    cell_id = _resolve_scalar(mapping, "values.artifact.cell_id")
    if not isinstance(cell_id, str):
        raise AuditedCardError("SOH cell identity is invalid")
    fields = [
        _display_field("电芯", cell_id, is_short=True),
        *_cell_metadata_fields(
            result,
            expected_cell_id=cell_id,
            legacy_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
            metadata_artifact_type=ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
        ),
        _display_field(
            "已观测循环",
            _format_count(_resolve_scalar(mapping, "values.artifact.cutoff_cycle")),
            is_short=True,
        ),
        _display_field(
            "预测边界循环",
            _format_count(
                _resolve_scalar(mapping, "values.artifact.horizon_end_cycle")
            ),
            is_short=True,
        ),
        _display_field(
            "边界周期 SOH",
            _format_scalar(
                _resolve_scalar(
                    mapping,
                    f"values.artifact.predicted_soh.{last_index}",
                )
            ),
            is_short=True,
        ),
    ]
    evidence = _EVIDENCE_PRESENTATION.get(
        authorization.evidence_level.value,
        "受审计证据",
    )
    elements: list[dict[str, Any]] = []
    if image_key is not None:
        elements.append(
            {
                "tag": "img",
                "img_key": _identifier(image_key, field_name="image_key"),
                "alt": {"tag": "plain_text", "content": "有限时域 SOH 轨迹"},
                "mode": "fit_horizontal",
                "preview": True,
            }
        )
    elements.extend([
        {"tag": "div", "fields": fields},
        _paragraph(
            "**结果说明**\n"
            f"证据类型: {evidence}\n"
            "该轨迹仅覆盖至第 500 个循环, 不是 15 至 25 年自然年寿命推演。"
        ),
        _paragraph(
            "**不确定性边界**\n"
            "当前未提供统计区间; 如需区间, 必须使用已登记的校准 ToolResult。"
        ),
    ])
    elements.extend(_warning_elements(result.warnings))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "green",
            "title": {
                "tag": "plain_text",
                "content": _bounded_text(
                    f"{cell_id} | 有限时域 SOH",
                    field_name="card title",
                ),
            },
        },
        "elements": elements,
    }


def _cell_metadata_fields(
    result: ToolResult,
    *,
    expected_cell_id: str,
    legacy_artifact_type: str,
    metadata_artifact_type: str,
) -> list[dict[str, Any]]:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AuditedCardError("analysis artifact is invalid")
    try:
        metadata = validate_versioned_cell_metadata_evidence(
            artifact,
            artifact_type=result.values.get("artifact_type"),
            legacy_artifact_type=legacy_artifact_type,
            metadata_artifact_type=metadata_artifact_type,
        )
    except CellMetadataEvidenceIdentityError as exc:
        raise AuditedCardError(
            "cell metadata identity does not match the result"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise AuditedCardError("cell metadata does not satisfy its contract") from exc
    if metadata is None:
        return [
            _display_field("化学体系", "未登记", is_short=True),
            _display_field("标称容量", "未登记", is_short=True),
            _display_field("数据来源", "未登记", is_short=False),
            _display_field("测试规格", "未登记", is_short=False),
        ]
    if metadata.cell_id != expected_cell_id:
        raise AuditedCardError("cell metadata identity does not match the result")
    specification = (
        metadata.protocol_description
        or metadata.protocol_id
        or metadata.schema_version
    )
    return [
        _display_field("化学体系", metadata.chemistry, is_short=True),
        _display_field(
            "标称容量",
            f"{_format_number(metadata.nominal_capacity_ah)} Ah",
            is_short=True,
        ),
        _display_field("数据来源", metadata.dataset_id, is_short=False),
        _display_field("测试规格", specification, is_short=False),
    ]


def _resolve_scalar(
    mapping: Mapping[str, Any], json_path: str
) -> str | bool | int | float:
    value: Any = mapping
    for segment in json_path.split("."):
        if isinstance(value, Mapping) and segment in value:
            value = value[segment]
            continue
        if (
            isinstance(value, list)
            and segment.isascii()
            and segment.isdecimal()
            and (segment == "0" or not segment.startswith("0"))
        ):
            index = int(segment)
            if index < len(value):
                value = value[index]
                continue
        raise AuditedCardError(f"allowlisted ToolResult path is missing: {json_path}")
    if isinstance(value, bool | str):
        return value
    if isinstance(value, Real):
        return float(value)
    raise AuditedCardError("allowlisted ToolResult path is not a scalar")


def _build_scenario_result_card(
    *,
    run_id: str,
    result: ToolResult,
    authorization: AuditedResultAuthorization,
    image_key: str | None,
) -> dict[str, Any]:
    mapping = result.model_dump(mode="json")
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("status") != "COMPLETED":
        raise AuditedCardError("scenario result is not completed")
    projections = _scenario_projection_paths(result)
    elements: list[dict[str, Any]] = []
    if image_key is not None:
        checked_image_key = _identifier(image_key, field_name="image_key")
        elements.append(
            {
                "tag": "img",
                "img_key": checked_image_key,
                "alt": {"tag": "plain_text", "content": "储能工况年份推演曲线"},
                "mode": "fit_horizontal",
                "preview": True,
            }
        )
    for index, (prefix, projection) in enumerate(projections):
        scenario_id = _resolve_scalar(mapping, f"{prefix}.scenario_id")
        support_status = _resolve_scalar(mapping, f"{prefix}.support.status")
        eol_status = _resolve_scalar(mapping, f"{prefix}.eol.status")
        if (
            not isinstance(scenario_id, str)
            or not isinstance(support_status, str)
            or not isinstance(eol_status, str)
        ):
            raise AuditedCardError("scenario card metadata is invalid")
        section_title = "基准工况" if index == 0 else f"对比工况 {index}"
        elements.append(
            _paragraph(
                f"**{section_title} · {_escape_lark_markdown(scenario_id)}**"
            )
        )
        if result.feature_version == SCENARIO_FEATURE_VERSION:
            elements.append(
                _paragraph(
                    "**工况假设**\n"
                    + "\n".join(
                        _scenario_segment_lines(
                            mapping=mapping,
                            prefix=prefix,
                            projection=projection,
                        )
                    )
                )
            )
        fields = [
            _display_field(
                "支持状态",
                _SUPPORT_PRESENTATION.get(support_status, "需要复核"),
                is_short=True,
            ),
            _display_field(
                "阈值状态",
                _EOL_PRESENTATION.get(eol_status, "需要复核"),
                is_short=True,
            ),
            _display_field(
                "推演终点",
                _format_year(
                    _resolve_scalar(mapping, f"{prefix}.final_natural_year")
                ),
                is_short=True,
            ),
            _display_field(
                "累计计划 EFC",
                _format_efc(
                    _resolve_scalar(
                        mapping,
                        f"{prefix}.final_equivalent_full_cycles",
                    )
                ),
                is_short=True,
            ),
            _display_field(
                "期末 SOH (比例)",
                _format_scalar(_resolve_scalar(mapping, f"{prefix}.final_soh")),
                is_short=True,
            ),
            _display_field(
                "EOL 阈值",
                _format_scalar(
                    _resolve_scalar(mapping, f"{prefix}.eol_threshold")
                ),
                is_short=True,
            ),
        ]
        milestones = projection.get("milestone_soh")
        if not isinstance(milestones, Mapping):
            raise AuditedCardError("scenario milestones are invalid")
        for year in ("15", "20", "25"):
            if year in milestones:
                path = f"{prefix}.milestone_soh.{year}"
                fields.append(
                    _display_field(
                        f"{year} 年 SOH",
                        _format_scalar(_resolve_scalar(mapping, path)),
                        is_short=True,
                    )
                )
        if eol_status == "REACHED":
            fields.extend(
                (
                    _display_field(
                        "首次达到阈值年份",
                        _format_year(
                            _resolve_scalar(mapping, f"{prefix}.eol.natural_year")
                        ),
                        is_short=True,
                    ),
                    _display_field(
                        "达到阈值时累计 EFC",
                        _format_efc(
                            _resolve_scalar(
                                mapping,
                                f"{prefix}.eol.equivalent_full_cycles",
                            )
                        ),
                        is_short=True,
                    ),
                )
            )
        elements.append({"tag": "div", "fields": fields})
    evidence = _EVIDENCE_PRESENTATION.get(
        authorization.evidence_level.value,
        "受审计证据",
    )
    elements.append(
        _paragraph(
            "**证据说明**\n"
            f"证据类型: {evidence}\n"
            "本卡使用独立的 BLAST-Lite LFP/石墨参考情景。"
            "它不是 MATR 电芯的自然年换算, 也不代表海辰现场寿命承诺。"
        )
    )
    elements.extend(_warning_elements(result.warnings))
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": "orange",
            "title": {
                "tag": "plain_text",
                "content": "储能工况年份推演",
            },
        },
        "elements": elements,
    }


def _scenario_segment_lines(
    *,
    mapping: Mapping[str, Any],
    prefix: str,
    projection: Mapping[str, Any],
) -> list[str]:
    segments = projection.get("operating_segments")
    if not isinstance(segments, list) or not segments or len(segments) > 25:
        raise AuditedCardError("scenario operating segments are invalid")
    lines: list[str] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            raise AuditedCardError("scenario operating segment is invalid")
        segment_prefix = f"{prefix}.operating_segments.{index}"
        start_year = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.start_year")
        )
        end_year = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.end_year")
        )
        temperature_c = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.temperature_c")
        )
        charge_c_rate = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.charge_c_rate")
        )
        discharge_c_rate = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.discharge_c_rate")
        )
        soc_lower = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.soc_lower_bound")
        )
        soc_upper = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.soc_upper_bound")
        )
        dod = _numeric_scalar(_resolve_scalar(mapping, f"{segment_prefix}.dod"))
        annual_efc = _numeric_scalar(
            _resolve_scalar(
                mapping,
                f"{segment_prefix}.equivalent_full_cycles_per_year",
            )
        )
        rest_hours = _numeric_scalar(
            _resolve_scalar(mapping, f"{segment_prefix}.rest_duration_hours")
        )
        stage = (
            ""
            if len(segments) == 1
            else f"{_format_number(start_year)}-{_format_number(end_year)} 年: "
        )
        lines.append(
            stage
            + f"{_format_number(temperature_c)}°C | "
            + f"充电 {_format_number(charge_c_rate)}C / "
            + f"放电 {_format_number(discharge_c_rate)}C | "
            + f"SOC {_format_percent(soc_lower)}-{_format_percent(soc_upper)} "
            + f"(DoD {_format_percent(dod)}) | "
            + f"{_format_number(annual_efc)} EFC/年 | "
            + f"静置 {_format_number(rest_hours)} 小时"
        )
    return lines


def _scenario_projection_paths(
    result: ToolResult,
) -> list[tuple[str, Mapping[str, Any]]]:
    artifact = result.values.get("artifact")
    if not isinstance(artifact, Mapping):
        raise AuditedCardError("scenario artifact is invalid")
    if result.tool_name == "project_storage_lifetime":
        projection = artifact.get("projection")
        if not isinstance(projection, Mapping):
            raise AuditedCardError("scenario projection is invalid")
        return [("values.artifact.projection", projection)]
    baseline = artifact.get("baseline")
    comparisons = artifact.get("comparisons")
    if not isinstance(baseline, Mapping) or not isinstance(comparisons, list):
        raise AuditedCardError("scenario comparison projections are invalid")
    projections: list[tuple[str, Mapping[str, Any]]] = [
        ("values.artifact.baseline", baseline)
    ]
    for index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise AuditedCardError("scenario comparison projection is invalid")
        projections.append((f"values.artifact.comparisons.{index}", comparison))
    return projections


def _bounded_text(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 500 or any(ord(char) < 32 for char in normalized):
        raise AuditedCardError(f"{field_name} is not safe card text")
    return normalized


def _escape_lark_markdown(value: str) -> str:
    normalized = _bounded_text(value, field_name="display value")
    for token in ("\\", "`", "*", "[", "]"):
        normalized = normalized.replace(token, f"\\{token}")
    return normalized


def _format_scalar(value: str | bool | int | float) -> str:
    if isinstance(value, str):
        return _bounded_text(value, field_name="display value")
    if isinstance(value, bool):
        return "是" if value else "否"
    return _format_number(value)


def _format_number(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise AuditedCardError("display value must be finite")
    if math.isclose(number, round(number), rel_tol=0.0, abs_tol=1e-9):
        return f"{round(number):,}"
    return f"{number:,.2f}".rstrip("0").rstrip(".")


def _numeric_scalar(value: str | bool | int | float) -> int | float:
    if isinstance(value, str | bool):
        raise AuditedCardError("scenario operating segment values must be numeric")
    return value


def _format_count(value: str | bool | int | float) -> str:
    if isinstance(value, str | bool):
        raise AuditedCardError("cycle count must be numeric")
    return f"{_format_number(value)} 次"


def _format_year(value: str | bool | int | float) -> str:
    if isinstance(value, str | bool):
        raise AuditedCardError("natural year must be numeric")
    return f"{_format_number(value)} 年"


def _format_efc(value: str | bool | int | float) -> str:
    if isinstance(value, str | bool):
        raise AuditedCardError("equivalent full cycles must be numeric")
    return f"{_format_number(value)} EFC"


def _format_percent(value: int | float) -> str:
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise AuditedCardError("percentage display value must be between zero and one")
    return f"{_format_number(number * 100.0)}%"


def _warning_elements(warnings: list[str]) -> list[dict[str, Any]]:
    if not warnings:
        return []
    messages = list(
        dict.fromkeys(
            _WARNING_PRESENTATION.get(
                warning,
                "结果包含需要复核的限制, 详情见受审计报告。",
            )
            for warning in warnings
        )
    )
    return [
        _paragraph("**注意事项**\n" + "\n".join(f"- {message}" for message in messages))
    ]


def _identifier(value: str, *, field_name: str) -> str:
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must not be blank")
    if _SAFE_REFERENCE.fullmatch(checked) is None:
        raise ValueError(f"{field_name} must be a safe machine identifier")
    return checked
