"""Pure, non-computing presenters for server-signed evidence payloads."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

UNAVAILABLE_LABEL = "后端未提供"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class EvidenceField:
    """One backend field and its exact JSON path."""

    key: str
    label: str
    json_path: str
    value: object | None
    available: bool

    def display_value(self) -> object:
        return self.value if self.available else UNAVAILABLE_LABEL


@dataclass(frozen=True, slots=True)
class EvidenceSection:
    """Display-only section; it performs no engineering derivation."""

    key: str
    title: str
    fields: tuple[EvidenceField, ...]

    def field(self, key: str) -> EvidenceField:
        for item in self.fields:
            if item.key == key:
                return item
        raise KeyError(key)


def build_quality_section(payload: Mapping[str, object]) -> EvidenceSection:
    return _build_section(
        payload,
        key="quality",
        title="数据质量",
        specs=(
            ("dataset_id", "数据集 ID", ("values", "dataset_id")),
            ("blocked", "是否阻断", ("values", "blocked")),
            ("quality_score", "质量评分", ("values", "quality_score")),
            ("issue_count", "问题数量", ("values", "issue_count")),
            ("issues", "问题明细", ("values", "issues")),
        ),
    )


def build_lifetime_section(
    prediction_payload: Mapping[str, object],
    interval_payload: Mapping[str, object],
) -> EvidenceSection:
    prediction_specs = (
        (
            "predicted_eol_cycle",
            "EOL80 点预测周期",
            ("values", "artifact", "life_prediction", "predicted_eol_cycle"),
        ),
        (
            "derived_rul_cycle",
            "服务端派生 RUL 周期",
            ("values", "artifact", "derived_rul_cycle"),
        ),
    )
    interval_specs = (
        (
            "lower_eol_cycle",
            "区间下界周期",
            ("values", "artifact", "prediction_interval", "lower_eol_cycle"),
        ),
        (
            "upper_eol_cycle",
            "区间上界周期",
            ("values", "artifact", "prediction_interval", "upper_eol_cycle"),
        ),
        (
            "coverage_target",
            "目标覆盖率",
            ("uncertainty", "coverage_target"),
        ),
    )
    fields = (
        *_fields_from_specs(prediction_payload, prediction_specs),
        *_fields_from_specs(interval_payload, interval_specs),
    )
    return EvidenceSection(key="lifetime", title="寿命点预测与区间", fields=fields)


def build_decision_section(payload: Mapping[str, object]) -> EvidenceSection:
    return _build_section(
        payload,
        key="decision",
        title="批次决策",
        specs=(
            ("decision", "决策", ("values", "decision")),
            ("required_eol_cycle", "策略要求周期", ("values", "required_eol_cycle")),
            ("reason_codes", "原因代码", ("values", "reason_codes")),
            (
                "target_domain_calibrated",
                "目标域已校准",
                ("values", "target_domain_calibrated"),
            ),
            ("policy_id", "策略 ID", ("values", "policy_id")),
            ("policy_version", "策略版本", ("values", "policy_version")),
        ),
    )


def build_audit_section(payload: Mapping[str, object]) -> EvidenceSection:
    return _build_section(
        payload,
        key="audit",
        title="审计证据",
        specs=(
            ("result_id", "结果 ID", ("result_id",)),
            ("tool_name", "工具名称", ("tool_name",)),
            ("tool_version", "工具版本", ("tool_version",)),
            ("model_version", "模型版本", ("model_version",)),
            ("data_version", "数据版本", ("data_version",)),
            ("feature_version", "特征版本", ("feature_version",)),
            ("input_hash", "输入哈希", ("input_hash",)),
            ("created_at", "签发时间", ("created_at",)),
            ("warnings", "服务端警告", ("warnings",)),
            ("provenance", "来源记录", ("provenance",)),
        ),
    )


FieldSpec = tuple[str, str, tuple[str, ...]]


def _build_section(
    payload: Mapping[str, object],
    *,
    key: str,
    title: str,
    specs: tuple[FieldSpec, ...],
) -> EvidenceSection:
    return EvidenceSection(
        key=key,
        title=title,
        fields=_fields_from_specs(payload, specs),
    )


def _fields_from_specs(
    payload: Mapping[str, object], specs: tuple[FieldSpec, ...]
) -> tuple[EvidenceField, ...]:
    fields: list[EvidenceField] = []
    for key, label, path in specs:
        raw_value = _read_path(payload, path)
        available = raw_value is not _MISSING
        fields.append(
            EvidenceField(
                key=key,
                label=label,
                json_path=".".join(path),
                value=deepcopy(raw_value) if available else None,
                available=available,
            )
        )
    return tuple(fields)


def _read_path(payload: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = payload
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            return _MISSING
        current = current[component]
    return current
