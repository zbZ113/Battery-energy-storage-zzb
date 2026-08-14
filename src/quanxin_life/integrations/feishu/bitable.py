"""Idempotent Feishu Bitable summaries keyed only by a safe ``run_id``."""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from hmac import compare_digest
from types import MappingProxyType
from typing import Protocol

_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

BITABLE_RUN_FIELD_NAMES = frozenset(
    {
        "run_id",
        "task_type",
        "task_status",
        "data_batch_id",
        "input_file_sha256",
        "cell_reference",
        "scenario_context_id",
        "scenario_id",
        "scenario_version",
        "primary_result_id",
        "model_route",
        "model_version",
        "data_version",
        "feature_version",
        "evidence_level",
        "warnings",
        "report_link",
        "analysis_summary",
        "applicability",
        "cell_name",
        "chemistry",
        "nominal_capacity_ah",
        "data_source",
        "test_specification",
        "observed_cycle_count",
        "predicted_total_cycles",
        "predicted_remaining_cycles",
        "soh_horizon_cycle",
        "soh_horizon_value",
        "recommendation",
        "recommendation_reason",
        "recommendation_ruleset_version",
        "curve_attachment",
        "curve_source_result_id",
        "curve_renderer_version",
        "curve_sha256",
        "curve_template",
        "created_at_utc",
        "updated_at_utc",
    }
)
_REQUIRED_FIELDS = frozenset(
    {"run_id", "task_type", "task_status", "created_at_utc", "updated_at_utc"}
)
_TIMESTAMP_FIELDS = frozenset({"created_at_utc", "updated_at_utc"})
_SHA256_FIELDS = frozenset({"input_file_sha256", "curve_sha256"})
_CURVE_EVIDENCE_FIELDS = frozenset(
    {
        "curve_attachment",
        "curve_source_result_id",
        "curve_renderer_version",
        "curve_sha256",
        "curve_template",
    }
)
_MAX_TEXT_LENGTH = 2_000
_MAX_FIELD_NAME_LENGTH = 100
_MAX_BITABLE_MEDIA_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BitableAttachment:
    """One already-uploaded attachment scoped to the configured Bitable app."""

    file_token: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "file_token",
            _safe_identifier(self.file_token, field_name="file_token"),
        )


@dataclass(frozen=True, slots=True)
class BitableFieldProfile:
    """Versioned mapping from internal evidence keys to operator-facing columns."""

    profile_id: str
    profile_version: str
    field_names: Mapping[str, str]
    value_labels: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_id",
            _safe_identifier(self.profile_id, field_name="profile_id"),
        )
        object.__setattr__(
            self,
            "profile_version",
            _safe_identifier(self.profile_version, field_name="profile_version"),
        )
        if not isinstance(self.field_names, Mapping) or set(self.field_names) != set(
            BITABLE_RUN_FIELD_NAMES
        ):
            raise BitableValidationError(
                "Bitable field profile must map every allowlisted field"
            )
        normalized_names = {
            key: _safe_field_name(value, field_name=f"field_names.{key}")
            for key, value in self.field_names.items()
        }
        if len(set(normalized_names.values())) != len(normalized_names):
            raise BitableValidationError("Bitable field profile names must be unique")
        normalized_labels: dict[str, Mapping[str, str]] = {}
        if not isinstance(self.value_labels, Mapping):
            raise BitableValidationError("Bitable value labels must be a mapping")
        for field_name, labels in self.value_labels.items():
            if field_name not in BITABLE_RUN_FIELD_NAMES or not isinstance(
                labels, Mapping
            ):
                raise BitableValidationError("Bitable value labels are invalid")
            normalized_labels[field_name] = MappingProxyType(
                {
                    _bounded_text(key, field_name=f"value_labels.{field_name}.key"):
                    _bounded_text(value, field_name=f"value_labels.{field_name}.value")
                    for key, value in labels.items()
                }
            )
        object.__setattr__(self, "field_names", MappingProxyType(normalized_names))
        object.__setattr__(self, "value_labels", MappingProxyType(normalized_labels))

    def remote_fields(self, fields: Mapping[str, object]) -> dict[str, object]:
        mapped: dict[str, object] = {}
        for field_name, value in fields.items():
            labels = self.value_labels.get(field_name)
            if labels is not None and isinstance(value, str):
                try:
                    value = labels[value]
                except KeyError as exc:
                    raise BitableValidationError(
                        f"{field_name} has no reviewed display label"
                    ) from exc
            mapped[self.field_names[field_name]] = value
        return mapped


class BitableWriterError(RuntimeError):
    """Base failure for deterministic Bitable summary persistence."""


class BitableValidationError(BitableWriterError):
    """Raised before network access when one summary is unsafe."""


class BitableConflictError(BitableWriterError):
    """Raised when ``run_id`` no longer identifies one remote record."""


class BitableProtocolError(BitableWriterError):
    """Raised when Feishu returns an unusable response shape."""


class BitableWriteAction(StrEnum):
    CREATED = "CREATED"
    UPDATED = "UPDATED"


@dataclass(frozen=True, slots=True)
class BitableWriteResult:
    run_id: str
    record_id: str
    action: BitableWriteAction


class FeishuBitableClient(Protocol):
    def search_bitable_records(
        self,
        *,
        app_token: str,
        table_id: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, object]: ...

    def create_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]: ...

    def update_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        record_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]: ...


class BitableMediaClient(Protocol):
    def upload_bitable_media(
        self,
        *,
        app_token: str,
        filename: str,
        content_type: str,
        payload: bytes,
    ) -> dict[str, object]: ...


class BitableMediaUploader:
    """Upload one verified renderer image into a Bitable attachment scope."""

    def __init__(self, client: BitableMediaClient, *, app_token: str) -> None:
        if not callable(getattr(client, "upload_bitable_media", None)):
            raise TypeError("client must support Bitable media uploads")
        self._client = client
        self._app_token = _safe_identifier(app_token, field_name="app_token")

    def upload_image(
        self,
        *,
        filename: str,
        content_type: str,
        payload: bytes,
        expected_sha256: str,
    ) -> BitableAttachment:
        if not isinstance(payload, bytes) or not 0 < len(payload) <= _MAX_BITABLE_MEDIA_BYTES:
            raise BitableValidationError("Bitable image payload size is invalid")
        if content_type != "image/png":
            raise BitableValidationError("Bitable analysis image must be PNG")
        checked_filename = _safe_png_filename(filename)
        if _SHA256.fullmatch(expected_sha256) is None or not compare_digest(
            sha256(payload).hexdigest(), expected_sha256
        ):
            raise BitableValidationError("Bitable image SHA-256 does not match")
        response = self._client.upload_bitable_media(
            app_token=self._app_token,
            filename=checked_filename,
            content_type=content_type,
            payload=payload,
        )
        if not isinstance(response, Mapping):
            raise BitableProtocolError("Bitable media response is invalid")
        file_token = response.get("file_token")
        if not isinstance(file_token, str):
            raise BitableProtocolError(
                "Bitable media response has no safe file_token"
            )
        try:
            return BitableAttachment(file_token=file_token)
        except (TypeError, ValueError, BitableValidationError) as exc:
            raise BitableProtocolError(
                "Bitable media response has no safe file_token"
            ) from exc

class FeishuBitableWriter:
    """Create or update exactly one scalar-only Bitable row per ``run_id``."""

    def __init__(
        self,
        client: FeishuBitableClient,
        *,
        app_token: str,
        table_id: str,
        field_profile: BitableFieldProfile | None = None,
    ) -> None:
        if not callable(getattr(client, "search_bitable_records", None)):
            raise TypeError("client must support Bitable record search")
        if not callable(getattr(client, "create_bitable_record", None)):
            raise TypeError("client must support Bitable record creation")
        if not callable(getattr(client, "update_bitable_record", None)):
            raise TypeError("client must support Bitable record updates")
        self._client = client
        self._app_token = _safe_identifier(app_token, field_name="app_token")
        self._table_id = _safe_identifier(table_id, field_name="table_id")
        self._field_profile = field_profile or IDENTITY_BITABLE_FIELD_PROFILE
        if not isinstance(self._field_profile, BitableFieldProfile):
            raise TypeError("field_profile must be a BitableFieldProfile")
        self._lock = threading.RLock()

    def upsert(self, fields: Mapping[str, object]) -> BitableWriteResult:
        """Persist one complete summary without accepting arrays or nested objects."""

        normalized = _normalize_fields(fields)
        remote_fields = self._field_profile.remote_fields(normalized)
        run_id = normalized["run_id"]
        assert isinstance(run_id, str)
        with self._lock:
            search = self._client.search_bitable_records(
                app_token=self._app_token,
                table_id=self._table_id,
                field_name=self._field_profile.field_names["run_id"],
                field_value=run_id,
            )
            items = _search_items(search)
            if len(items) > 1:
                raise BitableConflictError(
                    "run_id matched multiple records; refusing a concurrent overwrite"
                )
            if not items:
                response = self._client.create_bitable_record(
                    app_token=self._app_token,
                    table_id=self._table_id,
                    fields=remote_fields,
                )
                action = BitableWriteAction.CREATED
            else:
                record_id = _record_id(items[0], response_label="search response")
                response = self._client.update_bitable_record(
                    app_token=self._app_token,
                    table_id=self._table_id,
                    record_id=record_id,
                    fields=remote_fields,
                )
                action = BitableWriteAction.UPDATED
            return BitableWriteResult(
                run_id=run_id,
                record_id=_record_response_id(response),
                action=action,
            )


def _normalize_fields(fields: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(fields, Mapping):
        raise BitableValidationError("Bitable fields must be a mapping")
    unknown = set(fields) - BITABLE_RUN_FIELD_NAMES
    if unknown:
        raise BitableValidationError("Bitable field is not allowlisted")
    missing = _REQUIRED_FIELDS - set(fields)
    if missing:
        raise BitableValidationError("Bitable summary is missing required fields")

    normalized: dict[str, object] = {}
    for name, value in fields.items():
        if name in _TIMESTAMP_FIELDS:
            normalized[name] = _utc_timestamp(value, field_name=name)
        elif value is None:
            normalized[name] = None
        elif name == "curve_attachment":
            if not isinstance(value, BitableAttachment):
                raise BitableValidationError(
                    "curve_attachment must be an audited Bitable attachment"
                )
            normalized[name] = [{"file_token": value.file_token}]
        elif not isinstance(value, str):
            raise BitableValidationError(
                f"{name} must be a scalar string, datetime, or null"
            )
        else:
            normalized[name] = _bounded_text(value, field_name=name)

    run_id = normalized["run_id"]
    assert isinstance(run_id, str)
    normalized["run_id"] = _safe_identifier(run_id, field_name="run_id")
    for field_name in _SHA256_FIELDS:
        digest = normalized.get(field_name)
        if digest is not None and (
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        ):
            raise BitableValidationError(f"{field_name} must be lowercase SHA-256")
    curve_evidence = _CURVE_EVIDENCE_FIELDS.intersection(normalized)
    if curve_evidence and (
        curve_evidence != _CURVE_EVIDENCE_FIELDS
        or any(normalized[field_name] is None for field_name in curve_evidence)
    ):
        raise BitableValidationError("Bitable curve evidence is incomplete")
    if curve_evidence and (
        normalized.get("primary_result_id")
        != normalized.get("curve_source_result_id")
    ):
        raise BitableValidationError(
            "Bitable curve source must match the primary result"
        )
    report_link = normalized.get("report_link")
    if report_link is not None and (
        not isinstance(report_link, str) or not report_link.startswith("https://")
    ):
        raise BitableValidationError("report_link must use HTTPS")
    created = normalized["created_at_utc"]
    updated = normalized["updated_at_utc"]
    assert isinstance(created, str)
    assert isinstance(updated, str)
    if created > updated:
        raise BitableValidationError("created_at_utc must not follow updated_at_utc")
    return normalized


def _utc_timestamp(value: object, *, field_name: str) -> str:
    if not isinstance(value, datetime):
        raise BitableValidationError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise BitableValidationError(f"{field_name} must include a timezone")
    return value.astimezone(UTC).isoformat()


def _bounded_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if (
        not normalized
        or len(normalized) > _MAX_TEXT_LENGTH
        or any(ord(character) < 32 and character not in "\n\t" for character in normalized)
    ):
        raise BitableValidationError(f"{field_name} must be bounded safe text")
    return normalized


def _safe_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_IDENTIFIER.fullmatch(normalized) is None:
        raise BitableValidationError(f"{field_name} must be a safe identifier")
    return normalized


def _safe_field_name(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > _MAX_FIELD_NAME_LENGTH
        or any(ord(character) < 32 for character in normalized)
    ):
        raise BitableValidationError(f"{field_name} must be a safe Bitable field name")
    return normalized


def _safe_png_filename(value: object) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if (
        not normalized
        or len(normalized) > 250
        or not normalized.lower().endswith(".png")
        or any(character in normalized for character in ("/", "\\", "\0"))
        or any(ord(character) < 32 for character in normalized)
    ):
        raise BitableValidationError("Bitable image filename is invalid")
    return normalized


_IDENTITY_FIELD_NAMES = {name: name for name in BITABLE_RUN_FIELD_NAMES}
IDENTITY_BITABLE_FIELD_PROFILE = BitableFieldProfile(
    profile_id="bitable-fields-identity",
    profile_version="v1",
    field_names=_IDENTITY_FIELD_NAMES,
    value_labels={},
)

_CHINESE_FIELD_NAMES = {
    "run_id": "任务ID",
    "task_type": "分析类型",
    "task_status": "任务状态",
    "data_batch_id": "数据批次ID",
    "input_file_sha256": "输入文件SHA256",
    "cell_reference": "电芯引用",
    "scenario_context_id": "工况上下文ID",
    "scenario_id": "工况ID",
    "scenario_version": "工况版本",
    "primary_result_id": "结果ID",
    "model_route": "模型路线",
    "model_version": "模型版本",
    "data_version": "数据版本",
    "feature_version": "特征版本",
    "evidence_level": "证据类型",
    "warnings": "技术警告代码",
    "report_link": "详细报告",
    "analysis_summary": "分析摘要",
    "applicability": "适用边界",
    "cell_name": "电芯名称",
    "chemistry": "化学体系",
    "nominal_capacity_ah": "标称容量",
    "data_source": "数据来源",
    "test_specification": "测试规格",
    "observed_cycle_count": "已观测循环数",
    "predicted_total_cycles": "预计总循环寿命",
    "predicted_remaining_cycles": "预计剩余循环寿命",
    "soh_horizon_cycle": "SOH预测边界循环",
    "soh_horizon_value": "边界循环SOH",
    "recommendation": "综合建议",
    "recommendation_reason": "建议原因",
    "recommendation_ruleset_version": "规则集版本",
    "curve_attachment": "分析曲线",
    "curve_source_result_id": "曲线来源结果ID",
    "curve_renderer_version": "曲线渲染器版本",
    "curve_sha256": "曲线SHA256",
    "curve_template": "曲线模板",
    "created_at_utc": "创建时间UTC",
    "updated_at_utc": "更新时间UTC",
}
CHINESE_ANALYSIS_BITABLE_PROFILE = BitableFieldProfile(
    profile_id="quanxin-analysis-zh-cn",
    profile_version="v2",
    field_names=_CHINESE_FIELD_NAMES,
    value_labels={
        "task_type": {
            "predict_cycle_life": "个体早期循环寿命预测",
            "predict_soh_trajectory": "有限循环 SOH 轨迹预测",
            "compare_operation_scenarios": "参考工况对比",
            "project_storage_lifetime": "项目储能寿命参考推演",
            "ingest_observed_soh": "登记实测 SOH",
            "update_trajectory": "更新退化轨迹",
            "make_engineering_recommendation": "工程综合建议",
        },
        "task_status": {
            "PENDING": "待处理",
            "QUEUED": "已入队",
            "RUNNING": "分析中",
            "RETRYABLE": "等待重试",
            "SUCCEEDED": "已完成",
            "COMPLETED": "已完成",
            "REJECTED": "已拒绝",
            "FAILED": "失败",
        },
        "evidence_level": {
            "DATA_DIRECT": "直接数据证据",
            "MODEL_INFERENCE": "模型推理",
            "PHYSICS_REFERENCE": "物理参考推演",
            "DOMAIN_KNOWLEDGE": "领域知识",
            "UNDETERMINED": "尚未确定",
        },
        "curve_template": {
            "CYCLE_LIFE_SUMMARY": "早期寿命概览",
            "FINITE_SOH_CURVE": "SOH退化轨迹",
            "SCENARIO_COMPARISON": "工况退化对比",
        },
    },
)


def _search_items(response: object) -> list[object]:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable search response is invalid")
    if response.get("has_more") is True:
        raise BitableConflictError(
            "Bitable search has additional records; uniqueness is not proven"
        )
    items = response.get("items")
    if not isinstance(items, list):
        raise BitableProtocolError("Bitable search response has no item list")
    if any(not isinstance(item, Mapping) for item in items):
        raise BitableProtocolError("Bitable search response contains an invalid record")
    return items


def _record_response_id(response: object) -> str:
    if not isinstance(response, Mapping):
        raise BitableProtocolError("Bitable record response is invalid")
    record = response.get("record")
    return _record_id(record, response_label="record response")


def _record_id(value: object, *, response_label: str) -> str:
    if not isinstance(value, Mapping):
        raise BitableProtocolError(f"Bitable {response_label} is invalid")
    record_id = value.get("record_id")
    if not isinstance(record_id, str) or _SAFE_IDENTIFIER.fullmatch(record_id) is None:
        raise BitableProtocolError(f"Bitable {response_label} has no safe record_id")
    return record_id


__all__ = [
    "BITABLE_RUN_FIELD_NAMES",
    "CHINESE_ANALYSIS_BITABLE_PROFILE",
    "IDENTITY_BITABLE_FIELD_PROFILE",
    "BitableAttachment",
    "BitableConflictError",
    "BitableFieldProfile",
    "BitableMediaClient",
    "BitableMediaUploader",
    "BitableProtocolError",
    "BitableValidationError",
    "BitableWriteAction",
    "BitableWriteResult",
    "BitableWriterError",
    "FeishuBitableWriter",
]
