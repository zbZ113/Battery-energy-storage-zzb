"""Deterministic Feishu cards built from traceable audit-ledger references."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any, Protocol

from quanxin_life.core import EvidenceLevel, ToolResult
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
        _CardFieldPolicy("Data blocked", "values.blocked"),
        _CardFieldPolicy("Quality score", "values.quality_score"),
        _CardFieldPolicy("Issue count", "values.issue_count"),
    ),
    (
        "predict_cycle_life",
        "advanced-rul-prediction-tool-v1",
    ): (
        _CardFieldPolicy(
            "Predicted cycle",
            "values.artifact.cycle_life_prediction.predicted_cycle",
        ),
        _CardFieldPolicy(
            "Remaining cycles", "values.artifact.derived_remaining_cycles"
        ),
    ),
    (
        "predict_soh_trajectory",
        "advanced-soh-prediction-tool-v1",
    ): (
        _CardFieldPolicy("Prediction boundary", "values.artifact.horizon_end_cycle"),
    ),
}


_STATUS_PRESENTATION = {
    FeishuCardStatus.RECEIVED: ("Received", "blue"),
    FeishuCardStatus.DATA_CHECK: ("Data check", "blue"),
    FeishuCardStatus.QUEUED: ("Queued", "blue"),
    FeishuCardStatus.RUNNING: ("Running", "blue"),
    FeishuCardStatus.SUCCESS: ("Completed", "green"),
    FeishuCardStatus.REJECTED: ("Rejected", "red"),
    FeishuCardStatus.DEGRADED: ("Degraded", "orange"),
    FeishuCardStatus.REPORT_READY: ("Report ready", "green"),
}


def build_status_card(
    *,
    status: FeishuCardStatus,
    run_id: str,
    result_id: str | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    checked_run_id = _identifier(run_id, field_name="run_id")
    title, template = _STATUS_PRESENTATION[status]
    fields = [_reference_field("run_id", checked_run_id)]
    if result_id is not None:
        fields.append(
            _reference_field(
                "result_id", _identifier(result_id, field_name="result_id")
            )
        )
    if reason_code is not None:
        fields.append(
            _reference_field(
                "reason", _identifier(reason_code, field_name="reason_code")
            )
        )
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": f"Quanxin Life - {title}"},
        },
        "elements": [{"tag": "div", "fields": fields}],
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

    def build_result_card(self, *, run_id: str, result_id: str) -> dict[str, Any]:
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
            )
        policy = _CARD_POLICIES.get((result.tool_name, result.tool_version))
        if policy is None:
            raise AuditedCardError("ToolResult version has no allowlisted card paths")
        result_mapping = result.model_dump(mode="json")
        value_fields = [
            _audited_value_field(
                item.label,
                _resolve_scalar(result_mapping, item.json_path),
                result_id=result.result_id,
                json_path=item.json_path,
            )
            for item in policy
        ]
        metadata = (
            ("route_id", authorization.route_id),
            ("activation_status", authorization.activation_status),
            ("model_version", result.model_version or ""),
            ("data_version", result.data_version or ""),
            ("feature_version", result.feature_version or ""),
            ("evidence_level", authorization.evidence_level.value),
            ("supported_domain", authorization.supported_domain),
        )
        metadata_fields = [
            _reference_field(label, _bounded_text(value, field_name=label))
            for label, value in metadata
        ]
        warning_fields = [
            _reference_field(
                "warning", _bounded_text(warning, field_name="warning")
            )
            for warning in result.warnings
        ]
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "template": "green",
                "title": {"tag": "plain_text", "content": "Quanxin Life - Audited result"},
            },
            "elements": [
                {
                    "tag": "div",
                    "fields": [
                        _reference_field("run_id", checked_run_id),
                        _reference_field("result_id", result.result_id),
                        *value_fields,
                        *metadata_fields,
                        *warning_fields,
                    ],
                }
            ],
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


def _reference_field(label: str, value: str) -> dict[str, Any]:
    return {
        "is_short": False,
        "text": {"tag": "lark_md", "content": f"**{label}**\n{value}"},
    }


def _audited_value_field(
    label: str,
    value: str | bool | int | float,
    *,
    result_id: str,
    json_path: str,
) -> dict[str, Any]:
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    return {
        "is_short": False,
        "text": {
            "tag": "lark_md",
            "content": (
                f"**{label}**\n{rendered}\n"
                f"`{result_id}#{json_path}`"
            ),
        },
    }


def _resolve_scalar(
    mapping: Mapping[str, Any], json_path: str
) -> str | bool | int | float:
    value: Any = mapping
    for segment in json_path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise AuditedCardError(f"allowlisted ToolResult path is missing: {json_path}")
        value = value[segment]
    if isinstance(value, bool | str):
        return value
    if isinstance(value, Real):
        return float(value)
    raise AuditedCardError("allowlisted ToolResult path is not a scalar")


def _bounded_text(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or len(normalized) > 500 or any(ord(char) < 32 for char in normalized):
        raise AuditedCardError(f"{field_name} is not safe card text")
    return normalized


def _identifier(value: str, *, field_name: str) -> str:
    checked = value.strip()
    if not checked:
        raise ValueError(f"{field_name} must not be blank")
    if _SAFE_REFERENCE.fullmatch(checked) is None:
        raise ValueError(f"{field_name} must be a safe machine identifier")
    return checked
