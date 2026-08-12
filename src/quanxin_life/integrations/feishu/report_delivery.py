"""Deliver ledger-resolved audited report artifacts to Feishu."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from quanxin_life.core import ToolResult
from quanxin_life.reporting import AuditedReportArtifact, ReportArtifactFormat
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class FeishuReportDeliveryError(RuntimeError):
    """Raised when an audited artifact cannot be safely delivered."""


class AuditedArtifactExporter(Protocol):
    def export(
        self, result_id: str, format: ReportArtifactFormat
    ) -> AuditedReportArtifact: ...


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class FeishuReportClient(Protocol):
    def upload_file(
        self, *, filename: str, content_type: str, payload: bytes
    ) -> dict[str, object]: ...

    def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: dict[str, object],
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class FeishuReportDeliveryReceipt:
    source_result_id: str
    format: ReportArtifactFormat
    filename: str
    sha256: str
    file_key: str
    message_id: str


class FeishuReportDelivery:
    """Export by ToolResult identity, verify bytes, then upload and send."""

    def __init__(
        self,
        exporter: AuditedArtifactExporter,
        client: FeishuReportClient,
        *,
        result_resolver: RegisteredResultResolver,
    ) -> None:
        if not callable(getattr(result_resolver, "resolve_registered_result", None)):
            raise TypeError("result_resolver must resolve registered ToolResults")
        self._exporter = exporter
        self._client = client
        self._result_resolver = result_resolver

    def deliver(
        self,
        *,
        result_id: str,
        chat_id: str,
        format: ReportArtifactFormat = ReportArtifactFormat.MARKDOWN,
        existing_file_key: str | None = None,
        existing_message_id: str | None = None,
        on_uploaded: Callable[[str], None] | None = None,
        on_sent: Callable[[str], None] | None = None,
    ) -> FeishuReportDeliveryReceipt:
        checked_result_id = _reference(result_id, field_name="result_id")
        checked_chat_id = _reference(chat_id, field_name="chat_id")
        try:
            source_result = ToolResult.model_validate(
                self._result_resolver.resolve_registered_result(
                    checked_result_id
                ).model_dump(mode="json")
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise FeishuReportDeliveryError(
                "audited report ToolResult is not registered"
            ) from exc
        if (
            source_result.tool_name != AUDITED_REPORT_TOOL_NAME
            or source_result.tool_version != AUDITED_REPORT_TOOL_VERSION
            or source_result.model_version != REPORTING_VERSION
        ):
            raise FeishuReportDeliveryError(
                "source is not a supported audited report ToolResult"
            )
        artifact = self._exporter.export(checked_result_id, format)
        if artifact.source_result_id != checked_result_id:
            raise FeishuReportDeliveryError(
                "audited artifact source_result_id does not match the request"
            )
        if artifact.format is not format:
            raise FeishuReportDeliveryError(
                "audited report artifact format does not match the request"
            )
        actual_sha256 = sha256(artifact.payload).hexdigest()
        if actual_sha256 != artifact.sha256:
            raise FeishuReportDeliveryError(
                "audited report artifact SHA-256 verification failed"
            )
        if existing_message_id is not None and existing_file_key is None:
            raise FeishuReportDeliveryError(
                "checkpointed report message requires its uploaded file key"
            )
        if existing_file_key is None:
            uploaded = self._client.upload_file(
                filename=artifact.filename,
                content_type=artifact.media_type,
                payload=artifact.payload,
            )
            file_key = _response_reference(uploaded, "file_key")
            if on_uploaded is not None:
                on_uploaded(file_key)
        else:
            file_key = _reference(existing_file_key, field_name="file_key")
        if existing_message_id is None:
            sent = self._client.send_message(
                receive_id=checked_chat_id,
                receive_id_type="chat_id",
                msg_type="file",
                content={"file_key": file_key},
            )
            message_id = _response_reference(sent, "message_id")
            if on_sent is not None:
                on_sent(message_id)
        else:
            message_id = _reference(existing_message_id, field_name="message_id")
        return FeishuReportDeliveryReceipt(
            source_result_id=checked_result_id,
            format=artifact.format,
            filename=artifact.filename,
            sha256=actual_sha256,
            file_key=file_key,
            message_id=message_id,
        )


def _response_reference(response: dict[str, object], field_name: str) -> str:
    try:
        return _reference(response.get(field_name), field_name=field_name)
    except ValueError as exc:
        raise FeishuReportDeliveryError(
            f"Feishu response did not contain a safe {field_name}"
        ) from exc


def _reference(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe machine reference")
    return normalized


__all__ = [
    "AuditedArtifactExporter",
    "FeishuReportClient",
    "FeishuReportDelivery",
    "FeishuReportDeliveryError",
    "FeishuReportDeliveryReceipt",
    "RegisteredResultResolver",
]
