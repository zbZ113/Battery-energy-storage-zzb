from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.report_delivery import (
    FeishuReportDelivery,
    FeishuReportDeliveryError,
)
from quanxin_life.reporting import AuditedReportArtifact, ReportArtifactFormat
from quanxin_life.reporting.audited_markdown import REPORTING_VERSION
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_NAME,
    AUDITED_REPORT_TOOL_VERSION,
)

REPORT_RESULT_ID = "11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 7, tzinfo=UTC)


class _Exporter:
    def __init__(self, artifact: AuditedReportArtifact) -> None:
        self.artifact = artifact
        self.calls: list[tuple[str, ReportArtifactFormat]] = []

    def export(
        self, result_id: str, format: ReportArtifactFormat
    ) -> AuditedReportArtifact:
        self.calls.append((result_id, format))
        return self.artifact


class _Client:
    def __init__(self) -> None:
        self.uploads: list[dict[str, object]] = []
        self.messages: list[dict[str, object]] = []

    def upload_file(
        self, *, filename: str, content_type: str, payload: bytes
    ) -> dict[str, object]:
        self.uploads.append(
            {
                "filename": filename,
                "content_type": content_type,
                "payload": payload,
            }
        )
        return {"file_key": "file-audited-report"}

    def send_message(
        self,
        *,
        receive_id: str,
        receive_id_type: str,
        msg_type: str,
        content: dict[str, object],
    ) -> dict[str, object]:
        self.messages.append(
            {
                "receive_id": receive_id,
                "receive_id_type": receive_id_type,
                "msg_type": msg_type,
                "content": content,
            }
        )
        return {"message_id": "om-audited-report"}


def _artifact(
    *,
    declared_sha256: str | None = None,
    format: ReportArtifactFormat = ReportArtifactFormat.MARKDOWN,
) -> AuditedReportArtifact:
    payload = b"# Ledger-bound audited report\n"
    return AuditedReportArtifact(
        source_result_id=REPORT_RESULT_ID,
        format=format,
        filename="audited-report-safe.md",
        media_type="text/markdown; charset=utf-8",
        payload=payload,
        sha256=declared_sha256 or sha256(payload).hexdigest(),
    )


def _result(*, audited_report: bool = True) -> ToolResult:
    return ToolResult(
        result_id=REPORT_RESULT_ID,
        tool_name=(
            AUDITED_REPORT_TOOL_NAME if audited_report else "validate_battery_data"
        ),
        tool_version=(
            AUDITED_REPORT_TOOL_VERSION if audited_report else "data-quality-tool-v1"
        ),
        model_version=REPORTING_VERSION if audited_report else None,
        data_version="registered-data-v1",
        feature_version="registered-features-v1",
        input_hash="1" * 64,
        values={"report_kind": "audited_markdown"},
        uncertainty=None,
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="registered-source",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:registered-source",
                sha256="2" * 64,
                description="Registered observed source",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _delivery(
    exporter: _Exporter,
    client: _Client,
    *,
    audited_report: bool = True,
) -> FeishuReportDelivery:
    return FeishuReportDelivery(
        exporter,
        client,
        result_resolver=AuditLedger((_result(audited_report=audited_report),)),
    )


def test_delivery_exports_by_result_id_and_sends_only_the_uploaded_file_key() -> None:
    exporter = _Exporter(_artifact())
    client = _Client()

    delivered = _delivery(exporter, client).deliver(
        result_id=REPORT_RESULT_ID,
        chat_id="oc-reviewed-chat",
        format=ReportArtifactFormat.MARKDOWN,
    )

    assert exporter.calls == [(REPORT_RESULT_ID, ReportArtifactFormat.MARKDOWN)]
    assert client.uploads == [
        {
            "filename": "audited-report-safe.md",
            "content_type": "text/markdown; charset=utf-8",
            "payload": b"# Ledger-bound audited report\n",
        }
    ]
    assert client.messages == [
        {
            "receive_id": "oc-reviewed-chat",
            "receive_id_type": "chat_id",
            "msg_type": "file",
            "content": {"file_key": "file-audited-report"},
        }
    ]
    assert delivered.source_result_id == REPORT_RESULT_ID
    assert delivered.file_key == "file-audited-report"
    assert delivered.message_id == "om-audited-report"
    assert delivered.sha256 == sha256(b"# Ledger-bound audited report\n").hexdigest()


def test_delivery_rejects_an_artifact_with_a_mismatched_sha_before_upload() -> None:
    client = _Client()

    with pytest.raises(FeishuReportDeliveryError, match="SHA-256"):
        _delivery(_Exporter(_artifact(declared_sha256="0" * 64)), client).deliver(
            result_id=REPORT_RESULT_ID,
            chat_id="oc-reviewed-chat",
            format=ReportArtifactFormat.MARKDOWN,
        )

    assert client.uploads == []
    assert client.messages == []


def test_delivery_rejects_a_malformed_feishu_upload_response() -> None:
    class _MalformedClient(_Client):
        def upload_file(
            self, *, filename: str, content_type: str, payload: bytes
        ) -> dict[str, object]:
            super().upload_file(
                filename=filename,
                content_type=content_type,
                payload=payload,
            )
            return {}

    client = _MalformedClient()

    with pytest.raises(FeishuReportDeliveryError, match="file_key"):
        _delivery(_Exporter(_artifact()), client).deliver(
            result_id=REPORT_RESULT_ID,
            chat_id="oc-reviewed-chat",
        )

    assert client.messages == []


def test_delivery_rejects_a_non_report_tool_result_before_export() -> None:
    exporter = _Exporter(_artifact())
    client = _Client()

    with pytest.raises(FeishuReportDeliveryError, match="audited report ToolResult"):
        _delivery(exporter, client, audited_report=False).deliver(
            result_id=REPORT_RESULT_ID,
            chat_id="oc-reviewed-chat",
        )

    assert exporter.calls == []
    assert client.uploads == []


def test_delivery_rejects_an_exported_format_that_differs_from_the_request() -> None:
    exporter = _Exporter(_artifact(format=ReportArtifactFormat.JSON))
    client = _Client()

    with pytest.raises(FeishuReportDeliveryError, match="format"):
        _delivery(exporter, client).deliver(
            result_id=REPORT_RESULT_ID,
            chat_id="oc-reviewed-chat",
            format=ReportArtifactFormat.MARKDOWN,
        )

    assert client.uploads == []


def test_delivery_reuses_checkpointed_upload_after_message_retry() -> None:
    class _FailMessageOnceClient(_Client):
        def __init__(self) -> None:
            super().__init__()
            self.fail_once = True

        def send_message(self, **kwargs: object) -> dict[str, object]:
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("controlled message failure")
            return super().send_message(**kwargs)  # type: ignore[arg-type]

    exporter = _Exporter(_artifact())
    client = _FailMessageOnceClient()
    uploaded: list[str] = []
    sent: list[str] = []
    delivery = _delivery(exporter, client)

    with pytest.raises(RuntimeError, match="controlled message failure"):
        delivery.deliver(
            result_id=REPORT_RESULT_ID,
            chat_id="oc-reviewed-chat",
            on_uploaded=uploaded.append,
            on_sent=sent.append,
        )

    assert uploaded == ["file-audited-report"]
    assert sent == []
    delivered = delivery.deliver(
        result_id=REPORT_RESULT_ID,
        chat_id="oc-reviewed-chat",
        existing_file_key=uploaded[0],
        on_uploaded=uploaded.append,
        on_sent=sent.append,
    )

    assert len(client.uploads) == 1
    assert len(client.messages) == 1
    assert uploaded == ["file-audited-report"]
    assert sent == ["om-audited-report"]
    assert delivered.file_key == "file-audited-report"
    assert delivered.message_id == "om-audited-report"
