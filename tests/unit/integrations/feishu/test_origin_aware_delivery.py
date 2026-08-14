from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
    FeishuJobDeliveryProgress,
    FeishuJobDeliveryReceipt,
    OriginAwareAnalysisJobDelivery,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class _Delivery:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, FeishuAnalysisJobOrigin]] = []

    def deliver_rejection(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        reason_code: str,
        primary_result_id: str | None,
        checkpoint: object,
    ) -> FeishuJobDeliveryReceipt:
        del reason_code, primary_result_id, checkpoint
        self.calls.append(("rejection", job.job_origin))
        return FeishuJobDeliveryReceipt(
            bitable_record_id=f"rec-{self.name}",
            report_file_key=None,
        )

    def deliver_success(
        self,
        *,
        job: FeishuAnalysisJobRecord,
        analysis_result: ToolResult,
        report_result: ToolResult,
        checkpoint: object,
    ) -> FeishuJobDeliveryReceipt:
        del analysis_result, report_result, checkpoint
        self.calls.append(("success", job.job_origin))
        return FeishuJobDeliveryReceipt(
            bitable_record_id=f"rec-{self.name}",
            report_file_key=None,
        )


def _job(origin: FeishuAnalysisJobOrigin) -> FeishuAnalysisJobRecord:
    job_id = str(uuid4())
    return FeishuAnalysisJobRecord(
        job_id=job_id,
        run_id=job_id,
        event_id=f"{origin.value.casefold()}:{uuid4()}",
        event_type=(
            "im.message.receive_v1"
            if origin is FeishuAnalysisJobOrigin.FEISHU
            else "aily.analysis_task.create_v1"
        ),
        task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        job_status=FeishuAnalysisJobStatus.RUNNING,
        job_stage="DELIVERING_RESULT",
        message_id="om-source" if origin is FeishuAnalysisJobOrigin.FEISHU else None,
        file_key=None,
        file_name=None,
        chat_id="oc-target" if origin is FeishuAnalysisJobOrigin.FEISHU else None,
        sender_id=None,
        receive_id_type=(
            "chat_id" if origin is FeishuAnalysisJobOrigin.FEISHU else None
        ),
        event_time=NOW,
        scenario_context_id=str(uuid4()),
        record_batch_id="canonical-csv-" + "1" * 64,
        cell_reference="cell-reference",
        input_file_sha256="1" * 64,
        validation_result_id=str(uuid4()),
        prepared_input_result_id=None,
        analysis_result_id=str(uuid4()),
        report_result_id=str(uuid4()),
        scenario_image_key=None,
        analysis_image_key=None,
        analysis_image_renderer_version=None,
        analysis_image_sha256=None,
        csv_mapping_status=None,
        csv_mapping_evidence=None,
        csv_mapping_evidence_sha256=None,
        result_card_message_id=None,
        report_file_key=None,
        report_message_id=None,
        report_card_message_id=None,
        bitable_record_id=None,
        job_last_error_code=None,
        job_attempt_count=1,
        job_created_at=NOW,
        job_updated_at=NOW,
        job_completed_at=None,
        job_origin=origin,
        job_request_sha256="2" * 64 if origin is FeishuAnalysisJobOrigin.AILY else None,
    )


def _result(tool_name: str) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version="tool-v1",
        model_version="model-v1",
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash="3" * 64,
        values={"status": "COMPLETED"},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="source",
                source_kind=SourceKind.SIMULATED,
                uri="package://source",
                sha256="4" * 64,
                description="Controlled test source.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_origin_aware_delivery_dispatches_success_to_exactly_one_channel() -> None:
    feishu = _Delivery("feishu")
    aily = _Delivery("aily")
    delivery = OriginAwareAnalysisJobDelivery(
        feishu_delivery=feishu,
        aily_delivery=aily,
    )
    analysis = _result("compare_operation_scenarios")
    report = _result("generate_audited_report")

    feishu_receipt = delivery.deliver_success(
        job=_job(FeishuAnalysisJobOrigin.FEISHU),
        analysis_result=analysis,
        report_result=report,
        checkpoint=lambda _progress: None,
    )
    aily_receipt = delivery.deliver_success(
        job=_job(FeishuAnalysisJobOrigin.AILY),
        analysis_result=analysis,
        report_result=report,
        checkpoint=lambda _progress: None,
    )

    assert feishu_receipt.bitable_record_id == "rec-feishu"
    assert aily_receipt.bitable_record_id == "rec-aily"
    assert feishu.calls == [("success", FeishuAnalysisJobOrigin.FEISHU)]
    assert aily.calls == [("success", FeishuAnalysisJobOrigin.AILY)]


def test_origin_aware_delivery_dispatches_rejection_without_cross_channel_send() -> None:
    feishu = _Delivery("feishu")
    aily = _Delivery("aily")
    delivery = OriginAwareAnalysisJobDelivery(
        feishu_delivery=feishu,
        aily_delivery=aily,
    )
    aily_job = replace(_job(FeishuAnalysisJobOrigin.AILY), chat_id=None)

    receipt = delivery.deliver_rejection(
        job=aily_job,
        reason_code="SCENARIO_REJECTED",
        primary_result_id=None,
        checkpoint=lambda _progress: None,
    )

    assert receipt.bitable_record_id == "rec-aily"
    assert feishu.calls == []
    assert aily.calls == [("rejection", FeishuAnalysisJobOrigin.AILY)]
    assert isinstance(FeishuJobDeliveryProgress(), FeishuJobDeliveryProgress)
