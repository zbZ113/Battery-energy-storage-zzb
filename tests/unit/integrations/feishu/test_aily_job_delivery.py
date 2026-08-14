from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.aily_tasks import AilyAnalysisJobDelivery
from quanxin_life.integrations.feishu.bitable import (
    BitableWriteAction,
    BitableWriteResult,
)
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorization
from quanxin_life.integrations.feishu.delivery_contract import (
    FeishuDeliveryResultContractError,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask

NOW = datetime(2026, 8, 12, 11, 0, tzinfo=UTC)


class _Bitable:
    def __init__(self) -> None:
        self.fields: list[dict[str, object]] = []

    def upsert(self, fields: dict[str, object]) -> BitableWriteResult:
        self.fields.append(dict(fields))
        return BitableWriteResult(
            run_id=str(fields["run_id"]),
            record_id="rec-aily",
            action=BitableWriteAction.CREATED,
        )


class _Authorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name in {
            "compare_operation_scenarios",
            "generate_audited_report",
        }
        return AuditedResultAuthorization(
            allowed=True,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            activation_status="REGISTERED_CANDIDATE",
            evidence_level=EvidenceLevel.PHYSICS_REFERENCE,
            supported_domain="manifest-bounded-reference-scenario",
        )


class _ModelAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name in {"predict_cycle_life", "generate_audited_report"}
        return AuditedResultAuthorization(
            allowed=True,
            route_id="cyclepatch-direct-v3",
            activation_status="ACTIVE",
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            supported_domain="project-bound MATR cycle-life route",
        )


def _job() -> FeishuAnalysisJobRecord:
    job_id = str(uuid4())
    return FeishuAnalysisJobRecord(
        job_id=job_id,
        run_id=job_id,
        event_id="aily:" + "1" * 64,
        event_type="aily.analysis_task.create_v1",
        task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        job_status=FeishuAnalysisJobStatus.RUNNING,
        job_stage="DELIVERING_RESULT",
        message_id=None,
        file_key=None,
        file_name=None,
        chat_id=None,
        sender_id=None,
        receive_id_type=None,
        event_time=NOW,
        scenario_context_id=str(uuid4()),
        record_batch_id="canonical-csv-" + "2" * 64,
        cell_reference="cell-250ah",
        input_file_sha256="2" * 64,
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
        job_origin=FeishuAnalysisJobOrigin.AILY,
        job_request_sha256="3" * 64,
    )


def _result(job: FeishuAnalysisJobRecord) -> ToolResult:
    assert job.analysis_result_id is not None
    return ToolResult(
        result_id=job.analysis_result_id,
        tool_name="compare_operation_scenarios",
        tool_version="compare-operation-scenarios-tool-v1",
        model_version="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        data_version="scenario-data-v1",
        feature_version="operation-scenario-contract-v1",
        input_hash="4" * 64,
        values={
            "artifact": {
                "status": "COMPLETED",
                "route_id": "blast-lite-lfp-gr-250ah-prismatic-2019-v1",
                "baseline": {
                    "scenario_id": "baseline",
                    "scenario_version": "baseline-v1",
                    "natural_years": [0.0, 1.0],
                    "soh": [1.0, 0.99],
                },
                "comparisons": [],
            }
        },
        warnings=["CANDIDATE_ROUTE_RESEARCH_USE_ONLY"],
        provenance=[
            ProvenanceRecord(
                source_id="blast-route",
                source_kind=SourceKind.SIMULATED,
                uri="package://blast-route",
                sha256="5" * 64,
                description="Pinned candidate reference route.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _report(job: FeishuAnalysisJobRecord) -> ToolResult:
    assert job.report_result_id is not None
    return ToolResult(
        result_id=job.report_result_id,
        tool_name="generate_audited_report",
        tool_version="audited-report-tool-v1",
        model_version="audited-markdown-v1",
        data_version="scenario-data-v1",
        feature_version="operation-scenario-contract-v1",
        input_hash="6" * 64,
        values={
            "report_kind": "storage_lifetime_scenario",
            "upstream_result_ids": [job.analysis_result_id],
        },
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="analysis-result",
                source_kind=SourceKind.SIMULATED,
                uri=f"tool-result:{job.analysis_result_id}",
                sha256="7" * 64,
                description="Audited scenario result reference.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_aily_success_delivery_writes_only_scalar_metadata_and_report_link() -> None:
    bitable = _Bitable()
    delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable,
        result_authorizer=_Authorizer(),
        report_link_factory=lambda job, report: (
            f"https://api.example.invalid/v1/aily/analysis-tasks/{job.run_id}"
            f"/reports/{report.result_id}"
        ),
    )
    job = _job()

    receipt = delivery.deliver_success(
        job=job,
        analysis_result=_result(job),
        report_result=_report(job),
        checkpoint=lambda _progress: None,
    )

    assert receipt.bitable_record_id == "rec-aily"
    assert receipt.report_file_key is None
    fields = bitable.fields[0]
    assert fields["task_status"] == "SUCCEEDED"
    assert fields["scenario_id"] == "baseline"
    assert fields["scenario_version"] == "baseline-v1"
    assert fields["evidence_level"] == EvidenceLevel.PHYSICS_REFERENCE.value
    assert fields["report_link"] == (
        f"https://api.example.invalid/v1/aily/analysis-tasks/{job.run_id}"
        f"/reports/{job.report_result_id}"
    )
    assert all(not isinstance(value, list | dict) for value in fields.values())
    rendered = repr(fields).casefold()
    assert "natural_years" not in rendered
    assert "soh" not in rendered


def test_aily_success_delivery_rejects_unbound_report_before_bitable_write() -> None:
    bitable = _Bitable()
    delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable,
        result_authorizer=_Authorizer(),
    )
    job = _job()
    analysis = _result(job)
    report = _report(job).model_copy(
        update={"values": {"upstream_result_ids": [str(uuid4())]}}
    )

    with pytest.raises(
        FeishuDeliveryResultContractError,
        match="exact analysis result",
    ):
        delivery.deliver_success(
            job=job,
            analysis_result=analysis,
            report_result=report,
            checkpoint=lambda _progress: None,
        )

    assert bitable.fields == []


def test_aily_rejection_delivery_needs_no_chat_target() -> None:
    bitable = _Bitable()
    delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable,
        result_authorizer=_Authorizer(),
    )
    job = _job()

    receipt = delivery.deliver_rejection(
        job=job,
        reason_code="SCENARIO_CONTEXT_REJECTED",
        primary_result_id=job.validation_result_id,
        checkpoint=lambda _progress: None,
    )

    assert receipt.report_file_key is None
    assert bitable.fields[0]["task_status"] == "REJECTED"
    assert bitable.fields[0]["warnings"] == "SCENARIO_CONTEXT_REJECTED"


def test_aily_non_scenario_success_does_not_require_scenario_projection() -> None:
    bitable = _Bitable()
    delivery = AilyAnalysisJobDelivery(
        bitable_writer=bitable,
        result_authorizer=_ModelAuthorizer(),
    )
    job = replace(
        _job(),
        task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        scenario_context_id=None,
    )
    assert job.analysis_result_id is not None
    assert job.report_result_id is not None
    analysis = ToolResult(
        result_id=job.analysis_result_id,
        tool_name="predict_cycle_life",
        tool_version="advanced-rul-prediction-tool-v1",
        model_version="cyclepatch-direct-v3",
        data_version="matr-project-v1",
        feature_version="advanced-input-v1",
        input_hash="8" * 64,
        values={"artifact": {"record_batch_id": job.record_batch_id}},
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="project-batch",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch:project-batch",
                sha256="9" * 64,
                description="Authorized project batch.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )
    report = _report(job).model_copy(
        update={
            "values": {
                "report_kind": "lifetime_decision",
                "upstream_result_ids": [analysis.result_id],
            }
        }
    )

    delivery.deliver_success(
        job=job,
        analysis_result=analysis,
        report_result=report,
    )

    fields = bitable.fields[0]
    assert fields["task_type"] == FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value
    assert fields["evidence_level"] == EvidenceLevel.MODEL_INFERENCE.value
    assert "scenario_id" not in fields
    assert "scenario_version" not in fields
