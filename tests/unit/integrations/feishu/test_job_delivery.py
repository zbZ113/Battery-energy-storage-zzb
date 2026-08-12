from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

from quanxin_life.core import EvidenceLevel, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.bitable import (
    BitableWriteAction,
    BitableWriteResult,
)
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorization
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobDelivery,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
    FeishuJobDeliveryProgress,
)
from quanxin_life.integrations.feishu.report_delivery import (
    FeishuReportDeliveryReceipt,
)
from quanxin_life.integrations.feishu.scenario_plot import (
    FeishuScenarioPlotArtifact,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.reporting import ReportArtifactFormat

NOW = datetime(2026, 8, 11, 6, 0, tzinfo=UTC)


class _Client:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []
        self.images: list[dict[str, object]] = []

    def send_message(self, **kwargs: object) -> dict[str, object]:
        self.messages.append(dict(kwargs))
        return {"message_id": f"om-{len(self.messages)}"}

    def upload_image(self, **kwargs: object) -> dict[str, object]:
        self.images.append(dict(kwargs))
        return {"image_key": "img-scenario"}


class _CardBuilder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def build_result_card(
        self,
        *,
        run_id: str,
        result_id: str,
        image_key: str | None = None,
    ) -> dict[str, object]:
        self.calls.append((run_id, result_id))
        return {
            "kind": "audited",
            "run_id": run_id,
            "result_id": result_id,
            "image_key": image_key,
        }


class _Bitable:
    def __init__(self) -> None:
        self.fields: list[dict[str, object]] = []

    def upsert(self, fields: dict[str, object]) -> BitableWriteResult:
        self.fields.append(dict(fields))
        return BitableWriteResult(
            run_id=str(fields["run_id"]),
            record_id="rec-job",
            action=BitableWriteAction.CREATED,
        )


class _ReportDelivery:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def deliver(
        self,
        *,
        result_id: str,
        chat_id: str,
        existing_file_key: str | None = None,
        existing_message_id: str | None = None,
        on_uploaded: object = None,
        on_sent: object = None,
    ) -> FeishuReportDeliveryReceipt:
        self.calls.append((result_id, chat_id))
        file_key = existing_file_key or "file-audited"
        message_id = existing_message_id or "om-report"
        if existing_file_key is None and callable(on_uploaded):
            on_uploaded(file_key)
        if existing_message_id is None and callable(on_sent):
            on_sent(message_id)
        return FeishuReportDeliveryReceipt(
            source_result_id=result_id,
            format=ReportArtifactFormat.MARKDOWN,
            filename="audited.md",
            sha256="b" * 64,
            file_key=file_key,
            message_id=message_id,
        )


class _Authorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "predict_cycle_life"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="controlled-route",
            activation_status="ACTIVE",
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            supported_domain="registered-canonical-csv",
        )


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


class _ScenarioPlotter:
    def render(self, result: ToolResult) -> FeishuScenarioPlotArtifact:
        payload = b"\x89PNG\r\n\x1a\ncontrolled"
        return FeishuScenarioPlotArtifact(
            source_result_id=result.result_id,
            filename=f"scenario-{result.result_id}.png",
            media_type="image/png",
            payload=payload,
            sha256=sha256(payload).hexdigest(),
        )


def _job(
    *,
    task_type: FeishuAnalysisTask = FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
    scenario_context_id: str | None = None,
) -> FeishuAnalysisJobRecord:
    return FeishuAnalysisJobRecord(
        job_id="3a3c972b-a23e-42c3-af76-e39038806f13",
        run_id="3a3c972b-a23e-42c3-af76-e39038806f13",
        event_id="evt-delivery",
        task_type=task_type,
        job_status=FeishuAnalysisJobStatus.RUNNING,
        job_stage="DELIVERING_RESULT",
        message_id="om-source",
        file_key="file-source",
        file_name="battery.csv",
        chat_id="oc-delivery",
        sender_id="ou-delivery",
        receive_id_type="chat_id",
        event_time=NOW,
        scenario_context_id=scenario_context_id,
        record_batch_id="batch-1",
        cell_reference="cell-1",
        input_file_sha256="a" * 64,
        validation_result_id="0f1eef6f-7193-48f5-9c49-f56e96f5eec7",
        analysis_result_id="a03018dd-7a76-42bf-a506-4dc571ddca7e",
        report_result_id="d9a37276-54ee-48ab-b491-ec5199213c6c",
        scenario_image_key=None,
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
    )


def _result(*, result_id: str, tool_name: str) -> ToolResult:
    return ToolResult(
        result_id=result_id,
        tool_name=tool_name,
        tool_version=(
            "advanced-rul-prediction-tool-v1"
            if tool_name == "predict_cycle_life"
            else "audited-report-tool-v1"
        ),
        model_version="model-v1",
        data_version="data-v1",
        feature_version="feature-v1",
        input_hash="c" * 64,
        values={"registered": True},
        warnings=["TRACEABLE_WARNING"],
        provenance=[
            ProvenanceRecord(
                source_id="file-source",
                source_kind=SourceKind.OBSERVED,
                uri="feishu://message/om-source/resource/file-source",
                sha256="a" * 64,
                description="Verified canonical CSV",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_success_delivery_uses_audited_card_bitable_metadata_and_report_result() -> None:
    client = _Client()
    cards = _CardBuilder()
    bitable = _Bitable()
    reports = _ReportDelivery()
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=cards,
        bitable_writer=bitable,
        report_delivery=reports,
        result_authorizer=_Authorizer(),
        report_link_factory=lambda receipt: (
            f"https://files.example.invalid/{receipt.file_key}"
        ),
    )
    job = _job()
    analysis = _result(
        result_id=job.analysis_result_id or "",
        tool_name="predict_cycle_life",
    )
    report = _result(
        result_id=job.report_result_id or "",
        tool_name="generate_audited_report",
    )

    receipt = delivery.deliver_success(
        job=job,
        analysis_result=analysis,
        report_result=report,
    )

    assert cards.calls == [(job.run_id, analysis.result_id)]
    assert reports.calls == [(report.result_id, "oc-delivery")]
    assert receipt.report_file_key == "file-audited"
    assert receipt.bitable_record_id == "rec-job"
    fields = bitable.fields[0]
    assert fields["primary_result_id"] == analysis.result_id
    assert fields["model_route"] == "controlled-route"
    assert fields["report_link"] == "https://files.example.invalid/file-audited"
    assert fields["cell_reference"] == "cell-1"
    assert all(not isinstance(value, list | dict) for value in fields.values())
    assert len(client.messages) == 2


def test_rejection_delivery_contains_only_metadata_and_reason_reference() -> None:
    client = _Client()
    bitable = _Bitable()
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=_CardBuilder(),
        bitable_writer=bitable,
        report_delivery=_ReportDelivery(),
        result_authorizer=_Authorizer(),
    )
    job = _job()

    receipt = delivery.deliver_rejection(
        job=job,
        reason_code="MODEL_ROUTE_NOT_ACTIVATED",
        primary_result_id=job.validation_result_id,
    )

    assert receipt.report_file_key is None
    assert receipt.bitable_record_id == "rec-job"
    assert bitable.fields[0]["task_status"] == "REJECTED"
    assert bitable.fields[0]["warnings"] == "MODEL_ROUTE_NOT_ACTIVATED"
    rendered = repr(client.messages[0])
    assert "predicted_cycle" not in rendered
    assert "remaining_cycles" not in rendered


def test_scenario_delivery_uploads_curve_and_writes_scalar_context_metadata() -> None:
    client = _Client()
    cards = _CardBuilder()
    bitable = _Bitable()
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=cards,
        bitable_writer=bitable,
        report_delivery=_ReportDelivery(),
        result_authorizer=_ScenarioAuthorizer(),
        scenario_plotter=_ScenarioPlotter(),
    )
    context_id = "5ab7c239-85a1-4ff2-bc58-2c15c006f311"
    job = _job(
        task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        scenario_context_id=context_id,
    )
    analysis = _result(
        result_id=job.analysis_result_id or "",
        tool_name="compare_operation_scenarios",
    ).model_copy(
        update={
            "tool_version": "compare-operation-scenarios-tool-v1",
            "values": {
                "artifact": {
                    "status": "COMPLETED",
                    "baseline": {
                        "scenario_id": "baseline",
                        "scenario_version": "baseline-v1",
                    },
                    "comparisons": [],
                }
            },
        }
    )
    report = _result(
        result_id=job.report_result_id or "",
        tool_name="generate_audited_report",
    )

    receipt = delivery.deliver_success(
        job=job,
        analysis_result=analysis,
        report_result=report,
    )

    assert receipt.bitable_record_id == "rec-job"
    assert len(client.images) == 1
    assert client.images[0]["payload"] == b"\x89PNG\r\n\x1a\ncontrolled"
    assert client.messages[0]["content"]["image_key"] == "img-scenario"  # type: ignore[index]
    fields = bitable.fields[0]
    assert fields["scenario_context_id"] == context_id
    assert fields["scenario_id"] == "baseline"
    assert fields["scenario_version"] == "baseline-v1"
    assert all(not isinstance(value, list | dict) for value in fields.values())


def test_success_delivery_replay_skips_checkpointed_external_actions() -> None:
    client = _Client()
    cards = _CardBuilder()
    bitable = _Bitable()
    reports = _ReportDelivery()
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=cards,
        bitable_writer=bitable,
        report_delivery=reports,
        result_authorizer=_Authorizer(),
    )
    job = _job()
    analysis = _result(
        result_id=job.analysis_result_id or "",
        tool_name="predict_cycle_life",
    )
    report = _result(
        result_id=job.report_result_id or "",
        tool_name="generate_audited_report",
    )
    progress: list[FeishuJobDeliveryProgress] = []

    first = delivery.deliver_success(
        job=job,
        analysis_result=analysis,
        report_result=report,
        checkpoint=progress.append,
    )

    replay_job = replace(
        job,
        result_card_message_id=next(
            item.result_card_message_id
            for item in progress
            if item.result_card_message_id is not None
        ),
        report_file_key=next(
            item.report_file_key for item in progress if item.report_file_key is not None
        ),
        report_message_id=next(
            item.report_message_id
            for item in progress
            if item.report_message_id is not None
        ),
        report_card_message_id=next(
            item.report_card_message_id
            for item in progress
            if item.report_card_message_id is not None
        ),
        bitable_record_id=next(
            item.bitable_record_id
            for item in progress
            if item.bitable_record_id is not None
        ),
    )
    before = (len(client.messages), len(client.images), len(bitable.fields), len(cards.calls))

    replayed = delivery.deliver_success(
        job=replay_job,
        analysis_result=analysis,
        report_result=report,
        checkpoint=lambda item: (_ for _ in ()).throw(
            AssertionError(f"replay emitted unexpected checkpoint: {item}")
        ),
    )

    assert first == replayed
    assert (
        len(client.messages),
        len(client.images),
        len(bitable.fields),
        len(cards.calls),
    ) == before
