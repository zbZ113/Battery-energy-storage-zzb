from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import urlsplit

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine

from quanxin_life.api.aily import AilyCompareScenarioContextRequest
from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.aily_scenarios import (
    SqlAlchemyAilyScenarioContextGateway,
)
from quanxin_life.integrations.feishu.attachments import FeishuAttachmentPolicy
from quanxin_life.integrations.feishu.bitable import FeishuBitableWriter
from quanxin_life.integrations.feishu.cards import AuditedCardBuilder
from quanxin_life.integrations.feishu.client import (
    FeishuClient,
    FeishuClientConfig,
    FeishuHttpRequest,
    FeishuHttpResponse,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobDelivery,
    FeishuAnalysisJobStatus,
    FeishuAnalysisJobWorker,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.report_delivery import FeishuReportDelivery
from quanxin_life.integrations.feishu.routing import parse_feishu_event_reference
from quanxin_life.integrations.feishu.sandbox import create_fake_feishu_sandbox_app
from quanxin_life.integrations.feishu.scenario_authorization import (
    AuditedScenarioResultAuthorizer,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.scenario_plot import FeishuScenarioPlotter
from quanxin_life.integrations.feishu.scenario_reports import (
    FeishuScenarioReportResultFactory,
)
from quanxin_life.integrations.feishu.sqlalchemy_receipts import (
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
)
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.scenarios import OperationScenario, ScenarioSegment
from quanxin_life.tools import ToolRegistry
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.blast_scenarios import register_compare_operation_scenarios_tool
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 11, 15, 0, tzinfo=UTC)
SOURCE_RUN_ID = "d9d05347-682e-46ac-9658-2b04dca645f7"
CSV_PAYLOAD = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"sandbox-scenario,cell-250ah,1,0,0.0,3.6,125.0,25.0,250.0,249.0,0.02,true,true\n"
    b"sandbox-scenario,cell-250ah,20,0,0.0,3.5,125.0,25.0,248.0,247.0,0.03,true,true\n"
)


class _SandboxAsgiTransport:
    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.requests: list[FeishuHttpRequest] = []

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        self.requests.append(request)
        path = urlsplit(request.url).path
        files = [
            (
                item.field_name,
                (item.filename, item.payload, item.content_type),
            )
            for item in request.files
        ]
        response = self._client.request(
            request.method,
            path,
            headers=request.headers,
            params=request.params,
            json=request.json_body,
            data=request.form or None,
            files=files or None,
        )
        json_body: dict[str, object] | None = None
        if "application/json" in response.headers.get("content-type", "").lower():
            value = response.json()
            if isinstance(value, dict):
                json_body = value
        return FeishuHttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            json_body=json_body,
            body=response.content,
        )


class _FixtureDataIdentity:
    def resolve_source_job(self, **_: object) -> None:
        return None


class _Queue:
    def __init__(self) -> None:
        self.job_id: str | None = None

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_id = job_id
        return FeishuJobDispatchReceipt(job_id=job_id, task_id="task-scenario-e2e")


class _ScenarioRouteAuthorizer:
    def authorize(self, **_: object) -> None:
        raise AssertionError("scenario tools must use the BLAST candidate gate")


class _ScenarioInputBinder:
    def bind_analysis_input(
        self,
        *,
        requested_analysis_input: Mapping[str, object],
        **_: object,
    ) -> Mapping[str, object]:
        return dict(requested_analysis_input)


def _scenario(
    *,
    scenario_id: str,
    temperature_c: float,
    discharge_c_rate: float,
) -> OperationScenario:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=25,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="years-1-25",
                start_year=0,
                end_year=25,
                temperature_c=temperature_c,
                charge_c_rate=0.5,
                discharge_c_rate=discharge_c_rate,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def test_fake_feishu_scenario_pipeline_delivers_only_audited_results() -> None:
    sandbox = TestClient(create_fake_feishu_sandbox_app())
    seeded = sandbox.put(
        "/sandbox/resources/om-source/file-source",
        content=CSV_PAYLOAD,
        headers={"content-type": "text/csv"},
    )
    assert seeded.status_code == 204
    transport = _SandboxAsgiTransport(sandbox)
    client = FeishuClient(
        FeishuClientConfig(
            app_id="sandbox-app",
            app_secret=SecretStr("sandbox-secret"),
            base_url="http://testserver/open-apis",
            backoff_base_seconds=0,
        ),
        transport=transport,
    )
    downloaded = client.download_message_resource(
        message_id="om-source",
        file_key="file-source",
        resource_type="file",
    )
    attachment = FeishuAttachmentPolicy().verify(
        filename="scenario.csv",
        content_type="text/csv",
        payload=downloaded,
    )
    batches = InMemoryVerifiedEarlyCycleBatchStore()
    batch_id = batches.register_canonical_csv(
        attachment.payload,
        registration=_registration(attachment.sha256),
    )
    downloads_before_worker = _download_count(transport)

    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(session_factory)
    jobs = SqlAlchemyFeishuJobStore(session_factory, lease_seconds=60)
    contexts = SqlAlchemyFeishuScenarioContextStore(session_factory)
    context = SqlAlchemyAilyScenarioContextGateway(
        context_store=contexts,
        batch_store=batches,
        data_identity_resolver=_FixtureDataIdentity(),
        clock=lambda: NOW,
    ).create_scenario_context(
        AilyCompareScenarioContextRequest(
            task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
            source_run_id=SOURCE_RUN_ID,
            data_batch_id=batch_id,
            cell_format="prismatic",
            baseline=_scenario(
                scenario_id="baseline",
                temperature_c=25.0,
                discharge_c_rate=0.5,
            ),
            comparisons=(
                _scenario(
                    scenario_id="warmer-higher-rate",
                    temperature_c=35.0,
                    discharge_c_rate=1.0,
                ),
            ),
        )
    )
    event = parse_feishu_event_reference(
        {
            "header": {
                "event_id": "evt-scenario-e2e",
                "event_type": "card.action.trigger",
            },
            "event": {
                "operator": {"open_id": "ou-scenario"},
                "context": {
                    "open_chat_id": "oc-scenario",
                    "open_message_id": "om-scenario-card",
                },
                "action": {
                    "tag": "button",
                    "value": {
                        "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
                        "scenario_context_id": context.scenario_context_id,
                    },
                },
            },
        }
    )
    claim = receipts.claim(
        event_id=event.event_id,
        event_type=event.event_type,
        payload_sha256=sha256(b"sanitized-scenario-e2e").hexdigest(),
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    routed = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda value: FeishuAnalysisTask(
            value.action_value["task_type"]
        ),
        clock=lambda: NOW,
    ).route_event(event=event, claim_token=claim.claim_token)
    assert routed is FeishuEventRouteStatus.ENQUEUED
    assert queue.job_id is not None

    ledger = AuditLedger()
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_compare_operation_scenarios_tool(
        registry,
        context_resolver=contexts,
        allow_candidate_execution=True,
        clock=lambda: NOW,
    )
    register_generate_audited_report_tool(
        registry,
        audit_ledger=ledger,
        clock=lambda: NOW,
    )
    service = ToolInvocationService(registry=registry, audit_ledger=ledger)
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=_ScenarioRouteAuthorizer(),
        input_binder=_ScenarioInputBinder(),
    )
    authorizer = AuditedScenarioResultAuthorizer(
        result_resolver=ledger,
        allow_candidate_results=True,
    )
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=AuditedCardBuilder(
            ledger,
            authorizer=authorizer,
            binding_verifier=jobs,
        ),
        bitable_writer=FeishuBitableWriter(
            client,
            app_token="app-sandbox",
            table_id="tbl-sandbox",
        ),
        report_delivery=FeishuReportDelivery(
            AuditedReportArtifactExporter(ledger),
            client,
            result_resolver=ledger,
        ),
        result_authorizer=authorizer,
        report_link_factory=lambda value: (
            f"https://sandbox.invalid/reports/{value.file_key}"
        ),
        scenario_plotter=FeishuScenarioPlotter(),
    )
    worker = FeishuAnalysisJobWorker(
        jobs,
        client=client,
        attachment_policy=FeishuAttachmentPolicy(),
        batch_store=batches,
        registration_resolver=lambda *_: (_ for _ in ()).throw(
            AssertionError("scenario card jobs must reuse the existing batch")
        ),
        analysis_input_factory=lambda _job, batch: ValidateBatteryDataToolInput(
            records=batch.records,
            data_version=batch.data_version,
            feature_version=batch.feature_config.feature_version,
            provenance=batch.provenance,
            validated_at=NOW,
        ).model_dump(mode="json"),
        scenario_input_resolver=contexts,
        workflow=workflow,
        result_resolver=ledger,
        report_result_factory=FeishuScenarioReportResultFactory(service),
        delivery=delivery,
        clock=lambda: NOW,
        heartbeat_interval_seconds=30,
    )

    status = worker.execute(job_id=queue.job_id)

    snapshot = jobs.get(queue.job_id)
    assert snapshot.job_last_error_code is None
    assert status is FeishuAnalysisJobStatus.SUCCEEDED
    assert _download_count(transport) == downloads_before_worker
    assert snapshot.record_batch_id == batch_id
    assert snapshot.scenario_context_id == context.scenario_context_id
    assert snapshot.analysis_result_id is not None
    assert snapshot.report_result_id is not None
    assert snapshot.scenario_image_key is not None
    assert snapshot.result_card_message_id is not None
    assert snapshot.report_file_key is not None
    assert snapshot.report_message_id is not None
    assert snapshot.report_card_message_id is not None
    assert snapshot.bitable_record_id is not None

    analysis = ledger.resolve_registered_result(snapshot.analysis_result_id)
    assert analysis.values["artifact"]["status"] == "COMPLETED"
    baseline = analysis.values["artifact"]["baseline"]
    assert set(baseline["milestone_soh"]) == {"15", "20", "25"}
    report = ledger.resolve_registered_result(snapshot.report_result_id)
    assert report.values["upstream_result_ids"] == [analysis.result_id]
    assert "PHYSICS_REFERENCE" in report.values["markdown"]

    assert sandbox.get("/sandbox/state").json() == {
        "messages": 3,
        "resources": 1,
        "bitable_records": 1,
    }
    assert any(_path(item) == "/open-apis/im/v1/images" for item in transport.requests)
    assert any(_path(item) == "/open-apis/im/v1/files" for item in transport.requests)
    fields = _created_bitable_fields(transport)
    assert fields["scenario_context_id"] == context.scenario_context_id
    assert fields["scenario_id"] == "baseline"
    assert fields["scenario_version"] == "baseline-v1"
    assert fields["primary_result_id"] == analysis.result_id
    assert fields["evidence_level"] == "PHYSICS_REFERENCE"
    assert all(not isinstance(value, list | dict) for value in fields.values())
    assert "natural_years" not in repr(fields)
    assert "model_effective_full_cycles" not in repr(fields)
    result_card = _result_card_content(transport)
    assert snapshot.scenario_image_key in result_card
    assert "储能工况年份推演" in result_card
    assert "15 年 SOH" in result_card
    assert "20 年 SOH" in result_card
    assert "25 年 SOH" in result_card
    assert "不是 MATR 电芯的自然年换算" in result_card
    assert "values.artifact" not in result_card
    assert analysis.result_id not in result_card
    assert "natural_years" not in result_card
    assert "model_effective_full_cycles" not in result_card


def _registration(payload_sha256: str) -> CanonicalCsvBatchRegistration:
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="sandbox-scenario",
            cell_id="cell-250ah",
            chemistry="LFP/graphite",
            nominal_capacity_ah=250.0,
            source_uri="feishu-sandbox://message/om-source/resource/file-source",
            source_sha256=payload_sha256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="sandbox-scenario-v1",
        split_version="sandbox-scenario-cell-v1",
        provenance=(
            ProvenanceRecord(
                source_id="file-source",
                source_kind=SourceKind.OBSERVED,
                uri="feishu-sandbox://message/om-source/resource/file-source",
                sha256=payload_sha256,
                description="Verified Fake Feishu canonical scenario CSV.",
                created_at=NOW,
            ),
        ),
    )


def _path(request: FeishuHttpRequest) -> str:
    return urlsplit(request.url).path


def _download_count(transport: _SandboxAsgiTransport) -> int:
    return sum(
        _path(request)
        == "/open-apis/im/v1/messages/om-source/resources/file-source"
        for request in transport.requests
    )


def _created_bitable_fields(
    transport: _SandboxAsgiTransport,
) -> dict[str, object]:
    request = next(
        item
        for item in transport.requests
        if item.method == "POST"
        and _path(item)
        == "/open-apis/bitable/v1/apps/app-sandbox/tables/tbl-sandbox/records"
    )
    assert isinstance(request.json_body, dict)
    fields = request.json_body.get("fields")
    assert isinstance(fields, dict)
    return fields


def _result_card_content(
    transport: _SandboxAsgiTransport,
) -> str:
    for request in transport.requests:
        if request.method != "POST" or _path(request) != "/open-apis/im/v1/messages":
            continue
        if not isinstance(request.json_body, dict):
            continue
        content = request.json_body.get("content")
        if not isinstance(content, str):
            continue
        decoded = json.loads(content)
        if (
            isinstance(decoded, dict)
            and decoded.get("header", {}).get("title", {}).get("content")
            == "储能工况年份推演"
        ):
            return content
    raise AssertionError("audited scenario result card was not sent")
