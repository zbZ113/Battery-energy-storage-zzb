from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine, select

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    CellMetadata,
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.analysis_plots import FeishuAnalysisPlotter
from quanxin_life.integrations.feishu.attachments import FeishuAttachmentPolicy
from quanxin_life.integrations.feishu.bitable import (
    CHINESE_ANALYSIS_BITABLE_PROFILE,
    BitableMediaUploader,
    FeishuBitableWriter,
)
from quanxin_life.integrations.feishu.cards import (
    AuditedCardBuilder,
    AuditedResultAuthorization,
)
from quanxin_life.integrations.feishu.client import (
    FeishuClient,
    FeishuClientConfig,
    FeishuHttpRequest,
    FeishuHttpResponse,
)
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioProfile,
    ReviewedDefaultScenarioRegistry,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobDelivery,
    FeishuAnalysisJobStatus,
    FeishuAnalysisJobWorker,
    FeishuJobDispatchReceipt,
    FeishuProjectModelExecution,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.report_delivery import FeishuReportDelivery
from quanxin_life.integrations.feishu.routing import parse_feishu_event_reference
from quanxin_life.integrations.feishu.sandbox import create_fake_feishu_sandbox_app
from quanxin_life.integrations.feishu.scenario_authorization import (
    BlastScenarioResultAuthorizer,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
from quanxin_life.integrations.feishu.scenario_reports import (
    FeishuScenarioReportResultFactory,
)
from quanxin_life.integrations.feishu.sibling_planner import (
    ProactiveFeishuSiblingPlanner,
)
from quanxin_life.integrations.feishu.sqlalchemy_receipts import (
    SqlAlchemyFeishuReceiptStore,
)
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
)
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.scenarios import OperationScenario, ScenarioSegment
from quanxin_life.tools import StandardToolName, ToolRegistry
from quanxin_life.tools.audited_report import (
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
    execute_generate_audited_report_tool,
    register_generate_audited_report_tool,
)
from quanxin_life.tools.blast_scenarios import (
    register_compare_operation_scenarios_tool,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)
CSV_PAYLOAD = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"MATR,MATR_b3c34,1,0,0.0,3.6,1.0,25.0,1.1,1.05,0.02,true,true\n"
    b"MATR,MATR_b3c34,50,0,0.0,3.5,1.0,25.0,1.02,1.0,0.03,true,true\n"
)
_CONTROLLED_WARNING = "CONTROLLED_FAKE_TOOLRESULT_NOT_PRODUCTION"


class _SandboxAsgiTransport:
    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.requests: list[FeishuHttpRequest] = []

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        self.requests.append(request)
        files = [
            (item.field_name, (item.filename, item.payload, item.content_type))
            for item in request.files
        ]
        response = self._client.request(
            request.method,
            urlsplit(request.url).path,
            headers=request.headers,
            params=request.params,
            json=request.json_body,
            data=request.form or None,
            files=files or None,
        )
        json_body: dict[str, object] | None = None
        if "application/json" in response.headers.get("content-type", "").lower():
            candidate = response.json()
            if isinstance(candidate, dict):
                json_body = candidate
        return FeishuHttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            json_body=json_body,
            body=response.content,
        )


class _Queue:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_ids.append(job_id)
        return FeishuJobDispatchReceipt(
            job_id=job_id,
            task_id=f"task-{len(self.job_ids)}-{job_id}",
        )


class _ScenarioRouteAuthorizer:
    def authorize(self, **_: object) -> None:
        raise AssertionError("scenario tools own their candidate activation gate")


class _ScenarioInputBinder:
    def bind_analysis_input(
        self,
        *,
        requested_analysis_input: Mapping[str, object],
        **_: object,
    ) -> Mapping[str, object]:
        return dict(requested_analysis_input)


class _ControlledProjectModelExecutor:
    def __init__(self, ledger: AuditLedger) -> None:
        self._ledger = ledger
        self.calls: list[FeishuAnalysisTask] = []

    def execute(
        self,
        job: object,
        registration: CanonicalCsvBatchRegistration,
        *,
        claim_token: str | None = None,
    ) -> FeishuProjectModelExecution:
        assert claim_token is not None
        task = FeishuAnalysisTask(job.task_type)
        self.calls.append(task)
        prepared = self._ledger.register_result(
            ToolResult(
                result_id=str(uuid4()),
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                tool_version="prepare-advanced-input-tool-v1",
                model_version="controlled-fake-input-v1",
                data_version=registration.data_version,
                feature_version=registration.feature_config.feature_version,
                input_hash=sha256_canonical(
                    {"record_batch_id": job.record_batch_id, "task": task.value}
                ),
                values={"artifact": {"record_batch_id": job.record_batch_id}},
                warnings=[_CONTROLLED_WARNING],
                provenance=list(registration.provenance),
                created_at=NOW,
            )
        )
        analysis = self._ledger.register_result(
            _controlled_analysis_result(task=task, registration=registration)
        )
        report_input = _report_input(analysis)
        report = self._ledger.register_result(
            execute_generate_audited_report_tool(
                report_input,
                audit_ledger=self._ledger,
                clock=lambda: NOW,
            )
        )
        return FeishuProjectModelExecution(
            record_batch_id=job.record_batch_id,
            prepared_input_result_id=prepared.result_id,
            analysis_result=analysis,
            report_result=report,
        )


class _ControlledDeliveryAuthorizer:
    def __init__(self, ledger: AuditLedger) -> None:
        self._ledger = ledger
        self._scenario = BlastScenarioResultAuthorizer(
            allow_candidate_results=True
        )

    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        checked = ToolResult.model_validate(result.model_dump(mode="json"))
        if checked.tool_name == StandardToolName.GENERATE_AUDITED_REPORT.value:
            upstream_ids = checked.values.get("upstream_result_ids")
            if (
                not isinstance(upstream_ids, list)
                or len(upstream_ids) != 1
                or not isinstance(upstream_ids[0], str)
            ):
                return _rejected_authorization("CONTROLLED_REPORT_INVALID")
            try:
                upstream = self._ledger.resolve_registered_result(upstream_ids[0])
            except ValueError:
                return _rejected_authorization("CONTROLLED_REPORT_INVALID")
            return self.authorize(upstream)
        if checked.tool_name in {
            StandardToolName.COMPARE_OPERATION_SCENARIOS.value,
            StandardToolName.PROJECT_STORAGE_LIFETIME.value,
        }:
            return self._scenario.authorize(checked)
        if (
            checked.tool_name
            in {
                StandardToolName.PREDICT_CYCLE_LIFE.value,
                StandardToolName.PREDICT_SOH_TRAJECTORY.value,
            }
            and _CONTROLLED_WARNING in checked.warnings
        ):
            return AuditedResultAuthorization(
                allowed=True,
                route_id=f"controlled-fake-{checked.tool_name}",
                activation_status="CONTROLLED_TEST_ONLY",
                evidence_level=EvidenceLevel.MODEL_INFERENCE,
                supported_domain="Fake E2E only; not a production model result.",
            )
        return _rejected_authorization("CONTROLLED_RESULT_NOT_AUTHORIZED")


def test_one_upload_proactively_delivers_independent_rul_soh_and_blast_results() -> None:
    outcome = _run_proactive_pipeline(with_profile=True)

    assert outcome["statuses"] == {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE: FeishuAnalysisJobStatus.SUCCEEDED,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: FeishuAnalysisJobStatus.SUCCEEDED,
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: (
            FeishuAnalysisJobStatus.SUCCEEDED
        ),
    }
    assert outcome["download_count"] == 1
    assert outcome["job_count"] == 3
    assert outcome["result_count"] == 3
    assert outcome["bitable_records"] == 3
    assert outcome["media_uploads"] == 3
    assert outcome["curve_templates"] == {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE: "CYCLE_LIFE_SUMMARY",
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: "FINITE_SOH_CURVE",
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: "SCENARIO_COMPARISON",
    }
    scenario = outcome["results"][FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS]
    assert scenario.tool_name == StandardToolName.COMPARE_OPERATION_SCENARIOS.value
    assert {
        "CAPACITY_REFERENCE_MISMATCH",
        "CELL_FORMAT_REFERENCE_MISMATCH",
        "REFERENCE_USE_ONLY",
    }.issubset(scenario.warnings)
    authorization = outcome["authorizer"].authorize(scenario)
    assert authorization.evidence_level is EvidenceLevel.PHYSICS_REFERENCE
    assert outcome["scenario_profile_sha256"] is not None


def test_missing_profile_child_does_not_erase_rul_or_soh_success() -> None:
    outcome = _run_proactive_pipeline(with_profile=False)

    assert outcome["statuses"] == {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE: FeishuAnalysisJobStatus.SUCCEEDED,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: FeishuAnalysisJobStatus.SUCCEEDED,
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS: (
            FeishuAnalysisJobStatus.REJECTED
        ),
    }
    assert outcome["download_count"] == 1
    assert outcome["job_count"] == 3
    assert outcome["result_count"] == 2
    assert outcome["bitable_records"] == 3
    assert outcome["media_uploads"] == 2
    assert outcome["curve_templates"] == {
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE: "CYCLE_LIFE_SUMMARY",
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY: "FINITE_SOH_CURVE",
    }
    assert outcome["scenario_error"] == "SCENARIO_PARAMETERS_REQUIRED"
    assert outcome["scenario_profile_sha256"] is None


def _run_proactive_pipeline(*, with_profile: bool) -> dict[str, object]:
    sandbox = TestClient(create_fake_feishu_sandbox_app())
    seeded = sandbox.put(
        "/sandbox/resources/om-proactive/file-proactive",
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
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(sessions)
    jobs = SqlAlchemyFeishuJobStore(sessions, lease_seconds=60)
    contexts = SqlAlchemyFeishuScenarioContextStore(sessions)
    batches = InMemoryVerifiedEarlyCycleBatchStore()
    queue = _Queue()
    sibling_jobs = SqlAlchemyFeishuSiblingJobService(
        jobs,
        queue=queue,
        clock=lambda: NOW,
    )
    registry = ReviewedDefaultScenarioRegistry(
        (_default_profile(),) if with_profile else ()
    )
    planner = ProactiveFeishuSiblingPlanner(
        batch_store=batches,
        context_store=contexts,
        sibling_jobs=sibling_jobs,
        default_scenarios=registry,
        clock=lambda: NOW,
    )
    ledger = AuditLedger()
    tools = ToolRegistry()
    register_validate_battery_data_tool(tools)
    register_compare_operation_scenarios_tool(
        tools,
        context_resolver=contexts,
        allow_candidate_execution=True,
        clock=lambda: NOW,
    )
    register_generate_audited_report_tool(
        tools,
        audit_ledger=ledger,
        clock=lambda: NOW,
    )
    service = ToolInvocationService(registry=tools, audit_ledger=ledger)
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=_ScenarioRouteAuthorizer(),
        input_binder=_ScenarioInputBinder(),
    )
    authorizer = _ControlledDeliveryAuthorizer(ledger)
    analysis_plotter = FeishuAnalysisPlotter()
    delivery = FeishuAnalysisJobDelivery(
        client=client,
        card_builder=AuditedCardBuilder(
            ledger,
            authorizer=authorizer,
            binding_verifier=jobs,
        ),
        bitable_writer=FeishuBitableWriter(
            client,
            app_token="app-proactive",
            table_id="tbl-proactive",
            field_profile=CHINESE_ANALYSIS_BITABLE_PROFILE,
        ),
        report_delivery=FeishuReportDelivery(
            AuditedReportArtifactExporter(ledger),
            client,
            result_resolver=ledger,
        ),
        result_authorizer=authorizer,
        scenario_plotter=analysis_plotter,
        analysis_plotter=analysis_plotter,
        bitable_curve_plotter=analysis_plotter,
        bitable_media_uploader=BitableMediaUploader(
            client,
            app_token="app-proactive",
        ),
    )
    project_executor = _ControlledProjectModelExecutor(ledger)
    registration = _registration()
    worker = FeishuAnalysisJobWorker(
        jobs,
        client=client,
        attachment_policy=FeishuAttachmentPolicy(),
        batch_store=batches,
        registration_resolver=lambda _job, attachment, _now: (
            registration
            if attachment.sha256 == registration.metadata.source_sha256
            else (_ for _ in ()).throw(ValueError("unexpected payload SHA-256"))
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
        project_model_executor=project_executor,
        delivery=delivery,
        sibling_planner=planner,
        clock=lambda: NOW,
        heartbeat_interval_seconds=30,
    )
    event = parse_feishu_event_reference(
        {
            "header": {
                "event_id": f"evt-proactive-{with_profile}",
                "event_type": "im.message.receive_v1",
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou-proactive"}},
                "message": {
                    "message_id": "om-proactive",
                    "chat_id": "oc-proactive",
                    "message_type": "file",
                    "content": json.dumps(
                        {
                            "file_key": "file-proactive",
                            "file_name": "MATR_b3c34-cutoff-50.csv",
                        }
                    ),
                },
            },
        }
    )
    claim = receipts.claim(
        event_id=event.event_id,
        event_type=event.event_type,
        payload_sha256=sha256(f"sanitized-{with_profile}".encode()).hexdigest(),
        received_at=NOW,
    )
    assert claim.claim_token is not None
    routed = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda _event: FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        clock=lambda: NOW,
    ).route_event(event=event, claim_token=claim.claim_token)
    assert routed is FeishuEventRouteStatus.ENQUEUED
    root_job_id = queue.job_ids[0]
    assert worker.execute(job_id=root_job_id) is FeishuAnalysisJobStatus.SUCCEEDED

    dispatched_after_root = tuple(queue.job_ids)
    assert len(dispatched_after_root) == 3
    planner.plan_validated_siblings(job=jobs.get(root_job_id))
    assert tuple(queue.job_ids) == dispatched_after_root
    for child_id in dispatched_after_root[1:]:
        worker.execute(job_id=child_id)

    with sessions() as session:
        rows = tuple(
            session.scalars(
                select(FeishuEventReceipt).order_by(FeishuEventReceipt.job_created_at)
            ).all()
        )
    assert len(rows) == 3
    root = next(row for row in rows if row.job_id == root_job_id)
    children = tuple(row for row in rows if row.source_job_id == root_job_id)
    assert len(children) == 2
    assert all(row.record_batch_id == root.record_batch_id for row in children)
    assert all(row.message_id is None and row.file_key is None for row in children)
    records = {row.task_type: jobs.get(row.job_id) for row in rows}
    typed_records = {
        FeishuAnalysisTask(task): value for task, value in records.items()
    }
    statuses = {task: record.job_status for task, record in typed_records.items()}
    results: dict[FeishuAnalysisTask, ToolResult] = {}
    for task, record in typed_records.items():
        if record.analysis_result_id is not None:
            results[task] = ledger.resolve_registered_result(record.analysis_result_id)
    assert {
        task: result.tool_name for task, result in results.items()
    } == {task: task.value for task in results}
    assert len({result.result_id for result in results.values()}) == len(results)
    assert project_executor.calls == [
        FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
        FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    ]
    state = sandbox.get("/sandbox/state").json()
    scenario_record = typed_records[FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS]
    curve_templates = {
        task: record.bitable_curve_template
        for task, record in typed_records.items()
        if record.bitable_curve_template is not None
    }
    assert all(
        record.bitable_curve_source_result_id == record.analysis_result_id
        for record in typed_records.values()
        if record.bitable_curve_source_result_id is not None
    )
    return {
        "statuses": statuses,
        "download_count": sum(
            urlsplit(request.url).path
            == "/open-apis/im/v1/messages/om-proactive/resources/file-proactive"
            for request in transport.requests
        ),
        "job_count": len(rows),
        "result_count": len(results),
        "bitable_records": state["bitable_records"],
        "media_uploads": sum(
            urlsplit(request.url).path == "/open-apis/drive/v1/medias/upload_all"
            for request in transport.requests
        ),
        "curve_templates": curve_templates,
        "results": results,
        "authorizer": authorizer,
        "scenario_error": scenario_record.job_last_error_code,
        "scenario_profile_sha256": scenario_record.default_scenario_profile_sha256,
    }


def _controlled_analysis_result(
    *,
    task: FeishuAnalysisTask,
    registration: CanonicalCsvBatchRegistration,
) -> ToolResult:
    if task is FeishuAnalysisTask.PREDICT_CYCLE_LIFE:
        values = {
            "artifact_type": "quanxin_life.advanced_rul_prediction.v1",
            "artifact": {
                "cell_id": registration.metadata.cell_id,
                "cutoff_cycle": registration.feature_config.cutoff_cycle,
                "cycle_life_prediction": {"predicted_cycle": 52},
                "derived_remaining_cycles": 2,
            },
        }
        tool_version = "advanced-rul-prediction-tool-v1"
        uncertainty = None
    elif task is FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY:
        values = {
            "artifact_type": "quanxin_life.advanced_soh_trajectory.v1",
            "artifact": {
                "cell_id": registration.metadata.cell_id,
                "cutoff_cycle": registration.feature_config.cutoff_cycle,
                "prediction_cycles": [51, 500],
                "predicted_soh": [0.99, 0.87],
                "horizon_end_cycle": 500,
            },
        }
        tool_version = "advanced-soh-prediction-tool-v1"
        uncertainty = {
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        }
    else:  # pragma: no cover - executor is bound only to project model tasks
        raise AssertionError("unsupported controlled project task")
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=task.value,
        tool_version=tool_version,
        model_version=f"controlled-fake-{task.value}-v1",
        data_version=registration.data_version,
        feature_version=registration.feature_config.feature_version,
        input_hash=sha256_canonical({"task": task.value, "values": values}),
        values=values,
        uncertainty=uncertainty,
        warnings=[_CONTROLLED_WARNING],
        provenance=list(registration.provenance),
        created_at=NOW,
    )


def _report_input(result: ToolResult) -> GenerateAuditedReportToolInput:
    if result.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value:
        paths = (
            "values.artifact.cycle_life_prediction.predicted_cycle",
            "values.artifact.derived_remaining_cycles",
        )
    else:
        paths = (
            "values.artifact.predicted_soh.0",
            "values.artifact.predicted_soh.1",
        )
    return GenerateAuditedReportToolInput(
        report_kind=ReportKind.LIFETIME_DECISION,
        claims=(
            AuditedReportClaimReference(
                claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                numeric_evidence=tuple(
                    NumericEvidenceReference(
                        result_id=result.result_id,
                        json_path=path,
                    )
                    for path in paths
                ),
            ),
        ),
        upstream_result_ids=(result.result_id,),
    )


def _registration() -> CanonicalCsvBatchRegistration:
    digest = sha256(CSV_PAYLOAD).hexdigest()
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="MATR",
            cell_id="MATR_b3c34",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="feishu-sandbox://message/om-proactive/file-proactive",
            source_sha256=digest,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(
            cutoff_cycle=50,
            feature_version="cyclepatch-multichannel-v1",
        ),
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="MATR_b3c34-cutoff-50",
                source_kind=SourceKind.OBSERVED,
                uri="feishu-sandbox://message/om-proactive/file-proactive",
                sha256=digest,
                description="Reviewed Fake E2E MATR early-cycle upload.",
                created_at=NOW,
            ),
        ),
    )


def _default_profile() -> ReviewedDefaultScenarioProfile:
    digest = sha256(CSV_PAYLOAD).hexdigest()
    return ReviewedDefaultScenarioProfile(
        profile_id="matr-b3c34-proactive-reference",
        profile_version="matr-b3c34-proactive-reference-v1",
        review_status="APPROVED",
        task_type="compare_operation_scenarios",
        dataset_id="MATR",
        cell_id="MATR_b3c34",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        allowed_cutoff_cycles=(50,),
        allowed_source_sha256s=(digest,),
        chemistry="LFP/graphite",
        nominal_capacity_ah=1.1,
        source_cell_format="cylindrical",
        reference_cell_format="prismatic",
        route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        trusted_reference_use=True,
        reference_use_reason_code="MATR_TO_LARGE_FORMAT_LFP_REFERENCE_ONLY",
        initial_state_policy="BOL_ONLY",
        baseline=_scenario("baseline-25c", temperature_c=25.0),
        comparisons=(_scenario("comparison-35c", temperature_c=35.0),),
    )


def _scenario(scenario_id: str, *, temperature_c: float) -> OperationScenario:
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
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=300.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def _rejected_authorization(reason: str) -> AuditedResultAuthorization:
    return AuditedResultAuthorization(
        allowed=False,
        route_id="controlled-unresolved",
        activation_status="NOT_ACTIVATED",
        evidence_level=EvidenceLevel.MODEL_INFERENCE,
        supported_domain="Fake E2E only.",
        rejection_reason=reason,
    )
