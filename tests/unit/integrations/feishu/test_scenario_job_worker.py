from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from sqlalchemy import create_engine

from quanxin_life.api.feishu import FeishuEventRouteStatus
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.attachments import FeishuAttachmentPolicy
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobStatus,
    FeishuAnalysisJobWorker,
    FeishuJobDeliveryReceipt,
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobRouter,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.routing import (
    FeishuEventReference,
    FeishuInboundEventKind,
)
from quanxin_life.integrations.feishu.scenario_contexts import (
    SqlAlchemyFeishuScenarioContextStore,
)
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
from quanxin_life.scenarios import (
    OperationScenario,
    ScenarioCellDescriptor,
    ScenarioSegment,
    VerifiedScenarioContext,
)
from quanxin_life.tools import ToolRegistry
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.blast_scenarios import (
    CompareOperationScenariosToolInput,
    register_compare_operation_scenarios_tool,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 11, 10, 0, tzinfo=UTC)
HEADER = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
)
VALID_CSV = HEADER + (
    b"scenario-data,cell-1,1,0,0.0,3.6,1.0,25.0,1.2,1.1,0.02,true,true\n"
    b"scenario-data,cell-1,20,0,0.0,3.5,1.0,25.0,1.1,1.0,0.03,true,true\n"
)


class _Queue:
    def __init__(self) -> None:
        self.job_id: str | None = None

    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        self.job_id = job_id
        return FeishuJobDispatchReceipt(job_id=job_id, task_id="task-scenario")


class _NoDownloadClient:
    def __init__(self) -> None:
        self.calls = 0

    def download_message_resource_response(self, **_: str) -> object:
        self.calls += 1
        raise AssertionError("scenario card jobs must not download an attachment")


class _RouteAuthorizer:
    def authorize(self, **_: object) -> None:
        raise AssertionError("scenario tools use their own candidate activation gate")


class _ScenarioBinder:
    def bind_analysis_input(
        self,
        *,
        requested_analysis_input: dict[str, object],
        **_: object,
    ) -> dict[str, object]:
        return dict(requested_analysis_input)


class _Delivery:
    def __init__(self) -> None:
        self.analysis_result: ToolResult | None = None

    def deliver_rejection(self, **_: object) -> FeishuJobDeliveryReceipt:
        raise AssertionError("supported scenario fixture must not be rejected")

    def deliver_success(
        self,
        *,
        analysis_result: ToolResult,
        checkpoint: object,
        **_: object,
    ) -> FeishuJobDeliveryReceipt:
        assert callable(checkpoint)
        self.analysis_result = analysis_result
        return FeishuJobDeliveryReceipt(
            bitable_record_id="rec-scenario",
            report_file_key="file-scenario-report",
        )


class _JobBoundResultResolver:
    def __init__(
        self,
        *,
        ledger: AuditLedger,
        jobs: SqlAlchemyFeishuJobStore,
        run_id: str,
    ) -> None:
        self._ledger = ledger
        self._jobs = jobs
        self._run_id = run_id

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        if not self._jobs.is_result_bound_to_run(
            run_id=self._run_id,
            result_id=result_id,
        ):
            raise ValueError("ToolResult is not bound to the scenario job")
        return self._ledger.resolve_registered_result(result_id)


def _scenario() -> OperationScenario:
    return OperationScenario(
        scenario_id="baseline",
        scenario_version="baseline-v1",
        horizon_years=1,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="year-1",
                start_year=0,
                end_year=1,
                temperature_c=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    )


def test_worker_resolves_scenario_context_without_downloading_and_runs_real_tool() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    receipts = SqlAlchemyFeishuReceiptStore(session_factory)
    jobs = SqlAlchemyFeishuJobStore(session_factory, lease_seconds=60)
    context_store = SqlAlchemyFeishuScenarioContextStore(session_factory)
    batch_store = InMemoryVerifiedEarlyCycleBatchStore()
    provenance = (
        ProvenanceRecord(
            source_id="scenario-source",
            source_kind=SourceKind.OBSERVED,
            uri="memory://scenario-source",
            sha256=sha256(VALID_CSV).hexdigest(),
            description="Verified canonical scenario fixture.",
            created_at=NOW,
        ),
    )
    batch_id = batch_store.register_canonical_csv(
        VALID_CSV,
        registration=CanonicalCsvBatchRegistration(
            metadata=CellMetadata(
                dataset_id="scenario-data",
                cell_id="cell-1",
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.2,
                source_uri="memory://scenario-source",
                source_sha256=sha256(VALID_CSV).hexdigest(),
                schema_version="cycle-record-v1",
            ),
            feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
            data_version="scenario-data-v1",
            split_version="scenario-cell-v1",
            provenance=provenance,
        ),
    )
    batch = batch_store.resolve_verified_early_cycle_batch(batch_id)
    context_id = str(uuid4())
    cell = ScenarioCellDescriptor(
        chemistry=batch.metadata.chemistry,
        nominal_capacity_ah=batch.metadata.nominal_capacity_ah,
        cell_format="prismatic",
    )
    context_store.create(
        task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        data_batch_id=batch_id,
        verified_context=VerifiedScenarioContext(
            scenario_context_id=context_id,
            cell=cell,
            trusted_reference_use=True,
            data_version=batch.data_version,
            provenance=batch.provenance,
        ),
        analysis_input=CompareOperationScenariosToolInput(
            run_id=context_id,
            scenario_context_id=context_id,
            route_id="blast-lite-lfp-gr-250ah-prismatic-2019-v1",
            cell=cell,
            baseline=_scenario(),
            comparisons=(
                _scenario().model_copy(
                    update={
                        "scenario_id": "comparison",
                        "scenario_version": "comparison-v1",
                    }
                ),
            ),
        ),
        created_by_reference="ou-scenario",
        created_at=NOW,
    )
    claim = receipts.claim(
        event_id="evt-scenario-worker",
        event_type="card.action.trigger",
        payload_sha256="b" * 64,
        received_at=NOW,
    )
    assert claim.claim_token is not None
    queue = _Queue()
    routed = SqlAlchemyFeishuJobRouter(
        jobs,
        queue=queue,
        task_resolver=lambda event: FeishuAnalysisTask(event.action_value["task_type"]),
        clock=lambda: NOW,
    ).route_event(
        event=FeishuEventReference(
            event_id="evt-scenario-worker",
            event_type="card.action.trigger",
            kind=FeishuInboundEventKind.CARD_ACTION,
            chat_id="oc-scenario",
            user_id="ou-scenario",
            message_id="om-scenario",
            event_time=NOW,
            action_value={
                "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
                "scenario_context_id": context_id,
            },
        ),
        claim_token=claim.claim_token,
    )
    assert routed is FeishuEventRouteStatus.ENQUEUED
    assert queue.job_id is not None
    ledger = AuditLedger()
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_compare_operation_scenarios_tool(
        registry,
        context_resolver=context_store,
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
        route_authorizer=_RouteAuthorizer(),
        input_binder=_ScenarioBinder(),
    )
    delivery = _Delivery()
    client = _NoDownloadClient()
    bound_resolver = _JobBoundResultResolver(
        ledger=ledger,
        jobs=jobs,
        run_id=queue.job_id,
    )

    worker = FeishuAnalysisJobWorker(
        jobs,
        client=client,  # type: ignore[arg-type]
        attachment_policy=FeishuAttachmentPolicy(),
        batch_store=batch_store,
        registration_resolver=lambda *_: (_ for _ in ()).throw(
            AssertionError("scenario jobs must not register a new attachment")
        ),
        analysis_input_factory=lambda _job, resolved: ValidateBatteryDataToolInput(
            records=resolved.records,
            data_version=resolved.data_version,
            feature_version=resolved.feature_config.feature_version,
            provenance=resolved.provenance,
            validated_at=NOW,
        ).model_dump(mode="json"),
        scenario_input_resolver=context_store,
        workflow=workflow,
        result_resolver=bound_resolver,
        report_result_factory=FeishuScenarioReportResultFactory(service),
        delivery=delivery,
        clock=lambda: NOW,
        heartbeat_interval_seconds=30,
    )

    status = worker.execute(job_id=queue.job_id)

    snapshot = jobs.get(queue.job_id)
    assert status is FeishuAnalysisJobStatus.SUCCEEDED
    assert client.calls == 0
    assert snapshot.record_batch_id == batch_id
    assert snapshot.scenario_context_id == context_id
    assert delivery.analysis_result is not None
    assert delivery.analysis_result.tool_name == "compare_operation_scenarios"
    assert delivery.analysis_result.values["artifact"]["status"] == "COMPLETED"
    assert snapshot.report_result_id is not None
    report = ledger.resolve_registered_result(snapshot.report_result_id)
    assert report.tool_name == "generate_audited_report"
    assert report.values["upstream_result_ids"] == [delivery.analysis_result.result_id]
    assert "PHYSICS_REFERENCE" in report.values["markdown"]
