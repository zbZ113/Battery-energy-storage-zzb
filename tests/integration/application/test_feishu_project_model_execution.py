from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.api.service import ToolInvocation
from quanxin_life.application.feishu_project_models import (
    FeishuProjectModelExecutor,
    FeishuProjectModelRejected,
    FeishuProjectRecordBatchResolver,
    FeishuProjectResultResolver,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import AuditLedgerError, SqlAuditLedger, SqlProjectAuditLedger
from quanxin_life.core import (
    AdvancedModelRouteRole,
    CellMetadata,
    DatasetStatus,
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import session_scope
from quanxin_life.persistence.models import (
    Dataset,
    FeishuBindingRow,
    FeishuEventReceipt,
    Project,
    ProjectToolResultBindingRecord,
    RecordBatchBinding,
    User,
)
from quanxin_life.reporting.contracts import AUDITED_REPORT_TOOL_VERSION
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
)
from quanxin_life.tools.advanced_input import (
    ADVANCED_INPUT_EVIDENCE_TYPE,
    PREPARE_ADVANCED_INPUT_TOOL_VERSION,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
SOURCE_SHA256 = "a" * 64


@dataclass(frozen=True, slots=True)
class _Job:
    task_type: object
    chat_id: str | None = "oc-approved"
    sender_id: str | None = "ou-approved"
    job_id: str = ""
    run_id: str = ""
    prepared_input_result_id: str | None = None
    analysis_result_id: str | None = None
    report_result_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Fixture:
    sessions: object
    context_service: ProjectInvocationContextService
    project_ledger: SqlProjectAuditLedger
    project_id: str
    user_id: str
    binding_id: str
    record_batch_id: str


def _fixture(*, cutoff_cycle: int = 20) -> _Fixture:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    binding_id = str(uuid4())
    dataset_id = str(uuid4())
    record_batch_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add(
            User(
                id=user_id,
                username="feishu-model@example.test",
                credential_hash="not-used-by-feishu",
                must_change_credential=False,
                role=UserRole.ADMIN.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Project(
                id=project_id,
                owner_user_id=user_id,
                name="Feishu model project",
                status=ProjectStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            Dataset(
                id=dataset_id,
                project_id=project_id,
                name="Frozen MATR",
                data_version="matr-v1",
                schema_version="cycle-record-v1",
                status=DatasetStatus.FROZEN.value,
                manifest_uri="manifest://matr-v1",
                manifest_sha256="b" * 64,
                created_at=NOW,
                frozen_at=NOW,
            )
        )
        session.add(
            FeishuBindingRow(
                id=binding_id,
                project_id=project_id,
                chat_id="oc-approved",
                bitable_app_token=None,
                bitable_table_id=None,
                user_open_id_map_json={user_id: "ou-approved"},
                binding_version="feishu-binding-v1",
                status="ACTIVE",
                created_at=NOW,
            )
        )
        session.add(
            RecordBatchBinding(
                id=record_batch_id,
                binding_schema_version="record-batch-binding-v1",
                content_batch_id="canonical-csv-" + "c" * 64,
                project_id=project_id,
                dataset_id=dataset_id,
                source_manifest_sha256=SOURCE_SHA256,
                registration_sha256="d" * 64,
                content_dataset_id="MATR",
                dataset_schema_version="cycle-record-v1",
                cell_id="MATR_b3c34",
                cutoff_cycle=cutoff_cycle,
                data_version="matr-v1",
                split_version="matr-cell-split-v1",
                feature_version="multichannel-cycle-v1",
                created_by_user_id=user_id,
                created_at=NOW,
            )
        )
    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    return _Fixture(
        sessions=sessions,
        context_service=context_service,
        project_ledger=SqlProjectAuditLedger(
            sessions,
            context_validator=context_service,
            clock=lambda: NOW,
        ),
        project_id=project_id,
        user_id=user_id,
        binding_id=binding_id,
        record_batch_id=record_batch_id,
    )


def _registration(*, cutoff_cycle: int = 20) -> CanonicalCsvBatchRegistration:
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="MATR",
            cell_id="MATR_b3c34",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="feishu://message/om/resource/file",
            source_sha256=SOURCE_SHA256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(
            cutoff_cycle=cutoff_cycle,
            feature_version="multichannel-cycle-v1",
        ),
        data_version="matr-v1",
        split_version="matr-cell-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="file",
                source_kind=SourceKind.OBSERVED,
                uri="feishu://message/om/resource/file",
                sha256=SOURCE_SHA256,
                description="Verified Feishu MATR canonical CSV",
                created_at=NOW,
            ),
        ),
    )


def test_resolver_returns_only_the_exact_frozen_project_batch() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    resolver = FeishuProjectRecordBatchResolver(fixture.sessions)

    resolved = resolver.resolve(context, _registration())

    assert resolved == fixture.record_batch_id


def test_resolver_rejects_non_frozen_or_ambiguous_project_batches() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    resolver = FeishuProjectRecordBatchResolver(fixture.sessions)
    with session_scope(fixture.sessions) as session:
        dataset = session.scalar(
            select(Dataset).where(Dataset.project_id == fixture.project_id)
        )
        assert dataset is not None
        dataset.status = DatasetStatus.DRAFT.value

    with pytest.raises(FeishuProjectModelRejected, match="FROZEN"):
        resolver.resolve(context, _registration())

    with session_scope(fixture.sessions) as session:
        dataset = session.scalar(
            select(Dataset).where(Dataset.project_id == fixture.project_id)
        )
        assert dataset is not None
        dataset.status = DatasetStatus.FROZEN.value
        duplicate_dataset_id = str(uuid4())
        session.add(
            Dataset(
                id=duplicate_dataset_id,
                project_id=fixture.project_id,
                name="Duplicate frozen MATR",
                data_version="matr-v1",
                schema_version="cycle-record-v1",
                status=DatasetStatus.FROZEN.value,
                manifest_uri="manifest://duplicate",
                manifest_sha256="e" * 64,
                created_at=NOW,
                frozen_at=NOW,
            )
        )
        session.add(
            RecordBatchBinding(
                id=str(uuid4()),
                binding_schema_version="record-batch-binding-v1",
                content_batch_id="canonical-csv-" + "f" * 64,
                project_id=fixture.project_id,
                dataset_id=duplicate_dataset_id,
                source_manifest_sha256=SOURCE_SHA256,
                registration_sha256="1" * 64,
                content_dataset_id="MATR",
                dataset_schema_version="cycle-record-v1",
                cell_id="MATR_b3c34",
                cutoff_cycle=20,
                data_version="matr-v1",
                split_version="matr-cell-split-v1",
                feature_version="multichannel-cycle-v1",
                created_by_user_id=fixture.user_id,
                created_at=NOW,
            )
        )

    with pytest.raises(FeishuProjectModelRejected, match="unique"):
        resolver.resolve(context, _registration())


def test_result_resolver_restores_only_a_durably_bound_feishu_project_result() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    result = fixture.project_ledger.register_result(
        context,
        ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            tool_version=ADVANCED_RUL_PREDICTION_TOOL_VERSION,
            model_version="reviewed-model-v1",
            data_version="matr-v1",
            feature_version="multichannel-cycle-v1",
            input_hash="9" * 64,
            values={"artifact_type": ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE},
            provenance=list(_registration().provenance),
            created_at=NOW,
        ),
    )
    job_id = str(uuid4())
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-project-result",
                event_type="im.message.receive_v1",
                payload_sha256="8" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="SUCCEEDED",
                job_stage="SUCCEEDED",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                analysis_result_id=result.result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )

    resolver = FeishuProjectResultResolver(
        session_factory=fixture.sessions,
        global_resolver=SqlAuditLedger(fixture.sessions, clock=lambda: NOW),
        context_service=fixture.context_service,
        project_ledger=fixture.project_ledger,
    )

    assert resolver.resolve_registered_result(result.result_id) == result
    with session_scope(fixture.sessions) as session:
        row = session.get(FeishuEventReceipt, session.scalar(select(FeishuEventReceipt.id)))
        assert row is not None
        row.sender_id = "ou-no-longer-authorized"

    with pytest.raises(ValueError, match="project ToolResult"):
        resolver.resolve_registered_result(result.result_id)


def test_result_resolver_restores_a_prepared_input_slot() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    prepared = fixture.project_ledger.register_result(
        context,
        ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
            tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
            model_version="advanced-input-transform-v1",
            data_version="matr-v1",
            feature_version="multichannel-cycle-v1",
            input_hash=sha256_canonical({"record_batch_id": fixture.record_batch_id}),
            values={"artifact": {"record_batch_id": fixture.record_batch_id}},
            provenance=list(_registration().provenance),
            created_at=NOW,
        ),
    )
    job_id = str(uuid4())
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-prepared-result",
                event_type="im.message.receive_v1",
                payload_sha256="7" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                prepared_input_result_id=prepared.result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )

    resolver = FeishuProjectResultResolver(
        session_factory=fixture.sessions,
        global_resolver=SqlAuditLedger(fixture.sessions, clock=lambda: NOW),
        context_service=fixture.context_service,
        project_ledger=fixture.project_ledger,
    )

    assert resolver.resolve_registered_result(prepared.result_id) == prepared


class _ProjectToolService:
    def __init__(self, ledger: SqlProjectAuditLedger) -> None:
        self.ledger = ledger
        self.calls: list[ToolInvocation] = []

    def _execute(self, invocation: ToolInvocation, *, context: object) -> ToolResult:
        self.calls.append(invocation)
        if invocation.tool_name is StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES:
            result = ToolResult(
                result_id=str(uuid4()),
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
                model_version="advanced-input-transform-v1",
                data_version="matr-v1",
                feature_version="multichannel-cycle-v1",
                input_hash=sha256_canonical(invocation.input_value),
                values={
                    "artifact_type": ADVANCED_INPUT_EVIDENCE_TYPE,
                    "artifact": {"record_batch_id": invocation.input_value["record_batch_id"]},
                },
                provenance=list(_registration().provenance),
                created_at=NOW,
            )
        elif invocation.tool_name is StandardToolName.PREDICT_CYCLE_LIFE:
            result = ToolResult(
                result_id=str(uuid4()),
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
                tool_version=ADVANCED_RUL_PREDICTION_TOOL_VERSION,
                model_version="reviewed-model-v1",
                data_version="matr-v1",
                feature_version="multichannel-cycle-v1",
                input_hash=sha256_canonical(invocation.input_value),
                values={
                    "artifact_type": ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
                    "artifact": {
                        "record_batch_id": fixture_record_batch_id(context),
                        "upstream_result_id": invocation.input_value[
                            "upstream_result_id"
                        ],
                        "cycle_life_prediction": {"predicted_cycle": 720.0},
                        "derived_remaining_cycles": 700.0,
                        "route_role": invocation.input_value["route_role"],
                    },
                },
                provenance=list(_registration().provenance),
                created_at=NOW,
            )
        else:
            assert invocation.tool_name is StandardToolName.PREDICT_SOH_TRAJECTORY
            result = ToolResult(
                result_id=str(uuid4()),
                tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
                tool_version=ADVANCED_SOH_PREDICTION_TOOL_VERSION,
                model_version="reviewed-soh-model-v1",
                data_version="matr-v1",
                feature_version="multichannel-cycle-v1",
                input_hash=sha256_canonical(invocation.input_value),
                values={
                    "artifact_type": ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
                    "artifact": {
                        "record_batch_id": fixture_record_batch_id(context),
                        "upstream_result_id": invocation.input_value[
                            "upstream_result_id"
                        ],
                        "dataset_id": "MATR",
                        "cell_id": "MATR_b3c34",
                        "cutoff_cycle": 20,
                        "prediction_cycles": [21, 500],
                        "predicted_soh": [0.99, 0.88],
                        "horizon_end_cycle": 500,
                        "route_role": invocation.input_value["route_role"],
                    },
                },
                uncertainty={
                    "finite_horizon_only": True,
                    "conformal_interval_included": False,
                },
                provenance=list(_registration().provenance),
                created_at=NOW,
            )
        return result

    def invoke_in_project(self, invocation: ToolInvocation, *, context: object) -> ToolResult:
        result = self._execute(invocation, context=context)
        return self.ledger.register_result(context, result)  # type: ignore[arg-type]


class _AtomicProjectToolService(_ProjectToolService):
    def execute_in_project_unregistered(
        self,
        invocation: ToolInvocation,
        *,
        context: object,
    ) -> ToolResult:
        return self._execute(invocation, context=context)


def fixture_record_batch_id(context: object) -> str:
    project_id = context.project_id
    return _RECORD_BATCH_IDS[project_id]


_RECORD_BATCH_IDS: dict[str, str] = {}


@pytest.mark.parametrize(
    ("cutoff_cycle", "expected_role"),
    (
        (20, AdvancedModelRouteRole.DEFAULT),
        (50, AdvancedModelRouteRole.POINT_ACCURACY),
        (100, AdvancedModelRouteRole.POINT_ACCURACY),
        (150, AdvancedModelRouteRole.POINT_ACCURACY),
    ),
)
def test_executor_uses_project_tools_and_registers_a_fixed_audited_report(
    cutoff_cycle: int,
    expected_role: AdvancedModelRouteRole,
) -> None:
    fixture = _fixture(cutoff_cycle=cutoff_cycle)
    _RECORD_BATCH_IDS[fixture.project_id] = fixture.record_batch_id
    service = _ProjectToolService(fixture.project_ledger)
    executor = FeishuProjectModelExecutor(
        context_service=fixture.context_service,
        batch_resolver=FeishuProjectRecordBatchResolver(fixture.sessions),
        project_tool_service=service,
        project_ledger=fixture.project_ledger,
        clock=lambda: NOW,
    )

    execution = executor.execute(
        _Job(task_type=StandardToolName.PREDICT_CYCLE_LIFE),
        _registration(cutoff_cycle=cutoff_cycle),
    )

    assert execution.record_batch_id == fixture.record_batch_id
    assert [call.tool_name for call in service.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_CYCLE_LIFE,
    ]
    assert service.calls[0].input_value == {"record_batch_id": fixture.record_batch_id}
    assert service.calls[1].input_value == {
        "upstream_result_id": execution.prepared_input_result_id,
        "route_role": expected_role.value,
    }
    assert execution.analysis_result.tool_version == ADVANCED_RUL_PREDICTION_TOOL_VERSION
    assert execution.report_result.tool_version == AUDITED_REPORT_TOOL_VERSION
    assert execution.report_result.values["upstream_result_ids"] == [
        execution.analysis_result.result_id
    ]
    with session_scope(fixture.sessions) as session:
        report_binding = session.get(
            ProjectToolResultBindingRecord,
            execution.report_result.result_id,
        )
        assert report_binding is not None
        assert report_binding.binding_schema_version == "project-tool-result-binding-v3"
        assert report_binding.feishu_binding_id == fixture.binding_id


def test_executor_runs_project_bound_finite_soh_with_explicit_mean_role() -> None:
    fixture = _fixture(cutoff_cycle=20)
    _RECORD_BATCH_IDS[fixture.project_id] = fixture.record_batch_id
    service = _ProjectToolService(fixture.project_ledger)
    executor = FeishuProjectModelExecutor(
        context_service=fixture.context_service,
        batch_resolver=FeishuProjectRecordBatchResolver(fixture.sessions),
        project_tool_service=service,
        project_ledger=fixture.project_ledger,
        clock=lambda: NOW,
    )

    execution = executor.execute(
        _Job(task_type=StandardToolName.PREDICT_SOH_TRAJECTORY),
        _registration(cutoff_cycle=20),
    )

    assert [call.tool_name for call in service.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
    ]
    assert service.calls[1].input_value == {
        "upstream_result_id": execution.prepared_input_result_id,
        "route_role": AdvancedModelRouteRole.MEAN_ACCURACY.value,
    }
    assert execution.analysis_result.tool_name == "predict_soh_trajectory"
    assert execution.analysis_result.tool_version == ADVANCED_SOH_PREDICTION_TOOL_VERSION
    assert execution.report_result.values["upstream_result_ids"] == [
        execution.analysis_result.result_id
    ]


def test_atomic_executor_commits_each_slot_and_marks_the_execution_complete() -> None:
    fixture = _fixture(cutoff_cycle=20)
    _RECORD_BATCH_IDS[fixture.project_id] = fixture.record_batch_id
    service = _AtomicProjectToolService(fixture.project_ledger)
    executor = FeishuProjectModelExecutor(
        context_service=fixture.context_service,
        batch_resolver=FeishuProjectRecordBatchResolver(fixture.sessions),
        project_tool_service=service,
        project_ledger=fixture.project_ledger,
        clock=lambda: NOW,
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-atomic-executor",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )

    execution = executor.execute(
        _Job(
            task_type=StandardToolName.PREDICT_SOH_TRAJECTORY,
            job_id=job_id,
            run_id=job_id,
        ),
        _registration(cutoff_cycle=20),
        claim_token=claim_token,
    )

    assert execution.slots_committed is True
    assert [call.tool_name for call in service.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
    ]
    with session_scope(fixture.sessions) as session:
        receipt = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert receipt is not None
        assert receipt.prepared_input_result_id == execution.prepared_input_result_id
        assert receipt.analysis_result_id == execution.analysis_result.result_id
        assert receipt.report_result_id == execution.report_result.result_id


class _InjectedProcessCrash(BaseException):
    pass


@pytest.mark.parametrize("crash_slot", ("PREPARED", "ANALYSIS", "REPORT"))
def test_atomic_executor_resumes_after_each_committed_slot_without_rerunning(
    crash_slot: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(cutoff_cycle=20)
    _RECORD_BATCH_IDS[fixture.project_id] = fixture.record_batch_id
    service = _AtomicProjectToolService(fixture.project_ledger)
    executor = FeishuProjectModelExecutor(
        context_service=fixture.context_service,
        batch_resolver=FeishuProjectRecordBatchResolver(fixture.sessions),
        project_tool_service=service,
        project_ledger=fixture.project_ledger,
        clock=lambda: NOW,
    )
    job_id = str(uuid4())
    first_claim = "f" * 64
    second_claim = "a" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id=f"evt-crash-after-{crash_slot.casefold()}",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=first_claim,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    original_commit = fixture.project_ledger.commit_feishu_result_slot
    original_report = executor._audited_report_unregistered
    report_calls: list[str] = []

    def crash_after_commit(**kwargs: object) -> ToolResult:
        committed = original_commit(**kwargs)  # type: ignore[arg-type]
        if kwargs["slot"] == crash_slot:
            raise _InjectedProcessCrash
        return committed

    def count_report(
        analysis: ToolResult,
        **kwargs: object,
    ) -> ToolResult:
        report_calls.append(analysis.result_id)
        return original_report(analysis, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        fixture.project_ledger,
        "commit_feishu_result_slot",
        crash_after_commit,
    )
    monkeypatch.setattr(executor, "_audited_report_unregistered", count_report)
    first_job = _Job(
        task_type=StandardToolName.PREDICT_SOH_TRAJECTORY,
        job_id=job_id,
        run_id=job_id,
    )

    with pytest.raises(_InjectedProcessCrash):
        executor.execute(
            first_job,
            _registration(cutoff_cycle=20),
            claim_token=first_claim,
        )

    with session_scope(fixture.sessions) as session:
        receipt = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert receipt is not None
        assert receipt.prepared_input_result_id is not None
        if crash_slot in {"ANALYSIS", "REPORT"}:
            assert receipt.analysis_result_id is not None
        else:
            assert receipt.analysis_result_id is None
        if crash_slot == "REPORT":
            assert receipt.report_result_id is not None
        else:
            assert receipt.report_result_id is None
        receipt.job_claim_token = second_claim
        receipt.job_attempt_count = 2
        receipt.job_lease_expires_at = NOW.replace(hour=13)
        recovery_job = _Job(
            task_type=StandardToolName.PREDICT_SOH_TRAJECTORY,
            job_id=job_id,
            run_id=job_id,
            prepared_input_result_id=receipt.prepared_input_result_id,
            analysis_result_id=receipt.analysis_result_id,
            report_result_id=receipt.report_result_id,
        )

    monkeypatch.setattr(
        fixture.project_ledger,
        "commit_feishu_result_slot",
        original_commit,
    )
    execution = executor.execute(
        recovery_job,
        _registration(cutoff_cycle=20),
        claim_token=second_claim,
    )

    assert execution.slots_committed is True
    assert [call.tool_name for call in service.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
    ]
    assert report_calls == [execution.analysis_result.result_id]
    with session_scope(fixture.sessions) as session:
        receipt = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert receipt is not None
        assert receipt.prepared_input_result_id == execution.prepared_input_result_id
        assert receipt.analysis_result_id == execution.analysis_result.result_id
        assert receipt.report_result_id == execution.report_result.result_id


def test_feishu_project_result_slot_commit_is_idempotent_and_fenced() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-atomic-slot",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    prepared = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version="advanced-input-transform-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={"artifact": {"record_batch_id": fixture.record_batch_id}},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    first = fixture.project_ledger.commit_feishu_result_slot(
        context=context,
        job_id=job_id,
        claim_token=claim_token,
        slot="PREPARED",
        expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
        run_id=job_id,
        chat_id="oc-approved",
        sender_id="ou-approved",
        result=prepared,
    )
    second = fixture.project_ledger.commit_feishu_result_slot(
        context=context,
        job_id=job_id,
        claim_token=claim_token,
        slot="PREPARED",
        expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
        run_id=job_id,
        chat_id="oc-approved",
        sender_id="ou-approved",
        result=prepared,
    )

    assert first == prepared
    assert second == prepared
    with session_scope(fixture.sessions) as session:
        row = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert row is not None
        assert row.prepared_input_result_id == prepared.result_id
        assert (
            session.scalar(
                select(ProjectToolResultBindingRecord).where(
                    ProjectToolResultBindingRecord.result_id == prepared.result_id
                )
            )
            is not None
        )

    with pytest.raises(AuditLedgerError, match="stale"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token="a" * 64,
            slot="PREPARED",
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=prepared.model_copy(update={"result_id": str(uuid4())}),
        )


def test_feishu_project_result_slot_uses_the_ledger_commit_time_for_the_lease() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-expired-atomic-slot",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    ledger = SqlProjectAuditLedger(
        fixture.sessions,
        context_validator=fixture.context_service,
        clock=lambda: NOW.replace(hour=14),
    )
    prepared = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version="advanced-input-transform-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={"artifact": {"record_batch_id": fixture.record_batch_id}},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    with pytest.raises(AuditLedgerError, match="stale"):
        ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token=claim_token,
            slot="PREPARED",
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=prepared,
        )


@pytest.mark.parametrize(
    ("slot", "tool_name"),
    (
        ("PREPARED", StandardToolName.PREDICT_CYCLE_LIFE.value),
        ("ANALYSIS", StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value),
        ("REPORT", StandardToolName.PREDICT_CYCLE_LIFE.value),
    ),
)
def test_feishu_project_result_slot_rejects_the_wrong_tool(
    slot: str,
    tool_name: str,
) -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id=f"evt-wrong-{slot.casefold()}",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version="test-tool-v1",
        model_version="test-model-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    with pytest.raises(AuditLedgerError, match="tool"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token=claim_token,
            slot=slot,
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=result,
        )


def test_feishu_project_result_slot_rejects_non_hex_claim_and_non_feishu_origin() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-non-feishu-slot",
                event_type="aily.analysis.requested",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="AILY",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token="g" * 64,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    prepared = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version="advanced-input-transform-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={"artifact": {"record_batch_id": fixture.record_batch_id}},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    with pytest.raises(AuditLedgerError, match="claim token"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token="g" * 64,
            slot="PREPARED",
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=prepared,
        )

    with session_scope(fixture.sessions) as session:
        receipt = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert receipt is not None
        receipt.job_claim_token = "f" * 64

    with pytest.raises(AuditLedgerError, match="stale"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token="f" * 64,
            slot="PREPARED",
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=prepared,
        )


def test_feishu_project_result_slot_accepts_aily_job_with_exact_feishu_anchor() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    source_job_id = str(uuid4())
    aily_job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add_all(
            (
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="evt-aily-project-source",
                    event_type="im.message.receive_v1",
                    payload_sha256="a" * 64,
                    status="PROCESSED",
                    attempt_count=1,
                    received_at=NOW,
                    processed_at=NOW,
                    job_id=source_job_id,
                    job_origin="FEISHU",
                    job_status="SUCCEEDED",
                    job_stage="SUCCEEDED",
                    task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                    run_id=source_job_id,
                    chat_id="oc-approved",
                    sender_id="ou-approved",
                    receive_id_type="chat_id",
                    event_time=NOW,
                    record_batch_id=fixture.record_batch_id,
                    cell_reference="MATR_b3c34",
                    input_file_sha256="a" * 64,
                    validation_result_id=str(uuid4()),
                    job_attempt_count=1,
                    job_created_at=NOW,
                    job_updated_at=NOW,
                    job_completed_at=NOW,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="aily:anchored-project-slot",
                    event_type="aily.analysis_task.create_v2",
                    payload_sha256="b" * 64,
                    status="PROCESSED",
                    attempt_count=1,
                    received_at=NOW,
                    processed_at=NOW,
                    job_id=aily_job_id,
                    job_origin="AILY",
                    job_request_sha256="b" * 64,
                    source_job_id=source_job_id,
                    job_status="RUNNING",
                    job_stage="RUNNING_TOOL",
                    task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                    run_id=aily_job_id,
                    chat_id="oc-approved",
                    sender_id="ou-approved",
                    receive_id_type="chat_id",
                    event_time=NOW,
                    job_claim_token=claim_token,
                    job_attempt_count=1,
                    job_lease_expires_at=NOW.replace(hour=13),
                    record_batch_id=fixture.record_batch_id,
                    cell_reference="MATR_b3c34",
                    input_file_sha256="a" * 64,
                    job_created_at=NOW,
                    job_updated_at=NOW,
                ),
            )
        )
    prepared = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version="advanced-input-transform-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={"artifact": {"record_batch_id": fixture.record_batch_id}},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    committed = fixture.project_ledger.commit_feishu_result_slot(
        context=context,
        job_id=aily_job_id,
        claim_token=claim_token,
        slot="PREPARED",
        expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
        run_id=aily_job_id,
        chat_id="oc-approved",
        sender_id="ou-approved",
        result=prepared,
    )

    assert committed == prepared
    with session_scope(fixture.sessions) as session:
        receipt = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == aily_job_id)
        )
        assert receipt is not None
        assert receipt.prepared_input_result_id == prepared.result_id


def test_feishu_project_result_slot_rejects_unsupported_analysis_task() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-unsupported-slot-task",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type="made_up_analysis_task",
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    result = ToolResult(
        result_id=str(uuid4()),
        tool_name="made_up_analysis_task",
        tool_version="test-tool-v1",
        model_version="test-model-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )

    with pytest.raises(AuditLedgerError, match="task"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token=claim_token,
            slot="ANALYSIS",
            expected_task="made_up_analysis_task",
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=result,
        )

def test_feishu_project_result_slot_retry_requires_the_exact_project_binding() -> None:
    fixture = _fixture()
    context = fixture.context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    job_id = str(uuid4())
    claim_token = "f" * 64
    with session_scope(fixture.sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-missing-binding-slot",
                event_type="im.message.receive_v1",
                payload_sha256="e" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type=StandardToolName.PREDICT_CYCLE_LIFE.value,
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                job_claim_token=claim_token,
                job_attempt_count=1,
                job_lease_expires_at=NOW.replace(hour=13),
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    prepared = ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=PREPARE_ADVANCED_INPUT_TOOL_VERSION,
        model_version="advanced-input-transform-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="1" * 64,
        values={"artifact": {"record_batch_id": fixture.record_batch_id}},
        provenance=list(_registration().provenance),
        created_at=NOW,
    )
    fixture.project_ledger.commit_feishu_result_slot(
        context=context,
        job_id=job_id,
        claim_token=claim_token,
        slot="PREPARED",
        expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
        run_id=job_id,
        chat_id="oc-approved",
        sender_id="ou-approved",
        result=prepared,
    )
    with session_scope(fixture.sessions) as session:
        binding = session.get(ProjectToolResultBindingRecord, prepared.result_id)
        assert binding is not None
        session.delete(binding)

    with pytest.raises(AuditLedgerError, match="binding"):
        fixture.project_ledger.commit_feishu_result_slot(
            context=context,
            job_id=job_id,
            claim_token=claim_token,
            slot="PREPARED",
            expected_task=StandardToolName.PREDICT_CYCLE_LIFE.value,
            run_id=job_id,
            chat_id="oc-approved",
            sender_id="ou-approved",
            result=prepared,
        )
