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
from quanxin_life.audit import SqlAuditLedger, SqlProjectAuditLedger
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

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
SOURCE_SHA256 = "a" * 64


@dataclass(frozen=True, slots=True)
class _Job:
    task_type: object
    chat_id: str | None = "oc-approved"
    sender_id: str | None = "ou-approved"


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


class _ProjectToolService:
    def __init__(self, ledger: SqlProjectAuditLedger) -> None:
        self.ledger = ledger
        self.calls: list[ToolInvocation] = []

    def invoke_in_project(self, invocation: ToolInvocation, *, context: object) -> ToolResult:
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
        else:
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
                        "cycle_life_prediction": {"predicted_cycle": 720.0},
                        "derived_remaining_cycles": 700.0,
                        "route_role": invocation.input_value["route_role"],
                    },
                },
                provenance=list(_registration().provenance),
                created_at=NOW,
            )
        return self.ledger.register_result(context, result)  # type: ignore[arg-type]


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
