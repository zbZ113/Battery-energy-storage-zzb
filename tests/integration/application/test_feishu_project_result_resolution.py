from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, select

from quanxin_life.application.feishu_project_models import (
    FeishuProjectResultResolver,
)
from quanxin_life.application.invocation_context import ProjectInvocationContextService
from quanxin_life.audit import SqlAuditLedger, SqlProjectAuditLedger
from quanxin_life.core import (
    ProjectStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuJobDispatchReceipt,
    SqlAlchemyFeishuJobStore,
    SqlAlchemyFeishuSiblingJobService,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import session_scope
from quanxin_life.persistence.models import (
    FeishuBindingRow,
    FeishuEventReceipt,
    Project,
    User,
)

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


class _Queue:
    def enqueue(self, *, job_id: str) -> FeishuJobDispatchReceipt:
        return FeishuJobDispatchReceipt(job_id=job_id, task_id=f"task-{job_id}")


def _result(*, tool_name: str) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version=f"{tool_name}-test-v1",
        model_version="reviewed-test-model-v1",
        data_version="matr-v1",
        feature_version="multichannel-cycle-v1",
        input_hash="a" * 64,
        values={"artifact": {"source": "registered-test-result"}},
        provenance=[
            ProvenanceRecord(
                source_id="matr-batch",
                source_kind=SourceKind.OBSERVED,
                uri="record-batch://matr-batch",
                sha256="b" * 64,
                description="Reviewed test batch",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_resolver_reads_global_and_exact_feishu_project_results() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    binding_id = str(uuid4())
    job_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add_all(
            (
                User(
                    id=user_id,
                    username="feishu-results@example.test",
                    credential_hash="not-used-by-feishu",
                    must_change_credential=False,
                    role=UserRole.ADMIN.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Project(
                    id=project_id,
                    owner_user_id=user_id,
                    name="Feishu result project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
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
                ),
            )
        )
    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    global_ledger = SqlAuditLedger(sessions, clock=lambda: NOW)
    project_ledger = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )
    validation = global_ledger.register_result(_result(tool_name="validate_battery_data"))
    context = context_service.resolve_feishu(
        chat_id="oc-approved",
        sender_open_id="ou-approved",
    )
    analysis = project_ledger.register_result(
        context,
        _result(tool_name="predict_cycle_life"),
    )
    aily_analysis = project_ledger.register_result(
        context,
        _result(tool_name="predict_soh_trajectory"),
    )
    with session_scope(sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-results",
                event_type="im.message.receive_v1",
                payload_sha256="c" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=job_id,
                job_origin="FEISHU",
                job_status="RUNNING",
                job_stage="RUNNING_TOOL",
                task_type="predict_cycle_life",
                run_id=job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id="canonical-csv-" + "d" * 64,
                cell_reference="MATR_b3c34",
                input_file_sha256="e" * 64,
                validation_result_id=validation.result_id,
                analysis_result_id=analysis.result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
            )
        )
    jobs = SqlAlchemyFeishuJobStore(sessions, lease_seconds=60)
    sibling_id = SqlAlchemyFeishuSiblingJobService(
        jobs,
        queue=_Queue(),
        clock=lambda: NOW,
    ).stage_sibling(
        source_job_id=job_id,
        task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
    )
    assert jobs.get(sibling_id).validation_result_id is None
    aily_job_id = str(uuid4())
    with session_scope(sessions) as session:
        source = session.scalar(
            select(FeishuEventReceipt).where(FeishuEventReceipt.job_id == job_id)
        )
        assert source is not None
        source.job_status = "SUCCEEDED"
        source.job_stage = "SUCCEEDED"
        source.job_completed_at = NOW
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="aily:anchored-project-result",
                event_type="aily.analysis_task.create_v2",
                payload_sha256="f" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=aily_job_id,
                job_origin="AILY",
                job_request_sha256="f" * 64,
                source_job_id=job_id,
                job_status="SUCCEEDED",
                job_stage="SUCCEEDED",
                task_type="predict_soh_trajectory",
                run_id=aily_job_id,
                chat_id="oc-approved",
                sender_id="ou-approved",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id="canonical-csv-" + "d" * 64,
                cell_reference="MATR_b3c34",
                input_file_sha256="e" * 64,
                analysis_result_id=aily_analysis.result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
    resolver = FeishuProjectResultResolver(
        session_factory=sessions,
        global_resolver=global_ledger,
        project_ledger=project_ledger,
        context_service=context_service,
    )

    assert resolver.resolve_registered_result(validation.result_id) == validation
    assert resolver.resolve_registered_result(analysis.result_id) == analysis
    assert resolver.resolve_registered_result(aily_analysis.result_id) == aily_analysis

    with session_scope(sessions) as session:
        original = session.get(FeishuBindingRow, binding_id)
        assert original is not None
        original.status = "INACTIVE"

    with pytest.raises(ValueError, match="active Feishu binding"):
        resolver.resolve_registered_result(validation.result_id)
    with pytest.raises(ValueError, match="active Feishu binding"):
        resolver.resolve_registered_result(analysis.result_id)
    with pytest.raises(ValueError, match="active Feishu binding"):
        resolver.resolve_registered_result(aily_analysis.result_id)

    with session_scope(sessions) as session:
        session.add(
            FeishuBindingRow(
                id=str(uuid4()),
                project_id=project_id,
                chat_id="oc-approved",
                bitable_app_token=None,
                bitable_table_id=None,
                user_open_id_map_json={user_id: "ou-approved"},
                binding_version="feishu-binding-v2",
                status="ACTIVE",
                created_at=NOW,
            )
        )

    with pytest.raises(ValueError, match="exact Feishu binding"):
        resolver.resolve_registered_result(analysis.result_id)
    assert resolver.resolve_registered_result(validation.result_id) == validation


def _anchored_aily_global_result(
    *,
    task_type: str,
    job_status: str,
    tool_name: str,
    bind_as_validation: bool = False,
) -> tuple[FeishuProjectResultResolver, ToolResult]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    user_id = str(uuid4())
    project_id = str(uuid4())
    root_job_id = str(uuid4())
    aily_job_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add_all(
            (
                User(
                    id=user_id,
                    username=f"aily-result-{aily_job_id}@example.test",
                    credential_hash="not-used-by-feishu",
                    must_change_credential=False,
                    role=UserRole.ADMIN.value,
                    status=UserStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                Project(
                    id=project_id,
                    owner_user_id=user_id,
                    name="Anchored Aily result project",
                    status=ProjectStatus.ACTIVE.value,
                    created_at=NOW,
                    updated_at=NOW,
                ),
                FeishuBindingRow(
                    id=str(uuid4()),
                    project_id=project_id,
                    chat_id="oc-anchored-aily",
                    bitable_app_token=None,
                    bitable_table_id=None,
                    user_open_id_map_json={user_id: "ou-anchored-aily"},
                    binding_version="feishu-binding-v1",
                    status="ACTIVE",
                    created_at=NOW,
                ),
            )
        )
    context_service = ProjectInvocationContextService(sessions, clock=lambda: NOW)
    global_ledger = SqlAuditLedger(sessions, clock=lambda: NOW)
    project_ledger = SqlProjectAuditLedger(
        sessions,
        context_validator=context_service,
        clock=lambda: NOW,
    )
    root_validation = global_ledger.register_result(
        _result(tool_name="validate_battery_data")
    )
    target = global_ledger.register_result(_result(tool_name=tool_name))
    with session_scope(sessions) as session:
        session.add_all(
            (
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id=f"evt-upload-{root_job_id}",
                    event_type="im.message.receive_v1",
                    payload_sha256="c" * 64,
                    status="PROCESSED",
                    attempt_count=1,
                    received_at=NOW,
                    processed_at=NOW,
                    job_id=root_job_id,
                    job_origin="FEISHU",
                    job_status="SUCCEEDED",
                    job_stage="SUCCEEDED",
                    task_type="predict_cycle_life",
                    run_id=root_job_id,
                    chat_id="oc-anchored-aily",
                    sender_id="ou-anchored-aily",
                    receive_id_type="chat_id",
                    event_time=NOW,
                    record_batch_id="canonical-csv-" + "d" * 64,
                    cell_reference="MATR_b3c34",
                    input_file_sha256="e" * 64,
                    validation_result_id=root_validation.result_id,
                    job_attempt_count=1,
                    job_created_at=NOW,
                    job_updated_at=NOW,
                    job_completed_at=NOW,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id=f"aily:anchored-{aily_job_id}",
                    event_type="aily.analysis_task.create_v2",
                    payload_sha256="f" * 64,
                    status="PROCESSED",
                    attempt_count=1,
                    received_at=NOW,
                    processed_at=NOW,
                    job_id=aily_job_id,
                    job_origin="AILY",
                    job_request_sha256="f" * 64,
                    source_job_id=root_job_id,
                    job_status=job_status,
                    job_stage=job_status,
                    task_type=task_type,
                    run_id=aily_job_id,
                    chat_id="oc-anchored-aily",
                    sender_id="ou-anchored-aily",
                    receive_id_type="chat_id",
                    event_time=NOW,
                    record_batch_id="canonical-csv-" + "d" * 64,
                    cell_reference="MATR_b3c34",
                    input_file_sha256="e" * 64,
                    validation_result_id=(
                        target.result_id if bind_as_validation else None
                    ),
                    analysis_result_id=(
                        None if bind_as_validation else target.result_id
                    ),
                    job_attempt_count=1,
                    job_created_at=NOW,
                    job_updated_at=NOW,
                    job_completed_at=NOW,
                ),
            )
        )
    return (
        FeishuProjectResultResolver(
            session_factory=sessions,
            global_resolver=global_ledger,
            project_ledger=project_ledger,
            context_service=context_service,
        ),
        target,
    )


def test_resolver_reads_feishu_anchored_aily_scenario_global_result() -> None:
    resolver, result = _anchored_aily_global_result(
        task_type="compare_operation_scenarios",
        job_status="SUCCEEDED",
        tool_name="compare_operation_scenarios",
    )

    assert resolver.resolve_registered_result(result.result_id) == result


def test_resolver_reads_rejected_feishu_anchored_aily_validation_result() -> None:
    resolver, result = _anchored_aily_global_result(
        task_type="predict_cycle_life",
        job_status="REJECTED",
        tool_name="validate_battery_data",
        bind_as_validation=True,
    )

    assert resolver.resolve_registered_result(result.result_id) == result
