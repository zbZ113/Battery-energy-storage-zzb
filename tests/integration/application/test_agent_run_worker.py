from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field
from sqlalchemy import delete, event, select

from quanxin_life.agents.supervisor import (
    SupervisorPlanningRequest,
    SupervisorPlanningResult,
)
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.agent_run_invocation import (
    PersistentAgentRunInvocationResolver,
)
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.audit import AuditLedger, SqlProjectAuditLedger
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentFailurePolicy,
    AgentIntent,
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
    AgentRunStatus,
    AgentStepStatus,
    ApprovalStatus,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import (
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
    User,
)
from quanxin_life.tools import (
    StandardToolName,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

NOW = datetime(2026, 7, 16, 13, 0, tzinfo=UTC)


class _FeaturesInput(ContractModel):
    record_batch_id: str = Field(min_length=1)


class _CanonicalFeaturesInput(ContractModel):
    record_batch_id: str = Field(min_length=1)
    input_schema_version: str = "canonical-features-v1"


class _PredictionInput(ContractModel):
    upstream_result_id: str = Field(min_length=1)


class _DatasetArtifactResolver:
    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        del run_id, project_id, dataset_id
        return "verified-worker-record-batch-v1"

    def resolve_context(self, *, run_id: str, project_id: str, reference: str) -> object:
        del run_id, project_id
        raise KeyError(reference)


class _Planner:
    def __init__(
        self,
        *,
        approval: bool = False,
        failure_policy: AgentFailurePolicy = AgentFailurePolicy.STOP,
    ) -> None:
        self.approval = approval
        self.failure_policy = failure_policy

    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: object,
    ) -> SupervisorPlanningResult:
        del available_tools
        intent = AgentIntent(
            intent_id=str(uuid4()),
            project_id=request.project_id,
            goal=request.user_goal,
            dataset_ids=request.dataset_ids,
            requested_outputs=request.requested_outputs,
            created_at=NOW,
        )
        steps = (
            AgentPlanStep(
                step_id="features",
                role=AgentRole.DATA_QUALITY,
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                input_references={"record_batch_id": "intent.dataset_ids[0]"},
                requires_approval=self.approval,
                failure_policy=self.failure_policy,
            ),
            AgentPlanStep(
                step_id="predict",
                role=AgentRole.LIFETIME,
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
                depends_on=("features",),
                input_references={"upstream_result_id": "step.features.result_id"},
                failure_policy=AgentFailurePolicy.STOP,
            ),
        )
        plan = AgentPlan.build(
            plan_version="worker-test-plan-v1",
            intent_id=intent.intent_id,
            steps=steps,
            planning_mode=AgentPlanningMode.FIXED_FALLBACK,
            created_at=NOW,
        )
        return SupervisorPlanningResult(intent=intent, plan=plan)


class _CountingTools:
    features_input_model: type[ContractModel] = _FeaturesInput

    def __init__(self, *, fail_feature_attempts: int = 0) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail_feature_attempts = fail_feature_attempts

    def service(self) -> ToolInvocationService:
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                tool_version="worker-tool-v1",
                input_model=self.features_input_model,
                executor=self._features,
            )
        )
        registry.register(
            ToolDefinition(
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
                tool_version="worker-tool-v1",
                input_model=_PredictionInput,
                executor=self._prediction,
            )
        )
        return ToolInvocationService(registry=registry, audit_ledger=AuditLedger())

    def _features(self, value: _FeaturesInput) -> ToolResult:
        payload = value.model_dump(mode="json")
        self.calls.append((StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value, payload))
        if self.fail_feature_attempts > 0:
            self.fail_feature_attempts -= 1
            raise RuntimeError("test-only transient tool failure")
        return _tool_result(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES, payload)

    def _prediction(self, value: _PredictionInput) -> ToolResult:
        payload = value.model_dump(mode="json")
        self.calls.append((StandardToolName.PREDICT_CYCLE_LIFE.value, payload))
        return _tool_result(StandardToolName.PREDICT_CYCLE_LIFE, payload)

    def project_service(
        self,
        session_factory: SessionFactory,
    ) -> tuple[ToolInvocationService, PersistentAgentRunInvocationResolver]:
        context_service = ProjectInvocationContextService(
            session_factory,
            clock=lambda: NOW,
        )
        resolver = PersistentAgentRunInvocationResolver(
            session_factory,
            context_service=context_service,
            clock=lambda: NOW,
        )
        registry = ToolRegistry(project_context_validator=context_service)
        registry.register(
            ToolDefinition(
                tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
                tool_version="worker-tool-v1",
                input_model=self.features_input_model,
                executor=None,
                execution_scope=ToolExecutionScope.PROJECT,
                project_executor=lambda value, context: self._features(value),
            )
        )
        registry.register(
            ToolDefinition(
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
                tool_version="worker-tool-v1",
                input_model=_PredictionInput,
                executor=None,
                execution_scope=ToolExecutionScope.PROJECT,
                project_executor=lambda value, context: self._prediction(value),
            )
        )
        ledger = SqlProjectAuditLedger(
            session_factory,
            context_validator=context_service,
            clock=lambda: NOW,
        )
        return (
            ToolInvocationService(
                registry=registry,
                project_audit_ledger=ledger,
                agent_run_invocation_validator=resolver,
            ),
            resolver,
        )


class _CanonicalizingCountingTools(_CountingTools):
    features_input_model = _CanonicalFeaturesInput


def _tool_result(tool_name: StandardToolName, input_value: dict[str, object]) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name.value,
        tool_version="worker-tool-v1",
        model_version="worker-model-v1",
        data_version="worker-data-v1",
        feature_version="worker-features-v1",
        input_hash=sha256_canonical(input_value),
        values={"execution_status": "computed"},
        provenance=[
            ProvenanceRecord(
                source_id=f"source-{tool_name.value}",
                source_kind=SourceKind.OBSERVED,
                uri=f"test://{tool_name.value}",
                sha256=sha256_canonical({"tool": tool_name.value}),
                description="Worker integration test provenance",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _principal(user_id: str) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"member-{user_id[:8]}@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )


def _setup(
    tmp_path: Path,
    *,
    approval: bool = False,
    failure_policy: AgentFailurePolicy = AgentFailurePolicy.STOP,
    fail_feature_attempts: int = 0,
) -> tuple[
    AgentRunService,
    AuthPrincipal,
    SessionFactory,
    str,
    _CountingTools,
]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'agent-worker.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    member = _principal(str(uuid4()))
    with session_factory.begin() as session:
        session.add(
            User(
                id=member.user_id,
                username=member.username,
                credential_hash="$argon2id$worker-test",
                must_change_credential=False,
                role=member.role.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=member.session_id,
                user_id=member.user_id,
                token_hash=uuid4().hex + uuid4().hex,
                status="ACTIVE",
                created_at=NOW,
                expires_at=NOW + timedelta(days=1),
            )
        )
    project_service = ProjectService(session_factory)
    dataset_service = DatasetService(session_factory)
    project = project_service.create_project(member, name="worker project", now=NOW)
    dataset = dataset_service.create_dataset(
        member,
        project_id=project.project_id,
        name="worker dataset",
        data_version="worker-data-v1",
        schema_version="canonical-v1",
        now=NOW,
    )
    dataset = dataset_service.freeze_dataset(member, dataset.dataset_id, now=NOW)
    run_service = AgentRunService(
        session_factory,
        planner=_Planner(approval=approval, failure_policy=failure_policy),
    )
    run = run_service.create_run(
        member,
        request=SupervisorPlanningRequest(
            project_id=project.project_id,
            user_goal="run the trusted lifetime workflow",
            dataset_ids=(dataset.dataset_id,),
            requested_outputs=("cycle_life",),
        ),
        idempotency_key=f"worker-run-{uuid4()}",
        available_tools=(
            StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,
            StandardToolName.PREDICT_CYCLE_LIFE,
        ),
        now=NOW,
    )
    tools = _CountingTools(fail_feature_attempts=fail_feature_attempts)
    return run_service, member, session_factory, run.run_id, tools


def _worker(
    run_service: AgentRunService,
    session_factory: SessionFactory,
    tools: _CountingTools,
    *,
    now: datetime = NOW,
):
    from quanxin_life.application.agent_run_execution import AgentRunExecutionWorker

    return AgentRunExecutionWorker(
        session_factory,
        run_service=run_service,
        tool_service=tools.service(),
        context_resolver=_DatasetArtifactResolver(),
        clock=lambda: now,
    )


def _project_worker(
    run_service: AgentRunService,
    session_factory: SessionFactory,
    tools: _CountingTools,
):
    from quanxin_life.application.agent_run_execution import AgentRunExecutionWorker

    with session_factory.begin() as session:
        dispatch = session.scalar(select(AgentRunDispatch))
        assert dispatch is not None
        dispatch.status = "DISPATCHED"
        dispatch.task_id = "project-worker-task"
    service, resolver = tools.project_service(session_factory)
    return AgentRunExecutionWorker(
        session_factory,
        run_service=run_service,
        tool_service=service,
        context_resolver=_DatasetArtifactResolver(),
        agent_run_invocation_resolver=resolver,
        clock=lambda: NOW,
    )


def test_worker_executes_each_step_once_and_resumes_duplicate_delivery_without_rerun(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    worker = _worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    first = worker.execute(run_id=run_id, plan_hash=plan_hash)
    repeated = worker.execute(run_id=run_id, plan_hash=plan_hash)

    assert first.status is AgentRunStatus.COMPLETED
    assert repeated == first
    assert [name for name, _ in tools.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        StandardToolName.PREDICT_CYCLE_LIFE.value,
    ]
    assert tools.calls[1][1] == {"upstream_result_id": first.result_ids[0]}
    with session_factory() as session:
        steps = tuple(
            session.scalars(
                select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.ordinal)
            )
        )
        results = tuple(
            session.scalars(
                select(ToolResultRecord).where(ToolResultRecord.run_id == run_id)
            )
        )
    assert [step.status for step in steps] == [
        AgentStepStatus.COMPLETED.value,
        AgentStepStatus.COMPLETED.value,
    ]
    assert len(results) == 2
    assert [event.event_type.value for event in run_service.list_events(member, run_id)] == [
        "RUN_CREATED",
        "STEP_STARTED",
        "STEP_COMPLETED",
        "STEP_STARTED",
        "STEP_COMPLETED",
        "RUN_COMPLETED",
    ]


def test_non_project_worker_inserts_tool_result_before_fk_dependent_provenance(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    engine = session_factory.kw["bind"]
    insert_statements: list[str] = []

    def record_insert_order(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _execution_context: object,
        _executemany: object,
    ) -> None:
        if statement.lstrip().upper().startswith("INSERT"):
            insert_statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_insert_order)
    try:
        _worker(run_service, session_factory, tools).advance_once(
            run_id=run_id,
            plan_hash=run_service.get_run(member, run_id).plan.plan_hash,
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_insert_order)

    tool_result_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO tool_results" in statement
    )
    provenance_position = next(
        index
        for index, statement in enumerate(insert_statements)
        if "INSERT INTO provenance_records" in statement
    )
    assert tool_result_position < provenance_position


def test_project_worker_atomically_binds_each_step_and_duplicate_delivery_does_not_rerun(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    worker = _project_worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    completed = worker.execute(run_id=run_id, plan_hash=plan_hash)
    repeated = worker.execute(run_id=run_id, plan_hash=plan_hash)

    assert repeated == completed
    assert completed.status is AgentRunStatus.COMPLETED
    assert len(tools.calls) == 2
    with session_factory() as session:
        steps = tuple(
            session.scalars(
                select(AgentStep)
                .where(AgentStep.run_id == run_id)
                .order_by(AgentStep.ordinal)
            )
        )
        bindings = tuple(
            session.scalars(
                select(ProjectToolResultBindingRecord)
                .where(ProjectToolResultBindingRecord.agent_run_id == run_id)
                .order_by(ProjectToolResultBindingRecord.step_id)
            )
        )
    assert len(bindings) == len(steps) == 2
    assert {binding.agent_step_id for binding in bindings} == {
        step.id for step in steps
    }
    assert all(
        binding.binding_schema_version == "project-tool-result-binding-v2"
        for binding in bindings
    )


def test_project_worker_resumes_after_restart_from_atomically_committed_step(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    first = _project_worker(run_service, session_factory, tools).advance_once(
        run_id=run_id,
        plan_hash=plan_hash,
    )
    completed = _project_worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=plan_hash,
    )

    assert first.status is AgentRunStatus.RUNNING
    assert first.completed_step_ids == ("features",)
    assert completed.status is AgentRunStatus.COMPLETED
    assert completed.completed_step_ids == ("features", "predict")
    assert len(tools.calls) == 2
    with session_factory() as session:
        bindings = tuple(
            session.scalars(
                select(ProjectToolResultBindingRecord).where(
                    ProjectToolResultBindingRecord.agent_run_id == run_id
                )
            )
        )
    assert len(bindings) == 2


def test_project_worker_reapplies_canonical_input_defaults_when_resuming(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, _ = _setup(tmp_path)
    tools = _CanonicalizingCountingTools()
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    first = _project_worker(run_service, session_factory, tools).advance_once(
        run_id=run_id,
        plan_hash=plan_hash,
    )
    completed = _project_worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=plan_hash,
    )

    assert first.status is AgentRunStatus.RUNNING
    assert first.completed_step_ids == ("features",)
    assert completed.status is AgentRunStatus.COMPLETED
    assert completed.completed_step_ids == ("features", "predict")
    with session_factory() as session:
        first_step = session.scalar(
            select(AgentStep).where(
                AgentStep.run_id == run_id,
                AgentStep.step_id == "features",
            )
        )
        assert first_step is not None
        assert first_step.resolved_input_json == {
            "record_batch_id": "verified-worker-record-batch-v1",
            "input_schema_version": "canonical-features-v1",
        }


def test_worker_can_advance_exactly_one_professional_agent_step_at_a_time(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    worker = _worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    first = worker.advance_once(run_id=run_id, plan_hash=plan_hash)

    assert first.status is AgentRunStatus.RUNNING
    assert first.completed_step_ids == ("features",)
    assert [name for name, _ in tools.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
    ]

    second = worker.advance_once(run_id=run_id, plan_hash=plan_hash)

    assert [name for name, _ in tools.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        StandardToolName.PREDICT_CYCLE_LIFE.value,
    ]
    assert second.status is AgentRunStatus.COMPLETED
    assert second.completed_step_ids == ("features", "predict")


def test_langgraph_runner_routes_each_role_but_keeps_worker_as_execution_boundary(
    tmp_path: Path,
) -> None:
    from quanxin_life.application.agent_graph import AgentGraphRunner

    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    worker = _worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash
    runner = AgentGraphRunner(worker=worker, checkpointer=InMemorySaver())

    completed = runner.execute(run_id=run_id, plan_hash=plan_hash)
    repeated = runner.execute(run_id=run_id, plan_hash=plan_hash)

    assert completed.status is AgentRunStatus.COMPLETED
    assert repeated == completed
    assert [name for name, _ in tools.calls] == [
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        StandardToolName.PREDICT_CYCLE_LIFE.value,
    ]


def test_langgraph_runner_exits_for_approval_and_resumes_the_same_thread(
    tmp_path: Path,
) -> None:
    from quanxin_life.application.agent_graph import AgentGraphRunner

    run_service, member, session_factory, run_id, tools = _setup(tmp_path, approval=True)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash
    runner = AgentGraphRunner(
        worker=_worker(run_service, session_factory, tools),
        checkpointer=InMemorySaver(),
    )

    paused = runner.execute(run_id=run_id, plan_hash=plan_hash)

    assert paused.status is AgentRunStatus.AWAITING_APPROVAL
    assert tools.calls == []
    with session_factory() as session:
        from quanxin_life.persistence.models import ApprovalRequestRow

        approval = session.scalar(
            select(ApprovalRequestRow).where(ApprovalRequestRow.run_id == run_id)
        )
        assert approval is not None
        approval_id = approval.id
    run_service.approve_run(
        member,
        run_id,
        approval_id=approval_id,
        reason="approved graph resume",
        now=NOW,
    )

    completed = runner.execute(run_id=run_id, plan_hash=plan_hash)

    assert completed.status is AgentRunStatus.COMPLETED
    assert len(tools.calls) == 2


def test_langgraph_runner_stops_on_authoritative_failed_run(tmp_path: Path) -> None:
    from quanxin_life.application.agent_graph import AgentGraphRunner

    run_service, member, session_factory, run_id, tools = _setup(
        tmp_path,
        fail_feature_attempts=1,
    )
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    failed = AgentGraphRunner(
        worker=_worker(run_service, session_factory, tools),
        checkpointer=InMemorySaver(),
    ).execute(run_id=run_id, plan_hash=plan_hash)

    assert failed.status is AgentRunStatus.FAILED
    assert len(tools.calls) == 1


def test_worker_rejects_a_stale_queue_plan_hash_before_tool_execution(tmp_path: Path) -> None:
    from quanxin_life.application.agent_run_execution import AgentRunExecutionError

    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    worker = _worker(run_service, session_factory, tools)

    with pytest.raises(AgentRunExecutionError, match="plan hash"):
        worker.execute(run_id=run_id, plan_hash="0" * 64)

    assert tools.calls == []
    assert run_service.get_run(member, run_id).status is AgentRunStatus.RUNNING


def test_worker_does_not_execute_a_cancelled_run(tmp_path: Path) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    run = run_service.get_run(member, run_id)
    run_service.cancel_run(member, run_id, now=NOW)

    state = _worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=run.plan.plan_hash,
    )

    assert state.status is AgentRunStatus.CANCELLED
    assert tools.calls == []


def test_worker_pauses_for_approval_then_resumes_the_same_plan(tmp_path: Path) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path, approval=True)
    worker = _worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    paused = worker.execute(run_id=run_id, plan_hash=plan_hash)

    assert paused.status is AgentRunStatus.AWAITING_APPROVAL
    assert paused.pending_approval_step_ids == ("features",)
    assert tools.calls == []
    with session_factory() as session:
        from quanxin_life.persistence.models import ApprovalRequestRow

        approval = session.scalar(
            select(ApprovalRequestRow).where(ApprovalRequestRow.run_id == run_id)
        )
        assert approval is not None
        step = session.scalar(
            select(AgentStep).where(
                AgentStep.run_id == run_id,
                AgentStep.step_id == "features",
            )
        )
        assert step is not None
        assert step.resolved_input_json == {
            "record_batch_id": "verified-worker-record-batch-v1"
        }
        assert step.resolved_input_hash == sha256_canonical(step.resolved_input_json)
        assert step.execution_snapshot_sha256 is not None
        assert approval.agent_step_id == step.id
        assert approval.execution_snapshot_sha256 == step.execution_snapshot_sha256
        approval_id = approval.id
    run_service.approve_run(
        member,
        run_id,
        approval_id=approval_id,
        reason="reviewed trusted data scope",
        now=NOW,
    )

    completed = worker.execute(run_id=run_id, plan_hash=plan_hash)

    assert completed.status is AgentRunStatus.COMPLETED
    assert completed.pending_approval_step_ids == ()
    assert len(tools.calls) == 2
    assert run_service.get_approval(member, run_id, approval_id).status is (
        ApprovalStatus.APPROVED
    )


def test_worker_refuses_to_steal_an_active_step_lease(tmp_path: Path) -> None:
    from quanxin_life.application.agent_run_execution import AgentRunExecutionBusyError

    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash
    with session_factory.begin() as session:
        step = session.scalar(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.ordinal == 1)
        )
        assert step is not None
        step.status = "RUNNING"
        step.attempts = 1
        step.claim_token = "a" * 64
        step.lease_expires_at = NOW + timedelta(minutes=5)

    with pytest.raises(AgentRunExecutionBusyError, match="lease"):
        _worker(run_service, session_factory, tools).execute(
            run_id=run_id,
            plan_hash=plan_hash,
        )

    assert tools.calls == []


def test_worker_reclaims_an_expired_step_lease_and_clears_the_claim(tmp_path: Path) -> None:
    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash
    with session_factory.begin() as session:
        step = session.scalar(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.ordinal == 1)
        )
        assert step is not None
        step.status = "RUNNING"
        step.attempts = 1
        step.claim_token = "b" * 64
        step.lease_expires_at = NOW - timedelta(seconds=1)

    completed = _worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=plan_hash,
    )

    assert completed.status is AgentRunStatus.COMPLETED
    with session_factory() as session:
        first = session.scalar(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.ordinal == 1)
        )
        assert first is not None
        assert first.attempts == 2
        assert first.claim_token is None
        assert first.lease_expires_at is None


def test_worker_rejects_a_persisted_result_whose_input_hash_was_tampered(
    tmp_path: Path,
) -> None:
    from quanxin_life.application.agent_run_execution import AgentRunExecutionError

    run_service, member, session_factory, run_id, tools = _setup(tmp_path)
    first_worker = _worker(run_service, session_factory, tools)
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash
    assert first_worker.execute(run_id=run_id, plan_hash=plan_hash).status is (
        AgentRunStatus.COMPLETED
    )

    with session_factory.begin() as session:
        steps = tuple(
            session.scalars(
                select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.ordinal)
            )
        )
        results = tuple(
            session.scalars(
                select(ToolResultRecord)
                .where(ToolResultRecord.run_id == run_id)
                .order_by(ToolResultRecord.created_at)
            )
        )
        assert len(steps) == len(results) == 2
        results[0].input_hash = "0" * 64
        session.execute(
            delete(ProvenanceRecordRow).where(
                ProvenanceRecordRow.tool_result_id == results[1].id
            )
        )
        session.delete(results[1])
        steps[1].status = "PENDING"
        steps[1].started_at = None
        steps[1].completed_at = None
        run = session.scalar(select(AgentRun).where(AgentRun.id == run_id))
        assert run is not None
        run.status = AgentRunStatus.RUNNING.value
        run.completed_at = None

    fresh_tools = _CountingTools()
    with pytest.raises(AgentRunExecutionError, match="input hash"):
        _worker(run_service, session_factory, fresh_tools).execute(
            run_id=run_id,
            plan_hash=plan_hash,
        )

    assert fresh_tools.calls == []


def test_worker_applies_retry_once_exactly_once_before_continuing(tmp_path: Path) -> None:
    run_service, member, session_factory, run_id, tools = _setup(
        tmp_path,
        failure_policy=AgentFailurePolicy.RETRY_ONCE,
        fail_feature_attempts=1,
    )
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    completed = _worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=plan_hash,
    )

    assert completed.status is AgentRunStatus.COMPLETED
    assert [name for name, _ in tools.calls].count(
        StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value
    ) == 2
    with session_factory() as session:
        first = session.scalar(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.ordinal == 1)
        )
        assert first is not None
        assert first.attempts == 2


def test_worker_marks_replan_policy_unavailable_without_fake_replanning(
    tmp_path: Path,
) -> None:
    run_service, member, session_factory, run_id, tools = _setup(
        tmp_path,
        failure_policy=AgentFailurePolicy.REPLAN,
        fail_feature_attempts=1,
    )
    plan_hash = run_service.get_run(member, run_id).plan.plan_hash

    failed = _worker(run_service, session_factory, tools).execute(
        run_id=run_id,
        plan_hash=plan_hash,
    )

    assert failed.status is AgentRunStatus.FAILED
    with session_factory() as session:
        first = session.scalar(
            select(AgentStep).where(AgentStep.run_id == run_id, AgentStep.ordinal == 1)
        )
        assert first is not None
        assert first.last_error_code is not None
        assert first.last_error_code.startswith("REPLAN_UNAVAILABLE:")
