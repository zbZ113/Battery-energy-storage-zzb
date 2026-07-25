from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import Field
from sqlalchemy import select

from quanxin_life.agents.supervisor import (
    SupervisorPlanningRequest,
    SupervisorPlanningResult,
)
from quanxin_life.api.service import ToolInvocation, ToolInvocationService
from quanxin_life.application.agent_runs import AgentRunService
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.invocation_context import (
    ProjectInvocationContextService,
    ProjectInvocationSource,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.audit import AuditLedgerError
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentIntent,
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
    ProjectStatus,
    ProvenanceRecord,
    SessionStatus,
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
    Project,
    ProjectToolResultBindingRecord,
    SessionRecord,
    ToolResultRecord,
    User,
)
from quanxin_life.tools import (
    StandardToolName,
    ToolAuthorizationError,
    ToolDefinition,
    ToolExecutionScope,
    ToolRegistry,
)

NOW = datetime(2026, 7, 25, 9, 0, tzinfo=UTC)
CLAIM_TOKEN = "c" * 64


class _ProjectInput(ContractModel):
    record_batch_id: str = Field(min_length=1)


class _Planner:
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
        plan = AgentPlan.build(
            plan_version="project-agent-grant-test-v1",
            intent_id=intent.intent_id,
            planning_mode=AgentPlanningMode.FIXED_FALLBACK,
            steps=(
                AgentPlanStep(
                    step_id="predict-life",
                    role=AgentRole.LIFETIME,
                    tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
                    input_references={"record_batch_id": "intent.dataset_ids[0]"},
                ),
                AgentPlanStep(
                    step_id="predict-soh",
                    role=AgentRole.LIFETIME,
                    tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY.value,
                    input_references={"record_batch_id": "intent.dataset_ids[0]"},
                ),
            ),
            created_at=NOW,
        )
        return SupervisorPlanningResult(intent=intent, plan=plan)


@dataclass(frozen=True, slots=True)
class _Fixture:
    session_factory: SessionFactory
    context_service: ProjectInvocationContextService
    principal: AuthPrincipal
    project_id: str
    run_id: str
    step_id: str
    plan_hash: str
    claim_token: str


@pytest.fixture
def invocation_fixture(tmp_path: Path) -> _Fixture:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'agent-project-grant.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principal = AuthPrincipal(
        user_id=str(uuid4()),
        session_id=str(uuid4()),
        username="project-agent@example.test",
        role=UserRole.MEMBER,
        must_change_password=False,
    )
    with session_factory.begin() as session:
        session.add(
            User(
                id=principal.user_id,
                username=principal.username,
                credential_hash="$argon2id$project-agent-test",
                must_change_credential=False,
                role=principal.role.value,
                status=UserStatus.ACTIVE.value,
                created_at=NOW,
                updated_at=NOW,
            )
        )
        session.add(
            SessionRecord(
                id=principal.session_id,
                user_id=principal.user_id,
                token_hash=uuid4().hex + uuid4().hex,
                status=SessionStatus.ACTIVE.value,
                created_at=NOW,
                expires_at=NOW + timedelta(hours=2),
            )
        )
    project = ProjectService(session_factory).create_project(
        principal,
        name="project Agent invocation",
        now=NOW,
    )
    dataset = DatasetService(session_factory).create_dataset(
        principal,
        project_id=project.project_id,
        name="project Agent dataset",
        data_version="project-agent-data-v1",
        schema_version="canonical-v1",
        now=NOW,
    )
    DatasetService(session_factory).freeze_dataset(
        principal,
        dataset.dataset_id,
        now=NOW,
    )
    run_service = AgentRunService(session_factory, planner=_Planner())
    run = run_service.create_run(
        principal,
        request=SupervisorPlanningRequest(
            project_id=project.project_id,
            user_goal="execute the persisted project workflow",
            dataset_ids=(dataset.dataset_id,),
            requested_outputs=("audited-result",),
        ),
        idempotency_key=f"project-agent-{uuid4()}",
        available_tools=(
            StandardToolName.PREDICT_CYCLE_LIFE,
            StandardToolName.PREDICT_SOH_TRAJECTORY,
        ),
        now=NOW,
    )
    with session_factory.begin() as session:
        dispatch = session.scalar(
            select(AgentRunDispatch).where(AgentRunDispatch.run_id == run.run_id)
        )
        step = session.scalar(
            select(AgentStep).where(
                AgentStep.run_id == run.run_id,
                AgentStep.step_id == "predict-life",
            )
        )
        assert dispatch is not None
        assert step is not None
        dispatch.status = "DISPATCHED"
        dispatch.task_id = "project-agent-task"
        step.status = "RUNNING"
        step.attempts = 1
        step.claim_token = CLAIM_TOKEN
        step.started_at = NOW
        step.lease_expires_at = NOW + timedelta(minutes=10)
    return _Fixture(
        session_factory=session_factory,
        context_service=ProjectInvocationContextService(
            session_factory,
            clock=lambda: NOW,
        ),
        principal=principal,
        project_id=project.project_id,
        run_id=run.run_id,
        step_id="predict-life",
        plan_hash=run.plan.plan_hash,
        claim_token=CLAIM_TOKEN,
    )


def _resolver(fixture: _Fixture, *, now: datetime = NOW):
    from quanxin_life.application.agent_run_invocation import (
        PersistentAgentRunInvocationResolver,
    )

    return PersistentAgentRunInvocationResolver(
        fixture.session_factory,
        context_service=fixture.context_service,
        clock=lambda: now,
    )


def _resolve(fixture: _Fixture):
    return _resolver(fixture).resolve(
        fixture.run_id,
        fixture.step_id,
        fixture.claim_token,
    )


def test_grant_is_frozen_and_derived_from_persisted_run_plan_step_and_dispatch(
    invocation_fixture: _Fixture,
) -> None:
    grant = _resolve(invocation_fixture)

    assert grant.agent_run_id == invocation_fixture.run_id
    assert grant.step_id == invocation_fixture.step_id
    assert grant.plan_hash == invocation_fixture.plan_hash
    assert grant.allowed_tool_names == frozenset(
        {StandardToolName.PREDICT_CYCLE_LIFE}
    )
    assert grant.project_context.project_id == invocation_fixture.project_id
    assert grant.project_context.actor_user_id == invocation_fixture.principal.user_id
    assert grant.project_context.actor_session_id == invocation_fixture.principal.session_id
    assert grant.project_context.actor_role is UserRole.MEMBER
    assert grant.project_context.invocation_source is ProjectInvocationSource.AGENT
    assert grant.project_context.agent_run_id == invocation_fixture.run_id

    with pytest.raises((AttributeError, TypeError)):
        grant.step_id = "predict-soh"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("run_id", str(uuid4())),
        ("step_id", "predict-soh"),
        ("claim_token", "f" * 64),
    ],
)
def test_resolver_rejects_substituted_run_step_or_claim(
    invocation_fixture: _Fixture,
    field_name: str,
    replacement: str,
) -> None:
    coordinates = {
        "run_id": invocation_fixture.run_id,
        "step_id": invocation_fixture.step_id,
        "claim_token": invocation_fixture.claim_token,
    }
    coordinates[field_name] = replacement

    with pytest.raises(RuntimeError):
        _resolver(invocation_fixture).resolve(**coordinates)


def test_revalidation_rejects_a_caller_replaced_allowlist(
    invocation_fixture: _Fixture,
) -> None:
    resolver = _resolver(invocation_fixture)
    grant = resolver.resolve(
        invocation_fixture.run_id,
        invocation_fixture.step_id,
        invocation_fixture.claim_token,
    )
    forged = replace(
        grant,
        allowed_tool_names=frozenset({StandardToolName.PREDICT_SOH_TRAJECTORY}),
    )

    with pytest.raises(RuntimeError):
        resolver.revalidate(forged)


@pytest.mark.parametrize("tamper_target", ["plan", "step"])
def test_resolver_rejects_persisted_plan_or_step_tampering(
    invocation_fixture: _Fixture,
    tamper_target: str,
) -> None:
    with invocation_fixture.session_factory.begin() as session:
        if tamper_target == "plan":
            run = session.get(AgentRun, invocation_fixture.run_id)
            assert run is not None
            changed = dict(run.plan_json)
            changed["plan_version"] = "tampered-plan-v2"
            run.plan_json = changed
        else:
            step = session.scalar(
                select(AgentStep).where(
                    AgentStep.run_id == invocation_fixture.run_id,
                    AgentStep.step_id == invocation_fixture.step_id,
                )
            )
            assert step is not None
            step.tool_name = StandardToolName.PREDICT_SOH_TRAJECTORY.value

    with pytest.raises(RuntimeError):
        _resolve(invocation_fixture)


@pytest.mark.parametrize("live_state", ["revoked-session", "archived-project"])
def test_resolver_revalidates_live_session_and_project(
    invocation_fixture: _Fixture,
    live_state: str,
) -> None:
    with invocation_fixture.session_factory.begin() as session:
        if live_state == "revoked-session":
            record = session.get(SessionRecord, invocation_fixture.principal.session_id)
            assert record is not None
            record.status = SessionStatus.REVOKED.value
        else:
            project = session.get(Project, invocation_fixture.project_id)
            assert project is not None
            project.status = ProjectStatus.ARCHIVED.value

    with pytest.raises(RuntimeError):
        _resolve(invocation_fixture)


def test_resolver_rejects_an_expired_step_claim(invocation_fixture: _Fixture) -> None:
    with pytest.raises(RuntimeError):
        _resolver(
            invocation_fixture,
            now=NOW + timedelta(minutes=11),
        ).resolve(
            invocation_fixture.run_id,
            invocation_fixture.step_id,
            invocation_fixture.claim_token,
        )


def _project_registry(
    fixture: _Fixture,
    calls: list[StandardToolName],
    *,
    during_execution: Callable[[], None] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(project_context_validator=fixture.context_service)

    def register(tool_name: StandardToolName) -> None:
        def execute(value: _ProjectInput, context: object) -> ToolResult:
            del context
            calls.append(tool_name)
            if during_execution is not None:
                during_execution()
            payload = value.model_dump(mode="json")
            return ToolResult(
                result_id=str(uuid4()),
                tool_name=tool_name.value,
                tool_version="project-agent-test-v1",
                model_version="project-agent-model-v1",
                data_version="project-agent-data-v1",
                feature_version="project-agent-feature-v1",
                input_hash=sha256_canonical(payload),
                values={"execution_status": "computed"},
                provenance=[
                    ProvenanceRecord(
                        source_id=f"fixture-{tool_name.value}",
                        source_kind=SourceKind.OBSERVED,
                        uri=f"test://project-agent/{tool_name.value}",
                        sha256=sha256_canonical({"tool": tool_name.value}),
                        description="Project Agent authorization fixture",
                        created_at=NOW,
                    )
                ],
                created_at=NOW,
            )

        registry.register(
            ToolDefinition(
                tool_name=tool_name,
                tool_version="project-agent-test-v1",
                input_model=_ProjectInput,
                execution_scope=ToolExecutionScope.PROJECT,
                executor=None,
                project_executor=execute,
            )
        )

    register(StandardToolName.PREDICT_CYCLE_LIFE)
    register(StandardToolName.PREDICT_SOH_TRAJECTORY)
    return registry


def _project_ledger(fixture: _Fixture):
    from quanxin_life.audit.sql_project_ledger import SqlProjectAuditLedger

    return SqlProjectAuditLedger(
        fixture.session_factory,
        context_validator=fixture.context_service,
    )


def test_project_agent_service_requires_validator_and_ledger_before_executor(
    invocation_fixture: _Fixture,
) -> None:
    grant = _resolve(invocation_fixture)
    invocation = ToolInvocation(
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
        input_value={"record_batch_id": "opaque-record-batch"},
    )
    calls: list[StandardToolName] = []

    without_validator = ToolInvocationService(
        registry=_project_registry(invocation_fixture, calls),
        project_audit_ledger=_project_ledger(invocation_fixture),
    )
    with pytest.raises(ToolAuthorizationError, match="validator"):
        without_validator.invoke_for_project_agent(invocation, grant=grant)

    without_ledger = ToolInvocationService(
        registry=_project_registry(invocation_fixture, calls),
        agent_run_invocation_validator=_resolver(invocation_fixture),
    )
    with pytest.raises(AuditLedgerError, match="project audit ledger"):
        without_ledger.invoke_for_project_agent(invocation, grant=grant)

    assert calls == []


def test_generic_project_service_rejects_agent_context_without_grant(
    invocation_fixture: _Fixture,
) -> None:
    calls: list[StandardToolName] = []
    service = ToolInvocationService(
        registry=_project_registry(invocation_fixture, calls),
        project_audit_ledger=_project_ledger(invocation_fixture),
    )
    grant = _resolve(invocation_fixture)

    with pytest.raises(ToolAuthorizationError, match="per-step grant"):
        service.invoke_in_project(
            ToolInvocation(
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
                input_value={"record_batch_id": "opaque-record-batch"},
            ),
            context=grant.project_context,
        )

    assert calls == []


def test_project_agent_service_rejects_another_tool_before_executor(
    invocation_fixture: _Fixture,
) -> None:
    calls: list[StandardToolName] = []
    resolver = _resolver(invocation_fixture)
    service = ToolInvocationService(
        registry=_project_registry(invocation_fixture, calls),
        project_audit_ledger=_project_ledger(invocation_fixture),
        agent_run_invocation_validator=resolver,
    )

    with pytest.raises(ToolAuthorizationError, match="permitted"):
        service.invoke_for_project_agent(
            ToolInvocation(
                tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
                input_value={"record_batch_id": "opaque-record-batch"},
            ),
            grant=resolver.resolve(
                invocation_fixture.run_id,
                invocation_fixture.step_id,
                invocation_fixture.claim_token,
            ),
        )

    assert calls == []


def test_project_agent_service_binds_legal_result_to_agent_run(
    invocation_fixture: _Fixture,
) -> None:
    calls: list[StandardToolName] = []
    ledger = _project_ledger(invocation_fixture)
    resolver = _resolver(invocation_fixture)
    service = ToolInvocationService(
        registry=_project_registry(invocation_fixture, calls),
        project_audit_ledger=ledger,
        agent_run_invocation_validator=resolver,
    )
    grant = resolver.resolve(
        invocation_fixture.run_id,
        invocation_fixture.step_id,
        invocation_fixture.claim_token,
    )

    result = service.invoke_for_project_agent(
        ToolInvocation(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            input_value={"record_batch_id": "opaque-record-batch"},
        ),
        grant=grant,
    )

    binding = ledger.resolve_binding(grant.project_context, result.result_id)
    assert calls == [StandardToolName.PREDICT_CYCLE_LIFE]
    assert binding.project_id == invocation_fixture.project_id
    assert binding.agent_run_id == invocation_fixture.run_id
    assert binding.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value


def test_project_agent_service_rejects_result_when_claim_changes_during_execution(
    invocation_fixture: _Fixture,
) -> None:
    calls: list[StandardToolName] = []

    def replace_claim() -> None:
        with invocation_fixture.session_factory.begin() as session:
            step = session.scalar(
                select(AgentStep).where(
                    AgentStep.run_id == invocation_fixture.run_id,
                    AgentStep.step_id == invocation_fixture.step_id,
                )
            )
            assert step is not None
            step.claim_token = "d" * 64

    resolver = _resolver(invocation_fixture)
    service = ToolInvocationService(
        registry=_project_registry(
            invocation_fixture,
            calls,
            during_execution=replace_claim,
        ),
        project_audit_ledger=_project_ledger(invocation_fixture),
        agent_run_invocation_validator=resolver,
    )
    grant = resolver.resolve(
        invocation_fixture.run_id,
        invocation_fixture.step_id,
        invocation_fixture.claim_token,
    )

    with pytest.raises(RuntimeError):
        service.invoke_for_project_agent(
            ToolInvocation(
                tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
                input_value={"record_batch_id": "opaque-record-batch"},
            ),
            grant=grant,
        )

    assert calls == [StandardToolName.PREDICT_CYCLE_LIFE]
    with invocation_fixture.session_factory() as session:
        assert list(session.scalars(select(ToolResultRecord))) == []
        assert list(session.scalars(select(ProjectToolResultBindingRecord))) == []
