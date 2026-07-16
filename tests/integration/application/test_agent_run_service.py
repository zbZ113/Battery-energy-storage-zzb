from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.agents.supervisor import (
    SupervisorPlanningRequest,
    SupervisorPlanningResult,
)
from quanxin_life.application.agent_runs import (
    AgentRunAccessError,
    AgentRunConflictError,
    AgentRunNotFoundError,
    AgentRunService,
    AgentRunStateError,
)
from quanxin_life.application.datasets import DatasetService
from quanxin_life.application.projects import ProjectService
from quanxin_life.application.task_queue import AgentRunDispatchReceipt
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import (
    AgentFailurePolicy,
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
    AgentRunStatus,
    ApprovalKind,
    ApprovalStatus,
    UserRole,
    UserStatus,
)
from quanxin_life.core.product import AgentIntent
from quanxin_life.persistence import Base, create_engine_from_config, create_session_factory
from quanxin_life.persistence.database import DatabaseConfig, SessionFactory
from quanxin_life.persistence.models import SessionRecord, User
from quanxin_life.tools import StandardToolName

NOW = datetime(2026, 7, 16, 10, 0, tzinfo=UTC)


def _principal(user_id: str, role: UserRole) -> AuthPrincipal:
    return AuthPrincipal(
        user_id=user_id,
        session_id=str(uuid4()),
        username=f"{role.value.casefold()}-{user_id[:8]}@example.test",
        role=role,
        must_change_password=False,
    )


class FixedPlanner:
    def __init__(self) -> None:
        self.calls = 0

    def plan(
        self,
        request: SupervisorPlanningRequest,
        *,
        available_tools: object,
    ) -> SupervisorPlanningResult:
        del available_tools
        self.calls += 1
        intent = AgentIntent(
            intent_id=str(uuid4()),
            project_id=request.project_id,
            goal=request.user_goal,
            dataset_ids=request.dataset_ids,
            requested_outputs=request.requested_outputs,
            created_at=NOW,
        )
        plan = AgentPlan.build(
            plan_version="test-plan-v1",
            intent_id=intent.intent_id,
            steps=(
                AgentPlanStep(
                    step_id="features",
                    role=AgentRole.DATA_QUALITY,
                    tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                    input_references={"record_batch_id": "intent.dataset_ids[0]"},
                    requires_approval=True,
                    failure_policy=AgentFailurePolicy.STOP,
                ),
                AgentPlanStep(
                    step_id="features-retry",
                    role=AgentRole.DATA_QUALITY,
                    tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
                    depends_on=("features",),
                    input_references={"record_batch_id": "intent.dataset_ids[0]"},
                    requires_approval=True,
                    failure_policy=AgentFailurePolicy.STOP,
                ),
            ),
            planning_mode=AgentPlanningMode.FIXED_FALLBACK,
            created_at=NOW,
        )
        return SupervisorPlanningResult(
            intent=intent,
            plan=plan,
            warnings=("LLM_PLANNING_FALLBACK:TEST",),
        )


class RecordingQueue:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def enqueue(self, *, run_id: str, plan_hash: str) -> AgentRunDispatchReceipt:
        self.calls.append((run_id, plan_hash))
        if self.fail:
            raise RuntimeError("simulated broker outage")
        return AgentRunDispatchReceipt(run_id=run_id, task_id=f"task-{run_id}")


@pytest.fixture
def run_context(
    tmp_path: Path,
) -> tuple[
    AgentRunService,
    FixedPlanner,
    ProjectService,
    DatasetService,
    dict[UserRole | str, AuthPrincipal],
    SessionFactory,
]:
    engine = create_engine_from_config(
        DatabaseConfig(url=f"sqlite+pysqlite:///{tmp_path / 'agent-runs.sqlite3'}")
    )
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    principals: dict[UserRole | str, AuthPrincipal] = {
        UserRole.ADMIN: _principal(str(uuid4()), UserRole.ADMIN),
        UserRole.MEMBER: _principal(str(uuid4()), UserRole.MEMBER),
        UserRole.JUDGE: _principal(str(uuid4()), UserRole.JUDGE),
        "outsider": _principal(str(uuid4()), UserRole.MEMBER),
    }
    with session_factory.begin() as session:
        for principal in principals.values():
            session.add(
                User(
                    id=principal.user_id,
                    username=principal.username,
                    credential_hash="$argon2id$test-only-not-used-for-authentication",
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
                    status="ACTIVE",
                    created_at=NOW,
                    expires_at=NOW + timedelta(days=1),
                )
            )
    planner = FixedPlanner()
    return (
        AgentRunService(session_factory, planner=planner),
        planner,
        ProjectService(session_factory),
        DatasetService(session_factory),
        principals,
        session_factory,
    )


def _request(
    project_id: str,
    dataset_id: str,
    goal: str = "analyze lifetime",
) -> SupervisorPlanningRequest:
    return SupervisorPlanningRequest(
        project_id=project_id,
        user_goal=goal,
        dataset_ids=(dataset_id,),
        requested_outputs=("cycle_life",),
    )


def _project_and_frozen_dataset(
    projects: ProjectService,
    datasets: DatasetService,
    member: AuthPrincipal,
) -> tuple[str, str]:
    project = projects.create_project(member, name="Agent run project", now=NOW)
    dataset = datasets.create_dataset(
        member,
        project_id=project.project_id,
        name="safe dataset",
        data_version="safe-v1",
        schema_version="canonical-v1",
        now=NOW,
    )
    frozen = datasets.freeze_dataset(member, dataset.dataset_id, now=NOW)
    return project.project_id, frozen.dataset_id


def test_idempotent_create_persists_plan_steps_event_and_pending_dispatch(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, planner, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    request = _request(project_id, dataset_id)

    created = service.create_run(
        member,
        request=request,
        idempotency_key="create-run-20260716-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )
    repeated = service.create_run(
        member,
        request=request,
        idempotency_key="create-run-20260716-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )

    assert repeated == created
    assert planner.calls == 1
    assert created.status is AgentRunStatus.RUNNING
    assert created.intent.project_id == project_id
    assert created.plan.steps[0].depends_on == ()
    assert created.dispatch_status == "PENDING"
    events = service.list_events(member, created.run_id)
    assert [event.sequence for event in events] == [1]
    assert events[0].event_type == "RUN_CREATED"
    assert request.user_goal not in str(events[0].payload)


def test_reusing_an_idempotency_key_for_a_different_request_conflicts(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    service.create_run(
        member,
        request=_request(project_id, dataset_id),
        idempotency_key="same-key-different-request",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )

    with pytest.raises(AgentRunConflictError, match="idempotency"):
        service.create_run(
            member,
            request=_request(project_id, dataset_id, goal="different goal"),
            idempotency_key="same-key-different-request",
            available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
            now=NOW,
        )


def test_run_requires_a_visible_active_project_and_frozen_project_dataset(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    outsider = principals["outsider"]
    project = projects.create_project(member, name="private", now=NOW)
    draft = datasets.create_dataset(
        member,
        project_id=project.project_id,
        name="draft",
        data_version="v1",
        schema_version="schema-v1",
        now=NOW,
    )
    request = _request(project.project_id, draft.dataset_id)

    with pytest.raises(AgentRunNotFoundError):
        service.create_run(
            outsider,
            request=request,
            idempotency_key="outsider-key-0001",
            available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
            now=NOW,
        )
    with pytest.raises(AgentRunStateError, match="frozen"):
        service.create_run(
            member,
            request=request,
            idempotency_key="draft-dataset-key-0001",
            available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
            now=NOW,
        )
    with pytest.raises(AgentRunAccessError, match="role"):
        service.create_run(
            principals[UserRole.JUDGE],
            request=request,
            idempotency_key="judge-key-0001",
            available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
            now=NOW,
        )


def test_dispatch_is_recoverable_and_does_not_duplicate_a_successful_send(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    created = service.create_run(
        member,
        request=_request(project_id, dataset_id),
        idempotency_key="dispatch-key-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )
    failing = RecordingQueue(fail=True)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        service.dispatch_pending(created.run_id, queue=failing, now=NOW)
    assert service.get_run(member, created.run_id).dispatch_status == "PENDING"

    queue = RecordingQueue()
    dispatched = service.dispatch_pending(created.run_id, queue=queue, now=NOW)
    repeated = service.dispatch_pending(created.run_id, queue=queue, now=NOW)
    assert dispatched.dispatch_status == "DISPATCHED"
    assert repeated == dispatched
    assert len(queue.calls) == 1
    assert [event.event_type for event in service.list_events(member, created.run_id)] == [
        "RUN_CREATED",
        "DISPATCH_FAILED",
        "RUN_DISPATCHED",
    ]


def test_cancel_is_idempotent_and_preserves_completed_terminal_runs(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    created = service.create_run(
        member,
        request=_request(project_id, dataset_id),
        idempotency_key="cancel-key-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )

    cancelled = service.cancel_run(member, created.run_id, now=NOW)
    assert cancelled.status is AgentRunStatus.CANCELLED
    assert cancelled.completed_at == NOW
    assert service.cancel_run(member, created.run_id, now=NOW) == cancelled
    assert service.list_events(member, created.run_id)[-1].event_type == "RUN_CANCELLED"
    with pytest.raises(AgentRunStateError, match="cancelled"):
        service.dispatch_pending(created.run_id, queue=RecordingQueue(), now=NOW)


def test_approval_pauses_run_and_approve_or_reject_is_audited_and_idempotent(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    created = service.create_run(
        member,
        request=_request(project_id, dataset_id),
        idempotency_key="approval-key-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )
    service.dispatch_pending(created.run_id, queue=RecordingQueue(), now=NOW)

    approval = service.request_approval(
        created.run_id,
        step_id="features",
        approval_kind=ApprovalKind.FORMAL_DECISION,
        impact_scope="Release a formal reviewed result",
        now=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    assert approval.status is ApprovalStatus.PENDING
    assert service.get_run(member, created.run_id).status is AgentRunStatus.AWAITING_APPROVAL

    approved = service.approve_run(
        member,
        created.run_id,
        approval_id=approval.approval_id,
        reason="Reviewed source evidence",
        now=NOW,
    )
    assert approved.status is AgentRunStatus.RUNNING
    assert approved.dispatch_status == "PENDING"
    assert service.approve_run(
        member,
        created.run_id,
        approval_id=approval.approval_id,
        reason="Repeated browser callback",
        now=NOW,
    ) == approved

    second = service.request_approval(
        created.run_id,
        step_id="features-retry",
        approval_kind=ApprovalKind.EXTERNAL_WRITE,
        impact_scope="Write a reviewed result to an external integration",
        now=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    rejected = service.reject_run(
        member,
        created.run_id,
        approval_id=second.approval_id,
        reason="Do not publish externally",
        now=NOW,
    )
    assert rejected.status is AgentRunStatus.CANCELLED
    event_types = [
        event.event_type for event in service.list_events(member, created.run_id)
    ]
    assert "APPROVAL_REQUESTED" in event_types
    assert "APPROVAL_APPROVED" in event_types
    assert event_types[-1] == "APPROVAL_REJECTED"


def test_cancel_marks_pending_approval_cancelled_and_expired_approval_cannot_resume(
    run_context: tuple[
        AgentRunService,
        FixedPlanner,
        ProjectService,
        DatasetService,
        dict[UserRole | str, AuthPrincipal],
        SessionFactory,
    ],
) -> None:
    service, _, projects, datasets, principals, _ = run_context
    member = principals[UserRole.MEMBER]
    project_id, dataset_id = _project_and_frozen_dataset(projects, datasets, member)
    created = service.create_run(
        member,
        request=_request(project_id, dataset_id),
        idempotency_key="approval-expiry-key-0001",
        available_tools=(StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES,),
        now=NOW,
    )
    approval = service.request_approval(
        created.run_id,
        step_id="features",
        approval_kind=ApprovalKind.FORMAL_DECISION,
        impact_scope="Time-bounded formal release",
        now=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )
    with pytest.raises(AgentRunStateError, match="expired"):
        service.approve_run(
            member,
            created.run_id,
            approval_id=approval.approval_id,
            reason=None,
            now=NOW + timedelta(minutes=2),
        )
    assert service.get_approval(member, created.run_id, approval.approval_id).status is (
        ApprovalStatus.EXPIRED
    )

    another = service.request_approval(
        created.run_id,
        step_id="features-retry",
        approval_kind=ApprovalKind.FORMAL_DECISION,
        impact_scope="A second pending gate",
        now=NOW,
        expires_at=NOW + timedelta(hours=1),
    )
    service.cancel_run(member, created.run_id, now=NOW)
    assert service.get_approval(member, created.run_id, another.approval_id).status is (
        ApprovalStatus.CANCELLED
    )
