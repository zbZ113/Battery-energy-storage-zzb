from datetime import UTC, datetime
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from quanxin_life.agents.coordination_graph import (
    AgentCoordinationRouter,
    AgentCoordinationStatus,
)
from quanxin_life.core import (
    AgentPlan,
    AgentPlanningMode,
    AgentPlanStep,
    AgentRole,
)

NOW = datetime(2026, 7, 16, 15, 0, tzinfo=UTC)


def _plan() -> AgentPlan:
    intent_id = str(uuid4())
    return AgentPlan.build(
        plan_version="coordination-graph-test-v1",
        intent_id=intent_id,
        planning_mode=AgentPlanningMode.FIXED_FALLBACK,
        steps=(
            AgentPlanStep(
                step_id="quality",
                role=AgentRole.DATA_QUALITY,
                tool_name="validate_battery_data",
                input_references={"record_batch_id": "intent.dataset_ids[0]"},
            ),
            AgentPlanStep(
                step_id="lifetime",
                role=AgentRole.LIFETIME,
                tool_name="predict_cycle_life",
                depends_on=("quality",),
                input_references={"upstream_result_id": "step.quality.result_id"},
            ),
        ),
        created_at=NOW,
    )


def test_graph_routes_the_next_incomplete_step_to_its_professional_agent() -> None:
    plan = _plan()
    router = AgentCoordinationRouter(checkpointer=InMemorySaver())
    run_id = str(uuid4())

    first = router.select_next(
        run_id=run_id,
        plan=plan,
        completed_step_ids=(),
        pending_approval_step_ids=(),
    )
    second = router.select_next(
        run_id=run_id,
        plan=plan,
        completed_step_ids=("quality",),
        pending_approval_step_ids=(),
    )

    assert first.status is AgentCoordinationStatus.DISPATCH_READY
    assert first.step_id == "quality"
    assert first.role is AgentRole.DATA_QUALITY
    assert first.routed_role is AgentRole.DATA_QUALITY
    assert first.tool_name == "validate_battery_data"
    assert second.status is AgentCoordinationStatus.DISPATCH_READY
    assert second.step_id == "lifetime"
    assert second.role is AgentRole.LIFETIME
    assert second.routed_role is AgentRole.LIFETIME


def test_graph_pauses_on_approval_and_finishes_only_after_a_sequential_prefix() -> None:
    plan = _plan()
    router = AgentCoordinationRouter(checkpointer=InMemorySaver())
    run_id = str(uuid4())

    paused = router.select_next(
        run_id=run_id,
        plan=plan,
        completed_step_ids=("quality",),
        pending_approval_step_ids=("lifetime",),
    )
    complete = router.select_next(
        run_id=run_id,
        plan=plan,
        completed_step_ids=("quality", "lifetime"),
        pending_approval_step_ids=(),
    )

    assert paused.status is AgentCoordinationStatus.AWAITING_APPROVAL
    assert paused.step_id == "lifetime"
    assert paused.routed_role is None
    assert complete.status is AgentCoordinationStatus.COMPLETE
    assert complete.step_id is None
    assert complete.role is None
    assert complete.tool_name is None


def test_graph_rejects_non_prefix_completion_that_could_skip_a_required_step() -> None:
    plan = _plan()
    router = AgentCoordinationRouter(checkpointer=InMemorySaver())

    try:
        router.select_next(
            run_id=str(uuid4()),
            plan=plan,
            completed_step_ids=("lifetime",),
            pending_approval_step_ids=(),
        )
    except ValueError as exc:
        assert "sequential prefix" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("non-prefix completion must be rejected")
