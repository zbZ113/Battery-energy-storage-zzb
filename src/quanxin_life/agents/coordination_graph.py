"""LangGraph routing over control metadata only; domain tools stay outside the graph."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict, cast
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from quanxin_life.core import AgentPlan, AgentRole
from quanxin_life.core.schemas import ContractModel

COORDINATION_GRAPH_VERSION = "agent-coordination-graph-v1"


class AgentCoordinationStatus(StrEnum):
    """Control-only outcomes emitted by the coordination graph."""

    DISPATCH_READY = "DISPATCH_READY"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETE = "COMPLETE"


class AgentCoordinationAssignment(ContractModel):
    """The next authorized role assignment without tool inputs or values."""

    graph_version: str = COORDINATION_GRAPH_VERSION
    status: AgentCoordinationStatus
    step_id: str | None = Field(default=None, min_length=1)
    role: AgentRole | None = None
    routed_role: AgentRole | None = None
    tool_name: str | None = Field(default=None, min_length=1)


class _CoordinationGraphState(TypedDict):
    run_id: str
    plan: dict[str, object]
    completed_step_ids: tuple[str, ...]
    pending_approval_step_ids: tuple[str, ...]
    status: str
    selected_step_id: str | None
    selected_role: str | None
    selected_tool_name: str | None
    routed_role: str | None


class AgentCoordinationRouter:
    """Route one authoritative plan prefix through role-specific graph nodes."""

    def __init__(self, *, checkpointer: BaseCheckpointSaver[str] | None = None) -> None:
        graph = StateGraph(_CoordinationGraphState)
        graph.add_node("load_route", self._supervisor)
        for role in AgentRole:
            graph.add_node(role.value, self._role_node(role))
        graph.add_edge(START, "load_route")
        graph.add_conditional_edges(
            "load_route",
            self._route,
            {**{role.value: role.value for role in AgentRole}, "finish": END},
        )
        for role in AgentRole:
            graph.add_edge(role.value, END)
        self._graph = graph.compile(checkpointer=checkpointer)

    def select_next(
        self,
        *,
        run_id: str,
        plan: AgentPlan,
        completed_step_ids: tuple[str, ...],
        pending_approval_step_ids: tuple[str, ...],
    ) -> AgentCoordinationAssignment:
        """Return the next role assignment using only non-numeric control state."""

        _validate_run_id(run_id)
        output = self._graph.invoke(
            {
                "run_id": run_id,
                "plan": cast(dict[str, object], plan.model_dump(mode="json")),
                "completed_step_ids": completed_step_ids,
                "pending_approval_step_ids": pending_approval_step_ids,
                "status": AgentCoordinationStatus.COMPLETE.value,
                "selected_step_id": None,
                "selected_role": None,
                "selected_tool_name": None,
                "routed_role": None,
            },
            config={
                "configurable": {
                    "thread_id": run_id,
                    "checkpoint_ns": COORDINATION_GRAPH_VERSION,
                }
            },
        )
        status = AgentCoordinationStatus(output["status"])
        role = _optional_role(output["selected_role"])
        routed_role = _optional_role(output["routed_role"])
        return AgentCoordinationAssignment(
            status=status,
            step_id=output["selected_step_id"],
            role=role,
            routed_role=routed_role,
            tool_name=output["selected_tool_name"],
        )

    @staticmethod
    def _supervisor(state: _CoordinationGraphState) -> dict[str, object]:
        plan = AgentPlan.model_validate(state["plan"])
        completed = tuple(state["completed_step_ids"])
        expected_prefix = tuple(step.step_id for step in plan.steps[: len(completed)])
        if completed != expected_prefix:
            raise ValueError("completed Agent steps must be a sequential prefix of the plan")
        pending = tuple(state["pending_approval_step_ids"])
        if len(pending) > 1:
            raise ValueError("only one Agent plan step may await approval")
        if len(completed) == len(plan.steps):
            if pending:
                raise ValueError("a completed Agent plan cannot await approval")
            return _selection(AgentCoordinationStatus.COMPLETE)
        next_step = plan.steps[len(completed)]
        if pending:
            if pending != (next_step.step_id,):
                raise ValueError("approval must reference the next incomplete Agent step")
            return _selection(
                AgentCoordinationStatus.AWAITING_APPROVAL,
                step_id=next_step.step_id,
                role=next_step.role,
                tool_name=next_step.tool_name,
            )
        return _selection(
            AgentCoordinationStatus.DISPATCH_READY,
            step_id=next_step.step_id,
            role=next_step.role,
            tool_name=next_step.tool_name,
        )

    @staticmethod
    def _role_node(role: AgentRole):  # type: ignore[no-untyped-def]
        def route_role(state: _CoordinationGraphState) -> dict[str, object]:
            if state["status"] != AgentCoordinationStatus.DISPATCH_READY.value:
                raise ValueError("professional Agent node requires a dispatch-ready assignment")
            if state["selected_role"] != role.value:
                raise ValueError("professional Agent route does not match the selected role")
            return {"routed_role": role.value}

        return route_role

    @staticmethod
    def _route(state: _CoordinationGraphState) -> str:
        if state["status"] != AgentCoordinationStatus.DISPATCH_READY.value:
            return "finish"
        selected_role = state["selected_role"]
        if selected_role not in {role.value for role in AgentRole}:
            raise ValueError("coordination graph selected an unsupported Agent role")
        return selected_role


def _selection(
    status: AgentCoordinationStatus,
    *,
    step_id: str | None = None,
    role: AgentRole | None = None,
    tool_name: str | None = None,
) -> dict[str, object]:
    return {
        "status": status.value,
        "selected_step_id": step_id,
        "selected_role": role.value if role is not None else None,
        "selected_tool_name": tool_name,
        "routed_role": None,
    }


def _optional_role(value: str | None) -> AgentRole | None:
    return AgentRole(value) if value is not None else None


def _validate_run_id(run_id: str) -> None:
    try:
        parsed = UUID(run_id)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("LangGraph thread_id must be a UUID") from exc
    if str(parsed) != run_id:
        raise ValueError("LangGraph thread_id must be canonical")


__all__ = [
    "COORDINATION_GRAPH_VERSION",
    "AgentCoordinationAssignment",
    "AgentCoordinationRouter",
    "AgentCoordinationStatus",
]
