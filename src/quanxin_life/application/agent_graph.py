"""LangGraph control loop around the single-step safe Agent worker."""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from quanxin_life.agents.coordination_graph import (
    COORDINATION_GRAPH_VERSION,
    AgentCoordinationRouter,
    AgentCoordinationStatus,
)
from quanxin_life.application.agent_run_execution import AgentRunExecutionWorker
from quanxin_life.core import AgentRole, AgentRunState, AgentRunStatus


class _AgentGraphState(TypedDict):
    run_id: str
    plan_hash: str
    control_status: str
    selected_step_id: str | None
    selected_role: str | None


class AgentGraphRunner:
    """Route role nodes while delegating every tool attempt to the safe worker."""

    def __init__(
        self,
        *,
        worker: AgentRunExecutionWorker,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> None:
        self._worker = worker
        self._router = AgentCoordinationRouter()
        graph = StateGraph(_AgentGraphState)
        graph.add_node("load_route", self._load_route)
        for role in AgentRole:
            graph.add_node(role.value, self._advance_role(role))
        graph.add_edge(START, "load_route")
        graph.add_conditional_edges(
            "load_route",
            self._route,
            {**{role.value: role.value for role in AgentRole}, "finish": END},
        )
        for role in AgentRole:
            graph.add_edge(role.value, "load_route")
        self._graph = graph.compile(checkpointer=checkpointer)

    def execute(self, *, run_id: str, plan_hash: str) -> AgentRunState:
        """Run graph-controlled single steps until approval, failure or completion."""

        self._graph.invoke(
            {
                "run_id": run_id,
                "plan_hash": plan_hash,
                "control_status": AgentCoordinationStatus.COMPLETE.value,
                "selected_step_id": None,
                "selected_role": None,
            },
            config={
                "configurable": {
                    "thread_id": run_id,
                    "checkpoint_ns": COORDINATION_GRAPH_VERSION,
                },
                "recursion_limit": 50,
            },
        )
        _, state = self._worker.authoritative_snapshot(
            run_id=run_id,
            plan_hash=plan_hash,
        )
        return state

    def _load_route(self, state: _AgentGraphState) -> dict[str, object]:
        plan, run_state = self._worker.authoritative_snapshot(
            run_id=state["run_id"],
            plan_hash=state["plan_hash"],
        )
        if run_state.status in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }:
            return {
                "control_status": AgentCoordinationStatus.COMPLETE.value,
                "selected_step_id": None,
                "selected_role": None,
            }
        assignment = self._router.select_next(
            run_id=state["run_id"],
            plan=plan,
            completed_step_ids=run_state.completed_step_ids,
            pending_approval_step_ids=run_state.pending_approval_step_ids,
        )
        return {
            "control_status": assignment.status.value,
            "selected_step_id": assignment.step_id,
            "selected_role": assignment.role.value if assignment.role is not None else None,
        }

    def _advance_role(self, role: AgentRole):  # type: ignore[no-untyped-def]
        def advance(state: _AgentGraphState) -> dict[str, object]:
            if state["control_status"] != AgentCoordinationStatus.DISPATCH_READY.value:
                raise ValueError("professional Agent node requires a dispatch-ready state")
            if state["selected_role"] != role.value:
                raise ValueError("LangGraph role node does not match the authoritative route")
            self._worker.advance_once(
                run_id=state["run_id"],
                plan_hash=state["plan_hash"],
            )
            return {}

        return advance

    @staticmethod
    def _route(state: _AgentGraphState) -> str:
        if state["control_status"] != AgentCoordinationStatus.DISPATCH_READY.value:
            return "finish"
        selected_role = state["selected_role"]
        if selected_role not in {role.value for role in AgentRole}:
            raise ValueError("LangGraph selected an unsupported Agent role")
        return selected_role


__all__ = ["AgentGraphRunner"]
