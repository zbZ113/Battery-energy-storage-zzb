"""Application port for dispatching durable Agent runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class AgentRunDispatchReceipt:
    """Identity-only acknowledgement returned after queue dispatch."""

    run_id: str
    task_id: str


class AgentRunQueue(Protocol):
    """Queue boundary; PostgreSQL remains the source of task state and results."""

    def enqueue(self, *, run_id: str, plan_hash: str) -> AgentRunDispatchReceipt: ...


__all__ = ["AgentRunDispatchReceipt", "AgentRunQueue"]
