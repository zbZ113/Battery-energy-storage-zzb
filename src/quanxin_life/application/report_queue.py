"""Identity-only dispatch contract for asynchronous report generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ProjectReportDispatchReceipt:
    report_id: str
    task_id: str


class ProjectReportQueue(Protocol):
    def enqueue(self, *, report_id: str) -> ProjectReportDispatchReceipt: ...


__all__ = ["ProjectReportDispatchReceipt", "ProjectReportQueue"]
