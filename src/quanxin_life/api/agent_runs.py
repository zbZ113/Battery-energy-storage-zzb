"""Authenticated FastAPI and SSE transport for persistent Agent runs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from quanxin_life.agents.supervisor import SupervisorPlanningRequest
from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.agent_runs import (
    AgentRunAccessError,
    AgentRunConflictError,
    AgentRunDispatchError,
    AgentRunNotFoundError,
    AgentRunRecord,
    AgentRunService,
    AgentRunStateError,
)
from quanxin_life.application.task_queue import AgentRunQueue
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import AgentRunStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName

TERMINAL_AGENT_RUN_STATUSES = frozenset(
    {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED}
)


@dataclass(frozen=True, slots=True)
class AgentRunHttpAdapter:
    router: APIRouter


class ApprovalActionRequest(ContractModel):
    approval_id: str = Field(min_length=1, max_length=64)
    reason: str | None = Field(default=None, max_length=2_000)


def create_agent_run_http_adapter(
    service: AgentRunService,
    *,
    auth_adapter: AuthHttpAdapter,
    available_tools: Collection[StandardToolName],
    queue: AgentRunQueue,
    event_poll_seconds: float = 1.0,
) -> AgentRunHttpAdapter:
    """Build run creation, read, cancellation and resumable SSE routes."""

    tools = tuple(available_tools)
    if not tools:
        raise ValueError("Agent run API requires at least one available tool")
    if event_poll_seconds <= 0:
        raise ValueError("event_poll_seconds must be positive")
    router = APIRouter(prefix="/v1/agent/runs", tags=["agent-runs"])
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )
    operator_principal = Depends(operator_dependency)
    ready_principal = Depends(auth_adapter.require_ready_user)
    idempotency_header = Header(alias="Idempotency-Key")
    last_event_header = Header(default=None, alias="Last-Event-ID")

    @router.post(
        "",
        response_model=AgentRunRecord,
        status_code=202,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_run(
        payload: SupervisorPlanningRequest,
        idempotency_key: str = idempotency_header,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            record = service.create_run(
                principal,
                request=payload,
                idempotency_key=idempotency_key,
                available_tools=tools,
                now=datetime.now(UTC),
            )
            try:
                return service.dispatch_pending(
                    record.run_id,
                    queue=queue,
                    now=datetime.now(UTC),
                )
            except AgentRunDispatchError:
                # The durable outbox remains PENDING and a scheduler may retry it.
                return service.get_run(principal, record.run_id)
        except AgentRunAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run_scope_not_found") from exc
        except AgentRunConflictError as exc:
            raise HTTPException(status_code=409, detail="idempotency_conflict") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=409, detail="agent_run_state_conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_agent_run") from exc

    @router.get("/{run_id}", response_model=AgentRunRecord)
    def get_run(
        run_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return service.get_run(principal, run_id)
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="agent_run_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_agent_run_state") from exc

    @router.get("/{run_id}/results/{result_id}")
    def get_run_result(
        run_id: str,
        result_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return service.get_result(principal, run_id, result_id)
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="agent_result_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_agent_result_state") from exc

    @router.post(
        "/{run_id}/cancel",
        response_model=AgentRunRecord,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def cancel_run(
        run_id: str,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            return service.cancel_run(principal, run_id, now=datetime.now(UTC))
        except AgentRunAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="agent_run_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=409, detail="agent_run_state_conflict") from exc

    @router.post(
        "/{run_id}/approve",
        response_model=AgentRunRecord,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def approve_run(
        run_id: str,
        payload: ApprovalActionRequest,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            record = service.approve_run(
                principal,
                run_id,
                approval_id=payload.approval_id,
                reason=payload.reason,
                now=datetime.now(UTC),
            )
            try:
                return service.dispatch_pending(
                    record.run_id,
                    queue=queue,
                    now=datetime.now(UTC),
                )
            except AgentRunDispatchError:
                return service.get_run(principal, record.run_id)
        except AgentRunAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="approval_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=409, detail="approval_state_conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_approval") from exc

    @router.post(
        "/{run_id}/reject",
        response_model=AgentRunRecord,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def reject_run(
        run_id: str,
        payload: ApprovalActionRequest,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            return service.reject_run(
                principal,
                run_id,
                approval_id=payload.approval_id,
                reason=payload.reason,
                now=datetime.now(UTC),
            )
        except AgentRunAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="approval_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=409, detail="approval_state_conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_approval") from exc

    @router.get("/{run_id}/events")
    async def stream_events(
        run_id: str,
        principal: AuthPrincipal = ready_principal,
        last_event_id: str | None = last_event_header,
    ) -> StreamingResponse:
        try:
            cursor = _last_event_sequence(last_event_id)
            service.get_run(principal, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_last_event_id") from exc
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="agent_run_not_found") from exc

        async def event_source() -> AsyncIterator[str]:
            nonlocal cursor
            while True:
                events = service.list_events(
                    principal,
                    run_id,
                    after_sequence=cursor,
                )
                for event in events:
                    cursor = event.sequence
                    yield (
                        f"id: {event.sequence}\n"
                        f"event: {event.event_type.value}\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )
                record = service.get_run(principal, run_id)
                if record.status in TERMINAL_AGENT_RUN_STATUSES:
                    break
                if not events:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(event_poll_seconds)

        return StreamingResponse(
            event_source(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
            },
        )

    return AgentRunHttpAdapter(router=router)


def _last_event_sequence(value: str | None) -> int:
    if value is None:
        return 0
    normalized = value.strip()
    if not normalized or not normalized.isdecimal():
        raise ValueError("Last-Event-ID must be a non-negative integer")
    return int(normalized)


__all__ = ["AgentRunHttpAdapter", "create_agent_run_http_adapter"]
