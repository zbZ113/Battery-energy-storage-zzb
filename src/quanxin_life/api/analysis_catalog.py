"""Authenticated project catalogs and deterministic advanced analysis creation."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import Field

from quanxin_life.agents.supervisor import FIXED_ADVANCED_TOOLS
from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.agent_runs import (
    AgentRunAccessError,
    AgentRunConflictError,
    AgentRunDispatchError,
    AgentRunNotFoundError,
    AgentRunRecord,
    AgentRunResultRecord,
    AgentRunService,
    AgentRunStateError,
)
from quanxin_life.application.datasets import (
    DatasetNotFoundError,
    DatasetRecord,
    DatasetService,
    DatasetStateError,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    ProjectInvocationNotFoundError,
)
from quanxin_life.application.record_batch_bindings import (
    RecordBatchBindingNotFoundError,
    RecordBatchBindingRecord,
    RecordBatchBindingService,
    RecordBatchBindingStateError,
)
from quanxin_life.application.task_queue import AgentRunQueue
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import DatasetStatus, UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName


class AnalysisInputsRecord(ContractModel):
    project_id: str = Field(min_length=1)
    datasets: tuple[DatasetRecord, ...]
    batches: tuple[RecordBatchBindingRecord, ...]


class CreateAdvancedAnalysisRequest(ContractModel):
    record_batch_id: str = Field(min_length=1, max_length=200)
    cell_id: str = Field(min_length=1, max_length=200)
    cutoff_cycle: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class AnalysisCatalogHttpAdapter:
    router: APIRouter


def create_analysis_catalog_http_adapter(
    *,
    auth_adapter: AuthHttpAdapter,
    dataset_service: DatasetService,
    record_batch_service: RecordBatchBindingService,
    context_service: ProjectInvocationContextService,
    run_service: AgentRunService,
    available_tools: Collection[StandardToolName],
    queue: AgentRunQueue,
) -> AnalysisCatalogHttpAdapter:
    tools = tuple(available_tools)
    if not frozenset(tools) >= FIXED_ADVANCED_TOOLS:
        raise ValueError("analysis catalog requires the complete advanced tool set")
    router = APIRouter(tags=["analysis-catalog"])
    ready_principal = Depends(auth_adapter.require_ready_user)
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )
    operator_principal = Depends(operator_dependency)
    idempotency_header = Header(alias="Idempotency-Key")

    @router.get(
        "/v1/projects/{project_id}/datasets",
        response_model=list[DatasetRecord],
    )
    def list_project_datasets(
        project_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return list(dataset_service.list_project_datasets(principal, project_id))
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except DatasetStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_dataset_state") from exc

    @router.get(
        "/v1/datasets/{dataset_id}/batches",
        response_model=list[RecordBatchBindingRecord],
    )
    def list_dataset_batches(
        dataset_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return list(record_batch_service.list_dataset_batches(principal, dataset_id))
        except RecordBatchBindingNotFoundError as exc:
            raise HTTPException(status_code=404, detail="dataset_not_found") from exc
        except RecordBatchBindingStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_record_batch_state") from exc

    @router.get(
        "/v1/projects/{project_id}/analysis-inputs",
        response_model=AnalysisInputsRecord,
    )
    def list_analysis_inputs(
        project_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            datasets = dataset_service.list_project_datasets(principal, project_id)
            selected_datasets: list[DatasetRecord] = []
            selected_batches: list[RecordBatchBindingRecord] = []
            for dataset in datasets:
                if dataset.status is not DatasetStatus.FROZEN:
                    continue
                batches = record_batch_service.list_dataset_batches(
                    principal, dataset.dataset_id
                )
                if not batches:
                    continue
                selected_datasets.append(dataset)
                selected_batches.extend(batches)
            return AnalysisInputsRecord(
                project_id=project_id,
                datasets=tuple(selected_datasets),
                batches=tuple(selected_batches),
            )
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except RecordBatchBindingNotFoundError as exc:
            raise HTTPException(status_code=404, detail="dataset_not_found") from exc
        except (DatasetStateError, RecordBatchBindingStateError) as exc:
            raise HTTPException(status_code=500, detail="invalid_analysis_inputs") from exc

    @router.get(
        "/v1/projects/{project_id}/agent/runs",
        response_model=list[AgentRunRecord],
    )
    def list_project_runs(
        project_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return list(run_service.list_project_runs(principal, project_id))
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_agent_run_state") from exc

    @router.get(
        "/v1/agent/runs/{run_id}/results",
        response_model=list[AgentRunResultRecord],
    )
    def list_run_results(
        run_id: str,
        principal: AuthPrincipal = ready_principal,
    ) -> Any:
        try:
            return list(run_service.list_results(principal, run_id))
        except AgentRunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="agent_run_not_found") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_agent_result_state") from exc

    @router.post(
        "/v1/projects/{project_id}/advanced-analyses",
        response_model=AgentRunRecord,
        status_code=202,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_advanced_analysis(
        project_id: str,
        payload: CreateAdvancedAnalysisRequest,
        idempotency_key: str = idempotency_header,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            context = context_service.resolve_http(principal, project_id)
            batch = record_batch_service.resolve_verified_early_cycle_batch(
                context, payload.record_batch_id
            )
            if (
                batch.metadata.cell_id != payload.cell_id
                or batch.feature_config.cutoff_cycle != payload.cutoff_cycle
            ):
                raise HTTPException(status_code=409, detail="analysis_input_conflict")
            record = run_service.create_fixed_advanced_run(
                principal,
                project_id=project_id,
                record_batch_id=payload.record_batch_id,
                idempotency_key=idempotency_key,
                available_tools=tools,
                now=datetime.now(UTC),
            )
            try:
                return run_service.dispatch_pending(
                    record.run_id,
                    queue=queue,
                    now=datetime.now(UTC),
                )
            except AgentRunDispatchError:
                return run_service.get_run(principal, record.run_id)
        except HTTPException:
            raise
        except ProjectInvocationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except (
            ProjectInvocationNotFoundError,
            RecordBatchBindingNotFoundError,
            AgentRunNotFoundError,
        ) as exc:
            raise HTTPException(status_code=404, detail="analysis_scope_not_found") from exc
        except RecordBatchBindingStateError as exc:
            raise HTTPException(status_code=409, detail="analysis_input_state_conflict") from exc
        except AgentRunAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AgentRunConflictError as exc:
            raise HTTPException(status_code=409, detail="idempotency_conflict") from exc
        except AgentRunStateError as exc:
            raise HTTPException(status_code=409, detail="agent_run_state_conflict") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_advanced_analysis") from exc

    return AnalysisCatalogHttpAdapter(router=router)


__all__ = [
    "AnalysisCatalogHttpAdapter",
    "AnalysisInputsRecord",
    "CreateAdvancedAnalysisRequest",
    "create_analysis_catalog_http_adapter",
]
