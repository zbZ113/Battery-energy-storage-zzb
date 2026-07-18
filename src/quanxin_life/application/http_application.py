"""Competition FastAPI composition over one shared in-process application state."""

from __future__ import annotations

from typing import Any

from quanxin_life.api.agent_runs import AgentRunHttpAdapter
from quanxin_life.api.app import create_fastapi_app
from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.api.datasets import DatasetHttpAdapter
from quanxin_life.api.experiments import ExperimentHttpAdapter
from quanxin_life.api.knowledge import KnowledgeHttpAdapter
from quanxin_life.api.projects import ProjectHttpAdapter
from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.assembly import (
    CompetitionToolDependencies,
    create_competition_tool_invocation_service,
)
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    VerifiedEarlyCycleBatchStore,
)
from quanxin_life.application.lifetime_workflow import (
    LifetimeDecisionWorkflowRequest,
    LifetimeDecisionWorkflowResult,
    run_lifetime_decision_workflow,
)


def create_competition_fastapi_app(
    dependencies: CompetitionToolDependencies,
    *,
    batch_store: VerifiedEarlyCycleBatchStore,
    auth_adapter: AuthHttpAdapter,
    project_adapter: ProjectHttpAdapter,
    dataset_adapter: DatasetHttpAdapter,
    experiment_adapter: ExperimentHttpAdapter,
    agent_run_adapter: AgentRunHttpAdapter,
    knowledge_adapter: KnowledgeHttpAdapter,
) -> Any:
    """Wire authenticated HTTP, tools, uploads and workflow to one trusted state."""

    if auth_adapter is None:
        raise ValueError("auth_adapter is required for the competition HTTP application")
    if project_adapter is None:
        raise ValueError("project_adapter is required for the competition HTTP application")
    if dataset_adapter is None:
        raise ValueError("dataset_adapter is required for the competition HTTP application")
    if experiment_adapter is None:
        raise ValueError(
            "experiment_adapter is required for the competition HTTP application"
        )
    if agent_run_adapter is None:
        raise ValueError("agent_run_adapter is required for the competition HTTP application")
    if knowledge_adapter is None:
        raise ValueError("knowledge_adapter is required for the competition HTTP application")

    service = create_competition_tool_invocation_service(dependencies)

    def run_workflow(
        received_service: ToolInvocationService,
        request: LifetimeDecisionWorkflowRequest,
    ) -> LifetimeDecisionWorkflowResult:
        if received_service is not service:
            raise ValueError("lifetime workflow must use the assembled competition service")
        return run_lifetime_decision_workflow(
            received_service,
            request,
            batch_resolver=batch_store,
        )

    def register_csv(
        payload: bytes,
        registration: CanonicalCsvBatchRegistration,
    ) -> str:
        return batch_store.register_canonical_csv(
            payload,
            registration=registration,
        )

    return create_fastapi_app(
        service,
        lifetime_workflow_runner=run_workflow,
        canonical_csv_registrar=register_csv,
        auth_adapter=auth_adapter,
        project_adapter=project_adapter,
        dataset_adapter=dataset_adapter,
        experiment_adapter=experiment_adapter,
        agent_run_adapter=agent_run_adapter,
        knowledge_adapter=knowledge_adapter,
    )
