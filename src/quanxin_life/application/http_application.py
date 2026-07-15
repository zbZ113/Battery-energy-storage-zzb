"""Competition FastAPI composition over one shared in-process application state."""

from __future__ import annotations

from typing import Any

from quanxin_life.api.app import create_fastapi_app
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
) -> Any:
    """Wire HTTP, tools, uploads and workflow to the same trusted state."""

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
    )
