"""Compile trusted Advanced Agent context references from server policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from deploy.competition_inputs import AdvancedAgentRuntimePolicy
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask, AgentIntent
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import AgentRun

if TYPE_CHECKING:
    from quanxin_life.application.invocation_context import (
        ProjectInvocationContextService,
        VerifiedProjectInvocationContext,
    )
    from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

_SUPPORTED_CUTOFFS: Final = frozenset({20, 50, 100, 150})


class BoundRecordBatchResolver(Protocol):
    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch: ...


class CompetitionAgentPolicyContextResolver:
    """Derive every non-result Agent reference from persisted server state."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_service: ProjectInvocationContextService,
        record_batch_resolver: BoundRecordBatchResolver,
        policy: AdvancedAgentRuntimePolicy,
    ) -> None:
        self._session_factory = session_factory
        self._context_service = context_service
        self._record_batch_resolver = record_batch_resolver
        self._policy = AdvancedAgentRuntimePolicy.model_validate(
            policy.model_dump(mode="json")
        )

    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        context, record_batch_id = self._target(run_id, project_id)
        if dataset_id != record_batch_id:
            raise ValueError("Agent dataset reference differs from its persisted target")
        batch = self._verified_batch(context, record_batch_id)
        return batch.record_batch_id

    def resolve_context(
        self,
        *,
        run_id: str,
        project_id: str,
        reference: str,
    ) -> object:
        context, record_batch_id = self._target(run_id, project_id)
        batch = self._verified_batch(context, record_batch_id)
        return resolve_advanced_context_reference(
            reference,
            cutoff_cycle=batch.feature_config.cutoff_cycle,
            policy=self._policy,
        )

    def _target(
        self,
        run_id: str,
        project_id: str,
    ) -> tuple[VerifiedProjectInvocationContext, str]:
        try:
            context = self._context_service.resolve_agent_run(run_id)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("Agent run context is not authorized") from exc
        if context.project_id != project_id:
            raise ValueError("Agent run project identity changed")

        with session_scope(self._session_factory) as session:
            row = session.get(AgentRun, run_id)
            if row is None or row.project_id != project_id:
                raise ValueError("Agent run target batch is unavailable")
            try:
                intent = AgentIntent.model_validate(row.intent_json)
            except (TypeError, ValueError) as exc:
                raise ValueError("Agent run intent is invalid") from exc
        if intent.project_id != project_id or len(intent.dataset_ids) != 1:
            raise ValueError("Advanced Agent execution requires one bound target batch")
        return context, intent.dataset_ids[0]

    def _verified_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch:
        try:
            batch = self._record_batch_resolver.resolve_verified_early_cycle_batch(
                context,
                record_batch_id,
            )
        except (LookupError, RuntimeError, ValueError) as exc:
            raise ValueError("Agent target record batch failed verification") from exc
        if batch.record_batch_id != record_batch_id:
            raise ValueError("Agent target record batch identity changed")
        return batch


def resolve_advanced_context_reference(
    reference: str,
    *,
    cutoff_cycle: int,
    policy: AdvancedAgentRuntimePolicy,
) -> object:
    """Resolve one planner reference without accepting caller-owned business values."""

    if cutoff_cycle not in _SUPPORTED_CUTOFFS:
        raise ValueError(
            f"cutoff_cycle must be one of {sorted(_SUPPORTED_CUTOFFS)}"
        )

    rul_default = (
        AdvancedModelRouteRole.DEFAULT
        if cutoff_cycle == 20
        else AdvancedModelRouteRole.POINT_ACCURACY
    )
    coverage_default = (
        AdvancedModelRouteRole.DEFAULT
        if cutoff_cycle == 20
        else AdvancedModelRouteRole.COVERAGE
    )
    values: dict[str, object] = {
        "context.rul_point_route_role": rul_default,
        "context.rul_coverage_route_role": coverage_default,
        "context.soh_route_role": AdvancedModelRouteRole.MEAN_ACCURACY,
        "context.rul_task": AdvancedModelTask.RUL,
        "context.soh_task": AdvancedModelTask.SOH,
        "context.conformal_calibrate_operation": "calibrate",
        "context.conformal_issue_operation": "issue",
        "context.conformal_alpha": policy.conformal_alpha,
    }
    return values[reference]


__all__ = [
    "CompetitionAgentPolicyContextResolver",
    "resolve_advanced_context_reference",
]
