"""Create durable Aily scenario contexts from verified batch references only."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID, uuid4

from quanxin_life.api.aily import (
    AilyCompareScenarioContextRequest,
    AilyCreateScenarioContextRequest,
    AilyProjectLifetimeScenarioContextRequest,
    AilyScenarioContextState,
)
from quanxin_life.application.ingestion import VerifiedEarlyCycleBatchStore
from quanxin_life.scenarios import (
    BlastRouteManifest,
    BlastRouteRejected,
    ScenarioCellDescriptor,
    VerifiedScenarioContext,
    load_packaged_blast_route_catalog,
)
from quanxin_life.tools.blast_scenarios import (
    CompareOperationScenariosToolInput,
    ProjectStorageLifetimeToolInput,
)
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

from .scenario_contexts import (
    ScenarioAnalysisInput,
    SqlAlchemyFeishuScenarioContextStore,
)
from .workflow import FeishuAnalysisTask

Clock = Callable[[], datetime]
UuidFactory = Callable[[], UUID]


class AilyScenarioReferenceUseAuthorizer(Protocol):
    """Server-owned approval for nonexact BLAST reference use."""

    def authorize_reference_use(
        self,
        *,
        task: FeishuAnalysisTask,
        data_batch_id: str,
        batch: VerifiedEarlyCycleBatch,
        route: BlastRouteManifest,
    ) -> bool: ...


class RejectingAilyScenarioReferenceUseAuthorizer:
    """Default policy: exact model references only."""

    def authorize_reference_use(self, **_: object) -> bool:
        return False


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqlAlchemyAilyScenarioContextGateway:
    """Resolve trusted batch metadata and persist a typed scenario template."""

    def __init__(
        self,
        *,
        context_store: SqlAlchemyFeishuScenarioContextStore,
        batch_store: VerifiedEarlyCycleBatchStore,
        created_by_reference: str,
        reference_use_authorizer: AilyScenarioReferenceUseAuthorizer | None = None,
        clock: Clock = _utc_now,
        uuid_factory: UuidFactory = uuid4,
    ) -> None:
        if not created_by_reference.strip():
            raise ValueError("created_by_reference must not be blank")
        if not callable(clock) or not callable(uuid_factory):
            raise TypeError("clock and uuid_factory must be callable")
        self._context_store = context_store
        self._batch_store = batch_store
        self._created_by_reference = created_by_reference
        self._reference_use_authorizer = (
            reference_use_authorizer
            or RejectingAilyScenarioReferenceUseAuthorizer()
        )
        self._clock = clock
        self._uuid_factory = uuid_factory

    def create_scenario_context(
        self,
        request: AilyCreateScenarioContextRequest,
    ) -> AilyScenarioContextState:
        batch = self._batch_store.resolve_verified_early_cycle_batch(
            request.data_batch_id
        )
        cell = ScenarioCellDescriptor(
            chemistry=batch.metadata.chemistry,
            nominal_capacity_ah=batch.metadata.nominal_capacity_ah,
            cell_format=request.cell_format,
        )
        route, trusted_reference_use = self._select_route(
            request=request,
            batch=batch,
            cell=cell,
        )
        context_id = str(self._uuid_factory())
        verified_context = VerifiedScenarioContext(
            scenario_context_id=context_id,
            cell=cell,
            trusted_reference_use=trusted_reference_use,
            data_version=batch.data_version,
            provenance=batch.provenance,
        )
        analysis_input: ScenarioAnalysisInput
        if isinstance(request, AilyCompareScenarioContextRequest):
            analysis_input = CompareOperationScenariosToolInput(
                run_id=context_id,
                scenario_context_id=context_id,
                route_id=route.route_id,
                cell=cell,
                baseline=request.baseline,
                comparisons=request.comparisons,
                current_state_reference=request.current_state_reference,
            )
        elif isinstance(request, AilyProjectLifetimeScenarioContextRequest):
            analysis_input = ProjectStorageLifetimeToolInput(
                run_id=context_id,
                scenario_context_id=context_id,
                route_id=route.route_id,
                cell=cell,
                scenario=request.scenario,
                current_state_reference=request.current_state_reference,
                new_observation_reference=request.new_observation_reference,
            )
        else:  # pragma: no cover - discriminated union invariant
            raise TypeError("unsupported Aily scenario request")
        record = self._context_store.create(
            task=FeishuAnalysisTask(request.task_type),
            data_batch_id=request.data_batch_id,
            verified_context=verified_context,
            analysis_input=analysis_input,
            created_by_reference=self._created_by_reference,
            created_at=self._clock(),
        )
        return AilyScenarioContextState(
            scenario_context_id=record.scenario_context_id,
            task_type=record.task,
            data_batch_id=record.data_batch_id,
            input_sha256=record.input_sha256,
            created_at=record.created_at,
        )

    def _select_route(
        self,
        *,
        request: AilyCreateScenarioContextRequest,
        batch: VerifiedEarlyCycleBatch,
        cell: ScenarioCellDescriptor,
    ) -> tuple[BlastRouteManifest, bool]:
        catalog = load_packaged_blast_route_catalog()
        task = FeishuAnalysisTask(request.task_type)
        candidates: list[tuple[BlastRouteManifest, bool]] = []
        normalized_chemistry = cell.chemistry.casefold()
        for route in catalog.routes:
            if task.value not in route.task_types or route.cell_format != cell.cell_format:
                continue
            if normalized_chemistry not in {
                alias.casefold() for alias in route.chemistry_aliases
            }:
                continue
            try:
                approved = self._reference_use_authorizer.authorize_reference_use(
                    task=task,
                    data_batch_id=request.data_batch_id,
                    batch=batch,
                    route=route,
                )
            except Exception as exc:
                raise ValueError("scenario reference use authorization failed") from exc
            if not isinstance(approved, bool):
                raise ValueError("scenario reference use authorization is invalid")
            try:
                authorized = catalog.authorize_reference_use(
                    route_id=route.route_id,
                    chemistry=cell.chemistry,
                    nominal_capacity_ah=cell.nominal_capacity_ah,
                    cell_format=cell.cell_format,
                    trusted_reference_use=approved,
                )
            except BlastRouteRejected:
                continue
            candidates.append((authorized, approved))
        if len(candidates) != 1:
            raise ValueError("no unique reviewed scenario reference route is authorized")
        return candidates[0]


__all__ = [
    "AilyScenarioReferenceUseAuthorizer",
    "RejectingAilyScenarioReferenceUseAuthorizer",
    "SqlAlchemyAilyScenarioContextGateway",
]
