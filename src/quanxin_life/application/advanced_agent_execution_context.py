"""Server-owned Advanced calibration references for persisted Agent execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from sqlalchemy import select

from quanxin_life.agents.execution_adapter import AgentExecutionContextResolver
from quanxin_life.audit import AuditLedgerError, SqlProjectAuditLedger
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelRouteRole,
    AdvancedModelTask,
    AgentIntent,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    AdvancedCalibrationSampleBinding,
    AgentRun,
)

if TYPE_CHECKING:
    from quanxin_life.application.advanced_runtime import VerifiedAdvancedRuntime
    from quanxin_life.application.invocation_context import (
        ProjectInvocationContextService,
        VerifiedProjectInvocationContext,
    )
    from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

_RUL_CALIBRATION_REFERENCE = "context.rul_calibration_sample_result_ids"
_SOH_CALIBRATION_REFERENCE = "context.soh_calibration_sample_result_ids"
_CALIBRATION_REFERENCES = frozenset(
    {_RUL_CALIBRATION_REFERENCE, _SOH_CALIBRATION_REFERENCE}
)
_SAMPLE_MANIFEST_SCHEMA_VERSION = "advanced-calibration-sample-manifest-v1"


class AdvancedAgentExecutionContextError(ValueError):
    """Raised when an Agent run cannot resolve trusted calibration evidence."""


class BoundRecordBatchResolver(Protocol):
    def resolve_verified_early_cycle_batch(
        self,
        context: VerifiedProjectInvocationContext,
        record_batch_id: str,
    ) -> VerifiedEarlyCycleBatch: ...


class ActiveAdvancedRuntimeResolver(Protocol):
    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedAdvancedRuntime: ...

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
        runtime: VerifiedAdvancedRuntime,
    ) -> VerifiedAdvancedRuntime: ...


class AdvancedAgentExecutionContextResolver:
    """Resolve calibration result IDs only from live persisted server state."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_service: ProjectInvocationContextService,
        record_batch_resolver: BoundRecordBatchResolver,
        runtime_resolver: ActiveAdvancedRuntimeResolver,
        delegate: AgentExecutionContextResolver,
    ) -> None:
        self._session_factory = session_factory
        self._context_service = context_service
        self._record_batch_resolver = record_batch_resolver
        self._runtime_resolver = runtime_resolver
        self._delegate = delegate
        self._ledger = SqlProjectAuditLedger(
            session_factory,
            context_validator=context_service,
        )

    def resolve_dataset_artifact(
        self,
        *,
        run_id: str,
        project_id: str,
        dataset_id: str,
    ) -> object:
        context = self._agent_context(run_id, project_id)
        batch = self._resolve_batch(context, dataset_id)
        if batch.record_batch_id != dataset_id:
            raise AdvancedAgentExecutionContextError(
                "Agent target record batch identity changed"
            )
        return dataset_id

    def resolve_context(
        self,
        *,
        run_id: str,
        project_id: str,
        reference: str,
    ) -> object:
        if reference not in _CALIBRATION_REFERENCES:
            return self._delegate.resolve_context(
                run_id=run_id,
                project_id=project_id,
                reference=reference,
            )
        context = self._agent_context(run_id, project_id)
        record_batch_id = self._target_record_batch_id(run_id, project_id)
        batch = self._resolve_batch(context, record_batch_id)
        task, role = _reference_route(reference, batch.feature_config.cutoff_cycle)
        runtime = self._runtime_resolver.resolve(
            context,
            task=task,
            cutoff_cycle=batch.feature_config.cutoff_cycle,
            role=role,
        )
        self._require_target_runtime(context, batch, runtime, task=task, role=role)
        result_ids = self._ready_result_ids(
            context,
            target_cell_id=batch.metadata.cell_id,
            runtime=runtime,
        )
        try:
            revalidated = self._runtime_resolver.revalidate(context, runtime)
        except (LookupError, RuntimeError, ValueError) as exc:
            raise AdvancedAgentExecutionContextError(
                "Advanced runtime route changed during calibration validation"
            ) from exc
        if _runtime_identity(revalidated) != _runtime_identity(runtime):
            raise AdvancedAgentExecutionContextError(
                "Advanced runtime route changed during calibration validation"
            )
        return result_ids

    def _agent_context(
        self,
        run_id: str,
        project_id: str,
    ) -> VerifiedProjectInvocationContext:
        try:
            context = self._context_service.resolve_agent_run(run_id)
        except (RuntimeError, ValueError) as exc:
            raise AdvancedAgentExecutionContextError(
                "Agent run context is not authorized"
            ) from exc
        if context.project_id != project_id:
            raise AdvancedAgentExecutionContextError(
                "Agent run project identity changed"
            )
        return context

    def _target_record_batch_id(self, run_id: str, project_id: str) -> str:
        with session_scope(self._session_factory) as session:
            run = session.get(AgentRun, run_id)
            if run is None or run.project_id != project_id:
                raise AdvancedAgentExecutionContextError(
                    "Agent run target batch is unavailable"
                )
            try:
                intent = AgentIntent.model_validate(run.intent_json)
            except (TypeError, ValueError) as exc:
                raise AdvancedAgentExecutionContextError(
                    "Agent run intent is invalid"
                ) from exc
        if intent.project_id != project_id or len(intent.dataset_ids) != 1:
            raise AdvancedAgentExecutionContextError(
                "Advanced Agent execution requires one bound target batch"
            )
        return intent.dataset_ids[0]

    def _resolve_batch(
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
            raise AdvancedAgentExecutionContextError(
                "Agent target record batch failed verification"
            ) from exc
        if batch.record_batch_id != record_batch_id:
            raise AdvancedAgentExecutionContextError(
                "Agent target record batch identity changed"
            )
        return batch

    @staticmethod
    def _require_target_runtime(
        context: VerifiedProjectInvocationContext,
        batch: VerifiedEarlyCycleBatch,
        runtime: VerifiedAdvancedRuntime,
        *,
        task: AdvancedModelTask,
        role: AdvancedModelRouteRole,
    ) -> None:
        if (
            runtime.project_id != context.project_id
            or runtime.task is not task
            or runtime.cutoff_cycle != batch.feature_config.cutoff_cycle
            or runtime.role is not role
            or runtime.dataset_id != batch.metadata.dataset_id
            or runtime.data_version != batch.data_version
            or runtime.feature_version != batch.feature_config.feature_version
            or runtime.split_version != batch.split_version
        ):
            raise AdvancedAgentExecutionContextError(
                "Advanced target runtime does not match the bound record batch"
            )

    def _ready_result_ids(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        target_cell_id: str,
        runtime: VerifiedAdvancedRuntime,
    ) -> tuple[str, ...]:
        with session_scope(self._session_factory) as session:
            materializations = tuple(
                session.scalars(
                    select(AdvancedCalibrationMaterialization).where(
                        AdvancedCalibrationMaterialization.project_id
                        == context.project_id,
                        AdvancedCalibrationMaterialization.task
                        == runtime.task.value,
                        AdvancedCalibrationMaterialization.cutoff_cycle
                        == runtime.cutoff_cycle,
                        AdvancedCalibrationMaterialization.route_role
                        == runtime.role.value,
                        AdvancedCalibrationMaterialization.status
                        == AdvancedCalibrationMaterializationStatus.READY.value,
                        AdvancedCalibrationMaterialization.data_version
                        == runtime.data_version,
                        AdvancedCalibrationMaterialization.split_version
                        == runtime.split_version,
                        AdvancedCalibrationMaterialization.feature_version
                        == runtime.feature_version,
                        AdvancedCalibrationMaterialization.artifact_id
                        == runtime.artifact_id,
                        AdvancedCalibrationMaterialization.artifact_manifest_sha256
                        == runtime.artifact_manifest_sha256,
                        AdvancedCalibrationMaterialization.normalization_statistics_sha256
                        == runtime.normalization_sha256,
                        AdvancedCalibrationMaterialization.decision_event_id
                        == runtime.decision_event_id,
                        AdvancedCalibrationMaterialization.ledger_sequence_number
                        == runtime.ledger_sequence_number,
                        AdvancedCalibrationMaterialization.ledger_head_sha256
                        == runtime.ledger_head_sha256,
                    )
                )
            )
            if len(materializations) != 1:
                raise AdvancedAgentExecutionContextError(
                    "exact READY Advanced calibration materialization is unavailable"
                )
            materialization = materializations[0]
            bindings = tuple(
                session.scalars(
                    select(AdvancedCalibrationSampleBinding)
                    .where(
                        AdvancedCalibrationSampleBinding.materialization_id
                        == materialization.id
                    )
                    .order_by(AdvancedCalibrationSampleBinding.ordinal)
                )
            )
            session.expunge(materialization)
            for binding in bindings:
                session.expunge(binding)

        if (
            len(bindings) != materialization.sample_count
            or not bindings
            or tuple(binding.ordinal for binding in bindings)
            != tuple(range(len(bindings)))
        ):
            raise AdvancedAgentExecutionContextError(
                "Advanced calibration sample count or ordinal manifest is invalid"
            )
        if any(binding.cell_id == target_cell_id for binding in bindings):
            raise AdvancedAgentExecutionContextError(
                "Agent target cell belongs to the calibration cohort"
            )

        entries: list[dict[str, object]] = []
        result_ids: list[str] = []
        for binding in bindings:
            result = self._resolve_sample_result(
                context,
                materialization=materialization,
                runtime=runtime,
                binding=binding,
            )
            sample_sha256 = sha256_canonical(result.model_dump(mode="json"))
            if binding.sample_sha256 != sample_sha256:
                raise AdvancedAgentExecutionContextError(
                    "Advanced calibration sample binding content is invalid"
                )
            entries.append(
                {
                    "ordinal": binding.ordinal,
                    "cell_id": binding.cell_id,
                    "result_id": binding.result_id,
                    "sample_sha256": binding.sample_sha256,
                }
            )
            result_ids.append(binding.result_id)
        expected_manifest = _manifest_sha256(
            materialization,
            runtime=runtime,
            entries=tuple(entries),
        )
        if materialization.sample_manifest_sha256 != expected_manifest:
            raise AdvancedAgentExecutionContextError(
                "Advanced calibration sample manifest is invalid"
            )
        return tuple(result_ids)

    def _resolve_sample_result(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization: AdvancedCalibrationMaterialization,
        runtime: VerifiedAdvancedRuntime,
        binding: AdvancedCalibrationSampleBinding,
    ) -> ToolResult:
        from quanxin_life.tools.advanced_conformal import (
            validate_advanced_calibration_sample_result,
        )

        try:
            result = self._ledger.resolve_registered_result(
                context,
                binding.result_id,
            )
            validate_advanced_calibration_sample_result(
                result,
                task=runtime.task,
            )
        except (AuditLedgerError, TypeError, ValueError) as exc:
            raise AdvancedAgentExecutionContextError(
                "Advanced calibration ToolResult or provenance content is invalid"
            ) from exc
        artifact = result.values.get("artifact")
        expected = {
            "materialization_id": materialization.id,
            "source_registration_id": materialization.source_registration_id,
            "source_identity_sha256": materialization.source_identity_sha256,
            "split_partition": "calibration",
            "cell_id": binding.cell_id,
            "task": runtime.task.value,
            "route_role": runtime.role.value,
            "artifact_id": runtime.artifact_id,
            "artifact_manifest_sha256": runtime.artifact_manifest_sha256,
            "model_version": runtime.model_version,
            "dataset_id": runtime.dataset_id,
            "cutoff_cycle": runtime.cutoff_cycle,
            "data_version": runtime.data_version,
            "feature_version": runtime.feature_version,
            "split_version": runtime.split_version,
            "normalization_statistics_sha256": runtime.normalization_sha256,
            "decision_event_id": runtime.decision_event_id,
            "ledger_sequence_number": runtime.ledger_sequence_number,
            "ledger_head_sha256": runtime.ledger_head_sha256,
        }
        if (
            not isinstance(artifact, dict)
            or any(artifact.get(name) != value for name, value in expected.items())
            or result.result_id != binding.result_id
            or result.model_version != runtime.model_version
            or result.data_version != runtime.data_version
            or result.feature_version != runtime.feature_version
        ):
            raise AdvancedAgentExecutionContextError(
                "Advanced calibration ToolResult identity differs from its binding"
            )
        return result


def _reference_route(
    reference: str,
    cutoff_cycle: int,
) -> tuple[AdvancedModelTask, AdvancedModelRouteRole]:
    if reference == _RUL_CALIBRATION_REFERENCE:
        role = (
            AdvancedModelRouteRole.DEFAULT
            if cutoff_cycle == 20
            else AdvancedModelRouteRole.COVERAGE
        )
        return AdvancedModelTask.RUL, role
    if reference == _SOH_CALIBRATION_REFERENCE:
        return AdvancedModelTask.SOH, AdvancedModelRouteRole.MEAN_ACCURACY
    raise AdvancedAgentExecutionContextError(
        "Advanced calibration context reference is not supported"
    )


def _manifest_sha256(
    materialization: AdvancedCalibrationMaterialization,
    *,
    runtime: VerifiedAdvancedRuntime,
    entries: tuple[dict[str, object], ...],
) -> str:
    return sha256_canonical(
        {
            "schema_version": _SAMPLE_MANIFEST_SCHEMA_VERSION,
            "materialization_id": materialization.id,
            "project_id": materialization.project_id,
            "task": runtime.task.value,
            "cutoff_cycle": runtime.cutoff_cycle,
            "route_role": runtime.role.value,
            "data_version": runtime.data_version,
            "split_version": runtime.split_version,
            "feature_version": runtime.feature_version,
            "artifact_id": runtime.artifact_id,
            "artifact_manifest_sha256": runtime.artifact_manifest_sha256,
            "model_version": runtime.model_version,
            "normalization_statistics_sha256": runtime.normalization_sha256,
            "decision_event_id": runtime.decision_event_id,
            "ledger_sequence_number": runtime.ledger_sequence_number,
            "ledger_head_sha256": runtime.ledger_head_sha256,
            "source_registration_id": materialization.source_registration_id,
            "source_identity_sha256": materialization.source_identity_sha256,
            "request_sha256": materialization.request_sha256,
            "samples": list(entries),
        }
    )


def _runtime_identity(runtime: VerifiedAdvancedRuntime) -> tuple[object, ...]:
    return (
        runtime.project_id,
        runtime.task,
        runtime.cutoff_cycle,
        runtime.role,
        runtime.output_target,
        runtime.artifact_kind,
        runtime.dataset_id,
        runtime.data_version,
        runtime.feature_version,
        runtime.split_version,
        runtime.normalization_sha256,
        runtime.artifact_id,
        runtime.artifact_manifest_sha256,
        runtime.model_version,
        runtime.decision_event_id,
        runtime.ledger_sequence_number,
        runtime.ledger_head_sha256,
    )


__all__ = [
    "AdvancedAgentExecutionContextError",
    "AdvancedAgentExecutionContextResolver",
]
