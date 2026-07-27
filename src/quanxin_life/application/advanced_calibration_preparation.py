"""Freeze path-free runtime and source identities before calibration dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationMaterializationPreparation,
)
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask

if TYPE_CHECKING:
    from quanxin_life.application.advanced_calibration_evidence import (
        AdvancedCalibrationEvidence,
    )
    from quanxin_life.application.advanced_calibration_materialization import (
        AdvancedCalibrationMaterializationRequest,
        AdvancedCalibrationRuntimeResolver,
    )
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )


class _PreparationRuntime(Protocol):
    project_id: str
    task: AdvancedModelTask
    cutoff_cycle: int
    role: AdvancedModelRouteRole
    dataset_id: str
    data_version: str
    split_version: str
    feature_version: str
    artifact_id: str
    artifact_manifest_sha256: str
    model_version: str
    normalization_sha256: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str


class _EvidenceResolver(Protocol):
    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedCalibrationEvidence: ...


class ActiveAdvancedCalibrationPreparationResolver:
    """Resolve and freeze one exact active route plus registered source identity."""

    def __init__(
        self,
        *,
        runtime_resolver: AdvancedCalibrationRuntimeResolver,
        evidence_resolver: _EvidenceResolver,
    ) -> None:
        self._runtime_resolver = runtime_resolver
        self._evidence_resolver = evidence_resolver

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        request: AdvancedCalibrationMaterializationRequest,
    ) -> AdvancedCalibrationMaterializationPreparation:
        runtime = self._runtime_resolver.resolve(
            context,
            task=request.task,
            cutoff_cycle=request.cutoff_cycle,
            role=request.route_role,
        )
        evidence = self._evidence_resolver.resolve(
            request.source_registration_id,
            task=request.task,
            cutoff_cycle=request.cutoff_cycle,
        )
        source = evidence.source_identity
        if (
            context.project_id != request.project_id
            or runtime.project_id != request.project_id
            or runtime.task is not request.task
            or runtime.cutoff_cycle != request.cutoff_cycle
            or runtime.role is not request.route_role
            or runtime.dataset_id != "MATR"
            or source.dataset_id != "MATR"
            or source.registration_id != request.source_registration_id
            or runtime.data_version != source.data_version
            or runtime.split_version != source.split_version
        ):
            raise ValueError(
                "Advanced runtime or source identity does not match the "
                "materialization request"
            )
        return AdvancedCalibrationMaterializationPreparation(
            data_version=runtime.data_version,
            split_version=runtime.split_version,
            feature_version=runtime.feature_version,
            artifact_id=runtime.artifact_id,
            artifact_manifest_sha256=runtime.artifact_manifest_sha256,
            model_version=runtime.model_version,
            normalization_statistics_sha256=runtime.normalization_sha256,
            decision_event_id=runtime.decision_event_id,
            ledger_sequence_number=runtime.ledger_sequence_number,
            ledger_head_sha256=runtime.ledger_head_sha256,
            source_registration_id=source.registration_id,
            source_identity_sha256=source.source_identity_sha256,
        )


__all__ = ["ActiveAdvancedCalibrationPreparationResolver"]
