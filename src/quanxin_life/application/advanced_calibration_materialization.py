"""Produce deterministic, route-bound Advanced calibration sample ToolResults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Protocol, cast
from uuid import UUID, uuid5

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationEvidence,
    AdvancedCalibrationEvidenceResolver,
    AdvancedCalibrationSourceIdentity,
    AdvancedRULCalibrationEvidence,
    AdvancedSOHCalibrationEvidence,
)
from quanxin_life.application.advanced_runtime import (
    VerifiedAdvancedRuntime,
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.advanced_split_conformal import (
    AdvancedConformalRuntimeIdentity,
    AdvancedRULCalibrationSample,
    AdvancedSOHCalibrationSample,
)
from quanxin_life.application.deep_model_artifacts import DeepArtifactKind
from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit.project_ledger import MaterializedProjectResult
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    UserRole,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION,
    ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
    ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName

_APPROVED_CUTOFFS = {20, 50, 100, 150}


class AdvancedCalibrationMaterializationRequest(ContractModel):
    """Identity-only request for one route-specific calibration cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: str = Field(min_length=1)
    task: AdvancedModelTask
    cutoff_cycle: int
    route_role: AdvancedModelRouteRole
    source_registration_id: str = Field(min_length=1, max_length=200)

    @field_validator("cutoff_cycle")
    @classmethod
    def require_approved_cutoff(cls, value: int) -> int:
        if value not in _APPROVED_CUTOFFS:
            raise ValueError("Advanced calibration requires an approved cutoff cycle")
        return value

    @model_validator(mode="after")
    def require_conformal_route(
        self,
    ) -> AdvancedCalibrationMaterializationRequest:
        if self.task is AdvancedModelTask.RUL:
            expected = (
                AdvancedModelRouteRole.DEFAULT
                if self.cutoff_cycle == 20
                else AdvancedModelRouteRole.COVERAGE
            )
            if self.route_role is not expected:
                raise ValueError(
                    "Advanced RUL calibration route is not conformal-compatible"
                )
        elif self.route_role not in {
            AdvancedModelRouteRole.MEAN_ACCURACY,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
        }:
            raise ValueError(
                "Advanced SOH calibration route is not conformal-compatible"
            )
        return self


ProducedAdvancedCalibrationSample = MaterializedProjectResult


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationCellInput:
    """Server-resolved label-free input for one registered calibration cell."""

    source_registration_id: str
    source_identity_sha256: str
    cell_id: str
    cutoff_cycle: int
    data_version: str
    feature_version: str
    split_version: str
    raw_sequence: EarlyCycleSequence
    initial_soh: float | None = None

    def __post_init__(self) -> None:
        if (
            not self.source_registration_id
            or len(self.source_identity_sha256) != 64
            or not self.cell_id
            or self.cutoff_cycle not in _APPROVED_CUTOFFS
            or not self.data_version
            or not self.feature_version
            or not self.split_version
            or not isinstance(self.raw_sequence, EarlyCycleSequence)
        ):
            raise ValueError(
                "Advanced calibration cell input identity is invalid"
            )
        if self.initial_soh is not None and (
            not math.isfinite(self.initial_soh)
            or not 0.0 < self.initial_soh <= 1.5
        ):
            raise ValueError(
                "Advanced calibration initial SOH must be finite and physical"
            )


class AdvancedCalibrationCellInputResolver(Protocol):
    def resolve(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        task: AdvancedModelTask,
        cell_id: str,
        cutoff_cycle: int,
        feature_version: str,
    ) -> AdvancedCalibrationCellInput: ...


class VerifiedAdvancedCalibrationCellPredictor:
    """Run an exact verified runtime over server-resolved early-cycle input."""

    def __init__(
        self,
        *,
        input_resolver: AdvancedCalibrationCellInputResolver,
    ) -> None:
        self._input_resolver = input_resolver

    def predict_rul(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedRULRuntime,
    ) -> float:
        resolved = self._resolve_input(
            source_registration_id=source_registration_id,
            source_identity=source_identity,
            cell_id=cell_id,
            runtime=runtime,
        )
        value = runtime.inference.predict_cycle(resolved.raw_sequence)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) <= runtime.cutoff_cycle
        ):
            raise ValueError(
                "Advanced RUL runtime returned an invalid official cycle life"
            )
        return float(value)

    def predict_soh(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedSOHRuntime,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        resolved = self._resolve_input(
            source_registration_id=source_registration_id,
            source_identity=source_identity,
            cell_id=cell_id,
            runtime=runtime,
        )
        if resolved.initial_soh is None:
            raise ValueError(
                "Advanced SOH calibration input requires cutoff SOH"
            )
        cycles, values = runtime.inference.predict_trajectory(
            resolved.raw_sequence,
            initial_soh=resolved.initial_soh,
        )
        expected = tuple(range(runtime.cutoff_cycle + 1, 501))
        normalized_values = tuple(float(value) for value in values)
        if (
            tuple(cycles) != expected
            or len(normalized_values) != len(expected)
            or any(
                not math.isfinite(value) or value < 0.0 or value > 1.5
                for value in normalized_values
            )
            or any(
                current > previous
                for previous, current in pairwise(normalized_values)
            )
        ):
            raise ValueError(
                "Advanced SOH runtime returned an invalid finite trajectory"
            )
        return expected, normalized_values

    def _resolve_input(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedAdvancedRuntime,
    ) -> AdvancedCalibrationCellInput:
        resolved = self._input_resolver.resolve(
            source_registration_id=source_registration_id,
            source_identity=source_identity,
            task=runtime.task,
            cell_id=cell_id,
            cutoff_cycle=runtime.cutoff_cycle,
            feature_version=runtime.feature_version,
        )
        sequence = resolved.raw_sequence
        if (
            resolved.source_registration_id != source_registration_id
            or resolved.source_identity_sha256
            != source_identity.source_identity_sha256
            or resolved.cell_id != cell_id
            or resolved.cutoff_cycle != runtime.cutoff_cycle
            or resolved.data_version != runtime.data_version
            or resolved.feature_version != runtime.feature_version
            or resolved.split_version != runtime.split_version
            or sequence.dataset_id != runtime.dataset_id
            or sequence.cell_id != cell_id
            or sequence.cutoff_cycle != runtime.cutoff_cycle
            or sequence.data_version != runtime.data_version
            or sequence.feature_version != runtime.feature_version
            or sequence.normalization_version != "none"
            or sequence.normalization_statistics_sha256 is not None
        ):
            raise ValueError(
                "Advanced calibration cell input does not match the runtime"
            )
        return resolved


class AdvancedCalibrationRuntimeResolver(Protocol):
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


class AdvancedCalibrationCellPredictor(Protocol):
    """Server-owned label-free inference boundary for registered MATR cells."""

    def predict_rul(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedRULRuntime,
    ) -> float: ...

    def predict_soh(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedSOHRuntime,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]: ...


class AdvancedCalibrationSampleProducer:
    """Combine trusted supervision with one exact active Advanced runtime."""

    def __init__(
        self,
        *,
        evidence_resolver: AdvancedCalibrationEvidenceResolver,
        runtime_resolver: AdvancedCalibrationRuntimeResolver,
        predictor: AdvancedCalibrationCellPredictor,
    ) -> None:
        self._evidence_resolver = evidence_resolver
        self._runtime_resolver = runtime_resolver
        self._predictor = predictor

    def produce(
        self,
        *,
        context: VerifiedProjectInvocationContext,
        materialization_id: str,
        request: AdvancedCalibrationMaterializationRequest,
        created_at: datetime,
    ) -> tuple[ProducedAdvancedCalibrationSample, ...]:
        normalized_request = AdvancedCalibrationMaterializationRequest.model_validate(
            request.model_dump(mode="json")
        )
        namespace = _uuid(materialization_id, "materialization_id")
        timestamp = _utc_timestamp(created_at)
        _require_context(context, normalized_request)
        evidence = self._evidence_resolver.resolve(
            normalized_request.source_registration_id,
            task=normalized_request.task,
            cutoff_cycle=normalized_request.cutoff_cycle,
        )
        runtime = self._runtime_resolver.resolve(
            context,
            task=normalized_request.task,
            cutoff_cycle=normalized_request.cutoff_cycle,
            role=normalized_request.route_role,
        )
        runtime_identity = _runtime_identity(runtime)
        _require_request_identity(normalized_request, evidence, runtime)
        _require_unique_cells(evidence)

        samples = self._produce_samples(
            namespace=namespace,
            materialization_id=materialization_id,
            request=normalized_request,
            evidence=evidence,
            runtime=runtime,
            runtime_identity=runtime_identity,
            created_at=timestamp,
        )
        current_runtime = self._runtime_resolver.revalidate(context, runtime)
        if _runtime_identity(current_runtime) != runtime_identity:
            raise ValueError("Advanced runtime changed during calibration inference")
        return samples

    def _produce_samples(
        self,
        *,
        namespace: UUID,
        materialization_id: str,
        request: AdvancedCalibrationMaterializationRequest,
        evidence: AdvancedCalibrationEvidence,
        runtime: VerifiedAdvancedRuntime,
        runtime_identity: AdvancedConformalRuntimeIdentity,
        created_at: datetime,
    ) -> tuple[ProducedAdvancedCalibrationSample, ...]:
        if isinstance(evidence, AdvancedRULCalibrationEvidence):
            rul_runtime = cast(VerifiedRULRuntime, runtime)
            return tuple(
                self._rul_sample(
                    namespace=namespace,
                    materialization_id=materialization_id,
                    request=request,
                    evidence=evidence,
                    runtime=rul_runtime,
                    runtime_identity=runtime_identity,
                    ordinal=ordinal,
                    cell_id=cell.cell_id,
                    observed_cycle=cell.observed_cycle,
                    created_at=created_at,
                )
                for ordinal, cell in enumerate(evidence.cells)
            )
        soh_runtime = cast(VerifiedSOHRuntime, runtime)
        return tuple(
            self._soh_sample(
                namespace=namespace,
                materialization_id=materialization_id,
                request=request,
                evidence=evidence,
                runtime=soh_runtime,
                runtime_identity=runtime_identity,
                ordinal=ordinal,
                cell_id=cell.cell_id,
                observed_soh=cell.observed_soh,
                created_at=created_at,
            )
            for ordinal, cell in enumerate(evidence.cells)
        )

    def _rul_sample(
        self,
        *,
        namespace: UUID,
        materialization_id: str,
        request: AdvancedCalibrationMaterializationRequest,
        evidence: AdvancedRULCalibrationEvidence,
        runtime: VerifiedRULRuntime,
        runtime_identity: AdvancedConformalRuntimeIdentity,
        ordinal: int,
        cell_id: str,
        observed_cycle: int,
        created_at: datetime,
    ) -> ProducedAdvancedCalibrationSample:
        point_prediction = self._predictor.predict_rul(
            source_registration_id=request.source_registration_id,
            source_identity=evidence.source_identity,
            cell_id=cell_id,
            runtime=runtime,
        )
        sample = AdvancedRULCalibrationSample.model_validate(
            {
                "cell_id": cell_id,
                "point_prediction_cycle": point_prediction,
                "observed_cycle": observed_cycle,
                "runtime": runtime_identity.model_dump(mode="json"),
            }
        )
        artifact = {
            **runtime_identity.model_dump(mode="json"),
            "materialization_id": materialization_id,
            "source_registration_id": request.source_registration_id,
            "source_identity_sha256": (
                evidence.source_identity.source_identity_sha256
            ),
            "split_partition": "calibration",
            "cell_id": sample.cell_id,
            "point_prediction_cycle": sample.point_prediction_cycle,
            "observed_cycle": sample.observed_cycle,
        }
        return self._tool_result(
            namespace=namespace,
            request=request,
            ordinal=ordinal,
            cell_id=cell_id,
            runtime=runtime_identity,
            source_identity=evidence.source_identity,
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            artifact_type=ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            artifact=artifact,
            created_at=created_at,
        )

    def _soh_sample(
        self,
        *,
        namespace: UUID,
        materialization_id: str,
        request: AdvancedCalibrationMaterializationRequest,
        evidence: AdvancedSOHCalibrationEvidence,
        runtime: VerifiedSOHRuntime,
        runtime_identity: AdvancedConformalRuntimeIdentity,
        ordinal: int,
        cell_id: str,
        observed_soh: tuple[float, ...],
        created_at: datetime,
    ) -> ProducedAdvancedCalibrationSample:
        prediction_cycles, predicted_soh = self._predictor.predict_soh(
            source_registration_id=request.source_registration_id,
            source_identity=evidence.source_identity,
            cell_id=cell_id,
            runtime=runtime,
        )
        if prediction_cycles != evidence.prediction_cycles:
            raise ValueError("SOH runtime prediction axis differs from source evidence")
        sample = AdvancedSOHCalibrationSample.model_validate(
            {
                "cell_id": cell_id,
                "prediction_cycles": prediction_cycles,
                "predicted_soh": predicted_soh,
                "observed_soh": observed_soh,
                "runtime": runtime_identity.model_dump(mode="json"),
            }
        )
        artifact = {
            **runtime_identity.model_dump(mode="json"),
            "materialization_id": materialization_id,
            "source_registration_id": request.source_registration_id,
            "source_identity_sha256": (
                evidence.source_identity.source_identity_sha256
            ),
            "split_partition": "calibration",
            "cell_id": sample.cell_id,
            "prediction_cycles": list(sample.prediction_cycles),
            "predicted_soh": list(sample.predicted_soh),
            "observed_soh": list(sample.observed_soh),
            "finite_horizon_only": True,
            "horizon_end_cycle": 500,
        }
        return self._tool_result(
            namespace=namespace,
            request=request,
            ordinal=ordinal,
            cell_id=cell_id,
            runtime=runtime_identity,
            source_identity=evidence.source_identity,
            tool_name=StandardToolName.PREDICT_SOH_TRAJECTORY,
            artifact_type=ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
            artifact=artifact,
            created_at=created_at,
        )

    @staticmethod
    def _tool_result(
        *,
        namespace: UUID,
        request: AdvancedCalibrationMaterializationRequest,
        ordinal: int,
        cell_id: str,
        runtime: AdvancedConformalRuntimeIdentity,
        source_identity: AdvancedCalibrationSourceIdentity,
        tool_name: StandardToolName,
        artifact_type: str,
        artifact: dict[str, object],
        created_at: datetime,
    ) -> ProducedAdvancedCalibrationSample:
        semantic_payload = {
            "producer_version": ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION,
            "request": request.model_dump(mode="json"),
            "ordinal": ordinal,
            "cell_id": cell_id,
            "artifact_type": artifact_type,
            "artifact": artifact,
        }
        result_id = str(uuid5(namespace, sha256_canonical(semantic_payload)))
        result = ToolResult(
            result_id=result_id,
            tool_name=tool_name.value,
            tool_version=ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION,
            model_version=runtime.model_version,
            data_version=runtime.data_version,
            feature_version=runtime.feature_version,
            input_hash=sha256_canonical(semantic_payload),
            values={"artifact_type": artifact_type, "artifact": artifact},
            uncertainty=None,
            warnings=[],
            provenance=[
                ProvenanceRecord(
                    source_id=(
                        f"advanced-calibration-source-"
                        f"{source_identity.registration_id}"
                    ),
                    source_kind=SourceKind.OBSERVED,
                    uri=(
                        "calibration-source://"
                        f"{source_identity.registration_id}/"
                        f"{source_identity.source_identity_sha256}"
                    ),
                    sha256=source_identity.source_identity_sha256,
                    description=(
                        "Verified server-owned MATR calibration supervision"
                    ),
                    created_at=created_at,
                ),
                ProvenanceRecord(
                    source_id=f"advanced-model-{runtime.artifact_id}",
                    source_kind=SourceKind.PREDICTED,
                    uri=f"artifact://advanced-model/{runtime.artifact_id}",
                    sha256=runtime.artifact_manifest_sha256,
                    description=(
                        "Verified active Advanced runtime used for calibration inference"
                    ),
                    created_at=created_at,
                ),
            ],
            created_at=created_at,
        )
        return ProducedAdvancedCalibrationSample(
            ordinal=ordinal,
            cell_id=cell_id,
            result=result,
        )


def _require_context(
    context: VerifiedProjectInvocationContext,
    request: AdvancedCalibrationMaterializationRequest,
) -> None:
    if context.project_id != request.project_id:
        raise ValueError("calibration request project does not match its context")
    if context.actor_role is not UserRole.ADMIN:
        raise ValueError("Advanced calibration materialization requires ADMIN")


def _require_request_identity(
    request: AdvancedCalibrationMaterializationRequest,
    evidence: AdvancedCalibrationEvidence,
    runtime: VerifiedAdvancedRuntime,
) -> None:
    source = evidence.source_identity
    if (
        runtime.project_id != request.project_id
        or runtime.task is not request.task
        or runtime.cutoff_cycle != request.cutoff_cycle
        or runtime.role is not request.route_role
    ):
        raise ValueError("Advanced runtime does not match the materialization request")
    if (
        source.registration_id != request.source_registration_id
        or source.dataset_id != runtime.dataset_id
        or source.data_version != runtime.data_version
        or source.split_version != runtime.split_version
    ):
        raise ValueError("Advanced calibration source identity does not match the runtime")
    if request.task is AdvancedModelTask.RUL and not isinstance(
        evidence,
        AdvancedRULCalibrationEvidence,
    ):
        raise ValueError("RUL materialization requires RUL calibration evidence")
    if isinstance(evidence, AdvancedRULCalibrationEvidence) and any(
        cell.observed_cycle <= request.cutoff_cycle for cell in evidence.cells
    ):
        raise ValueError("MATR official cycle life must be greater than cutoff")
    if request.task is AdvancedModelTask.SOH and not isinstance(
        evidence,
        AdvancedSOHCalibrationEvidence,
    ):
        raise ValueError("SOH materialization requires SOH calibration evidence")
    if isinstance(evidence, AdvancedSOHCalibrationEvidence) and (
        evidence.prediction_cycles
        != tuple(range(request.cutoff_cycle + 1, 501))
    ):
        raise ValueError(
            "SOH calibration axis must span cutoff + 1 through cycle 500"
        )


def _require_unique_cells(evidence: AdvancedCalibrationEvidence) -> None:
    cell_ids = tuple(cell.cell_id for cell in evidence.cells)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Advanced calibration cell identities must be unique")


def _runtime_identity(
    runtime: VerifiedAdvancedRuntime,
) -> AdvancedConformalRuntimeIdentity:
    normalization_sha256 = runtime.inference.normalization_statistics_sha256
    if normalization_sha256 != runtime.normalization_sha256:
        raise ValueError("Advanced runtime normalization identity is inconsistent")
    return AdvancedConformalRuntimeIdentity.model_validate(
        {
            "task": runtime.task,
            "route_role": runtime.role,
            "output_target": runtime.output_target.value,
            "artifact_kind": _conformal_artifact_kind(runtime.artifact_kind),
            "artifact_id": runtime.artifact_id,
            "artifact_manifest_sha256": runtime.artifact_manifest_sha256,
            "model_version": runtime.model_version,
            "dataset_id": runtime.dataset_id,
            "cutoff_cycle": runtime.cutoff_cycle,
            "data_version": runtime.data_version,
            "feature_version": runtime.feature_version,
            "split_version": runtime.split_version,
            "normalization_statistics_sha256": normalization_sha256,
            "decision_event_id": runtime.decision_event_id,
            "ledger_sequence_number": runtime.ledger_sequence_number,
            "ledger_head_sha256": runtime.ledger_head_sha256,
        }
    )


def _conformal_artifact_kind(artifact_kind: DeepArtifactKind) -> str:
    mapping = {
        DeepArtifactKind.CYCLEPATCH_DIRECT: "cyclepatch_direct",
        DeepArtifactKind.CYCLEPATCH_BATLINET: "cyclepatch_batlinet",
        DeepArtifactKind.HYBRIDPATCH_V2: "hybridpatch_v2",
        DeepArtifactKind.CURRENT_HYBRID: "current_hybrid",
    }
    try:
        return mapping[artifact_kind]
    except KeyError as exc:
        raise ValueError(
            "Advanced runtime artifact kind is not conformal-compatible"
        ) from exc


def _utc_timestamp(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError("created_at must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _uuid(value: str, label: str) -> UUID:
    try:
        return UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID string") from exc


__all__ = [
    "ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION",
    "AdvancedCalibrationCellInput",
    "AdvancedCalibrationCellInputResolver",
    "AdvancedCalibrationCellPredictor",
    "AdvancedCalibrationMaterializationRequest",
    "AdvancedCalibrationRuntimeResolver",
    "AdvancedCalibrationSampleProducer",
    "ProducedAdvancedCalibrationSample",
    "VerifiedAdvancedCalibrationCellPredictor",
]
