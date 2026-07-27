"""Verified application services for formal Advanced RUL and SOH inference."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Literal, Protocol

from quanxin_life.application.advanced_runtime import (
    ActiveAdvancedRuntimeResolver,
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.deep_model_artifacts import DeepArtifactKind
from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    CycleLifePrediction,
    PredictionTarget,
    sha256_canonical,
)
from quanxin_life.features.early_cycle_sequence import (
    PHASE_NAMES,
    VARIABLE_NAMES,
    EarlyCycleSequence,
)
from quanxin_life.features.multichannel_cycle import (
    CONDITION_NAMES,
    MultichannelCycleConfig,
    build_early_cycle_sequence,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    AdvancedRULInference,
)
from quanxin_life.tools.advanced_input import (
    AdvancedInputEvidence,
    ProjectEarlyCycleBatchResolver,
)
from quanxin_life.tools.advanced_soh_prediction import AdvancedSOHInference
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch


class AdvancedRuntimeResolver(Protocol):
    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedRULRuntime | VerifiedSOHRuntime: ...

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
        runtime: VerifiedRULRuntime | VerifiedSOHRuntime,
    ) -> VerifiedRULRuntime | VerifiedSOHRuntime: ...


class AdvancedRULPredictionService:
    def __init__(
        self,
        *,
        batch_resolver: ProjectEarlyCycleBatchResolver,
        runtime_resolver: AdvancedRuntimeResolver | ActiveAdvancedRuntimeResolver,
    ) -> None:
        self._batch_resolver = batch_resolver
        self._runtime_resolver = runtime_resolver

    def predict(
        self,
        context: VerifiedProjectInvocationContext,
        evidence: AdvancedInputEvidence,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedRULInference:
        _validate_rul_role(evidence.cutoff_cycle, route_role)
        raw_sequence, _batch = _rebuild_verified_sequence(
            context,
            evidence,
            self._batch_resolver,
        )
        runtime = self._runtime_resolver.resolve(
            context,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=evidence.cutoff_cycle,
            role=route_role,
        )
        if not isinstance(runtime, VerifiedRULRuntime):
            raise ValueError("Advanced RUL resolver returned the wrong runtime task")
        _validate_runtime(runtime, context, evidence, role=route_role)
        predicted_cycle = runtime.inference.predict_cycle(raw_sequence)
        self._runtime_resolver.revalidate(context, runtime)
        prediction = CycleLifePrediction(
            dataset_id=evidence.dataset_id,
            cell_id=evidence.cell_id,
            cutoff_cycle=evidence.cutoff_cycle,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            predicted_cycle=predicted_cycle,
            observed_cycle=None,
            right_censored=True,
            feature_version=evidence.feature_version,
            split_version=evidence.split_version,
            model_version=runtime.model_version,
            data_version=evidence.data_version,
        )
        return AdvancedRULInference(
            prediction=prediction,
            task=AdvancedModelTask.RUL,
            route_role=route_role,
            output_target="matr_official_cycle_life",
            artifact_kind=_rul_artifact_kind(runtime.artifact_kind),
            artifact_id=runtime.artifact_id,
            artifact_manifest_sha256=runtime.artifact_manifest_sha256,
            raw_sequence_input_sha256=raw_sequence.input_hash,
            normalization_statistics_sha256=(
                runtime.inference.normalization_statistics_sha256
            ),
            decision_event_id=runtime.decision_event_id,
            ledger_sequence_number=runtime.ledger_sequence_number,
            ledger_head_sha256=runtime.ledger_head_sha256,
        )


class AdvancedSOHPredictionService:
    def __init__(
        self,
        *,
        batch_resolver: ProjectEarlyCycleBatchResolver,
        runtime_resolver: AdvancedRuntimeResolver | ActiveAdvancedRuntimeResolver,
    ) -> None:
        self._batch_resolver = batch_resolver
        self._runtime_resolver = runtime_resolver

    def predict(
        self,
        context: VerifiedProjectInvocationContext,
        evidence: AdvancedInputEvidence,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedSOHInference:
        if route_role not in {
            AdvancedModelRouteRole.MEAN_ACCURACY,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
        }:
            raise ValueError("Advanced SOH requires an approved SOH route role")
        raw_sequence, batch = _rebuild_verified_sequence(
            context,
            evidence,
            self._batch_resolver,
        )
        runtime = self._runtime_resolver.resolve(
            context,
            task=AdvancedModelTask.SOH,
            cutoff_cycle=evidence.cutoff_cycle,
            role=route_role,
        )
        if not isinstance(runtime, VerifiedSOHRuntime):
            raise ValueError("Advanced SOH resolver returned the wrong runtime task")
        _validate_runtime(runtime, context, evidence, role=route_role)
        prediction_cycles, predicted_soh = runtime.inference.predict_trajectory(
            raw_sequence,
            initial_soh=_cutoff_soh(batch),
        )
        self._runtime_resolver.revalidate(context, runtime)
        return AdvancedSOHInference(
            dataset_id=evidence.dataset_id,
            cell_id=evidence.cell_id,
            cutoff_cycle=evidence.cutoff_cycle,
            data_version=evidence.data_version,
            feature_version=evidence.feature_version,
            split_version=evidence.split_version,
            model_version=runtime.model_version,
            task=AdvancedModelTask.SOH,
            route_role=route_role,
            output_target="soh_trajectory",
            artifact_kind=_soh_artifact_kind(runtime.artifact_kind),
            artifact_id=runtime.artifact_id,
            artifact_manifest_sha256=runtime.artifact_manifest_sha256,
            raw_sequence_input_sha256=raw_sequence.input_hash,
            normalization_statistics_sha256=(
                runtime.inference.normalization_statistics_sha256
            ),
            prediction_cycles=prediction_cycles,
            predicted_soh=predicted_soh,
            decision_event_id=runtime.decision_event_id,
            ledger_sequence_number=runtime.ledger_sequence_number,
            ledger_head_sha256=runtime.ledger_head_sha256,
        )


def _rebuild_verified_sequence(
    context: VerifiedProjectInvocationContext,
    evidence: AdvancedInputEvidence,
    resolver: ProjectEarlyCycleBatchResolver,
) -> tuple[EarlyCycleSequence, VerifiedEarlyCycleBatch]:
    resolved = resolver.resolve_verified_early_cycle_batch(
        context,
        evidence.record_batch_id,
    )
    batch = VerifiedEarlyCycleBatch.model_validate(
        resolved.model_dump(mode="json")
    )
    expected = {
        "record_batch_id": evidence.record_batch_id,
        "data_version": evidence.data_version,
        "split_version": evidence.split_version,
        "source_manifest_hash": evidence.source_manifest_hash,
    }
    for name, value in expected.items():
        if getattr(batch, name) != value:
            raise ValueError(f"Advanced input batch {name} changed before inference")
    if (
        batch.metadata.dataset_id != evidence.dataset_id
        or batch.metadata.cell_id != evidence.cell_id
        or batch.feature_config.cutoff_cycle != evidence.cutoff_cycle
        or batch.feature_config.feature_version != evidence.feature_version
    ):
        raise ValueError("Advanced input batch identity changed before inference")
    config = MultichannelCycleConfig(
        cutoff_cycle=evidence.cutoff_cycle,
        feature_version=evidence.feature_version,
    )
    if sha256_canonical(asdict(config)) != evidence.transform_config_sha256:
        raise ValueError("Advanced input transform configuration changed before inference")
    raw_sequence = build_early_cycle_sequence(
        batch.records,
        config=config,
        data_version=evidence.data_version,
    )
    if raw_sequence.input_hash != evidence.raw_sequence_input_sha256:
        raise ValueError("Advanced raw sequence changed before inference")
    if (
        evidence.phase_names != PHASE_NAMES
        or evidence.variable_names != VARIABLE_NAMES
        or evidence.condition_names != CONDITION_NAMES
    ):
        raise ValueError("Advanced input axes changed before inference")
    return raw_sequence, batch


def _validate_runtime(
    runtime: VerifiedRULRuntime | VerifiedSOHRuntime,
    context: VerifiedProjectInvocationContext,
    evidence: AdvancedInputEvidence,
    *,
    role: AdvancedModelRouteRole,
) -> None:
    expected = {
        "project_id": context.project_id,
        "cutoff_cycle": evidence.cutoff_cycle,
        "dataset_id": evidence.dataset_id,
        "data_version": evidence.data_version,
        "feature_version": evidence.feature_version,
        "split_version": evidence.split_version,
        "role": role,
    }
    for name, value in expected.items():
        if getattr(runtime, name) != value:
            raise ValueError(f"Advanced runtime {name} does not match input evidence")
    if (
        runtime.normalization_sha256
        != runtime.inference.normalization_statistics_sha256
    ):
        raise ValueError("Advanced runtime normalization context is inconsistent")
    expected_target = (
        "matr_official_cycle_life"
        if runtime.task is AdvancedModelTask.RUL
        else "soh_trajectory"
    )
    if runtime.output_target.value != expected_target:
        raise ValueError("Advanced runtime output target is inconsistent")


def _validate_rul_role(
    cutoff_cycle: int,
    route_role: AdvancedModelRouteRole,
) -> None:
    if cutoff_cycle == 20:
        if route_role is not AdvancedModelRouteRole.DEFAULT:
            raise ValueError("cutoff 20 requires the frozen DEFAULT RUL route")
        return
    if route_role not in {
        AdvancedModelRouteRole.POINT_ACCURACY,
        AdvancedModelRouteRole.COVERAGE,
    }:
        raise ValueError("Advanced RUL requires a point-accuracy or coverage route")


def _rul_artifact_kind(
    artifact_kind: DeepArtifactKind,
) -> Literal["cyclepatch_direct", "cyclepatch_batlinet"]:
    if artifact_kind is DeepArtifactKind.CYCLEPATCH_DIRECT:
        return "cyclepatch_direct"
    if artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
        return "cyclepatch_batlinet"
    raise ValueError("Advanced RUL runtime artifact kind is inconsistent")


def _soh_artifact_kind(
    artifact_kind: DeepArtifactKind,
) -> Literal["hybridpatch_v2", "current_hybrid"]:
    if artifact_kind is DeepArtifactKind.HYBRIDPATCH_V2:
        return "hybridpatch_v2"
    if artifact_kind is DeepArtifactKind.CURRENT_HYBRID:
        return "current_hybrid"
    raise ValueError("Advanced SOH runtime artifact kind is inconsistent")


def _cutoff_soh(batch: VerifiedEarlyCycleBatch) -> float:
    reference_capacity = batch.metadata.reference_capacity_ah
    if reference_capacity is None or not math.isfinite(reference_capacity):
        raise ValueError("verified reference capacity is required for SOH inference")
    cutoff = batch.feature_config.cutoff_cycle
    capacities = tuple(
        float(record.discharge_capacity_ah)
        for record in batch.records
        if (
            record.valid
            and record.diagnostic
            and record.cycle_index == cutoff
            and record.discharge_capacity_ah is not None
            and math.isfinite(float(record.discharge_capacity_ah))
            and float(record.discharge_capacity_ah) > 0
        )
    )
    if not capacities:
        raise ValueError("cutoff diagnostic discharge capacity is required")
    value = max(capacities) / float(reference_capacity)
    if not math.isfinite(value) or not 0.0 < value <= 1.5:
        raise ValueError("cutoff SOH must be finite and in (0, 1.5]")
    return value


__all__ = [
    "AdvancedRULPredictionService",
    "AdvancedSOHPredictionService",
]
