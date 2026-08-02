from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch
from pydantic import ValidationError

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceIdentity,
    AdvancedRULCalibrationEvidence,
    AdvancedRULObservedCell,
    AdvancedSOHCalibrationEvidence,
    AdvancedSOHObservedCell,
)
from quanxin_life.application.advanced_calibration_materialization import (
    ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION,
    AdvancedCalibrationCellInput,
    AdvancedCalibrationMaterializationRequest,
    AdvancedCalibrationSampleProducer,
    VerifiedAdvancedCalibrationCellPredictor,
)
from quanxin_life.application.advanced_runtime import (
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactKind,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    SourceKind,
    UserRole,
)
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.tools.advanced_conformal import (
    ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
    ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName

NOW = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)
PROJECT_ID = "project-advanced-calibration"
MATERIALIZATION_ID = "b2ae9c9c-7f4e-4d79-8ea1-40a99314f457"
ARTIFACT_ID = "f4e12f3a-6502-4aec-876b-05e4ec00cf1b"
DECISION_EVENT_ID = "a94ebc02-6105-4822-beb7-09b174db6ac5"


def _source_identity() -> AdvancedCalibrationSourceIdentity:
    return AdvancedCalibrationSourceIdentity(
        registration_id="matr-three-batch-final-v1",
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-split-v1",
        three_batch_manifest_sha256="1" * 64,
        combined_split_sha256="2" * 64,
        conversion_report_sha256s=("3" * 64, "4" * 64, "5" * 64),
        component_split_sha256s=("6" * 64, "7" * 64, "8" * 64),
        eligibility_report_sha256s=("9" * 64, "a" * 64, "b" * 64),
        supervision_report_sha256s=("c" * 64, "d" * 64, "e" * 64),
        supervision_parquet_sha256s=("f" * 64, "0" * 64, "1" * 64),
        source_identity_sha256="2" * 64,
    )


def _context() -> VerifiedProjectInvocationContext:
    return VerifiedProjectInvocationContext(
        project_id=PROJECT_ID,
        actor_user_id="admin-user",
        actor_session_id="admin-session",
        actor_role=UserRole.ADMIN,
        invocation_source=ProjectInvocationSource.HTTP,
        _authorization_tag="0" * 64,
    )


def _rul_runtime(*, head: str = "7" * 64) -> VerifiedRULRuntime:
    return VerifiedRULRuntime(
        project_id=PROJECT_ID,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        role=AdvancedModelRouteRole.COVERAGE,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        split_version="matr-three-batch-split-v1",
        normalization_sha256="3" * 64,
        artifact_id=ARTIFACT_ID,
        artifact_manifest_sha256="4" * 64,
        model_version="cyclepatch-direct-v1",
        decision_event_id=DECISION_EVENT_ID,
        ledger_sequence_number=3,
        ledger_head_sha256=head,
        inference=SimpleNamespace(
            normalization_statistics_sha256="3" * 64
        ),
    )


def _soh_runtime(*, head: str = "7" * 64) -> VerifiedSOHRuntime:
    return VerifiedSOHRuntime(
        project_id=PROJECT_ID,
        task=AdvancedModelTask.SOH,
        cutoff_cycle=100,
        role=AdvancedModelRouteRole.MEAN_ACCURACY,
        output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
        artifact_kind=DeepArtifactKind.HYBRIDPATCH_V2,
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        split_version="matr-three-batch-split-v1",
        normalization_sha256="3" * 64,
        artifact_id=ARTIFACT_ID,
        artifact_manifest_sha256="4" * 64,
        model_version="hybridpatch-v2",
        decision_event_id=DECISION_EVENT_ID,
        ledger_sequence_number=3,
        ledger_head_sha256=head,
        inference=SimpleNamespace(
            normalization_statistics_sha256="3" * 64
        ),
    )


class _EvidenceResolver:
    def __init__(
        self,
        evidence: AdvancedRULCalibrationEvidence | AdvancedSOHCalibrationEvidence,
    ) -> None:
        self.evidence = evidence
        self.calls: list[tuple[str, AdvancedModelTask, int]] = []

    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedRULCalibrationEvidence | AdvancedSOHCalibrationEvidence:
        self.calls.append((registration_id, task, cutoff_cycle))
        return self.evidence


class _RuntimeResolver:
    def __init__(
        self,
        runtime: VerifiedRULRuntime | VerifiedSOHRuntime,
        *,
        replacement: VerifiedRULRuntime | VerifiedSOHRuntime | None = None,
    ) -> None:
        self.runtime = runtime
        self.replacement = replacement
        self.resolve_calls = 0
        self.revalidate_calls = 0

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedRULRuntime | VerifiedSOHRuntime:
        del context, task, cutoff_cycle, role
        self.resolve_calls += 1
        return self.runtime

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
        runtime: VerifiedRULRuntime | VerifiedSOHRuntime,
    ) -> VerifiedRULRuntime | VerifiedSOHRuntime:
        del context, runtime
        self.revalidate_calls += 1
        return self.replacement or self.runtime


class _Predictor:
    def __init__(self) -> None:
        self.rul_calls: list[str] = []
        self.soh_calls: list[str] = []

    def predict_rul(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedRULRuntime,
    ) -> float:
        del source_registration_id, source_identity, runtime
        self.rul_calls.append(cell_id)
        return {"cell-a": 620.0, "cell-b": 710.0}[cell_id]

    def predict_soh(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        cell_id: str,
        runtime: VerifiedSOHRuntime,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        del source_registration_id, source_identity, runtime
        self.soh_calls.append(cell_id)
        cycles = tuple(range(101, 501))
        offset = 0.001 if cell_id == "cell-b" else 0.0
        return cycles, tuple(1.0 - index / 10000 - offset for index in cycles)


def test_request_is_identity_only_and_rejects_client_business_values() -> None:
    request = AdvancedCalibrationMaterializationRequest(
        project_id=PROJECT_ID,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        route_role=AdvancedModelRouteRole.COVERAGE,
        source_registration_id="matr-three-batch-final-v1",
    )

    assert set(request.model_dump()) == {
        "project_id",
        "task",
        "cutoff_cycle",
        "route_role",
        "source_registration_id",
    }
    for forbidden in (
        {"cell_ids": ["cell-a"]},
        {"observed_cycle": 600},
        {"predicted_soh": [0.9]},
        {"evidence_root": "D:/evidence"},
        {"model_version": "client-model"},
        {"source_identity_sha256": "f" * 64},
    ):
        with pytest.raises(ValidationError):
            AdvancedCalibrationMaterializationRequest.model_validate(
                request.model_dump(mode="json") | forbidden
            )


@pytest.mark.parametrize(
    ("task", "cutoff_cycle", "route_role"),
    [
        (
            AdvancedModelTask.RUL,
            100,
            AdvancedModelRouteRole.POINT_ACCURACY,
        ),
        (
            AdvancedModelTask.RUL,
            20,
            AdvancedModelRouteRole.COVERAGE,
        ),
        (
            AdvancedModelTask.SOH,
            100,
            AdvancedModelRouteRole.COVERAGE,
        ),
    ],
)
def test_request_rejects_routes_that_cannot_materialize_conformal_samples(
    task: AdvancedModelTask,
    cutoff_cycle: int,
    route_role: AdvancedModelRouteRole,
) -> None:
    with pytest.raises(ValidationError, match="route"):
        AdvancedCalibrationMaterializationRequest(
            project_id=PROJECT_ID,
            task=task,
            cutoff_cycle=cutoff_cycle,
            route_role=route_role,
            source_registration_id="matr-three-batch-final-v1",
        )


def test_produces_deterministic_rul_sample_tool_results() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity(),
        cells=(
            AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=611),
            AdvancedRULObservedCell(cell_id="cell-b", observed_cycle=702),
        ),
    )
    evidence_resolver = _EvidenceResolver(evidence)
    runtime_resolver = _RuntimeResolver(_rul_runtime())
    predictor = _Predictor()
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=evidence_resolver,
        runtime_resolver=runtime_resolver,
        predictor=predictor,
    )
    request = AdvancedCalibrationMaterializationRequest(
        project_id=PROJECT_ID,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        route_role=AdvancedModelRouteRole.COVERAGE,
        source_registration_id="matr-three-batch-final-v1",
    )

    first = producer.produce(
        context=_context(),
        materialization_id=MATERIALIZATION_ID,
        request=request,
        created_at=NOW,
    )
    second = producer.produce(
        context=_context(),
        materialization_id=MATERIALIZATION_ID,
        request=request,
        created_at=NOW,
    )

    assert first == second
    assert tuple(item.ordinal for item in first) == (0, 1)
    assert tuple(item.cell_id for item in first) == ("cell-a", "cell-b")
    assert all(UUID(item.result.result_id).version == 5 for item in first)
    assert all(
        item.result.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value
        and item.result.tool_version == ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION
        and item.result.created_at == NOW
        and item.result.values["artifact_type"]
        == ADVANCED_RUL_CALIBRATION_SAMPLE_EVIDENCE_TYPE
        for item in first
    )
    artifact = first[0].result.values["artifact"]
    assert artifact["split_partition"] == "calibration"
    assert artifact["observed_cycle"] == 611
    assert artifact["point_prediction_cycle"] == 620.0
    assert artifact["materialization_id"] == MATERIALIZATION_ID
    assert artifact["source_identity_sha256"] == "2" * 64
    assert {item.source_kind for item in first[0].result.provenance} == {
        SourceKind.OBSERVED,
        SourceKind.PREDICTED,
    }
    assert runtime_resolver.resolve_calls == 2
    assert runtime_resolver.revalidate_calls == 2


def test_produces_exact_finite_soh_sample_tool_results() -> None:
    cycles = tuple(range(101, 501))
    evidence = AdvancedSOHCalibrationEvidence(
        source_identity=_source_identity(),
        prediction_cycles=cycles,
        cells=(
            AdvancedSOHObservedCell(
                cell_id="cell-a",
                observed_soh=tuple(1.0 - index / 9000 for index in cycles),
            ),
            AdvancedSOHObservedCell(
                cell_id="cell-b",
                observed_soh=tuple(0.999 - index / 9000 for index in cycles),
            ),
        ),
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_soh_runtime()),
        predictor=_Predictor(),
    )

    samples = producer.produce(
        context=_context(),
        materialization_id=MATERIALIZATION_ID,
        request=AdvancedCalibrationMaterializationRequest(
            project_id=PROJECT_ID,
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
            route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
            source_registration_id="matr-three-batch-final-v1",
        ),
        created_at=NOW,
    )

    assert len(samples) == 2
    assert all(
        item.result.tool_name == StandardToolName.PREDICT_SOH_TRAJECTORY.value
        and item.result.values["artifact_type"]
        == ADVANCED_SOH_CALIBRATION_SAMPLE_EVIDENCE_TYPE
        for item in samples
    )
    artifact = samples[0].result.values["artifact"]
    assert artifact["prediction_cycles"] == list(cycles)
    assert len(artifact["predicted_soh"]) == 400
    assert len(artifact["observed_soh"]) == 400
    assert artifact["finite_horizon_only"] is True
    assert artifact["horizon_end_cycle"] == 500


def test_aligns_dense_soh_observations_to_verified_sparse_runtime_axis() -> None:
    evidence_cycles = tuple(range(101, 501))
    runtime_cycles = tuple(
        cycle for cycle in evidence_cycles if cycle != 139
    )
    observed_by_cycle = {
        cycle: 1.0 - cycle / 9000 for cycle in evidence_cycles
    }
    evidence = AdvancedSOHCalibrationEvidence(
        source_identity=_source_identity(),
        prediction_cycles=evidence_cycles,
        cells=(
            AdvancedSOHObservedCell(
                cell_id="cell-a",
                observed_soh=tuple(
                    observed_by_cycle[cycle] for cycle in evidence_cycles
                ),
            ),
        ),
    )

    class _SparseAxisPredictor(_Predictor):
        def predict_soh(
            self,
            *,
            source_registration_id: str,
            source_identity: AdvancedCalibrationSourceIdentity,
            cell_id: str,
            runtime: VerifiedSOHRuntime,
        ) -> tuple[tuple[int, ...], tuple[float, ...]]:
            del source_registration_id, source_identity, runtime
            self.soh_calls.append(cell_id)
            return runtime_cycles, tuple(
                1.0 - cycle / 10000 for cycle in runtime_cycles
            )

    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_soh_runtime()),
        predictor=_SparseAxisPredictor(),
    )

    samples = producer.produce(
        context=_context(),
        materialization_id=MATERIALIZATION_ID,
        request=AdvancedCalibrationMaterializationRequest(
            project_id=PROJECT_ID,
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
            route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
            source_registration_id="matr-three-batch-final-v1",
        ),
        created_at=NOW,
    )

    artifact = samples[0].result.values["artifact"]
    assert artifact["prediction_cycles"] == list(runtime_cycles)
    assert artifact["observed_soh"] == [
        observed_by_cycle[cycle] for cycle in runtime_cycles
    ]


def test_rejects_runtime_change_after_inference() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity(),
        cells=(AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=611),),
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(
            _rul_runtime(),
            replacement=_rul_runtime(head="8" * 64),
        ),
        predictor=_Predictor(),
    )

    with pytest.raises(ValueError, match="changed during calibration inference"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=AdvancedCalibrationMaterializationRequest(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE,
                source_registration_id="matr-three-batch-final-v1",
            ),
            created_at=NOW,
        )


def test_rejects_wrong_project_or_source_runtime_identity() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity().model_copy(
            update={"split_version": "other-split"}
        ),
        cells=(AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=611),),
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_rul_runtime()),
        predictor=_Predictor(),
    )
    request = AdvancedCalibrationMaterializationRequest(
        project_id=PROJECT_ID,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        route_role=AdvancedModelRouteRole.COVERAGE,
        source_registration_id="matr-three-batch-final-v1",
    )

    with pytest.raises(ValueError, match="source identity"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=request,
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="project"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=request.model_copy(update={"project_id": "other-project"}),
            created_at=NOW,
        )


def test_rejects_duplicate_calibration_cells_before_inference() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity(),
        cells=(
            AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=611),
            AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=611),
        ),
    )
    predictor = _Predictor()
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_rul_runtime()),
        predictor=predictor,
    )

    with pytest.raises(ValueError, match="unique"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=AdvancedCalibrationMaterializationRequest(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE,
                source_registration_id="matr-three-batch-final-v1",
            ),
            created_at=NOW,
        )

    assert predictor.rul_calls == []


def test_rejects_official_cycle_life_at_the_cutoff() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity(),
        cells=(
            AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=100),
        ),
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_rul_runtime()),
        predictor=_Predictor(),
    )

    with pytest.raises(ValueError, match="greater than cutoff"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=AdvancedCalibrationMaterializationRequest(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE,
                source_registration_id="matr-three-batch-final-v1",
            ),
            created_at=NOW,
        )


class _CalibrationInputResolver:
    def __init__(self, value: AdvancedCalibrationCellInput) -> None:
        self.value = value
        self.calls: list[
            tuple[str, str, str, str, int, str]
        ] = []

    def resolve(
        self,
        *,
        source_registration_id: str,
        source_identity: AdvancedCalibrationSourceIdentity,
        task: AdvancedModelTask,
        cell_id: str,
        cutoff_cycle: int,
        feature_version: str,
    ) -> AdvancedCalibrationCellInput:
        self.calls.append(
            (
                source_registration_id,
                source_identity.source_identity_sha256,
                task.value,
                cell_id,
                cutoff_cycle,
                feature_version,
            )
        )
        return self.value


class _RuntimeRULInference:
    normalization_statistics_sha256 = "3" * 64

    def __init__(self) -> None:
        self.calls: list[str] = []

    def predict_cycle(self, raw_sequence: EarlyCycleSequence) -> float:
        self.calls.append(raw_sequence.cell_id)
        return 650.0


class _RuntimeSOHInference:
    normalization_statistics_sha256 = "3" * 64

    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def predict_trajectory(
        self,
        raw_sequence: EarlyCycleSequence,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        self.calls.append((raw_sequence.cell_id, initial_soh))
        cycles = tuple(range(101, 501))
        return cycles, tuple(1.0 - index / 10000 for index in cycles)


class _SparseRuntimeSOHInference(_RuntimeSOHInference):
    def predict_trajectory(
        self,
        raw_sequence: EarlyCycleSequence,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        self.calls.append((raw_sequence.cell_id, initial_soh))
        cycles = tuple(
            cycle for cycle in range(101, 501) if cycle != 139
        )
        return cycles, tuple(1.0 - index / 10000 for index in cycles)


def test_verified_cell_predictor_runs_only_server_resolved_sequence() -> None:
    sequence = _early_sequence()
    source_identity = _source_identity()
    resolver = _CalibrationInputResolver(
        AdvancedCalibrationCellInput(
            source_registration_id=source_identity.registration_id,
            source_identity_sha256=source_identity.source_identity_sha256,
            cell_id=sequence.cell_id,
            cutoff_cycle=sequence.cutoff_cycle,
            data_version=sequence.data_version,
            feature_version=sequence.feature_version,
            split_version=source_identity.split_version,
            raw_sequence=sequence,
            initial_soh=0.99,
        )
    )
    predictor = VerifiedAdvancedCalibrationCellPredictor(
        input_resolver=resolver
    )
    rul_inference = _RuntimeRULInference()
    soh_inference = _RuntimeSOHInference()
    rul_runtime = _rul_runtime()
    soh_runtime = _soh_runtime()
    object.__setattr__(rul_runtime, "inference", rul_inference)
    object.__setattr__(soh_runtime, "inference", soh_inference)

    assert predictor.predict_rul(
        source_registration_id=source_identity.registration_id,
        source_identity=source_identity,
        cell_id=sequence.cell_id,
        runtime=rul_runtime,
    ) == 650.0
    cycles, predicted = predictor.predict_soh(
        source_registration_id=source_identity.registration_id,
        source_identity=source_identity,
        cell_id=sequence.cell_id,
        runtime=soh_runtime,
    )

    assert cycles == tuple(range(101, 501))
    assert len(predicted) == 400
    assert rul_inference.calls == ["cell-a"]
    assert soh_inference.calls == [("cell-a", 0.99)]
    assert len(resolver.calls) == 2


def test_verified_cell_predictor_preserves_sparse_runtime_axis() -> None:
    sequence = _early_sequence()
    source_identity = _source_identity()
    resolver = _CalibrationInputResolver(
        AdvancedCalibrationCellInput(
            source_registration_id=source_identity.registration_id,
            source_identity_sha256=source_identity.source_identity_sha256,
            cell_id=sequence.cell_id,
            cutoff_cycle=sequence.cutoff_cycle,
            data_version=sequence.data_version,
            feature_version=sequence.feature_version,
            split_version=source_identity.split_version,
            raw_sequence=sequence,
            initial_soh=0.99,
        )
    )
    predictor = VerifiedAdvancedCalibrationCellPredictor(
        input_resolver=resolver
    )
    inference = _SparseRuntimeSOHInference()
    runtime = _soh_runtime()
    object.__setattr__(runtime, "inference", inference)

    cycles, predicted = predictor.predict_soh(
        source_registration_id=source_identity.registration_id,
        source_identity=source_identity,
        cell_id=sequence.cell_id,
        runtime=runtime,
    )

    assert cycles == tuple(
        cycle for cycle in range(101, 501) if cycle != 139
    )
    assert len(predicted) == 399
    assert inference.calls == [("cell-a", 0.99)]


def _early_sequence() -> EarlyCycleSequence:
    sample_mask = torch.ones((101, 2, 150), dtype=torch.bool)
    values = torch.ones((101, 2, 150, 3), dtype=torch.float32)
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id="cell-a",
        cutoff_cycle=100,
        data_version="matr-three-batch-v1",
        feature_version="advanced-feature-v1",
        cycle_indices=tuple(range(101)),
        values=values,
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=("temperature_c", "charge_rate_c"),
        condition_values=torch.tensor([25.0, 1.0], dtype=torch.float32),
        condition_mask=torch.tensor([True, True], dtype=torch.bool),
    )


def test_rejects_rul_observation_at_the_cutoff_cycle() -> None:
    evidence = AdvancedRULCalibrationEvidence(
        source_identity=_source_identity(),
        cells=(AdvancedRULObservedCell(cell_id="cell-a", observed_cycle=100),),
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_rul_runtime()),
        predictor=_Predictor(),
    )

    with pytest.raises(ValueError, match="greater than cutoff"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=AdvancedCalibrationMaterializationRequest(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.RUL,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.COVERAGE,
                source_registration_id="matr-three-batch-final-v1",
            ),
            created_at=NOW,
        )


def test_rejects_soh_axis_that_does_not_start_after_cutoff() -> None:
    cycles = tuple(range(102, 501))
    evidence = AdvancedSOHCalibrationEvidence(
        source_identity=_source_identity(),
        prediction_cycles=cycles,
        cells=(
            AdvancedSOHObservedCell(
                cell_id="cell-a",
                observed_soh=tuple(1.0 - index / 9000 for index in cycles),
            ),
        ),
    )
    class _ShiftedAxisPredictor(_Predictor):
        def predict_soh(
            self,
            *,
            source_registration_id: str,
            source_identity: AdvancedCalibrationSourceIdentity,
            cell_id: str,
            runtime: VerifiedSOHRuntime,
        ) -> tuple[tuple[int, ...], tuple[float, ...]]:
            del source_registration_id, source_identity, runtime
            self.soh_calls.append(cell_id)
            return cycles, tuple(1.0 - index / 10000 for index in cycles)

    predictor = _ShiftedAxisPredictor()
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=_EvidenceResolver(evidence),
        runtime_resolver=_RuntimeResolver(_soh_runtime()),
        predictor=predictor,
    )

    with pytest.raises(ValueError, match=r"cutoff \+ 1 through cycle 500"):
        producer.produce(
            context=_context(),
            materialization_id=MATERIALIZATION_ID,
            request=AdvancedCalibrationMaterializationRequest(
                project_id=PROJECT_ID,
                task=AdvancedModelTask.SOH,
                cutoff_cycle=100,
                route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
                source_registration_id="matr-three-batch-final-v1",
            ),
            created_at=NOW,
        )
