from __future__ import annotations

from quanxin_life.application.advanced_prediction import (
    AdvancedRULPredictionService,
    AdvancedSOHPredictionService,
)
from quanxin_life.application.advanced_runtime import (
    VerifiedRULRuntime,
    VerifiedSOHRuntime,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactKind,
)
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask
from quanxin_life.tools.advanced_input import (
    PrepareAdvancedInputToolInput,
    decode_advanced_input_result,
    execute_prepare_advanced_input_tool,
)
from tests.unit.tools.test_project_advanced_input_tool import (
    _batch,
    _BatchResolver,
    _context,
)


class _RULAdapter:
    normalization_statistics_sha256 = "9" * 64

    def predict_cycle(self, raw_sequence: object) -> float:
        assert raw_sequence.input_hash
        return 812.5


class _SOHAdapter:
    normalization_statistics_sha256 = "9" * 64

    def predict_trajectory(
        self,
        raw_sequence: object,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        assert raw_sequence.input_hash
        assert initial_soh == 1.0
        return (21, 22, 23), (0.99, 0.98, 0.97)


class _RuntimeResolver:
    def __init__(self, runtime: object) -> None:
        self.runtime = runtime
        self.resolve_calls: list[tuple[AdvancedModelTask, int, AdvancedModelRouteRole]] = []
        self.revalidate_calls = 0

    def resolve(
        self,
        context: object,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> object:
        assert context.project_id == "project-1"
        self.resolve_calls.append((task, cutoff_cycle, role))
        return self.runtime

    def revalidate(self, context: object, runtime: object) -> object:
        assert context.project_id == "project-1"
        assert runtime is self.runtime
        self.revalidate_calls += 1
        return runtime


def _evidence() -> object:
    batch = _batch()
    result = execute_prepare_advanced_input_tool(
        PrepareAdvancedInputToolInput(record_batch_id=batch.record_batch_id),
        context=_context(),
        batch_resolver=_BatchResolver(batch),
    )
    return decode_advanced_input_result(result)


def test_rul_service_rebuilds_input_and_revalidates_active_runtime() -> None:
    evidence = _evidence()
    runtime = VerifiedRULRuntime(
        project_id="project-1",
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        dataset_id="MATR",
        data_version=evidence.data_version,
        feature_version=evidence.feature_version,
        split_version=evidence.split_version,
        normalization_sha256="9" * 64,
        artifact_id="00000000-0000-4000-8000-000000000001",
        artifact_manifest_sha256="a" * 64,
        model_version="cyclepatch-direct-cutoff-20-seed-38",
        decision_event_id="00000000-0000-4000-8000-000000000002",
        ledger_sequence_number=1,
        ledger_head_sha256="b" * 64,
        inference=_RULAdapter(),
    )
    resolver = _RuntimeResolver(runtime)

    inference = AdvancedRULPredictionService(
        batch_resolver=_BatchResolver(_batch().model_copy(
            update={"record_batch_id": evidence.record_batch_id}
        )),
        runtime_resolver=resolver,
    ).predict(
        _context(),
        evidence,
        route_role=AdvancedModelRouteRole.DEFAULT,
    )

    assert inference.prediction.predicted_cycle == 812.5
    assert inference.prediction.observed_cycle is None
    assert inference.raw_sequence_input_sha256 == evidence.raw_sequence_input_sha256
    assert resolver.resolve_calls == [
        (AdvancedModelTask.RUL, 20, AdvancedModelRouteRole.DEFAULT)
    ]
    assert resolver.revalidate_calls == 1


def test_soh_service_uses_verified_cutoff_capacity_and_selected_route() -> None:
    evidence = _evidence()
    runtime = VerifiedSOHRuntime(
        project_id="project-1",
        task=AdvancedModelTask.SOH,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.MEAN_ACCURACY,
        output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
        artifact_kind=DeepArtifactKind.HYBRIDPATCH_V2,
        dataset_id="MATR",
        data_version=evidence.data_version,
        feature_version=evidence.feature_version,
        split_version=evidence.split_version,
        normalization_sha256="9" * 64,
        artifact_id="00000000-0000-4000-8000-000000000003",
        artifact_manifest_sha256="c" * 64,
        model_version="hybridpatch-v2-cutoff-20-seed-38",
        decision_event_id="00000000-0000-4000-8000-000000000004",
        ledger_sequence_number=2,
        ledger_head_sha256="d" * 64,
        inference=_SOHAdapter(),
    )
    resolver = _RuntimeResolver(runtime)

    inference = AdvancedSOHPredictionService(
        batch_resolver=_BatchResolver(_batch().model_copy(
            update={"record_batch_id": evidence.record_batch_id}
        )),
        runtime_resolver=resolver,
    ).predict(
        _context(),
        evidence,
        route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
    )

    assert inference.prediction_cycles == (21, 22, 23)
    assert inference.predicted_soh == (0.99, 0.98, 0.97)
    assert resolver.resolve_calls == [
        (AdvancedModelTask.SOH, 20, AdvancedModelRouteRole.MEAN_ACCURACY)
    ]
    assert resolver.revalidate_calls == 1
