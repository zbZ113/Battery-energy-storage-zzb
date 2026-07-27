from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    CycleLifePrediction,
    PredictionTarget,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_RUL_PREDICTION_TOOL_VERSION,
    AdvancedRULInference,
    PredictAdvancedRULToolInput,
    execute_predict_advanced_rul_tool,
)
from quanxin_life.tools.advanced_input import (
    PrepareAdvancedInputToolInput,
    execute_prepare_advanced_input_tool,
)
from tests.unit.tools.test_project_advanced_input_tool import (
    _batch,
    _BatchResolver,
    _context,
)


class _ResultResolver:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def resolve_registered_result(self, result_id: str) -> object:
        self.calls.append(result_id)
        return self.result


class _InferenceService:
    def __init__(self, inference: AdvancedRULInference) -> None:
        self.inference = inference
        self.calls: list[tuple[str, str, AdvancedModelRouteRole]] = []

    def predict(
        self,
        context: object,
        evidence: object,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedRULInference:
        self.calls.append((context.project_id, evidence.record_batch_id, route_role))
        return self.inference


def _upstream_result() -> object:
    batch = _batch()
    return execute_prepare_advanced_input_tool(
        PrepareAdvancedInputToolInput(record_batch_id=batch.record_batch_id),
        context=_context(),
        batch_resolver=_BatchResolver(batch),
        clock=lambda: datetime(2026, 7, 26, 10, 0, tzinfo=UTC),
    )


def _inference(upstream: object) -> AdvancedRULInference:
    artifact = upstream.values["artifact"]
    return AdvancedRULInference(
        prediction=CycleLifePrediction(
            dataset_id=artifact["dataset_id"],
            cell_id=artifact["cell_id"],
            cutoff_cycle=artifact["cutoff_cycle"],
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            predicted_cycle=812.5,
            observed_cycle=None,
            right_censored=True,
            feature_version=upstream.feature_version,
            split_version=artifact["split_version"],
            model_version="cyclepatch-direct-cutoff-20-seed-38",
            data_version=upstream.data_version,
        ),
        task=AdvancedModelTask.RUL,
        route_role=AdvancedModelRouteRole.DEFAULT,
        output_target="matr_official_cycle_life",
        artifact_kind="cyclepatch_direct",
        artifact_id=str(uuid4()),
        artifact_manifest_sha256="d" * 64,
        raw_sequence_input_sha256=artifact["raw_sequence_input_sha256"],
        normalization_statistics_sha256="f" * 64,
        decision_event_id=str(uuid4()),
        ledger_sequence_number=7,
        ledger_head_sha256="1" * 64,
    )


def test_project_advanced_rul_wraps_only_verified_matr_official_prediction() -> None:
    upstream = _upstream_result()
    resolver = _ResultResolver(upstream)
    service = _InferenceService(_inference(upstream))
    input_value = PredictAdvancedRULToolInput(
        upstream_result_id=upstream.result_id,
        route_role=AdvancedModelRouteRole.DEFAULT,
    )

    result = execute_predict_advanced_rul_tool(
        input_value,
        context=_context(),
        result_resolver=resolver,
        inference_service=service,
        clock=lambda: datetime(2026, 7, 26, 11, 0, tzinfo=UTC),
    )

    assert result.tool_version == ADVANCED_RUL_PREDICTION_TOOL_VERSION
    assert result.values["artifact_type"] == ADVANCED_RUL_PREDICTION_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    prediction = artifact["cycle_life_prediction"]
    assert prediction["target"] == PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value
    assert prediction["predicted_cycle"] == 812.5
    assert artifact["derived_remaining_cycles"] == 792.5
    assert artifact["upstream_result_id"] == upstream.result_id
    assert artifact["artifact_manifest_sha256"] == "d" * 64
    assert result.uncertainty is None
    assert "official_life_label" not in repr(result.values)
    assert "observed_cycle': 900" not in repr(result.values)
    assert resolver.calls == [upstream.result_id]
    assert service.calls == [
        (
            "project-1",
            artifact["record_batch_id"],
            AdvancedModelRouteRole.DEFAULT,
        )
    ]
