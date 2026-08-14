from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask
from quanxin_life.tools.advanced_input import (
    PrepareAdvancedInputToolInput,
    execute_prepare_advanced_input_tool,
)
from quanxin_life.tools.advanced_soh_prediction import (
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE,
    ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1,
    ADVANCED_SOH_PREDICTION_TOOL_VERSION,
    AdvancedSOHInference,
    PredictAdvancedSOHToolInput,
    execute_predict_advanced_soh_tool,
)
from tests.unit.tools.test_project_advanced_cycle_life_tool import (
    _legacy_upstream_result,
    _ResultResolver,
)
from tests.unit.tools.test_project_advanced_input_tool import (
    _batch,
    _BatchResolver,
    _context,
)


class _SOHService:
    def __init__(self, inference: AdvancedSOHInference) -> None:
        self.inference = inference
        self.calls: list[tuple[str, str, AdvancedModelRouteRole]] = []

    def predict(
        self,
        context: object,
        evidence: object,
        *,
        route_role: AdvancedModelRouteRole,
    ) -> AdvancedSOHInference:
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


def _inference(upstream: object) -> AdvancedSOHInference:
    artifact = upstream.values["artifact"]
    return AdvancedSOHInference(
        dataset_id=artifact["dataset_id"],
        cell_id=artifact["cell_id"],
        cutoff_cycle=artifact["cutoff_cycle"],
        data_version=upstream.data_version,
        feature_version=upstream.feature_version,
        split_version=artifact["split_version"],
        model_version="hybridpatch-v2-cutoff-20-seed-38",
        task=AdvancedModelTask.SOH,
        route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
        output_target="soh_trajectory",
        artifact_kind="hybridpatch_v2",
        artifact_id=str(uuid4()),
        artifact_manifest_sha256="d" * 64,
        raw_sequence_input_sha256=artifact["raw_sequence_input_sha256"],
        normalization_statistics_sha256="f" * 64,
        prediction_cycles=(21, 22, 23),
        predicted_soh=(0.99, 0.98, 0.97),
        decision_event_id=str(uuid4()),
        ledger_sequence_number=8,
        ledger_head_sha256="1" * 64,
    )


def test_project_advanced_soh_wraps_only_finite_model_trajectory() -> None:
    upstream = _upstream_result()
    resolver = _ResultResolver(upstream)
    service = _SOHService(_inference(upstream))
    input_value = PredictAdvancedSOHToolInput(
        upstream_result_id=upstream.result_id,
        route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
    )

    result = execute_predict_advanced_soh_tool(
        input_value,
        context=_context(),
        result_resolver=resolver,
        inference_service=service,
        clock=lambda: datetime(2026, 7, 26, 11, 0, tzinfo=UTC),
    )

    assert result.tool_version == ADVANCED_SOH_PREDICTION_TOOL_VERSION
    assert result.values["artifact_type"] == ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE
    assert result.values["artifact_type"] == (
        "quanxin_life.advanced_soh_trajectory.v2"
    )
    artifact = result.values["artifact"]
    assert artifact["prediction_cycles"] == [21, 22, 23]
    assert artifact["predicted_soh"] == [0.99, 0.98, 0.97]
    assert artifact["route_role"] == AdvancedModelRouteRole.MEAN_ACCURACY.value
    assert artifact["horizon_end_cycle"] == 23
    assert artifact["cell_metadata"]["chemistry"] == "LFP/graphite"
    assert artifact["cell_metadata"]["nominal_capacity_ah"] == 1.1
    assert result.uncertainty == {
        "finite_horizon_only": True,
        "conformal_interval_included": False,
    }
    payload = repr(result.values).casefold()
    assert "rul" not in payload
    assert "official_life_label" not in payload
    assert service.calls == [
        (
            "project-1",
            artifact["record_batch_id"],
            AdvancedModelRouteRole.MEAN_ACCURACY,
        )
    ]


def test_legacy_input_keeps_soh_artifact_v1_without_metadata() -> None:
    upstream = _legacy_upstream_result()

    result = execute_predict_advanced_soh_tool(
        PredictAdvancedSOHToolInput(
            upstream_result_id=upstream.result_id,
            route_role=AdvancedModelRouteRole.MEAN_ACCURACY,
        ),
        context=_context(),
        result_resolver=_ResultResolver(upstream),
        inference_service=_SOHService(_inference(upstream)),
    )

    assert result.values["artifact_type"] == (
        ADVANCED_SOH_PREDICTION_EVIDENCE_TYPE_V1
    )
    assert "cell_metadata" not in result.values["artifact"]
