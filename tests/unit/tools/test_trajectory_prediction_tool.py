"""Contract tests for ledger-bound finite-horizon SOH trajectory prediction."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from typing import Any
from uuid import uuid4

import pytest

from quanxin_life.core import (
    CellMetadata,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.models.hybrid_degradation import (
    HybridDegradationPredictor,
    TrajectoryTrainingSample,
)
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName, ToolRegistry


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="synthetic-lfp",
        train=("cell-a", "cell-b", "cell-c", "cell-d"),
        validation=("cell-e",),
        calibration=("cell-f",),
        test=("cell-g",),
    )


def _sample(
    *,
    cell_id: str,
    initial_soh: float,
    degradation_scale: float,
) -> TrajectoryTrainingSample:
    target_cycles = (21, 30, 40, 50)
    return TrajectoryTrainingSample(
        dataset_id="synthetic-lfp",
        cell_id=cell_id,
        cutoff_cycle=20,
        observed_cycles=(0, 10, 20),
        observed_soh=(initial_soh, initial_soh - 0.01, initial_soh - 0.02),
        target_cycles=target_cycles,
        target_soh=tuple(
            initial_soh - 0.02 - degradation_scale * (cycle - 20)
            for cycle in target_cycles
        ),
        condition_features={
            "nominal_capacity_ah": 1.0,
            "temperature_mean_c": 25.0 + degradation_scale * 100.0,
        },
        feature_version="early-cycle-v1",
        data_version="synthetic-data-v1",
    )


@pytest.fixture(scope="module")
def fitted_predictor() -> HybridDegradationPredictor:
    predictor = HybridDegradationPredictor(
        model_version="hybrid-degradation-v1",
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        data_version="synthetic-data-v1",
        cutoff_cycle=20,
        prediction_cycles=(21, 30, 40, 50),
        condition_feature_names=("nominal_capacity_ah", "temperature_mean_c"),
        hidden_dim=4,
        epochs=5,
        learning_rate=0.03,
    )
    samples = (
        _sample(cell_id="cell-a", initial_soh=1.00, degradation_scale=0.003),
        _sample(cell_id="cell-b", initial_soh=1.00, degradation_scale=0.004),
        _sample(cell_id="cell-c", initial_soh=0.99, degradation_scale=0.005),
        _sample(cell_id="cell-d", initial_soh=0.99, degradation_scale=0.006),
    )
    return predictor.fit(samples, split_manifest=_split())


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="registered-early-cycle-evidence",
            source_kind=SourceKind.OBSERVED,
            uri="tool-result://features/synthetic-cell-g",
            sha256="a" * 64,
            description="Cutoff-safe synthetic early-cycle evidence for tool-contract testing",
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def _feature_artifact(**overrides: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "record_batch_id": "trusted-batch-synthetic-cell-g",
        "dataset_id": "synthetic-lfp",
        "cell_id": "cell-g",
        "cutoff_cycle": 20,
        "observed_cycles": (0, 10, 20),
        "observed_soh": (1.00, 0.99, 0.98),
        "reference_capacity_ah": 1.0,
        "reference_capacity_method": "metadata_reference_capacity",
        "feature_values": {"delta_q_variance_ah2": None, "temperature_mean_c": 28.0},
        "condition_features": {"nominal_capacity_ah": 1.0, "temperature_mean_c": 28.0},
        "source_cycles": (0, 10, 20),
        "feature_warnings": (),
        "source_manifest_hash": "b" * 64,
        "feature_version": "early-cycle-v1",
        "split_version": "synthetic-split-v1",
        "data_version": "synthetic-data-v1",
    }
    artifact.update(overrides)
    return artifact


def _registered_upstream_result(
    *,
    tool_name: str = StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
    artifact_type: str = EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
    artifact: dict[str, object] | None = None,
) -> ToolResult:
    feature_artifact = artifact or _feature_artifact()
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool_name,
        tool_version=EARLY_CYCLE_FEATURE_TOOL_VERSION,
        model_version=EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
        data_version="synthetic-data-v1",
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"record_batch_id": "trusted-batch-synthetic-cell-g"}),
        values={"artifact_type": artifact_type, "artifact": feature_artifact},
        provenance=list(_provenance()),
        created_at=datetime(2026, 7, 13, tzinfo=UTC),
    )


def _audit_ledger(*results: ToolResult):
    from quanxin_life.audit import AuditLedger

    return AuditLedger(tuple(results))


def _input(*, upstream_result_id: str):
    from quanxin_life.tools.trajectory_prediction import PredictSOHTrajectoryToolInput

    return PredictSOHTrajectoryToolInput.model_validate(
        {"upstream_result_id": upstream_result_id}
    )


def _prediction_artifact(result: ToolResult) -> dict[str, Any]:
    from quanxin_life.tools.trajectory_prediction import PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE

    assert result.values["artifact_type"] == PREDICTED_SOH_TRAJECTORY_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert isinstance(artifact, dict)
    return artifact


def test_registered_trajectory_tool_uses_only_registered_feature_evidence_and_clock(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import register_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result()
    execution_time = datetime(2026, 7, 13, 9, 30, tzinfo=UTC)
    registry = ToolRegistry()
    register_predict_soh_trajectory_tool(
        registry,
        predictor=fitted_predictor,
        audit_ledger=_audit_ledger(upstream_result),
        clock=lambda: execution_time,
    )
    tool_input = _input(upstream_result_id=upstream_result.result_id)

    result = registry.execute(StandardToolName.PREDICT_SOH_TRAJECTORY, tool_input)
    artifact = _prediction_artifact(result)
    predicted_soh = artifact["predicted_soh"]
    crossing = artifact["eol80_crossing"]

    assert result.tool_name == StandardToolName.PREDICT_SOH_TRAJECTORY.value
    assert result.tool_version == "trajectory-prediction-tool-v1"
    assert result.model_version == fitted_predictor.model_version
    assert result.data_version == fitted_predictor.data_version
    assert result.feature_version == fitted_predictor.feature_version
    assert result.input_hash == sha256_canonical(tool_input.model_dump(mode="json"))
    assert result.created_at == execution_time
    assert artifact["prediction_cycles"] == list(fitted_predictor.prediction_cycles)
    assert artifact["upstream_result_id"] == upstream_result.result_id
    assert artifact["record_batch_id"] == "trusted-batch-synthetic-cell-g"
    assert result.provenance == upstream_result.provenance
    assert "MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY" in result.warnings
    assert artifact["model_artifact_status"] == "UNREGISTERED_IN_MEMORY"
    assert all(current <= previous for previous, current in pairwise(predicted_soh))
    assert crossing["cutoff_cycle"] == 20


def test_trajectory_tool_rejects_non_feature_legacy_or_unregistered_upstream_result(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    non_feature_result = _registered_upstream_result(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value
    )
    with pytest.raises(ValueError, match="extract_early_cycle_features"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=non_feature_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(non_feature_result),
        )

    legacy_result = _registered_upstream_result().model_copy(
        update={"values": {"trajectory_input": _feature_artifact()}}
    )
    with pytest.raises(ValueError, match="artifact_type"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=legacy_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(legacy_result),
        )

    with pytest.raises(ValueError, match="not registered"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=str(uuid4())),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(_registered_upstream_result()),
        )


def test_trajectory_tool_input_rejects_any_caller_supplied_numerical_or_provenance_fields() -> None:
    from quanxin_life.tools.trajectory_prediction import PredictSOHTrajectoryToolInput

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        PredictSOHTrajectoryToolInput.model_validate(
            {
                "upstream_result_id": str(uuid4()),
                "observed_soh": (1.00, 0.99, 0.98),
                "provenance": _provenance(),
            }
        )


def test_trajectory_tool_passes_only_standard_artifact_evidence_to_the_model(
    fitted_predictor: HybridDegradationPredictor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    registered_artifact = _feature_artifact(
        observed_soh=(0.97, 0.95, 0.91),
        condition_features={"nominal_capacity_ah": 1.2, "temperature_mean_c": 31.0},
    )
    upstream_result = _registered_upstream_result(artifact=registered_artifact)
    received: dict[str, Any] = {}
    original_predict = fitted_predictor.predict

    def recording_predict(**kwargs: Any):
        received.update(kwargs)
        return original_predict(**kwargs)

    monkeypatch.setattr(fitted_predictor, "predict", recording_predict)
    execute_predict_soh_trajectory_tool(
        _input(upstream_result_id=upstream_result.result_id),
        predictor=fitted_predictor,
        audit_ledger=_audit_ledger(upstream_result),
    )

    assert received["observed_soh"] == tuple(registered_artifact["observed_soh"])
    assert received["condition_features"] == registered_artifact["condition_features"]
    assert received["feature_version"] == registered_artifact["feature_version"]


def test_trajectory_tool_selects_the_fitted_condition_schema_from_source_evidence(
    fitted_predictor: HybridDegradationPredictor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result(
        artifact=_feature_artifact(
            condition_features={
                "nominal_capacity_ah": 1.2,
                "temperature_mean_c": 28.0,
                "capacity_first_ah": 1.2,
            }
        )
    )
    received: dict[str, Any] = {}
    original_predict = fitted_predictor.predict

    def recording_predict(**kwargs: Any):
        received.update(kwargs)
        return original_predict(**kwargs)

    monkeypatch.setattr(fitted_predictor, "predict", recording_predict)
    execute_predict_soh_trajectory_tool(
        _input(upstream_result_id=upstream_result.result_id),
        predictor=fitted_predictor,
        audit_ledger=_audit_ledger(upstream_result),
    )

    assert received["condition_features"] == {
        "nominal_capacity_ah": 1.2,
        "temperature_mean_c": 28.0,
    }


def test_trajectory_tool_rejects_source_evidence_missing_a_model_condition(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result(
        artifact=_feature_artifact(condition_features={"nominal_capacity_ah": 1.0})
    )
    with pytest.raises(ValueError, match="missing required condition features"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=upstream_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(upstream_result),
        )


def test_trajectory_tool_derives_rul_exactly_from_a_real_first_eol80_crossing(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result(
        artifact=_feature_artifact(observed_soh=(0.820001, 0.810001, 0.800001))
    )
    result = execute_predict_soh_trajectory_tool(
        _input(upstream_result_id=upstream_result.result_id),
        predictor=fitted_predictor,
        audit_ledger=_audit_ledger(upstream_result),
    )
    artifact = _prediction_artifact(result)

    crossing_cycle = artifact["eol80_crossing"]["eol80_cycle"]
    assert crossing_cycle == 21
    assert artifact["derived_rul_cycle"] == crossing_cycle - 20


@pytest.mark.parametrize(
    ("artifact", "error"),
    (
        (
            _feature_artifact(observed_soh=(1.00, 0.99, 1.51)),
            "observed SOH values must be in",
        ),
        (
            _feature_artifact(observed_soh=(1.00, 0.85, 0.80)),
            "observed SOH at cutoff already reached EOL80",
        ),
        (
            _feature_artifact(condition_features={"temperature_mean_c": "not-a-number"}),
            "registered early-cycle artifact evidence is invalid",
        ),
    ),
)
def test_trajectory_tool_rejects_invalid_standard_evidence_before_prediction(
    fitted_predictor: HybridDegradationPredictor,
    artifact: dict[str, object],
    error: str,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result(artifact=artifact)
    with pytest.raises(ValueError, match=error):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=upstream_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(upstream_result),
        )


def test_trajectory_tool_rejects_standard_evidence_with_mismatched_context(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result(
        artifact=_feature_artifact(feature_version="other-feature-v1")
    )
    with pytest.raises(ValueError, match="feature_version must match"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=upstream_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(upstream_result),
        )


def test_trajectory_tool_rejects_outer_and_artifact_version_mismatch(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    upstream_result = _registered_upstream_result().model_copy(
        update={"data_version": "other-data-v1"}
    )
    with pytest.raises(ValueError, match="data_version must match"):
        execute_predict_soh_trajectory_tool(
            _input(upstream_result_id=upstream_result.result_id),
            predictor=fitted_predictor,
            audit_ledger=_audit_ledger(upstream_result),
        )


def test_trajectory_tool_input_rejects_non_uuid_upstream_id() -> None:
    from quanxin_life.tools.trajectory_prediction import PredictSOHTrajectoryToolInput

    with pytest.raises(ValueError, match="upstream_result_id"):
        PredictSOHTrajectoryToolInput.model_validate({"upstream_result_id": "not-a-uuid"})


class _EarlyFeatureResolver:
    def __init__(self, batch: Any) -> None:
        self.batch = batch

    def resolve_verified_early_cycle_batch(self, record_batch_id: str) -> Any:
        if record_batch_id != self.batch.record_batch_id:
            raise ValueError("trusted early-cycle record batch was not found")
        return self.batch


def test_trajectory_tool_consumes_the_real_early_feature_tool_output(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    from quanxin_life.tools.early_cycle_features import (
        ExtractEarlyCycleFeaturesToolInput,
        VerifiedEarlyCycleBatch,
        execute_extract_early_cycle_features_tool,
    )
    from quanxin_life.tools.trajectory_prediction import execute_predict_soh_trajectory_tool

    records = tuple(
        CycleRecord(
            dataset_id="synthetic-lfp",
            cell_id="cell-g",
            cycle_index=cycle,
            sample_index=0,
            time_s=float(cycle),
            voltage_v=3.1,
            current_a=-1.0,
            temperature_c=26.0 + cycle / 10.0,
            discharge_capacity_ah=1.0 - cycle / 1000.0,
            diagnostic=True,
        )
        for cycle in (0, 10, 20)
    )
    batch = VerifiedEarlyCycleBatch(
        record_batch_id="trusted-feature-to-trajectory-batch",
        records=records,
        metadata=CellMetadata(
            dataset_id="synthetic-lfp",
            cell_id="cell-g",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.0,
            reference_capacity_ah=1.0,
            source_uri="trusted-store://trajectory/cell-g",
            source_sha256="d" * 64,
            schema_version="cell-metadata-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="synthetic-data-v1",
        split_version="synthetic-split-v1",
        source_manifest_hash="e" * 64,
        provenance=_provenance(),
    )
    feature_result = execute_extract_early_cycle_features_tool(
        ExtractEarlyCycleFeaturesToolInput(record_batch_id=batch.record_batch_id),
        resolver=_EarlyFeatureResolver(batch),
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )

    result = execute_predict_soh_trajectory_tool(
        _input(upstream_result_id=feature_result.result_id),
        predictor=fitted_predictor,
        audit_ledger=_audit_ledger(feature_result),
    )

    artifact = _prediction_artifact(result)
    assert artifact["record_batch_id"] == batch.record_batch_id
    assert artifact["model_condition_feature_names"] == [
        "nominal_capacity_ah",
        "temperature_mean_c",
    ]


def test_trajectory_tool_can_be_registered_only_in_project_scope(
    fitted_predictor: HybridDegradationPredictor,
) -> None:
    import quanxin_life.tools.trajectory_prediction as module
    from quanxin_life.tools import ToolExecutionScope, ToolRegistry

    registry = ToolRegistry()
    module.register_project_predict_soh_trajectory_tool(
        registry,
        predictor=fitted_predictor,
        project_audit_ledger=object(),
    )

    assert registry.list_schemas() == ()
    assert [
        item.tool_name
        for item in registry.list_schemas(execution_scope=ToolExecutionScope.PROJECT)
    ] == [StandardToolName.PREDICT_SOH_TRAJECTORY]
