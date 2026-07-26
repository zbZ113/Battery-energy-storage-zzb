"""Contracts for ledger-bound EOL80 cycle-life prediction."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    CellMetadata,
    LifePrediction,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.models.xgboost import XGBoostLifePredictor
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
    ExtractEarlyCycleFeaturesToolInput,
    VerifiedEarlyCycleBatch,
    execute_extract_early_cycle_features_tool,
)
from quanxin_life.tools.registry import StandardToolName, ToolRegistry

FEATURE_NAMES = (
    "nominal_capacity_ah",
    "capacity_delta_ah",
    "temperature_mean_c",
)


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id="synthetic-lfp",
        train=("cell-a", "cell-b", "cell-c", "cell-d"),
        validation=("cell-e",),
        calibration=("cell-f",),
        test=("cell-g",),
    )


def _label(cell_id: str, observed_eol_cycle: int) -> LifePrediction:
    return LifePrediction(
        dataset_id="synthetic-lfp",
        cell_id=cell_id,
        cutoff_cycle=20,
        predicted_eol_cycle=float(observed_eol_cycle),
        observed_eol_cycle=observed_eol_cycle,
        right_censored=False,
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        model_version="label-source-v1",
        data_version="synthetic-data-v1",
    )


@pytest.fixture(scope="module")
def fitted_predictor() -> XGBoostLifePredictor:
    predictor = XGBoostLifePredictor(
        model_version="xgboost-eol80-v1",
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        data_version="synthetic-data-v1",
        cutoff_cycle=20,
        feature_names=FEATURE_NAMES,
    )
    return predictor.fit(
        (
            _label("cell-a", 100),
            _label("cell-b", 180),
            _label("cell-c", 260),
            _label("cell-d", 340),
        ),
        training_features={
            "cell-a": {
                "nominal_capacity_ah": 10.0,
                "capacity_delta_ah": -0.05,
                "temperature_mean_c": 24.0,
            },
            "cell-b": {
                "nominal_capacity_ah": 10.0,
                "capacity_delta_ah": -0.10,
                "temperature_mean_c": 25.0,
            },
            "cell-c": {
                "nominal_capacity_ah": 10.0,
                "capacity_delta_ah": -0.15,
                "temperature_mean_c": 26.0,
            },
            "cell-d": {
                "nominal_capacity_ah": 10.0,
                "capacity_delta_ah": -0.20,
                "temperature_mean_c": 27.0,
            },
        },
        split_manifest=_split(),
    )


def _metadata() -> CellMetadata:
    return CellMetadata(
        dataset_id="synthetic-lfp",
        cell_id="cell-g",
        chemistry="LFP/graphite",
        nominal_capacity_ah=10.0,
        reference_capacity_ah=10.0,
        source_uri="trusted-store://features/cell-g",
        source_sha256=sha256_canonical({"fixture": "cell-g"}),
        schema_version="cell-metadata-v1",
    )


def _records() -> tuple[CycleRecord, ...]:
    return tuple(
        CycleRecord(
            dataset_id="synthetic-lfp",
            cell_id="cell-g",
            cycle_index=cycle,
            sample_index=0,
            time_s=float(cycle),
            voltage_v=3.1,
            current_a=-1.0,
            temperature_c=temperature,
            discharge_capacity_ah=capacity,
            diagnostic=True,
        )
        for cycle, capacity, temperature in ((0, 10.0, 25.0), (10, 9.9, 26.0), (20, 9.8, 27.0))
    )


def _provenance(source_kind: SourceKind = SourceKind.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="trusted-synthetic-cell-g",
        source_kind=source_kind,
        uri="trusted-store://features/cell-g",
        sha256=sha256_canonical({"fixture": "cell-g-records"}),
        description="Verified cutoff-safe synthetic early-cycle records for tool contracts",
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )


class _Resolver:
    def __init__(self, batch: VerifiedEarlyCycleBatch) -> None:
        self._batch = batch

    def resolve_verified_early_cycle_batch(self, record_batch_id: str) -> VerifiedEarlyCycleBatch:
        if record_batch_id != self._batch.record_batch_id:
            raise ValueError("trusted early-cycle record batch was not found")
        return self._batch


def _feature_result() -> ToolResult:
    batch = VerifiedEarlyCycleBatch(
        record_batch_id="trusted-cell-g-batch",
        records=_records(),
        metadata=_metadata(),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="synthetic-data-v1",
        split_version="synthetic-split-v1",
        source_manifest_hash=sha256_canonical({"fixture": "cell-g-manifest"}),
        provenance=(_provenance(),),
    )
    return execute_extract_early_cycle_features_tool(
        ExtractEarlyCycleFeaturesToolInput(record_batch_id=batch.record_batch_id),
        resolver=_Resolver(batch),
        clock=lambda: datetime(2026, 7, 14, 9, tzinfo=UTC),
    )


def _input(upstream_result_id: str):
    from quanxin_life.tools.cycle_life_prediction import PredictCycleLifeToolInput

    return PredictCycleLifeToolInput(upstream_result_id=upstream_result_id)


def _artifact(result: ToolResult) -> dict[str, object]:
    from quanxin_life.tools.cycle_life_prediction import PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE

    assert result.values["artifact_type"] == PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert isinstance(artifact, dict)
    return artifact


def test_cycle_life_tool_uses_actual_early_feature_evidence_and_declared_predictor_schema(
    fitted_predictor: XGBoostLifePredictor,
) -> None:
    from quanxin_life.tools.cycle_life_prediction import register_predict_cycle_life_tool

    upstream = _feature_result()
    registry = ToolRegistry()
    register_predict_cycle_life_tool(
        registry,
        predictor=fitted_predictor,
        audit_ledger=AuditLedger((upstream,)),
        clock=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
    )
    tool_input = _input(upstream.result_id)

    result = registry.execute(StandardToolName.PREDICT_CYCLE_LIFE, tool_input)
    artifact = _artifact(result)
    prediction = artifact["life_prediction"]

    assert result.tool_name == StandardToolName.PREDICT_CYCLE_LIFE.value
    assert result.tool_version == "cycle-life-prediction-tool-v1"
    assert result.model_version == fitted_predictor.model_version
    assert result.data_version == upstream.data_version
    assert result.feature_version == upstream.feature_version
    assert result.created_at == datetime(2026, 7, 14, 10, tzinfo=UTC)
    assert artifact["upstream_result_id"] == upstream.result_id
    assert artifact["used_feature_names"] == list(FEATURE_NAMES)
    assert artifact["model_artifact_status"] == "UNREGISTERED_IN_MEMORY"
    assert isinstance(prediction, dict)
    assert prediction["right_censored"] is True
    assert prediction["observed_eol_cycle"] is None
    assert "MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY" in result.warnings
    assert result.provenance == upstream.provenance
    assert result.uncertainty is None


def test_cycle_life_tool_marks_only_registry_loaded_native_model_as_verified(
    fitted_predictor: XGBoostLifePredictor,
    tmp_path: Path,
) -> None:
    from quanxin_life.application.model_artifacts import (
        ArtifactFormat,
        ArtifactKind,
        ModelArtifactManifest,
        ModelArtifactRegistry,
        load_verified_xgboost_life_predictor,
    )
    from quanxin_life.tools.cycle_life_prediction import register_predict_cycle_life_tool

    relative_path = Path("models/xgboost.json")
    fitted_predictor.export_native_model(tmp_path / relative_path)
    payload = (tmp_path / relative_path).read_bytes()
    manifest = ModelArtifactManifest(
        artifact_id=str(uuid4()),
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_JSON,
        relative_path=relative_path.as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        model_version=fitted_predictor.model_version,
        data_version=fitted_predictor.data_version,
        feature_version=fitted_predictor.feature_version,
        split_version=fitted_predictor.split_version,
        schema_version="model-artifact-manifest-v1",
        dataset_id="synthetic-lfp",
        cutoff_cycle=fitted_predictor.cutoff_cycle,
        feature_names=fitted_predictor.feature_names,
        created_at=datetime(2026, 7, 14, tzinfo=UTC),
    )
    model_registry = ModelArtifactRegistry(tmp_path)
    model_registry.register(manifest)
    predictor = load_verified_xgboost_life_predictor(
        model_registry,
        manifest.artifact_id,
    )
    upstream = _feature_result()
    registry = ToolRegistry()
    register_predict_cycle_life_tool(
        registry,
        predictor=predictor,
        audit_ledger=AuditLedger((upstream,)),
        model_artifact_registry=model_registry,
        clock=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
    )

    result = registry.execute(
        StandardToolName.PREDICT_CYCLE_LIFE,
        _input(upstream.result_id),
    )
    artifact = _artifact(result)

    assert artifact["model_artifact_status"] == "VERIFIED_ARTIFACT"
    assert artifact["model_artifact_id"] == manifest.artifact_id
    assert artifact["model_artifact_sha256"] == manifest.sha256
    assert "MODEL_ARTIFACT_UNREGISTERED_IN_MEMORY" not in result.warnings
    assert manifest.sha256 in {item.sha256 for item in result.provenance}


def test_cycle_life_input_rejects_direct_features_labels_versions_and_non_uuid_ids() -> None:
    from quanxin_life.tools.cycle_life_prediction import PredictCycleLifeToolInput

    with pytest.raises(ValueError, match="UUID"):
        PredictCycleLifeToolInput(upstream_result_id="not-a-uuid")
    with pytest.raises(ValueError, match="Extra inputs"):
        PredictCycleLifeToolInput.model_validate(
            {
                "upstream_result_id": str(uuid4()),
                "features": {"capacity_delta_ah": -0.2},
                "observed_eol_cycle": 400,
                "model_version": "caller-controlled-v1",
            }
        )


def test_cycle_life_tool_rejects_unregistered_legacy_or_wrong_upstream_tool(
    fitted_predictor: XGBoostLifePredictor,
) -> None:
    from quanxin_life.tools.cycle_life_prediction import execute_predict_cycle_life_tool

    upstream = _feature_result()
    with pytest.raises(ValueError, match="not registered"):
        execute_predict_cycle_life_tool(
            _input(str(uuid4())), predictor=fitted_predictor, audit_ledger=AuditLedger((upstream,))
        )

    legacy = upstream.model_copy(update={"values": {"features": {"temperature_mean_c": 26.0}}})
    with pytest.raises(ValueError, match="artifact_type"):
        execute_predict_cycle_life_tool(
            _input(legacy.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((legacy,)),
        )

    wrong_tool = upstream.model_copy(
        update={"tool_name": StandardToolName.VALIDATE_BATTERY_DATA.value}
    )
    with pytest.raises(ValueError, match="extract_early_cycle_features"):
        execute_predict_cycle_life_tool(
            _input(wrong_tool.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((wrong_tool,)),
        )


def test_cycle_life_tool_rejects_missing_features_version_mismatch_and_untrusted_provenance(
    fitted_predictor: XGBoostLifePredictor,
) -> None:
    from quanxin_life.tools.cycle_life_prediction import execute_predict_cycle_life_tool

    upstream = _feature_result()
    source_artifact = upstream.values["artifact"]
    assert isinstance(source_artifact, dict)

    missing_feature_artifact = {
        **source_artifact,
        "condition_features": {
            key: value
            for key, value in source_artifact["condition_features"].items()
            if key != "temperature_mean_c"
        },
    }
    missing_feature = upstream.model_copy(
        update={
            "values": {
                "artifact_type": EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
                "artifact": missing_feature_artifact,
            }
        }
    )
    with pytest.raises(ValueError, match="missing required predictor feature"):
        execute_predict_cycle_life_tool(
            _input(missing_feature.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((missing_feature,)),
        )

    version_mismatch = upstream.model_copy(update={"data_version": "caller-data-v2"})
    with pytest.raises(ValueError, match="data_version must match"):
        execute_predict_cycle_life_tool(
            _input(version_mismatch.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((version_mismatch,)),
        )

    untrusted_provenance = upstream.model_copy(
        update={"provenance": [_provenance(SourceKind.SIMULATED)]}
    )
    with pytest.raises(ValueError, match="OBSERVED"):
        execute_predict_cycle_life_tool(
            _input(untrusted_provenance.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((untrusted_provenance,)),
        )


def test_cycle_life_tool_rejects_tampered_early_feature_tool_versions(
    fitted_predictor: XGBoostLifePredictor,
) -> None:
    from quanxin_life.tools.cycle_life_prediction import execute_predict_cycle_life_tool

    upstream = _feature_result()
    old_tool = upstream.model_copy(update={"tool_version": "old-feature-tool-v0"})
    with pytest.raises(ValueError, match="tool_version"):
        execute_predict_cycle_life_tool(
            _input(old_tool.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((old_tool,)),
        )

    old_model = upstream.model_copy(update={"model_version": "old-feature-model-v0"})
    with pytest.raises(ValueError, match="model_version"):
        execute_predict_cycle_life_tool(
            _input(old_model.result_id),
            predictor=fitted_predictor,
            audit_ledger=AuditLedger((old_model,)),
        )


def test_cycle_life_tool_can_be_registered_only_in_project_scope(
    fitted_predictor: XGBoostLifePredictor,
) -> None:
    import quanxin_life.tools.cycle_life_prediction as module
    from quanxin_life.tools import ToolExecutionScope, ToolRegistry

    registry = ToolRegistry()
    module.register_project_predict_cycle_life_tool(
        registry,
        predictor=fitted_predictor,
        project_audit_ledger=object(),
    )

    assert registry.list_schemas() == ()
    assert [
        item.tool_name
        for item in registry.list_schemas(execution_scope=ToolExecutionScope.PROJECT)
    ] == [StandardToolName.PREDICT_CYCLE_LIFE]
