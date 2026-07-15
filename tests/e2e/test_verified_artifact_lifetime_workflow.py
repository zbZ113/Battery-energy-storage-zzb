"""Production-policy vertical test using only synthetic, non-business fixtures."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.lifetime_workflow import (
    LifetimeDecisionWorkflowRequest,
    LifetimeDecisionWorkflowStatus,
    run_lifetime_decision_workflow,
)
from quanxin_life.application.model_artifacts import (
    ArtifactFormat,
    ArtifactKind,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    load_verified_xgboost_life_predictor,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    CellMetadata,
    LifePrediction,
    ProvenanceRecord,
    SourceKind,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.decision import BatchDecisionPolicy
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.models.xgboost import XGBoostLifePredictor
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.batch_decision import (
    VerifiedBatchDecisionPolicy,
    register_batch_decision_tool,
)
from quanxin_life.tools.conformal_calibration import (
    VerifiedNormalizedCalibrationCohort,
    VerifiedPredictionDifficultyScale,
    register_calibrate_prediction_interval_tool,
)
from quanxin_life.tools.cycle_life_prediction import register_predict_cycle_life_tool
from quanxin_life.tools.data_quality import register_validate_battery_data_tool
from quanxin_life.tools.early_cycle_features import (
    VerifiedEarlyCycleBatch,
    register_extract_early_cycle_features_tool,
)
from quanxin_life.tools.registry import ToolRegistry
from quanxin_life.uncertainty import ScaledLifePrediction

DATASET_ID = "synthetic-lfp-e2e"
DATA_VERSION = "synthetic-data-v1"
FEATURE_VERSION = "early-cycle-v1"
SPLIT_VERSION = "synthetic-split-v1"
MODEL_VERSION = "synthetic-xgboost-v1"
FEATURE_NAMES = ("nominal_capacity_ah", "capacity_delta_ah", "temperature_mean_c")
FIXED_TIME = datetime(2026, 7, 15, tzinfo=UTC)


def _source(source_id: str, source_kind: SourceKind) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=source_kind,
        uri=f"fixture://{source_id}",
        sha256=sha256_canonical({"synthetic_fixture": source_id}),
        description="Synthetic fixture evidence; not a real battery or business result",
        created_at=FIXED_TIME,
    )


def _split() -> SplitManifest:
    return SplitManifest(
        dataset_id=DATASET_ID,
        train=("train-a", "train-b", "train-c", "train-d"),
        validation=("validation-a",),
        calibration=("cal-a", "cal-b", "cal-c"),
        test=("target-a",),
    )


def _observed_label(cell_id: str, eol_cycle: int) -> LifePrediction:
    return LifePrediction(
        dataset_id=DATASET_ID,
        cell_id=cell_id,
        cutoff_cycle=20,
        predicted_eol_cycle=float(eol_cycle),
        observed_eol_cycle=eol_cycle,
        right_censored=False,
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        model_version=MODEL_VERSION,
        data_version=DATA_VERSION,
    )


def _verified_predictor(tmp_path: Path) -> tuple[XGBoostLifePredictor, ModelArtifactRegistry]:
    predictor = XGBoostLifePredictor(
        model_version=MODEL_VERSION,
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        data_version=DATA_VERSION,
        cutoff_cycle=20,
        feature_names=FEATURE_NAMES,
    ).fit(
        tuple(
            _observed_label(cell_id, eol)
            for cell_id, eol in (
                ("train-a", 120),
                ("train-b", 180),
                ("train-c", 260),
                ("train-d", 340),
            )
        ),
        training_features={
            "train-a": dict(zip(FEATURE_NAMES, (10.0, -0.05, 24.0), strict=True)),
            "train-b": dict(zip(FEATURE_NAMES, (10.0, -0.10, 25.0), strict=True)),
            "train-c": dict(zip(FEATURE_NAMES, (10.0, -0.15, 26.0), strict=True)),
            "train-d": dict(zip(FEATURE_NAMES, (10.0, -0.20, 27.0), strict=True)),
        },
        split_manifest=_split(),
    )
    relative_path = Path("models/lifetime.json")
    predictor.export_native_model(tmp_path / relative_path)
    payload = (tmp_path / relative_path).read_bytes()
    manifest = ModelArtifactManifest(
        artifact_id=str(uuid4()),
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_JSON,
        relative_path=relative_path.as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        model_version=MODEL_VERSION,
        data_version=DATA_VERSION,
        feature_version=FEATURE_VERSION,
        split_version=SPLIT_VERSION,
        schema_version="model-artifact-manifest-v1",
        dataset_id=DATASET_ID,
        cutoff_cycle=20,
        feature_names=FEATURE_NAMES,
        created_at=FIXED_TIME,
    )
    registry = ModelArtifactRegistry(tmp_path)
    registry.register(manifest)
    return load_verified_xgboost_life_predictor(registry, manifest.artifact_id), registry


def _target_batch() -> VerifiedEarlyCycleBatch:
    source = _source("target-a-records", SourceKind.OBSERVED)
    records = tuple(
        CycleRecord(
            dataset_id=DATASET_ID,
            cell_id="target-a",
            cycle_index=cycle,
            sample_index=0,
            time_s=float(cycle),
            voltage_v=3.2,
            current_a=-1.0,
            temperature_c=temperature,
            discharge_capacity_ah=capacity,
            diagnostic=True,
        )
        for cycle, capacity, temperature in (
            (0, 10.0, 25.0),
            (10, 9.9, 26.0),
            (20, 9.8, 27.0),
        )
    )
    return VerifiedEarlyCycleBatch(
        record_batch_id="synthetic-target-batch",
        records=records,
        metadata=CellMetadata(
            dataset_id=DATASET_ID,
            cell_id="target-a",
            chemistry="LFP/graphite",
            nominal_capacity_ah=10.0,
            reference_capacity_ah=10.0,
            source_uri=source.uri,
            source_sha256=source.sha256,
            schema_version="cell-metadata-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version=DATA_VERSION,
        split_version=SPLIT_VERSION,
        source_manifest_hash=source.sha256,
        provenance=(source,),
    )


class _TrustedContext:
    def __init__(self, batch: VerifiedEarlyCycleBatch) -> None:
        self.batch = batch

    def resolve_verified_early_cycle_batch(self, record_batch_id: str) -> VerifiedEarlyCycleBatch:
        if record_batch_id != self.batch.record_batch_id:
            raise ValueError("synthetic test batch not found")
        return self.batch

    def resolve_verified_normalized_calibration_cohort(
        self, calibration_cohort_id: str
    ) -> VerifiedNormalizedCalibrationCohort:
        if calibration_cohort_id != "synthetic-calibration-cohort":
            raise ValueError("synthetic calibration cohort not found")
        calibration_source = _source("calibration-cohort", SourceKind.OBSERVED)
        predictions = tuple(
            ScaledLifePrediction(
                prediction=_observed_label(cell_id, observed).model_copy(
                    update={"predicted_eol_cycle": predicted}
                ),
                difficulty_scale_cycle=scale,
                scale_version="synthetic-scale-v1",
            )
            for cell_id, predicted, observed, scale in (
                ("cal-a", 140.0, 150, 5.0),
                ("cal-b", 210.0, 190, 10.0),
                ("cal-c", 280.0, 310, 15.0),
            )
        )
        return VerifiedNormalizedCalibrationCohort(
            calibration_cohort_id=calibration_cohort_id,
            calibration_predictions=predictions,
            split_manifest=_split(),
            calibration_domain_id=DATASET_ID,
            source_manifest_hash=calibration_source.sha256,
            provenance=(calibration_source,),
        )

    def resolve_verified_prediction_difficulty_scale(
        self, prediction_result_id: str
    ) -> VerifiedPredictionDifficultyScale:
        scale_source = ProvenanceRecord(
            source_id="difficulty-scale",
            source_kind=SourceKind.PREDICTED,
            uri="fixture://difficulty-scale",
            sha256=self.batch.source_manifest_hash,
            description="Synthetic model scale derived from the target fixture manifest",
            created_at=FIXED_TIME,
        )
        return VerifiedPredictionDifficultyScale(
            prediction_result_id=prediction_result_id,
            dataset_id=DATASET_ID,
            cell_id="target-a",
            cutoff_cycle=20,
            feature_version=FEATURE_VERSION,
            split_version=SPLIT_VERSION,
            model_version=MODEL_VERSION,
            data_version=DATA_VERSION,
            difficulty_scale_cycle=10.0,
            scale_version="synthetic-scale-v1",
            target_domain_id=DATASET_ID,
            source_manifest_hash=scale_source.sha256,
            provenance=(scale_source,),
        )

    def resolve_verified_batch_decision_policy(
        self, policy_id: str
    ) -> VerifiedBatchDecisionPolicy:
        if policy_id != "synthetic-policy":
            raise ValueError("synthetic policy not found")
        policy_source = _source("decision-policy", SourceKind.OBSERVED)
        return VerifiedBatchDecisionPolicy(
            policy_id=policy_id,
            policy=BatchDecisionPolicy(
                policy_version="synthetic-policy-v1",
                required_eol_cycle=200.0,
            ),
            policy_domain_id=DATASET_ID,
            source_manifest_hash=policy_source.sha256,
            provenance=(policy_source,),
        )


def test_verified_native_artifact_runs_real_tools_from_records_to_audited_report(
    tmp_path: Path,
) -> None:
    """Exercise the production artifact policy without claiming model performance."""

    predictor, model_registry = _verified_predictor(tmp_path)
    context = _TrustedContext(_target_batch())
    ledger = AuditLedger()
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_extract_early_cycle_features_tool(registry, resolver=context)
    register_predict_cycle_life_tool(
        registry,
        predictor=predictor,
        audit_ledger=ledger,
        model_artifact_registry=model_registry,
    )
    register_calibrate_prediction_interval_tool(
        registry,
        resolver=context,
        audit_ledger=ledger,
        difficulty_scale_resolver=context,
    )
    register_batch_decision_tool(registry, audit_ledger=ledger, policy_resolver=context)
    register_generate_audited_report_tool(registry, audit_ledger=ledger)
    service = ToolInvocationService(registry=registry, audit_ledger=ledger)

    result = run_lifetime_decision_workflow(
        service,
        LifetimeDecisionWorkflowRequest(
            record_batch_id=context.batch.record_batch_id,
            calibration_cohort_id="synthetic-calibration-cohort",
            policy_id="synthetic-policy",
        ),
        batch_resolver=context,
    )

    assert result.status is LifetimeDecisionWorkflowStatus.COMPLETED
    assert result.prediction_result_id is not None
    assert result.report_result_id is not None
    prediction_result = ledger.resolve_registered_result(result.prediction_result_id)
    prediction_artifact = prediction_result.values["artifact"]
    assert prediction_artifact["model_artifact_status"] == "VERIFIED_ARTIFACT"
    assert prediction_artifact["model_artifact_id"] == predictor.model_artifact_id
    assert prediction_artifact["model_artifact_sha256"] == predictor.model_artifact_sha256
    report_result = ledger.resolve_registered_result(result.report_result_id)
    assert report_result.values["report_kind"] == "lifetime_decision"
    assert "##" in report_result.values["markdown"]
