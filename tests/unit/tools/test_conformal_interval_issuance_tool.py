"""Contracts for issuing a target-cell interval from trusted Conformal evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    LifePrediction,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.data.schemas import CycleRecord, SplitManifest
from quanxin_life.tools.cycle_life_prediction import (
    CYCLE_LIFE_PREDICTION_TOOL_VERSION,
    PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName
from quanxin_life.uncertainty import ScaledLifePrediction

if TYPE_CHECKING:
    from quanxin_life.tools.conformal_calibration import (
        VerifiedNormalizedCalibrationCohort,
        VerifiedPredictionDifficultyScale,
    )


def _provenance(source_id: str, source_kind: SourceKind) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=source_kind,
        uri=f"trusted-store://conformal-interval/{source_id}",
        sha256=sha256_canonical({"fixture": source_id}),
        description=f"Trusted {source_id} fixture evidence for interval issuance",
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _calibration_prediction(
    cell_id: str,
    *,
    predicted_eol_cycle: float,
    observed_eol_cycle: int,
    difficulty_scale_cycle: float,
) -> ScaledLifePrediction:
    return ScaledLifePrediction(
        prediction=LifePrediction(
            dataset_id="synthetic-lfp",
            cell_id=cell_id,
            cutoff_cycle=20,
            predicted_eol_cycle=predicted_eol_cycle,
            observed_eol_cycle=observed_eol_cycle,
            right_censored=False,
            feature_version="early-cycle-v1",
            split_version="split-v1",
            model_version="xgboost-eol80-v1",
            data_version="synthetic-data-v1",
        ),
        difficulty_scale_cycle=difficulty_scale_cycle,
        scale_version="residual-scale-v1",
    )


def _calibration_cohort() -> VerifiedNormalizedCalibrationCohort:
    from quanxin_life.tools.conformal_calibration import VerifiedNormalizedCalibrationCohort

    return VerifiedNormalizedCalibrationCohort(
        calibration_cohort_id="trusted-calibration-cohort-20260715",
        calibration_predictions=(
            _calibration_prediction(
                "calibration-1",
                predicted_eol_cycle=100.0,
                observed_eol_cycle=110,
                difficulty_scale_cycle=5.0,
            ),
            _calibration_prediction(
                "calibration-2",
                predicted_eol_cycle=200.0,
                observed_eol_cycle=180,
                difficulty_scale_cycle=10.0,
            ),
            _calibration_prediction(
                "calibration-3",
                predicted_eol_cycle=300.0,
                observed_eol_cycle=330,
                difficulty_scale_cycle=15.0,
            ),
        ),
        split_manifest=SplitManifest(
            dataset_id="synthetic-lfp",
            train=("train-1",),
            validation=("validation-1",),
            calibration=("calibration-1", "calibration-2", "calibration-3"),
            test=("cell-target-1",),
        ),
        calibration_domain_id="synthetic-lfp",
        source_manifest_hash=sha256_canonical({"fixture": "calibration-manifest"}),
        provenance=(_provenance("calibration-records", SourceKind.OBSERVED),),
    )


class _CalibrationResolver:
    def __init__(self, cohort: VerifiedNormalizedCalibrationCohort) -> None:
        self._cohort = cohort

    def resolve_verified_normalized_calibration_cohort(
        self, calibration_cohort_id: str
    ) -> VerifiedNormalizedCalibrationCohort:
        if calibration_cohort_id != self._cohort.calibration_cohort_id:
            raise ValueError("trusted normalized calibration cohort was not found")
        return self._cohort


def _calibration_result() -> ToolResult:
    from quanxin_life.tools.conformal_calibration import (
        CalibratePredictionIntervalToolInput,
        execute_calibrate_prediction_interval_tool,
    )

    cohort = _calibration_cohort()
    return execute_calibrate_prediction_interval_tool(
        CalibratePredictionIntervalToolInput(calibration_cohort_id=cohort.calibration_cohort_id),
        resolver=_CalibrationResolver(cohort),
        clock=lambda: datetime(2026, 7, 15, 9, tzinfo=UTC),
    )


def _point_prediction(*, cell_id: str = "cell-target-1") -> LifePrediction:
    return LifePrediction(
        dataset_id="synthetic-lfp",
        cell_id=cell_id,
        cutoff_cycle=20,
        predicted_eol_cycle=360.0,
        observed_eol_cycle=None,
        right_censored=True,
        feature_version="early-cycle-v1",
        split_version="split-v1",
        model_version="xgboost-eol80-v1",
        data_version="synthetic-data-v1",
    )


def _prediction_result(*, cell_id: str = "cell-target-1") -> ToolResult:
    prediction = _point_prediction(cell_id=cell_id)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
        tool_version=CYCLE_LIFE_PREDICTION_TOOL_VERSION,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
        feature_version=prediction.feature_version,
        input_hash=sha256_canonical({"fixture": "point-prediction"}),
        values={
            "artifact_type": PREDICTED_CYCLE_LIFE_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": "trusted-target-batch",
                "dataset_id": prediction.dataset_id,
                "cell_id": prediction.cell_id,
                "cutoff_cycle": prediction.cutoff_cycle,
                "life_prediction": prediction.model_dump(mode="json"),
                "derived_rul_cycle": prediction.derived_rul,
                "source_manifest_hash": sha256_canonical({"fixture": "target-manifest"}),
                "upstream_result_id": str(uuid4()),
                "split_version": prediction.split_version,
                "used_feature_names": ["nominal_capacity_ah"],
                "model_artifact_status": "REGISTERED",
            },
        },
        provenance=[_provenance("point-prediction", SourceKind.PREDICTED)],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _difficulty_scale(
    prediction_result_id: str,
    *,
    cell_id: str = "cell-target-1",
) -> VerifiedPredictionDifficultyScale:
    from quanxin_life.tools.conformal_calibration import VerifiedPredictionDifficultyScale

    prediction = _point_prediction(cell_id=cell_id)
    return VerifiedPredictionDifficultyScale(
        prediction_result_id=prediction_result_id,
        dataset_id=prediction.dataset_id,
        cell_id=prediction.cell_id,
        cutoff_cycle=prediction.cutoff_cycle,
        feature_version=prediction.feature_version,
        split_version=prediction.split_version,
        model_version=prediction.model_version,
        data_version=prediction.data_version,
        difficulty_scale_cycle=15.0,
        scale_version="residual-scale-v1",
        target_domain_id="synthetic-lfp",
        source_manifest_hash=sha256_canonical({"fixture": "target-manifest"}),
        provenance=(_provenance("difficulty-scale", SourceKind.PREDICTED),),
    )


class _DifficultyScaleResolver:
    def __init__(self, scale: VerifiedPredictionDifficultyScale) -> None:
        self._scale = scale
        self.calls: list[str] = []

    def resolve_verified_prediction_difficulty_scale(
        self, prediction_result_id: str
    ) -> VerifiedPredictionDifficultyScale:
        self.calls.append(prediction_result_id)
        if prediction_result_id != self._scale.prediction_result_id:
            raise ValueError("trusted prediction difficulty scale was not found")
        return self._scale


def test_interval_issuance_binds_registered_point_prediction_scale_and_calibration() -> None:
    from quanxin_life.tools.conformal_calibration import (
        NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE,
        CalibratePredictionIntervalToolInput,
        execute_calibrate_prediction_interval_tool,
    )

    calibration_result = _calibration_result()
    prediction_result = _prediction_result()
    scale_resolver = _DifficultyScaleResolver(_difficulty_scale(prediction_result.result_id))

    result = execute_calibrate_prediction_interval_tool(
        CalibratePredictionIntervalToolInput(
            prediction_result_id=prediction_result.result_id,
            calibration_result_id=calibration_result.result_id,
        ),
        resolver=_CalibrationResolver(_calibration_cohort()),
        audit_ledger=AuditLedger((prediction_result, calibration_result)),
        difficulty_scale_resolver=scale_resolver,
        clock=lambda: datetime(2026, 7, 15, 10, tzinfo=UTC),
    )

    assert scale_resolver.calls == [prediction_result.result_id]
    assert result.tool_name == StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value
    assert result.tool_version == "conformal-calibration-tool-v1"
    assert result.values["artifact_type"] == NORMALIZED_PREDICTION_INTERVAL_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert artifact["prediction_result_id"] == prediction_result.result_id
    assert artifact["calibration_result_id"] == calibration_result.result_id
    assert artifact["target_domain_calibrated"] is True
    assert artifact["prediction_interval"]["point_prediction_cycle"] == 360.0
    assert artifact["prediction_interval"]["lower_eol_cycle"] == 330.0
    assert artifact["prediction_interval"]["upper_eol_cycle"] == 390.0
    assert artifact["prediction_interval"]["difficulty_scale_cycle"] == 15.0
    assert result.uncertainty == {
        "coverage_target": 0.9,
        "difficulty_scale_cycle": 15.0,
        "lower_eol_cycle": 330.0,
        "upper_eol_cycle": 390.0,
    }
    assert result.created_at == datetime(2026, 7, 15, 10, tzinfo=UTC)


def test_interval_issuance_input_rejects_caller_scales_labels_intervals_and_alpha() -> None:
    from quanxin_life.tools.conformal_calibration import CalibratePredictionIntervalToolInput

    with pytest.raises(ValueError, match="Extra inputs"):
        CalibratePredictionIntervalToolInput.model_validate(
            {
                "prediction_result_id": str(uuid4()),
                "calibration_result_id": str(uuid4()),
                "difficulty_scale_cycle": 1.0,
                "observed_eol_cycle": 400,
                "prediction_interval": [100, 300],
                "alpha": 0.5,
            }
        )


def test_interval_issuance_rejects_unregistered_evidence_and_scale_context_mismatch() -> None:
    from quanxin_life.tools.conformal_calibration import (
        CalibratePredictionIntervalToolInput,
        execute_calibrate_prediction_interval_tool,
    )

    calibration_result = _calibration_result()
    prediction_result = _prediction_result()
    mismatched_scale = _difficulty_scale(prediction_result.result_id).model_copy(
        update={"scale_version": "unexpected-scale-v2"}
    )

    with pytest.raises(ValueError, match="not registered"):
        execute_calibrate_prediction_interval_tool(
            CalibratePredictionIntervalToolInput(
                prediction_result_id=str(uuid4()),
                calibration_result_id=calibration_result.result_id,
            ),
            resolver=_CalibrationResolver(_calibration_cohort()),
            audit_ledger=AuditLedger((prediction_result, calibration_result)),
            difficulty_scale_resolver=_DifficultyScaleResolver(
                _difficulty_scale(prediction_result.result_id)
            ),
        )

    with pytest.raises(ValueError, match="scale_version"):
        execute_calibrate_prediction_interval_tool(
            CalibratePredictionIntervalToolInput(
                prediction_result_id=prediction_result.result_id,
                calibration_result_id=calibration_result.result_id,
            ),
            resolver=_CalibrationResolver(_calibration_cohort()),
            audit_ledger=AuditLedger((prediction_result, calibration_result)),
            difficulty_scale_resolver=_DifficultyScaleResolver(mismatched_scale),
        )


def test_interval_issuance_rejects_calibration_cell_and_manifest_reuse() -> None:
    from quanxin_life.tools.conformal_calibration import (
        CalibratePredictionIntervalToolInput,
        execute_calibrate_prediction_interval_tool,
    )

    calibration_result = _calibration_result()
    reused_prediction = _prediction_result(cell_id="calibration-1")
    reused_scale = _difficulty_scale(
        reused_prediction.result_id,
        cell_id="calibration-1",
    )

    with pytest.raises(ValueError, match="calibration cell"):
        execute_calibrate_prediction_interval_tool(
            CalibratePredictionIntervalToolInput(
                prediction_result_id=reused_prediction.result_id,
                calibration_result_id=calibration_result.result_id,
            ),
            resolver=_CalibrationResolver(_calibration_cohort()),
            audit_ledger=AuditLedger((reused_prediction, calibration_result)),
            difficulty_scale_resolver=_DifficultyScaleResolver(reused_scale),
        )

    prediction_result = _prediction_result()
    wrong_manifest_scale = _difficulty_scale(prediction_result.result_id).model_copy(
        update={"source_manifest_hash": sha256_canonical({"fixture": "wrong-manifest"})}
    )
    with pytest.raises(ValueError, match="source_manifest_hash"):
        execute_calibrate_prediction_interval_tool(
            CalibratePredictionIntervalToolInput(
                prediction_result_id=prediction_result.result_id,
                calibration_result_id=calibration_result.result_id,
            ),
            resolver=_CalibrationResolver(_calibration_cohort()),
            audit_ledger=AuditLedger((prediction_result, calibration_result)),
            difficulty_scale_resolver=_DifficultyScaleResolver(wrong_manifest_scale),
        )


def test_issued_interval_is_accepted_by_the_real_batch_decision_tool() -> None:
    """Exercise the production evidence chain without caller-supplied numerics."""
    from quanxin_life.decision import BatchDecisionPolicy
    from quanxin_life.tools.batch_decision import (
        BatchDecisionToolInput,
        VerifiedBatchDecisionPolicy,
        execute_batch_decision_tool,
    )
    from quanxin_life.tools.conformal_calibration import (
        CalibratePredictionIntervalToolInput,
        execute_calibrate_prediction_interval_tool,
    )
    from quanxin_life.tools.data_quality import (
        ValidateBatteryDataToolInput,
        execute_validate_battery_data_tool,
    )

    class PolicyResolver:
        def __init__(self, policy: VerifiedBatchDecisionPolicy) -> None:
            self.policy = policy

        def resolve_verified_batch_decision_policy(
            self, policy_id: str
        ) -> VerifiedBatchDecisionPolicy:
            if policy_id != self.policy.policy_id:
                raise ValueError("trusted policy was not found")
            return self.policy

    calibration_result = _calibration_result()
    prediction_result = _prediction_result()
    interval_result = execute_calibrate_prediction_interval_tool(
        CalibratePredictionIntervalToolInput(
            prediction_result_id=prediction_result.result_id,
            calibration_result_id=calibration_result.result_id,
        ),
        resolver=_CalibrationResolver(_calibration_cohort()),
        audit_ledger=AuditLedger((prediction_result, calibration_result)),
        difficulty_scale_resolver=_DifficultyScaleResolver(
            _difficulty_scale(prediction_result.result_id)
        ),
        clock=lambda: datetime(2026, 7, 15, 10, tzinfo=UTC),
    )
    quality_result = execute_validate_battery_data_tool(
        ValidateBatteryDataToolInput(
            records=(
                CycleRecord(
                    dataset_id="synthetic-lfp",
                    cell_id="cell-target-1",
                    cycle_index=0,
                    sample_index=0,
                    time_s=0.0,
                    voltage_v=3.2,
                    current_a=-1.0,
                    temperature_c=25.0,
                ),
                CycleRecord(
                    dataset_id="synthetic-lfp",
                    cell_id="cell-target-1",
                    cycle_index=0,
                    sample_index=1,
                    time_s=1.0,
                    voltage_v=3.2,
                    current_a=-1.0,
                    temperature_c=25.0,
                ),
            ),
            data_version="synthetic-data-v1",
            feature_version="raw-cycle-v1",
            provenance=(_provenance("target-quality", SourceKind.OBSERVED),),
            validated_at=datetime(2026, 7, 15, 10, tzinfo=UTC),
        )
    )
    policy = VerifiedBatchDecisionPolicy(
        policy_id="approved-policy-synthetic-v1",
        policy=BatchDecisionPolicy(
            policy_version="batch-policy-v1",
            required_eol_cycle=300.0,
        ),
        policy_domain_id="synthetic-lfp",
        source_manifest_hash=sha256_canonical({"fixture": "approved-policy"}),
        provenance=(_provenance("approved-policy", SourceKind.OBSERVED),),
    )

    decision = execute_batch_decision_tool(
        BatchDecisionToolInput(
            prediction_interval_result_id=interval_result.result_id,
            calibration_result_id=calibration_result.result_id,
            quality_result_id=quality_result.result_id,
            policy_id=policy.policy_id,
        ),
        audit_ledger=AuditLedger((interval_result, calibration_result, quality_result)),
        policy_resolver=PolicyResolver(policy),
        clock=lambda: datetime(2026, 7, 15, 11, tzinfo=UTC),
    )

    assert decision.values["decision"] == "ADMIT"
    assert decision.values["interval_lower_eol_cycle"] == 330.0
    assert decision.values["interval_upper_eol_cycle"] == 390.0
    assert decision.values["upstream_result_ids"] == [
        interval_result.result_id,
        calibration_result.result_id,
        quality_result.result_id,
    ]
    assert decision.created_at == datetime(2026, 7, 15, 11, tzinfo=UTC)
