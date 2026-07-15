"""Contracts for trusted normalized conformal calibration evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools.registry import StandardToolName, ToolRegistry
from quanxin_life.uncertainty import ScaledLifePrediction

if TYPE_CHECKING:
    from quanxin_life.tools.conformal_calibration import (
        CalibratePredictionIntervalToolInput,
        VerifiedNormalizedCalibrationCohort,
    )


def _split_manifest() -> SplitManifest:
    return SplitManifest(
        dataset_id="synthetic-lfp",
        train=("train-1",),
        validation=("validation-1",),
        calibration=("calibration-1", "calibration-2", "calibration-3"),
        test=("test-1",),
    )


def _scaled_prediction(
    cell_id: str,
    *,
    predicted_eol_cycle: float,
    observed_eol_cycle: int,
    difficulty_scale_cycle: float,
) -> ScaledLifePrediction:
    from quanxin_life.core import LifePrediction

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


def _provenance(source_kind: SourceKind = SourceKind.OBSERVED) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id="synthetic-calibration-cohort",
        source_kind=source_kind,
        uri="trusted-store://calibration/synthetic-lfp",
        sha256=sha256_canonical({"fixture": "normalized-conformal-calibration"}),
        description="Verified cell-disjoint calibration predictions for tool-contract tests",
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _cohort(
    *,
    calibration_scope: Literal[
        "in_distribution", "target_domain_recalibration"
    ] = "in_distribution",
) -> VerifiedNormalizedCalibrationCohort:
    from quanxin_life.tools.conformal_calibration import VerifiedNormalizedCalibrationCohort

    return VerifiedNormalizedCalibrationCohort(
        calibration_cohort_id="trusted-calibration-cohort-20260715",
        calibration_predictions=(
            _scaled_prediction(
                "calibration-1",
                predicted_eol_cycle=100.0,
                observed_eol_cycle=110,
                difficulty_scale_cycle=5.0,
            ),
            _scaled_prediction(
                "calibration-2",
                predicted_eol_cycle=200.0,
                observed_eol_cycle=180,
                difficulty_scale_cycle=10.0,
            ),
            _scaled_prediction(
                "calibration-3",
                predicted_eol_cycle=300.0,
                observed_eol_cycle=330,
                difficulty_scale_cycle=15.0,
            ),
        ),
        split_manifest=_split_manifest(),
        calibration_domain_id="synthetic-lfp-source",
        calibration_scope=calibration_scope,
        source_manifest_hash=sha256_canonical({"fixture": "calibration-manifest"}),
        provenance=(_provenance(),),
    )


class _Resolver:
    def __init__(self, cohort: VerifiedNormalizedCalibrationCohort) -> None:
        self._cohort = cohort
        self.calls: list[str] = []

    def resolve_verified_normalized_calibration_cohort(
        self, cohort_id: str
    ) -> VerifiedNormalizedCalibrationCohort:
        self.calls.append(cohort_id)
        if cohort_id != self._cohort.calibration_cohort_id:
            raise ValueError("trusted normalized calibration cohort was not found")
        return self._cohort


def _input() -> CalibratePredictionIntervalToolInput:
    from quanxin_life.tools.conformal_calibration import CalibratePredictionIntervalToolInput

    return CalibratePredictionIntervalToolInput(
        calibration_cohort_id="trusted-calibration-cohort-20260715"
    )


def test_calibration_tool_resolves_trusted_cohort_and_emits_standard_evidence() -> None:
    from quanxin_life.tools.conformal_calibration import (
        NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE,
        register_calibrate_prediction_interval_tool,
    )

    resolver = _Resolver(_cohort())
    registry = ToolRegistry()
    register_calibrate_prediction_interval_tool(
        registry,
        resolver=resolver,
        clock=lambda: datetime(2026, 7, 15, 9, tzinfo=UTC),
    )

    result = registry.execute(StandardToolName.CALIBRATE_PREDICTION_INTERVAL, _input())

    assert resolver.calls == ["trusted-calibration-cohort-20260715"]
    assert result.tool_name == StandardToolName.CALIBRATE_PREDICTION_INTERVAL.value
    assert result.tool_version == "conformal-calibration-tool-v1"
    assert result.model_version == "xgboost-eol80-v1"
    assert result.data_version == "synthetic-data-v1"
    assert result.feature_version == "early-cycle-v1"
    assert result.created_at == datetime(2026, 7, 15, 9, tzinfo=UTC)
    assert result.values["artifact_type"] == NORMALIZED_CONFORMAL_CALIBRATION_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert artifact["calibration_cohort_id"] == "trusted-calibration-cohort-20260715"
    assert artifact["calibration_domain_id"] == "synthetic-lfp-source"
    assert artifact["calibration_scope"] == "in_distribution"
    assert artifact["calibration"]["normalized_score_quantile"] == 2.0
    assert artifact["calibration"]["calibration_cell_count"] == 3
    assert artifact["split_manifest_hash"] == sha256_canonical(
        _split_manifest().model_dump(mode="json")
    )
    assert artifact["calibration_cell_ids"] == [
        "calibration-1",
        "calibration-2",
        "calibration-3",
    ]
    assert result.uncertainty is None
    assert result.provenance == [_provenance()]


def test_calibration_tool_input_rejects_caller_labels_residuals_intervals_and_alpha() -> None:
    from quanxin_life.tools.conformal_calibration import CalibratePredictionIntervalToolInput

    with pytest.raises(ValueError, match="Extra inputs"):
        CalibratePredictionIntervalToolInput.model_validate(
            {
                "calibration_cohort_id": "trusted-calibration-cohort-20260715",
                "alpha": 0.01,
                "observed_eol_cycle": 400,
                "normalized_residuals": [1.5, 2.0],
                "prediction_interval": [100, 300],
            }
        )


def test_calibration_tool_revalidates_resolver_output_and_requires_observed_provenance() -> None:
    from quanxin_life.tools.conformal_calibration import (
        VerifiedNormalizedCalibrationCohort,
        execute_calibrate_prediction_interval_tool,
    )

    trusted = _cohort()
    bypassed = VerifiedNormalizedCalibrationCohort.model_construct(
        calibration_cohort_id=trusted.calibration_cohort_id,
        calibration_predictions=trusted.calibration_predictions,
        split_manifest=trusted.split_manifest,
        calibration_domain_id=trusted.calibration_domain_id,
        calibration_scope=trusted.calibration_scope,
        source_manifest_hash=trusted.source_manifest_hash,
        provenance=(_provenance(SourceKind.PREDICTED),),
    )

    with pytest.raises(ValueError, match="OBSERVED"):
        execute_calibrate_prediction_interval_tool(_input(), resolver=_Resolver(bypassed))

    with pytest.raises(ValueError, match="not found"):
        execute_calibrate_prediction_interval_tool(
            _input().model_copy(update={"calibration_cohort_id": "unknown-calibration-cohort"}),
            resolver=_Resolver(trusted),
        )


def test_calibration_tool_marks_target_domain_recalibration_without_overclaiming_coverage() -> None:
    from quanxin_life.tools.conformal_calibration import execute_calibrate_prediction_interval_tool

    result = execute_calibrate_prediction_interval_tool(
        _input(),
        resolver=_Resolver(_cohort(calibration_scope="target_domain_recalibration")),
        clock=lambda: datetime(2026, 7, 15, 9, tzinfo=UTC),
    )

    assert "TARGET_DOMAIN_RECALIBRATION_APPLIED" in result.warnings
    assert "COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT" in result.warnings
    assert "guaranteed_target_domain_coverage" not in result.values["artifact"]


def test_calibration_tool_is_not_available_without_a_context_bound_resolver() -> None:
    from quanxin_life.tools.bootstrap import create_available_tool_registry

    schemas = create_available_tool_registry().list_schemas()

    assert StandardToolName.CALIBRATE_PREDICTION_INTERVAL not in {
        schema.tool_name for schema in schemas
    }
