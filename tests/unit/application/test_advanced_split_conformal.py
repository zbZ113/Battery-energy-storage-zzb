from __future__ import annotations

from uuid import uuid4

import pytest

from quanxin_life.application.advanced_split_conformal import (
    AdvancedConformalRuntimeIdentity,
    AdvancedRULCalibrationSample,
    AdvancedRULPointPrediction,
    AdvancedSOHCalibrationSample,
    AdvancedSOHPointPrediction,
    calibrate_advanced_rul_split_conformal,
    calibrate_advanced_soh_split_conformal,
    issue_advanced_rul_interval,
    issue_advanced_soh_band,
)
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask


def _identity(
    *,
    task: AdvancedModelTask,
    role: AdvancedModelRouteRole,
    artifact_kind: str,
    cutoff_cycle: int = 20,
) -> AdvancedConformalRuntimeIdentity:
    return AdvancedConformalRuntimeIdentity(
        task=task,
        route_role=role,
        output_target=(
            "matr_official_cycle_life"
            if task is AdvancedModelTask.RUL
            else "soh_trajectory"
        ),
        artifact_kind=artifact_kind,
        artifact_id=str(uuid4()),
        artifact_manifest_sha256="a" * 64,
        model_version="advanced-model-v1",
        dataset_id="MATR",
        cutoff_cycle=cutoff_cycle,
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        split_version="matr-cell-split-v1",
        normalization_statistics_sha256="b" * 64,
        decision_event_id=str(uuid4()),
        ledger_sequence_number=7,
        ledger_head_sha256="c" * 64,
    )


def test_rul_split_conformal_uses_coverage_point_and_derives_rul_bounds() -> None:
    identity = _identity(
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.DEFAULT,
        artifact_kind="cyclepatch_direct",
    )
    calibration = calibrate_advanced_rul_split_conformal(
        tuple(
            AdvancedRULCalibrationSample(
                cell_id=f"cal-{index}",
                point_prediction_cycle=point,
                observed_cycle=observed,
                runtime=identity,
            )
            for index, (point, observed) in enumerate(
                ((100.0, 105),) * 8 + ((110.0, 120),)
            )
        ),
        alpha=0.10,
    )

    interval = issue_advanced_rul_interval(
        AdvancedRULPointPrediction(
            cell_id="target",
            point_prediction_cycle=120.0,
            runtime=identity,
        ),
        calibration,
    )

    assert calibration.residual_quantile_cycle == 10.0
    assert interval.point_prediction_cycle == 120.0
    assert interval.lower_cycle == 110.0
    assert interval.upper_cycle == 130.0
    assert interval.derived_rul_cycle == 100.0
    assert interval.lower_rul_cycle == 90.0
    assert interval.upper_rul_cycle == 110.0


def test_rul_split_conformal_rejects_non_coverage_or_changed_runtime() -> None:
    with pytest.raises(ValueError, match="COVERAGE"):
        _identity(
            task=AdvancedModelTask.RUL,
            role=AdvancedModelRouteRole.POINT_ACCURACY,
            artifact_kind="cyclepatch_direct",
        )

    coverage_identity = _identity(
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.DEFAULT,
        artifact_kind="cyclepatch_direct",
    )
    calibration = calibrate_advanced_rul_split_conformal(
        (
            AdvancedRULCalibrationSample(
                cell_id="cal-1",
                point_prediction_cycle=100.0,
                observed_cycle=105,
                runtime=coverage_identity,
            ),
        ),
        alpha=0.50,
    )
    changed = coverage_identity.model_copy(update={"artifact_id": str(uuid4())})
    with pytest.raises(ValueError, match="runtime identity"):
        issue_advanced_rul_interval(
            AdvancedRULPointPrediction(
                cell_id="target",
                point_prediction_cycle=120.0,
                runtime=changed,
            ),
            calibration,
        )


@pytest.mark.parametrize("task", (AdvancedModelTask.RUL, AdvancedModelTask.SOH))
def test_split_conformal_rejects_unattainable_finite_sample_coverage(
    task: AdvancedModelTask,
) -> None:
    identity = _identity(
        task=task,
        role=(
            AdvancedModelRouteRole.DEFAULT
            if task is AdvancedModelTask.RUL
            else AdvancedModelRouteRole.TAIL_EFFICIENCY
        ),
        artifact_kind=(
            "cyclepatch_direct"
            if task is AdvancedModelTask.RUL
            else "current_hybrid"
        ),
    )

    with pytest.raises(ValueError, match=r"calibration cells.*coverage"):
        if task is AdvancedModelTask.RUL:
            calibrate_advanced_rul_split_conformal(
                (
                    AdvancedRULCalibrationSample(
                        cell_id="cal-1",
                        point_prediction_cycle=100.0,
                        observed_cycle=105,
                        runtime=identity,
                    ),
                    AdvancedRULCalibrationSample(
                        cell_id="cal-2",
                        point_prediction_cycle=110.0,
                        observed_cycle=120,
                        runtime=identity,
                    ),
                ),
                alpha=0.10,
            )
        else:
            calibrate_advanced_soh_split_conformal(
                (
                    AdvancedSOHCalibrationSample(
                        cell_id="cal-1",
                        prediction_cycles=(21, 22),
                        predicted_soh=(1.0, 0.90),
                        observed_soh=(0.98, 0.85),
                        runtime=identity,
                    ),
                    AdvancedSOHCalibrationSample(
                        cell_id="cal-2",
                        prediction_cycles=(21, 22),
                        predicted_soh=(0.95, 0.80),
                        observed_soh=(0.90, 0.70),
                        runtime=identity,
                    ),
                ),
                alpha=0.10,
            )


def test_soh_split_conformal_uses_cell_level_simultaneous_residual() -> None:
    identity = _identity(
        task=AdvancedModelTask.SOH,
        role=AdvancedModelRouteRole.TAIL_EFFICIENCY,
        artifact_kind="current_hybrid",
    )
    calibration = calibrate_advanced_soh_split_conformal(
        tuple(
            AdvancedSOHCalibrationSample(
                cell_id=f"cal-{index}",
                prediction_cycles=(21, 22),
                predicted_soh=predicted,
                observed_soh=observed,
                runtime=identity,
            )
            for index, (predicted, observed) in enumerate(
                (((1.0, 0.90), (0.98, 0.85)),) * 8
                + (((0.95, 0.80), (0.90, 0.70)),)
            )
        ),
        alpha=0.10,
    )

    band = issue_advanced_soh_band(
        AdvancedSOHPointPrediction(
            cell_id="target",
            prediction_cycles=(21, 22),
            predicted_soh=(0.95, 0.80),
            runtime=identity,
        ),
        calibration,
    )

    assert calibration.residual_quantile_soh == pytest.approx(0.10)
    assert band.prediction_cycles == (21, 22)
    assert band.predicted_soh == (0.95, 0.80)
    assert band.lower_soh == pytest.approx((0.85, 0.70))
    assert band.upper_soh == pytest.approx((1.05, 0.90))
    assert band.finite_horizon_only is True
    assert band.coverage_scope == "simultaneous_finite_trajectory"


def test_soh_split_conformal_rejects_mixed_cycles_and_target_calibration_cell() -> None:
    identity = _identity(
        task=AdvancedModelTask.SOH,
        role=AdvancedModelRouteRole.MEAN_ACCURACY,
        artifact_kind="hybridpatch_v2",
    )
    with pytest.raises(ValueError, match="prediction cycles"):
        calibrate_advanced_soh_split_conformal(
            (
                AdvancedSOHCalibrationSample(
                    cell_id="cal-1",
                    prediction_cycles=(21, 22),
                    predicted_soh=(1.0, 0.90),
                    observed_soh=(0.98, 0.85),
                    runtime=identity,
                ),
                AdvancedSOHCalibrationSample(
                    cell_id="cal-2",
                    prediction_cycles=(21, 23),
                    predicted_soh=(0.95, 0.80),
                    observed_soh=(0.90, 0.70),
                    runtime=identity,
                ),
            ),
            alpha=0.10,
        )

    calibration = calibrate_advanced_soh_split_conformal(
        (
            AdvancedSOHCalibrationSample(
                cell_id="cal-1",
                prediction_cycles=(21, 22),
                predicted_soh=(1.0, 0.90),
                observed_soh=(0.98, 0.85),
                runtime=identity,
            ),
        ),
        alpha=0.50,
    )
    with pytest.raises(ValueError, match="calibration cell"):
        issue_advanced_soh_band(
            AdvancedSOHPointPrediction(
                cell_id="cal-1",
                prediction_cycles=(21, 22),
                predicted_soh=(0.95, 0.80),
                runtime=identity,
            ),
            calibration,
        )
