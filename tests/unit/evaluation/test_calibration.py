import pytest

from quanxin_life.core import CycleLifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.evaluation.calibration import (
    TrajectoryIntervalEvidence,
    materialize_cycle_life_intervals,
)


def _prediction(cell_id: str, predicted: float, observed: int) -> CycleLifePrediction:
    return CycleLifePrediction(
        dataset_id="matr",
        cell_id=cell_id,
        cutoff_cycle=20,
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        predicted_cycle=predicted,
        observed_cycle=observed,
        right_censored=False,
        feature_version="feature-v1",
        split_version="split-v1",
        model_version="model-v1",
        data_version="data-v1",
    )


def test_materializes_separate_80_and_90_percent_calibrations() -> None:
    split = SplitManifest(
        dataset_id="matr",
        train=("train",),
        validation=("validation",),
        calibration=("cal-1", "cal-2"),
        test=("test-1",),
    )
    materials = materialize_cycle_life_intervals(
        (_prediction("cal-1", 100.0, 110), _prediction("cal-2", 120.0, 125)),
        (_prediction("test-1", 130.0, 135),),
        split_manifest=split,
        coverages=(0.8, 0.9),
        split_manifest_sha256="a" * 64,
        prediction_evidence_sha256="b" * 64,
    )

    assert [material.target_coverage for material in materials] == [0.8, 0.9]
    assert all(material.coverage.evaluated_cell_count == 1 for material in materials)
    assert all("SMALL_CALIBRATION_COHORT" in material.warnings for material in materials)
    assert len({material.material_sha256 for material in materials}) == 2


def test_test_cell_cannot_calibrate_and_trajectory_interval_is_unavailable() -> None:
    split = SplitManifest(
        dataset_id="matr", train=("train",), validation=("validation",),
        calibration=("cal",), test=("test",),
    )
    with pytest.raises(ValueError, match="calibration split"):
        materialize_cycle_life_intervals(
            (_prediction("test", 100.0, 110),),
            (_prediction("test", 100.0, 110),),
            split_manifest=split,
            coverages=(0.8,),
            split_manifest_sha256="a" * 64,
            prediction_evidence_sha256="b" * 64,
        )
    evidence = TrajectoryIntervalEvidence(cell_id="cell", cycle=30)
    assert evidence.interval_method == "NOT_AVAILABLE"
    assert evidence.lower is None and evidence.upper is None

