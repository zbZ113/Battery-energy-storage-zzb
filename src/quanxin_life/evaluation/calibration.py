"""Materialized split-conformal evidence bound to registered input hashes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import CycleLifePrediction, PredictionTarget, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.uncertainty.cycle_life_conformal import (
    CycleLifeConformalCalibration,
    CycleLifeIntervalCoverage,
    CycleLifePredictionInterval,
    calibrate_cycle_life_conformal,
    evaluate_cycle_life_interval_coverage,
    make_cycle_life_interval,
)


class CycleLifeIntervalMaterial(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "cycle-life-interval-material-v1"
    target_coverage: float = Field(gt=0, lt=1, allow_inf_nan=False)
    model_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    target: PredictionTarget
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    split_manifest_sha256: Sha256
    prediction_evidence_sha256: Sha256
    calibration: CycleLifeConformalCalibration
    intervals: tuple[CycleLifePredictionInterval, ...]
    coverage: CycleLifeIntervalCoverage
    warnings: tuple[str, ...]
    material_sha256: Sha256

    @model_validator(mode="after")
    def hash_matches_material(self) -> CycleLifeIntervalMaterial:
        payload = self.model_dump(mode="json", exclude={"material_sha256"})
        if sha256_canonical(payload) != self.material_sha256:
            raise ValueError("material_sha256 does not match conformal evidence")
        return self


class TrajectoryIntervalEvidence(ContractModel):
    """Honest placeholder until trajectory conformal coverage is implemented."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    cycle: int = Field(ge=0)
    lower: None = None
    upper: None = None
    interval_method: Literal["NOT_AVAILABLE"] = "NOT_AVAILABLE"
    coverage_guarantee: Literal["NONE"] = "NONE"
    warning: str = "NO_TRAJECTORY_COVERAGE_GUARANTEE"


def materialize_cycle_life_intervals(
    calibration_predictions: Sequence[CycleLifePrediction],
    test_predictions: Sequence[CycleLifePrediction],
    *,
    split_manifest: SplitManifest,
    coverages: Sequence[float] = (0.8, 0.9),
    split_manifest_sha256: Sha256,
    prediction_evidence_sha256: Sha256,
) -> tuple[CycleLifeIntervalMaterial, ...]:
    calibration_cohort = tuple(calibration_predictions)
    test_cohort = tuple(test_predictions)
    requested = tuple(coverages)
    if not calibration_cohort or not test_cohort:
        raise ValueError("calibration and test predictions are both required")
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("target coverages must be unique")
    if any(not math.isfinite(value) or not 0 < value < 1 for value in requested):
        raise ValueError("target coverages must be finite and between zero and one")
    materials: list[CycleLifeIntervalMaterial] = []
    for target_coverage in sorted(requested):
        calibration = calibrate_cycle_life_conformal(
            calibration_cohort,
            split_manifest=split_manifest,
            alpha=1 - target_coverage,
        )
        intervals = tuple(
            make_cycle_life_interval(prediction, calibration) for prediction in test_cohort
        )
        coverage = evaluate_cycle_life_interval_coverage(
            intervals,
            split_manifest=split_manifest,
        )
        warnings = coverage.warnings
        payload = {
            "schema_version": "cycle-life-interval-material-v1",
            "target_coverage": target_coverage,
            "model_version": calibration.model_version,
            "dataset_id": calibration.dataset_id,
            "cutoff_cycle": calibration.cutoff_cycle,
            "target": calibration.target.value,
            "feature_version": calibration.feature_version,
            "split_version": calibration.split_version,
            "data_version": calibration.data_version,
            "split_manifest_sha256": split_manifest_sha256,
            "prediction_evidence_sha256": prediction_evidence_sha256,
            "calibration": calibration.model_dump(mode="json"),
            "intervals": [interval.model_dump(mode="json") for interval in intervals],
            "coverage": coverage.model_dump(mode="json"),
            "warnings": list(warnings),
        }
        materials.append(
            CycleLifeIntervalMaterial.model_validate(
                {**payload, "material_sha256": sha256_canonical(payload)}
            )
        )
    return tuple(materials)


__all__ = [
    "CycleLifeIntervalMaterial",
    "TrajectoryIntervalEvidence",
    "materialize_cycle_life_intervals",
]
