"""Contracts for frozen-global individual SOH trajectory calibration.

All data in this module are synthetic.  These tests verify that online
personalisation updates only bounded cell-specific parameters and never a
global model or an unregistered model artefact.
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest

from quanxin_life.core import SourceKind
from quanxin_life.online.individual_calibration import (
    CalibrationConfig,
    CalibrationStatus,
    FrozenGlobalTrajectory,
    IndividualTrajectoryCalibrator,
    NewlyObservedSOH,
    replay_individual_calibration,
)


def _global_trajectory() -> FrozenGlobalTrajectory:
    return FrozenGlobalTrajectory(
        dataset_id="synthetic-lfp",
        cell_id="cell-online-01",
        cutoff_cycle=20,
        cycles=(20, 50, 100, 150, 200),
        soh=(0.990, 0.960, 0.900, 0.830, 0.750),
        model_version="global-hybrid-v1",
        feature_version="early-cycle-v1",
        split_version="synthetic-split-v1",
        data_version="synthetic-data-v1",
    )


def _observations() -> tuple[NewlyObservedSOH, ...]:
    return (
        NewlyObservedSOH(cycle=20, soh=0.990, source_kind=SourceKind.NEWLY_OBSERVED),
        NewlyObservedSOH(cycle=50, soh=0.940, source_kind=SourceKind.NEWLY_OBSERVED),
        NewlyObservedSOH(cycle=100, soh=0.865, source_kind=SourceKind.NEWLY_OBSERVED),
        NewlyObservedSOH(cycle=150, soh=0.775, source_kind=SourceKind.NEWLY_OBSERVED),
        NewlyObservedSOH(cycle=200, soh=0.690, source_kind=SourceKind.NEWLY_OBSERVED),
    )


def test_calibration_preserves_frozen_global_and_returns_finite_monotone_trajectory() -> None:
    global_trajectory = _global_trajectory()
    global_before = global_trajectory.model_dump(mode="json")

    outcome = IndividualTrajectoryCalibrator().calibrate(
        global_trajectory=global_trajectory,
        observations=_observations()[:4],
        update_version="online-update-v1",
    )

    assert global_trajectory.model_dump(mode="json") == global_before
    assert outcome.status is CalibrationStatus.ADAPTED
    assert outcome.audit.prior_parameters.bias_soh == 0.0
    assert outcome.audit.prior_parameters.rate_multiplier == 1.0
    assert outcome.audit.prior_parameters.knee_offset_fraction == 0.0
    assert outcome.audit.observed_until_cycle == 150
    assert outcome.audit.update_version == "online-update-v1"
    assert math.isfinite(outcome.audit.objective_value)
    assert math.isfinite(outcome.audit.fit_rmse)
    assert all(math.isfinite(value) for value in outcome.adapted_soh)
    assert all(
        current <= previous for previous, current in pairwise(outcome.adapted_soh)
    )


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (
            (
                NewlyObservedSOH(cycle=50, soh=0.94),
                NewlyObservedSOH(cycle=50, soh=0.93),
                NewlyObservedSOH(cycle=100, soh=0.86),
            ),
            "duplicate",
        ),
        (
            (
                NewlyObservedSOH(cycle=10, soh=0.99),
                NewlyObservedSOH(cycle=50, soh=0.94),
                NewlyObservedSOH(cycle=100, soh=0.86),
            ),
            "cutoff",
        ),
        (
            (
                NewlyObservedSOH(cycle=50, soh=0.94),
                NewlyObservedSOH(cycle=100, soh=0.86),
                NewlyObservedSOH(cycle=250, soh=0.70),
            ),
            "horizon",
        ),
    ],
)
def test_calibration_rejects_duplicate_or_out_of_horizon_observations(
    observations: tuple[NewlyObservedSOH, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        IndividualTrajectoryCalibrator().calibrate(
            global_trajectory=_global_trajectory(),
            observations=observations,
            update_version="online-update-v1",
        )


def test_observation_rejects_nonfinite_or_non_new_observation_source() -> None:
    with pytest.raises(ValueError, match="finite"):
        NewlyObservedSOH(cycle=50, soh=float("nan"))

    with pytest.raises(ValueError, match="NEWLY_OBSERVED"):
        NewlyObservedSOH(cycle=50, soh=0.94, source_kind=SourceKind.PREDICTED)


def test_insufficient_observations_returns_explicit_recheck_without_adaptation() -> None:
    global_trajectory = _global_trajectory()

    outcome = IndividualTrajectoryCalibrator().calibrate(
        global_trajectory=global_trajectory,
        observations=_observations()[:2],
        update_version="online-update-v1",
    )

    assert outcome.status is CalibrationStatus.RECHECK
    assert outcome.reason_code == "INSUFFICIENT_OBSERVATIONS"
    assert outcome.adapted_soh == global_trajectory.soh
    assert outcome.audit.updated_parameters == outcome.audit.prior_parameters
    assert outcome.audit.objective_value is None
    assert outcome.audit.fit_rmse is None


def test_poor_fit_returns_recheck_and_does_not_apply_untrusted_parameters() -> None:
    impossible_observations = (
        NewlyObservedSOH(cycle=20, soh=0.20),
        NewlyObservedSOH(cycle=50, soh=0.18),
        NewlyObservedSOH(cycle=100, soh=0.15),
    )
    calibrator = IndividualTrajectoryCalibrator(
        config=CalibrationConfig(max_fit_rmse=0.01),
    )

    outcome = calibrator.calibrate(
        global_trajectory=_global_trajectory(),
        observations=impossible_observations,
        update_version="online-update-v1",
    )

    assert outcome.status is CalibrationStatus.RECHECK
    assert outcome.reason_code == "FIT_QUALITY_INSUFFICIENT"
    assert outcome.adapted_soh == _global_trajectory().soh
    assert outcome.audit.updated_parameters == outcome.audit.prior_parameters
    assert outcome.audit.fit_rmse is not None


def test_replay_is_deterministic_and_reports_measured_before_after_errors() -> None:
    first = replay_individual_calibration(
        global_trajectory=_global_trajectory(),
        observations=_observations(),
        update_cycles=(20, 50, 100, 150),
        update_version_prefix="online-replay-v1",
    )
    second = replay_individual_calibration(
        global_trajectory=_global_trajectory(),
        observations=_observations(),
        update_cycles=(20, 50, 100, 150),
        update_version_prefix="online-replay-v1",
    )

    assert first == second
    assert [step.update_cycle for step in first.steps] == [20, 50, 100, 150]
    assert all(step.withheld_observation_count >= 1 for step in first.steps)
    assert all(step.before_mae is not None for step in first.steps)
    assert all(step.before_rmse is not None for step in first.steps)
    assert all(step.after_mae is not None for step in first.steps)
    assert all(step.after_rmse is not None for step in first.steps)
    assert first.steps[0].outcome.status is CalibrationStatus.RECHECK
    assert first.steps[2].outcome.status is CalibrationStatus.ADAPTED
