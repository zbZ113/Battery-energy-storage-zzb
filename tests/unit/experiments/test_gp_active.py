"""Focused contracts for low-dimensional GP experiment recommendation.

All fixtures are explicitly synthetic.  They prove numerical-service behavior,
not any result on a Naumann or industrial battery dataset.
"""

from __future__ import annotations

import math

import pytest
from sklearn.gaussian_process.kernels import Matern, Sum, WhiteKernel

from quanxin_life.experiments.gp_active import (
    AcquisitionConfig,
    ExperimentCandidate,
    ExperimentObservation,
    GaussianProcessExperimentRecommender,
    OperatingBounds,
    OperatingCondition,
)


def _bounds() -> OperatingBounds:
    return OperatingBounds(
        temperature_c=(10.0, 50.0),
        mean_soc=(0.1, 0.9),
        dod=(0.1, 0.9),
        charge_c_rate=(0.1, 2.0),
        discharge_c_rate=(0.1, 2.0),
    )


def _config(*, duplicate_penalty_weight: float = 0.0) -> AcquisitionConfig:
    return AcquisitionConfig(
        time_normalizer_hours=100.0,
        equipment_cost_normalizer=10.0,
        duplicate_penalty_weight=duplicate_penalty_weight,
        similarity_length_scale=1.0,
    )


def _condition(
    temperature_c: float,
    mean_soc: float,
    dod: float,
    charge_c_rate: float,
    discharge_c_rate: float,
) -> OperatingCondition:
    return OperatingCondition(
        temperature_c=temperature_c,
        mean_soc=mean_soc,
        dod=dod,
        charge_c_rate=charge_c_rate,
        discharge_c_rate=discharge_c_rate,
    )


def _observations() -> tuple[ExperimentObservation, ...]:
    return (
        ExperimentObservation(
            observation_id="synthetic-01",
            condition=_condition(15.0, 0.2, 0.2, 0.2, 0.2),
            target_name="capacity_loss_rate",
            observed_target=0.11,
            duration_hours=90.0,
            equipment_cost=4.0,
        ),
        ExperimentObservation(
            observation_id="synthetic-02",
            condition=_condition(25.0, 0.4, 0.4, 0.5, 0.5),
            target_name="capacity_loss_rate",
            observed_target=0.18,
            duration_hours=110.0,
            equipment_cost=5.0,
        ),
        ExperimentObservation(
            observation_id="synthetic-03",
            condition=_condition(35.0, 0.6, 0.6, 1.0, 1.0),
            target_name="capacity_loss_rate",
            observed_target=0.31,
            duration_hours=130.0,
            equipment_cost=6.0,
        ),
        ExperimentObservation(
            observation_id="synthetic-04",
            condition=_condition(45.0, 0.8, 0.8, 1.5, 1.5),
            target_name="capacity_loss_rate",
            observed_target=0.49,
            duration_hours=150.0,
            equipment_cost=7.0,
        ),
    )


def _candidate(
    candidate_id: str,
    condition: OperatingCondition,
    *,
    duration_hours: float = 100.0,
    equipment_cost: float = 5.0,
) -> ExperimentCandidate:
    return ExperimentCandidate(
        candidate_id=candidate_id,
        condition=condition,
        duration_hours=duration_hours,
        equipment_cost=equipment_cost,
    )


def _fitted_recommender(
    *, duplicate_penalty_weight: float = 0.0,
) -> GaussianProcessExperimentRecommender:
    return GaussianProcessExperimentRecommender(
        operating_bounds=_bounds(),
        acquisition_config=_config(duplicate_penalty_weight=duplicate_penalty_weight),
    ).fit(_observations())


def test_ranking_is_deterministic_and_breaks_equal_scores_by_candidate_id() -> None:
    recommender = _fitted_recommender()
    shared_condition = _condition(30.0, 0.5, 0.5, 0.8, 0.8)
    candidates = (
        _candidate("candidate-b", shared_condition),
        _candidate("candidate-a", shared_condition),
    )

    first = recommender.rank_candidates(candidates)
    second = recommender.rank_candidates(candidates)

    assert [assessment.candidate_id for assessment in first.accepted] == [
        "candidate-a",
        "candidate-b",
    ]
    assert first == second


def test_unsafe_candidate_is_rejected_without_clamping_or_prediction() -> None:
    recommender = _fitted_recommender()
    unsafe = _candidate("unsafe", _condition(55.0, 0.5, 0.5, 0.8, 0.8))

    ranking = recommender.rank_candidates((unsafe,))

    assert not ranking.accepted
    rejected = ranking.rejected[0]
    assert rejected.candidate_id == "unsafe"
    assert rejected.predicted_mean is None
    assert rejected.predicted_std is None
    assert rejected.acquisition_score is None
    assert rejected.rejection_reasons == ("OUT_OF_OPERATING_BOUNDS",)


def test_exact_duplicate_candidate_receives_explicit_penalty() -> None:
    recommender = _fitted_recommender(duplicate_penalty_weight=3.0)
    duplicate = _candidate("duplicate", _observations()[0].condition)

    assessment = recommender.rank_candidates((duplicate,)).accepted[0]

    assert assessment.duplicate_penalty == pytest.approx(3.0)
    assert assessment.normalized_cost is not None
    assert assessment.predicted_std is not None
    assert assessment.acquisition_score == pytest.approx(
        assessment.predicted_std / assessment.normalized_cost - assessment.duplicate_penalty
    )


def test_nonfinite_or_missing_input_is_rejected_without_silent_imputation() -> None:
    with pytest.raises(ValueError, match="temperature_c"):
        _condition(float("nan"), 0.5, 0.5, 0.8, 0.8)

    with pytest.raises(ValueError, match="mean_soc"):
        _condition(25.0, None, 0.5, 0.8, 0.8)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="observed_target"):
        ExperimentObservation(
            observation_id="invalid-target",
            condition=_condition(25.0, 0.5, 0.5, 0.8, 0.8),
            target_name="capacity_loss_rate",
            observed_target=float("inf"),
            duration_hours=100.0,
            equipment_cost=5.0,
        )


def test_model_uses_matern_five_halves_plus_white_kernel() -> None:
    recommender = _fitted_recommender()

    kernel = recommender.fitted_kernel

    assert isinstance(kernel, Sum)
    assert isinstance(kernel.k1, Matern)
    assert kernel.k1.nu == 2.5
    assert isinstance(kernel.k2, WhiteKernel)


def test_leave_one_out_replay_reports_observed_comparison_quantities() -> None:
    recommender = _fitted_recommender()

    replay = recommender.leave_one_out_replay()

    assert replay.target_name == "capacity_loss_rate"
    assert replay.fold_count == 4
    assert math.isfinite(replay.mae)
    assert math.isfinite(replay.rmse)
    assert [fold.held_out_observation_id for fold in replay.folds] == [
        "synthetic-01",
        "synthetic-02",
        "synthetic-03",
        "synthetic-04",
    ]
    assert all(math.isfinite(fold.actual_target) for fold in replay.folds)
    assert all(math.isfinite(fold.predicted_mean) for fold in replay.folds)
    assert all(math.isfinite(fold.predicted_std) for fold in replay.folds)
