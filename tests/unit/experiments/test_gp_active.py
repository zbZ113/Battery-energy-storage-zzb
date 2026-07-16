"""Focused contracts for low-dimensional GP experiment recommendation.

All fixtures are explicitly synthetic.  They prove numerical-service behavior,
not any result on a Naumann or industrial battery dataset.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from sklearn.gaussian_process.kernels import Matern, Sum, WhiteKernel

from quanxin_life.experiments.gp_active import (
    AcquisitionConfig,
    AcquisitionStrategy,
    ExperimentCandidate,
    ExperimentObservation,
    GaussianProcessExperimentRecommender,
    OperatingBounds,
    OperatingCondition,
    _stabilize_posterior_covariance,
    evaluate_finite_pool_replay,
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
            condition=_condition(45.0, 0.7, 0.4, 1.5, 1.5),
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
    safety_approved: bool | None = True,
    equipment_available: bool | None = True,
) -> ExperimentCandidate:
    return ExperimentCandidate(
        candidate_id=candidate_id,
        condition=condition,
        duration_hours=duration_hours,
        equipment_cost=equipment_cost,
        safety_approved=safety_approved,
        equipment_available=equipment_available,
    )


def _reference_conditions() -> tuple[OperatingCondition, ...]:
    """A fixed, explicit, synthetic reference space for EIVR tests."""

    return (
        *(observation.condition for observation in _observations()),
        _condition(20.0, 0.3, 0.3, 0.3, 0.3),
        _condition(40.0, 0.6, 0.5, 1.2, 1.2),
    )


def _replay_observations() -> tuple[ExperimentObservation, ...]:
    return (
        *_observations(),
        ExperimentObservation(
            observation_id="synthetic-05",
            condition=_condition(20.0, 0.7, 0.3, 0.3, 1.2),
            target_name="capacity_loss_rate",
            observed_target=0.23,
            duration_hours=105.0,
            equipment_cost=4.5,
        ),
        ExperimentObservation(
            observation_id="synthetic-06",
            condition=_condition(30.0, 0.4, 0.6, 1.2, 0.3),
            target_name="capacity_loss_rate",
            observed_target=0.27,
            duration_hours=120.0,
            equipment_cost=5.5,
        ),
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

    first = recommender.rank_candidates(
        candidates, reference_conditions=_reference_conditions()
    )
    second = recommender.rank_candidates(
        candidates, reference_conditions=_reference_conditions()
    )

    assert [assessment.candidate_id for assessment in first.accepted] == [
        "candidate-a",
        "candidate-b",
    ]
    assert first == second


def test_unsafe_candidate_is_rejected_without_clamping_or_prediction() -> None:
    recommender = _fitted_recommender()
    unsafe = _candidate("unsafe", _condition(55.0, 0.5, 0.5, 0.8, 0.8))

    ranking = recommender.rank_candidates(
        (unsafe,), reference_conditions=_reference_conditions()
    )

    assert not ranking.accepted
    rejected = ranking.rejected[0]
    assert rejected.candidate_id == "unsafe"
    assert rejected.predicted_mean is None
    assert rejected.predicted_std is None
    assert rejected.acquisition_score is None
    assert rejected.rejection_reasons == ("OUT_OF_OPERATING_BOUNDS",)


def test_missing_or_false_readiness_is_explicitly_rejected() -> None:
    recommender = _fitted_recommender()
    condition = _condition(30.0, 0.5, 0.5, 0.8, 0.8)
    missing = ExperimentCandidate(
        candidate_id="missing-readiness",
        condition=condition,
        duration_hours=100.0,
        equipment_cost=5.0,
    )
    explicitly_not_approved = _candidate(
        "not-approved",
        condition,
        safety_approved=False,
        equipment_available=False,
    )

    ranking = recommender.rank_candidates(
        (missing, explicitly_not_approved), reference_conditions=_reference_conditions()
    )

    by_id = {assessment.candidate_id: assessment for assessment in ranking.rejected}
    assert by_id["missing-readiness"].rejection_reasons == (
        "SAFETY_APPROVAL_MISSING",
        "EQUIPMENT_AVAILABILITY_MISSING",
    )
    assert by_id["not-approved"].rejection_reasons == (
        "SAFETY_NOT_APPROVED",
        "EQUIPMENT_UNAVAILABLE",
    )


def test_soc_and_dod_combination_must_describe_a_unit_interval() -> None:
    recommender = _fitted_recommender()
    unsafe_window = _candidate(
        "unsafe-window", _condition(25.0, 0.15, 0.5, 0.8, 0.8)
    )

    ranking = recommender.rank_candidates(
        (unsafe_window,), reference_conditions=_reference_conditions()
    )

    rejected = ranking.rejected[0]
    assert rejected.rejection_reasons == ("SOC_WINDOW_OUT_OF_RANGE",)
    assert rejected.constraint_details == ("SOC_LOWER_BELOW_ZERO",)


def test_exact_duplicate_observed_condition_is_rejected_even_without_penalty_weight() -> None:
    recommender = _fitted_recommender(duplicate_penalty_weight=0.0)
    duplicate = _candidate("duplicate", _observations()[0].condition)

    assessment = recommender.rank_candidates(
        (duplicate,), reference_conditions=_reference_conditions()
    ).rejected[0]

    assert assessment.rejection_reasons == ("DUPLICATE_OBSERVED_CONDITION",)


def test_cost_aware_eivr_keeps_auditable_variance_cost_and_similarity_components() -> None:
    recommender = _fitted_recommender(duplicate_penalty_weight=3.0)
    candidate = _candidate("candidate", _condition(30.0, 0.5, 0.5, 0.8, 0.8))

    assessment = recommender.rank_candidates(
        (candidate,), reference_conditions=_reference_conditions()
    ).accepted[0]

    assert assessment.normalized_cost is not None
    assert assessment.duplicate_penalty is not None
    assert assessment.acquisition_metadata is not None
    assert assessment.acquisition_metadata.strategy is AcquisitionStrategy.COST_AWARE_EIVR
    assert assessment.acquisition_metadata.expected_variance_reduction > 0
    assert assessment.acquisition_metadata.reference_total_variance_after == pytest.approx(
        assessment.acquisition_metadata.reference_total_variance_before
        - assessment.acquisition_metadata.expected_variance_reduction
    )
    assert assessment.acquisition_score == pytest.approx(
        assessment.acquisition_metadata.expected_variance_reduction
        / assessment.normalized_cost
        - assessment.duplicate_penalty
    )


def test_eivr_requires_an_explicit_fixed_reference_space() -> None:
    recommender = _fitted_recommender()

    with pytest.raises(ValueError, match="reference_conditions"):
        recommender.rank_candidates(
            (_candidate("candidate", _condition(30.0, 0.5, 0.5, 0.8, 0.8)),),
            reference_conditions=(),
        )


def test_eivr_accounts_for_observation_noise_and_reports_target_scale_variance() -> None:
    recommender = _fitted_recommender()
    candidate = _candidate("candidate", _condition(30.0, 0.5, 0.5, 0.8, 0.8))
    reference_space = _reference_conditions()

    assessment = recommender.rank_candidates(
        (candidate,), reference_conditions=reference_space
    ).accepted[0]
    metadata = assessment.acquisition_metadata
    assert metadata is not None
    candidate_matrix = recommender._conditions_to_scaled_matrix((candidate.condition,))
    reference_matrix = recommender._conditions_to_scaled_matrix(reference_space)
    latent_candidate_variance = recommender._posterior_covariance(
        candidate_matrix, candidate_matrix
    )[0, 0]
    reference_to_candidate = recommender._posterior_covariance(
        reference_matrix, candidate_matrix
    )[:, 0]
    expected_reduction = float(
        sum(
            covariance**2
            / (latent_candidate_variance + recommender._observation_noise_variance())
            for covariance in reference_to_candidate
        )
        * recommender._target_variance_scale()
    )

    assert metadata.expected_variance_reduction == pytest.approx(expected_reduction)


def test_max_variance_is_an_explicit_baseline_not_the_default_strategy() -> None:
    recommender = _fitted_recommender()
    candidate = _candidate("candidate", _condition(30.0, 0.5, 0.5, 0.8, 0.8))

    eivr = recommender.rank_candidates(
        (candidate,), reference_conditions=_reference_conditions()
    ).accepted[0]
    max_variance = recommender.rank_candidates(
        (candidate,), strategy=AcquisitionStrategy.MAX_VARIANCE
    ).accepted[0]

    assert eivr.acquisition_metadata is not None
    assert eivr.acquisition_metadata.strategy is AcquisitionStrategy.COST_AWARE_EIVR
    assert max_variance.acquisition_metadata is not None
    assert max_variance.acquisition_metadata.strategy is AcquisitionStrategy.MAX_VARIANCE
    assert max_variance.normalized_cost is None
    assert max_variance.duplicate_penalty is None
    assert max_variance.acquisition_score == pytest.approx(max_variance.predicted_std)


def test_eivr_uses_virtual_noise_conditioning_after_batch_selection() -> None:
    recommender = _fitted_recommender(duplicate_penalty_weight=0.0)
    selected_condition = _condition(20.0, 0.7, 0.3, 0.3, 1.2)
    candidate = _candidate("candidate", _condition(30.0, 0.4, 0.6, 1.2, 0.3))

    before_selection = recommender.rank_candidates(
        (candidate,), reference_conditions=_reference_conditions()
    ).accepted[0]
    after_virtual_selection = recommender.rank_candidates(
        (candidate,),
        reference_conditions=_reference_conditions(),
        _selected_conditions=(selected_condition,),
    ).accepted[0]

    assert before_selection.acquisition_metadata is not None
    assert after_virtual_selection.acquisition_metadata is not None
    assert (
        after_virtual_selection.acquisition_metadata.expected_variance_reduction
        != pytest.approx(before_selection.acquisition_metadata.expected_variance_reduction)
    )


def test_batch_recommendation_rejects_equivalent_conditions_after_selection() -> None:
    recommender = _fitted_recommender()
    same_condition = _condition(30.0, 0.5, 0.5, 0.8, 0.8)
    candidates = (
        _candidate("candidate-b", same_condition),
        _candidate("candidate-a", same_condition),
        _candidate("candidate-c", _condition(32.0, 0.55, 0.55, 0.9, 0.9)),
    )

    recommendation = recommender.recommend_batch(
        candidates,
        batch_size=2,
        reference_conditions=_reference_conditions(),
    )

    assert {assessment.candidate_id for assessment in recommendation.selected} == {
        "candidate-a",
        "candidate-c",
    }
    duplicate = next(
        assessment
        for assessment in recommendation.rejected
        if assessment.candidate_id == "candidate-b"
    )
    assert duplicate.rejection_reasons == ("DUPLICATE_SELECTED_CONDITION",)


def test_finite_pool_replay_compares_fixed_budget_strategies_without_winner_claim() -> None:
    replay = evaluate_finite_pool_replay(
        _replay_observations(),
        operating_bounds=_bounds(),
        acquisition_config=_config(),
        initial_observation_ids=("synthetic-01", "synthetic-02"),
        query_budget=2,
    )

    assert replay.query_budget == 2
    assert replay.winner is None
    assert [trajectory.strategy for trajectory in replay.trajectories] == [
        AcquisitionStrategy.RANDOM,
        AcquisitionStrategy.UNIFORM_GRID,
        AcquisitionStrategy.MAX_VARIANCE,
        AcquisitionStrategy.COST_AWARE_EIVR,
    ]
    for trajectory in replay.trajectories:
        assert len(trajectory.steps) == 2
        assert all(
            step.selected_observation_id not in {"synthetic-01", "synthetic-02"}
            for step in trajectory.steps
        )
        assert all(math.isfinite(step.absolute_error) for step in trajectory.steps)
        assert all(math.isfinite(step.predicted_std) for step in trajectory.steps)
        assert all(math.isfinite(step.cumulative_mae) for step in trajectory.steps)
        assert all(math.isfinite(step.cumulative_mean_predicted_std) for step in trajectory.steps)


def test_finite_pool_replay_rejects_duplicate_condition_vectors_with_observation_ids() -> None:
    duplicate = replace(
        _replay_observations()[0],
        observation_id="synthetic-duplicate-condition",
    )

    with pytest.raises(
        ValueError,
        match=r"synthetic-01.*synthetic-duplicate-condition",
    ):
        evaluate_finite_pool_replay(
            (*_replay_observations(), duplicate),
            operating_bounds=_bounds(),
            acquisition_config=_config(),
            initial_observation_ids=("synthetic-01", "synthetic-02"),
            query_budget=2,
        )


def test_replay_strategy_selection_is_invariant_to_unrevealed_target_values() -> None:
    initial_ids = {"synthetic-01", "synthetic-02"}
    baseline_observations = _replay_observations()
    relabelled_observations = tuple(
        observation
        if observation.observation_id in initial_ids
        else replace(observation, observed_target=observation.observed_target + 100.0)
        for observation in baseline_observations
    )

    baseline = evaluate_finite_pool_replay(
        baseline_observations,
        operating_bounds=_bounds(),
        acquisition_config=_config(),
        initial_observation_ids=tuple(sorted(initial_ids)),
        query_budget=2,
    )
    relabelled = evaluate_finite_pool_replay(
        relabelled_observations,
        operating_bounds=_bounds(),
        acquisition_config=_config(),
        initial_observation_ids=tuple(sorted(initial_ids)),
        query_budget=2,
    )

    assert [
        [step.selected_observation_id for step in trajectory.steps]
        for trajectory in baseline.trajectories
    ] == [
        [step.selected_observation_id for step in trajectory.steps]
        for trajectory in relabelled.trajectories
    ]


def test_posterior_covariance_stabilizer_symmetrizes_and_handles_roundoff_only() -> None:
    nearly_symmetric = np.asarray(
        [[1.0, 0.25 + 1e-13], [0.25, -1e-13]], dtype=float
    )

    stabilized = _stabilize_posterior_covariance(nearly_symmetric)

    assert np.array_equal(stabilized, stabilized.T)
    assert stabilized[1, 1] == 0.0
    with pytest.raises(RuntimeError, match="negative posterior variance"):
        _stabilize_posterior_covariance(np.asarray([[-1e-6]], dtype=float))


def test_near_duplicate_virtual_conditions_produce_stable_symmetric_covariance() -> None:
    recommender = _fitted_recommender()
    reference = _reference_conditions()
    near_one = _condition(30.0, 0.5, 0.5, 0.8, 0.8)
    near_two = _condition(30.0 + 1e-12, 0.5, 0.5, 0.8, 0.8)
    reference_matrix = recommender._conditions_to_scaled_matrix(reference)

    covariance = recommender._conditioned_posterior_covariance(
        reference_matrix,
        reference_matrix,
        (near_one, near_two),
    )

    assert np.array_equal(covariance, covariance.T)
    assert np.all(np.diag(covariance) >= 0.0)


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


def test_max_variance_can_rank_resource_unknown_historical_candidates() -> None:
    observations = tuple(
        replace(item, duration_hours=None, equipment_cost=None) for item in _observations()
    )
    recommender = GaussianProcessExperimentRecommender(
        operating_bounds=_bounds(),
        acquisition_config=_config(),
    ).fit(observations[:2])
    candidate = ExperimentCandidate(
        candidate_id=observations[2].observation_id,
        condition=observations[2].condition,
        duration_hours=None,
        equipment_cost=None,
        safety_approved=True,
        equipment_available=True,
    )

    result = recommender.rank_candidates(
        (candidate,),
        strategy=AcquisitionStrategy.MAX_VARIANCE,
    )

    assert [item.candidate_id for item in result.accepted] == [candidate.candidate_id]
    assert result.accepted[0].normalized_cost is None


def test_cost_aware_eivr_rejects_candidate_without_reviewed_resources() -> None:
    recommender = GaussianProcessExperimentRecommender(
        operating_bounds=_bounds(),
        acquisition_config=_config(),
    ).fit(_observations()[:2])
    candidate = ExperimentCandidate(
        candidate_id="missing-reviewed-resources",
        condition=_observations()[2].condition,
        duration_hours=None,
        equipment_cost=None,
        safety_approved=True,
        equipment_available=True,
    )

    with pytest.raises(ValueError, match="reviewed duration_hours and equipment_cost"):
        recommender.rank_candidates(
            (candidate,),
            reference_conditions=_reference_conditions(),
            strategy=AcquisitionStrategy.COST_AWARE_EIVR,
        )
