"""Low-dimensional Gaussian-process experiment recommendation.

This module is a deterministic numerical service, not an Agent or a dataset
loader.  Callers must supply provenance-reviewed, explicitly observed labels.
It neither invents Naumann measurements nor loads serialized model artifacts.
The fitted scikit-learn estimator stays in memory only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from statistics import fmean
from typing import Self

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import (  # type: ignore[import-untyped]
    Kernel,
    Matern,
    Sum,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

GP_RANDOM_STATE = 20260712
GP_MODEL_VERSION = "naumann-condition-gp-v2"
POSTERIOR_VARIANCE_ROUNDOFF_TOLERANCE = 1e-10
CONDITION_FEATURE_NAMES = (
    "temperature_c",
    "mean_soc",
    "dod",
    "charge_c_rate",
    "discharge_c_rate",
)


def _require_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite real number")
    return numeric_value


class AcquisitionStrategy(StrEnum):
    """Policies used in ranking or finite-pool replay.

    ``COST_AWARE_EIVR`` is the primary recommendation policy.  ``MAX_VARIANCE``
    remains an explicit baseline; it must not be presented as the primary
    cost-aware policy.
    """

    RANDOM = "random"
    UNIFORM_GRID = "uniform_grid"
    MAX_VARIANCE = "max_variance"
    COST_AWARE_EIVR = "cost_aware_eivr"


@dataclass(frozen=True)
class OperatingCondition:
    """One explicit low-dimensional ageing experiment condition."""

    temperature_c: float
    mean_soc: float
    dod: float
    charge_c_rate: float
    discharge_c_rate: float

    def __post_init__(self) -> None:
        for name in CONDITION_FEATURE_NAMES:
            _require_finite_real(getattr(self, name), name=name)

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        """Return the fixed, documented condition feature order."""

        return (
            float(self.temperature_c),
            float(self.mean_soc),
            float(self.dod),
            float(self.charge_c_rate),
            float(self.discharge_c_rate),
        )


@dataclass(frozen=True)
class OperatingBounds:
    """Reviewed safety and equipment limits; candidates are never clamped."""

    temperature_c: tuple[float, float]
    mean_soc: tuple[float, float]
    dod: tuple[float, float]
    charge_c_rate: tuple[float, float]
    discharge_c_rate: tuple[float, float]

    def __post_init__(self) -> None:
        for name in CONDITION_FEATURE_NAMES:
            bounds = getattr(self, name)
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"{name} bounds must be a two-item tuple")
            lower = _require_finite_real(bounds[0], name=f"{name} lower bound")
            upper = _require_finite_real(bounds[1], name=f"{name} upper bound")
            if lower >= upper:
                raise ValueError(f"{name} lower bound must be below upper bound")

    def violations(self, condition: OperatingCondition) -> tuple[str, ...]:
        """Return named bound violations without changing the supplied condition."""

        violations: list[str] = []
        for name in CONDITION_FEATURE_NAMES:
            lower, upper = getattr(self, name)
            value = getattr(condition, name)
            if value < lower or value > upper:
                violations.append(name)
        return tuple(violations) + self.soc_window_violations(condition)

    def soc_window_violations(self, condition: OperatingCondition) -> tuple[str, ...]:
        """Validate the physical SOC interval implied by mean SOC and DOD."""

        soc_lower = condition.mean_soc - condition.dod / 2.0
        soc_upper = condition.mean_soc + condition.dod / 2.0
        violations: list[str] = []
        if soc_lower < 0.0:
            violations.append("SOC_LOWER_BELOW_ZERO")
        if soc_upper > 1.0:
            violations.append("SOC_UPPER_ABOVE_ONE")
        return tuple(violations)

    def normalized_vector(self, condition: OperatingCondition) -> tuple[float, ...]:
        """Map a verified in-bound condition to the unit hyper-rectangle."""

        violations = self.violations(condition)
        if violations:
            raise ValueError("condition violates operating bounds: " + ", ".join(violations))
        return tuple(
            (getattr(condition, name) - lower) / (upper - lower)
            for name in CONDITION_FEATURE_NAMES
            for lower, upper in (getattr(self, name),)
        )


@dataclass(frozen=True)
class ExperimentObservation:
    """An actual observed numerical target for one completed experiment."""

    observation_id: str
    condition: OperatingCondition
    target_name: str
    observed_target: float
    duration_hours: float
    equipment_cost: float

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValueError("observation_id must be non-empty")
        if not self.target_name.strip():
            raise ValueError("target_name must be non-empty")
        _require_finite_real(self.observed_target, name="observed_target")
        if _require_finite_real(self.duration_hours, name="duration_hours") <= 0:
            raise ValueError("duration_hours must be positive")
        if _require_finite_real(self.equipment_cost, name="equipment_cost") <= 0:
            raise ValueError("equipment_cost must be positive")


@dataclass(frozen=True)
class ExperimentCandidate:
    """A proposed trial with explicit resource and readiness declarations."""

    candidate_id: str
    condition: OperatingCondition
    duration_hours: float
    equipment_cost: float
    safety_approved: bool | None = None
    equipment_available: bool | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if _require_finite_real(self.duration_hours, name="duration_hours") <= 0:
            raise ValueError("duration_hours must be positive")
        if _require_finite_real(self.equipment_cost, name="equipment_cost") <= 0:
            raise ValueError("equipment_cost must be positive")
        if self.safety_approved is not None and not isinstance(self.safety_approved, bool):
            raise ValueError("safety_approved must be boolean or None")
        if self.equipment_available is not None and not isinstance(self.equipment_available, bool):
            raise ValueError("equipment_available must be boolean or None")


@dataclass(frozen=True)
class AcquisitionConfig:
    """Explicitly normalized cost-aware uncertainty-acquisition settings."""

    time_normalizer_hours: float
    equipment_cost_normalizer: float
    duplicate_penalty_weight: float
    similarity_length_scale: float

    def __post_init__(self) -> None:
        for name in (
            "time_normalizer_hours",
            "equipment_cost_normalizer",
            "similarity_length_scale",
        ):
            if _require_finite_real(getattr(self, name), name=name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _require_finite_real(
            self.duplicate_penalty_weight, name="duplicate_penalty_weight"
        ) < 0:
            raise ValueError("duplicate_penalty_weight must be non-negative")


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Auditable numerical components of a candidate acquisition score."""

    strategy: AcquisitionStrategy
    reference_condition_count: int
    reference_total_variance_before: float | None
    reference_total_variance_after: float | None
    expected_variance_reduction: float | None


@dataclass(frozen=True)
class CandidateAssessment:
    """One accepted score or one explicit candidate rejection."""

    candidate_id: str
    accepted: bool
    predicted_mean: float | None
    predicted_std: float | None
    normalized_cost: float | None
    duplicate_penalty: float | None
    acquisition_score: float | None
    rejection_reasons: tuple[str, ...] = ()
    constraint_details: tuple[str, ...] = ()
    acquisition_metadata: AcquisitionMetadata | None = None


@dataclass(frozen=True)
class ExperimentRecommendation:
    """Deterministically ordered accepted and rejected candidate assessments."""

    target_name: str
    model_version: str
    strategy: AcquisitionStrategy
    accepted: tuple[CandidateAssessment, ...]
    rejected: tuple[CandidateAssessment, ...]


@dataclass(frozen=True)
class BatchExperimentRecommendation:
    """A batch where equivalent selected conditions are explicitly rejected."""

    target_name: str
    model_version: str
    strategy: AcquisitionStrategy
    selected: tuple[CandidateAssessment, ...]
    rejected: tuple[CandidateAssessment, ...]


@dataclass(frozen=True)
class ReplayFold:
    """A held-out observed target and the GP prediction made without it."""

    held_out_observation_id: str
    actual_target: float
    predicted_mean: float
    predicted_std: float
    absolute_error: float
    training_observation_count: int


@dataclass(frozen=True)
class LeaveOneOutReplay:
    """Transparent held-out prediction quantities, not a superiority claim."""

    target_name: str
    model_version: str
    folds: tuple[ReplayFold, ...]
    mae: float
    rmse: float

    @property
    def fold_count(self) -> int:
        return len(self.folds)


@dataclass(frozen=True)
class ReplayStep:
    """One sequential reveal from a finite pool of actually labelled trials."""

    step_index: int
    selected_observation_id: str
    actual_target: float
    predicted_mean: float
    predicted_std: float
    absolute_error: float
    cumulative_mae: float
    cumulative_rmse: float
    cumulative_mean_predicted_std: float
    training_observation_count: int


@dataclass(frozen=True)
class ReplayTrajectory:
    """Measured error and uncertainty path for one fixed-budget strategy."""

    strategy: AcquisitionStrategy
    steps: tuple[ReplayStep, ...]


@dataclass(frozen=True)
class FinitePoolReplay:
    """Four-strategy finite-pool replay with no asserted winner."""

    target_name: str
    model_version: str
    query_budget: int
    initial_observation_ids: tuple[str, ...]
    reference_condition_count: int
    trajectories: tuple[ReplayTrajectory, ...]
    winner: None = None


@dataclass
class GaussianProcessExperimentRecommender:
    """Fit a bounded Matérn-5/2 + WhiteKernel GP on actual observations."""

    operating_bounds: OperatingBounds
    acquisition_config: AcquisitionConfig
    random_state: int = GP_RANDOM_STATE
    _scaler: StandardScaler | None = field(default=None, init=False, repr=False)
    _regressor: GaussianProcessRegressor | None = field(default=None, init=False, repr=False)
    _observations: tuple[ExperimentObservation, ...] = field(
        default=(), init=False, repr=False
    )
    _target_name: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ValueError("random_state must be an integer")

    @property
    def target_name(self) -> str:
        if self._target_name is None:
            raise RuntimeError("GaussianProcessExperimentRecommender must be fitted before use")
        return self._target_name

    @property
    def fitted_kernel(self) -> Kernel:
        if self._regressor is None:
            raise RuntimeError("GaussianProcessExperimentRecommender must be fitted before use")
        return self._regressor.kernel_

    def fit(self, observations: Sequence[ExperimentObservation]) -> Self:
        """Fit only real, finite, reviewed observations in declared bounds."""

        cohort = tuple(sorted(observations, key=lambda observation: observation.observation_id))
        if len(cohort) < 2:
            raise ValueError("at least two observed experiments are required to fit a GP")

        seen_observation_ids: set[str] = set()
        target_name = cohort[0].target_name
        for observation in cohort:
            if observation.observation_id in seen_observation_ids:
                raise ValueError(f"duplicate observation_id: {observation.observation_id}")
            seen_observation_ids.add(observation.observation_id)
            if observation.target_name != target_name:
                raise ValueError("all observations must use the same target_name")
            violations = self.operating_bounds.violations(observation.condition)
            if violations:
                raise ValueError(
                    "observed condition violates operating bounds: " + ", ".join(violations)
                )

        feature_matrix = np.asarray(
            [observation.condition.as_tuple() for observation in cohort], dtype=float
        )
        target_vector = np.asarray(
            [observation.observed_target for observation in cohort], dtype=float
        )
        if not np.isfinite(feature_matrix).all() or not np.isfinite(target_vector).all():
            raise ValueError("GP observations must be finite; imputation is not supported")

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)
        # The kernel parameters intentionally remain fixed.  Tiny experimental
        # cohorts do not justify silently optimized boundary hyperparameters.
        regressor = GaussianProcessRegressor(
            kernel=Matern(nu=2.5) + WhiteKernel(),
            alpha=0.0,
            normalize_y=True,
            random_state=self.random_state,
            n_restarts_optimizer=0,
            optimizer=None,
        )
        regressor.fit(normalized_features, target_vector)
        self._scaler = scaler
        self._regressor = regressor
        self._observations = cohort
        self._target_name = target_name
        return self

    def predict_condition(self, condition: OperatingCondition) -> tuple[float, float]:
        """Return GP mean and standard deviation for one in-bound condition."""

        self.operating_bounds.normalized_vector(condition)
        regressor, scaler = self._require_fitted()
        features = np.asarray([condition.as_tuple()], dtype=float)
        transformed = scaler.transform(features)
        means, standard_deviations = regressor.predict(transformed, return_std=True)
        mean = float(means[0])
        standard_deviation = float(standard_deviations[0])
        if not math.isfinite(mean) or not math.isfinite(standard_deviation):
            raise RuntimeError("Gaussian process returned a non-finite prediction")
        if standard_deviation < 0:
            raise RuntimeError("Gaussian process returned a negative standard deviation")
        return mean, standard_deviation

    def rank_candidates(
        self,
        candidates: Sequence[ExperimentCandidate],
        *,
        reference_conditions: Sequence[OperatingCondition] | None = None,
        strategy: AcquisitionStrategy = AcquisitionStrategy.COST_AWARE_EIVR,
        _selected_conditions: Sequence[OperatingCondition] = (),
    ) -> ExperimentRecommendation:
        """Rank safe candidates with auditable EIVR or an explicit baseline.

        EIVR is evaluated against the caller-supplied, fixed reference space.
        It analytically updates the GP posterior covariance after a hypothetical
        candidate observation, so it never fabricates a target outcome.
        """

        self._require_fitted()
        self._validate_strategy(strategy)
        reference_space = self._prepare_reference_space(reference_conditions, strategy)
        reference_covariance: np.ndarray | None = None
        reference_variance_before: float | None = None
        if reference_space:
            reference_covariance = self._conditioned_posterior_covariance(
                self._conditions_to_scaled_matrix(reference_space),
                self._conditions_to_scaled_matrix(reference_space),
                _selected_conditions,
            )
            reference_variance_before = float(
                np.trace(reference_covariance) * self._target_variance_scale()
            )
            if not math.isfinite(reference_variance_before) or reference_variance_before < 0:
                raise RuntimeError("reference posterior variance must be finite and non-negative")

        selected_vectors = {condition.as_tuple() for condition in _selected_conditions}
        observed_vectors = {
            observation.condition.as_tuple() for observation in self._observations
        }
        seen_candidate_ids: set[str] = set()
        accepted: list[CandidateAssessment] = []
        rejected: list[CandidateAssessment] = []
        for candidate in candidates:
            if candidate.candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
            seen_candidate_ids.add(candidate.candidate_id)

            rejection_reasons, constraint_details = self._candidate_rejection(
                candidate, selected_vectors, observed_vectors
            )
            if rejection_reasons:
                rejected.append(
                    self._rejected_assessment(candidate, rejection_reasons, constraint_details)
                )
                continue

            predicted_mean, predicted_std = self.predict_condition(candidate.condition)
            metadata = self._acquisition_metadata(
                candidate.condition,
                strategy=strategy,
                reference_space=reference_space,
                reference_covariance=reference_covariance,
                reference_variance_before=reference_variance_before,
                virtual_conditions=_selected_conditions,
            )
            if strategy is AcquisitionStrategy.COST_AWARE_EIVR:
                normalized_cost = self._normalized_cost(candidate)
                duplicate_penalty = self._duplicate_penalty(
                    candidate.condition, additional_conditions=_selected_conditions
                )
                signal = metadata.expected_variance_reduction
                if signal is None:
                    raise RuntimeError("EIVR strategy did not produce a variance-reduction signal")
                acquisition_score = signal / normalized_cost - duplicate_penalty
            else:
                normalized_cost = None
                duplicate_penalty = None
                acquisition_score = predicted_std
            if not math.isfinite(acquisition_score):
                raise RuntimeError("candidate acquisition score must be finite")
            accepted.append(
                CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    accepted=True,
                    predicted_mean=predicted_mean,
                    predicted_std=predicted_std,
                    normalized_cost=normalized_cost,
                    duplicate_penalty=duplicate_penalty,
                    acquisition_score=acquisition_score,
                    acquisition_metadata=metadata,
                )
            )

        return ExperimentRecommendation(
            target_name=self.target_name,
            model_version=GP_MODEL_VERSION,
            strategy=strategy,
            accepted=tuple(
                sorted(
                    accepted,
                    key=lambda assessment: (
                        -_required_score(assessment.acquisition_score),
                        assessment.candidate_id,
                    ),
                )
            ),
            rejected=tuple(sorted(rejected, key=lambda assessment: assessment.candidate_id)),
        )

    def recommend_batch(
        self,
        candidates: Sequence[ExperimentCandidate],
        *,
        batch_size: int,
        reference_conditions: Sequence[OperatingCondition],
        strategy: AcquisitionStrategy = AcquisitionStrategy.COST_AWARE_EIVR,
    ) -> BatchExperimentRecommendation:
        """Select a batch without allowing equivalent selected conditions twice."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._validate_strategy(strategy)
        reference_space = self._prepare_reference_space(reference_conditions, strategy)
        pending = tuple(candidates)
        seen_candidate_ids: set[str] = set()
        for candidate in pending:
            if candidate.candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
            seen_candidate_ids.add(candidate.candidate_id)

        selected_conditions: list[OperatingCondition] = []
        selected: list[CandidateAssessment] = []
        rejected: list[CandidateAssessment] = []
        while pending and len(selected) < batch_size:
            ranking = self.rank_candidates(
                pending,
                reference_conditions=reference_space,
                strategy=strategy,
                _selected_conditions=selected_conditions,
            )
            rejected.extend(ranking.rejected)
            rejected_ids = {assessment.candidate_id for assessment in ranking.rejected}
            available = tuple(
                candidate
                for candidate in pending
                if candidate.candidate_id not in rejected_ids
            )
            if not ranking.accepted:
                break
            chosen = ranking.accepted[0]
            selected.append(chosen)
            chosen_candidate = next(
                candidate
                for candidate in available
                if candidate.candidate_id == chosen.candidate_id
            )
            selected_conditions.append(chosen_candidate.condition)
            pending = tuple(
                candidate
                for candidate in available
                if candidate.candidate_id != chosen.candidate_id
            )

        if pending and selected_conditions:
            final_ranking = self.rank_candidates(
                pending,
                reference_conditions=reference_space,
                strategy=strategy,
                _selected_conditions=selected_conditions,
            )
            rejected.extend(final_ranking.rejected)

        return BatchExperimentRecommendation(
            target_name=self.target_name,
            model_version=GP_MODEL_VERSION,
            strategy=strategy,
            selected=tuple(selected),
            rejected=tuple(sorted(_deduplicate_assessments(rejected), key=_assessment_sort_key)),
        )

    def leave_one_out_replay(self) -> LeaveOneOutReplay:
        """Re-fit per held-out observation and report actual prediction errors."""

        self._require_fitted()
        if len(self._observations) < 3:
            raise ValueError("leave-one-out replay requires at least three observations")

        folds: list[ReplayFold] = []
        for held_out in self._observations:
            training = tuple(
                observation
                for observation in self._observations
                if observation.observation_id != held_out.observation_id
            )
            replay_model = GaussianProcessExperimentRecommender(
                operating_bounds=self.operating_bounds,
                acquisition_config=self.acquisition_config,
                random_state=self.random_state,
            ).fit(training)
            predicted_mean, predicted_std = replay_model.predict_condition(held_out.condition)
            folds.append(
                ReplayFold(
                    held_out_observation_id=held_out.observation_id,
                    actual_target=float(held_out.observed_target),
                    predicted_mean=predicted_mean,
                    predicted_std=predicted_std,
                    absolute_error=abs(predicted_mean - held_out.observed_target),
                    training_observation_count=len(training),
                )
            )

        absolute_errors = [fold.absolute_error for fold in folds]
        squared_errors = [error * error for error in absolute_errors]
        return LeaveOneOutReplay(
            target_name=self.target_name,
            model_version=GP_MODEL_VERSION,
            folds=tuple(folds),
            mae=fmean(absolute_errors),
            rmse=math.sqrt(fmean(squared_errors)),
        )

    def _require_fitted(self) -> tuple[GaussianProcessRegressor, StandardScaler]:
        if self._regressor is None or self._scaler is None:
            raise RuntimeError("GaussianProcessExperimentRecommender must be fitted before use")
        return self._regressor, self._scaler

    def _prepare_reference_space(
        self,
        reference_conditions: Sequence[OperatingCondition] | None,
        strategy: AcquisitionStrategy,
    ) -> tuple[OperatingCondition, ...]:
        if strategy is AcquisitionStrategy.COST_AWARE_EIVR and not reference_conditions:
            raise ValueError(
                "reference_conditions must be an explicit, non-empty fixed reference space"
            )
        if reference_conditions is None:
            return ()
        reference_space = tuple(reference_conditions)
        seen_vectors: set[tuple[float, float, float, float, float]] = set()
        for condition in reference_space:
            self.operating_bounds.normalized_vector(condition)
            vector = condition.as_tuple()
            if vector in seen_vectors:
                raise ValueError(
                    "reference_conditions must not contain equivalent condition vectors"
                )
            seen_vectors.add(vector)
        return reference_space

    def _conditions_to_scaled_matrix(
        self, conditions: Sequence[OperatingCondition]
    ) -> np.ndarray:
        _, scaler = self._require_fitted()
        values = np.asarray([condition.as_tuple() for condition in conditions], dtype=float)
        return np.asarray(scaler.transform(values), dtype=float)

    def _posterior_covariance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        regressor, _ = self._require_fitted()
        kernel = regressor.kernel_
        prior = np.asarray(kernel(left, right), dtype=float)
        train_to_left = np.asarray(kernel(regressor.X_train_, left), dtype=float)
        train_to_right = np.asarray(kernel(regressor.X_train_, right), dtype=float)
        left_solution = np.linalg.solve(regressor.L_, train_to_left)
        right_solution = np.linalg.solve(regressor.L_, train_to_right)
        posterior = np.asarray(prior - left_solution.T @ right_solution, dtype=float)
        if not np.isfinite(posterior).all():
            raise RuntimeError("Gaussian-process posterior covariance is non-finite")
        return _stabilize_if_same_condition_space(posterior, left, right)

    def _conditioned_posterior_covariance(
        self,
        left: np.ndarray,
        right: np.ndarray,
        virtual_conditions: Sequence[OperatingCondition],
    ) -> np.ndarray:
        """Condition latent covariance on outcome-free virtual observations.

        The update uses only selected operating conditions plus the GP's
        observation-noise variance.  No selected candidate target is read or
        fabricated, so batch EIVR remains outcome-independent.
        """

        base_covariance = self._posterior_covariance(left, right)
        if not virtual_conditions:
            return base_covariance
        virtual = self._conditions_to_scaled_matrix(virtual_conditions)
        left_to_virtual = self._posterior_covariance(left, virtual)
        virtual_to_right = self._posterior_covariance(virtual, right)
        virtual_covariance = self._posterior_covariance(virtual, virtual)
        virtual_observation_covariance = virtual_covariance + (
            self._observation_noise_variance() * np.eye(len(virtual_conditions))
        )
        try:
            conditional_term = np.linalg.solve(
                virtual_observation_covariance, virtual_to_right
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError("virtual observation covariance is not solvable") from error
        conditioned = np.asarray(
            base_covariance - left_to_virtual @ conditional_term,
            dtype=float,
        )
        if not np.isfinite(conditioned).all():
            raise RuntimeError("conditioned GP posterior covariance is non-finite")
        return _stabilize_if_same_condition_space(conditioned, left, right)

    def _acquisition_metadata(
        self,
        condition: OperatingCondition,
        *,
        strategy: AcquisitionStrategy,
        reference_space: Sequence[OperatingCondition],
        reference_covariance: np.ndarray | None,
        reference_variance_before: float | None,
        virtual_conditions: Sequence[OperatingCondition],
    ) -> AcquisitionMetadata:
        if strategy is AcquisitionStrategy.MAX_VARIANCE:
            return AcquisitionMetadata(
                strategy=strategy,
                reference_condition_count=0,
                reference_total_variance_before=None,
                reference_total_variance_after=None,
                expected_variance_reduction=None,
            )
        if strategy is not AcquisitionStrategy.COST_AWARE_EIVR:
            raise ValueError(f"unsupported ranking strategy: {strategy.value}")
        if reference_covariance is None or reference_variance_before is None:
            raise RuntimeError("EIVR requires a reference covariance matrix")

        candidate = self._conditions_to_scaled_matrix((condition,))
        reference = self._conditions_to_scaled_matrix(reference_space)
        candidate_covariance = self._conditioned_posterior_covariance(
            candidate, candidate, virtual_conditions
        )
        candidate_observation_variance = float(
            candidate_covariance[0, 0] + self._observation_noise_variance()
        )
        if (
            not math.isfinite(candidate_observation_variance)
            or candidate_observation_variance <= 0
        ):
            raise RuntimeError("candidate posterior variance must be positive and finite")
        reference_to_candidate = self._conditioned_posterior_covariance(
            reference, candidate, virtual_conditions
        )[:, 0]
        variance_reduction = float(
            np.sum(
                np.square(reference_to_candidate) / candidate_observation_variance,
                dtype=float,
            )
        )
        variance_reduction *= self._target_variance_scale()
        if not math.isfinite(variance_reduction) or variance_reduction < 0:
            raise RuntimeError("expected variance reduction must be finite and non-negative")
        variance_after = reference_variance_before - variance_reduction
        if not math.isfinite(variance_after) or variance_after < 0:
            raise RuntimeError("EIVR posterior reference variance must be finite and non-negative")
        return AcquisitionMetadata(
            strategy=strategy,
            reference_condition_count=len(reference_space),
            reference_total_variance_before=reference_variance_before,
            reference_total_variance_after=variance_after,
            expected_variance_reduction=variance_reduction,
        )

    def _observation_noise_variance(self) -> float:
        regressor, _ = self._require_fitted()
        kernel = regressor.kernel_
        if not isinstance(kernel, Sum):
            raise RuntimeError("configured GP kernel must be a Matern plus WhiteKernel sum")
        white_kernel = kernel.k1 if isinstance(kernel.k1, WhiteKernel) else kernel.k2
        if not isinstance(white_kernel, WhiteKernel):
            raise RuntimeError("configured GP kernel must include a WhiteKernel")
        noise_variance = float(white_kernel.noise_level)
        if not math.isfinite(noise_variance) or noise_variance <= 0:
            raise RuntimeError("configured GP WhiteKernel noise must be positive and finite")
        return noise_variance

    def _target_variance_scale(self) -> float:
        regressor, _ = self._require_fitted()
        target_standard_deviation = float(np.asarray(regressor._y_train_std).reshape(-1)[0])
        target_variance_scale = target_standard_deviation * target_standard_deviation
        if not math.isfinite(target_variance_scale) or target_variance_scale <= 0:
            raise RuntimeError("GP target variance scale must be positive and finite")
        return target_variance_scale

    def _candidate_rejection(
        self,
        candidate: ExperimentCandidate,
        selected_vectors: set[tuple[float, float, float, float, float]],
        observed_vectors: set[tuple[float, float, float, float, float]],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        reasons: list[str] = []
        details: list[str] = []
        field_violations = tuple(
            name
            for name in CONDITION_FEATURE_NAMES
            if name in self.operating_bounds.violations(candidate.condition)
        )
        if field_violations:
            reasons.append("OUT_OF_OPERATING_BOUNDS")
            details.extend(field_violations)
        soc_window_violations = self.operating_bounds.soc_window_violations(candidate.condition)
        if soc_window_violations:
            reasons.append("SOC_WINDOW_OUT_OF_RANGE")
            details.extend(soc_window_violations)
        if candidate.safety_approved is None:
            reasons.append("SAFETY_APPROVAL_MISSING")
        elif not candidate.safety_approved:
            reasons.append("SAFETY_NOT_APPROVED")
        if candidate.equipment_available is None:
            reasons.append("EQUIPMENT_AVAILABILITY_MISSING")
        elif not candidate.equipment_available:
            reasons.append("EQUIPMENT_UNAVAILABLE")
        if candidate.condition.as_tuple() in observed_vectors:
            reasons.append("DUPLICATE_OBSERVED_CONDITION")
        if candidate.condition.as_tuple() in selected_vectors:
            reasons.append("DUPLICATE_SELECTED_CONDITION")
        return tuple(reasons), tuple(details)

    def _rejected_assessment(
        self,
        candidate: ExperimentCandidate,
        rejection_reasons: tuple[str, ...],
        constraint_details: tuple[str, ...],
    ) -> CandidateAssessment:
        return CandidateAssessment(
            candidate_id=candidate.candidate_id,
            accepted=False,
            predicted_mean=None,
            predicted_std=None,
            normalized_cost=None,
            duplicate_penalty=None,
            acquisition_score=None,
            rejection_reasons=rejection_reasons,
            constraint_details=constraint_details,
        )

    def _normalized_cost(self, candidate: ExperimentCandidate) -> float:
        normalized_cost = (
            candidate.duration_hours / self.acquisition_config.time_normalizer_hours
            + candidate.equipment_cost / self.acquisition_config.equipment_cost_normalizer
        )
        if not math.isfinite(normalized_cost) or normalized_cost <= 0:
            raise RuntimeError("validated candidate cost must normalize to a positive finite value")
        return normalized_cost

    def _duplicate_penalty(
        self,
        condition: OperatingCondition,
        *,
        additional_conditions: Sequence[OperatingCondition] = (),
    ) -> float:
        candidate_vector = self.operating_bounds.normalized_vector(condition)
        similarities = []
        comparison_conditions = (
            tuple(observation.condition for observation in self._observations)
            + tuple(additional_conditions)
        )
        for observed_condition in comparison_conditions:
            observed_vector = self.operating_bounds.normalized_vector(observed_condition)
            squared_distance = sum(
                (candidate_value - observed_value) ** 2
                for candidate_value, observed_value in zip(
                    candidate_vector, observed_vector, strict=True
                )
            )
            similarities.append(
                math.exp(
                    -0.5
                    * squared_distance
                    / (self.acquisition_config.similarity_length_scale**2)
                )
            )
        nearest_similarity = max(similarities, default=0.0)
        return self.acquisition_config.duplicate_penalty_weight * nearest_similarity

    @staticmethod
    def _validate_strategy(strategy: AcquisitionStrategy) -> None:
        if strategy not in (
            AcquisitionStrategy.MAX_VARIANCE,
            AcquisitionStrategy.COST_AWARE_EIVR,
        ):
            raise ValueError("ranking supports MAX_VARIANCE or COST_AWARE_EIVR only")


def evaluate_finite_pool_replay(
    observations: Sequence[ExperimentObservation],
    *,
    operating_bounds: OperatingBounds,
    acquisition_config: AcquisitionConfig,
    initial_observation_ids: Sequence[str],
    query_budget: int,
) -> FinitePoolReplay:
    """Compare four deterministic strategies using only revealed pool labels.

    Every selected condition is predicted before its supplied, actual target is
    revealed.  The return type records trajectories and deliberately has no
    winner field beyond ``None``: measured trajectories are not a claim that a
    strategy is universally superior.
    """

    cohort = tuple(sorted(observations, key=lambda observation: observation.observation_id))
    if isinstance(query_budget, bool) or not isinstance(query_budget, int) or query_budget <= 0:
        raise ValueError("query_budget must be a positive integer")
    _validate_replay_cohort(cohort, operating_bounds)
    initial_ids = tuple(initial_observation_ids)
    if len(set(initial_ids)) != len(initial_ids):
        raise ValueError("initial_observation_ids must be unique")
    by_id = {observation.observation_id: observation for observation in cohort}
    unknown_ids = sorted(set(initial_ids).difference(by_id))
    if unknown_ids:
        raise ValueError("initial_observation_ids are not in the supplied observation pool")
    if len(initial_ids) < 2:
        raise ValueError("at least two initial observations are required to fit a GP")
    held_out = tuple(
        observation for observation in cohort if observation.observation_id not in initial_ids
    )
    if query_budget > len(held_out):
        raise ValueError("query_budget exceeds the finite unrevealed observation pool")

    target_name = cohort[0].target_name
    reference_space = _unique_reference_conditions(cohort)
    strategies = (
        AcquisitionStrategy.RANDOM,
        AcquisitionStrategy.UNIFORM_GRID,
        AcquisitionStrategy.MAX_VARIANCE,
        AcquisitionStrategy.COST_AWARE_EIVR,
    )
    trajectories = tuple(
        _run_replay_strategy(
            strategy,
            initial=tuple(by_id[observation_id] for observation_id in initial_ids),
            held_out=held_out,
            reference_space=reference_space,
            query_budget=query_budget,
            operating_bounds=operating_bounds,
            acquisition_config=acquisition_config,
        )
        for strategy in strategies
    )
    return FinitePoolReplay(
        target_name=target_name,
        model_version=GP_MODEL_VERSION,
        query_budget=query_budget,
        initial_observation_ids=initial_ids,
        reference_condition_count=len(reference_space),
        trajectories=trajectories,
    )


def _run_replay_strategy(
    strategy: AcquisitionStrategy,
    *,
    initial: tuple[ExperimentObservation, ...],
    held_out: tuple[ExperimentObservation, ...],
    reference_space: tuple[OperatingCondition, ...],
    query_budget: int,
    operating_bounds: OperatingBounds,
    acquisition_config: AcquisitionConfig,
) -> ReplayTrajectory:
    revealed = list(initial)
    pending = list(held_out)
    rng = np.random.default_rng(GP_RANDOM_STATE)
    steps: list[ReplayStep] = []
    for step_index in range(1, query_budget + 1):
        recommender = GaussianProcessExperimentRecommender(
            operating_bounds=operating_bounds,
            acquisition_config=acquisition_config,
        ).fit(revealed)
        chosen = _select_replay_observation(
            strategy,
            recommender=recommender,
            pending=pending,
            revealed=revealed,
            reference_space=reference_space,
            random_generator=rng,
        )
        predicted_mean, predicted_std = recommender.predict_condition(chosen.condition)
        absolute_error = abs(predicted_mean - chosen.observed_target)
        errors = [step.absolute_error for step in steps] + [absolute_error]
        standard_deviations = [step.predicted_std for step in steps] + [predicted_std]
        steps.append(
            ReplayStep(
                step_index=step_index,
                selected_observation_id=chosen.observation_id,
                actual_target=chosen.observed_target,
                predicted_mean=predicted_mean,
                predicted_std=predicted_std,
                absolute_error=absolute_error,
                cumulative_mae=fmean(errors),
                cumulative_rmse=math.sqrt(fmean(error * error for error in errors)),
                cumulative_mean_predicted_std=fmean(standard_deviations),
                training_observation_count=len(revealed),
            )
        )
        revealed.append(chosen)
        pending.remove(chosen)
    return ReplayTrajectory(strategy=strategy, steps=tuple(steps))


def _select_replay_observation(
    strategy: AcquisitionStrategy,
    *,
    recommender: GaussianProcessExperimentRecommender,
    pending: Sequence[ExperimentObservation],
    revealed: Sequence[ExperimentObservation],
    reference_space: Sequence[OperatingCondition],
    random_generator: np.random.Generator,
) -> ExperimentObservation:
    ordered_pending = tuple(sorted(pending, key=lambda observation: observation.observation_id))
    if strategy is AcquisitionStrategy.RANDOM:
        return ordered_pending[int(random_generator.integers(len(ordered_pending)))]
    if strategy is AcquisitionStrategy.UNIFORM_GRID:
        return max(
            ordered_pending,
            key=lambda observation: (
                _minimum_normalized_distance(
                    recommender.operating_bounds,
                    observation.condition,
                    tuple(item.condition for item in revealed),
                ),
                _reverse_id_sort_key(observation.observation_id),
            ),
        )
    candidates = tuple(
        ExperimentCandidate(
            candidate_id=observation.observation_id,
            condition=observation.condition,
            duration_hours=observation.duration_hours,
            equipment_cost=observation.equipment_cost,
            # Finite-pool replay never proposes a production trial: each item
            # is an already completed, supplied observation.  The replay-only
            # candidate therefore carries explicit historical readiness rather
            # than inheriting the production candidate defaults.
            safety_approved=True,
            equipment_available=True,
        )
        for observation in ordered_pending
    )
    ranking = recommender.rank_candidates(
        candidates,
        reference_conditions=(
            reference_space if strategy is AcquisitionStrategy.COST_AWARE_EIVR else None
        ),
        strategy=strategy,
    )
    if not ranking.accepted:
        raise RuntimeError("finite-pool replay has no accepted candidate")
    chosen_id = ranking.accepted[0].candidate_id
    return next(
        observation
        for observation in ordered_pending
        if observation.observation_id == chosen_id
    )


def _minimum_normalized_distance(
    bounds: OperatingBounds,
    condition: OperatingCondition,
    reference_conditions: Sequence[OperatingCondition],
) -> float:
    candidate_vector = bounds.normalized_vector(condition)
    distances = []
    for reference in reference_conditions:
        reference_vector = bounds.normalized_vector(reference)
        distances.append(
            math.sqrt(
                sum(
                    (candidate_value - reference_value) ** 2
                    for candidate_value, reference_value in zip(
                        candidate_vector, reference_vector, strict=True
                    )
                )
            )
        )
    return min(distances, default=math.inf)


def _reverse_id_sort_key(value: str) -> tuple[int, ...]:
    """Choose lexical-lowest IDs when uniform-grid distances tie under ``max``."""

    return tuple(-ord(character) for character in value)


def _validate_replay_cohort(
    cohort: Sequence[ExperimentObservation], operating_bounds: OperatingBounds
) -> None:
    if len(cohort) < 3:
        raise ValueError("finite-pool replay requires at least three supplied observations")
    target_name = cohort[0].target_name
    seen_ids: set[str] = set()
    condition_ids: dict[tuple[float, float, float, float, float], str] = {}
    for observation in cohort:
        if observation.observation_id in seen_ids:
            raise ValueError(f"duplicate observation_id: {observation.observation_id}")
        seen_ids.add(observation.observation_id)
        if observation.target_name != target_name:
            raise ValueError("all replay observations must use the same target_name")
        violations = operating_bounds.violations(observation.condition)
        if violations:
            raise ValueError(
                "replay observation violates operating bounds: " + ", ".join(violations)
            )
        condition_vector = observation.condition.as_tuple()
        previous_observation_id = condition_ids.get(condition_vector)
        if previous_observation_id is not None:
            raise ValueError(
                "duplicate replay condition vector "
                f"{condition_vector} for observation IDs: "
                f"{previous_observation_id}, {observation.observation_id}"
            )
        condition_ids[condition_vector] = observation.observation_id


def _unique_reference_conditions(
    observations: Sequence[ExperimentObservation],
) -> tuple[OperatingCondition, ...]:
    result: list[OperatingCondition] = []
    seen_vectors: set[tuple[float, float, float, float, float]] = set()
    for observation in observations:
        vector = observation.condition.as_tuple()
        if vector not in seen_vectors:
            result.append(observation.condition)
            seen_vectors.add(vector)
    return tuple(result)


def _deduplicate_assessments(
    assessments: Sequence[CandidateAssessment],
) -> tuple[CandidateAssessment, ...]:
    by_id: dict[str, CandidateAssessment] = {}
    for assessment in assessments:
        by_id.setdefault(assessment.candidate_id, assessment)
    return tuple(by_id.values())


def _assessment_sort_key(assessment: CandidateAssessment) -> str:
    return assessment.candidate_id


def _stabilize_if_same_condition_space(
    covariance: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if left.shape == right.shape and np.array_equal(left, right):
        return _stabilize_posterior_covariance(covariance)
    return covariance


def _stabilize_posterior_covariance(covariance: np.ndarray) -> np.ndarray:
    """Symmetrize a covariance matrix and handle floating-point roundoff only.

    The function accepts a tiny negative diagonal no smaller than
    ``-POSTERIOR_VARIANCE_ROUNDOFF_TOLERANCE`` and sets only that numerical
    roundoff to zero.  A materially negative variance is always an error.
    """

    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("posterior covariance must be a square matrix")
    if not np.isfinite(matrix).all():
        raise RuntimeError("posterior covariance is non-finite")
    stabilized = np.asarray((matrix + matrix.T) / 2.0, dtype=float)
    diagonal = np.diag(stabilized)
    if np.any(diagonal < -POSTERIOR_VARIANCE_ROUNDOFF_TOLERANCE):
        raise RuntimeError("negative posterior variance exceeds roundoff tolerance")
    near_zero_negative = (diagonal < 0.0) & (
        diagonal >= -POSTERIOR_VARIANCE_ROUNDOFF_TOLERANCE
    )
    if np.any(near_zero_negative):
        stabilized = stabilized.copy()
        diagonal_indices = np.diag_indices_from(stabilized)
        stabilized[diagonal_indices] = np.where(diagonal < 0.0, 0.0, diagonal)
    return stabilized


def _required_score(score: float | None) -> float:
    if score is None:
        raise RuntimeError("accepted candidate assessment must have an acquisition score")
    return score
