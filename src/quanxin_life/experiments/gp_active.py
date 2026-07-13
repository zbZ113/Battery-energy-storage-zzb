"""Low-dimensional Gaussian-process experiment recommendation.

This module is intentionally a numerical service rather than an Agent or a
dataset loader.  A caller must provide fully observed, provenance-reviewed
condition/target pairs.  It neither invents Naumann measurements nor reads
serialized model artifacts.  The fitted scikit-learn estimator stays only in
memory; publication belongs to the governed model-artifact pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Real
from statistics import fmean
from typing import Self

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore[import-untyped]
from sklearn.gaussian_process.kernels import (  # type: ignore[import-untyped]
    Kernel,
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

GP_RANDOM_STATE = 20260712
GP_MODEL_VERSION = "naumann-condition-gp-v1"
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
    safety_approved: bool = True
    equipment_available: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if _require_finite_real(self.duration_hours, name="duration_hours") <= 0:
            raise ValueError("duration_hours must be positive")
        if _require_finite_real(self.equipment_cost, name="equipment_cost") <= 0:
            raise ValueError("equipment_cost must be positive")
        if not isinstance(self.safety_approved, bool):
            raise ValueError("safety_approved must be boolean")
        if not isinstance(self.equipment_available, bool):
            raise ValueError("equipment_available must be boolean")


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


@dataclass(frozen=True)
class ExperimentRecommendation:
    """Deterministically ordered accepted and rejected candidate assessments."""

    target_name: str
    model_version: str
    accepted: tuple[CandidateAssessment, ...]
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
        """Fit only from real, finite, reviewed observations in the declared bounds."""

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
        # The feature order is fixed by CONDITION_FEATURE_NAMES.  The initial
        # kernel is deliberately kept fixed: fitting a tiny experiment cohort
        # against boundary-constrained hyperparameters often emits convergence
        # warnings that cannot be resolved from the available observations.
        # Hyperparameter-selection policy belongs to a separately versioned
        # experiment protocol, not an implicit optimizer side effect.
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
        self, candidates: Sequence[ExperimentCandidate]
    ) -> ExperimentRecommendation:
        """Score in-bound, ready candidates and reject every unsafe candidate explicitly."""

        self._require_fitted()
        seen_candidate_ids: set[str] = set()
        accepted: list[CandidateAssessment] = []
        rejected: list[CandidateAssessment] = []
        for candidate in candidates:
            if candidate.candidate_id in seen_candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate.candidate_id}")
            seen_candidate_ids.add(candidate.candidate_id)

            rejection_reasons, constraint_details = self._candidate_rejection(candidate)
            if rejection_reasons:
                rejected.append(
                    CandidateAssessment(
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
                )
                continue

            predicted_mean, predicted_std = self.predict_condition(candidate.condition)
            normalized_cost = self._normalized_cost(candidate)
            duplicate_penalty = self._duplicate_penalty(candidate.condition)
            acquisition_score = predicted_std / normalized_cost - duplicate_penalty
            accepted.append(
                CandidateAssessment(
                    candidate_id=candidate.candidate_id,
                    accepted=True,
                    predicted_mean=predicted_mean,
                    predicted_std=predicted_std,
                    normalized_cost=normalized_cost,
                    duplicate_penalty=duplicate_penalty,
                    acquisition_score=acquisition_score,
                )
            )

        return ExperimentRecommendation(
            target_name=self.target_name,
            model_version=GP_MODEL_VERSION,
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

    def leave_one_out_replay(self) -> LeaveOneOutReplay:
        """Re-fit per held-out observation and report actual prediction errors.

        The result intentionally reports only observed targets and pointwise GP
        comparison quantities.  It does not assert that this model or any
        acquisition strategy is superior to another method.
        """

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

    def _candidate_rejection(
        self, candidate: ExperimentCandidate
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        reasons: list[str] = []
        details: list[str] = []
        violations = self.operating_bounds.violations(candidate.condition)
        if violations:
            reasons.append("OUT_OF_OPERATING_BOUNDS")
            details.extend(violations)
        if not candidate.safety_approved:
            reasons.append("SAFETY_NOT_APPROVED")
        if not candidate.equipment_available:
            reasons.append("EQUIPMENT_UNAVAILABLE")
        return tuple(reasons), tuple(details)

    def _normalized_cost(self, candidate: ExperimentCandidate) -> float:
        normalized_cost = (
            candidate.duration_hours / self.acquisition_config.time_normalizer_hours
            + candidate.equipment_cost / self.acquisition_config.equipment_cost_normalizer
        )
        if not math.isfinite(normalized_cost) or normalized_cost <= 0:
            raise RuntimeError("validated candidate cost must normalize to a positive finite value")
        return normalized_cost

    def _duplicate_penalty(self, condition: OperatingCondition) -> float:
        candidate_vector = self.operating_bounds.normalized_vector(condition)
        similarities = []
        for observation in self._observations:
            observed_vector = self.operating_bounds.normalized_vector(observation.condition)
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


def _required_score(score: float | None) -> float:
    if score is None:
        raise RuntimeError("accepted candidate assessment must have an acquisition score")
    return score
