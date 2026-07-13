"""Bounded, auditable cell-specific SOH correction over a frozen trajectory.

This module is deliberately not a model-training interface.  It receives a
finite global SOH trajectory already produced by a registered global model and
only fits three bounded *individual* parameters against explicit newly
observed measurements:

* an additive SOH bias;
* a positive degradation-rate time-warp multiplier; and
* a bounded post-midpoint knee acceleration offset.

The global model weights, architecture, data split and feature pipeline are
never present in this module and therefore cannot be modified here.  Failed,
under-observed or poorly fitted updates retain the global trajectory and return
an explicit ``RECHECK`` outcome instead of an imputed result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from itertools import pairwise
from numbers import Real
from statistics import fmean
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from quanxin_life.core import SourceKind, sha256_canonical

ONLINE_CALIBRATION_VERSION = "online-individual-calibration-v1"


class CalibrationStatus(StrEnum):
    """Whether a bounded individual correction was accepted for use."""

    ADAPTED = "ADAPTED"
    RECHECK = "RECHECK"


class _InternalModel(BaseModel):
    """Strict internal contracts; these are not public service DTOs."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be a finite real number")
    return numeric_value


class FrozenGlobalTrajectory(_InternalModel):
    """An immutable global-model trajectory eligible for individual correction.

    ``cycles`` can include the cutoff point or start immediately after it.  The
    latter directly accepts the existing global trajectory predictor output;
    observations at the cutoff are then explicitly rejected as outside this
    finite global horizon rather than fabricated by extrapolation.
    """

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0, strict=True)
    cycles: tuple[int, ...]
    soh: tuple[float, ...]
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @field_validator("cycles")
    @classmethod
    def cycles_are_valid(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) < 2:
            raise ValueError("global trajectory must contain at least two cycles")
        if any(isinstance(cycle, bool) or cycle < 0 for cycle in value):
            raise ValueError("global trajectory cycles must be non-negative integers")
        if any(current <= previous for previous, current in pairwise(value)):
            raise ValueError("global trajectory cycles must be strictly increasing")
        return value

    @field_validator("soh")
    @classmethod
    def soh_is_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2:
            raise ValueError("global trajectory must contain at least two SOH values")
        validated = tuple(_finite_real(item, name="global SOH") for item in value)
        if any(item < 0.0 or item > 1.5 for item in validated):
            raise ValueError("global SOH values must be in [0, 1.5]")
        if any(current > previous + 1e-10 for previous, current in pairwise(validated)):
            raise ValueError("global SOH trajectory must be non-increasing")
        return validated

    @model_validator(mode="after")
    def horizon_matches_cutoff(self) -> FrozenGlobalTrajectory:
        if len(self.cycles) != len(self.soh):
            raise ValueError("global trajectory cycles and SOH values must align")
        if self.cycles[0] < self.cutoff_cycle:
            raise ValueError("global trajectory horizon cannot precede cutoff_cycle")
        return self


class NewlyObservedSOH(_InternalModel):
    """One actual, newly measured SOH point; predicted or simulated points are refused."""

    cycle: int = Field(ge=0, strict=True)
    soh: float = Field(ge=0.0, le=1.5, strict=True)
    source_kind: SourceKind = SourceKind.NEWLY_OBSERVED

    @field_validator("soh")
    @classmethod
    def soh_is_finite(cls, value: float) -> float:
        return _finite_real(value, name="newly observed SOH")

    @model_validator(mode="after")
    def source_is_a_real_new_observation(self) -> NewlyObservedSOH:
        if self.source_kind is not SourceKind.NEWLY_OBSERVED:
            raise ValueError("online calibration only accepts NEWLY_OBSERVED SOH points")
        return self


class CalibrationConfig(_InternalModel):
    """Versioned numerical bounds and regularisation for individual fitting."""

    config_version: str = ONLINE_CALIBRATION_VERSION
    bias_bounds: tuple[float, float] = (-0.05, 0.05)
    rate_multiplier_bounds: tuple[float, float] = (0.5, 2.0)
    knee_offset_bounds: tuple[float, float] = (-0.25, 0.25)
    bias_regularization: float = Field(default=1.0, ge=0.0)
    rate_regularization: float = Field(default=0.50, ge=0.0)
    knee_regularization: float = Field(default=0.50, ge=0.0)
    min_observation_count: int = Field(default=3, ge=3, strict=True)
    max_fit_rmse: float = Field(default=0.05, gt=0.0)
    max_nfev: int = Field(default=200, ge=1, strict=True)

    @model_validator(mode="after")
    def bounds_are_safe_and_finite(self) -> CalibrationConfig:
        for name in ("bias_bounds", "rate_multiplier_bounds", "knee_offset_bounds"):
            lower, upper = getattr(self, name)
            lower_value = _finite_real(lower, name=f"{name} lower")
            upper_value = _finite_real(upper, name=f"{name} upper")
            if lower_value >= upper_value:
                raise ValueError(f"{name} lower bound must be below upper bound")
        if self.rate_multiplier_bounds[0] <= 0.0:
            raise ValueError("rate_multiplier_bounds must remain strictly positive")
        # Effective time stays non-decreasing on both sides of the midpoint.
        if self.rate_multiplier_bounds[0] + self.knee_offset_bounds[0] <= 0.0:
            raise ValueError("rate and knee bounds would permit a time reversal")
        for name in (
            "bias_regularization",
            "rate_regularization",
            "knee_regularization",
            "max_fit_rmse",
        ):
            _finite_real(getattr(self, name), name=name)
        return self


class IndividualCalibrationParameters(_InternalModel):
    """The only parameters allowed to change for one individual cell."""

    bias_soh: float = 0.0
    rate_multiplier: float = 1.0
    knee_offset_fraction: float = 0.0

    @field_validator("bias_soh", "rate_multiplier", "knee_offset_fraction")
    @classmethod
    def parameters_are_finite(cls, value: float) -> float:
        return _finite_real(value, name="individual calibration parameter")

    @model_validator(mode="after")
    def rate_is_positive(self) -> IndividualCalibrationParameters:
        if self.rate_multiplier <= 0.0:
            raise ValueError("rate_multiplier must be strictly positive")
        return self


class IndividualCalibrationAudit(_InternalModel):
    """Immutable evidence record for an accepted or rejected online update."""

    global_trajectory_hash: str = Field(min_length=64, max_length=64)
    global_model_version: str = Field(min_length=1)
    prior_parameters: IndividualCalibrationParameters
    updated_parameters: IndividualCalibrationParameters
    observed_until_cycle: int | None = Field(default=None, ge=0)
    observation_count: int = Field(ge=0)
    update_version: str = Field(min_length=1)
    objective_value: float | None = Field(default=None, ge=0.0)
    fit_rmse: float | None = Field(default=None, ge=0.0)
    optimizer_status: str = Field(min_length=1)
    optimizer_message: str = Field(min_length=1)


class IndividualCalibrationOutcome(_InternalModel):
    """Frozen-horizon correction result or a clear ``RECHECK`` degradation."""

    status: CalibrationStatus
    reason_code: str = Field(min_length=1)
    trajectory_cycles: tuple[int, ...]
    adapted_soh: tuple[float, ...]
    audit: IndividualCalibrationAudit
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def adapted_trajectory_is_valid(self) -> IndividualCalibrationOutcome:
        if len(self.trajectory_cycles) != len(self.adapted_soh) or not self.trajectory_cycles:
            raise ValueError("adapted trajectory cycles and SOH values must align")
        if any(current <= previous for previous, current in pairwise(self.trajectory_cycles)):
            raise ValueError("adapted trajectory cycles must be strictly increasing")
        values = tuple(_finite_real(value, name="adapted SOH") for value in self.adapted_soh)
        if any(value < 0.0 or value > 1.5 for value in values):
            raise ValueError("adapted SOH must remain in [0, 1.5]")
        if any(current > previous + 1e-10 for previous, current in pairwise(values)):
            raise ValueError("adapted SOH trajectory must be non-increasing")
        return self


class ReplayStep(_InternalModel):
    """One observed/unobserved split in a deterministic online replay."""

    update_cycle: int = Field(ge=0)
    outcome: IndividualCalibrationOutcome
    withheld_observation_count: int = Field(ge=0)
    before_mae: float | None = Field(default=None, ge=0.0)
    before_rmse: float | None = Field(default=None, ge=0.0)
    after_mae: float | None = Field(default=None, ge=0.0)
    after_rmse: float | None = Field(default=None, ge=0.0)
    warnings: tuple[str, ...] = ()


class CalibrationReplay(_InternalModel):
    """Transparent rolling replay; no field asserts or infers improvement."""

    global_trajectory_hash: str = Field(min_length=64, max_length=64)
    update_version_prefix: str = Field(min_length=1)
    steps: tuple[ReplayStep, ...]


class IndividualTrajectoryCalibrator:
    """Fit bounded per-cell correction parameters while keeping the global path frozen."""

    def __init__(self, *, config: CalibrationConfig | None = None) -> None:
        self._config = config or CalibrationConfig()

    @property
    def config(self) -> CalibrationConfig:
        """Return the immutable, versioned fitting configuration."""

        return self._config

    def calibrate(
        self,
        *,
        global_trajectory: FrozenGlobalTrajectory,
        observations: Sequence[NewlyObservedSOH],
        update_version: str,
    ) -> IndividualCalibrationOutcome:
        """Apply a bounded individual update or explicitly return ``RECHECK``.

        ``global_trajectory`` is read-only.  The only optimizer variables are
        :class:`IndividualCalibrationParameters`; no global network, feature
        transformer or model artefact is visible to this method.
        """

        if not update_version.strip():
            raise ValueError("update_version must be non-empty")
        ordered_observations = _validate_observations(global_trajectory, observations)
        prior = IndividualCalibrationParameters()
        global_hash = _trajectory_hash(global_trajectory)
        observed_until = ordered_observations[-1].cycle if ordered_observations else None

        if len(ordered_observations) < self._config.min_observation_count:
            return self._recheck_outcome(
                global_trajectory=global_trajectory,
                global_hash=global_hash,
                prior=prior,
                observations=ordered_observations,
                update_version=update_version,
                reason_code="INSUFFICIENT_OBSERVATIONS",
                optimizer_status="NOT_RUN",
                optimizer_message=(
                    "individual calibration requires at least "
                    f"{self._config.min_observation_count} newly observed SOH points"
                ),
            )

        initial = np.asarray((0.0, 1.0, 0.0), dtype=float)
        lower_bounds = np.asarray(
            (
                self._config.bias_bounds[0],
                self._config.rate_multiplier_bounds[0],
                self._config.knee_offset_bounds[0],
            ),
            dtype=float,
        )
        upper_bounds = np.asarray(
            (
                self._config.bias_bounds[1],
                self._config.rate_multiplier_bounds[1],
                self._config.knee_offset_bounds[1],
            ),
            dtype=float,
        )

        result = least_squares(
            lambda vector: self._objective_residuals(
                vector=vector,
                global_trajectory=global_trajectory,
                observations=ordered_observations,
            ),
            x0=initial,
            bounds=(lower_bounds, upper_bounds),
            max_nfev=self._config.max_nfev,
            method="trf",
        )
        parameters = IndividualCalibrationParameters(
            bias_soh=float(result.x[0]),
            rate_multiplier=float(result.x[1]),
            knee_offset_fraction=float(result.x[2]),
        )
        objective_value = float(2.0 * result.cost)
        candidate_soh = _adapted_trajectory_values(global_trajectory, parameters)
        observed_predictions = _interpolate_soh(
            cycles=global_trajectory.cycles,
            soh=candidate_soh,
            query_cycles=tuple(observation.cycle for observation in ordered_observations),
        )
        residuals = tuple(
            prediction - observation.soh
            for prediction, observation in zip(
                observed_predictions, ordered_observations, strict=True
            )
        )
        fit_rmse = math.sqrt(fmean(residual * residual for residual in residuals))
        optimizer_message = str(result.message).strip() or "optimizer returned no message"

        if not result.success:
            return self._recheck_outcome(
                global_trajectory=global_trajectory,
                global_hash=global_hash,
                prior=prior,
                observations=ordered_observations,
                update_version=update_version,
                reason_code="OPTIMIZER_UNSUCCESSFUL",
                optimizer_status=str(result.status),
                optimizer_message=optimizer_message,
                objective_value=objective_value,
                fit_rmse=fit_rmse,
            )
        if not math.isfinite(objective_value) or not math.isfinite(fit_rmse):
            return self._recheck_outcome(
                global_trajectory=global_trajectory,
                global_hash=global_hash,
                prior=prior,
                observations=ordered_observations,
                update_version=update_version,
                reason_code="NONFINITE_OPTIMIZATION_RESULT",
                optimizer_status=str(result.status),
                optimizer_message=optimizer_message,
            )
        if fit_rmse > self._config.max_fit_rmse:
            return self._recheck_outcome(
                global_trajectory=global_trajectory,
                global_hash=global_hash,
                prior=prior,
                observations=ordered_observations,
                update_version=update_version,
                reason_code="FIT_QUALITY_INSUFFICIENT",
                optimizer_status=str(result.status),
                optimizer_message=optimizer_message,
                objective_value=objective_value,
                fit_rmse=fit_rmse,
            )

        return IndividualCalibrationOutcome(
            status=CalibrationStatus.ADAPTED,
            reason_code="ADAPTED",
            trajectory_cycles=global_trajectory.cycles,
            adapted_soh=candidate_soh,
            audit=IndividualCalibrationAudit(
                global_trajectory_hash=global_hash,
                global_model_version=global_trajectory.model_version,
                prior_parameters=prior,
                updated_parameters=parameters,
                observed_until_cycle=observed_until,
                observation_count=len(ordered_observations),
                update_version=update_version,
                objective_value=objective_value,
                fit_rmse=fit_rmse,
                optimizer_status=str(result.status),
                optimizer_message=optimizer_message,
            ),
        )

    def _objective_residuals(
        self,
        *,
        vector: np.ndarray[Any, Any],
        global_trajectory: FrozenGlobalTrajectory,
        observations: Sequence[NewlyObservedSOH],
    ) -> np.ndarray[Any, Any]:
        parameters = IndividualCalibrationParameters(
            bias_soh=float(vector[0]),
            rate_multiplier=float(vector[1]),
            knee_offset_fraction=float(vector[2]),
        )
        candidate_soh = _adapted_trajectory_values(global_trajectory, parameters)
        predictions = _interpolate_soh(
            cycles=global_trajectory.cycles,
            soh=candidate_soh,
            query_cycles=tuple(observation.cycle for observation in observations),
        )
        data_residuals = [
            prediction - observation.soh
            for prediction, observation in zip(predictions, observations, strict=True)
        ]
        regularization_residuals = [
            math.sqrt(self._config.bias_regularization) * parameters.bias_soh,
            math.sqrt(self._config.rate_regularization) * (parameters.rate_multiplier - 1.0),
            math.sqrt(self._config.knee_regularization) * parameters.knee_offset_fraction,
        ]
        result = np.asarray((*data_residuals, *regularization_residuals), dtype=float)
        if not np.isfinite(result).all():
            raise ValueError("individual calibration objective became non-finite")
        return result

    def _recheck_outcome(
        self,
        *,
        global_trajectory: FrozenGlobalTrajectory,
        global_hash: str,
        prior: IndividualCalibrationParameters,
        observations: Sequence[NewlyObservedSOH],
        update_version: str,
        reason_code: str,
        optimizer_status: str,
        optimizer_message: str,
        objective_value: float | None = None,
        fit_rmse: float | None = None,
    ) -> IndividualCalibrationOutcome:
        return IndividualCalibrationOutcome(
            status=CalibrationStatus.RECHECK,
            reason_code=reason_code,
            trajectory_cycles=global_trajectory.cycles,
            adapted_soh=global_trajectory.soh,
            audit=IndividualCalibrationAudit(
                global_trajectory_hash=global_hash,
                global_model_version=global_trajectory.model_version,
                prior_parameters=prior,
                updated_parameters=prior,
                observed_until_cycle=observations[-1].cycle if observations else None,
                observation_count=len(observations),
                update_version=update_version,
                objective_value=objective_value,
                fit_rmse=fit_rmse,
                optimizer_status=optimizer_status,
                optimizer_message=optimizer_message,
            ),
            warnings=(reason_code,),
        )


def replay_individual_calibration(
    *,
    global_trajectory: FrozenGlobalTrajectory,
    observations: Sequence[NewlyObservedSOH],
    update_cycles: Sequence[int] = (20, 50, 100, 150),
    update_version_prefix: str,
    calibrator: IndividualTrajectoryCalibrator | None = None,
) -> CalibrationReplay:
    """Replay staged updates and report measured future-observation errors.

    At every update stage the calibrator only receives points at or before that
    stage.  Errors are reported against the supplied later observations; the
    replay intentionally provides no "improved" boolean or superiority claim.
    """

    if not update_version_prefix.strip():
        raise ValueError("update_version_prefix must be non-empty")
    ordered_observations = _validate_observations(global_trajectory, observations)
    stages = tuple(update_cycles)
    if not stages:
        raise ValueError("at least one update cycle is required")
    if any(isinstance(cycle, bool) or cycle < global_trajectory.cutoff_cycle for cycle in stages):
        raise ValueError("replay update cycles cannot precede cutoff_cycle")
    if any(current <= previous for previous, current in pairwise(stages)):
        raise ValueError("replay update cycles must be strictly increasing")
    if any(
        cycle < global_trajectory.cycles[0] or cycle > global_trajectory.cycles[-1]
        for cycle in stages
    ):
        raise ValueError("replay update cycles must lie inside the global trajectory horizon")

    service = calibrator or IndividualTrajectoryCalibrator()
    steps: list[ReplayStep] = []
    for update_cycle in stages:
        seen = tuple(
            observation
            for observation in ordered_observations
            if observation.cycle <= update_cycle
        )
        withheld = tuple(
            observation for observation in ordered_observations if observation.cycle > update_cycle
        )
        outcome = service.calibrate(
            global_trajectory=global_trajectory,
            observations=seen,
            update_version=f"{update_version_prefix}:{update_cycle}",
        )
        before_mae, before_rmse = _trajectory_error_metrics(
            cycles=global_trajectory.cycles,
            soh=global_trajectory.soh,
            withheld=withheld,
        )
        after_mae, after_rmse = _trajectory_error_metrics(
            cycles=outcome.trajectory_cycles,
            soh=outcome.adapted_soh,
            withheld=withheld,
        )
        warnings = () if withheld else ("NO_WITHHELD_OBSERVATIONS",)
        steps.append(
            ReplayStep(
                update_cycle=update_cycle,
                outcome=outcome,
                withheld_observation_count=len(withheld),
                before_mae=before_mae,
                before_rmse=before_rmse,
                after_mae=after_mae,
                after_rmse=after_rmse,
                warnings=warnings,
            )
        )

    return CalibrationReplay(
        global_trajectory_hash=_trajectory_hash(global_trajectory),
        update_version_prefix=update_version_prefix,
        steps=tuple(steps),
    )


def _validate_observations(
    global_trajectory: FrozenGlobalTrajectory,
    observations: Sequence[NewlyObservedSOH],
) -> tuple[NewlyObservedSOH, ...]:
    seen_cycles: set[int] = set()
    ordered: list[NewlyObservedSOH] = []
    for observation in observations:
        if observation.cycle in seen_cycles:
            raise ValueError(f"duplicate newly observed cycle: {observation.cycle}")
        seen_cycles.add(observation.cycle)
        if observation.cycle < global_trajectory.cutoff_cycle:
            raise ValueError("newly observed cycle cannot precede cutoff_cycle")
        if (
            observation.cycle < global_trajectory.cycles[0]
            or observation.cycle > global_trajectory.cycles[-1]
        ):
            raise ValueError("newly observed cycle is outside the global trajectory horizon")
        ordered.append(observation)
    return tuple(sorted(ordered, key=lambda item: item.cycle))


def _adapted_trajectory_values(
    global_trajectory: FrozenGlobalTrajectory,
    parameters: IndividualCalibrationParameters,
) -> tuple[float, ...]:
    cycles = np.asarray(global_trajectory.cycles, dtype=float)
    global_soh = np.asarray(global_trajectory.soh, dtype=float)
    normalized_time = (cycles - cycles[0]) / (cycles[-1] - cycles[0])
    # A non-negative local derivative is guaranteed by CalibrationConfig:
    # derivative is rate before midpoint and rate + knee offset afterwards.
    effective_time = (
        parameters.rate_multiplier * normalized_time
        + parameters.knee_offset_fraction * np.maximum(normalized_time - 0.5, 0.0)
    )
    effective_time = np.clip(effective_time, 0.0, 1.0)
    warped = np.interp(effective_time, normalized_time, global_soh)
    adjusted = np.clip(warped + parameters.bias_soh, 0.0, 1.5)
    monotone = np.minimum.accumulate(adjusted)
    if not np.isfinite(monotone).all():
        raise RuntimeError("individual calibration produced a non-finite trajectory")
    return tuple(float(value) for value in monotone.tolist())


def _interpolate_soh(
    *,
    cycles: Sequence[int],
    soh: Sequence[float],
    query_cycles: Sequence[int],
) -> tuple[float, ...]:
    if not query_cycles:
        return ()
    cycle_values = np.asarray(cycles, dtype=float)
    soh_values = np.asarray(soh, dtype=float)
    queries = np.asarray(query_cycles, dtype=float)
    if queries.min() < cycle_values[0] or queries.max() > cycle_values[-1]:
        raise ValueError("query cycles are outside the global trajectory horizon")
    values = np.interp(queries, cycle_values, soh_values)
    if not np.isfinite(values).all():
        raise RuntimeError("trajectory interpolation produced a non-finite SOH value")
    return tuple(float(value) for value in values.tolist())


def _trajectory_error_metrics(
    *,
    cycles: Sequence[int],
    soh: Sequence[float],
    withheld: Sequence[NewlyObservedSOH],
) -> tuple[float | None, float | None]:
    if not withheld:
        return None, None
    predictions = _interpolate_soh(
        cycles=cycles,
        soh=soh,
        query_cycles=tuple(observation.cycle for observation in withheld),
    )
    errors = tuple(
        prediction - observation.soh
        for prediction, observation in zip(predictions, withheld, strict=True)
    )
    return (
        fmean(abs(error) for error in errors),
        math.sqrt(fmean(error * error for error in errors)),
    )


def _trajectory_hash(global_trajectory: FrozenGlobalTrajectory) -> str:
    return sha256_canonical(global_trajectory.model_dump(mode="json"))
