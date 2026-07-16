"""Structurally monotone SOH-trajectory model for early-cycle evidence.

The model separates an interpretable positive trend term from a learned
non-negative residual-degradation increment.  Their cumulative subtraction
from the last observed SOH makes a long-term recovery structurally impossible;
this is stronger than relying only on a monotonicity penalty.

This module deliberately predicts a finite SOH trajectory only.  EOL80 and
RUL are derived from its first threshold crossing and are never trained as an
independent output head. Governed persistence is handled by the separate
strict safetensors adapter; this module never loads pickle-compatible
artefacts.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Real

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn
from torch.nn import functional as functional

from quanxin_life.data.schemas import SplitManifest

HYBRID_RANDOM_SEED = 20260712
EOL80_THRESHOLD = 0.8


class _InternalModel(BaseModel):
    """Strict internal schema; it is intentionally not an external DTO."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _finite_real(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be a finite real number")
    return numeric_value


def _strictly_increasing(values: Sequence[int], *, field_name: str) -> None:
    if any(current <= previous for previous, current in pairwise(values)):
        raise ValueError(f"{field_name} must be strictly increasing")


def normalise_prediction_cycle_positions(
    *, prediction_cycles: Sequence[int], cutoff_cycle: int
) -> tuple[float, ...]:
    """Map real post-cutoff cycle positions to the closed interval ``(0, 1]``.

    This deliberately uses the declared cycle coordinates rather than a
    uniformly spaced vector.  A sparse forecast axis therefore retains its
    actual temporal/cycle spacing in both the trend and residual terms.
    """

    if cutoff_cycle < 0:
        raise ValueError("cutoff_cycle must be non-negative")
    if not prediction_cycles:
        raise ValueError("prediction_cycles must be non-empty")
    _strictly_increasing(prediction_cycles, field_name="prediction_cycles")
    if any(cycle <= cutoff_cycle for cycle in prediction_cycles):
        raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
    horizon_span = prediction_cycles[-1] - cutoff_cycle
    return tuple((cycle - cutoff_cycle) / float(horizon_span) for cycle in prediction_cycles)


class TrajectoryTrainingSample(_InternalModel):
    """One train-cell trajectory with a cutoff-safe early observation window."""

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    observed_cycles: tuple[int, ...]
    observed_soh: tuple[float, ...]
    target_cycles: tuple[int, ...]
    target_soh: tuple[float, ...]
    condition_features: dict[str, float]
    feature_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @field_validator("observed_cycles", "target_cycles")
    @classmethod
    def cycles_are_nonnegative(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(cycle < 0 for cycle in value):
            raise ValueError("cycle sequences must be non-empty and non-negative")
        return value

    @field_validator("observed_soh", "target_soh")
    @classmethod
    def soh_values_are_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("SOH sequences must be non-empty")
        for soh in value:
            numeric_soh = _finite_real(soh, field_name="SOH")
            if numeric_soh <= 0 or numeric_soh > 1.5:
                raise ValueError("SOH values must be in (0, 1.5]")
        return tuple(float(soh) for soh in value)

    @field_validator("condition_features")
    @classmethod
    def conditions_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            raise ValueError("condition_features must be non-empty")
        validated: dict[str, float] = {}
        for name, raw_value in value.items():
            if not name:
                raise ValueError("condition feature names must be non-empty")
            validated[name] = _finite_real(raw_value, field_name=f"condition feature {name}")
        return validated

    @model_validator(mode="after")
    def observation_and_target_windows_are_cutoff_safe(self) -> TrajectoryTrainingSample:
        if len(self.observed_cycles) != len(self.observed_soh):
            raise ValueError("observed_cycles and observed_soh must align")
        if len(self.target_cycles) != len(self.target_soh):
            raise ValueError("target_cycles and target_soh must align")
        _strictly_increasing(self.observed_cycles, field_name="observed_cycles")
        _strictly_increasing(self.target_cycles, field_name="target_cycles")
        if any(cycle > self.cutoff_cycle for cycle in self.observed_cycles):
            raise ValueError("observed_cycles cannot exceed cutoff_cycle")
        if self.observed_cycles[-1] != self.cutoff_cycle:
            raise ValueError("the last observed cycle must equal cutoff_cycle")
        if any(cycle <= self.cutoff_cycle for cycle in self.target_cycles):
            raise ValueError("target_cycles must be strictly after cutoff_cycle")
        return self


class EOL80Crossing(_InternalModel):
    """Threshold result derived only from a predicted SOH trajectory."""

    cutoff_cycle: int = Field(ge=0)
    eol80_cycle: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def crossing_must_be_after_cutoff(self) -> EOL80Crossing:
        if self.eol80_cycle is not None and self.eol80_cycle <= self.cutoff_cycle:
            raise ValueError("eol80_cycle must be strictly after cutoff_cycle")
        return self

    @property
    def derived_rul_cycle(self) -> int | None:
        if self.eol80_cycle is None:
            return None
        return self.eol80_cycle - self.cutoff_cycle


class SOHTrajectoryPrediction(_InternalModel):
    """Finite-horizon, model-produced SOH trajectory and its derived crossing."""

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    prediction_cycles: tuple[int, ...]
    predicted_soh: tuple[float, ...]
    eol80_crossing: EOL80Crossing
    feature_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    data_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def trajectory_is_finite_and_monotone(self) -> SOHTrajectoryPrediction:
        if len(self.prediction_cycles) != len(self.predicted_soh) or not self.prediction_cycles:
            raise ValueError("prediction_cycles and predicted_soh must be non-empty and align")
        _strictly_increasing(self.prediction_cycles, field_name="prediction_cycles")
        if any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
        for soh in self.predicted_soh:
            numeric_soh = _finite_real(soh, field_name="predicted SOH")
            if numeric_soh < 0 or numeric_soh > 1.5:
                raise ValueError("predicted SOH must be in [0, 1.5]")
        if any(
            current > previous + 1e-8
            for previous, current in pairwise(self.predicted_soh)
        ):
            raise ValueError("predicted SOH trajectory must be non-increasing")
        if self.eol80_crossing.cutoff_cycle != self.cutoff_cycle:
            raise ValueError("EOL80 crossing cutoff must match the trajectory cutoff")
        derived_crossing = derive_eol80_crossing(
            prediction_cycles=self.prediction_cycles,
            predicted_soh=self.predicted_soh,
            cutoff_cycle=self.cutoff_cycle,
        )
        if self.eol80_crossing != derived_crossing:
            raise ValueError("eol80_crossing must be derived from predicted_soh")
        return self

    @property
    def derived_rul_cycle(self) -> int | None:
        return self.eol80_crossing.derived_rul_cycle


def derive_eol80_crossing(
    *,
    prediction_cycles: Sequence[int],
    predicted_soh: Sequence[float],
    cutoff_cycle: int,
) -> EOL80Crossing:
    """Find the first modeled EOL80 crossing; never invent a separate RUL."""

    if len(prediction_cycles) != len(predicted_soh) or not prediction_cycles:
        raise ValueError("prediction_cycles and predicted_soh must be non-empty and align")
    _strictly_increasing(prediction_cycles, field_name="prediction_cycles")
    if any(cycle <= cutoff_cycle for cycle in prediction_cycles):
        raise ValueError("prediction_cycles must be strictly after cutoff_cycle")
    for cycle, soh in zip(prediction_cycles, predicted_soh, strict=True):
        numeric_soh = _finite_real(soh, field_name="predicted SOH")
        if numeric_soh <= EOL80_THRESHOLD:
            return EOL80Crossing(cutoff_cycle=cutoff_cycle, eol80_cycle=cycle)
    return EOL80Crossing(cutoff_cycle=cutoff_cycle)


class _HybridNetwork(nn.Module):
    """Positive trend plus cumulative positive residual-degradation increments."""

    def __init__(self, *, input_dim: int, horizon: int, hidden_dim: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # sqrt/linear/knee acceleration amplitudes plus a normalised knee location.
        self.trend_head = nn.Linear(hidden_dim, 4)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor, initial_soh: Tensor, cycle_positions: Tensor) -> Tensor:
        if features.ndim != 2 or initial_soh.ndim != 1 or len(features) != len(initial_soh):
            raise ValueError("feature and initial-SOH batches must align")
        if cycle_positions.ndim != 1 or len(cycle_positions) != self.horizon:
            raise ValueError(
                "cycle_positions must be one-dimensional and match the forecast horizon"
            )
        if not torch.isfinite(cycle_positions).all() or torch.any(cycle_positions <= 0):
            raise ValueError("cycle_positions must be finite and strictly positive")
        if torch.any(cycle_positions[1:] <= cycle_positions[:-1]):
            raise ValueError("cycle_positions must be strictly increasing")
        encoded = self.encoder(features)
        raw_trend = self.trend_head(encoded)
        # Small positive scales make the components numerically stable while
        # retaining learnable differences between cells and conditions.
        sqrt_amplitude = functional.softplus(raw_trend[:, 0:1]) * 0.08
        linear_amplitude = functional.softplus(raw_trend[:, 1:2]) * 0.10
        acceleration_amplitude = functional.softplus(raw_trend[:, 2:3]) * 0.10
        knee_location = torch.sigmoid(raw_trend[:, 3:4])

        time_axis = cycle_positions.unsqueeze(0).to(dtype=features.dtype, device=features.device)
        trend = (
            sqrt_amplitude * torch.sqrt(time_axis)
            + linear_amplitude * time_axis
            + acceleration_amplitude * torch.relu(time_axis - knee_location).square()
        )
        encoded_by_time = encoded.unsqueeze(1).expand(-1, self.horizon, -1)
        residual_inputs = torch.cat(
            (encoded_by_time, time_axis.expand(len(encoded), -1).unsqueeze(-1)), dim=-1
        )
        residual_increments = functional.softplus(self.residual_head(residual_inputs).squeeze(-1))
        initial_position = torch.zeros(1, dtype=time_axis.dtype, device=time_axis.device)
        position_steps = torch.diff(time_axis.squeeze(0), prepend=initial_position).unsqueeze(0)
        # Each residual increment is non-negative and accumulated; it cannot
        # cause an SOH recovery even if the learned residual network fluctuates.
        residual_degradation = torch.cumsum(residual_increments * position_steps, dim=1) * 0.03
        predicted = initial_soh.unsqueeze(1) - trend - residual_degradation
        return torch.clamp(predicted, min=0.0, max=1.5)


@dataclass(frozen=True)
class _FittedContext:
    dataset_id: str
    condition_feature_names: tuple[str, ...]


@dataclass
class HybridDegradationPredictor:
    """Fit a monotone SOH-trajectory model on complete train-cell trajectories."""

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    prediction_cycles: tuple[int, ...]
    condition_feature_names: tuple[str, ...]
    hidden_dim: int = 32
    epochs: int = 300
    learning_rate: float = 0.001
    _network: _HybridNetwork | None = field(default=None, init=False, repr=False)
    _context: _FittedContext | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("model_version", self.model_version),
            ("feature_version", self.feature_version),
            ("split_version", self.split_version),
            ("data_version", self.data_version),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        if self.hidden_dim < 1 or self.epochs < 1 or self.learning_rate <= 0:
            raise ValueError("hidden_dim, epochs and learning_rate must be positive")
        if not self.condition_feature_names or len(set(self.condition_feature_names)) != len(
            self.condition_feature_names
        ):
            raise ValueError("condition_feature_names must be non-empty and unique")
        _strictly_increasing(self.prediction_cycles, field_name="prediction_cycles")
        if any(cycle <= self.cutoff_cycle for cycle in self.prediction_cycles):
            raise ValueError("prediction_cycles must be strictly after cutoff_cycle")

    def fit(
        self,
        training_samples: Sequence[TrajectoryTrainingSample],
        *,
        split_manifest: SplitManifest,
    ) -> HybridDegradationPredictor:
        """Fit using only the exact, cell-disjoint train cohort in ``split_manifest``."""

        if split_manifest.dataset_id == "":
            raise ValueError("split manifest dataset_id must be non-empty")
        expected_cell_ids = set(split_manifest.train)
        supplied_cell_ids = [sample.cell_id for sample in training_samples]
        if len(set(supplied_cell_ids)) != len(supplied_cell_ids):
            raise ValueError("training samples must not contain duplicate cell_id values")
        if set(supplied_cell_ids) != expected_cell_ids:
            raise ValueError("training sample cell_ids must exactly match split_manifest.train")
        if not training_samples:
            raise ValueError("at least one training sample is required")

        input_rows: list[list[float]] = []
        initial_soh_values: list[float] = []
        target_rows: list[list[float]] = []
        for sample in sorted(training_samples, key=lambda item: item.cell_id):
            self._validate_training_sample(sample, split_manifest=split_manifest)
            input_rows.append(
                self._build_features(
                    sample.observed_cycles,
                    sample.observed_soh,
                    sample.condition_features,
                )
            )
            initial_soh_values.append(float(sample.observed_soh[-1]))
            target_rows.append([float(value) for value in sample.target_soh])

        self._set_deterministic_seed()
        features = torch.tensor(input_rows, dtype=torch.float32, device="cpu")
        initial_soh = torch.tensor(initial_soh_values, dtype=torch.float32, device="cpu")
        targets = torch.tensor(target_rows, dtype=torch.float32, device="cpu")
        cycle_positions = torch.tensor(
            normalise_prediction_cycle_positions(
                prediction_cycles=self.prediction_cycles,
                cutoff_cycle=self.cutoff_cycle,
            ),
            dtype=torch.float32,
            device="cpu",
        )
        network = _HybridNetwork(
            input_dim=features.shape[1],
            horizon=len(self.prediction_cycles),
            hidden_dim=self.hidden_dim,
        ).to(device="cpu")
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)

        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            predicted = network(features, initial_soh, cycle_positions)
            fit_loss = functional.mse_loss(predicted, targets)
            smooth_loss = (
                predicted[:, 2:] - 2 * predicted[:, 1:-1] + predicted[:, :-2]
            ).square().mean()
            # The architecture already guarantees direction; smoothness keeps
            # small residual increments from producing an implausibly jagged path.
            loss = fit_loss + 0.01 * smooth_loss
            if not torch.isfinite(loss):
                raise RuntimeError("hybrid trajectory training produced a non-finite loss")
            loss.backward()
            optimizer.step()

        network.eval()
        self._network = network
        self._context = _FittedContext(
            dataset_id=split_manifest.dataset_id,
            condition_feature_names=self.condition_feature_names,
        )
        return self

    def predict(
        self,
        *,
        dataset_id: str,
        cell_id: str,
        observed_cycles: Sequence[int],
        observed_soh: Sequence[float],
        condition_features: Mapping[str, float],
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> SOHTrajectoryPrediction:
        """Predict a finite, monotone trajectory from cutoff-safe early inputs only."""

        if self._network is None or self._context is None:
            raise RuntimeError("HybridDegradationPredictor must be fitted before prediction")
        self._require_prediction_context(
            cutoff_cycle=cutoff_cycle,
            feature_version=feature_version,
            split_version=split_version,
            data_version=data_version,
        )
        if dataset_id != self._context.dataset_id:
            raise ValueError("dataset_id must match the fitted trajectory-model context")
        if not cell_id:
            raise ValueError("cell_id must be non-empty")
        input_features, initial_soh = self._validated_prediction_inputs(
            observed_cycles=observed_cycles,
            observed_soh=observed_soh,
            condition_features=condition_features,
        )
        features = torch.tensor([input_features], dtype=torch.float32, device="cpu")
        initial = torch.tensor([initial_soh], dtype=torch.float32, device="cpu")
        cycle_positions = torch.tensor(
            normalise_prediction_cycle_positions(
                prediction_cycles=self.prediction_cycles,
                cutoff_cycle=self.cutoff_cycle,
            ),
            dtype=torch.float32,
            device="cpu",
        )
        with torch.no_grad():
            values = self._network(features, initial, cycle_positions).squeeze(0).tolist()
        predicted_soh = tuple(float(value) for value in values)
        crossing = derive_eol80_crossing(
            prediction_cycles=self.prediction_cycles,
            predicted_soh=predicted_soh,
            cutoff_cycle=self.cutoff_cycle,
        )
        return SOHTrajectoryPrediction(
            dataset_id=dataset_id,
            cell_id=cell_id,
            cutoff_cycle=self.cutoff_cycle,
            prediction_cycles=self.prediction_cycles,
            predicted_soh=predicted_soh,
            eol80_crossing=crossing,
            feature_version=self.feature_version,
            split_version=self.split_version,
            model_version=self.model_version,
            data_version=self.data_version,
        )

    def _validate_training_sample(
        self, sample: TrajectoryTrainingSample, *, split_manifest: SplitManifest
    ) -> None:
        if sample.dataset_id != split_manifest.dataset_id:
            raise ValueError("training sample dataset_id must match split manifest")
        for name, received, expected in (
            ("cutoff_cycle", sample.cutoff_cycle, self.cutoff_cycle),
            ("feature_version", sample.feature_version, self.feature_version),
            ("data_version", sample.data_version, self.data_version),
        ):
            if received != expected:
                raise ValueError(f"training sample {name} must match the predictor context")
        if sample.target_cycles != self.prediction_cycles:
            raise ValueError("training target_cycles must exactly match prediction_cycles")
        if sample.observed_soh[-1] <= EOL80_THRESHOLD:
            raise ValueError("observed SOH at cutoff already reached EOL80")
        self._validated_condition_features(sample.condition_features)

    def _validated_prediction_inputs(
        self,
        *,
        observed_cycles: Sequence[int],
        observed_soh: Sequence[float],
        condition_features: Mapping[str, float],
    ) -> tuple[list[float], float]:
        if len(observed_cycles) != len(observed_soh) or not observed_cycles:
            raise ValueError("observed_cycles and observed_soh must be non-empty and align")
        normalised_cycles = tuple(int(cycle) for cycle in observed_cycles)
        if any(cycle < 0 for cycle in normalised_cycles):
            raise ValueError("observed_cycles must be non-negative")
        _strictly_increasing(normalised_cycles, field_name="observed_cycles")
        if any(cycle > self.cutoff_cycle for cycle in normalised_cycles):
            raise ValueError("observed_cycles cannot exceed cutoff_cycle")
        if normalised_cycles[-1] != self.cutoff_cycle:
            raise ValueError("the last observed cycle must equal cutoff_cycle")
        normalised_soh: list[float] = []
        for value in observed_soh:
            numeric_value = _finite_real(value, field_name="observed SOH")
            if numeric_value <= 0 or numeric_value > 1.5:
                raise ValueError("observed SOH values must be in (0, 1.5]")
            normalised_soh.append(numeric_value)
        if normalised_soh[-1] <= EOL80_THRESHOLD:
            raise ValueError("observed SOH at cutoff already reached EOL80")
        self._validated_condition_features(condition_features)
        return (
            self._build_features(normalised_cycles, normalised_soh, condition_features),
            normalised_soh[-1],
        )

    def _validated_condition_features(self, condition_features: Mapping[str, float]) -> None:
        if set(condition_features) != set(self.condition_feature_names):
            raise ValueError("condition feature schema must exactly match condition_feature_names")
        for name in self.condition_feature_names:
            _finite_real(condition_features[name], field_name=f"condition feature {name}")

    def _build_features(
        self,
        observed_cycles: Sequence[int],
        observed_soh: Sequence[float],
        condition_features: Mapping[str, float],
    ) -> list[float]:
        """Use only observed cutoff-safe SOH history and declared conditions."""

        if len(observed_cycles) != len(observed_soh):
            raise ValueError("observed cycles and SOH must align")
        first_cycle, last_cycle = observed_cycles[0], observed_cycles[-1]
        first_soh, last_soh = float(observed_soh[0]), float(observed_soh[-1])
        elapsed_cycles = max(last_cycle - first_cycle, 1)
        history_drop = first_soh - last_soh
        history_slope = history_drop / float(elapsed_cycles)
        return [
            last_soh,
            history_drop,
            history_slope,
            *[
                _finite_real(condition_features[name], field_name=f"condition feature {name}")
                for name in self.condition_feature_names
            ],
        ]

    def _require_prediction_context(
        self,
        *,
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> None:
        for name, received, expected in (
            ("cutoff_cycle", cutoff_cycle, self.cutoff_cycle),
            ("feature_version", feature_version, self.feature_version),
            ("split_version", split_version, self.split_version),
            ("data_version", data_version, self.data_version),
        ):
            if received != expected:
                raise ValueError(f"{name} must match the fitted hybrid-model context")

    @staticmethod
    def _set_deterministic_seed() -> None:
        random.seed(HYBRID_RANDOM_SEED)
        np.random.seed(HYBRID_RANDOM_SEED)
        torch.manual_seed(HYBRID_RANDOM_SEED)
        torch.use_deterministic_algorithms(True, warn_only=False)
