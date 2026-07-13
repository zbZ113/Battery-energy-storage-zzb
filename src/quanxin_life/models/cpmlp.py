"""Masked, cell-level CPMLP EOL80 baseline.

This module independently implements the two-stage MLP idea used for
cycle-capacity curves.  It deliberately has no deserialization or persistence
path: only in-memory tensors built from :class:`CurveTensor` instances are
accepted.  Missing curves remain ``NaN`` at the conversion boundary and are
only replaced after their explicit mask has been applied inside the network.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from quanxin_life.core import LifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.curve_tensor import CurveTensor

CPMLP_RANDOM_SEED = 20260712


@dataclass(frozen=True)
class _CurveSchema:
    """The axis contract learned from training curve tensors."""

    cycle_indices: tuple[int, ...]
    voltage_grid_v: tuple[float, ...]


def curve_tensor_to_tensors(curve: CurveTensor) -> tuple[Tensor, Tensor]:
    """Convert one curve tensor to ``NaN``-preserving CPU tensors.

    ``CurveTensor`` encodes absent rows as ``None`` plus ``observed_mask``.
    This adapter makes absence explicit as ``NaN`` rather than capacity zero so
    an accidental mask omission is observable and rejected by the network.
    """

    values: list[list[float]] = []
    for row, observed in zip(curve.values, curve.observed_mask, strict=True):
        converted_row = [float("nan") if value is None else float(value) for value in row]
        if observed and any(not math.isfinite(value) for value in converted_row):
            raise ValueError("non-finite values in observed curve row")
        if not observed and any(not math.isnan(value) for value in converted_row):
            raise ValueError("unobserved curve rows must remain missing rather than numeric")
        values.append(converted_row)

    curve_values = torch.tensor(values, dtype=torch.float32, device="cpu")
    observed_mask = torch.tensor(curve.observed_mask, dtype=torch.bool, device="cpu")
    if int(observed_mask.sum().item()) == 0:
        raise ValueError("curve tensor requires at least one observed curve")
    return curve_values, observed_mask


class _CPMLPNetwork(nn.Module):
    """Cycle encoder, masked temporal aggregation and constrained EOL80 head."""

    def __init__(
        self,
        *,
        voltage_points: int,
        cutoff_cycle: int,
        curve_hidden_dim: int,
        aggregation_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.cutoff_cycle = cutoff_cycle
        self.cycle_encoder = nn.Sequential(
            nn.Linear(voltage_points, curve_hidden_dim),
            nn.ReLU(),
            nn.Linear(curve_hidden_dim, curve_hidden_dim),
            nn.ReLU(),
        )
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(curve_hidden_dim + 1, aggregation_hidden_dim),
            nn.ReLU(),
            nn.Linear(aggregation_hidden_dim, aggregation_hidden_dim),
            nn.ReLU(),
        )
        self.eol_head = nn.Linear(aggregation_hidden_dim, 1)

    def forward(self, curve_values: Tensor, observed_mask: Tensor) -> Tensor:
        """Return one EOL80 estimate per sample, structurally bounded by cutoff."""

        if curve_values.ndim != 3:
            raise ValueError("curve_values must have shape [batch, cycle, voltage]")
        if observed_mask.ndim != 2 or observed_mask.shape != curve_values.shape[:2]:
            raise ValueError("observed_mask must match the batch and cycle axes")
        if observed_mask.dtype is not torch.bool:
            raise ValueError("observed_mask must have torch.bool dtype")
        if torch.any(observed_mask.sum(dim=1) == 0):
            raise ValueError("each sample requires at least one observed curve")

        observed_values = curve_values[observed_mask]
        if not torch.isfinite(observed_values).all():
            raise ValueError("observed curve values must be finite")
        absent_values = curve_values[~observed_mask]
        if absent_values.numel() and not torch.isnan(absent_values).all():
            raise ValueError("unobserved curve values must remain NaN until mask application")

        # The replacement occurs only after the mask has established that the
        # corresponding complete curve is absent.  Zero is therefore neutral
        # padding, never an observed zero-capacity measurement.
        masked_values = torch.where(
            observed_mask.unsqueeze(-1), curve_values, torch.zeros_like(curve_values)
        )
        encoded_cycles = self.cycle_encoder(masked_values)
        cycle_weights = observed_mask.unsqueeze(-1).to(dtype=encoded_cycles.dtype)
        observed_counts = cycle_weights.sum(dim=1).clamp_min(1.0)
        pooled = (encoded_cycles * cycle_weights).sum(dim=1) / observed_counts
        coverage = observed_counts / float(curve_values.shape[1])
        aggregated = self.aggregation_mlp(torch.cat((pooled, coverage), dim=1))
        raw_eol_distance = self.eol_head(aggregated).squeeze(dim=1)
        return float(self.cutoff_cycle) + functional.softplus(raw_eol_distance)


@dataclass
class CPMLPLifePredictor:
    """Fit a masked two-level MLP only from observed train-cell EOL80 labels."""

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    curve_hidden_dim: int = 32
    aggregation_hidden_dim: int = 32
    epochs: int = 200
    learning_rate: float = 0.001
    _network: _CPMLPNetwork | None = field(default=None, init=False, repr=False)
    _dataset_id: str | None = field(default=None, init=False, repr=False)
    _curve_schema: _CurveSchema | None = field(default=None, init=False, repr=False)

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
        if self.curve_hidden_dim <= 0 or self.aggregation_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")

    def fit(
        self,
        training_labels: Sequence[LifePrediction],
        *,
        training_curves: Mapping[str, CurveTensor],
        split_manifest: SplitManifest,
    ) -> Self:
        """Fit in memory from a cell-disjoint, explicit EOL80 training cohort."""

        if not training_labels:
            raise ValueError("at least one explicit observed EOL80 training label is required")
        self._seed_everything()

        seen_cell_ids: set[str] = set()
        target_values: list[float] = []
        for label in training_labels:
            self._validate_training_label(
                label, split_manifest=split_manifest, seen_cell_ids=seen_cell_ids
            )
            assert label.observed_eol_cycle is not None
            target_values.append(float(label.observed_eol_cycle))

        if set(training_curves) != seen_cell_ids:
            raise ValueError(
                "training curve cell_ids must exactly match observed training label cell_ids"
            )

        reference_schema: _CurveSchema | None = None
        curve_values_batch: list[Tensor] = []
        observed_mask_batch: list[Tensor] = []
        for label in training_labels:
            curve = training_curves[label.cell_id]
            schema = self._validate_curve_context(
                curve,
                expected_dataset_id=split_manifest.dataset_id,
                expected_cell_id=label.cell_id,
            )
            if reference_schema is None:
                reference_schema = schema
            elif schema != reference_schema:
                raise ValueError(
                    "training curve cycle_indices and voltage_grid_v must be identical"
                )
            values, mask = curve_tensor_to_tensors(curve)
            curve_values_batch.append(values)
            observed_mask_batch.append(mask)

        assert reference_schema is not None
        curve_values = torch.stack(curve_values_batch, dim=0)
        observed_mask = torch.stack(observed_mask_batch, dim=0)
        targets = torch.tensor(target_values, dtype=torch.float32, device="cpu")
        network = _CPMLPNetwork(
            voltage_points=len(reference_schema.voltage_grid_v),
            cutoff_cycle=self.cutoff_cycle,
            curve_hidden_dim=self.curve_hidden_dim,
            aggregation_hidden_dim=self.aggregation_hidden_dim,
        ).to(device="cpu")
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)

        network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            predicted = network(curve_values, observed_mask)
            loss = functional.mse_loss(predicted, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("CPMLP training produced a non-finite loss")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        network.eval()
        self._network = network
        self._dataset_id = split_manifest.dataset_id
        self._curve_schema = reference_schema
        return self

    def predict(
        self,
        *,
        curve: CurveTensor,
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> LifePrediction:
        """Predict EOL80 for one schema-compatible, masked curve tensor."""

        if self._network is None or self._dataset_id is None or self._curve_schema is None:
            raise RuntimeError("CPMLPLifePredictor must be fitted before prediction")
        self._require_prediction_context(
            cutoff_cycle=cutoff_cycle,
            feature_version=feature_version,
            split_version=split_version,
            data_version=data_version,
        )
        schema = self._validate_curve_context(
            curve,
            expected_dataset_id=self._dataset_id,
            expected_cell_id=curve.cell_id,
        )
        if schema != self._curve_schema:
            raise ValueError("curve cycle_indices and voltage_grid_v must match the fitted schema")
        values, mask = curve_tensor_to_tensors(curve)
        with torch.no_grad():
            predicted_eol_cycle = float(
                self._network(values.unsqueeze(0), mask.unsqueeze(0)).item()
            )
        if not math.isfinite(predicted_eol_cycle):
            raise RuntimeError("CPMLP produced a non-finite EOL80 prediction")
        if predicted_eol_cycle < self.cutoff_cycle:
            raise RuntimeError("CPMLP produced an EOL80 prediction before the cutoff cycle")

        return LifePrediction(
            dataset_id=curve.dataset_id,
            cell_id=curve.cell_id,
            cutoff_cycle=cutoff_cycle,
            target=PredictionTarget.EOL80_CYCLE,
            predicted_eol_cycle=predicted_eol_cycle,
            observed_eol_cycle=None,
            right_censored=True,
            feature_version=feature_version,
            split_version=split_version,
            model_version=self.model_version,
            data_version=data_version,
        )

    def _validate_training_label(
        self,
        label: LifePrediction,
        *,
        split_manifest: SplitManifest,
        seen_cell_ids: set[str],
    ) -> None:
        if label.cell_id in seen_cell_ids:
            raise ValueError(f"duplicate training cell_id: {label.cell_id}")
        seen_cell_ids.add(label.cell_id)
        if label.dataset_id != split_manifest.dataset_id:
            raise ValueError("training label dataset_id must match the split manifest")
        if label.cell_id not in split_manifest.train:
            raise ValueError(f"training label cell_id is outside the train split: {label.cell_id}")
        if label.target is not PredictionTarget.EOL80_CYCLE:
            raise ValueError("training label target must be EOL80_CYCLE")
        if label.right_censored or label.observed_eol_cycle is None:
            raise ValueError("training labels require an explicit observed EOL80 cycle")
        self._require_prediction_context(
            cutoff_cycle=label.cutoff_cycle,
            feature_version=label.feature_version,
            split_version=label.split_version,
            data_version=label.data_version,
        )

    def _validate_curve_context(
        self,
        curve: CurveTensor,
        *,
        expected_dataset_id: str,
        expected_cell_id: str,
    ) -> _CurveSchema:
        if curve.dataset_id != expected_dataset_id:
            raise ValueError("curve dataset_id must match the fitted training dataset")
        if curve.cell_id != expected_cell_id:
            raise ValueError("curve cell_id must match its declared training cell")
        if curve.cutoff_cycle != self.cutoff_cycle:
            raise ValueError("curve cutoff_cycle must match the CPMLP context")
        if curve.feature_version != self.feature_version:
            raise ValueError("curve feature_version must match the CPMLP context")
        if not curve.voltage_grid_v:
            raise ValueError("curve voltage_grid_v must contain at least one point")
        return _CurveSchema(
            cycle_indices=curve.cycle_indices,
            voltage_grid_v=curve.voltage_grid_v,
        )

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
                raise ValueError(f"{name} must match the fitted CPMLP context")

    @staticmethod
    def _seed_everything() -> None:
        random.seed(CPMLP_RANDOM_SEED)
        np.random.seed(CPMLP_RANDOM_SEED)
        torch.manual_seed(CPMLP_RANDOM_SEED)
        torch.use_deterministic_algorithms(True)
