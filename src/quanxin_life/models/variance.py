"""Leakage-safe classic log-log Delta-Q variance baseline for EOL80.

The implementation follows the interpretable one-feature baseline introduced
for early-cycle lifetime studies: both the Delta-Q variance and observed EOL80
cycle are transformed with ``log10`` before fitting an ordinary least-squares
line.  It keeps the fitted coefficients in memory and never serializes through
pickle or joblib.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Self

import numpy as np

from quanxin_life.core import LifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.variance import DeltaQVarianceFeature


@dataclass
class VarianceLifePredictor:
    """Fit EOL80 from one positive Delta-Q variance feature per train cell."""

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    anchor_cycle: int
    comparison_cycle: int
    _slope: float | None = field(default=None, init=False, repr=False)
    _intercept: float | None = field(default=None, init=False, repr=False)
    _dataset_id: str | None = field(default=None, init=False, repr=False)

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
        if self.anchor_cycle < 0:
            raise ValueError("anchor_cycle must be non-negative")
        if self.anchor_cycle >= self.comparison_cycle:
            raise ValueError("anchor_cycle must be earlier than comparison_cycle")
        if self.comparison_cycle > self.cutoff_cycle:
            raise ValueError("comparison_cycle must not exceed cutoff_cycle")

    def fit(
        self,
        training_labels: Sequence[LifePrediction],
        *,
        training_features: Sequence[DeltaQVarianceFeature],
        split_manifest: SplitManifest,
    ) -> Self:
        """Fit a log-log line from cell-disjoint observed training labels."""

        if not training_labels:
            raise ValueError("at least one explicit observed EOL80 training label is required")

        labels_by_cell: dict[str, LifePrediction] = {}
        for label in training_labels:
            if label.cell_id in labels_by_cell:
                raise ValueError(f"duplicate training cell_id: {label.cell_id}")
            self._validate_training_label(label, split_manifest=split_manifest)
            labels_by_cell[label.cell_id] = label

        features_by_cell: dict[str, DeltaQVarianceFeature] = {}
        for feature in training_features:
            if feature.cell_id in features_by_cell:
                raise ValueError(f"duplicate training feature cell_id: {feature.cell_id}")
            features_by_cell[feature.cell_id] = feature

        if set(features_by_cell) != set(labels_by_cell):
            raise ValueError(
                "training feature cell_ids must exactly match observed training label cell_ids"
            )

        x_values: list[float] = []
        y_values: list[float] = []
        for cell_id, label in labels_by_cell.items():
            feature = features_by_cell[cell_id]
            self._validate_feature(feature, dataset_id=split_manifest.dataset_id, cell_id=cell_id)
            assert label.observed_eol_cycle is not None
            if label.observed_eol_cycle <= 0:
                raise ValueError("observed EOL80 cycle must be strictly positive for log10")
            variance = self._positive_variance(feature)
            x_values.append(math.log10(variance))
            y_values.append(math.log10(float(label.observed_eol_cycle)))

        if len(x_values) < 2 or len(set(x_values)) < 2:
            raise ValueError("at least two distinct log variance values are required")

        x_array = np.asarray(x_values, dtype=np.float64)
        design = np.column_stack((x_array, np.ones(len(x_values), dtype=np.float64)))
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design, np.asarray(y_values, dtype=np.float64), rcond=None
        )
        machine_epsilon = np.finfo(np.float64).eps
        relative_precision = math.sqrt(machine_epsilon)
        centered_x_norm = float(np.linalg.norm(x_array - np.mean(x_array)))
        x_scale = max(1.0, float(np.linalg.norm(x_array)))
        if rank < 2 or centered_x_norm <= relative_precision * x_scale:
            raise ValueError(
                "insufficient numeric variation in log variance values for stable regression"
            )
        smallest_singular_value = float(singular_values[-1])
        condition_number = (
            math.inf
            if smallest_singular_value <= 0
            else float(singular_values[0]) / smallest_singular_value
        )
        if not math.isfinite(condition_number) or condition_number >= 1.0 / relative_precision:
            raise ValueError("ill-conditioned log variance regression design matrix")
        slope, intercept = (float(coefficients[0]), float(coefficients[1]))
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise RuntimeError("variance regression produced non-finite fitted coefficients")

        self._slope = slope
        self._intercept = intercept
        self._dataset_id = split_manifest.dataset_id
        return self

    def predict(
        self,
        *,
        feature: DeltaQVarianceFeature,
        dataset_id: str,
        cell_id: str,
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> LifePrediction:
        """Predict one EOL80 cycle from a context-bound variance feature."""

        if self._slope is None or self._intercept is None or self._dataset_id is None:
            raise RuntimeError("VarianceLifePredictor must be fitted before prediction")
        if not dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if not cell_id:
            raise ValueError("cell_id must be non-empty")
        if dataset_id != self._dataset_id:
            raise ValueError("dataset_id must match the fitted training dataset")
        self._require_prediction_context(
            cutoff_cycle=cutoff_cycle,
            feature_version=feature_version,
            split_version=split_version,
            data_version=data_version,
        )
        self._validate_feature(feature, dataset_id=dataset_id, cell_id=cell_id)
        variance = self._positive_variance(feature)
        predicted_log_eol = self._slope * math.log10(variance) + self._intercept
        try:
            predicted_eol_cycle = math.pow(10.0, predicted_log_eol)
        except OverflowError as exc:
            raise RuntimeError(
                "variance regression produced a non-finite EOL80 prediction"
            ) from exc
        if not math.isfinite(predicted_eol_cycle):
            raise RuntimeError("variance regression produced a non-finite EOL80 prediction")
        if predicted_eol_cycle < cutoff_cycle:
            raise RuntimeError("variance regression produced an EOL80 prediction before the cutoff")

        return LifePrediction(
            dataset_id=dataset_id,
            cell_id=cell_id,
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
        self, label: LifePrediction, *, split_manifest: SplitManifest
    ) -> None:
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

    def _validate_feature(
        self, feature: DeltaQVarianceFeature, *, dataset_id: str, cell_id: str
    ) -> None:
        for name, received, expected in (
            ("dataset_id", feature.dataset_id, dataset_id),
            ("cell_id", feature.cell_id, cell_id),
            ("cutoff_cycle", feature.cutoff_cycle, self.cutoff_cycle),
            ("feature_version", feature.feature_version, self.feature_version),
            ("anchor_cycle", feature.anchor_cycle, self.anchor_cycle),
            ("comparison_cycle", feature.comparison_cycle, self.comparison_cycle),
        ):
            if received != expected:
                raise ValueError(f"{name} must match the fitted variance context")

    @staticmethod
    def _positive_variance(feature: DeltaQVarianceFeature) -> float:
        value = feature.delta_q_variance_ah2
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError("delta_q_variance_ah2 must be finite and strictly positive")
        return float(value)

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
                raise ValueError(f"{name} must match the fitted variance context")
