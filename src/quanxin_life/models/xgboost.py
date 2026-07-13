"""Leakage-safe XGBoost EOL80 baseline on one finite feature row per cell.

The predictor deliberately keeps the fitted estimator in memory only.  Model
artifact publication belongs to the governed training pipeline and must not
use Python pickle or joblib serialization.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Self

from xgboost import XGBRegressor

from quanxin_life.core import LifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest

XGBOOST_RANDOM_STATE = 20260712


@dataclass
class XGBoostLifePredictor:
    """Fit an EOL80 tree baseline with an explicitly frozen feature schema.

    The model accepts exactly one finite feature mapping for every supplied
    observed training label.  It never fills missing values, and it refuses
    labels or features from outside the declared train-cell cohort.
    """

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    feature_names: tuple[str, ...]
    _estimator: XGBRegressor | None = field(default=None, init=False, repr=False)
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
        if not self.feature_names or any(not name for name in self.feature_names):
            raise ValueError("feature_names must contain at least one non-empty name")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be unique")

    def fit(
        self,
        training_labels: Sequence[LifePrediction],
        *,
        training_features: Mapping[str, Mapping[str, float]],
        split_manifest: SplitManifest,
    ) -> Self:
        """Fit only from observed train-cell EOL80 labels and finite feature rows."""

        if not training_labels:
            raise ValueError("at least one explicit observed EOL80 training label is required")

        seen_cell_ids: set[str] = set()
        target_values: list[float] = []
        for label in training_labels:
            self._validate_training_label(
                label, split_manifest=split_manifest, seen_cell_ids=seen_cell_ids
            )
            assert label.observed_eol_cycle is not None
            target_values.append(float(label.observed_eol_cycle))

        self._validate_training_feature_cells(
            training_features=training_features, expected_cell_ids=seen_cell_ids
        )
        feature_matrix = [
            self._validated_feature_row(training_features[label.cell_id])
            for label in training_labels
        ]

        estimator = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=64,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=XGBOOST_RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
        estimator.fit(feature_matrix, target_values)
        self._estimator = estimator
        self._dataset_id = split_manifest.dataset_id
        return self

    def predict(
        self,
        *,
        dataset_id: str,
        cell_id: str,
        features: Mapping[str, float],
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> LifePrediction:
        """Predict an EOL80 cycle for one schema-valid feature row."""

        if self._estimator is None or self._dataset_id is None:
            raise RuntimeError("XGBoostLifePredictor must be fitted before prediction")
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
        validated_feature_row = self._validated_feature_row(features)
        predicted_eol_cycle = float(self._estimator.predict([validated_feature_row])[0])
        if not math.isfinite(predicted_eol_cycle):
            raise RuntimeError("XGBoost produced a non-finite EOL80 prediction")
        if predicted_eol_cycle < self.cutoff_cycle:
            raise RuntimeError("XGBoost produced an EOL80 prediction before the cutoff cycle")

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

    def _validate_training_feature_cells(
        self,
        *,
        training_features: Mapping[str, Mapping[str, float]],
        expected_cell_ids: set[str],
    ) -> None:
        actual_cell_ids = set(training_features)
        if actual_cell_ids != expected_cell_ids:
            raise ValueError(
                "training feature cell_ids must exactly match observed training label cell_ids"
            )
        for cell_id in sorted(expected_cell_ids):
            self._validated_feature_row(training_features[cell_id])

    def _validated_feature_row(self, features: Mapping[str, float]) -> list[float]:
        if set(features) != set(self.feature_names):
            raise ValueError("feature schema must exactly match declared feature_names")

        row: list[float] = []
        for feature_name in self.feature_names:
            value: Any = features[feature_name]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"feature {feature_name} must be a finite real number")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"feature {feature_name} must be finite")
            row.append(numeric_value)
        return row

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
                raise ValueError(f"{name} must match the fitted XGBoost context")
