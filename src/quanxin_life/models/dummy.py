"""Leakage-safe EOL80 mean baseline operating on cell-level training labels only."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Self

from quanxin_life.core import LifePrediction, PredictionTarget
from quanxin_life.data.schemas import SplitManifest


@dataclass
class DummyLifePredictor:
    """Predict the mean observed EOL80 cycle from explicitly supplied train cells.

    The baseline intentionally does not accept feature rows.  Fitting requires
    the canonical cell-level split manifest so a caller cannot accidentally
    include validation, calibration, or test-cell labels in the fitted mean.
    """

    model_version: str
    feature_version: str
    split_version: str
    data_version: str
    cutoff_cycle: int
    _mean_eol_cycle: float | None = field(default=None, init=False, repr=False)
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

    def fit(
        self, training_labels: Sequence[LifePrediction], *, split_manifest: SplitManifest
    ) -> Self:
        """Fit only from observed EOL80 labels assigned to ``split_manifest.train``."""

        if not training_labels:
            raise ValueError("at least one explicit observed EOL80 training label is required")

        observed_eol_cycles: list[int] = []
        seen_cell_ids: set[str] = set()
        for label in training_labels:
            self._validate_training_label(
                label, split_manifest=split_manifest, seen_cell_ids=seen_cell_ids
            )
            assert label.observed_eol_cycle is not None
            observed_eol_cycles.append(label.observed_eol_cycle)

        self._mean_eol_cycle = float(fmean(observed_eol_cycles))
        self._dataset_id = split_manifest.dataset_id
        return self

    def predict(
        self,
        *,
        dataset_id: str,
        cell_id: str,
        cutoff_cycle: int,
        feature_version: str,
        split_version: str,
        data_version: str,
    ) -> LifePrediction:
        """Return an EOL80 prediction using only the fitted train-cell mean."""

        if self._mean_eol_cycle is None or self._dataset_id is None:
            raise RuntimeError("DummyLifePredictor must be fitted before prediction")
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

        return LifePrediction(
            dataset_id=dataset_id,
            cell_id=cell_id,
            cutoff_cycle=cutoff_cycle,
            target=PredictionTarget.EOL80_CYCLE,
            predicted_eol_cycle=self._mean_eol_cycle,
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
                raise ValueError(f"{name} must match the fitted baseline context")
