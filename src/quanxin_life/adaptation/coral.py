"""Cell-disjoint CORAL feature alignment without target-test leakage.

CORAL aligns second-order statistics of source-train features to an explicitly
provided target-adaptation train cohort.  The adapter is an in-memory research
component: it does not load weights, fit labels, or receive target test cells.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import cast

import numpy as np
from numpy.typing import NDArray

from quanxin_life.data.schemas import SplitManifest

FloatMatrix = NDArray[np.float64]


def _finite_feature(value: object, *, feature_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"feature {feature_name} must be finite")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"feature {feature_name} must be finite")
    return numeric_value


def _covariance(matrix: FloatMatrix, *, regularization: float) -> FloatMatrix:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("CORAL requires at least two cell-level feature rows per domain")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / float(matrix.shape[0] - 1)
    return cast(
        FloatMatrix,
        covariance + regularization * np.eye(matrix.shape[1], dtype=np.float64),
    )


def _symmetric_matrix_power(matrix: FloatMatrix, *, exponent: float) -> FloatMatrix:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues <= 0):
        raise ValueError("regularized covariance must be positive definite")
    return cast(FloatMatrix, (eigenvectors * np.power(eigenvalues, exponent)) @ eigenvectors.T)


@dataclass(frozen=True)
class _CORALState:
    source_dataset_id: str
    target_dataset_id: str
    source_mean: FloatMatrix
    target_mean: FloatMatrix
    source_whitener: FloatMatrix
    target_colorer: FloatMatrix


@dataclass
class CORALFeatureAdapter:
    """Fit CORAL only on exact source/target train cell cohorts."""

    adapter_version: str
    feature_version: str
    source_split_version: str
    target_split_version: str
    feature_names: tuple[str, ...]
    covariance_regularization: float = 1e-4
    _state: _CORALState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("adapter_version", self.adapter_version),
            ("feature_version", self.feature_version),
            ("source_split_version", self.source_split_version),
            ("target_split_version", self.target_split_version),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be non-empty and unique")
        if self.covariance_regularization <= 0 or not math.isfinite(
            self.covariance_regularization
        ):
            raise ValueError("covariance_regularization must be finite and positive")

    @property
    def source_dataset_id(self) -> str:
        if self._state is None:
            raise RuntimeError(
                "CORALFeatureAdapter must be fitted before reading source_dataset_id"
            )
        return self._state.source_dataset_id

    @property
    def target_dataset_id(self) -> str:
        if self._state is None:
            raise RuntimeError(
                "CORALFeatureAdapter must be fitted before reading target_dataset_id"
            )
        return self._state.target_dataset_id

    def fit(
        self,
        *,
        source_features: Mapping[str, Mapping[str, float]],
        source_split_manifest: SplitManifest,
        target_adaptation_features: Mapping[str, Mapping[str, float]],
        target_split_manifest: SplitManifest,
    ) -> CORALFeatureAdapter:
        """Fit source-to-target alignment from exact train-only cell feature maps."""

        source_matrix = self._cohort_matrix(
            source_features,
            expected_cell_ids=source_split_manifest.train,
            expected_message="must exactly match source split_manifest.train",
        )
        target_matrix = self._cohort_matrix(
            target_adaptation_features,
            expected_cell_ids=target_split_manifest.train,
            expected_message="must exactly match target split_manifest.train",
        )
        source_covariance = _covariance(
            source_matrix, regularization=self.covariance_regularization
        )
        target_covariance = _covariance(
            target_matrix, regularization=self.covariance_regularization
        )
        self._state = _CORALState(
            source_dataset_id=source_split_manifest.dataset_id,
            target_dataset_id=target_split_manifest.dataset_id,
            source_mean=source_matrix.mean(axis=0),
            target_mean=target_matrix.mean(axis=0),
            source_whitener=_symmetric_matrix_power(source_covariance, exponent=-0.5),
            target_colorer=_symmetric_matrix_power(target_covariance, exponent=0.5),
        )
        return self

    def transform_source_features(
        self, source_features: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        """Align finite source-domain feature rows using the frozen CORAL state."""

        if self._state is None:
            raise RuntimeError("CORALFeatureAdapter must be fitted before transformation")
        if not source_features:
            raise ValueError("source_features must be non-empty")
        ordered_cell_ids = tuple(sorted(source_features))
        matrix = self._matrix_from_mapping(source_features, ordered_cell_ids)
        transformed = (
            (matrix - self._state.source_mean)
            @ self._state.source_whitener
            @ self._state.target_colorer
            + self._state.target_mean
        )
        if not np.all(np.isfinite(transformed)):
            raise RuntimeError("CORAL transformation produced non-finite feature values")
        return {
            cell_id: {
                feature_name: float(transformed[row_index, column_index])
                for column_index, feature_name in enumerate(self.feature_names)
            }
            for row_index, cell_id in enumerate(ordered_cell_ids)
        }

    def _cohort_matrix(
        self,
        feature_mapping: Mapping[str, Mapping[str, float]],
        *,
        expected_cell_ids: tuple[str, ...],
        expected_message: str,
    ) -> FloatMatrix:
        if set(feature_mapping) != set(expected_cell_ids):
            raise ValueError(f"feature cell_ids {expected_message}")
        return self._matrix_from_mapping(feature_mapping, tuple(sorted(expected_cell_ids)))

    def _matrix_from_mapping(
        self,
        feature_mapping: Mapping[str, Mapping[str, float]],
        ordered_cell_ids: tuple[str, ...],
    ) -> FloatMatrix:
        rows: list[list[float]] = []
        for cell_id in ordered_cell_ids:
            features = feature_mapping.get(cell_id)
            if features is None:
                raise ValueError(f"missing feature row for cell_id {cell_id}")
            if set(features) != set(self.feature_names):
                raise ValueError("feature schema must exactly match feature_names")
            rows.append(
                [_finite_feature(features[name], feature_name=name) for name in self.feature_names]
            )
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix has an invalid shape")
        return matrix
