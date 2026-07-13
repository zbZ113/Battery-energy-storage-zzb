"""Leakage-safe CPMLP-DANN domain adaptation for early-cycle curves.

The source cohort must be exactly the source manifest's train cells.  The
unlabelled target cohort is either the target manifest's train cells (used as
the named adaptation cohort) or its independent calibration cells.  Target
test cells can be predicted after fitting, but cannot enter normalization,
adaptation, or any optimisation step.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from quanxin_life.data.schemas import SplitManifest

DANN_RANDOM_SEED = 20260712
SOURCE_TRAIN_ZSCORE_NORMALIZATION_VERSION = "source-train-zscore-v1"


class TargetDomainCohort(StrEnum):
    """Authorized unlabelled target cohorts for adaptation.

    ``ADAPTATION`` intentionally resolves to ``SplitManifest.train`` because
    the canonical manifest has no separate adaptation field.  That target
    train partition is reserved for unlabelled adaptation only; it is never a
    source of life labels.  ``CALIBRATION`` resolves to the independent target
    calibration partition.
    """

    ADAPTATION = "adaptation"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class CurveFeatureContract:
    """Immutable early-cycle curve axes and feature version for one DANN run."""

    feature_version: str
    cutoff_cycle: int
    cycle_indices: tuple[int, ...]
    voltage_grid_v: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.feature_version:
            raise ValueError("feature_version must be non-empty")
        if (
            isinstance(self.cutoff_cycle, bool)
            or not isinstance(self.cutoff_cycle, int)
            or self.cutoff_cycle < 0
        ):
            raise ValueError("cutoff_cycle must be a non-negative integer")
        if not self.cycle_indices:
            raise ValueError("cycle_indices must be non-empty")
        if any(
            isinstance(cycle_index, bool) or not isinstance(cycle_index, int)
            for cycle_index in self.cycle_indices
        ):
            raise ValueError("cycle_indices must contain finite integer values")
        if any(
            later <= earlier
            for earlier, later in zip(self.cycle_indices, self.cycle_indices[1:], strict=False)
        ):
            raise ValueError("cycle_indices must be strictly increasing")
        if self.cycle_indices[-1] != self.cutoff_cycle:
            raise ValueError("cycle_indices must end at cutoff_cycle")
        if not self.voltage_grid_v:
            raise ValueError("voltage_grid_v must be non-empty")
        if any(not math.isfinite(voltage) for voltage in self.voltage_grid_v):
            raise ValueError("voltage_grid_v must contain finite values")
        if any(
            later <= earlier
            for earlier, later in zip(self.voltage_grid_v, self.voltage_grid_v[1:], strict=False)
        ):
            raise ValueError("voltage_grid_v must be strictly increasing")


@dataclass(frozen=True)
class DANNConfig:
    """Versioned, deterministic hyperparameters for one CPMLP-DANN run."""

    model_version: str
    feature_version: str
    source_split_version: str
    target_split_version: str
    random_seed: int = DANN_RANDOM_SEED
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.001
    curve_hidden_dim: int = 32
    representation_dim: int = 32
    domain_hidden_dim: int = 16
    gradient_reversal_coefficient: float = 1.0
    domain_loss_weight: float = 0.2
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        for name, value in {
            "model_version": self.model_version,
            "feature_version": self.feature_version,
            "source_split_version": self.source_split_version,
            "target_split_version": self.target_split_version,
        }.items():
            if not value:
                raise ValueError(f"{name} must be non-empty")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if any(
            dimension <= 0
            for dimension in (
                self.curve_hidden_dim,
                self.representation_dim,
                self.domain_hidden_dim,
            )
        ):
            raise ValueError("network dimensions must be positive")
        for name, numeric_value in {
            "learning_rate": self.learning_rate,
            "gradient_reversal_coefficient": self.gradient_reversal_coefficient,
            "domain_loss_weight": self.domain_loss_weight,
            "weight_decay": self.weight_decay,
        }.items():
            if not math.isfinite(numeric_value) or numeric_value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")


def _validated_curve_inputs(
    *,
    cell_ids: tuple[str, ...],
    curve_values: Tensor,
    observed_mask: Tensor,
    feature_contract: CurveFeatureContract,
) -> tuple[Tensor, Tensor]:
    if (
        not cell_ids
        or len(cell_ids) != len(set(cell_ids))
        or any(not cell_id for cell_id in cell_ids)
    ):
        raise ValueError("cell_ids must be non-empty and unique")
    if not isinstance(curve_values, Tensor) or not isinstance(observed_mask, Tensor):
        raise TypeError("curve_values and observed_mask must be torch tensors")
    if curve_values.ndim != 3:
        raise ValueError("curve_values must have shape [cell, cycle, voltage]")
    if (
        curve_values.shape[0] != len(cell_ids)
        or curve_values.shape[1] <= 0
        or curve_values.shape[2] <= 0
    ):
        raise ValueError("curve_values must contain one non-empty curve tensor per cell_id")
    if curve_values.shape[1] != len(feature_contract.cycle_indices):
        raise ValueError("curve_values cycle dimension must match the feature contract")
    if curve_values.shape[2] != len(feature_contract.voltage_grid_v):
        raise ValueError("curve_values voltage dimension must match the feature contract")
    if observed_mask.dtype is not torch.bool or observed_mask.shape != curve_values.shape[:2]:
        raise ValueError("observed_mask must be torch.bool with shape [cell, cycle]")

    values = curve_values.detach().to(device="cpu", dtype=torch.float32).clone()
    mask = observed_mask.detach().to(device="cpu", dtype=torch.bool).clone()
    if torch.any(mask.sum(dim=1) == 0):
        raise ValueError("each cell requires at least one observed curve")
    if not torch.isfinite(values[mask]).all():
        raise ValueError("observed curve values must be finite")
    absent_values = values[~mask]
    if absent_values.numel() and not torch.isnan(absent_values).all():
        raise ValueError("unobserved curve rows must remain NaN until masking")
    return values, mask


@dataclass(frozen=True)
class SourceDomainBatch:
    """Labelled source-train curves for CPMLP-DANN regression."""

    dataset_id: str
    cell_ids: tuple[str, ...]
    curve_values: Tensor
    observed_mask: Tensor
    eol80_labels: Tensor
    feature_contract: CurveFeatureContract

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        values, mask = _validated_curve_inputs(
            cell_ids=self.cell_ids,
            curve_values=self.curve_values,
            observed_mask=self.observed_mask,
            feature_contract=self.feature_contract,
        )
        if not isinstance(self.eol80_labels, Tensor):
            raise TypeError("eol80_labels must be a torch tensor")
        labels = self.eol80_labels.detach().to(device="cpu", dtype=torch.float32).clone()
        if labels.ndim != 1 or labels.shape[0] != len(self.cell_ids):
            raise ValueError("eol80_labels must contain one value per source cell_id")
        if not torch.isfinite(labels).all():
            raise ValueError("eol80_labels must be finite")
        object.__setattr__(self, "curve_values", values)
        object.__setattr__(self, "observed_mask", mask)
        object.__setattr__(self, "eol80_labels", labels)


@dataclass(frozen=True)
class TargetDomainBatch:
    """Unlabelled target-domain curves; this interface intentionally has no labels."""

    dataset_id: str
    cell_ids: tuple[str, ...]
    curve_values: Tensor
    observed_mask: Tensor
    feature_contract: CurveFeatureContract

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        values, mask = _validated_curve_inputs(
            cell_ids=self.cell_ids,
            curve_values=self.curve_values,
            observed_mask=self.observed_mask,
            feature_contract=self.feature_contract,
        )
        object.__setattr__(self, "curve_values", values)
        object.__setattr__(self, "observed_mask", mask)


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, values: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output.neg().mul(float(ctx.coefficient)), None


def gradient_reverse(values: Tensor, *, coefficient: float) -> Tensor:
    """Pass values through while reversing their encoder gradient."""

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("coefficient must be finite and non-negative")
    return cast(Tensor, _GradientReversalFunction.apply(values, coefficient))  # type: ignore[no-untyped-call]


class _MaskedCPMLPDANNNetwork(nn.Module):
    """Shared masked curve encoder with regression and domain-adversarial heads."""

    def __init__(
        self,
        *,
        voltage_points: int,
        curve_hidden_dim: int,
        representation_dim: int,
        domain_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.curve_encoder = nn.Sequential(
            nn.Linear(voltage_points, curve_hidden_dim),
            nn.ReLU(),
            nn.Linear(curve_hidden_dim, curve_hidden_dim),
            nn.ReLU(),
        )
        self.representation_encoder = nn.Sequential(
            nn.Linear(curve_hidden_dim + 1, representation_dim),
            nn.ReLU(),
        )
        self.regression_head = nn.Linear(representation_dim, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(representation_dim, domain_hidden_dim),
            nn.ReLU(),
            nn.Linear(domain_hidden_dim, 1),
        )

    def encode(self, curve_values: Tensor, observed_mask: Tensor) -> Tensor:
        masked_values = torch.where(
            observed_mask.unsqueeze(-1), curve_values, torch.zeros_like(curve_values)
        )
        encoded_cycles = self.curve_encoder(masked_values)
        weights = observed_mask.unsqueeze(-1).to(dtype=encoded_cycles.dtype)
        observed_counts = weights.sum(dim=1).clamp_min(1.0)
        pooled = (encoded_cycles * weights).sum(dim=1) / observed_counts
        coverage = observed_counts / float(curve_values.shape[1])
        return cast(Tensor, self.representation_encoder(torch.cat((pooled, coverage), dim=1)))

    def regress(self, representation: Tensor) -> Tensor:
        return cast(Tensor, self.regression_head(representation).squeeze(dim=1))

    def domain_logits(self, representation: Tensor, *, coefficient: float) -> Tensor:
        reversed_representation = gradient_reverse(representation, coefficient=coefficient)
        return cast(Tensor, self.domain_head(reversed_representation).squeeze(dim=1))


@dataclass(frozen=True)
class DANNPrediction:
    """Numerical model output plus explicit fitted-domain provenance."""

    dataset_id: str
    cell_ids: tuple[str, ...]
    predicted_eol_cycles: tuple[float, ...]
    model_version: str
    source_dataset_id: str
    target_dataset_id: str
    target_domain_cohort: TargetDomainCohort
    feature_version: str
    source_split_version: str
    target_split_version: str
    random_seed: int
    normalization_version: str
    normalization_input_hash: str
    feature_contract: CurveFeatureContract


@dataclass(frozen=True)
class _DANNState:
    source_dataset_id: str
    target_dataset_id: str
    target_domain_cohort: TargetDomainCohort
    curve_shape: tuple[int, int]
    feature_mean: float
    feature_std: float
    label_mean: float
    label_std: float
    feature_contract: CurveFeatureContract
    source_split_version: str
    target_split_version: str
    random_seed: int
    normalization_version: str
    normalization_input_hash: str


@dataclass
class CPMLPDANNAdapter:
    """Fit a genuine source-regression and target-adversarial DANN comparison."""

    config: DANNConfig
    _network: _MaskedCPMLPDANNNetwork | None = field(default=None, init=False, repr=False)
    _state: _DANNState | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        *,
        source_batch: SourceDomainBatch,
        target_batch: TargetDomainBatch,
        source_split_manifest: SplitManifest,
        target_split_manifest: SplitManifest,
        target_cohort: TargetDomainCohort,
    ) -> Self:
        """Fit only from source labels and an exact unlabelled target cohort."""

        self._validate_fit_cohorts(
            source_batch=source_batch,
            target_batch=target_batch,
            source_split_manifest=source_split_manifest,
            target_split_manifest=target_split_manifest,
            target_cohort=target_cohort,
        )
        self._validate_feature_contracts(source_batch=source_batch, target_batch=target_batch)
        if source_batch.curve_values.shape[1:] != target_batch.curve_values.shape[1:]:
            raise ValueError("source and target curve axes must match exactly")

        self._seed_everything(self.config.random_seed)
        feature_mean, feature_std = self._source_feature_statistics(source_batch)
        label_mean, label_std = self._source_label_statistics(source_batch.eol80_labels)
        normalization_input_hash = self._normalization_input_hash(source_batch)
        normalized_source = self._normalize_curves(source_batch, feature_mean, feature_std)
        normalized_target = self._normalize_curves(target_batch, feature_mean, feature_std)
        normalized_labels = (source_batch.eol80_labels - label_mean) / label_std

        network = _MaskedCPMLPDANNNetwork(
            voltage_points=source_batch.curve_values.shape[2],
            curve_hidden_dim=self.config.curve_hidden_dim,
            representation_dim=self.config.representation_dim,
            domain_hidden_dim=self.config.domain_hidden_dim,
        ).to(device="cpu")
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._train(
            network=network,
            source_values=normalized_source,
            target_values=normalized_target,
            source_mask=source_batch.observed_mask,
            target_mask=target_batch.observed_mask,
            normalized_labels=normalized_labels,
            optimizer=optimizer,
        )
        network.eval()
        self._network = network
        self._state = _DANNState(
            source_dataset_id=source_batch.dataset_id,
            target_dataset_id=target_batch.dataset_id,
            target_domain_cohort=target_cohort,
            curve_shape=(source_batch.curve_values.shape[1], source_batch.curve_values.shape[2]),
            feature_mean=feature_mean,
            feature_std=feature_std,
            label_mean=label_mean,
            label_std=label_std,
            feature_contract=source_batch.feature_contract,
            source_split_version=self.config.source_split_version,
            target_split_version=self.config.target_split_version,
            random_seed=self.config.random_seed,
            normalization_version=SOURCE_TRAIN_ZSCORE_NORMALIZATION_VERSION,
            normalization_input_hash=normalization_input_hash,
        )
        return self

    def predict(self, batch: SourceDomainBatch | TargetDomainBatch) -> DANNPrediction:
        """Predict from a schema-compatible cohort without using any target labels."""

        if self._network is None or self._state is None:
            raise RuntimeError("CPMLPDANNAdapter must be fitted before prediction")
        if batch.dataset_id not in {
            self._state.source_dataset_id,
            self._state.target_dataset_id,
        }:
            raise ValueError(
                "prediction batch dataset_id is outside the fitted source/target domains"
            )
        if batch.curve_values.shape[1:] != self._state.curve_shape:
            raise ValueError("prediction curve axes must match the fitted DANN schema")
        if batch.feature_contract != self._state.feature_contract:
            raise ValueError("prediction feature contract must match the fitted DANN contract")

        normalized = self._normalize_curves(
            batch, self._state.feature_mean, self._state.feature_std
        )
        with torch.no_grad():
            normalized_prediction = self._network.regress(
                self._network.encode(normalized, batch.observed_mask)
            )
            prediction = normalized_prediction * self._state.label_std + self._state.label_mean
        if not torch.isfinite(prediction).all():
            raise RuntimeError("CPMLP-DANN produced non-finite predictions")
        predicted_values = tuple(float(value) for value in prediction.tolist())
        return DANNPrediction(
            dataset_id=batch.dataset_id,
            cell_ids=batch.cell_ids,
            predicted_eol_cycles=predicted_values,
            model_version=self.config.model_version,
            source_dataset_id=self._state.source_dataset_id,
            target_dataset_id=self._state.target_dataset_id,
            target_domain_cohort=self._state.target_domain_cohort,
            feature_version=self._state.feature_contract.feature_version,
            source_split_version=self._state.source_split_version,
            target_split_version=self._state.target_split_version,
            random_seed=self._state.random_seed,
            normalization_version=self._state.normalization_version,
            normalization_input_hash=self._state.normalization_input_hash,
            feature_contract=self._state.feature_contract,
        )

    def _train(
        self,
        *,
        network: _MaskedCPMLPDANNNetwork,
        source_values: Tensor,
        target_values: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
        normalized_labels: Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        network.train()
        source_count = source_values.shape[0]
        target_count = target_values.shape[0]
        for _ in range(self.config.epochs):
            source_order = torch.randperm(source_count)
            target_order = torch.randperm(target_count)
            for start in range(0, source_count, self.config.batch_size):
                source_indices = source_order[start : start + self.config.batch_size]
                target_positions = torch.arange(source_indices.shape[0]) + start
                target_indices = target_order[target_positions.remainder(target_count)]
                source_representation = network.encode(
                    source_values[source_indices], source_mask[source_indices]
                )
                target_representation = network.encode(
                    target_values[target_indices], target_mask[target_indices]
                )
                regression_loss = functional.mse_loss(
                    network.regress(source_representation), normalized_labels[source_indices]
                )
                source_domain_loss = functional.binary_cross_entropy_with_logits(
                    network.domain_logits(
                        source_representation,
                        coefficient=self.config.gradient_reversal_coefficient,
                    ),
                    torch.zeros(source_indices.shape[0], dtype=torch.float32),
                )
                target_domain_loss = functional.binary_cross_entropy_with_logits(
                    network.domain_logits(
                        target_representation,
                        coefficient=self.config.gradient_reversal_coefficient,
                    ),
                    torch.ones(target_indices.shape[0], dtype=torch.float32),
                )
                loss = regression_loss + self.config.domain_loss_weight * (
                    source_domain_loss + target_domain_loss
                ) / 2.0
                if not torch.isfinite(loss):
                    raise RuntimeError("CPMLP-DANN training produced a non-finite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

    def _validate_fit_cohorts(
        self,
        *,
        source_batch: SourceDomainBatch,
        target_batch: TargetDomainBatch,
        source_split_manifest: SplitManifest,
        target_split_manifest: SplitManifest,
        target_cohort: TargetDomainCohort,
    ) -> None:
        if not isinstance(target_cohort, TargetDomainCohort):
            raise ValueError("target_cohort must be an explicit TargetDomainCohort")
        if source_batch.dataset_id != source_split_manifest.dataset_id:
            raise ValueError("source batch dataset_id must match source split manifest")
        if target_batch.dataset_id != target_split_manifest.dataset_id:
            raise ValueError("target batch dataset_id must match target split manifest")
        if set(source_split_manifest.all_cells) & set(target_split_manifest.all_cells):
            raise ValueError(
                "global bare cell_id overlap exists across source and target manifests"
            )
        if set(source_batch.cell_ids) != set(source_split_manifest.train):
            raise ValueError("source cell_ids must exactly match source split_manifest.train")
        target_cell_ids = self._target_cohort_cell_ids(target_split_manifest, target_cohort)
        if set(target_batch.cell_ids) != set(target_cell_ids):
            raise ValueError("target adaptation cell_ids must exactly match its authorized cohort")
        if set(source_batch.cell_ids) & set(target_batch.cell_ids):
            raise ValueError("source and target adaptation cell_ids must be disjoint")
        if set(target_batch.cell_ids) & set(target_split_manifest.test):
            raise ValueError("target test cell_ids are forbidden from DANN adaptation")

    def _validate_feature_contracts(
        self, *, source_batch: SourceDomainBatch, target_batch: TargetDomainBatch
    ) -> None:
        if source_batch.feature_contract.feature_version != self.config.feature_version:
            raise ValueError("source feature_version must match DANNConfig.feature_version")
        if target_batch.feature_contract.feature_version != self.config.feature_version:
            raise ValueError("target feature_version must match DANNConfig.feature_version")
        if source_batch.feature_contract != target_batch.feature_contract:
            raise ValueError("source and target feature contracts must exactly match")

    @staticmethod
    def _target_cohort_cell_ids(
        manifest: SplitManifest, cohort: TargetDomainCohort
    ) -> tuple[str, ...]:
        if cohort is TargetDomainCohort.ADAPTATION:
            return manifest.train
        return manifest.calibration

    @staticmethod
    def _source_feature_statistics(source_batch: SourceDomainBatch) -> tuple[float, float]:
        observed_values = source_batch.curve_values[source_batch.observed_mask]
        mean = float(observed_values.mean().item())
        std = float(observed_values.std(unbiased=False).item())
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("source feature normalization statistics must be finite")
        return mean, max(std, 1e-6)

    @staticmethod
    def _source_label_statistics(labels: Tensor) -> tuple[float, float]:
        mean = float(labels.mean().item())
        std = float(labels.std(unbiased=False).item())
        if not math.isfinite(mean) or not math.isfinite(std):
            raise ValueError("source label normalization statistics must be finite")
        return mean, max(std, 1e-6)

    @staticmethod
    def _normalize_curves(
        batch: SourceDomainBatch | TargetDomainBatch, mean: float, std: float
    ) -> Tensor:
        normalized_observed = (batch.curve_values - mean) / std
        return torch.where(
            batch.observed_mask.unsqueeze(-1), normalized_observed, batch.curve_values
        )

    @staticmethod
    def _normalization_input_hash(source_batch: SourceDomainBatch) -> str:
        """Hash only the source-train inputs used to fit normalization statistics."""

        digest = hashlib.sha256()
        digest.update(source_batch.dataset_id.encode("utf-8"))
        digest.update("\x1f".join(source_batch.cell_ids).encode("utf-8"))
        digest.update(source_batch.feature_contract.feature_version.encode("utf-8"))
        digest.update(str(source_batch.feature_contract.cutoff_cycle).encode("ascii"))
        digest.update(repr(source_batch.feature_contract.cycle_indices).encode("ascii"))
        digest.update(repr(source_batch.feature_contract.voltage_grid_v).encode("ascii"))
        digest.update(source_batch.curve_values.contiguous().numpy().tobytes())
        digest.update(source_batch.observed_mask.contiguous().numpy().tobytes())
        digest.update(source_batch.eol80_labels.contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
