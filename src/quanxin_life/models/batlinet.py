"""Train-cell-only BatLiNet pairing on top of the governed CyclePatch encoder."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass
from math import fsum

import torch
from torch import Tensor, nn
from torch.nn import functional

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchEncoder,
    EarlyCycleBatch,
)


@dataclass(frozen=True)
class CycleLifeTargetScaler:
    """Training-cell statistics for raw MATR official cycle-life targets."""

    dataset_id: str
    target: PredictionTarget
    cutoff_cycle: int
    mean: float
    scale: float
    training_cell_ids_sha256: str
    training_labels_sha256: str
    context_sha256: str

    def __post_init__(self) -> None:
        if (
            self.dataset_id != "MATR"
            or self.target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        ):
            raise ValueError("target scaler is restricted to MATR official cycle-life")
        if self.cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        if not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("target scaler statistics must be finite with positive scale")
        _require_sha256(self.training_cell_ids_sha256, "training_cell_ids_sha256")
        _require_sha256(self.training_labels_sha256, "training_labels_sha256")
        _require_sha256(self.context_sha256, "context_sha256")
        if sha256_canonical(self._context_payload()) != self.context_sha256:
            raise ValueError("context_sha256 does not match target scaler context")

    @classmethod
    def fit(
        cls,
        labels: Mapping[str, float],
        *,
        training_cell_ids: tuple[str, ...],
        split_manifest: SplitManifest,
        cutoff_cycle: int,
        dataset_id: str,
        target: PredictionTarget,
    ) -> CycleLifeTargetScaler:
        if dataset_id != "MATR" or target is not PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE:
            raise ValueError("scaler accepts only MATR official cycle-life labels")
        if cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        if split_manifest.dataset_id != dataset_id:
            raise ValueError("SplitManifest dataset_id must match scaler dataset_id")
        if not training_cell_ids or len(set(training_cell_ids)) != len(training_cell_ids):
            raise ValueError("training_cell_ids must be non-empty and unique")
        if any(not cell_id for cell_id in training_cell_ids):
            raise ValueError("training_cell_ids must not contain blanks")
        if training_cell_ids != split_manifest.train:
            raise ValueError("training_cell_ids must exactly equal SplitManifest.train")
        if set(labels) != set(training_cell_ids):
            raise ValueError(
                "labels must exactly match declared training cells; heldout labels are forbidden"
            )
        ordered_values = tuple(float(labels[cell_id]) for cell_id in sorted(training_cell_ids))
        if any(not math.isfinite(value) for value in ordered_values):
            raise ValueError("official cycle-life labels must be finite")
        if any(value <= cutoff_cycle for value in ordered_values):
            raise ValueError("official cycle-life labels must be greater than cutoff_cycle")
        mean = fsum(ordered_values) / len(ordered_values)
        variance = fsum((value - mean) ** 2 for value in ordered_values) / len(
            ordered_values
        )
        standard_deviation = math.sqrt(variance)
        scale = 1.0 if standard_deviation == 0.0 else standard_deviation
        training_hash = sha256_canonical(sorted(training_cell_ids))
        training_labels_hash = _training_labels_hash(labels)
        payload = _scaler_context_payload(
            dataset_id=dataset_id,
            target=target,
            cutoff_cycle=cutoff_cycle,
            mean=mean,
            scale=scale,
            training_cell_ids_sha256=training_hash,
            training_labels_sha256=training_labels_hash,
        )
        return cls(
            dataset_id=dataset_id,
            target=target,
            cutoff_cycle=cutoff_cycle,
            mean=mean,
            scale=scale,
            training_cell_ids_sha256=training_hash,
            training_labels_sha256=training_labels_hash,
            context_sha256=sha256_canonical(payload),
        )

    def transform(self, values: Tensor) -> Tensor:
        if not values.is_floating_point() or not bool(torch.isfinite(values).all().item()):
            raise ValueError("cycle-life values must be finite floating tensors")
        if torch.any(values <= self.cutoff_cycle):
            raise ValueError("cycle-life values must be greater than cutoff_cycle")
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: Tensor) -> Tensor:
        if not values.is_floating_point() or not bool(torch.isfinite(values).all().item()):
            raise ValueError("standardized values must be finite floating tensors")
        return values * self.scale + self.mean

    def _context_payload(self) -> dict[str, object]:
        return _scaler_context_payload(
            dataset_id=self.dataset_id,
            target=self.target,
            cutoff_cycle=self.cutoff_cycle,
            mean=self.mean,
            scale=self.scale,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
            training_labels_sha256=self.training_labels_sha256,
        )


@dataclass(frozen=True)
class CycleLifeReferenceLibrary:
    """Deterministic quantile-stratified references selected from training only."""

    cell_ids: tuple[str, ...]
    standardized_labels: tuple[float, ...]
    quantile_bins: tuple[int, ...]
    reference_count: int
    seed: int
    scaler_context_sha256: str
    training_cell_ids_sha256: str
    training_labels_sha256: str
    library_sha256: str

    def __post_init__(self) -> None:
        if self.reference_count <= 0 or len(self.cell_ids) != self.reference_count:
            raise ValueError("reference_count must match selected references")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("reference cell_ids must be unique")
        if not (
            len(self.standardized_labels)
            == len(self.quantile_bins)
            == self.reference_count
        ):
            raise ValueError("reference labels and quantile bins must align")
        if any(not math.isfinite(value) for value in self.standardized_labels):
            raise ValueError("standardized reference labels must be finite")
        if self.quantile_bins != tuple(range(self.reference_count)):
            raise ValueError("reference library must cover every quantile stratum")
        _require_sha256(self.scaler_context_sha256, "scaler_context_sha256")
        _require_sha256(self.training_cell_ids_sha256, "training_cell_ids_sha256")
        _require_sha256(self.training_labels_sha256, "training_labels_sha256")
        _require_sha256(self.library_sha256, "library_sha256")
        if sha256_canonical(self._hash_payload()) != self.library_sha256:
            raise ValueError("library_sha256 does not match reference library")

    @classmethod
    def build(
        cls,
        labels: Mapping[str, float],
        *,
        training_cell_ids: tuple[str, ...],
        split_manifest: SplitManifest,
        scaler: CycleLifeTargetScaler,
        reference_count: int,
        seed: int,
    ) -> CycleLifeReferenceLibrary:
        if not training_cell_ids or len(set(training_cell_ids)) != len(training_cell_ids):
            raise ValueError("training_cell_ids must be non-empty and unique")
        if split_manifest.dataset_id != scaler.dataset_id:
            raise ValueError("SplitManifest dataset_id must match scaler dataset_id")
        if training_cell_ids != split_manifest.train:
            raise ValueError("training_cell_ids must exactly equal SplitManifest.train")
        if set(labels) != set(training_cell_ids):
            raise ValueError(
                "reference labels must exactly match training cells; heldout is forbidden"
            )
        training_hash = sha256_canonical(sorted(training_cell_ids))
        if scaler.training_cell_ids_sha256 != training_hash:
            raise ValueError("scaler and reference training cells do not match")
        training_labels_hash = _training_labels_hash(labels)
        if scaler.training_labels_sha256 != training_labels_hash:
            raise ValueError("scaler and reference training labels do not match")
        if reference_count <= 0 or reference_count > len(training_cell_ids):
            raise ValueError("reference_count must be between 1 and training cell count")
        ordered = sorted(
            ((cell_id, float(labels[cell_id])) for cell_id in training_cell_ids),
            key=lambda item: (item[1], item[0]),
        )
        raw_tensor = torch.tensor(
            [label for _, label in ordered], dtype=torch.float64
        )
        scaler.transform(raw_tensor)
        rng = random.Random(seed)
        selected: list[tuple[str, float]] = []
        for index in range(reference_count):
            start = index * len(ordered) // reference_count
            stop = (index + 1) * len(ordered) // reference_count
            bucket = ordered[start:stop]
            selected.append(bucket[rng.randrange(len(bucket))])
        cell_ids = tuple(cell_id for cell_id, _ in selected)
        standardized = tuple((label - scaler.mean) / scaler.scale for _, label in selected)
        bins = tuple(range(reference_count))
        payload = _library_hash_payload(
            cell_ids=cell_ids,
            standardized_labels=standardized,
            quantile_bins=bins,
            reference_count=reference_count,
            seed=seed,
            scaler_context_sha256=scaler.context_sha256,
            training_cell_ids_sha256=training_hash,
            training_labels_sha256=training_labels_hash,
        )
        return cls(
            cell_ids=cell_ids,
            standardized_labels=standardized,
            quantile_bins=bins,
            reference_count=reference_count,
            seed=seed,
            scaler_context_sha256=scaler.context_sha256,
            training_cell_ids_sha256=training_hash,
            training_labels_sha256=training_labels_hash,
            library_sha256=sha256_canonical(payload),
        )

    def _hash_payload(self) -> dict[str, object]:
        return _library_hash_payload(
            cell_ids=self.cell_ids,
            standardized_labels=self.standardized_labels,
            quantile_bins=self.quantile_bins,
            reference_count=self.reference_count,
            seed=self.seed,
            scaler_context_sha256=self.scaler_context_sha256,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
            training_labels_sha256=self.training_labels_sha256,
        )


@dataclass(frozen=True, eq=False)
class CycleLifePairBatch:
    """Aligned standardized target/reference embeddings and labels."""

    target_embeddings: Tensor
    target_labels: Tensor
    reference_embeddings: Tensor
    reference_labels: Tensor

    def __post_init__(self) -> None:
        for name in (
            "target_embeddings",
            "target_labels",
            "reference_embeddings",
            "reference_labels",
        ):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                object.__setattr__(self, name, value.clone())
        if self.target_embeddings.ndim != 2 or self.reference_embeddings.ndim != 2:
            raise ValueError("target/reference embeddings must be rank-two tensors")
        if self.target_embeddings.shape[1] != self.reference_embeddings.shape[1]:
            raise ValueError("target/reference embedding dimensions must align")
        if self.target_labels.shape != (self.target_embeddings.shape[0],):
            raise ValueError("target_labels must align with target embeddings")
        if self.reference_labels.shape != (self.reference_embeddings.shape[0],):
            raise ValueError("reference_labels must align with reference embeddings")
        tensors = (
            self.target_embeddings,
            self.target_labels,
            self.reference_embeddings,
            self.reference_labels,
        )
        if any(not tensor.is_floating_point() for tensor in tensors) or any(
            not bool(torch.isfinite(tensor).all().item()) for tensor in tensors
        ):
            raise ValueError("pair embeddings and labels must be finite floating tensors")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("pair embeddings and labels must use one floating dtype")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("pair embeddings and labels must share a device")


@dataclass(frozen=True)
class BatLiNetConfig:
    encoder: CyclePatchConfig
    lambda_pair: float = 1.0
    lambda_rank: float = 0.0
    fusion_alpha: float = 0.5
    reference_count: int = 8

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_pair) or self.lambda_pair <= 0:
            raise ValueError("lambda_pair must be finite and strictly positive")
        if not math.isfinite(self.lambda_rank) or self.lambda_rank < 0:
            raise ValueError("lambda_rank must be finite and non-negative")
        if not math.isfinite(self.fusion_alpha) or not 0 <= self.fusion_alpha <= 1:
            raise ValueError("fusion_alpha must be finite and in [0, 1]")
        if self.reference_count <= 0:
            raise ValueError("reference_count must be positive")


class CyclePatchBatLiNet(nn.Module):
    """Shared CyclePatch encoder with direct and pairwise standardized heads."""

    def __init__(self, config: BatLiNetConfig, *, condition_count: int) -> None:
        super().__init__()
        self.config = config
        self.encoder = CyclePatchEncoder(config.encoder, condition_count=condition_count)
        self.direct_head = nn.Linear(config.encoder.d_model, 1)
        self.pair_head = nn.Sequential(
            nn.Linear(config.encoder.d_model * 3, config.encoder.d_model),
            nn.GELU(),
            nn.Linear(config.encoder.d_model, 1),
        )

    def encode(self, batch: EarlyCycleBatch) -> Tensor:
        embeddings: Tensor = self.encoder(batch)
        return embeddings

    def pair_delta(self, targets: Tensor, references: Tensor) -> Tensor:
        _validate_embedding_pair(
            targets,
            references,
            self.config.encoder.d_model,
            self.config.reference_count,
        )
        self._require_parameter_context(targets, references)
        target_grid = targets.unsqueeze(1).expand(-1, references.shape[0], -1)
        reference_grid = references.unsqueeze(0).expand(targets.shape[0], -1, -1)
        difference = target_grid - reference_grid
        features = torch.cat(
            (difference, torch.abs(difference), target_grid * reference_grid), dim=2
        )
        deltas: Tensor = self.pair_head(features).squeeze(dim=2)
        return deltas

    def fuse_standardized(
        self,
        targets: Tensor,
        references: Tensor,
        reference_labels: Tensor,
    ) -> Tensor:
        _validate_embedding_pair(
            targets,
            references,
            self.config.encoder.d_model,
            self.config.reference_count,
        )
        if reference_labels.shape != (references.shape[0],):
            raise ValueError("reference_labels must align with reference embeddings")
        if not reference_labels.is_floating_point():
            raise ValueError("reference_labels must use a floating dtype")
        if reference_labels.dtype != targets.dtype:
            raise ValueError("reference_labels dtype must match target/reference embeddings")
        if reference_labels.device != targets.device:
            raise ValueError("reference_labels device must match target/reference embeddings")
        if not bool(torch.isfinite(reference_labels).all().item()):
            raise ValueError("reference_labels must be finite")
        self._require_parameter_context(targets, references, reference_labels)
        direct: Tensor = self.direct_head(targets).squeeze(dim=1)
        if self.config.fusion_alpha == 1.0:
            return direct
        relative = reference_labels.unsqueeze(0) + self.pair_delta(targets, references)
        sorted_relative = torch.sort(relative, dim=1).values
        relative_median = sorted_relative[:, (sorted_relative.shape[1] - 1) // 2]
        return (
            self.config.fusion_alpha * direct
            + (1.0 - self.config.fusion_alpha) * relative_median
        )

    def loss(self, batch: CycleLifePairBatch) -> Tensor:
        _validate_embedding_pair(
            batch.target_embeddings,
            batch.reference_embeddings,
            self.config.encoder.d_model,
            self.config.reference_count,
        )
        self._require_parameter_context(
            batch.target_embeddings,
            batch.target_labels,
            batch.reference_embeddings,
            batch.reference_labels,
        )
        direct: Tensor = self.direct_head(batch.target_embeddings).squeeze(dim=1)
        predicted_delta = self.pair_delta(
            batch.target_embeddings, batch.reference_embeddings
        )
        true_delta = (
            batch.target_labels.unsqueeze(1) - batch.reference_labels.unsqueeze(0)
        )
        direct_loss = functional.smooth_l1_loss(direct, batch.target_labels)
        pair_loss = functional.smooth_l1_loss(predicted_delta, true_delta)
        nonzero = true_delta != 0
        if self.config.lambda_rank > 0 and bool(nonzero.any().item()):
            rank_loss = functional.softplus(
                -torch.sign(true_delta[nonzero]) * predicted_delta[nonzero]
            ).mean()
        else:
            rank_loss = torch.zeros((), dtype=direct.dtype, device=direct.device)
        total: Tensor = (
            direct_loss
            + self.config.lambda_pair * pair_loss
            + self.config.lambda_rank * rank_loss
        )
        return total

    def _require_parameter_context(self, *tensors: Tensor) -> None:
        parameter = next(self.parameters())
        if any(tensor.dtype != parameter.dtype for tensor in tensors):
            raise ValueError("tensor dtype must match model parameters")
        if any(tensor.device != parameter.device for tensor in tensors):
            raise ValueError("tensor device must match model parameters")


def _validate_embedding_pair(
    targets: Tensor,
    references: Tensor,
    d_model: int,
    reference_count: int,
) -> None:
    if targets.ndim != 2 or references.ndim != 2:
        raise ValueError("target/reference embeddings must be rank two")
    if targets.shape[1] != d_model or references.shape[1] != d_model:
        raise ValueError("target/reference embeddings must match encoder d_model")
    if targets.shape[0] == 0 or references.shape[0] == 0:
        raise ValueError("target/reference embeddings must be non-empty")
    if references.shape[0] != reference_count:
        raise ValueError("reference embeddings must match configured reference_count")
    if not targets.is_floating_point() or not references.is_floating_point():
        raise ValueError("target/reference embeddings must use a floating dtype")
    if targets.dtype != references.dtype:
        raise ValueError("target/reference embeddings must use the same dtype")
    if not bool(torch.isfinite(targets).all().item()) or not bool(
        torch.isfinite(references).all().item()
    ):
        raise ValueError("target/reference embeddings must be finite")
    if targets.device != references.device:
        raise ValueError("target/reference embeddings must share a device")


def _require_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _scaler_context_payload(
    *,
    dataset_id: str,
    target: PredictionTarget,
    cutoff_cycle: int,
    mean: float,
    scale: float,
    training_cell_ids_sha256: str,
    training_labels_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "matr-cycle-life-target-scaler-v1",
        "dataset_id": dataset_id,
        "target": target.value,
        "cutoff_cycle": cutoff_cycle,
        "mean": mean,
        "scale": scale,
        "training_cell_ids_sha256": training_cell_ids_sha256,
        "training_labels_sha256": training_labels_sha256,
    }


def _library_hash_payload(
    *,
    cell_ids: tuple[str, ...],
    standardized_labels: tuple[float, ...],
    quantile_bins: tuple[int, ...],
    reference_count: int,
    seed: int,
    scaler_context_sha256: str,
    training_cell_ids_sha256: str,
    training_labels_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "cycle-life-reference-library-v1",
        "cell_ids": cell_ids,
        "standardized_labels": standardized_labels,
        "quantile_bins": quantile_bins,
        "reference_count": reference_count,
        "seed": seed,
        "scaler_context_sha256": scaler_context_sha256,
        "training_cell_ids_sha256": training_cell_ids_sha256,
        "training_labels_sha256": training_labels_sha256,
    }


def _training_labels_hash(labels: Mapping[str, float]) -> str:
    return sha256_canonical(
        sorted((cell_id, float(label)) for cell_id, label in labels.items())
    )
