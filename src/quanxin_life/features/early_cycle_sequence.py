"""Label-free multichannel early-cycle sequences and train-only normalization."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Self

import torch
from torch import Tensor

from quanxin_life.core.hashing import sha256_canonical

PHASE_NAMES = ("charge", "discharge")
VARIABLE_NAMES = ("voltage_v", "current_a", "capacity_ah")
SAMPLES_PER_PHASE = 150
RAW_NORMALIZATION_VERSION = "none"
TRAIN_ZSCORE_NORMALIZATION_VERSION = "train-zscore-v1"


@dataclass(frozen=True, eq=False)
class EarlyCycleSequence:
    """One label-free sequence with explicit sample and condition masks."""

    dataset_id: str
    cell_id: str
    cutoff_cycle: int
    data_version: str
    feature_version: str
    cycle_indices: tuple[int, ...]
    values: Tensor
    cycle_mask: Tensor
    sample_mask: Tensor
    condition_names: tuple[str, ...]
    condition_values: Tensor
    condition_mask: Tensor
    phase_names: tuple[str, ...] = PHASE_NAMES
    variable_names: tuple[str, ...] = VARIABLE_NAMES
    normalization_version: str = RAW_NORMALIZATION_VERSION
    normalization_statistics_sha256: str | None = None
    input_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self._snapshot_tensors()
        self._validate_identity_and_axes()
        self._validate_sequence_tensors()
        self._validate_conditions()
        object.__setattr__(self, "input_hash", sha256_canonical(self._hash_payload()))

    def _snapshot_tensors(self) -> None:
        for name in (
            "values",
            "cycle_mask",
            "sample_mask",
            "condition_values",
            "condition_mask",
        ):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                object.__setattr__(self, name, value.detach().clone())

    def _validate_identity_and_axes(self) -> None:
        for name in ("dataset_id", "cell_id", "data_version", "feature_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        if self.phase_names != PHASE_NAMES:
            raise ValueError(f"phase_names must be fixed to {PHASE_NAMES}")
        if self.variable_names != VARIABLE_NAMES:
            raise ValueError(f"variable_names must be fixed to {VARIABLE_NAMES}")
        if self.normalization_version == RAW_NORMALIZATION_VERSION:
            if self.normalization_statistics_sha256 is not None:
                raise ValueError(
                    "raw sequence normalization_statistics_sha256 must be None"
                )
        elif self.normalization_version == TRAIN_ZSCORE_NORMALIZATION_VERSION:
            if self.normalization_statistics_sha256 is None:
                raise ValueError(
                    "train-zscore-v1 requires normalization_statistics_sha256"
                )
            _require_sha256(
                self.normalization_statistics_sha256,
                "normalization_statistics_sha256",
            )
        else:
            raise ValueError("normalization_version is not supported")
        if not self.cycle_indices:
            raise ValueError("cycle_indices must not be empty")
        if any(cycle < 0 for cycle in self.cycle_indices):
            raise ValueError("cycle_indices must be non-negative")
        if any(
            current <= previous
            for previous, current in zip(
                self.cycle_indices, self.cycle_indices[1:], strict=False
            )
        ):
            raise ValueError("cycle_indices must be strictly increasing")
        if self.cycle_indices != tuple(range(self.cutoff_cycle + 1)):
            raise ValueError("cycle_indices must equal 0..cutoff_cycle")

    def _validate_sequence_tensors(self) -> None:
        _require_tensor(self.values, "values", torch.float32)
        _require_tensor(self.cycle_mask, "cycle_mask", torch.bool)
        _require_tensor(self.sample_mask, "sample_mask", torch.bool)
        if self.values.ndim != 4 or self.values.shape[1] != len(PHASE_NAMES):
            raise ValueError(
                "values shape must be [cycle, phase=2, sample, variable=3]"
            )
        if self.values.shape[-1] != len(VARIABLE_NAMES):
            raise ValueError(
                "values shape must be [cycle, phase=2, sample, variable=3]"
            )
        if self.values.shape[2] != SAMPLES_PER_PHASE:
            raise ValueError(f"each phase must contain exactly {SAMPLES_PER_PHASE} samples")
        cycle_count = len(self.cycle_indices)
        if self.values.shape[0] != cycle_count:
            raise ValueError("values cycle axis must align with cycle_indices")
        if self.cycle_mask.shape != (cycle_count,):
            raise ValueError("cycle_mask shape must align with cycle_indices")
        if self.sample_mask.shape != self.values.shape[:-1]:
            raise ValueError("sample_mask shape must align with values")
        if not (
            self.values.device == self.cycle_mask.device == self.sample_mask.device
        ):
            raise ValueError("values and sequence masks must use the same device")
        observed_cycles = self.sample_mask.any(dim=(1, 2))
        if not torch.equal(self.cycle_mask, observed_cycles):
            raise ValueError(
                "cycle_mask must be exactly equal to sample_mask.any over phase and sample"
            )
        observed = self.values[self.sample_mask]
        missing = self.values[~self.sample_mask]
        if observed.numel() and not bool(torch.isfinite(observed).all().item()):
            raise ValueError("observed points must contain only finite values")
        if missing.numel() and not bool(torch.isnan(missing).all().item()):
            raise ValueError("unobserved points must contain only NaN values")

    def _validate_conditions(self) -> None:
        _require_tensor(self.condition_values, "condition_values", torch.float32)
        _require_tensor(self.condition_mask, "condition_mask", torch.bool)
        if len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("condition_names must be unique")
        if any(not name.strip() for name in self.condition_names):
            raise ValueError("condition_names must not contain blank names")
        expected_shape = (len(self.condition_names),)
        if self.condition_values.shape != expected_shape:
            raise ValueError("condition_values must align with condition_names")
        if self.condition_mask.shape != expected_shape:
            raise ValueError("condition_mask must align with condition_names")
        if self.condition_values.device != self.condition_mask.device:
            raise ValueError("condition_values and condition_mask must use the same device")
        observed = self.condition_values[self.condition_mask]
        missing = self.condition_values[~self.condition_mask]
        if observed.numel() and not bool(torch.isfinite(observed).all().item()):
            raise ValueError("observed conditions must contain only finite values")
        if missing.numel() and not bool(torch.isnan(missing).all().item()):
            raise ValueError("unobserved conditions must contain only NaN values")

    def _hash_payload(self) -> dict[str, object]:
        values = self.values.detach().cpu()
        sample_mask = self.sample_mask.detach().cpu()
        encoded_values: list[list[list[list[float | None]]]] = []
        for cycle_index in range(values.shape[0]):
            encoded_cycle: list[list[list[float | None]]] = []
            for phase_index in range(values.shape[1]):
                encoded_phase: list[list[float | None]] = []
                for sample_index in range(values.shape[2]):
                    if bool(sample_mask[cycle_index, phase_index, sample_index]):
                        encoded_phase.append(
                            [
                                float(item)
                                for item in values[cycle_index, phase_index, sample_index]
                            ]
                        )
                    else:
                        encoded_phase.append([None] * len(VARIABLE_NAMES))
                encoded_cycle.append(encoded_phase)
            encoded_values.append(encoded_cycle)
        conditions = self.condition_values.detach().cpu()
        condition_mask = self.condition_mask.detach().cpu()
        return {
            "schema_version": "early-cycle-sequence-v1",
            "dataset_id": self.dataset_id,
            "cell_id": self.cell_id,
            "cutoff_cycle": self.cutoff_cycle,
            "data_version": self.data_version,
            "feature_version": self.feature_version,
            "normalization_version": self.normalization_version,
            "normalization_statistics_sha256": self.normalization_statistics_sha256,
            "phase_names": self.phase_names,
            "variable_names": self.variable_names,
            "cycle_indices": self.cycle_indices,
            "values": encoded_values,
            "cycle_mask": self.cycle_mask.detach().cpu().tolist(),
            "sample_mask": sample_mask.tolist(),
            "condition_names": self.condition_names,
            "condition_values": [
                float(value) if bool(condition_mask[index]) else None
                for index, value in enumerate(conditions)
            ],
            "condition_mask": condition_mask.tolist(),
        }

    def verify_input_hash(self) -> None:
        """Reject tensors changed after construction before they reach a consumer."""

        try:
            observed_hash = sha256_canonical(self._hash_payload())
        except (TypeError, ValueError) as exc:
            raise ValueError("input_hash verification failed after tensor mutation") from exc
        if observed_hash != self.input_hash:
            raise ValueError("input_hash does not match the current sequence tensors")


@dataclass(frozen=True, eq=False)
class EarlyCycleNormalizer:
    """Train-cell-only variable statistics for one fixed sequence schema."""

    dataset_id: str
    data_version: str
    feature_version: str
    cutoff_cycle: int
    cycle_indices: tuple[int, ...]
    sample_count: int
    condition_names: tuple[str, ...]
    variable_means: tuple[float, ...]
    variable_stds: tuple[float, ...]
    condition_means: tuple[float, ...]
    condition_stds: tuple[float, ...]
    training_cell_ids_sha256: str
    statistics_sha256: str

    def __post_init__(self) -> None:
        for name in ("dataset_id", "data_version", "feature_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.cutoff_cycle < 0:
            raise ValueError("cutoff_cycle must be non-negative")
        if self.cycle_indices != tuple(range(self.cutoff_cycle + 1)):
            raise ValueError("cycle_indices must equal 0..cutoff_cycle")
        if self.sample_count != SAMPLES_PER_PHASE:
            raise ValueError(
                f"sample_count must equal the fixed sample count {SAMPLES_PER_PHASE}"
            )
        if len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("condition_names must be unique")
        if any(not name.strip() for name in self.condition_names):
            raise ValueError("condition_names must not contain blank names")
        _validate_statistics(
            self.variable_means,
            self.variable_stds,
            expected_count=len(VARIABLE_NAMES),
            label="variable",
        )
        _validate_statistics(
            self.condition_means,
            self.condition_stds,
            expected_count=len(self.condition_names),
            label="condition",
        )
        _require_sha256(
            self.training_cell_ids_sha256,
            "training_cell_ids_sha256",
        )
        _require_sha256(self.statistics_sha256, "statistics_sha256")
        expected_hash = sha256_canonical(self._statistics_payload())
        if expected_hash != self.statistics_sha256:
            raise ValueError("statistics_sha256 does not match normalizer statistics")

    @classmethod
    def fit(
        cls,
        sequences: tuple[EarlyCycleSequence, ...],
        *,
        training_cell_ids: frozenset[str],
    ) -> Self:
        if not sequences:
            raise ValueError("normalizer requires at least one training sequence")
        for sequence in sequences:
            sequence.verify_input_hash()
        if not training_cell_ids:
            raise ValueError("training_cell_ids must not be empty")
        observed_cell_ids = tuple(sequence.cell_id for sequence in sequences)
        if len(set(observed_cell_ids)) != len(observed_cell_ids):
            raise ValueError("training sequences must contain unique cell_ids")
        if set(observed_cell_ids) != set(training_cell_ids):
            raise ValueError(
                "training sequences must exactly match the declared training_cell_ids"
            )
        reference_schema = _sequence_schema(sequences[0])
        if any(_sequence_schema(sequence) != reference_schema for sequence in sequences[1:]):
            raise ValueError("all training sequences must use an aligned schema")
        observed_points = torch.cat(
            tuple(sequence.values[sequence.sample_mask] for sequence in sequences),
            dim=0,
        )
        if observed_points.shape[0] == 0:
            raise ValueError("normalizer requires observed training samples")
        means = observed_points.mean(dim=0)
        stds = observed_points.std(dim=0, unbiased=False)
        if not bool(torch.isfinite(means).all().item()) or not bool(
            torch.isfinite(stds).all().item()
        ):
            raise ValueError("normalizer statistics must be finite")
        condition_means: list[float] = []
        condition_stds: list[float] = []
        for index, condition_name in enumerate(sequences[0].condition_names):
            observed_conditions = torch.cat(
                tuple(
                    sequence.condition_values[index : index + 1]
                    for sequence in sequences
                    if bool(sequence.condition_mask[index].item())
                ),
                dim=0,
            ) if any(
                bool(sequence.condition_mask[index].item()) for sequence in sequences
            ) else None
            if observed_conditions is None or observed_conditions.numel() == 0:
                raise ValueError(
                    f"condition {condition_name!r} has no observed training values"
                )
            condition_mean = observed_conditions.mean()
            condition_std = observed_conditions.std(unbiased=False)
            if not bool(torch.isfinite(condition_mean).item()) or not bool(
                torch.isfinite(condition_std).item()
            ):
                raise ValueError("condition normalizer statistics must be finite")
            condition_means.append(float(condition_mean.detach().cpu()))
            condition_stds.append(float(condition_std.detach().cpu()))
        first = sequences[0]
        variable_means = tuple(float(value) for value in means.detach().cpu())
        variable_stds = tuple(float(value) for value in stds.detach().cpu())
        training_hash = sha256_canonical(sorted(training_cell_ids))
        statistics_payload = _normalizer_statistics_payload(
            dataset_id=first.dataset_id,
            data_version=first.data_version,
            feature_version=first.feature_version,
            cutoff_cycle=first.cutoff_cycle,
            cycle_indices=first.cycle_indices,
            sample_count=int(first.values.shape[2]),
            condition_names=first.condition_names,
            variable_means=variable_means,
            variable_stds=variable_stds,
            condition_means=tuple(condition_means),
            condition_stds=tuple(condition_stds),
            training_cell_ids_sha256=training_hash,
        )
        statistics_hash = sha256_canonical(statistics_payload)
        return cls(
            dataset_id=first.dataset_id,
            data_version=first.data_version,
            feature_version=first.feature_version,
            cutoff_cycle=first.cutoff_cycle,
            cycle_indices=first.cycle_indices,
            sample_count=int(first.values.shape[2]),
            condition_names=first.condition_names,
            variable_means=variable_means,
            variable_stds=variable_stds,
            condition_means=tuple(condition_means),
            condition_stds=tuple(condition_stds),
            training_cell_ids_sha256=training_hash,
            statistics_sha256=statistics_hash,
        )

    def transform(self, sequence: EarlyCycleSequence) -> EarlyCycleSequence:
        """Normalize observed variables while preserving all explicit missing values."""

        sequence.verify_input_hash()
        if sequence.normalization_version != RAW_NORMALIZATION_VERSION:
            raise ValueError("sequence is already normalized")
        if _sequence_schema(sequence) != self._schema:
            raise ValueError("sequence schema does not match the fitted normalizer schema")
        means = torch.tensor(
            self.variable_means,
            dtype=torch.float32,
            device=sequence.values.device,
        )
        raw_stds = torch.tensor(
            self.variable_stds,
            dtype=torch.float32,
            device=sequence.values.device,
        )
        scales = torch.where(raw_stds > 0, raw_stds, torch.ones_like(raw_stds))
        values = sequence.values.clone()
        values[sequence.sample_mask] = (
            values[sequence.sample_mask] - means
        ) / scales
        condition_means = torch.tensor(
            self.condition_means,
            dtype=torch.float32,
            device=sequence.condition_values.device,
        )
        raw_condition_stds = torch.tensor(
            self.condition_stds,
            dtype=torch.float32,
            device=sequence.condition_values.device,
        )
        condition_scales = torch.where(
            raw_condition_stds > 0,
            raw_condition_stds,
            torch.ones_like(raw_condition_stds),
        )
        condition_values = sequence.condition_values.clone()
        condition_values[sequence.condition_mask] = (
            condition_values[sequence.condition_mask]
            - condition_means[sequence.condition_mask]
        ) / condition_scales[sequence.condition_mask]
        return EarlyCycleSequence(
            dataset_id=sequence.dataset_id,
            cell_id=sequence.cell_id,
            cutoff_cycle=sequence.cutoff_cycle,
            data_version=sequence.data_version,
            feature_version=sequence.feature_version,
            cycle_indices=sequence.cycle_indices,
            values=values,
            cycle_mask=sequence.cycle_mask.clone(),
            sample_mask=sequence.sample_mask.clone(),
            condition_names=sequence.condition_names,
            condition_values=condition_values,
            condition_mask=sequence.condition_mask.clone(),
            normalization_version=TRAIN_ZSCORE_NORMALIZATION_VERSION,
            normalization_statistics_sha256=self.statistics_sha256,
        )

    @property
    def _schema(self) -> tuple[object, ...]:
        return (
            self.dataset_id,
            self.data_version,
            self.feature_version,
            self.cutoff_cycle,
            self.cycle_indices,
            self.sample_count,
            self.condition_names,
        )

    def _statistics_payload(self) -> dict[str, object]:
        return _normalizer_statistics_payload(
            dataset_id=self.dataset_id,
            data_version=self.data_version,
            feature_version=self.feature_version,
            cutoff_cycle=self.cutoff_cycle,
            cycle_indices=self.cycle_indices,
            sample_count=self.sample_count,
            condition_names=self.condition_names,
            variable_means=self.variable_means,
            variable_stds=self.variable_stds,
            condition_means=self.condition_means,
            condition_stds=self.condition_stds,
            training_cell_ids_sha256=self.training_cell_ids_sha256,
        )


def _sequence_schema(sequence: EarlyCycleSequence) -> tuple[object, ...]:
    return (
        sequence.dataset_id,
        sequence.data_version,
        sequence.feature_version,
        sequence.cutoff_cycle,
        sequence.cycle_indices,
        int(sequence.values.shape[2]),
        sequence.condition_names,
    )


def _require_tensor(value: object, name: str, dtype: torch.dtype) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not dtype:
        raise ValueError(f"{name} must use {dtype}")


def _validate_statistics(
    means: tuple[float, ...],
    stds: tuple[float, ...],
    *,
    expected_count: int,
    label: str,
) -> None:
    if len(means) != expected_count or len(stds) != expected_count:
        raise ValueError(f"{label} statistics must match the declared schema")
    if any(not math.isfinite(value) for value in (*means, *stds)):
        raise ValueError(f"{label} statistics must be finite")
    if any(value < 0 for value in stds):
        raise ValueError(f"{label} standard deviations must be non-negative")


def _require_sha256(value: str, name: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _normalizer_statistics_payload(
    *,
    dataset_id: str,
    data_version: str,
    feature_version: str,
    cutoff_cycle: int,
    cycle_indices: tuple[int, ...],
    sample_count: int,
    condition_names: tuple[str, ...],
    variable_means: tuple[float, ...],
    variable_stds: tuple[float, ...],
    condition_means: tuple[float, ...],
    condition_stds: tuple[float, ...],
    training_cell_ids_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": "early-cycle-normalizer-v1",
        "dataset_id": dataset_id,
        "data_version": data_version,
        "feature_version": feature_version,
        "cutoff_cycle": cutoff_cycle,
        "cycle_indices": cycle_indices,
        "sample_count": sample_count,
        "phase_names": PHASE_NAMES,
        "variable_names": VARIABLE_NAMES,
        "condition_names": condition_names,
        "variable_means": variable_means,
        "variable_stds": variable_stds,
        "condition_means": condition_means,
        "condition_stds": condition_stds,
        "training_cell_ids_sha256": training_cell_ids_sha256,
    }
