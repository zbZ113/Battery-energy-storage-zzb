"""Direct-only masked CyclePatch encoder for normalized early-cycle sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from quanxin_life.features.early_cycle_sequence import (
    PHASE_NAMES,
    SAMPLES_PER_PHASE,
    TRAIN_ZSCORE_NORMALIZATION_VERSION,
    VARIABLE_NAMES,
    EarlyCycleSequence,
)


@dataclass(frozen=True, eq=False)
class EarlyCycleBatch:
    """A label-free, mask-preserving batch bound to one normalization context."""

    dataset_id: str
    data_version: str
    feature_version: str
    normalization_statistics_sha256: str
    cell_ids: tuple[str, ...]
    condition_names: tuple[str, ...]
    values: Tensor
    cycle_indices: Tensor
    cycle_mask: Tensor
    sample_mask: Tensor
    condition_values: Tensor
    condition_mask: Tensor

    def __post_init__(self) -> None:
        for name in ("values", "cycle_indices", "cycle_mask", "sample_mask"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                object.__setattr__(self, name, value.detach().clone())
        for name in ("condition_values", "condition_mask"):
            value = getattr(self, name)
            if isinstance(value, Tensor):
                object.__setattr__(self, name, value.detach().clone())
        self._validate()

    def _validate(self) -> None:
        for name in ("dataset_id", "data_version", "feature_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if len(self.normalization_statistics_sha256) != 64:
            raise ValueError("normalization_statistics_sha256 must be a SHA-256 digest")
        if not self.cell_ids or len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("cell_ids must be non-empty and unique")
        if len(set(self.condition_names)) != len(self.condition_names) or any(
            not name.strip() for name in self.condition_names
        ):
            raise ValueError("condition_names must be unique and nonblank")
        _require_tensor(self.values, "values", torch.float32)
        _require_tensor(self.cycle_indices, "cycle_indices", torch.int64)
        _require_tensor(self.cycle_mask, "cycle_mask", torch.bool)
        _require_tensor(self.sample_mask, "sample_mask", torch.bool)
        _require_tensor(self.condition_values, "condition_values", torch.float32)
        _require_tensor(self.condition_mask, "condition_mask", torch.bool)
        batch_size = len(self.cell_ids)
        if self.values.ndim != 5 or self.values.shape != (
            batch_size,
            self.values.shape[1],
            len(PHASE_NAMES),
            SAMPLES_PER_PHASE,
            len(VARIABLE_NAMES),
        ):
            raise ValueError("values must have shape [batch, cycle, 2, 150, 3]")
        cycle_count = self.values.shape[1]
        if self.cycle_indices.shape != (batch_size, cycle_count):
            raise ValueError("cycle_indices must match batch and cycle axes")
        expected_cycle_indices = torch.arange(
            cycle_count,
            dtype=torch.int64,
            device=self.cycle_indices.device,
        ).expand(batch_size, -1)
        if not torch.equal(self.cycle_indices, expected_cycle_indices):
            raise ValueError("every cycle_indices row must equal the fixed 0..cutoff axis")
        if self.cycle_mask.shape != (batch_size, cycle_count):
            raise ValueError("cycle_mask must match batch and cycle axes")
        if self.sample_mask.shape != self.values.shape[:-1]:
            raise ValueError("sample_mask must align with values")
        condition_shape = (batch_size, len(self.condition_names))
        if self.condition_values.shape != condition_shape:
            raise ValueError("condition_values must align with condition_names")
        if self.condition_mask.shape != condition_shape:
            raise ValueError("condition_mask must align with condition_names")
        devices = {
            self.values.device,
            self.cycle_indices.device,
            self.cycle_mask.device,
            self.sample_mask.device,
            self.condition_values.device,
            self.condition_mask.device,
        }
        if len(devices) != 1:
            raise ValueError("all batch tensors must use the same device")
        expected_cycle_mask = self.sample_mask.any(dim=(2, 3))
        if not torch.equal(self.cycle_mask, expected_cycle_mask):
            raise ValueError("cycle_mask must equal sample_mask.any over phase and sample")
        if torch.any(~self.cycle_mask.any(dim=1)):
            raise ValueError("each cell requires at least one valid cycle")
        observed = self.values[self.sample_mask]
        if observed.numel() and not bool(torch.isfinite(observed).all().item()):
            raise ValueError("observed sequence values must be finite")
        observed_conditions = self.condition_values[self.condition_mask]
        if observed_conditions.numel() and not bool(
            torch.isfinite(observed_conditions).all().item()
        ):
            raise ValueError("observed condition values must be finite")


def stack_early_cycle_sequences(
    sequences: tuple[EarlyCycleSequence, ...],
) -> EarlyCycleBatch:
    """Stack normalized sequences after revalidating hashes and shared schema."""

    if not sequences:
        raise ValueError("at least one normalized early-cycle sequence is required")
    for sequence in sequences:
        sequence.verify_input_hash()
        if sequence.normalization_version != TRAIN_ZSCORE_NORMALIZATION_VERSION:
            raise ValueError("all sequences must use train-zscore-v1 normalization")
        if sequence.normalization_statistics_sha256 is None:
            raise ValueError("normalization_statistics_sha256 is required")
        if sequence.values.device != sequence.condition_values.device:
            raise ValueError("sequence values and conditions must use the same device")
    cell_ids = tuple(sequence.cell_id for sequence in sequences)
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("batch sequences must contain unique cell_ids")
    reference = sequences[0]
    reference_schema = _sequence_schema(reference)
    if any(_sequence_schema(sequence) != reference_schema for sequence in sequences[1:]):
        raise ValueError(
            "all sequences must share schema and normalization_statistics_sha256"
        )
    cycle_indices = torch.stack(
        tuple(
            torch.tensor(
                sequence.cycle_indices,
                dtype=torch.int64,
                device=sequence.values.device,
            )
            for sequence in sequences
        )
    )
    assert reference.normalization_statistics_sha256 is not None
    return EarlyCycleBatch(
        dataset_id=reference.dataset_id,
        data_version=reference.data_version,
        feature_version=reference.feature_version,
        normalization_statistics_sha256=reference.normalization_statistics_sha256,
        cell_ids=cell_ids,
        condition_names=reference.condition_names,
        values=torch.stack(tuple(sequence.values for sequence in sequences)),
        cycle_indices=cycle_indices,
        cycle_mask=torch.stack(tuple(sequence.cycle_mask for sequence in sequences)),
        sample_mask=torch.stack(tuple(sequence.sample_mask for sequence in sequences)),
        condition_values=torch.stack(
            tuple(sequence.condition_values for sequence in sequences)
        ),
        condition_mask=torch.stack(
            tuple(sequence.condition_mask for sequence in sequences)
        ),
    )


@dataclass(frozen=True)
class CyclePatchConfig:
    d_model: int = 128
    layers: int = 4
    heads: int = 8
    dropout: float = 0.1
    max_cycles: int = 151

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.layers <= 0:
            raise ValueError("layers must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.max_cycles != 151:
            raise ValueError("max_cycles must be fixed to 151")


class _PhaseMultiScale(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        branch_dim = max(1, d_model // 3)
        self.branches = nn.ModuleList(
            nn.Conv1d(len(VARIABLE_NAMES), branch_dim, kernel, padding=kernel // 2)
            for kernel in (3, 5, 7)
        )
        self.projection = nn.Sequential(
            nn.Linear(branch_dim * 3, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, values: Tensor, sample_mask: Tensor) -> Tensor:
        clean = torch.where(
            sample_mask.unsqueeze(-1), values, torch.zeros_like(values)
        ).transpose(1, 2)
        encoded = torch.cat(
            tuple(torch.nn.functional.gelu(branch(clean)) for branch in self.branches),
            dim=1,
        )
        weights = sample_mask.unsqueeze(1).to(dtype=encoded.dtype)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        valid = sample_mask.any(dim=1, keepdim=True)
        projected: Tensor = self.projection(pooled)
        return projected * valid.to(dtype=pooled.dtype)


class PhasePatchEncoder(nn.Module):
    """Independent multiscale convolutional encoders for charge and discharge."""

    def __init__(self, *, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.charge_encoder = _PhaseMultiScale(d_model)
        self.discharge_encoder = _PhaseMultiScale(d_model)

    def forward(self, values: Tensor, sample_mask: Tensor) -> Tensor:
        if values.ndim != 5 or values.shape[2:] != (
            len(PHASE_NAMES),
            SAMPLES_PER_PHASE,
            len(VARIABLE_NAMES),
        ):
            raise ValueError("values must have shape [batch, cycle, 2, 150, 3]")
        if sample_mask.shape != values.shape[:-1] or sample_mask.dtype is not torch.bool:
            raise ValueError("sample_mask must be bool and align with values")
        batch_size, cycle_count = values.shape[:2]
        flattened_values = values.reshape(
            batch_size * cycle_count,
            len(PHASE_NAMES),
            SAMPLES_PER_PHASE,
            len(VARIABLE_NAMES),
        )
        flattened_mask = sample_mask.reshape(
            batch_size * cycle_count, len(PHASE_NAMES), SAMPLES_PER_PHASE
        )
        charge = self.charge_encoder(flattened_values[:, 0], flattened_mask[:, 0])
        discharge = self.discharge_encoder(
            flattened_values[:, 1], flattened_mask[:, 1]
        )
        return torch.stack((charge, discharge), dim=1).reshape(
            batch_size, cycle_count, len(PHASE_NAMES), -1
        )


class CyclePatchEncoder(nn.Module):
    """Fuse phase, real cycle position and conditions with masked attention."""

    def __init__(self, config: CyclePatchConfig, *, condition_count: int) -> None:
        super().__init__()
        if condition_count <= 0:
            raise ValueError("condition_count must be positive")
        self.config = config
        self.condition_count = condition_count
        self.phase_encoder = PhasePatchEncoder(d_model=config.d_model)
        self.phase_fusion = nn.Sequential(
            nn.Linear(config.d_model * len(PHASE_NAMES), config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.cycle_position_encoder = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_count * 2, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_encoding = nn.Parameter(
            torch.zeros(1, config.max_cycles + 1, config.d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.layers)
        self.attention_score = nn.Linear(config.d_model, 1)
        self.fusion_gate = nn.Linear(config.d_model * 2, config.d_model)

    def forward(self, batch: EarlyCycleBatch) -> Tensor:
        batch_size, cycle_count = batch.values.shape[:2]
        if cycle_count > self.config.max_cycles:
            raise ValueError("cycle count exceeds max_cycles")
        if batch.condition_values.shape[1] != self.condition_count:
            raise ValueError("batch condition count does not match encoder")
        if torch.any(batch.cycle_indices < 0) or torch.any(
            batch.cycle_indices >= self.config.max_cycles
        ):
            raise ValueError("cycle_indices must be within configured real positions")
        phase = self.phase_encoder(batch.values, batch.sample_mask).flatten(start_dim=2)
        phase_tokens = self.phase_fusion(phase)
        normalized_positions = batch.cycle_indices.to(dtype=batch.values.dtype).unsqueeze(-1)
        normalized_positions = normalized_positions / float(self.config.max_cycles - 1)
        position_tokens = self.cycle_position_encoder(normalized_positions)
        clean_conditions = torch.where(
            batch.condition_mask,
            batch.condition_values,
            torch.zeros_like(batch.condition_values),
        )
        condition_input = torch.cat(
            (clean_conditions, batch.condition_mask.to(dtype=batch.values.dtype)), dim=1
        )
        condition_tokens = self.condition_encoder(condition_input).unsqueeze(1)
        cycle_tokens = phase_tokens + position_tokens + condition_tokens
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, cycle_tokens), dim=1)
        tokens = tokens + self.position_encoding[:, : cycle_count + 1]
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (batch_size, 1), dtype=torch.bool, device=batch.values.device
                ),
                ~batch.cycle_mask,
            ),
            dim=1,
        )
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        cls_encoded = encoded[:, 0]
        cycle_encoded = encoded[:, 1:]
        scores = self.attention_score(cycle_encoded).squeeze(-1)
        scores = scores.masked_fill(~batch.cycle_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (cycle_encoded * weights).sum(dim=1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat((cls_encoded, pooled), dim=1)))
        fused: Tensor = gate * cls_encoded + (1.0 - gate) * pooled
        return fused


class CyclePatchLifeRegressor(nn.Module):
    """Return one standardized, task-agnostic scalar per cell."""

    def __init__(self, config: CyclePatchConfig, *, condition_count: int) -> None:
        super().__init__()
        self.encoder = CyclePatchEncoder(config, condition_count=condition_count)
        self.scalar_head = nn.Linear(config.d_model, 1)

    def forward(self, batch: EarlyCycleBatch) -> Tensor:
        scalar: Tensor = self.scalar_head(self.encoder(batch)).squeeze(dim=1)
        return scalar


def _sequence_schema(sequence: EarlyCycleSequence) -> tuple[object, ...]:
    return (
        sequence.dataset_id,
        sequence.data_version,
        sequence.feature_version,
        sequence.cutoff_cycle,
        sequence.cycle_indices,
        sequence.phase_names,
        sequence.variable_names,
        sequence.condition_names,
        sequence.normalization_version,
        sequence.normalization_statistics_sha256,
    )


def _require_tensor(value: object, name: str, dtype: torch.dtype) -> None:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype is not dtype:
        raise ValueError(f"{name} must use {dtype}")
