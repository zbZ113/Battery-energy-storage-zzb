"""Physics-structured monotone SOH decoding over CyclePatch token memory."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchEncoder,
    EarlyCycleBatch,
)


@dataclass(frozen=True)
class HybridPatchV2Config:
    cyclepatch: CyclePatchConfig
    query_token_count: int = 8
    query_layers: int = 1
    decoder_hidden_dim: int = 64
    huber_delta: float = 1.0
    lambda_history: float = 0.1
    lambda_smooth: float = 0.1
    lambda_order: float = 0.1
    lambda_residual: float = 0.01
    max_prediction_cycle: int = 500

    def __post_init__(self) -> None:
        if self.query_token_count not in {0, 8, 16}:
            raise ValueError("query_token_count must be one of 0, 8 or 16")
        if self.query_layers <= 0:
            raise ValueError("query_layers must be positive")
        if self.decoder_hidden_dim <= 0:
            raise ValueError("decoder_hidden_dim must be positive")
        if not math.isfinite(self.huber_delta) or self.huber_delta <= 0:
            raise ValueError("huber_delta must be finite and positive")
        for name in ("lambda_history", "lambda_smooth", "lambda_order"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.lambda_residual) or self.lambda_residual <= 0:
            raise ValueError("lambda_residual must be finite and positive")
        if self.max_prediction_cycle != 500:
            raise ValueError("max_prediction_cycle must be fixed to 500")


@dataclass(frozen=True, eq=False)
class HybridPatchV2Inputs:
    """Inference-only observations; future SOH supervision is intentionally absent."""

    early_batch: EarlyCycleBatch
    initial_soh: Tensor
    prediction_cycles: Tensor

    def __post_init__(self) -> None:
        batch_size, cycle_count = self.early_batch.cycle_mask.shape
        _require_tensor(self.initial_soh, "initial_soh")
        if self.initial_soh.dtype != self.early_batch.values.dtype:
            raise ValueError("initial_soh dtype must match early_batch values")
        if self.initial_soh.shape != (batch_size,):
            raise ValueError("initial_soh must have shape [batch]")
        if self.initial_soh.device != self.early_batch.values.device:
            raise ValueError("initial_soh device must match early_batch")
        torch._assert_async(
            (
                torch.isfinite(self.initial_soh)
                & (self.initial_soh > 0)
                & (self.initial_soh <= 1.5)
            ).all(),
            "initial_soh must be finite and in (0, 1.5]",
        )
        if not isinstance(self.prediction_cycles, Tensor):
            raise ValueError("prediction_cycles must be a tensor")
        if self.prediction_cycles.dtype is not torch.int64:
            raise ValueError("prediction_cycles must use int64")
        if self.prediction_cycles.ndim != 1 or self.prediction_cycles.numel() == 0:
            raise ValueError("prediction_cycles must be a non-empty [horizon] tensor")
        if self.prediction_cycles.device != self.early_batch.values.device:
            raise ValueError("prediction_cycles device must match early_batch")
        cutoff_cycle = cycle_count - 1
        torch._assert_async(
            (self.prediction_cycles > cutoff_cycle).all(),
            "prediction_cycles must be strictly after cutoff",
        )
        torch._assert_async(
            (self.prediction_cycles <= 500).all(),
            "prediction_cycles cannot exceed 500",
        )
        torch._assert_async(
            (self.prediction_cycles[1:] > self.prediction_cycles[:-1]).all(),
            "prediction_cycles must be strictly increasing",
        )


@dataclass(frozen=True, eq=False)
class HybridPatchV2Targets:
    """Loss-only real SOH supervision with explicit missingness."""

    history_soh: Tensor
    history_mask: Tensor
    target_soh: Tensor
    target_mask: Tensor

    def __post_init__(self) -> None:
        _validate_masked_soh(self.history_soh, self.history_mask, "history")
        _validate_masked_soh(self.target_soh, self.target_mask, "target")
        if self.history_soh.ndim != 2 or self.target_soh.ndim != 2:
            raise ValueError("history and target SOH must be rank two")
        if self.history_soh.shape[0] != self.target_soh.shape[0]:
            raise ValueError("history and target SOH batch axes must align")
        torch._assert_async(
            self.target_mask.any(dim=1).all(),
            "every target row requires at least one valid target",
        )
        tensors = (self.history_soh, self.history_mask, self.target_soh, self.target_mask)
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all supervision tensors must share a device")
        if self.history_soh.dtype != self.target_soh.dtype:
            raise ValueError("history and target SOH must share a dtype")


class ConditionAdaLN(nn.Module):
    def __init__(self, *, d_model: int, condition_count: int) -> None:
        super().__init__()
        if d_model <= 0 or condition_count <= 0:
            raise ValueError("d_model and condition_count must be positive")
        self.condition_count = condition_count
        self.normalization = nn.LayerNorm(d_model)
        self.modulation = nn.Linear(condition_count * 2, d_model * 2)

    def forward(self, x: Tensor, values: Tensor, mask: Tensor) -> Tensor:
        if values.shape != mask.shape or values.ndim != 2:
            raise ValueError("condition values and mask must align as [batch, condition]")
        if values.shape[1] != self.condition_count or mask.dtype is not torch.bool:
            raise ValueError("condition schema does not match ConditionAdaLN")
        clean = torch.where(mask, values, torch.zeros_like(values))
        condition = torch.cat((clean, mask.to(dtype=values.dtype)), dim=1)
        modulation: Tensor = self.modulation(condition)
        gamma, beta = modulation.chunk(2, dim=1)
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        normalized: Tensor = self.normalization(x)
        conditioned: Tensor = normalized * (1.0 + 0.1 * torch.tanh(gamma)) + beta
        return conditioned


class DegradationQueryEncoder(nn.Module):
    def __init__(self, config: HybridPatchV2Config) -> None:
        super().__init__()
        self.query_token_count = config.query_token_count
        d_model = config.cyclepatch.d_model
        if self.query_token_count == 0:
            self.summary_gate = nn.Linear(d_model * 2, d_model)
        else:
            self.query_tokens = nn.Parameter(
                torch.empty(1, self.query_token_count, d_model)
            )
            nn.init.trunc_normal_(self.query_tokens, mean=0.0, std=0.02)
            layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=config.cyclepatch.heads,
                dim_feedforward=d_model * 4,
                dropout=config.cyclepatch.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.query_decoder = nn.TransformerDecoder(
                layer, num_layers=config.query_layers
            )
            self.summary_gate = nn.Linear(d_model * 3, d_model)

    def forward(
        self,
        fused: Tensor,
        cycle_tokens: Tensor,
        cycle_mask: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        weights = cycle_mask.unsqueeze(-1).to(dtype=cycle_tokens.dtype)
        masked_summary = (cycle_tokens * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        if self.query_token_count == 0:
            gate = torch.sigmoid(
                self.summary_gate(torch.cat((fused, masked_summary), dim=1))
            )
            return gate * fused + (1.0 - gate) * masked_summary, None
        query = self.query_tokens.expand(cycle_tokens.shape[0], -1, -1)
        query_output: Tensor = self.query_decoder(
            query,
            cycle_tokens,
            memory_key_padding_mask=~cycle_mask,
        )
        query_summary = query_output.mean(dim=1)
        gate = torch.sigmoid(
            self.summary_gate(
                torch.cat((fused, masked_summary, query_summary), dim=1)
            )
        )
        combined = (masked_summary + query_summary) * 0.5
        return gate * fused + (1.0 - gate) * combined, query_output


@dataclass(frozen=True, eq=False)
class _DecoderOutput:
    predicted_soh: Tensor
    sqrt_degradation: Tensor
    linear_degradation: Tensor
    knee_degradation: Tensor
    residual_degradation: Tensor
    residual_increments: Tensor


class StructuralMonotoneSOHDecoder(nn.Module):
    def __init__(self, *, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 5),
        )
        self.residual_rate = nn.Sequential(
            nn.Linear(d_model + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: Tensor,
        initial_soh: Tensor,
        prediction_cycles: Tensor,
        *,
        cutoff_cycle: int,
    ) -> _DecoderOutput:
        tau = (prediction_cycles.to(dtype=context.dtype) - float(cutoff_cycle)) / float(
            500 - cutoff_cycle
        )
        raw = self.parameter_head(context)
        amplitudes = functional.softplus(raw[:, :4])
        knee_location = torch.sigmoid(raw[:, 4:5])
        tau_grid = tau.unsqueeze(0).expand(context.shape[0], -1)
        sqrt_degradation = amplitudes[:, 0:1] * torch.sqrt(tau_grid)
        linear_degradation = amplitudes[:, 1:2] * tau_grid
        knee_distance = torch.relu(tau_grid - knee_location)
        knee_degradation = amplitudes[:, 2:3] * knee_distance.square()
        context_grid = context.unsqueeze(1).expand(-1, tau.shape[0], -1)
        rate_input = torch.cat((context_grid, tau_grid.unsqueeze(-1)), dim=2)
        rate: Tensor = functional.softplus(self.residual_rate(rate_input).squeeze(-1))
        previous_tau = torch.cat((torch.zeros_like(tau[:1]), tau[:-1]))
        delta_tau = tau - previous_tau
        residual_increments = amplitudes[:, 3:4] * rate * delta_tau.unsqueeze(0)
        residual_degradation = torch.cumsum(residual_increments, dim=1)
        total = (
            sqrt_degradation
            + linear_degradation
            + knee_degradation
            + residual_degradation
        )
        predicted = torch.clamp(initial_soh.unsqueeze(1) - total, 0.0, 1.5)
        return _DecoderOutput(
            predicted_soh=predicted,
            sqrt_degradation=sqrt_degradation,
            linear_degradation=linear_degradation,
            knee_degradation=knee_degradation,
            residual_degradation=residual_degradation,
            residual_increments=residual_increments,
        )


@dataclass(frozen=True, eq=False)
class HybridPatchV2Output:
    predicted_soh: Tensor
    history_reconstruction: Tensor
    sqrt_degradation: Tensor
    linear_degradation: Tensor
    knee_degradation: Tensor
    residual_degradation: Tensor
    residual_increments: Tensor
    query_tokens: Tensor | None
    cycle_mask: Tensor


class HybridPatchV2Predictor(nn.Module):
    def __init__(self, config: HybridPatchV2Config, *, condition_count: int) -> None:
        super().__init__()
        self.config = config
        self.cycle_encoder = CyclePatchEncoder(
            config.cyclepatch, condition_count=condition_count
        )
        self.condition_adaln = ConditionAdaLN(
            d_model=config.cyclepatch.d_model,
            condition_count=condition_count,
        )
        self.query_encoder = DegradationQueryEncoder(config)
        self.history_head = nn.Linear(config.cyclepatch.d_model, 1)
        self.decoder = StructuralMonotoneSOHDecoder(
            d_model=config.cyclepatch.d_model,
            hidden_dim=config.decoder_hidden_dim,
        )

    def forward(self, inputs: HybridPatchV2Inputs) -> HybridPatchV2Output:
        encoding = self.cycle_encoder.encode(inputs.early_batch)
        conditioned_cycles = self.condition_adaln(
            encoding.cycle_tokens,
            inputs.early_batch.condition_values,
            inputs.early_batch.condition_mask,
        )
        conditioned_cycles = torch.where(
            encoding.cycle_mask.unsqueeze(-1),
            conditioned_cycles,
            torch.zeros_like(conditioned_cycles),
        )
        conditioned_fused = self.condition_adaln(
            encoding.fused,
            inputs.early_batch.condition_values,
            inputs.early_batch.condition_mask,
        )
        context, query_tokens = self.query_encoder(
            conditioned_fused, conditioned_cycles, encoding.cycle_mask
        )
        history: Tensor = self.history_head(conditioned_cycles).squeeze(-1)
        history = torch.where(encoding.cycle_mask, history, torch.zeros_like(history))
        decoded = self.decoder(
            context,
            inputs.initial_soh,
            inputs.prediction_cycles,
            cutoff_cycle=inputs.early_batch.cycle_mask.shape[1] - 1,
        )
        return HybridPatchV2Output(
            predicted_soh=decoded.predicted_soh,
            history_reconstruction=history,
            sqrt_degradation=decoded.sqrt_degradation,
            linear_degradation=decoded.linear_degradation,
            knee_degradation=decoded.knee_degradation,
            residual_degradation=decoded.residual_degradation,
            residual_increments=decoded.residual_increments,
            query_tokens=query_tokens,
            cycle_mask=encoding.cycle_mask,
        )


@dataclass(frozen=True, eq=False)
class HybridPatchV2Loss:
    total: Tensor
    trajectory: Tensor
    history: Tensor
    smooth: Tensor
    order: Tensor
    residual: Tensor


def compute_hybridpatch_v2_loss(
    output: HybridPatchV2Output,
    targets: HybridPatchV2Targets,
    inputs: HybridPatchV2Inputs,
    config: HybridPatchV2Config,
) -> HybridPatchV2Loss:
    if targets.target_soh.shape != output.predicted_soh.shape:
        raise ValueError("target SOH must align with predicted trajectory")
    if targets.history_soh.shape != output.history_reconstruction.shape:
        raise ValueError("history SOH must align with cycle reconstruction")
    if targets.target_soh.device != output.predicted_soh.device:
        raise ValueError("targets and output must share a device")
    if targets.target_soh.dtype != output.predicted_soh.dtype:
        raise ValueError("targets and output must share a dtype")
    torch._assert_async(
        ~(targets.history_mask & ~output.cycle_mask).any(),
        "history_mask cannot exceed early cycle observations",
    )
    torch._assert_async(
        targets.target_mask.any(dim=1).all(),
        "main target mask cannot be empty",
    )

    trajectory = _masked_smooth_l1_mean(
        output.predicted_soh,
        targets.target_soh,
        targets.target_mask,
        beta=config.huber_delta,
    )
    history_valid = targets.history_mask & output.cycle_mask
    history = _masked_smooth_l1_mean(
        output.history_reconstruction,
        targets.history_soh,
        history_valid,
        beta=config.huber_delta,
    )
    adjacent_valid = targets.target_mask[:, 1:] & targets.target_mask[:, :-1]
    order = _masked_mean(
        torch.relu(output.predicted_soh[:, 1:] - output.predicted_soh[:, :-1]),
        adjacent_valid,
    )
    triple_valid = (
        targets.target_mask[:, 2:]
        & targets.target_mask[:, 1:-1]
        & targets.target_mask[:, :-2]
    )
    cycles = inputs.prediction_cycles.to(dtype=output.predicted_soh.dtype)
    slopes = (output.predicted_soh[:, 1:] - output.predicted_soh[:, :-1]) / (
        cycles[1:] - cycles[:-1]
    )
    second_difference = slopes[:, 1:] - slopes[:, :-1]
    smooth = _masked_mean(second_difference.square(), triple_valid)
    residual = _masked_mean(
        output.residual_increments.square(), targets.target_mask
    )
    total = (
        trajectory
        + config.lambda_history * history
        + config.lambda_smooth * smooth
        + config.lambda_order * order
        + config.lambda_residual * residual
    )
    return HybridPatchV2Loss(
        total=total,
        trajectory=trajectory,
        history=history,
        smooth=smooth,
        order=order,
        residual=residual,
    )


def _validate_masked_soh(values: Tensor, mask: Tensor, label: str) -> None:
    _require_tensor(values, f"{label}_soh")
    if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
        raise ValueError(f"{label}_mask must be a bool tensor")
    if values.shape != mask.shape:
        raise ValueError(f"{label}_soh and {label}_mask must align")
    valid_observed = (~mask) | (
        torch.isfinite(values) & (values > 0) & (values <= 1.5)
    )
    torch._assert_async(
        valid_observed.all(),
        f"observed {label} SOH must be finite and in (0, 1.5]",
    )
    torch._assert_async(
        (mask | torch.isnan(values)).all(),
        f"masked {label} SOH must remain NaN",
    )


def _masked_smooth_l1_mean(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    beta: float,
) -> Tensor:
    safe_target = torch.where(mask, target, prediction.detach())
    elementwise = functional.smooth_l1_loss(
        prediction,
        safe_target,
        beta=beta,
        reduction="none",
    )
    return _masked_mean(elementwise, mask)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _require_tensor(value: object, name: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
