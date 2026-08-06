"""Offline-safe adapter primitives for the reviewed PBT upstream."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import SafetensorError, safe_open
from torch import nn
from torch.utils.data import Dataset

from quanxin_life.core import SelectionMetricDirection, TrainingBlockedReason
from quanxin_life.data.dataset_bundle import ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.adapters.base import (
    EvalState,
    EvaluationResult,
    PredictionBatch,
    ResolvedTrainingConfig,
    TrainState,
    VerifiedUpstreamArtifact,
    validate_upstream_artifact,
)
from quanxin_life.training.batching import calculate_effective_batch_size
from quanxin_life.training.engine import EpochMetrics

PBT_UPSTREAM_COMMIT = "a2df9d36db3f57ab2c5686638952ad7715ea0646"
PBT_LICENSE_STATUS = "VERIFIED_LICENSE_PRESENT"

_REQUIRED_CONDITION_ARTIFACTS = frozenset(
    {"pca_components", "pca_mean", "condition_embeddings", "normalizer"}
)
_PBT_ARTIFACT_SUFFIXES = frozenset({".json", ".safetensors"})


class _PBTMoEBlock(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        d_ff: int | None = None,
        num_experts: int,
        num_general_experts: int,
        residual: bool,
    ) -> None:
        super().__init__()
        hidden_dim = output_dim if d_ff is None else d_ff
        if hidden_dim < 1:
            raise ValueError("PBT expert hidden width must be positive")
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_experts)
        )
        self.general_experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_general_experts)
        )
        self.residual = residual and input_dim == output_dim

    def forward(self, values: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        expert_outputs = torch.stack(tuple(expert(values) for expert in self.experts), dim=1)
        gate_shape = (gates.shape[0], gates.shape[1]) + (1,) * (values.ndim - 1)
        routed = torch.sum(expert_outputs * gates.reshape(gate_shape), dim=1)
        for expert in self.general_experts:
            routed = routed + expert(values)
        return routed + values if self.residual else routed


class _PBTInterCycleBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_experts: int,
        num_general_experts: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = _PBTMoEBlock(
            input_dim=d_model,
            output_dim=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_general_experts=num_general_experts,
            residual=True,
        )
        self.feed_forward_width = d_ff

    def forward(
        self,
        values: torch.Tensor,
        gates: torch.Tensor,
        valid_cycle_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm1(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_cycle_mask,
            need_weights=False,
        )
        return self.moe.forward(self.norm2(values + attended), gates)


class PBTModel(nn.Module):
    """Offline reconstruction of PBT's audited intra/inter-cycle MoE hierarchy."""

    def __init__(
        self,
        *,
        curve_length: int = 300,
        early_cycle_threshold: int = 100,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 5,
        d_ff: int = 32,
        dropout: float = 0.05,
        num_experts: int = 20,
        num_general_experts: int = 5,
        condition_embedding_dim: int = 4096,
        gate_d_ff: int = 512,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if d_model < 1 or num_experts < 2 or num_general_experts < 0:
            raise ValueError("PBT dimensions and expert counts are invalid")
        if curve_length < 1 or early_cycle_threshold < 1:
            raise ValueError("PBT curve dimensions must be positive")
        if e_layers < 0 or d_layers < 0 or n_heads < 1 or d_model % n_heads:
            raise ValueError("PBT layer counts and attention heads are invalid")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("PBT top_k must select an available expert")
        self.curve_length = curve_length
        self.early_cycle_threshold = early_cycle_threshold
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_experts = num_experts
        self.num_general_experts = num_general_experts
        self.top_k = top_k
        self.condition_projection = nn.Sequential(
            nn.Linear(condition_embedding_dim, gate_d_ff), nn.LeakyReLU()
        )
        self.gate = nn.Linear(gate_d_ff, num_experts, bias=False)
        self.flatten_intra_cycle = _PBTMoEBlock(
            input_dim=curve_length * 3,
            output_dim=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_general_experts=num_general_experts,
            residual=False,
        )
        self.intra_moe_layers = nn.ModuleList(
            _PBTMoEBlock(
                input_dim=d_model,
                output_dim=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                num_general_experts=num_general_experts,
                residual=True,
            )
            for _ in range(e_layers)
        )
        self.inter_moe_layers = nn.ModuleList(
            _PBTInterCycleBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                num_experts=num_experts,
                num_general_experts=num_general_experts,
            )
            for _ in range(d_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.regression_head = nn.Linear(d_model, 1)

    def forward(
        self,
        inputs: torch.Tensor,
        condition_embeddings: torch.Tensor | None,
        valid_cycle_mask: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition_embeddings is None:
            raise RuntimeError(
                f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}: frozen condition embedding is required"
            )
        if inputs.ndim != 4 or inputs.shape[2:] != (3, self.curve_length):
            raise ValueError("PBT inputs must be [batch, cycle, 3, curve_length]")
        if inputs.shape[1] > self.early_cycle_threshold:
            raise ValueError("PBT inputs exceed the configured early-cycle cutoff")
        if inputs.shape[0] != condition_embeddings.shape[0]:
            raise ValueError("PBT inputs and condition embeddings must align")
        if valid_cycle_mask is None:
            valid_cycle_mask = torch.ones(inputs.shape[:2], dtype=torch.bool, device=inputs.device)
        if valid_cycle_mask.shape != inputs.shape[:2] or valid_cycle_mask.dtype is not torch.bool:
            raise ValueError("PBT valid-cycle mask must align with inputs")
        if not bool(valid_cycle_mask.any(dim=1).all().item()):
            raise ValueError("PBT each sample requires an observed cycle")
        if expert_mask is None:
            expert_mask = torch.ones(
                (inputs.shape[0], self.num_experts), dtype=torch.bool, device=inputs.device
            )
        if expert_mask.shape != (inputs.shape[0], self.num_experts):
            raise ValueError("PBT expert mask must align with the configured experts")
        if bool(torch.any(expert_mask.sum(dim=1) < self.top_k).item()):
            raise ValueError("PBT each sample requires at least top_k active reviewed experts")
        dense_gates = torch.softmax(
            self.gate(self.condition_projection(condition_embeddings)), dim=1
        ) * expert_mask.to(inputs.dtype)
        if bool(torch.any(dense_gates.sum(dim=1) <= 0).item()):
            raise ValueError("PBT each sample requires an active reviewed expert")
        top_values, top_indices = torch.topk(dense_gates, self.top_k, dim=1)
        sparse_gates = torch.zeros_like(dense_gates).scatter(1, top_indices, top_values)
        gates = sparse_gates / sparse_gates.sum(dim=1, keepdim=True)
        hidden = self.flatten_intra_cycle(inputs.flatten(start_dim=2), gates)
        for layer in self.intra_moe_layers:
            hidden = layer(hidden, gates)
        for layer in self.inter_moe_layers:
            hidden = layer(hidden, gates, valid_cycle_mask)
        last_indices = valid_cycle_mask.sum(dim=1).to(torch.long) - 1
        row_indices = torch.arange(hidden.shape[0], device=hidden.device)
        embedding = self.output_norm(hidden[row_indices, last_indices])
        return self.regression_head(embedding).squeeze(-1), embedding, gates


class PBTAdapter:
    """Governed PBT adapter; model-view I/O stays in the repository pipeline."""

    adapter_version = "pbt-offline-v1"
    selection_metric_name = "validation_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = PBT_UPSTREAM_COMMIT
    license_status = PBT_LICENSE_STATUS

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        if resolved_config.model_family != "pbt":
            raise ValueError("PBT adapter received a different model family")
        return PBTModel()

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        del manifest
        raise RuntimeError(
            f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}: "
            "multi-domain HUST model View is unavailable"
        )

    def train_epoch(self, state: TrainState) -> EpochMetrics:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def validate(self, state: EvalState) -> EvaluationResult:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def predict(self, state: EvalState) -> PredictionBatch:
        del state
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def export_best(self, destination: Path) -> ArtifactManifest:
        del destination
        raise RuntimeError(f"{TrainingBlockedReason.BLOCKED_DATA_VIEW}")

    def readiness(self, available_artifacts: Set[str]) -> TrainingBlockedReason | None:
        if not available_artifacts >= _REQUIRED_CONDITION_ARTIFACTS:
            return TrainingBlockedReason.BLOCKED_DATA_VIEW
        return None

    def run_manifest_fields(self, resolved_config: ResolvedTrainingConfig) -> dict[str, str | int]:
        effective_batch_size = calculate_effective_batch_size(
            micro_batch_size=resolved_config.micro_batch_size,
            visible_gpu_count=1,
            gradient_accumulation_steps=resolved_config.gradient_accumulation_steps,
        )
        if effective_batch_size != resolved_config.effective_batch_size:
            raise ValueError("PBT single-card effective batch does not match task identity")
        return {
            "adapter_version": self.adapter_version,
            "upstream_commit": self.upstream_commit,
            "visible_gpu_count": 1,
            "micro_batch_size": resolved_config.micro_batch_size,
            "gradient_accumulation_steps": resolved_config.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
        }


def validate_pbt_artifacts(
    artifacts: Mapping[str, tuple[Path, str]],
) -> dict[str, VerifiedUpstreamArtifact]:
    """Validate the complete frozen condition context before any model use."""

    names = set(artifacts)
    if names != _REQUIRED_CONDITION_ARTIFACTS:
        missing = sorted(_REQUIRED_CONDITION_ARTIFACTS - names)
        extra = sorted(names - _REQUIRED_CONDITION_ARTIFACTS)
        raise ValueError(f"PBT condition artifacts differ; missing={missing}, extra={extra}")
    verified: dict[str, VerifiedUpstreamArtifact] = {}
    for name in sorted(artifacts):
        path, expected_sha256 = artifacts[name]
        suffix = path.suffix.lower()
        expected_suffix = ".json" if name == "normalizer" else ".safetensors"
        if suffix not in _PBT_ARTIFACT_SUFFIXES or suffix != expected_suffix:
            raise ValueError("PBT artifacts must use safetensors or JSON")
        verified[name] = validate_upstream_artifact(path, expected_sha256)
        try:
            if suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("PBT JSON artifact must contain an object")
            else:
                with safe_open(path, framework="pt", device="cpu") as handle:
                    if not list(handle.keys()):
                        raise ValueError("PBT safetensors artifact must contain tensors")
        except (OSError, UnicodeError, json.JSONDecodeError, SafetensorError) as exc:
            raise ValueError("PBT artifact is not valid JSON or safetensors") from exc
    return verified


def compute_pbt_losses(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    embeddings: torch.Tensor,
    augmented_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
    gate_probabilities: torch.Tensor,
    expert_mask: torch.Tensor,
    raw_gate_logits: torch.Tensor | None = None,
    temperature: float = 1.0,
    guidance_weight: float = 1.0,
    contrastive_weight: float = 1.0,
    alignment_weight: float = 1.0,
    load_balancing_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute the reviewed PBT objective without data-loader side effects."""

    _validate_same_shape(predictions, targets, "PBT predictions and targets")
    _validate_embedding_shapes(embeddings, augmented_embeddings, prompt_embeddings)
    _validate_gate_probabilities(gate_probabilities)
    if any(
        not bool(torch.isfinite(tensor).all().item())
        for tensor in (
            predictions,
            targets,
            embeddings,
            augmented_embeddings,
            prompt_embeddings,
            gate_probabilities,
        )
    ):
        raise ValueError("PBT loss tensors must be finite")
    if expert_mask.shape != gate_probabilities.shape or expert_mask.dtype is not torch.bool:
        raise ValueError("PBT expert mask must be boolean and align with gate probabilities")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("PBT contrastive temperature must be positive and finite")
    weights = (
        guidance_weight,
        contrastive_weight,
        alignment_weight,
        load_balancing_weight,
    )
    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("PBT loss weights must be finite and non-negative")

    label_loss = functional.mse_loss(predictions, targets)
    active_probability = torch.sum(
        gate_probabilities * expert_mask.to(gate_probabilities.dtype), dim=1
    )
    if bool(torch.any(active_probability <= 0).item()):
        raise ValueError("every PBT sample requires at least one active routed expert")
    inactive_probability = torch.sum(
        gate_probabilities * (~expert_mask).to(gate_probabilities.dtype), dim=1
    )
    if raw_gate_logits is not None:
        if raw_gate_logits.shape != gate_probabilities.shape or not bool(
            torch.isfinite(raw_gate_logits).all().item()
        ):
            raise ValueError("PBT raw gate logits must be finite and aligned")
        if bool(torch.any((~expert_mask).sum(dim=1) == 0).item()):
            raise ValueError("PBT guidance requires an inactive expert")
        guidance_loss = -(
            (raw_gate_logits * expert_mask).sum(dim=1)
            - (raw_gate_logits * (~expert_mask)).sum(dim=1)
        ).mean()
    else:
        guidance_loss = torch.mean(inactive_probability - active_probability)

    normalized = functional.normalize(embeddings, dim=-1)
    normalized_augmented = functional.normalize(augmented_embeddings, dim=-1)
    labels = torch.arange(embeddings.shape[0], device=embeddings.device)
    if raw_gate_logits is not None:
        pairwise_distance = torch.cdist(embeddings, augmented_embeddings) / temperature
        contrastive_loss = -(
            -pairwise_distance.diagonal() - torch.logsumexp(-pairwise_distance, dim=1)
        ).mean()
    else:
        contrastive_loss = functional.cross_entropy(
            normalized @ normalized_augmented.transpose(0, 1) / temperature,
            labels,
        )
    dimension_scale = math.sqrt(embeddings.shape[-1]) * temperature
    center_distances = torch.cdist(embeddings, prompt_embeddings) / dimension_scale
    instance_distances = torch.cdist(embeddings, augmented_embeddings) / dimension_scale
    alignment_loss = torch.mean(
        torch.diagonal(center_distances)
        - torch.logsumexp(center_distances, dim=1)
        + torch.diagonal(instance_distances)
        - torch.logsumexp(instance_distances, dim=1)
    )
    moe_load_balancing_loss = torch.mean(
        torch.sum(
            gate_probabilities
            * torch.log(gate_probabilities.clamp_min(1e-12))
            * expert_mask.to(gate_probabilities.dtype),
            dim=1,
        )
    )
    total_loss = (
        label_loss
        + guidance_weight * guidance_loss
        + contrastive_weight * contrastive_loss
        + alignment_weight * alignment_loss
        + load_balancing_weight * moe_load_balancing_loss
    )
    return {
        "label_loss": label_loss,
        "guidance_loss": guidance_loss,
        "contrastive_loss": contrastive_loss,
        "alignment_loss": alignment_loss,
        "moe_load_balancing_loss": moe_load_balancing_loss,
        "total_loss": total_loss,
    }


def pbt_routing_metrics(gate_probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
    _validate_gate_probabilities(gate_probabilities)
    utilization = gate_probabilities.mean(dim=0)
    entropy = -torch.sum(
        gate_probabilities * torch.log(gate_probabilities.clamp_min(1e-12)), dim=1
    ).mean()
    return {"expert_utilization": utilization, "gating_entropy": entropy}


def pbt_seen_unseen_metrics(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    seen_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    _validate_same_shape(predictions, targets, "PBT predictions and targets")
    if seen_mask.shape != predictions.shape or seen_mask.dtype is not torch.bool:
        raise ValueError("PBT seen mask must be boolean and align with predictions")
    if not bool(seen_mask.any().item()) or not bool((~seen_mask).any().item()):
        raise ValueError("PBT Seen/Unseen metrics require both cohorts")
    absolute_error = torch.abs(predictions - targets)
    return {
        "seen_mae": absolute_error[seen_mask].mean(),
        "unseen_mae": absolute_error[~seen_mask].mean(),
    }


def _validate_same_shape(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError(f"{label} must be non-empty and aligned")
    if not bool(torch.isfinite(left).all().item()) or not bool(torch.isfinite(right).all().item()):
        raise ValueError(f"{label} must be finite")


def _validate_embedding_shapes(
    embeddings: torch.Tensor,
    augmented_embeddings: torch.Tensor,
    prompt_embeddings: torch.Tensor,
) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("PBT contrastive batches require at least two embedding rows")
    if not (embeddings.shape == augmented_embeddings.shape == prompt_embeddings.shape):
        raise ValueError("PBT embeddings, augmentations and prompts must align")


def _validate_gate_probabilities(gate_probabilities: torch.Tensor) -> None:
    if gate_probabilities.ndim != 2 or gate_probabilities.shape[1] < 2:
        raise ValueError("PBT gate probabilities require at least two experts")
    if not bool(torch.isfinite(gate_probabilities).all().item()) or bool(
        torch.any(gate_probabilities < 0).item()
    ):
        raise ValueError("PBT gate probabilities must be finite and non-negative")
    row_sums = gate_probabilities.sum(dim=1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
        raise ValueError("PBT gate probabilities must sum to one")


__all__ = [
    "PBT_LICENSE_STATUS",
    "PBT_UPSTREAM_COMMIT",
    "PBTAdapter",
    "PBTModel",
    "compute_pbt_losses",
    "pbt_routing_metrics",
    "pbt_seen_unseen_metrics",
    "validate_pbt_artifacts",
]
