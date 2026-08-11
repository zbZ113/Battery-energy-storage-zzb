"""Offline-safe adapter primitives for the reviewed PBT upstream."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

from quanxin_life.core import SelectionMetricDirection, TrainingBlockedReason
from quanxin_life.data.dataset_bundle import ArtifactFile, ArtifactManifest
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
        if d_ff is not None and d_ff < 1:
            raise ValueError("PBT expert hidden width must be positive")
        if d_ff is None:
            self.experts = nn.ModuleList(
                nn.Sequential(nn.Linear(input_dim, output_dim)) for _ in range(num_experts)
            )
            self.general_experts = nn.ModuleList(
                nn.Sequential(nn.Linear(input_dim, output_dim))
                for _ in range(num_general_experts)
            )
        else:
            self.experts = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(input_dim, d_ff), nn.GELU(), nn.Linear(d_ff, output_dim)
                )
                for _ in range(num_experts)
            )
            self.general_experts = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(input_dim, d_ff), nn.GELU(), nn.Linear(d_ff, output_dim)
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

    target_log_mean: torch.Tensor
    target_log_std: torch.Tensor
    target_transform_fitted: torch.Tensor

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
        self.register_buffer("target_log_mean", torch.tensor(0.0))
        self.register_buffer("target_log_std", torch.tensor(1.0))
        self.register_buffer("target_transform_fitted", torch.tensor(False))

    def fit_target_transform(self, train_targets: torch.Tensor) -> None:
        values = train_targets.detach().to(dtype=torch.float32).reshape(-1)
        if values.numel() < 2 or not bool(torch.isfinite(values).all().item()):
            raise ValueError("PBT target transform requires at least two finite train targets")
        if bool(torch.any(values <= 0).item()):
            raise ValueError("PBT cycle-life targets must be positive")
        logs = torch.log(values)
        scale = logs.std(unbiased=False)
        if not bool(torch.isfinite(scale).item()) or float(scale) <= 0:
            raise ValueError("PBT train targets require non-zero log-scale variation")
        self.target_log_mean.copy_(logs.mean())
        self.target_log_std.copy_(scale)
        self.target_transform_fitted.fill_(True)

    def encode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if not bool(self.target_transform_fitted.item()):
            raise RuntimeError("PBT target transform is not fitted")
        values = targets.to(dtype=torch.float32)
        if not bool(torch.isfinite(values).all().item()) or bool(
            torch.any(values <= 0).item()
        ):
            raise ValueError("PBT cycle-life targets must be positive and finite")
        return (torch.log(values) - self.target_log_mean) / self.target_log_std

    def decode_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if not bool(self.target_transform_fitted.item()):
            raise RuntimeError("PBT target transform is not fitted")
        values = torch.exp(
            predictions.to(dtype=torch.float32) * self.target_log_std
            + self.target_log_mean
        )
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("PBT decoded cycle-life predictions must be finite")
        return values

    def target_transform_metadata(self) -> dict[str, object]:
        return {
            "method": "train_log_zscore_v1",
            "fitted_split": "train",
            "fitted": bool(self.target_transform_fitted.item()),
            "log_mean": float(self.target_log_mean.item()),
            "log_std": float(self.target_log_std.item()),
        }

    def forward(
        self,
        inputs: torch.Tensor,
        condition_embeddings: torch.Tensor | None,
        valid_cycle_mask: torch.Tensor | None = None,
        expert_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if condition_embeddings is None:
            raise ValueError("frozen condition embedding is required")
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


class PBTTensorDataset(Dataset[dict[str, Any]]):
    """Read-only PBT tensors and metadata from a v2 model-view directory."""

    def __init__(self, root: Path, manifest: ModelViewManifest) -> None:
        directory = Path(root).resolve(strict=True)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError("PBT model view root must be a regular directory")
        names = sorted(artifact.relative_path for artifact in manifest.artifacts)
        tensor_candidates = [name for name in names if name.endswith(".safetensors")]
        metadata_candidates = [name for name in names if name.endswith(".json")]
        if not tensor_candidates or not metadata_candidates:
            raise ValueError("PBT model view requires tensor and metadata artifacts")
        tensors: dict[str, torch.Tensor] | None = None
        tensor_path: Path | None = None
        for name in tensor_candidates:
            candidate = directory / name
            if candidate.is_symlink():
                raise ValueError("PBT model-view artifacts must not be symlinks")
            loaded = load_file(str(candidate), device="cpu")
            if {"curves", "condition_embeddings", "targets"} <= set(loaded):
                tensors = loaded
                tensor_path = candidate
                break
        metadata: dict[str, Any] | None = None
        metadata_path: Path | None = None
        for name in metadata_candidates:
            candidate = directory / name
            if candidate.is_symlink():
                raise ValueError("PBT model-view artifacts must not be symlinks")
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and {
                "entity_ids",
                "splits",
                "domains",
                "seen",
            } <= set(payload):
                metadata = payload
                metadata_path = candidate
                break
        if tensors is None or metadata is None or tensor_path is None or metadata_path is None:
            raise ValueError("PBT model-view tensor or metadata artifact is missing required keys")
        required = {"entity_ids", "splits", "domains", "seen"}
        if not required <= set(metadata):
            raise ValueError("PBT model-view metadata is incomplete")
        required_tensors = {
            "curves",
            "condition_embeddings",
            "targets",
            "valid_cycle_mask",
            "expert_mask",
        }
        if not required_tensors <= set(tensors):
            raise ValueError("PBT model-view tensors are incomplete")
        count = int(tensors["targets"].shape[0])
        if count < 1 or any(len(metadata[key]) != count for key in required):
            raise ValueError("PBT model-view metadata and tensors are misaligned")
        if tensors["curves"].ndim != 4 or tensors["curves"].shape[2] != 3:
            raise ValueError("PBT curves must be [row, cycle, 3, curve_length]")
        if tensors["condition_embeddings"].shape[0] != count:
            raise ValueError("PBT condition embeddings are misaligned")
        if tensors["valid_cycle_mask"].shape != tensors["curves"].shape[:2]:
            raise ValueError("PBT valid cycle mask is misaligned")
        if tensors["expert_mask"].shape[0] != count:
            raise ValueError("PBT expert mask is misaligned")
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors.values()):
            raise ValueError("PBT model-view tensors must be finite")
        self._tensors = {key: value.contiguous() for key, value in tensors.items()}
        self._metadata = metadata

    def __len__(self) -> int:
        return int(self._tensors["targets"].shape[0])

    def targets_for_split(self, split: str) -> torch.Tensor:
        indices = [
            index
            for index, value in enumerate(self._metadata["splits"])
            if str(value) == split
        ]
        if not indices:
            raise ValueError(f"PBT view has no rows for split {split!r}")
        return self._tensors["targets"][indices].detach().clone()

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "inputs": self._tensors["curves"][index],
            "condition_embeddings": self._tensors["condition_embeddings"][index],
            "targets": self._tensors["targets"][index],
            "valid_cycle_mask": self._tensors["valid_cycle_mask"][index].bool(),
            "expert_mask": self._tensors["expert_mask"][index].bool(),
            "entity_id": str(self._metadata["entity_ids"][index]),
            "split": str(self._metadata["splits"][index]),
            "domain": str(self._metadata["domains"][index]),
            "seen": bool(self._metadata["seen"][index]),
        }


class PBTTrainingTask:
    """TrainingEngine-compatible PBT task with safe tensor-only input."""

    def __init__(
        self,
        model: PBTModel,
        dataset: PBTTensorDataset,
        *,
        device: torch.device,
        micro_batch_size: int = 128,
        gradient_accumulation_steps: int = 2,
        cutoff_cycle: int = 100,
        learning_rate: float = 2.5e-5,
        weight_decay: float = 0.01,
    ) -> None:
        if micro_batch_size < 1 or gradient_accumulation_steps < 1:
            raise ValueError("PBT batch settings must be positive")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("PBT learning rate must be positive and finite")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("PBT weight decay must be finite and non-negative")
        self.model = model
        self.dataset = dataset
        self.device = device
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.cutoff_cycle = cutoff_cycle
        self.model.fit_target_transform(dataset.targets_for_split("train"))
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=min(1e-6, learning_rate),
        )
        self.model.to(device)

    def encode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return self.model.encode_targets(targets)

    def decode_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        return self.model.decode_predictions(predictions)

    def target_transform_metadata(self) -> dict[str, object]:
        return self.model.target_transform_metadata()

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader[dict[str, Any]]:
        indices = [
            index
            for index in range(len(self.dataset))
            if self.dataset[index]["split"] == split
        ]
        if not indices:
            raise ValueError(f"PBT view has no rows for split {split!r}")
        subset = torch.utils.data.Subset(self.dataset, indices)
        return DataLoader(subset, batch_size=self.micro_batch_size, shuffle=shuffle)

    def _forward_batch(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        inputs = batch["inputs"].to(self.device, dtype=torch.float32)
        conditions = batch["condition_embeddings"].to(self.device, dtype=torch.float32)
        targets = batch["targets"].to(self.device, dtype=torch.float32).reshape(-1)
        masks = batch["valid_cycle_mask"].to(self.device, dtype=torch.bool)
        expert_mask = batch["expert_mask"].to(self.device, dtype=torch.bool)
        normalized_predictions, embeddings, gates = self.model(
            inputs, conditions, masks, expert_mask
        )
        normalized_targets = self.model.encode_targets(targets)
        zeros = torch.zeros((), device=self.device, dtype=normalized_predictions.dtype)
        losses = {
            "label_loss": functional.mse_loss(
                normalized_predictions, normalized_targets
            ),
            "guidance_loss": zeros,
            "contrastive_loss": zeros,
            "alignment_loss": zeros,
            "moe_load_balancing_loss": torch.mean(
                torch.sum(gates * torch.log(gates.clamp_min(1e-12)) * expert_mask, dim=1)
            ),
        }
        losses["total_loss"] = losses["label_loss"]
        del embeddings
        predictions = self.model.decode_predictions(normalized_predictions.detach())
        return losses, predictions, targets.detach()

    def train_epoch(self, epoch: int, *, device: torch.device | None = None) -> EpochMetrics:
        del epoch
        if device is not None and device != self.device:
            self.device = device
            self.model.to(device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals: dict[str, float] = {}
        batches = 0
        for step, batch in enumerate(self._loader("train", shuffle=True), start=1):
            losses, _predictions, _targets = self._forward_batch(batch)
            torch.autograd.backward(losses["total_loss"] / self.gradient_accumulation_steps)
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            batches += 1
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
        if batches % self.gradient_accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return EpochMetrics(
            loss=totals["total_loss"] / batches,
            metrics={
                name: value / batches for name, value in totals.items() if name != "total_loss"
            },
        )

    def validate(self, epoch: int, *, split: Any = "validation") -> EvaluationResult:
        del epoch
        split_name = getattr(split, "value", split)
        self.model.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in self._loader(str(split_name), shuffle=False):
                losses, values, labels = self._forward_batch(batch)
                total += float(losses["total_loss"].cpu())
                count += 1
                predictions.append(values)
                targets.append(labels)
        pred = torch.cat(predictions)
        target = torch.cat(targets)
        metrics = _pbt_regression_metrics(pred, target)
        return EvaluationResult(loss=total / count, metrics=metrics)

    def predict(self, *, split: Any) -> PredictionBatch:
        split_name = getattr(split, "value", split)
        self.model.eval()
        ids: list[str] = []
        values: list[float] = []
        with torch.no_grad():
            for batch in self._loader(str(split_name), shuffle=False):
                _losses, predictions, _targets = self._forward_batch(batch)
                ids.extend(batch["entity_id"])
                values.extend(float(value) for value in predictions.cpu())
        return PredictionBatch(
            split=split,
            entity_ids=tuple(ids),
            values=tuple(values),
        )

    def predict_records(self, *, split: Any) -> tuple[dict[str, object], ...]:
        """Return auditable predictions without upgrading HUST counts to official EOL."""

        split_name = getattr(split, "value", split)
        self.model.eval()
        records: list[dict[str, object]] = []
        with torch.no_grad():
            for batch in self._loader(str(split_name), shuffle=False):
                _losses, predictions, targets = self._forward_batch(batch)
                for entity_id, domain, seen, prediction, target in zip(
                    batch["entity_id"],
                    batch["domain"],
                    batch["seen"],
                    predictions.cpu(),
                    targets.cpu(),
                    strict=True,
                ):
                    y_true = float(target)
                    y_pred = float(prediction)
                    error = y_pred - y_true
                    domain_name = str(domain)
                    record: dict[str, object] = {
                        "cell_id": str(entity_id),
                        "domain": domain_name,
                        "seen": bool(seen),
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "error": error,
                        "abs_error": abs(error),
                        "target_semantics": (
                            "matr_official_cycle_life"
                            if domain_name == "MATR"
                            else "source_derived_observed_cycle_count_not_official_eol"
                        ),
                    }
                    if domain_name == "MATR":
                        record.update(
                            rul_true=y_true - self.cutoff_cycle,
                            rul_pred=y_pred - self.cutoff_cycle,
                            rul_abs_error=abs(error),
                        )
                    records.append(record)
        if not records:
            raise ValueError(f"PBT view has no rows for split {split_name!r}")
        return tuple(records)

    @staticmethod
    def domain_metrics(records: tuple[dict[str, object], ...]) -> dict[str, dict[str, float]]:
        """Aggregate auditable MAE/RUL-MAE by domain without synthesising values."""

        grouped: dict[str, list[float]] = {}
        for record in records:
            domain = record.get("domain")
            error = record.get("abs_error")
            if not isinstance(domain, str) or not isinstance(error, (int, float)):
                raise ValueError("PBT prediction record lacks typed domain/error")
            grouped.setdefault(domain, []).append(float(error))
        result: dict[str, dict[str, float]] = {}
        for domain, errors in sorted(grouped.items()):
            if not errors:
                continue
            result[domain] = {"mae": math.fsum(errors) / len(errors)}
            if domain == "MATR":
                result[domain]["rul_mae"] = result[domain]["mae"]
        return result


def _pbt_regression_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    error = predictions - targets
    absolute = error.abs()
    rmse = float(torch.sqrt(torch.mean(error.square())))
    mae = float(torch.mean(absolute))
    denominator = targets.abs().clamp_min(1e-8)
    mape = float(torch.mean(absolute / denominator) * 100.0)
    centered = targets - targets.mean()
    total = float(torch.sum(centered.square()))
    r2 = float(1.0 - torch.sum(error.square()) / total) if total > 0 else 0.0
    median = float(torch.median(absolute))
    p90 = float(torch.quantile(absolute, 0.9)) if absolute.numel() > 1 else float(absolute.item())
    return {
        "validation_mae": mae,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "median_absolute_error": median,
        "p90_absolute_error": p90,
        "rul_mae": mae,
    }


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PBTAdapter:
    """Governed PBT adapter; model-view I/O stays in the repository pipeline."""

    adapter_version = "pbt-offline-v2"
    selection_metric_name = "validation_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = PBT_UPSTREAM_COMMIT
    license_status = PBT_LICENSE_STATUS

    def __init__(self, *, view_root: Path | None = None) -> None:
        self.view_root = None if view_root is None else Path(view_root)
        self._model: PBTModel | None = None
        self._dataset: PBTTensorDataset | None = None
        self._task: PBTTrainingTask | None = None

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> torch.nn.Module:
        if resolved_config.model_family != "pbt":
            raise ValueError("PBT adapter received a different model family")
        self._model = PBTModel(
            curve_length=int(getattr(resolved_config, "curve_length", 300)),
            early_cycle_threshold=int(
                getattr(resolved_config, "early_cycle_threshold", 100)
            ),
            d_model=int(getattr(resolved_config, "d_model", 128)),
            n_heads=int(getattr(resolved_config, "n_heads", 8)),
            e_layers=int(getattr(resolved_config, "e_layers", 2)),
            d_layers=int(getattr(resolved_config, "d_layers", 5)),
            d_ff=int(getattr(resolved_config, "d_ff", 32)),
            dropout=float(getattr(resolved_config, "dropout", 0.05)),
            num_experts=int(getattr(resolved_config, "num_experts", 20)),
            num_general_experts=int(
                getattr(resolved_config, "num_general_experts", 5)
            ),
            condition_embedding_dim=int(
                getattr(resolved_config, "condition_embedding_dim", 4096)
            ),
            gate_d_ff=int(getattr(resolved_config, "gate_d_ff", 512)),
            top_k=int(getattr(resolved_config, "top_k", 2)),
        )
        return self._model

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        if self.view_root is None:
            raise ValueError("PBTAdapter requires a frozen model-view root")
        if manifest.view_id != "pbt_multidomain_early_life":
            raise ValueError("manifest is not a PBT model view")
        self._dataset = PBTTensorDataset(self.view_root, manifest)
        return self._dataset

    def train_epoch(self, state: TrainState) -> EpochMetrics:
        if self._task is None:
            raise RuntimeError("PBT training task is not attached")
        return self._task.train_epoch(state.epoch, device=self._task.device)

    def validate(self, state: EvalState) -> EvaluationResult:
        if self._task is None:
            raise RuntimeError("PBT training task is not attached")
        result = self._task.validate(state.epoch, split=state.split)
        return EvaluationResult(loss=result.loss, metrics=result.metrics)

    def predict(self, state: EvalState) -> PredictionBatch:
        if self._task is None:
            raise RuntimeError("PBT training task is not attached")
        return self._task.predict(split=state.split)

    def export_best(self, destination: Path) -> ArtifactManifest:
        if self._model is None:
            raise RuntimeError("PBT model has not been built")
        root = Path(destination)
        target = root / "targets"
        target.mkdir(parents=True, exist_ok=True)
        model_path = target / "pbt_model.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self._model.state_dict().items()
            },
            str(model_path),
        )
        manifest_path = target / "pbt_model.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "pbt-safe-model-v1",
                    "adapter_version": self.adapter_version,
                    "upstream_commit": self.upstream_commit,
                    "weights": model_path.name,
                    "weights_sha256": _sha256(model_path),
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ArtifactManifest(
            dataset_id="PBT",
            dataset_version=self.adapter_version,
            files=(
                ArtifactFile(
                    relative_path="targets/pbt_model.json",
                    size_bytes=manifest_path.stat().st_size,
                    sha256=_sha256(manifest_path),
                ),
            ),
        )

    def attach_training(
        self,
        resolved_config: ResolvedTrainingConfig,
        *,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> PBTTrainingTask:
        if self._model is None:
            self.build_model(resolved_config)
        if self._dataset is None or self._model is None:
            raise ValueError("load_view must be called before attach_training")
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        self._task = PBTTrainingTask(
            self._model,
            self._dataset,
            device=device or torch.device("cpu"),
            micro_batch_size=int(getattr(resolved_config, "micro_batch_size", 128)),
            gradient_accumulation_steps=int(
                getattr(resolved_config, "gradient_accumulation_steps", 2)
            ),
            cutoff_cycle=int(getattr(resolved_config, "cutoff_cycle", 100) or 100),
            learning_rate=float(getattr(resolved_config, "learning_rate", 2.5e-5)),
            weight_decay=float(getattr(resolved_config, "weight_decay", 0.01)),
        )
        return self._task

    def save(self, destination: Path) -> Path:
        """Save model weights in the only executable-safe tensor format."""

        if self._model is None:
            raise RuntimeError("PBT model has not been built")
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        path = root / "model.safetensors"
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in self._model.state_dict().items()
            },
            str(path),
        )
        return path

    def restore(self, source: Path) -> None:
        """Restore a strictly matching tensor state; never deserialize code."""

        if self._model is None:
            raise RuntimeError("PBT model has not been built")
        path = Path(source)
        if path.is_symlink() or path.suffix.lower() != ".safetensors":
            raise ValueError("PBT restore requires a regular safetensors file")
        state = load_file(str(path), device="cpu")
        expected = self._model.state_dict()
        if set(state) != set(expected):
            raise ValueError("PBT checkpoint keys do not match the model")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError("PBT checkpoint tensor does not match the model")
        self._model.load_state_dict(state, strict=True)

    def readiness(self, available_artifacts: Set[str]) -> TrainingBlockedReason | None:
        if not available_artifacts >= _REQUIRED_CONDITION_ARTIFACTS:
            missing = sorted(_REQUIRED_CONDITION_ARTIFACTS - available_artifacts)
            raise ValueError(f"PBT condition artifacts are missing: {missing}")
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
