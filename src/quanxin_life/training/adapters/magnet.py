"""Offline-safe MAGNet adapter bound to the reviewed upstream Informer and MLDG loop."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import sys
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import torch
import torch.nn.functional as functional
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from quanxin_life.core import SelectionMetricDirection
from quanxin_life.data.dataset_bundle import ArtifactFile, ArtifactManifest
from quanxin_life.data.model_views.schemas import ModelViewManifest
from quanxin_life.training.adapters.base import (
    EvalState,
    EvaluationResult,
    PredictionBatch,
    ResolvedTrainingConfig,
    TrainState,
)
from quanxin_life.training.engine import EpochMetrics

MAGNET_UPSTREAM_COMMIT = "aafb90c551d20748251a35fd51a34eae2539aaca"
MAGNET_LICENSE_STATUS = "VERIFIED_LICENSE_PRESENT"

_UPSTREAM_FILES = {
    "LICENSE": "91d504b475dcd3c3ca2838bcbd685e87353dd4b430910bbd2ec0c96273656b4d",
    "exp/exp_main.py": "6cacfe6d7bc334c1803d9051ccb348a89d21b9e0da4fa8fdea2318d78d38e4de",
    "layers/Embed.py": "e3c33e70fe36811a9065a3327c9e296395edfd0d5590e8213da7f69ce7647220",
    "layers/SelfAttention_Family.py": (
        "2f3431707f4571c3b4b37d58b51acef1bdade2d13f532d6611e1c0cea20403ae"
    ),
    "layers/Transformer_EncDec.py": (
        "6d2acbb669b602e17f2a70c7bcc8751a618b196b8f86a401a5c1c785ce536c1b"
    ),
    "models/Informer.py": "c24cc4c5e119e30972264b7d51f6e594f70b7a4818d199d612db9a55ca9c8a9f",
    "utils/masking.py": "ca43fb86366601261426d6acc78242bbe4ec42a8fc6ab2132f8e9b114a571286",
}
_CONDITION_FEATURES = (
    "temperature_c",
    "charge_rate_c",
    "discharge_rate_c",
    "dod_fraction",
    "soc_fraction",
)
_IMPORT_LOCK = threading.Lock()
_UPSTREAM_MODULE: ModuleType | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _upstream_root() -> Path:
    root = Path(__file__).resolve().parents[4] / "research" / "upstream" / "MAGNet"
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("reviewed MAGNet upstream source is unavailable")
    return root


def validate_magnet_upstream(root: Path | None = None) -> dict[str, str]:
    """Fail closed if any executable upstream source differs from the reviewed commit."""

    source_root = (root or _upstream_root()).resolve(strict=True)
    verified: dict[str, str] = {}
    for relative_path, expected in sorted(_UPSTREAM_FILES.items()):
        candidate = source_root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise RuntimeError(f"MAGNet upstream file is missing or unsafe: {relative_path}")
        actual = _sha256(candidate)
        if actual != expected:
            raise RuntimeError(f"MAGNet upstream SHA-256 mismatch: {relative_path}")
        verified[relative_path] = actual
    return verified


def _load_upstream_informer() -> ModuleType:
    global _UPSTREAM_MODULE
    with _IMPORT_LOCK:
        if _UPSTREAM_MODULE is not None:
            return _UPSTREAM_MODULE
        root = _upstream_root()
        validate_magnet_upstream(root)
        prior = sys.modules.get("models.Informer")
        if prior is not None:
            module_path = Path(str(getattr(prior, "__file__", ""))).resolve()
            if module_path != (root / "models" / "Informer.py").resolve():
                raise RuntimeError("a conflicting top-level models.Informer module is loaded")
            _UPSTREAM_MODULE = prior
            return prior
        temporary_modules: dict[str, ModuleType] = {}
        if "matplotlib" not in sys.modules and "matplotlib.pyplot" not in sys.modules:
            matplotlib = ModuleType("matplotlib")
            matplotlib.__path__ = []
            pyplot = ModuleType("matplotlib.pyplot")
            temporary_modules.update({"matplotlib": matplotlib, "matplotlib.pyplot": pyplot})
        if "reformer_pytorch" not in sys.modules:
            reformer = ModuleType("reformer_pytorch")

            class _UnavailableLSHSelfAttention(nn.Module):
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    super().__init__()
                    raise RuntimeError("Reformer is outside the reviewed MAGNet Informer runtime")

            reformer.LSHSelfAttention = _UnavailableLSHSelfAttention  # type: ignore[attr-defined]
            temporary_modules["reformer_pytorch"] = reformer
        sys.modules.update(temporary_modules)
        sys.path.insert(0, str(root))
        try:
            imported = importlib.import_module("models.Informer")
        finally:
            with suppress(ValueError):
                sys.path.remove(str(root))
            for name, module in temporary_modules.items():
                if sys.modules.get(name) is module:
                    del sys.modules[name]
        module_path = Path(str(imported.__file__)).resolve(strict=True)
        if module_path != (root / "models" / "Informer.py").resolve(strict=True):
            raise RuntimeError("MAGNet Informer import escaped the reviewed upstream")
        _UPSTREAM_MODULE = imported
        return imported


class MAGNetModel(nn.Module):
    """Exact reviewed upstream Informer core with its cycle-distance auxiliary head."""

    def __init__(
        self,
        *,
        seq_len: int = 20,
        label_len: int = 20,
        pred_len: int = 500,
        d_model: int = 12,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 4,
        factor: int = 5,
        factor2: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        values = (seq_len, label_len, pred_len, d_model, n_heads, e_layers, d_layers, d_ff)
        if any(value < 1 for value in values) or d_model % n_heads:
            raise ValueError("MAGNet dimensions and attention heads are invalid")
        if label_len > seq_len or factor < 1 or factor2 < 1:
            raise ValueError("MAGNet sequence and attention settings are invalid")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("MAGNet dropout must be in [0, 1)")
        config = SimpleNamespace(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=2,
            dec_in=2,
            c_out=2,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            factor=factor,
            factor2=factor2,
            dropout=dropout,
            embed="Cycle",
            freq="h",
            activation="gelu",
            output_attention=False,
            distil=False,
        )
        upstream = _load_upstream_informer()
        self.upstream_model = upstream.Model(config)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    @classmethod
    def from_training_config(cls, config: object) -> MAGNetModel:
        defaults: dict[str, int | float] = {
            "seq_len": 20,
            "label_len": 20,
            "pred_len": 500,
            "d_model": 12,
            "n_heads": 4,
            "e_layers": 2,
            "d_layers": 2,
            "d_ff": 4,
            "factor": 5,
            "factor2": 1,
            "dropout": 0.0,
        }
        prediction_horizon = getattr(config, "prediction_horizon", None)
        if prediction_horizon is not None:
            defaults["pred_len"] = int(prediction_horizon)
        kwargs = {name: getattr(config, name, default) for name, default in defaults.items()}
        return cls(**kwargs)  # type: ignore[arg-type]

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.upstream_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("reviewed MAGNet Informer returned an unexpected result")
        predictions, encoded, cycle_distance = output
        return predictions, encoded, cycle_distance


class MAGNetTensorDataset(Dataset[dict[str, Any]]):
    """Closed-world Qd/Ed trajectory view with explicit cell and condition identity."""

    _TENSOR_KEYS = frozenset(
        {
            "history",
            "history_cycle",
            "decoder_known",
            "targets",
            "target_cycle",
            "observation_mask",
            "cycle_distance",
            "condition_vectors",
        }
    )
    _METADATA_KEYS = frozenset(
        {
            "entity_ids",
            "condition_keys",
            "splits",
            "seen",
            "condition_feature_names",
            "protocols",
            "monotonic_applicable",
        }
    )

    def __init__(self, root: Path, manifest: ModelViewManifest) -> None:
        directory = Path(root).resolve(strict=True)
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("MAGNet model-view root must be a regular directory")
        for artifact in manifest.artifacts:
            candidate = directory / artifact.relative_path
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("MAGNet model-view artifact is missing or unsafe")
            if (
                candidate.stat().st_size != artifact.size_bytes
                or _sha256(candidate) != artifact.sha256
            ):
                raise ValueError("MAGNet model-view artifact differs from its manifest")
        tensor_files = [
            item.relative_path
            for item in manifest.artifacts
            if item.relative_path.endswith(".safetensors")
        ]
        metadata_files = [
            item.relative_path
            for item in manifest.artifacts
            if item.relative_path.endswith(".json")
        ]
        if len(tensor_files) != 1 or not metadata_files:
            raise ValueError("MAGNet view requires one tensor artifact and JSON metadata")
        tensors = load_file(str(directory / tensor_files[0]), device="cpu")
        matching_metadata: list[dict[str, Any]] = []
        for relative_path in metadata_files:
            candidate_payload = json.loads(
                (directory / relative_path).read_text(encoding="utf-8")
            )
            if (
                isinstance(candidate_payload, dict)
                and set(candidate_payload) == self._METADATA_KEYS
            ):
                matching_metadata.append(candidate_payload)
        if len(matching_metadata) != 1:
            raise ValueError("MAGNet view must contain exactly one typed metadata artifact")
        payload = matching_metadata[0]
        if set(tensors) != self._TENSOR_KEYS:
            raise ValueError("MAGNet tensor schema is incomplete or contains unknown fields")
        count = int(tensors["targets"].shape[0])
        if count < 1 or any(
            len(payload[key]) != count for key in self._METADATA_KEYS - {"condition_feature_names"}
        ):
            raise ValueError("MAGNet metadata and tensors are misaligned")
        if tuple(payload["condition_feature_names"]) != _CONDITION_FEATURES:
            raise ValueError("MAGNet condition vocabulary must include temperature/rates/DoD/SOC")
        self._validate_tensor_shapes(tensors, count)
        self._validate_split_isolation(payload)
        self._tensors = {name: tensor.contiguous() for name, tensor in tensors.items()}
        self._metadata = payload

    @staticmethod
    def _validate_tensor_shapes(tensors: Mapping[str, torch.Tensor], count: int) -> None:
        history = tensors["history"]
        targets = tensors["targets"]
        decoder = tensors["decoder_known"]
        if history.ndim != 3 or history.shape[0] != count or history.shape[-1] != 2:
            raise ValueError("MAGNet history must be [row, history, Qd/Ed]")
        if targets.ndim != 3 or targets.shape[0] != count or targets.shape[-1] != 2:
            raise ValueError("MAGNet targets must be [row, horizon, Qd/Ed]")
        if decoder.ndim != 3 or decoder.shape[0] != count or decoder.shape[-1] != 2:
            raise ValueError("MAGNet decoder-known values must be Qd/Ed trajectories")
        expected = {
            "history_cycle": (*history.shape[:2], 1),
            "target_cycle": (count, decoder.shape[1] + targets.shape[1], 1),
            "observation_mask": targets.shape,
            "cycle_distance": (count, 1),
            "condition_vectors": (count, len(_CONDITION_FEATURES)),
        }
        for name, shape in expected.items():
            if tuple(tensors[name].shape) != tuple(shape):
                raise ValueError(f"MAGNet {name} tensor has an invalid shape")
        if tensors["observation_mask"].dtype is not torch.bool:
            raise ValueError("MAGNet observation mask must be boolean")
        if not bool(tensors["observation_mask"].flatten(1).any(dim=1).all().item()):
            raise ValueError("MAGNet each row requires at least one observed target")
        for name, tensor in tensors.items():
            if tensor.dtype is not torch.bool and not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"MAGNet {name} tensor must be finite")

    @staticmethod
    def _validate_split_isolation(metadata: Mapping[str, Any]) -> None:
        allowed = {"train", "validation", "calibration", "test"}
        if any(split not in allowed for split in metadata["splits"]):
            raise ValueError("MAGNet view contains an unknown split")
        for key in ("entity_ids", "condition_keys"):
            memberships: dict[str, set[str]] = {}
            for identity, split in zip(metadata[key], metadata["splits"], strict=True):
                memberships.setdefault(str(identity), set()).add(str(split))
            leaking = sorted(
                identity for identity, splits in memberships.items() if len(splits) > 1
            )
            if leaking:
                raise ValueError(f"MAGNet {key} crosses splits: {leaking[:3]}")

    def __len__(self) -> int:
        return int(self._tensors["targets"].shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row: dict[str, Any] = {name: tensor[index] for name, tensor in self._tensors.items()}
        row.update(
            entity_id=str(self._metadata["entity_ids"][index]),
            condition_key=str(self._metadata["condition_keys"][index]),
            split=str(self._metadata["splits"][index]),
            seen=bool(self._metadata["seen"][index]),
            protocol=str(self._metadata["protocols"][index]),
            monotonic_applicable=bool(self._metadata["monotonic_applicable"][index]),
        )
        return row


def _clone_module(module: nn.Module, memo: dict[int, torch.Tensor] | None = None) -> nn.Module:
    """Port of the exact differentiable clone primitive used by upstream MAGNet."""

    if memo is None:
        memo = {}
    clone: Any = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    for key, parameter in module._parameters.items():
        if parameter is None:
            continue
        pointer = parameter.data_ptr()
        clone._parameters[key] = memo.setdefault(pointer, parameter.clone())
    for key, buffer in module._buffers.items():
        if buffer is not None and buffer.requires_grad:
            pointer = buffer.data_ptr()
            clone._buffers[key] = memo.setdefault(pointer, buffer.clone())
    for key, child in module._modules.items():
        clone._modules[key] = None if child is None else _clone_module(child, memo)
    if hasattr(clone, "flatten_parameters"):
        clone = clone._apply(lambda value: value)
    return cast(nn.Module, clone)


def _update_module(
    module: nn.Module,
    updates: list[torch.Tensor | None] | None = None,
    memo: dict[torch.Tensor, torch.Tensor] | None = None,
) -> nn.Module:
    if memo is None:
        memo = {}
    mutable: Any = module
    if updates is not None:
        parameters = list(module.parameters())
        if len(parameters) != len(updates):
            raise ValueError("MAGNet parameter updates are misaligned")
        for typed_parameter, update in zip(parameters, updates, strict=True):
            dynamic_parameter: Any = typed_parameter
            dynamic_parameter.update = update
    for key, raw_parameter in module._parameters.items():
        current_parameter = cast(Any, raw_parameter)
        if current_parameter in memo:
            mutable._parameters[key] = memo[current_parameter]
        elif (
            current_parameter is not None and getattr(current_parameter, "update", None) is not None
        ):
            updated = current_parameter + current_parameter.update
            current_parameter.update = None
            memo[current_parameter] = updated
            mutable._parameters[key] = updated
    for key, child in module._modules.items():
        mutable._modules[key] = None if child is None else _update_module(child, memo=memo)
    if hasattr(module, "flatten_parameters"):
        mutable._apply(lambda value: value)
    return module


def _maml_update(
    model: nn.Module,
    learning_rate: float,
    gradients: list[torch.Tensor | None],
) -> nn.Module:
    updates = [None if gradient is None else -learning_rate * gradient for gradient in gradients]
    return _update_module(model, updates)


class MAGNetTrainingTask:
    """TrainingEngine-compatible support/query MAGNet task."""

    def __init__(
        self,
        model: MAGNetModel,
        dataset: MAGNetTensorDataset,
        *,
        device: torch.device,
        micro_batch_size: int = 128,
        gradient_accumulation_steps: int = 1,
        seed: int = 38,
        inner_learning_rate: float = 1e-6,
        meta_learning_rate: float = 0.0075,
        meta_beta: float = 2.0,
        auxiliary_gamma: float = 0.2,
        weight_decay: float = 0.0,
    ) -> None:
        if micro_batch_size < 2 or gradient_accumulation_steps < 1:
            raise ValueError("MAGNet episodic batches require at least two rows")
        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        self.inner_learning_rate = inner_learning_rate
        self.meta_beta = meta_beta
        self.auxiliary_gamma = auxiliary_gamma
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=meta_learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _epoch: 1.0)
        train_vectors = [
            dataset[index]["condition_vectors"].float()
            for index in range(len(dataset))
            if dataset[index]["split"] == "train"
        ]
        if not train_vectors:
            raise ValueError("MAGNet view has no meta-train rows")
        self._train_condition_vectors = torch.stack(train_vectors)
        self._train_condition_mean = self._train_condition_vectors.mean(dim=0)
        self._train_condition_scale = self._train_condition_vectors.std(dim=0).clamp_min(1e-6)

    def _loader(self, split: str, *, shuffle: bool, epoch: int = 0) -> DataLoader[dict[str, Any]]:
        indices = [
            index for index in range(len(self.dataset)) if self.dataset[index]["split"] == split
        ]
        if not indices:
            raise ValueError(f"MAGNet view has no rows for split {split!r}")
        if split == "train":
            return DataLoader(
                self.dataset,
                batch_sampler=self._condition_batches(indices, epoch),
            )
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return DataLoader(
            Subset(self.dataset, indices),
            batch_size=self.micro_batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def _condition_batches(self, indices: list[int], epoch: int) -> list[list[int]]:
        """Build deterministic, no-drop episodes containing at least two conditions."""

        generator = random.Random(self.seed + epoch)
        grouped: dict[str, list[int]] = {}
        for index in indices:
            grouped.setdefault(self.dataset[index]["condition_key"], []).append(index)
        if len(grouped) < 2:
            raise ValueError("MAGNet meta-train requires at least two conditions")
        for values in grouped.values():
            generator.shuffle(values)
        condition_order = sorted(grouped)
        generator.shuffle(condition_order)
        interleaved: list[int] = []
        while any(grouped.values()):
            for condition in condition_order:
                if grouped[condition]:
                    interleaved.append(grouped[condition].pop())
        batches = [
            interleaved[start : start + self.micro_batch_size]
            for start in range(0, len(interleaved), self.micro_batch_size)
        ]
        if len(batches) > 1 and len(batches[-1]) == 1:
            batches[-1].insert(0, batches[-2].pop())
        for position, batch in enumerate(batches):
            if len({self.dataset[index]["condition_key"] for index in batch}) >= 2:
                continue
            only_condition = self.dataset[batch[0]]["condition_key"]
            repaired = False
            for donor_position, donor in enumerate(batches):
                if donor_position == position:
                    continue
                for donor_offset, donor_index in enumerate(donor):
                    if self.dataset[donor_index]["condition_key"] == only_condition:
                        continue
                    batch[0], donor[donor_offset] = donor_index, batch[0]
                    donor_conditions = {self.dataset[index]["condition_key"] for index in donor}
                    if len(donor_conditions) >= 2:
                        repaired = True
                        break
                    batch[0], donor[donor_offset] = donor[donor_offset], batch[0]
                if repaired:
                    break
            if not repaired:
                raise ValueError(
                    "MAGNet condition distribution cannot form no-drop support/query batches"
                )
        return batches

    @staticmethod
    def _subset(batch: Mapping[str, Any], mask: torch.Tensor) -> dict[str, Any]:
        indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
        result: dict[str, Any] = {}
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[name] = value[mask]
            else:
                result[name] = [value[index] for index in indices]
        return result

    def _forward(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history = batch["history"].to(self.device, dtype=torch.float32)
        history_cycle = batch["history_cycle"].to(self.device, dtype=torch.float32)
        known = batch["decoder_known"].to(self.device, dtype=torch.float32)
        targets = batch["targets"].to(self.device, dtype=torch.float32)
        target_cycle = batch["target_cycle"].to(self.device, dtype=torch.float32)
        decoder = torch.cat([known, torch.zeros_like(targets)], dim=1)
        output, _encoded, cycle_distance = model(history, history_cycle, decoder, target_cycle)
        return output[:, -targets.shape[1] :, :], cycle_distance

    def _loss(
        self, model: nn.Module, batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        predictions, cycle_predictions = self._forward(model, batch)
        targets = batch["targets"].to(self.device, dtype=torch.float32)
        mask = batch["observation_mask"].to(self.device, dtype=torch.bool)
        cycle_targets = batch["cycle_distance"].to(self.device, dtype=torch.float32)
        losses = compute_magnet_awmse(
            predictions=predictions,
            targets=targets,
            observation_mask=mask,
            condition_keys=tuple(batch["condition_key"]),
            cycle_distance_predictions=cycle_predictions,
            cycle_distance_targets=cycle_targets,
            auxiliary_gamma=self.auxiliary_gamma,
        )
        return losses["total_loss"], losses

    def _episode_masks(
        self, batch: Mapping[str, Any], epoch: int, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditions = sorted(set(batch["condition_key"]))
        if len(conditions) < 2:
            raise ValueError(
                "MAGNet support/query episode requires at least two conditions per batch"
            )
        generator = random.Random(self.seed + epoch * 1_000_003 + step)
        query_count = max(1, int(len(conditions) * 0.5))
        query_conditions = set(generator.sample(conditions, min(query_count, len(conditions) - 1)))
        query = torch.tensor([value in query_conditions for value in batch["condition_key"]])
        return ~query, query

    def train_epoch(self, epoch: int, *, device: torch.device | None = None) -> EpochMetrics:
        if device is not None and device != self.device:
            self.device = device
            self.model.to(device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {"support_loss": 0.0, "query_loss": 0.0, "meta_outer_loss": 0.0}
        batches = 0
        for step, batch in enumerate(self._loader("train", shuffle=True, epoch=epoch), start=1):
            support_mask, query_mask = self._episode_masks(batch, epoch, step)
            support = self._subset(batch, support_mask)
            query = self._subset(batch, query_mask)
            clone = _clone_module(self.model)
            support_loss, _support_parts = self._loss(clone, support)
            differentiable_parameters = [
                parameter for parameter in clone.parameters() if parameter.requires_grad
            ]
            gradients = torch.autograd.grad(
                support_loss,
                differentiable_parameters,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
            adapted = _maml_update(clone, self.inner_learning_rate, list(gradients))
            query_loss, _query_parts = self._loss(adapted, query)
            outer_loss = support_loss + self.meta_beta * query_loss
            torch.autograd.backward(outer_loss / self.gradient_accumulation_steps)
            if step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            batches += 1
            totals["support_loss"] += float(support_loss.detach().cpu())
            totals["query_loss"] += float(query_loss.detach().cpu())
            totals["meta_outer_loss"] += float(outer_loss.detach().cpu())
        if batches % self.gradient_accumulation_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        averaged = {name: value / batches for name, value in totals.items()}
        return EpochMetrics(loss=averaged["meta_outer_loss"], metrics=averaged)

    def _evaluate_tensors(
        self, split: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        metadata: list[dict[str, Any]] = []
        self.model.eval()
        split_offsets = {
            "train": 0,
            "validation": 1_000_003,
            "calibration": 2_000_003,
            "test": 3_000_017,
        }
        if split not in split_offsets:
            raise ValueError(f"unknown MAGNet evaluation split: {split!r}")
        evaluation_seed = self.seed + split_offsets[split]
        cuda_devices = (
            [self.device.index if self.device.index is not None else 0]
            if self.device.type == "cuda"
            else []
        )
        # Reviewed ProbAttention samples query/key indices even in eval mode. Isolate
        # that RNG so repeated evidence generation is deterministic and does not
        # perturb the training process RNG state.
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(evaluation_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(evaluation_seed)
            for batch in self._loader(split, shuffle=False):
                output, _cycle = self._forward(self.model, batch)
                predictions.append(output.cpu())
                targets.append(batch["targets"].cpu())
                masks.append(batch["observation_mask"].bool().cpu())
                for index, entity_id in enumerate(batch["entity_id"]):
                    metadata.append(
                        {
                            "entity_id": entity_id,
                            "condition_key": batch["condition_key"][index],
                            "condition_vector": batch["condition_vectors"][index].float().cpu(),
                            "seen": bool(batch["seen"][index]),
                            "protocol": batch["protocol"][index],
                            "monotonic_applicable": bool(batch["monotonic_applicable"][index]),
                        }
                    )
        return torch.cat(predictions), torch.cat(targets), torch.cat(masks), metadata

    def _evaluation_result(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> EvaluationResult:
        applicable = torch.tensor([bool(row["monotonic_applicable"]) for row in metadata])
        metrics = _trajectory_metrics(predictions, targets, mask, monotonic_applicable=applicable)
        seen_rows = torch.tensor([bool(row["seen"]) for row in metadata])
        absolute = torch.abs(predictions - targets)
        for cohort_name, rows in (("seen_condition", seen_rows), ("unseen_condition", ~seen_rows)):
            cohort_mask = mask & rows[:, None, None]
            if bool(cohort_mask.any().item()):
                metrics[f"{cohort_name}_observed_mae"] = float(absolute[cohort_mask].mean())
        loss = float((predictions - targets).square()[mask].mean())
        return EvaluationResult(loss=loss, metrics=metrics)

    def validate(self, epoch: int, *, split: Any = "validation") -> EvaluationResult:
        del epoch
        split_name = str(getattr(split, "value", split))
        evaluated = self._evaluate_tensors(split_name)
        return self._evaluation_result(*evaluated)

    def predict(self, *, split: Any) -> PredictionBatch:
        records = self.predict_records(split=split)
        return PredictionBatch(
            split=split,
            entity_ids=tuple(
                f"{row['entity_id']}|h{row['horizon']:04d}|{row['target_name']}" for row in records
            ),
            values=tuple(cast(float, row["y_pred"]) for row in records),
        )

    def _prediction_records_from_tensors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        metadata: list[dict[str, Any]],
    ) -> tuple[dict[str, object], ...]:
        train_normalized = (
            self._train_condition_vectors - self._train_condition_mean
        ) / self._train_condition_scale
        records: list[dict[str, object]] = []
        for row_index, row in enumerate(metadata):
            vector = (
                row["condition_vector"] - self._train_condition_mean
            ) / self._train_condition_scale
            condition_distance = float(torch.cdist(vector[None, :], train_normalized).min())
            for horizon in range(predictions.shape[1]):
                for channel, target_name in enumerate(("Qd", "Ed")):
                    if not bool(mask[row_index, horizon, channel]):
                        continue
                    y_true = float(targets[row_index, horizon, channel])
                    y_pred = float(predictions[row_index, horizon, channel])
                    records.append(
                        {
                            "entity_id": row["entity_id"],
                            "condition_key": row["condition_key"],
                            "protocol": row["protocol"],
                            "horizon": horizon + 1,
                            "target_name": target_name,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "error": y_pred - y_true,
                            "abs_error": abs(y_pred - y_true),
                            "condition_distance": condition_distance,
                            "ood_status": "IN_DOMAIN" if row["seen"] else "OOD",
                        }
                    )
        return tuple(records)

    def evaluate_with_records(
        self, *, split: Any
    ) -> tuple[EvaluationResult, tuple[dict[str, object], ...]]:
        """Generate metrics and auditable rows from one deterministic forward pass."""

        split_name = str(getattr(split, "value", split))
        evaluated = self._evaluate_tensors(split_name)
        return self._evaluation_result(*evaluated), self._prediction_records_from_tensors(
            *evaluated
        )

    def predict_records(self, *, split: Any) -> tuple[dict[str, object], ...]:
        split_name = str(getattr(split, "value", split))
        return self._prediction_records_from_tensors(*self._evaluate_tensors(split_name))


def _trajectory_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    monotonic_applicable: torch.Tensor | None = None,
) -> dict[str, float]:
    base = magnet_qd_ed_metrics(predictions, targets, mask)
    squared = (predictions - targets).square()
    metrics = {"validation_observed_mae": float(base["observed_mae"])}
    for channel, target_name in enumerate(("qd", "ed")):
        channel_mask = mask[..., channel]
        if bool(channel_mask.any().item()):
            metrics[f"{target_name}_mae"] = float(base[f"{target_name}_mae"])
            metrics[f"{target_name}_rmse"] = float(
                torch.sqrt(squared[..., channel][channel_mask].mean())
            )
    for horizon in range(predictions.shape[1]):
        horizon_mask = mask[:, horizon, :]
        if bool(horizon_mask.any().item()):
            metrics[f"horizon_{horizon + 1:04d}_mae"] = float(
                torch.abs(predictions[:, horizon, :] - targets[:, horizon, :])[horizon_mask].mean()
            )
    if monotonic_applicable is None:
        monotonic_applicable = torch.ones(predictions.shape[0], dtype=torch.bool)
    if monotonic_applicable.shape != (predictions.shape[0],):
        raise ValueError("MAGNet monotonic applicability flags must align with rows")
    if bool(monotonic_applicable.any().item()) and predictions.shape[1] > 1:
        qd_differences = (
            predictions[monotonic_applicable, 1:, 0] - predictions[monotonic_applicable, :-1, 0]
        )
        metrics["monotonic_violation_rate"] = float((qd_differences > 0).float().mean())
    return metrics


def compute_magnet_awmse(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
    condition_keys: tuple[str, ...],
    cycle_distance_predictions: torch.Tensor,
    cycle_distance_targets: torch.Tensor,
    auxiliary_gamma: float,
) -> dict[str, torch.Tensor]:
    """Upstream AWMSE: equal condition weighting plus TDDG cycle-distance loss."""

    if predictions.shape != targets.shape or observation_mask.shape != predictions.shape:
        raise ValueError("MAGNet AWMSE forecast tensors must align")
    if observation_mask.dtype is not torch.bool or len(condition_keys) != predictions.shape[0]:
        raise ValueError("MAGNet AWMSE mask and condition identities must align")
    if cycle_distance_predictions.shape != cycle_distance_targets.shape:
        raise ValueError("MAGNet cycle-distance tensors must align")
    if not math.isfinite(auxiliary_gamma) or auxiliary_gamma < 0:
        raise ValueError("MAGNet auxiliary_gamma must be finite and non-negative")
    raw = (predictions - targets).square()
    condition_losses: list[torch.Tensor] = []
    for condition in sorted(set(condition_keys)):
        rows = torch.tensor(
            [value == condition for value in condition_keys],
            dtype=torch.bool,
            device=predictions.device,
        )
        condition_mask = observation_mask[rows]
        if not bool(condition_mask.any().item()):
            raise ValueError("MAGNet each condition requires observed Qd/Ed targets")
        condition_losses.append(raw[rows][condition_mask].mean())
    forecast_loss = torch.stack(condition_losses).mean()
    cycle_distance_loss = functional.mse_loss(cycle_distance_predictions, cycle_distance_targets)
    return {
        "forecast_loss": forecast_loss,
        "cycle_distance_loss": cycle_distance_loss,
        "total_loss": forecast_loss + auxiliary_gamma * cycle_distance_loss,
    }


class MAGNetAdapter:
    adapter_version = "magnet-upstream-informer-mldg-v2"
    selection_metric_name = "validation_observed_mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE
    upstream_commit = MAGNET_UPSTREAM_COMMIT
    license_status = MAGNET_LICENSE_STATUS

    def __init__(self, *, view_root: Path | None = None) -> None:
        self.view_root = view_root
        self._model: MAGNetModel | None = None
        self._dataset: MAGNetTensorDataset | None = None
        self._task: MAGNetTrainingTask | None = None

    def build_model(self, resolved_config: ResolvedTrainingConfig) -> MAGNetModel:
        if resolved_config.model_family != "magnet":
            raise ValueError("MAGNet adapter received a different model family")
        self._model = MAGNetModel.from_training_config(resolved_config)
        return self._model

    def build(self, resolved_config: ResolvedTrainingConfig) -> MAGNetModel:
        return self.build_model(resolved_config)

    def load_view(self, manifest: ModelViewManifest) -> Dataset[Any]:
        if self.view_root is None:
            raise ValueError("MAGNetAdapter requires a frozen model-view root")
        if manifest.view_id != "magnet_multicondition_qd_ed":
            raise ValueError("manifest is not a MAGNet condition view")
        self._dataset = MAGNetTensorDataset(self.view_root, manifest)
        return self._dataset

    def attach_training(
        self,
        resolved_config: ResolvedTrainingConfig,
        *,
        device: torch.device | None = None,
        seed: int | None = None,
    ) -> MAGNetTrainingTask:
        if self._model is None:
            self.build_model(resolved_config)
        if self._model is None or self._dataset is None:
            raise ValueError("load_view must be called before attach_training")
        selected_seed = int(seed if seed is not None else getattr(resolved_config, "seed", 38))
        random.seed(selected_seed)
        torch.manual_seed(selected_seed)
        self._task = MAGNetTrainingTask(
            self._model,
            self._dataset,
            device=device or torch.device("cpu"),
            micro_batch_size=int(getattr(resolved_config, "micro_batch_size", 128)),
            gradient_accumulation_steps=int(
                getattr(resolved_config, "gradient_accumulation_steps", 1)
            ),
            seed=selected_seed,
        )
        return self._task

    def train_epoch(self, state: TrainState) -> EpochMetrics:
        if self._task is None:
            raise RuntimeError("MAGNet training task is not attached")
        return self._task.train_epoch(state.epoch, device=self._task.device)

    def validate(self, state: EvalState) -> EvaluationResult:
        if self._task is None:
            raise RuntimeError("MAGNet training task is not attached")
        return self._task.validate(state.epoch, split=state.split)

    def predict(self, state: EvalState) -> PredictionBatch:
        if self._task is None:
            raise RuntimeError("MAGNet training task is not attached")
        return self._task.predict(split=state.split)

    def save(self, destination: Path) -> Path:
        if self._model is None:
            raise RuntimeError("MAGNet model has not been built")
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
        if self._model is None:
            raise RuntimeError("MAGNet model has not been built")
        path = Path(source)
        if path.is_symlink() or not path.is_file() or path.suffix.lower() != ".safetensors":
            raise ValueError("MAGNet restore requires a regular safetensors file")
        state = load_file(str(path), device="cpu")
        expected = self._model.state_dict()
        if set(state) != set(expected):
            raise ValueError("MAGNet checkpoint keys do not match the model")
        for name, value in state.items():
            if value.shape != expected[name].shape or value.dtype != expected[name].dtype:
                raise ValueError("MAGNet checkpoint tensor does not match the model")
        self._model.load_state_dict(state, strict=True)

    def export_best(self, destination: Path) -> ArtifactManifest:
        root = Path(destination)
        target = root / "targets"
        target.mkdir(parents=True, exist_ok=True)
        saved = self.save(target)
        model_path = target / "magnet_model.safetensors"
        saved.replace(model_path)
        manifest_path = target / "magnet_model.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "magnet-safe-model-v1",
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
            dataset_id="MAGNet",
            dataset_version=self.adapter_version,
            files=(
                ArtifactFile(
                    relative_path="targets/magnet_model.json",
                    size_bytes=manifest_path.stat().st_size,
                    sha256=_sha256(manifest_path),
                ),
            ),
        )

    def export(self, destination: Path) -> ArtifactManifest:
        return self.export_best(destination)

    def readiness(self, *, has_multi_condition_view: bool) -> None:
        if not has_multi_condition_view:
            raise ValueError("MAGNet requires a committed multi-condition model view")


def compute_magnet_losses(
    *,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
    soc_markers: torch.Tensor,
    cutoff_voltage: float,
    meta_train_loss: torch.Tensor,
    meta_test_loss: torch.Tensor,
    meta_beta: float,
    proportion_weight: float,
    voltage_weight: float,
) -> dict[str, torch.Tensor]:
    """Compatibility helper for the reviewed optional physics regularizers."""

    if predictions.shape != targets.shape or predictions.shape[-1] != 2:
        raise ValueError("MAGNet Qd/Ed predictions and targets must align")
    if observation_mask.shape != predictions.shape or observation_mask.dtype is not torch.bool:
        raise ValueError("MAGNet observation mask must align")
    if not bool(observation_mask.flatten(1).any(dim=1).all().item()):
        raise ValueError("MAGNet each sample requires at least one observed target")
    if soc_markers.shape != (*predictions.shape[:-1], 1):
        raise ValueError("MAGNet SOC markers must align")
    weights = (meta_beta, proportion_weight, voltage_weight, cutoff_voltage)
    if any(not math.isfinite(value) or value < 0 for value in weights[:-1]):
        raise ValueError("MAGNet loss settings must be finite and non-negative")
    if any(
        tensor.ndim != 0 or not bool(torch.isfinite(tensor).item())
        for tensor in (meta_train_loss, meta_test_loss)
    ):
        raise ValueError("MAGNet meta losses must be finite scalar tensors")
    if not bool(torch.isfinite(predictions).all().item()) or not bool(
        torch.isfinite(targets).all().item()
    ):
        raise ValueError("MAGNet prediction tensors must be finite")
    raw_mse = (predictions - targets).square()[observation_mask].mean()
    predicted_qd = predictions[..., 0]
    qd_max = predicted_qd.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    predicted_ratio = predicted_qd / qd_max
    queried_soc = 100.0 - soc_markers.squeeze(-1)
    soc_max = queried_soc.max(dim=1, keepdim=True).values.clamp_min(1e-12)
    soc_ratio = queried_soc / soc_max
    soc_proportion_loss = functional.mse_loss(predicted_ratio, soc_ratio)
    predicted_voltage = predictions[..., 1]
    minimum_voltage = predicted_voltage.min(dim=1).values
    cutoff_voltage_loss = functional.mse_loss(
        minimum_voltage, torch.full_like(minimum_voltage, cutoff_voltage)
    )
    total_loss = (
        meta_train_loss
        + meta_beta * meta_test_loss
        + proportion_weight * soc_proportion_loss
        + voltage_weight * cutoff_voltage_loss
    )
    return {
        "raw_mse": raw_mse,
        "soc_proportion_loss": soc_proportion_loss,
        "cutoff_voltage_loss": cutoff_voltage_loss,
        "meta_train_loss": meta_train_loss,
        "meta_test_loss": meta_test_loss,
        "total_loss": total_loss,
    }


def magnet_qd_ed_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observation_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        predictions.shape != targets.shape
        or observation_mask.shape != predictions.shape
        or observation_mask.dtype is not torch.bool
    ):
        raise ValueError("MAGNet metric tensors must align")
    if not bool(observation_mask.flatten(1).any(dim=1).all().item()):
        raise ValueError("MAGNet each sample requires at least one observed target")
    absolute = torch.abs(predictions - targets)
    metrics = {"observed_mae": absolute[observation_mask].mean()}
    for channel, target_name in enumerate(("qd", "ed")):
        channel_mask = observation_mask[..., channel]
        if bool(channel_mask.any().item()):
            metrics[f"{target_name}_mae"] = absolute[..., channel][channel_mask].mean()
    return metrics


__all__ = [
    "MAGNET_LICENSE_STATUS",
    "MAGNET_UPSTREAM_COMMIT",
    "MAGNetAdapter",
    "MAGNetModel",
    "MAGNetTensorDataset",
    "MAGNetTrainingTask",
    "compute_magnet_awmse",
    "compute_magnet_losses",
    "magnet_qd_ed_metrics",
    "validate_magnet_upstream",
]
