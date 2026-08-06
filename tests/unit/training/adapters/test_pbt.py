from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from quanxin_life.core import TrainingBlockedReason
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.pbt import (
    PBT_LICENSE_STATUS,
    PBT_UPSTREAM_COMMIT,
    PBTAdapter,
    PBTModel,
    compute_pbt_losses,
    pbt_routing_metrics,
    pbt_seen_unseen_metrics,
    validate_pbt_artifacts,
)
from quanxin_life.training.adapters.registry import TrainingAdapterRegistry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pbt_binds_reviewed_upstream_and_implements_adapter_protocol() -> None:
    adapter = PBTAdapter()

    assert PBT_UPSTREAM_COMMIT == "a2df9d36db3f57ab2c5686638952ad7715ea0646"
    assert PBT_LICENSE_STATUS == "VERIFIED_LICENSE_PRESENT"
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.adapter_version == "pbt-offline-v1"
    registry = TrainingAdapterRegistry()
    registry.register("pbt", adapter)
    assert registry.get("pbt") is adapter


def test_pbt_accepts_only_verified_non_executable_condition_artifacts(
    tmp_path: Path,
) -> None:
    artifacts: dict[str, tuple[Path, str]] = {}
    for name, suffix in {
        "pca_components": ".safetensors",
        "pca_mean": ".safetensors",
        "condition_embeddings": ".safetensors",
        "normalizer": ".json",
    }.items():
        path = tmp_path / f"{name}{suffix}"
        if suffix == ".safetensors":
            save_file({name: torch.ones((1, 2))}, path)
        else:
            path.write_text(json.dumps({"schema_version": "normalizer-v1"}), encoding="utf-8")
        artifacts[name] = (path, _sha256(path))

    verified = validate_pbt_artifacts(artifacts)

    assert tuple(verified) == (
        "condition_embeddings",
        "normalizer",
        "pca_components",
        "pca_mean",
    )

    unsafe = tmp_path / "condition_embeddings.pkl"
    unsafe.write_bytes(b"not executable, but the format is forbidden")
    artifacts["condition_embeddings"] = (unsafe, _sha256(unsafe))
    with pytest.raises(ValueError, match=r"safetensors|JSON|unsafe"):
        validate_pbt_artifacts(artifacts)

    corrupt = tmp_path / "condition_embeddings.safetensors"
    corrupt.write_bytes(b"not a safetensors document")
    artifacts["condition_embeddings"] = (corrupt, _sha256(corrupt))
    with pytest.raises(ValueError, match=r"safetensors|parse|valid"):
        validate_pbt_artifacts(artifacts)


def test_pbt_missing_frozen_embedding_is_explicitly_blocked() -> None:
    assert PBTAdapter().readiness({"normalizer", "pca_components", "pca_mean"}) is (
        TrainingBlockedReason.BLOCKED_DATA_VIEW
    )
    with pytest.raises(RuntimeError, match="BLOCKED_DATA_VIEW"):
        PBTModel()(torch.ones((2, 4)), None)


def test_pbt_model_preserves_reviewed_hierarchical_moe_stages() -> None:
    model = PBTModel(
        curve_length=6,
        early_cycle_threshold=4,
        d_model=16,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        num_experts=6,
        num_general_experts=2,
        condition_embedding_dim=8,
        gate_d_ff=16,
        top_k=2,
    )
    predictions, embeddings, gates = model(
        torch.ones((3, 4, 3, 6)),
        torch.ones((3, 8)),
        torch.ones((3, 4), dtype=torch.bool),
        torch.ones((3, 6), dtype=torch.bool),
    )

    assert predictions.shape == (3,)
    assert embeddings.shape == (3, 16)
    assert len(model.intra_moe_layers) == 2
    assert len(model.inter_moe_layers) == 1
    assert hasattr(model, "flatten_intra_cycle")
    assert hasattr(model, "regression_head")
    assert torch.equal((gates > 0).sum(dim=1), torch.full((3,), 2))
    assert torch.allclose(gates.sum(dim=1), torch.ones(3))


def test_pbt_defaults_bind_official_hyperparameter_mapping() -> None:
    model = PBTModel()

    assert model.curve_length == 300
    assert model.early_cycle_threshold == 100
    assert model.d_model == 128
    assert model.n_heads == 8
    assert len(model.intra_moe_layers) == 2
    assert len(model.inter_moe_layers) == 5
    assert model.num_experts == 20
    assert model.num_general_experts == 5


def test_pbt_rejects_fewer_active_experts_than_top_k() -> None:
    model = PBTModel(
        curve_length=6,
        early_cycle_threshold=4,
        d_model=16,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        num_experts=4,
        num_general_experts=1,
        condition_embedding_dim=8,
        gate_d_ff=16,
        top_k=2,
    )
    with pytest.raises(ValueError, match=r"top_k|active reviewed experts"):
        model(
            torch.ones((1, 4, 3, 6)),
            torch.ones((1, 8)),
            torch.ones((1, 4), dtype=torch.bool),
            torch.tensor([[True, False, False, False]]),
        )


def test_pbt_loss_and_routing_contract_covers_all_reviewed_terms() -> None:
    predictions = torch.tensor([0.2, 0.8], requires_grad=True)
    targets = torch.tensor([0.0, 1.0])
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    augmented = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    prompts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gates = torch.tensor([[0.8, 0.2], [0.3, 0.7]], requires_grad=True)
    expert_mask = torch.tensor([[True, False], [False, True]])

    losses = compute_pbt_losses(
        predictions=predictions,
        targets=targets,
        embeddings=embeddings,
        augmented_embeddings=augmented,
        prompt_embeddings=prompts,
        gate_probabilities=gates,
        expert_mask=expert_mask,
    )
    losses["total_loss"].backward()

    assert set(losses) == {
        "label_loss",
        "guidance_loss",
        "contrastive_loss",
        "alignment_loss",
        "moe_load_balancing_loss",
        "total_loss",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    assert predictions.grad is not None
    assert embeddings.grad is not None
    assert gates.grad is not None

    active_probability = torch.sum(gates * expert_mask, dim=1)
    inactive_probability = torch.sum(gates * (~expert_mask), dim=1)
    expected_guidance = torch.mean(inactive_probability - active_probability)
    expected_load_balance = torch.mean(torch.sum(gates * torch.log(gates) * expert_mask, dim=1))
    dimension_scale = embeddings.shape[-1] ** 0.5
    center_distances = torch.cdist(embeddings, prompts) / dimension_scale
    instance_distances = torch.cdist(embeddings, augmented) / dimension_scale
    expected_alignment = torch.mean(
        torch.diagonal(center_distances)
        - torch.logsumexp(center_distances, dim=1)
        + torch.diagonal(instance_distances)
        - torch.logsumexp(instance_distances, dim=1)
    )
    assert torch.allclose(losses["guidance_loss"], expected_guidance)
    assert torch.allclose(losses["moe_load_balancing_loss"], expected_load_balance)
    assert torch.allclose(losses["alignment_loss"], expected_alignment)

    routing = pbt_routing_metrics(gates.detach())
    assert set(routing) == {"expert_utilization", "gating_entropy"}
    assert routing["expert_utilization"].shape == (2,)
    assert torch.isfinite(routing["gating_entropy"])


def test_pbt_seen_unseen_metrics_are_reported_separately() -> None:
    metrics = pbt_seen_unseen_metrics(
        predictions=torch.tensor([1.0, 3.0, 7.0, 9.0]),
        targets=torch.tensor([2.0, 2.0, 8.0, 8.0]),
        seen_mask=torch.tensor([True, True, False, False]),
    )

    assert set(metrics) == {"seen_mae", "unseen_mae"}
    assert all(torch.isfinite(value) for value in metrics.values())


def test_pbt_runtime_dependency_graph_has_no_online_llm_or_unsafe_loader() -> None:
    module = inspect.getmodule(PBTAdapter)
    assert module is not None
    tree = ast.parse(inspect.getsource(module))
    imported_roots = {
        node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert imported_roots.isdisjoint({"transformers", "qwen", "BatteryML", "joblib", "pickle"})


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_pbt_configs_are_single_card_offline_and_explicitly_blocked(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "pbt" / f"{stage}.json").read_text(encoding="utf-8")
    )

    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW"]
    assert payload["upstream_commit"] == PBT_UPSTREAM_COMMIT
    assert payload["online_condition_embedding"] is False
    assert set(payload["required_condition_artifacts"]) == {
        "pca_components",
        "pca_mean",
        "condition_embeddings",
        "normalizer",
    }
    assert set(payload["loss_names"]) == {
        "label",
        "guidance",
        "contrastive",
        "alignment",
        "moe_load_balancing",
    }
    assert set(payload["metric_names"]) == {
        "expert_utilization",
        "gating_entropy",
        "seen_mae",
        "unseen_mae",
    }
    optimization = payload["optimization"]
    assert optimization == {
        **optimization,
        "micro_batch_size": 128,
        "visible_gpu_count": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 256,
    }
    assert payload["model"] == {
        **payload["model"],
        "d_model": 128,
        "n_heads": 8,
        "encoder_layers": 2,
        "decoder_layers": 5,
        "d_ff": 32,
        "dropout": 0.05,
        "num_experts": 20,
        "num_general_experts": 5,
    }


def test_pbt_training_matrix_matches_frozen_adapter_config() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "pbt")
    assert entry["cutoff_cycle"] == 100
    assert entry["loss_names"] == [
        "label",
        "guidance",
        "contrastive",
        "alignment",
        "moe_load_balancing",
    ]
    assert (
        entry["micro_batch_size"],
        entry["gradient_accumulation_steps"],
        entry["effective_batch_size"],
    ) == (128, 2, 256)


def test_pbt_guidance_and_contrastive_losses_match_frozen_upstream_equations() -> None:
    raw_gate_logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, -0.5, 1.5]])
    gate_probabilities = torch.softmax(raw_gate_logits, dim=1)
    expert_mask = torch.tensor([[True, False, False], [False, True, False]])
    embeddings = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    augmented = torch.tensor([[1.0, 0.0], [2.0, 2.0]])
    losses = compute_pbt_losses(
        predictions=torch.tensor([1.0, 2.0]), targets=torch.tensor([1.0, 2.0]),
        embeddings=embeddings, augmented_embeddings=augmented,
        prompt_embeddings=embeddings, gate_probabilities=gate_probabilities,
        raw_gate_logits=raw_gate_logits, expert_mask=expert_mask,
    )
    expected_guidance = -((raw_gate_logits * expert_mask).sum(dim=1) -
                          (raw_gate_logits * ~expert_mask).sum(dim=1)).mean()
    distances = torch.cdist(embeddings, augmented)
    expected_contrastive = -(-distances.diagonal() - torch.logsumexp(-distances, dim=1)).mean()
    assert torch.allclose(losses["guidance_loss"], expected_guidance)
    assert torch.allclose(losses["contrastive_loss"], expected_contrastive)


def test_pbt_d_ff_changes_expert_hidden_width() -> None:
    model = PBTModel(curve_length=4, early_cycle_threshold=3, d_model=8, n_heads=2,
                     e_layers=1, d_layers=1, d_ff=13, num_experts=3,
                     num_general_experts=1, condition_embedding_dim=6, gate_d_ff=5, top_k=2)
    assert model.intra_moe_layers[0].experts[0][0].out_features == 13
    assert model.inter_moe_layers[0].moe.experts[0][0].out_features == 13
