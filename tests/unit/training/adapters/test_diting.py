from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from quanxin_life.core import TrainingBlockedReason
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.diting import (
    DITING_LICENSE_STATUS,
    DITING_UPSTREAM_COMMIT,
    DITINGAdapter,
    compute_diting_losses,
    gaussian_mmd_loss,
    validate_diting_cell_cohorts,
)
from quanxin_life.training.adapters.registry import TrainingAdapterRegistry


def test_diting_binds_research_only_upstream_and_adapter_protocol() -> None:
    adapter = DITINGAdapter()

    assert DITING_UPSTREAM_COMMIT == "b67f48373c591ca62c030fca257ee94910c974ce"
    assert DITING_LICENSE_STATUS == "RESEARCH_ONLY_LICENSE_UNVERIFIED"
    assert isinstance(adapter, TrainingAdapter)
    assert adapter.adapter_version == "diting-offline-v1"
    assert adapter.readiness() is TrainingBlockedReason.BLOCKED_DATA_VIEW
    assert adapter.readiness(has_hust_view=True) is TrainingBlockedReason.BLOCKED_DEPENDENCY
    assert adapter.promotion_blocker is TrainingBlockedReason.BLOCKED_LICENSE
    registry = TrainingAdapterRegistry()
    registry.register("diting_cptransformer", adapter)
    assert registry.get("diting_cptransformer") is adapter


def test_diting_mmd_and_supervised_losses_follow_official_decomposition() -> None:
    source_predictions = torch.tensor([0.1, 0.8], requires_grad=True)
    source_targets = torch.tensor([0.0, 1.0])
    target_predictions = torch.tensor([0.4, 0.6], requires_grad=True)
    target_targets = torch.tensor([0.5, 0.5])
    source_embeddings = torch.tensor([[0.0, 0.0], [1.0, 1.0]], requires_grad=True)
    target_embeddings = torch.tensor([[0.1, 0.1], [0.9, 0.9]], requires_grad=True)

    losses = compute_diting_losses(
        source_predictions=source_predictions,
        source_targets=source_targets,
        target_predictions=target_predictions,
        target_targets=target_targets,
        source_embeddings=source_embeddings,
        target_embeddings=target_embeddings,
        domain_loss_weight=1.0,
        target_cohort="adaptation",
    )
    losses["total_loss"].backward()

    assert set(losses) == {
        "source_loss",
        "target_loss",
        "prediction_loss",
        "mmd_domain_loss",
        "total_loss",
    }
    assert torch.allclose(losses["prediction_loss"], losses["source_loss"] + losses["target_loss"])
    assert torch.allclose(
        losses["total_loss"],
        losses["prediction_loss"] + losses["mmd_domain_loss"],
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert source_predictions.grad is not None
    assert target_predictions.grad is not None
    assert source_embeddings.grad is not None

    assert torch.allclose(
        gaussian_mmd_loss(source_embeddings, source_embeddings),
        torch.zeros(()),
        atol=1e-6,
    )


def test_diting_zero_five_ten_shot_cohorts_are_cell_disjoint() -> None:
    validate_diting_cell_cohorts(
        source_training_cells=("source-train",),
        adaptation_cells={
            0: (),
            5: tuple(f"adapt-5-{index}" for index in range(5)),
            10: tuple(f"adapt-10-{index}" for index in range(10)),
        },
        test_cells={
            0: ("test-zero",),
            5: ("test-five",),
            10: ("test-ten",),
        },
        early_stopping_cells=("source-validation",),
    )

    with pytest.raises(ValueError, match=r"target test|overlap|disjoint"):
        validate_diting_cell_cohorts(
            source_training_cells=("source-train",),
            adaptation_cells={
                0: (),
                5: ("leaked", "a5-1", "a5-2", "a5-3", "a5-4"),
                10: tuple(f"a-{i}" for i in range(10)),
            },
            test_cells={0: ("test-0",), 5: ("leaked",), 10: ("test-10",)},
            early_stopping_cells=("source-validation",),
        )

    with pytest.raises(ValueError, match="early-stopping"):
        validate_diting_cell_cohorts(
            source_training_cells=("source-train",),
            adaptation_cells={
                0: (),
                5: tuple(f"a5-{i}" for i in range(5)),
                10: tuple(f"a10-{i}" for i in range(10)),
            },
            test_cells={0: ("test-0",), 5: ("test-5",), 10: ("test-10",)},
            early_stopping_cells=("a5-0",),
        )

    with pytest.raises(ValueError, match="source training and early-stopping"):
        validate_diting_cell_cohorts(
            source_training_cells=("shared",),
            adaptation_cells={
                0: (),
                5: tuple(f"a5-{i}" for i in range(5)),
                10: tuple(f"a10-{i}" for i in range(10)),
            },
            test_cells={0: ("test-0",), 5: ("test-5",), 10: ("test-10",)},
            early_stopping_cells=("shared",),
        )


def test_diting_rejects_test_cohort_as_supervised_target() -> None:
    tensors = torch.ones(1)
    with pytest.raises(ValueError, match="test labels"):
        compute_diting_losses(
            source_predictions=tensors,
            source_targets=tensors,
            target_predictions=tensors,
            target_targets=tensors,
            source_embeddings=torch.ones(1, 2),
            target_embeddings=torch.ones(1, 2),
            target_cohort="test",
        )


def test_diting_zero_shot_uses_unlabelled_target_embeddings_only() -> None:
    losses = compute_diting_losses(
        source_predictions=torch.tensor([0.2]),
        source_targets=torch.tensor([0.1]),
        target_predictions=torch.empty(0),
        target_targets=torch.empty(0),
        source_embeddings=torch.tensor([[0.0, 0.0]]),
        target_embeddings=torch.tensor([[0.1, 0.1]]),
        target_cohort="zero_shot",
    )
    assert torch.equal(losses["target_loss"], torch.zeros(()))
    assert torch.equal(losses["prediction_loss"], losses["source_loss"])


def test_diting_target_test_labels_cannot_enter_training_or_early_stopping() -> None:
    with pytest.raises(ValueError, match="target test"):
        validate_diting_cell_cohorts(
            source_training_cells=("source-train",),
            adaptation_cells={
                0: (),
                5: tuple(f"a5-{i}" for i in range(5)),
                10: tuple(f"a10-{i}" for i in range(10)),
            },
            test_cells={0: ("held-out",), 5: ("test-5",), 10: ("test-10",)},
            early_stopping_cells=("held-out",),
        )

    with pytest.raises(ValueError, match="target test"):
        validate_diting_cell_cohorts(
            source_training_cells=("held-out",),
            adaptation_cells={
                0: (),
                5: tuple(f"a5-{i}" for i in range(5)),
                10: tuple(f"a10-{i}" for i in range(10)),
            },
            test_cells={0: ("held-out",), 5: ("test-5",), 10: ("test-10",)},
            early_stopping_cells=("source-validation",),
        )


def test_diting_runtime_dependency_graph_bypasses_upstream_unsafe_io() -> None:
    module = inspect.getmodule(DITINGAdapter)
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
def test_diting_configs_keep_data_and_license_gates(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "diting" / f"{stage}.json").read_text(encoding="utf-8")
    )

    assert payload["enabled"] is False
    assert payload["blocked_reasons"] == ["BLOCKED_DATA_VIEW", "BLOCKED_DEPENDENCY"]
    assert payload["promotion_blocked_reasons"] == ["BLOCKED_LICENSE"]
    assert payload["upstream_commit"] == DITING_UPSTREAM_COMMIT
    assert payload["license_status"] == DITING_LICENSE_STATUS
    assert set(payload["loss_names"]) == {
        "prediction",
        "source",
        "target",
        "mmd_domain",
    }
    manifests = payload["shot_cell_manifests"]
    assert set(manifests) == {"zero_shot", "five_shot", "ten_shot"}
    paths = [path for cohort in manifests.values() for path in cohort.values()]
    assert len(paths) == len(set(paths))
    assert payload["source_training_cells_manifest"] not in paths
    assert payload["early_stopping_cells_manifest"] not in paths
    optimization = payload["optimization"]
    assert optimization == {
        **optimization,
        "learning_rate": 0.00005,
        "micro_batch_size": 32,
        "visible_gpu_count": 1,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 64,
    }
    assert payload["model"] == {
        **payload["model"],
        "d_model": 128,
        "n_heads": 4,
        "encoder_layers": 12,
        "d_ff": 256,
        "dropout": 0.0,
    }


def test_diting_training_matrix_matches_frozen_adapter_config() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "diting_cptransformer")
    assert entry["cutoff_cycle"] == 100
    assert entry["loss_names"] == ["prediction", "source", "target", "mmd_domain"]
    assert (
        entry["micro_batch_size"],
        entry["gradient_accumulation_steps"],
        entry["effective_batch_size"],
    ) == (32, 2, 64)
