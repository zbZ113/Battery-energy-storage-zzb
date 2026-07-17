from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.training.checkpoint import (
    CheckpointContext,
    TrainingProgress,
    load_training_checkpoint,
    save_training_checkpoint,
)


def _context() -> CheckpointContext:
    return CheckpointContext(
        run_id="matr-cpmlp-cutoff50-seed20260712",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="cpmlp",
        cutoff_cycle=50,
        seed=20260712,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="matr-v1",
        split_version="split-v1",
        feature_version="feature-v1",
        source_commit="c" * 40,
    )


def _trained_components() -> tuple[
    torch.nn.Linear,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step(1.0)
    return model, optimizer, scheduler


def test_safe_checkpoint_restores_model_optimizer_scheduler_and_progress(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _trained_components()
    progress = TrainingProgress(
        epoch=5,
        global_step=5,
        best_epoch=5,
        best_metric=12.5,
        validations_without_improvement=0,
    )
    manifest = save_training_checkpoint(
        tmp_path,
        context=_context(),
        progress=progress,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        restored_optimizer, patience=1
    )
    restored_progress = load_training_checkpoint(
        tmp_path,
        manifest,
        expected_context=_context(),
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert restored_progress == progress
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_safe_checkpoint_rejects_tampered_tensor_bytes(tmp_path: Path) -> None:
    model, optimizer, scheduler = _trained_components()
    manifest = save_training_checkpoint(
        tmp_path,
        context=_context(),
        progress=TrainingProgress(epoch=1, global_step=1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match=r"size|SHA-256"):
        load_training_checkpoint(
            tmp_path,
            manifest,
            expected_context=_context(),
            model=torch.nn.Linear(2, 1),
            optimizer=torch.optim.AdamW(torch.nn.Linear(2, 1).parameters()),
            scheduler=None,
        )


def test_safe_checkpoint_rejects_context_mismatch(tmp_path: Path) -> None:
    model, optimizer, scheduler = _trained_components()
    manifest = save_training_checkpoint(
        tmp_path,
        context=_context(),
        progress=TrainingProgress(epoch=1, global_step=1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    with pytest.raises(ValueError, match="context"):
        load_training_checkpoint(
            tmp_path,
            manifest,
            expected_context=_context().model_copy(update={"seed": 9}),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
