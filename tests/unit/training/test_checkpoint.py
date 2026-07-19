from __future__ import annotations

from pathlib import Path

import pytest
import torch

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.training.checkpoint import (
    AdvancedCheckpointContext,
    AdvancedTrainingCheckpointManifest,
    CheckpointContext,
    TrainingProgress,
    load_advanced_training_checkpoint,
    load_training_checkpoint,
    model_architecture_sha256,
    save_advanced_training_checkpoint,
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


def _advanced_context(
    *,
    run_mode: str = "final",
    model_name: str = "cyclepatch_batlinet",
    stage: str | None = None,
) -> AdvancedCheckpointContext:
    candidate_hash = "d" * 64
    default_stage = {
        "smoke": "smoke",
        "select": "selection_stage1",
        "final": "final",
    }[run_mode]
    payload = {
        **_context().model_dump(mode="python"),
        "model_name": model_name,
        "run_mode": run_mode,
        "stage": stage or default_stage,
        "candidate_config_sha256": candidate_hash,
        "model_architecture_sha256": model_architecture_sha256(
            torch.nn.Linear(2, 1), candidate_hash
        ),
        "normalization_sha256": "f" * 64,
        "selection_manifest_sha256": "1" * 64 if run_mode == "final" else None,
        "reference_library_sha256": ("2" * 64 if model_name == "cyclepatch_batlinet" else None),
    }
    return AdvancedCheckpointContext.model_validate(payload)


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
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer, patience=1)
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


def test_v1_manifest_serialization_and_hash_contract_remain_unchanged(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _trained_components()
    manifest = save_training_checkpoint(
        tmp_path,
        context=_context(),
        progress=TrainingProgress(epoch=1, global_step=1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    dumped = manifest.model_dump(mode="json")
    assert dumped["schema_version"] == "safe-training-checkpoint-v1"
    assert dumped["context"] == _context().model_dump(mode="json")
    assert set(dumped) == {
        "schema_version",
        "context",
        "progress",
        "files",
        "created_at",
        "manifest_sha256",
    }
    assert (
        sha256_canonical(manifest.model_dump(mode="json", exclude={"manifest_sha256"}))
        == manifest.manifest_sha256
    )


def test_advanced_checkpoint_round_trip_restores_all_state(tmp_path: Path) -> None:
    model, optimizer, scheduler = _trained_components()
    progress = TrainingProgress(
        epoch=7,
        global_step=19,
        best_epoch=5,
        best_metric=8.5,
        validations_without_improvement=2,
    )
    context = _advanced_context()
    manifest = save_advanced_training_checkpoint(
        tmp_path,
        context=context,
        progress=progress,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    restored_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(restored_optimizer, patience=1)
    restored = load_advanced_training_checkpoint(
        tmp_path,
        manifest,
        expected_context=context,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert isinstance(manifest, AdvancedTrainingCheckpointManifest)
    assert manifest.schema_version == "safe-training-checkpoint-v2"
    assert restored == progress
    for expected, actual in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(expected, actual)
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert restored_scheduler.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize(
    "field",
    [
        "candidate_config_sha256",
        "model_architecture_sha256",
        "normalization_sha256",
        "selection_manifest_sha256",
        "reference_library_sha256",
    ],
)
def test_advanced_checkpoint_rejects_changed_context_hash(
    tmp_path: Path,
    field: str,
) -> None:
    model, optimizer, scheduler = _trained_components()
    context = _advanced_context()
    manifest = save_advanced_training_checkpoint(
        tmp_path,
        context=context,
        progress=TrainingProgress(epoch=1, global_step=1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    changed = context.model_copy(update={field: "3" * 64})

    with pytest.raises(ValueError, match=r"context|model_architecture_sha256"):
        load_advanced_training_checkpoint(
            tmp_path,
            manifest,
            expected_context=changed,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_advanced_context_enforces_batlinet_reference_and_final_selection() -> None:
    assert (
        _advanced_context(
            run_mode="smoke", model_name="cyclepatch_direct"
        ).selection_manifest_sha256
        is None
    )

    with pytest.raises(ValueError, match="reference_library_sha256"):
        AdvancedCheckpointContext.model_validate(
            {
                **_advanced_context(model_name="cyclepatch_batlinet").model_dump(mode="python"),
                "reference_library_sha256": None,
            }
        )
    with pytest.raises(ValueError, match="reference_library_sha256"):
        AdvancedCheckpointContext.model_validate(
            {
                **_advanced_context(model_name="cyclepatch_direct").model_dump(mode="python"),
                "reference_library_sha256": "2" * 64,
            }
        )
    with pytest.raises(ValueError, match="selection_manifest_sha256"):
        AdvancedCheckpointContext.model_validate(
            {
                **_advanced_context(run_mode="final").model_dump(mode="python"),
                "selection_manifest_sha256": None,
            }
        )
    with pytest.raises(ValueError, match="selection_manifest_sha256"):
        AdvancedCheckpointContext.model_validate(
            {
                **_advanced_context(run_mode="select").model_dump(mode="python"),
                "selection_manifest_sha256": "1" * 64,
            }
        )


@pytest.mark.parametrize(
    "model_name",
    [
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    ],
)
def test_advanced_context_accepts_only_closed_model_families(model_name: str) -> None:
    assert _advanced_context(model_name=model_name).model_name == model_name

    with pytest.raises(ValueError, match="model_name"):
        AdvancedCheckpointContext.model_validate(
            {
                **_advanced_context().model_dump(mode="python"),
                "model_name": "cpmlp",
                "reference_library_sha256": None,
            }
        )


@pytest.mark.parametrize(
    ("run_mode", "stage"),
    [
        ("smoke", "smoke"),
        ("select", "selection_stage1"),
        ("select", "selection_stage2"),
        ("select", "selection_recheck"),
        ("final", "final"),
    ],
)
def test_advanced_context_accepts_only_approved_mode_stage_combinations(
    run_mode: str,
    stage: str,
) -> None:
    assert _advanced_context(run_mode=run_mode, stage=stage).stage == stage


@pytest.mark.parametrize(
    ("run_mode", "stage"),
    [
        ("smoke", "final"),
        ("smoke", "selection_stage1"),
        ("select", "smoke"),
        ("select", "final"),
        ("final", "smoke"),
        ("final", "selection_recheck"),
    ],
)
def test_advanced_context_rejects_cross_mode_stage_combinations(
    run_mode: str,
    stage: str,
) -> None:
    with pytest.raises(ValueError, match=r"run_mode|stage"):
        _advanced_context(run_mode=run_mode, stage=stage)


def test_model_architecture_hash_ignores_weights_but_binds_shape_and_candidate() -> None:
    torch.manual_seed(1)
    first = torch.nn.Linear(2, 1)
    torch.manual_seed(2)
    second = torch.nn.Linear(2, 1)
    wider = torch.nn.Linear(3, 1)

    first_hash = model_architecture_sha256(first, "a" * 64)

    assert first_hash == model_architecture_sha256(second, "a" * 64)
    assert first_hash != model_architecture_sha256(wider, "a" * 64)
    assert first_hash != model_architecture_sha256(first, "b" * 64)
    assert len(first_hash) == 64
    with pytest.raises(ValueError, match="candidate_config_sha256"):
        model_architecture_sha256(first, "not-a-hash")


class _ArchitectureTwinA(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)


class _ArchitectureTwinB(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)


class _BufferedArchitecture(torch.nn.Module):
    def __init__(self, size: int, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 1)
        self.register_buffer("axis", torch.zeros(size, dtype=dtype))


def test_model_architecture_hash_binds_full_class_identity() -> None:
    assert model_architecture_sha256(_ArchitectureTwinA(), "a" * 64) != model_architecture_sha256(
        _ArchitectureTwinB(), "a" * 64
    )


def test_model_architecture_hash_binds_buffer_schema_but_not_buffer_values() -> None:
    first = _BufferedArchitecture(3)
    same_schema = _BufferedArchitecture(3)
    same_schema.axis.fill_(99.0)
    different_shape = _BufferedArchitecture(4)
    different_dtype = _BufferedArchitecture(3, dtype=torch.float64)

    first_hash = model_architecture_sha256(first, "a" * 64)
    assert first_hash == model_architecture_sha256(same_schema, "a" * 64)
    assert first_hash != model_architecture_sha256(different_shape, "a" * 64)
    assert first_hash != model_architecture_sha256(different_dtype, "a" * 64)


def test_advanced_checkpoint_rejects_unlisted_file(tmp_path: Path) -> None:
    model, optimizer, scheduler = _trained_components()
    context = _advanced_context()
    manifest = save_advanced_training_checkpoint(
        tmp_path,
        context=context,
        progress=TrainingProgress(epoch=1, global_step=1),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    (tmp_path / "unsafe.pkl").write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="unexpected file"):
        load_advanced_training_checkpoint(
            tmp_path,
            manifest,
            expected_context=context,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )


def test_advanced_checkpoint_rejects_context_not_bound_to_model_architecture(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _trained_components()
    context = _advanced_context().model_copy(update={"model_architecture_sha256": "e" * 64})

    with pytest.raises(ValueError, match="model_architecture_sha256"):
        save_advanced_training_checkpoint(
            tmp_path,
            context=context,
            progress=TrainingProgress(epoch=1, global_step=1),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
