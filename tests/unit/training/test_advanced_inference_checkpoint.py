from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import quanxin_life.training.checkpoint as checkpoint_module
from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.training.checkpoint import (
    AdvancedCheckpointContext,
    AdvancedTrainingCheckpointManifest,
    TrainingProgress,
    load_advanced_inference_checkpoint,
    model_architecture_sha256,
    save_advanced_training_checkpoint,
)


def _context(model: torch.nn.Module) -> AdvancedCheckpointContext:
    candidate_sha256 = "d" * 64
    return AdvancedCheckpointContext(
        run_id="matr-cyclepatch-direct-c50-s38",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name="cyclepatch_direct",
        cutoff_cycle=50,
        seed=38,
        config_sha256="a" * 64,
        input_bundle_sha256="b" * 64,
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-cell-level-v1",
        feature_version="advanced-feature-v1",
        source_commit="c" * 40,
        run_mode="final",
        stage="final",
        candidate_config_sha256=candidate_sha256,
        model_architecture_sha256=model_architecture_sha256(model, candidate_sha256),
        normalization_sha256="e" * 64,
        selection_manifest_sha256="f" * 64,
    )


def _saved_checkpoint(
    root: Path,
) -> tuple[torch.nn.Linear, AdvancedCheckpointContext, AdvancedTrainingCheckpointManifest]:
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    loss = model(torch.tensor([[1.0, 2.0]])).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step(1.0)
    context = _context(model)
    manifest = save_advanced_training_checkpoint(
        root,
        context=context,
        progress=TrainingProgress(
            epoch=7,
            global_step=19,
            best_epoch=5,
            best_metric=8.5,
            validations_without_improvement=2,
        ),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return model, context, manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _resign_file(
    root: Path,
    manifest: AdvancedTrainingCheckpointManifest,
    relative_path: str,
) -> AdvancedTrainingCheckpointManifest:
    payload = json.loads(manifest.model_dump_json())
    changed_path = root / relative_path
    for item in payload["files"]:
        if item["relative_path"] == relative_path:
            item["size_bytes"] = changed_path.stat().st_size
            item["sha256"] = _sha256_file(changed_path)
            break
    else:  # pragma: no cover - test helper invariant
        raise AssertionError(f"unregistered checkpoint file: {relative_path}")
    payload_without_hash = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    payload["manifest_sha256"] = sha256_canonical(payload_without_hash)
    _write_json(root / "manifest.json", payload)
    return AdvancedTrainingCheckpointManifest.model_validate(payload)


def _assert_numpy_rng_equal(
    first: tuple[Any, ...],
    second: tuple[Any, ...],
) -> None:
    assert first[0] == second[0]
    assert np.array_equal(first[1], second[1])
    assert first[2:] == second[2:]


def test_inference_checkpoint_restores_only_model_and_verified_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, context, manifest = _saved_checkpoint(tmp_path)
    restored = torch.nn.Linear(2, 1)
    loaded_paths: list[Path] = []
    real_load_file = checkpoint_module.load_file

    def recording_load_file(filename: str, *, device: str) -> dict[str, torch.Tensor]:
        loaded_paths.append(Path(filename))
        return real_load_file(filename, device=device)

    monkeypatch.setattr(checkpoint_module, "load_file", recording_load_file)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()

    progress = load_advanced_inference_checkpoint(
        tmp_path,
        manifest,
        expected_context=context,
        model=restored,
    )

    assert progress == manifest.progress
    assert loaded_paths == [tmp_path / "model.safetensors"]
    assert random.getstate() == python_rng
    _assert_numpy_rng_equal(np.random.get_state(), numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    for expected, actual in zip(trained.parameters(), restored.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_inference_checkpoint_rejects_context_mismatch(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="context"):
        load_advanced_inference_checkpoint(
            tmp_path,
            manifest,
            expected_context=context.model_copy(update={"seed": 39}),
            model=torch.nn.Linear(2, 1),
        )


def test_inference_checkpoint_rejects_manifest_self_hash_mismatch(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    changed = manifest.model_copy(update={"manifest_sha256": "0" * 64})

    with pytest.raises(ValueError, match=r"manifest SHA-256|manifest hash"):
        load_advanced_inference_checkpoint(
            tmp_path,
            changed,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )


def test_inference_checkpoint_rejects_unregistered_dangerous_file(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    (tmp_path / "untrusted.pkl").write_bytes(b"must never be deserialized")

    with pytest.raises(ValueError, match=r"unexpected file|file inventory"):
        load_advanced_inference_checkpoint(
            tmp_path,
            manifest,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "model.safetensors",
        "optimizer.safetensors",
        "optimizer_state.json",
        "scheduler_state.json",
        "rng_state.safetensors",
        "progress.json",
    ],
)
def test_inference_checkpoint_rejects_changed_size_for_every_registered_file(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    path = tmp_path / relative_path
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="size"):
        load_advanced_inference_checkpoint(
            tmp_path,
            manifest,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )


def test_inference_checkpoint_rejects_same_size_file_sha_mismatch(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    path = tmp_path / "scheduler_state.json"
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 1
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA-256"):
        load_advanced_inference_checkpoint(
            tmp_path,
            manifest,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )


def test_inference_checkpoint_rejects_progress_not_matching_manifest(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    progress_path = tmp_path / "progress.json"
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    payload["progress"]["global_step"] += 1
    _write_json(progress_path, payload)
    resigned = _resign_file(tmp_path, manifest, "progress.json")

    with pytest.raises(ValueError, match="progress"):
        load_advanced_inference_checkpoint(
            tmp_path,
            resigned,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )


def test_inference_checkpoint_rejects_model_architecture_mismatch(tmp_path: Path) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="model_architecture_sha256"):
        load_advanced_inference_checkpoint(
            tmp_path,
            manifest,
            expected_context=context,
            model=torch.nn.Linear(3, 1),
        )


@pytest.mark.parametrize(
    "invalid_state",
    [
        {"unexpected.weight": torch.zeros(1, 2), "bias": torch.zeros(1)},
        {"weight": torch.zeros(1, 3), "bias": torch.zeros(1)},
    ],
    ids=["state-keys", "tensor-shape"],
)
def test_inference_checkpoint_rejects_state_schema_mismatch(
    tmp_path: Path,
    invalid_state: dict[str, torch.Tensor],
) -> None:
    _, context, manifest = _saved_checkpoint(tmp_path)
    save_file(invalid_state, str(tmp_path / "model.safetensors"))
    resigned = _resign_file(tmp_path, manifest, "model.safetensors")

    with pytest.raises(ValueError, match=r"keys|tensor|architecture"):
        load_advanced_inference_checkpoint(
            tmp_path,
            resigned,
            expected_context=context,
            model=torch.nn.Linear(2, 1),
        )
