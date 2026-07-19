from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file, save_file

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.features.early_cycle_sequence import EarlyCycleSequence
from quanxin_life.features.multichannel_cycle import MultichannelCycleConfig
from quanxin_life.training.advanced_cache import (
    EarlyCycleSequenceCache,
    EarlySequenceCacheContext,
    EarlySequenceCacheMetadata,
)


def _context(*, source_sha256: str = "a" * 64) -> EarlySequenceCacheContext:
    return EarlySequenceCacheContext.from_multichannel_config(
        source_parquet_sha256=source_sha256,
        dataset_id="MATR",
        cell_id="cell-001",
        data_version="matr-three-batch-v1",
        config=MultichannelCycleConfig(
            cutoff_cycle=20,
            feature_version="multichannel-v1",
        ),
    )


def _sequence(context: EarlySequenceCacheContext) -> EarlyCycleSequence:
    cycle_count = context.cutoff_cycle + 1
    values = torch.full((cycle_count, 2, 150, 3), float("nan"), dtype=torch.float32)
    sample_mask = torch.zeros((cycle_count, 2, 150), dtype=torch.bool)
    values[1, 0, :, 0] = torch.linspace(3.0, 4.2, 150)
    values[1, 0, :, 1] = 1.0
    values[1, 0, :, 2] = torch.linspace(0.0, 1.0, 150)
    sample_mask[1, 0] = True
    return EarlyCycleSequence(
        dataset_id=context.dataset_id,
        cell_id=context.cell_id,
        cutoff_cycle=context.cutoff_cycle,
        data_version=context.data_version,
        feature_version=context.feature_version,
        cycle_indices=tuple(range(cycle_count)),
        values=values,
        cycle_mask=sample_mask.any(dim=(1, 2)),
        sample_mask=sample_mask,
        condition_names=(
            "mean_temperature_c",
            "mean_charge_current_a",
            "mean_discharge_current_a",
        ),
        condition_values=torch.tensor([25.0, 1.0, float("nan")]),
        condition_mask=torch.tensor([True, True, False]),
    )


def test_cache_round_trip_is_content_addressed_and_reuses_without_builder(
    tmp_path: Path,
) -> None:
    cache = EarlyCycleSequenceCache(tmp_path)
    context = _context()
    calls = 0

    def builder() -> EarlyCycleSequence:
        nonlocal calls
        calls += 1
        return _sequence(context)

    first = cache.get_or_build(context, builder)
    second = cache.get_or_build(context, builder)

    assert calls == 1
    assert second.input_hash == first.input_hash
    assert torch.equal(second.sample_mask, first.sample_mask)
    object_root = tmp_path / "objects" / cache.cache_key(context, first.input_hash)
    assert {path.name for path in object_root.iterdir()} == {
        "sequence.safetensors",
        "metadata.json",
        "manifest.json",
    }
    metadata_text = (object_root / "metadata.json").read_text(encoding="utf-8")
    assert "label" not in metadata_text.lower()
    assert "soh" not in metadata_text.lower()
    assert "future" not in metadata_text.lower()


def test_cache_key_binds_source_context_config_and_sequence_hash(tmp_path: Path) -> None:
    cache = EarlyCycleSequenceCache(tmp_path)
    first_context = _context(source_sha256="a" * 64)
    second_context = _context(source_sha256="b" * 64)
    first = cache.get_or_build(first_context, lambda: _sequence(first_context))
    second = cache.get_or_build(second_context, lambda: _sequence(second_context))

    assert first.input_hash == second.input_hash
    assert cache.cache_key(first_context, first.input_hash) != cache.cache_key(
        second_context,
        second.input_hash,
    )
    assert first_context.context_sha256 != second_context.context_sha256


def test_cache_metadata_structurally_forbids_labels_soh_and_future_fields() -> None:
    context = _context()
    sequence = _sequence(context)
    payload = {
        "schema_version": "early-sequence-cache-metadata-v1",
        **context.model_dump(mode="json"),
        "context_sha256": context.context_sha256,
        "cache_key": "b" * 64,
        "sequence_input_hash": sequence.input_hash,
        "cycle_indices": list(sequence.cycle_indices),
        "phase_names": list(sequence.phase_names),
        "variable_names": list(sequence.variable_names),
        "condition_names": list(sequence.condition_names),
        "normalization_version": sequence.normalization_version,
        "normalization_statistics_sha256": None,
    }
    for field in ("official_life_label", "future_soh", "labels"):
        with pytest.raises(ValidationError):
            EarlySequenceCacheMetadata.model_validate({**payload, field: 100})


def test_cache_load_rejects_closed_world_and_byte_tampering(tmp_path: Path) -> None:
    cache = EarlyCycleSequenceCache(tmp_path)
    context = _context()
    sequence = cache.get_or_build(context, lambda: _sequence(context))
    object_root = tmp_path / "objects" / cache.cache_key(context, sequence.input_hash)
    (object_root / "labels.json").write_text('{"life": 100}', encoding="utf-8")
    with pytest.raises(ValueError, match=r"unexpected|closed"):
        cache.load(context)
    (object_root / "labels.json").unlink()
    weights = object_root / "sequence.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match=r"size|SHA-256"):
        cache.load(context)


def test_cache_load_rejects_self_hashed_wrong_tensor_shape(tmp_path: Path) -> None:
    cache = EarlyCycleSequenceCache(tmp_path)
    context = _context()
    sequence = cache.get_or_build(context, lambda: _sequence(context))
    object_root = tmp_path / "objects" / cache.cache_key(context, sequence.input_hash)
    tensor_path = object_root / "sequence.safetensors"
    tensors = load_file(str(tensor_path), device="cpu")
    tensors["cycle_indices"] = tensors["cycle_indices"][:-1]
    save_file(tensors, str(tensor_path))
    _rehash_object_manifest(object_root, "sequence.safetensors")

    with pytest.raises(ValueError, match=r"shape|cycle|align"):
        cache.load(context)


def test_cache_rejects_non_raw_builder_output(tmp_path: Path) -> None:
    cache = EarlyCycleSequenceCache(tmp_path)
    context = _context()
    raw = _sequence(context)
    normalized = EarlyCycleSequence(
        **{
            **{
                key: value
                for key, value in raw.__dict__.items()
                if key != "input_hash"
            },
            "normalization_version": "train-zscore-v1",
            "normalization_statistics_sha256": "c" * 64,
        }
    )
    with pytest.raises(ValueError, match=r"raw|normalization"):
        cache.get_or_build(context, lambda: normalized)


def _rehash_object_manifest(object_root: Path, changed_name: str) -> None:
    import hashlib

    manifest_path = object_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_path = object_root / changed_name
    for file in manifest["files"]:
        if file["relative_path"] == changed_name:
            payload = changed_path.read_bytes()
            file["size_bytes"] = len(payload)
            file["sha256"] = hashlib.sha256(payload).hexdigest()
    canonical = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_canonical(canonical)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
