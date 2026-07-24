from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import torch

import quanxin_life.application.deep_model_artifacts as artifacts
from quanxin_life.application.deep_model_artifacts import (
    DeepArtifactFileRole,
    DeepArtifactKind,
    DeepModelArtifactManifest,
)
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.features.early_cycle_sequence import VARIABLE_NAMES
from quanxin_life.models.cyclepatch import EarlyCycleBatch
from quanxin_life.models.hybrid_degradation import (
    _HybridNetwork,
    normalise_prediction_cycle_positions,
)

_AGGREGATION_VERSION = "masked-variable-mean-v1"
_CANDIDATE_SHA256 = "a" * 64
_NORMALIZATION_SHA256 = "b" * 64
_PREDICTION_CYCLES = (50, 100, 200, 500)


def _early_batch() -> EarlyCycleBatch:
    cell_ids = ("cell-1", "cell-2")
    cycle_count = 21
    values = torch.linspace(
        -1.0,
        1.0,
        steps=len(cell_ids) * cycle_count * 2 * 150 * len(VARIABLE_NAMES),
        dtype=torch.float32,
    ).reshape(len(cell_ids), cycle_count, 2, 150, len(VARIABLE_NAMES))
    sample_mask = torch.ones((len(cell_ids), cycle_count, 2, 150), dtype=torch.bool)
    sample_mask[0, 0, 0, 0] = False
    values[0, 0, 0, 0] = torch.nan
    return EarlyCycleBatch(
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        feature_version="multichannel-v1",
        normalization_statistics_sha256=_NORMALIZATION_SHA256,
        cell_ids=cell_ids,
        condition_names=("temperature_c",),
        values=values,
        cycle_indices=torch.arange(cycle_count).expand(len(cell_ids), -1),
        cycle_mask=torch.ones((len(cell_ids), cycle_count), dtype=torch.bool),
        sample_mask=sample_mask,
        condition_values=torch.zeros((len(cell_ids), 1), dtype=torch.float32),
        condition_mask=torch.ones((len(cell_ids), 1), dtype=torch.bool),
    )


def _network() -> _HybridNetwork:
    return _HybridNetwork(
        input_dim=len(VARIABLE_NAMES),
        horizon=len(_PREDICTION_CYCLES),
        hidden_dim=32,
    ).eval()


def _export(tmp_path: Path) -> DeepModelArtifactManifest:
    return artifacts.export_current_hybrid_artifact(
        _network(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        prediction_cycles=_PREDICTION_CYCLES,
        variable_names=VARIABLE_NAMES,
        aggregation_version=_AGGREGATION_VERSION,
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
    )


def _load(tmp_path: Path, manifest: DeepModelArtifactManifest) -> torch.nn.Module:
    return artifacts.load_current_hybrid_artifact(
        tmp_path,
        manifest,
        expected_prediction_cycles=_PREDICTION_CYCLES,
        expected_variable_names=VARIABLE_NAMES,
        expected_aggregation_version=_AGGREGATION_VERSION,
        expected_normalization_sha256=_NORMALIZATION_SHA256,
        expected_candidate_config_sha256=_CANDIDATE_SHA256,
    )


def _rewrite_registered_json(
    root: Path,
    manifest: DeepModelArtifactManifest,
    update: dict[str, object],
) -> DeepModelArtifactManifest:
    registered = next(
        item
        for item in manifest.files
        if item.role is DeepArtifactFileRole.FEATURE_CONFIG
    )
    path = root / registered.relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    path.write_text(json.dumps(payload), encoding="utf-8")
    files = tuple(
        item.model_copy(
            update={
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if item.role is DeepArtifactFileRole.FEATURE_CONFIG
        else item
        for item in manifest.files
    )
    manifest_payload = {
        **manifest.model_dump(mode="json", exclude={"manifest_sha256", "files"}),
        "files": [item.model_dump(mode="json") for item in files],
    }
    changed = DeepModelArtifactManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    (root / manifest.artifact_id / "manifest.json").write_text(
        json.dumps(changed.model_dump(mode="json")),
        encoding="utf-8",
    )
    return changed


def _rewrite_kind(
    root: Path,
    manifest: DeepModelArtifactManifest,
    kind: DeepArtifactKind,
) -> DeepModelArtifactManifest:
    payload = {
        **manifest.model_dump(mode="json", exclude={"manifest_sha256"}),
        "artifact_kind": kind,
    }
    changed = DeepModelArtifactManifest.model_validate(
        {**payload, "manifest_sha256": sha256_canonical(payload)}
    )
    (root / manifest.artifact_id / "manifest.json").write_text(
        json.dumps(changed.model_dump(mode="json")),
        encoding="utf-8",
    )
    return changed


def test_current_hybrid_artifact_round_trip_preserves_advanced_prediction(
    tmp_path: Path,
) -> None:
    batch = _early_batch()
    initial_soh = torch.tensor((0.99, 0.98), dtype=torch.float32)
    network = _network()
    mask = batch.sample_mask.unsqueeze(-1)
    clean = torch.where(mask, batch.values, torch.zeros_like(batch.values))
    count = mask.to(batch.values.dtype).sum(dim=(1, 2, 3)).clamp_min(1.0)
    features = clean.sum(dim=(1, 2, 3)) / count
    cycle_positions = torch.tensor(
        normalise_prediction_cycle_positions(
            prediction_cycles=_PREDICTION_CYCLES,
            cutoff_cycle=20,
        ),
        dtype=torch.float32,
    )
    with torch.no_grad():
        expected = network(features, initial_soh, cycle_positions)

    manifest = artifacts.export_current_hybrid_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 24, tzinfo=UTC),
        dataset_id=batch.dataset_id,
        data_version=batch.data_version,
        split_version="matr-cell-split-v1",
        feature_version=batch.feature_version,
        cutoff_cycle=20,
        condition_names=batch.condition_names,
        prediction_cycles=_PREDICTION_CYCLES,
        variable_names=VARIABLE_NAMES,
        aggregation_version=_AGGREGATION_VERSION,
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
    )
    loaded = _load(tmp_path, manifest)

    with torch.no_grad():
        actual = loaded(batch, initial_soh)
    assert manifest.artifact_kind is DeepArtifactKind.CURRENT_HYBRID
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "wrong_hash", "message"),
    (
        ("expected_normalization_sha256", "c" * 64, "normalization_sha256"),
        ("expected_candidate_config_sha256", "d" * 64, "candidate_config_sha256"),
    ),
)
def test_current_hybrid_loader_rejects_wrong_expected_hash(
    tmp_path: Path,
    field: str,
    wrong_hash: str,
    message: str,
) -> None:
    manifest = _export(tmp_path)
    expected = {
        "expected_prediction_cycles": _PREDICTION_CYCLES,
        "expected_variable_names": VARIABLE_NAMES,
        "expected_aggregation_version": _AGGREGATION_VERSION,
        "expected_normalization_sha256": _NORMALIZATION_SHA256,
        "expected_candidate_config_sha256": _CANDIDATE_SHA256,
    }
    expected[field] = wrong_hash

    with pytest.raises(ValueError, match=message):
        artifacts.load_current_hybrid_artifact(tmp_path, manifest, **expected)


def test_current_hybrid_loader_rejects_tampered_weights(tmp_path: Path) -> None:
    manifest = _export(tmp_path)
    weights = tmp_path / manifest.artifact_id / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        _load(tmp_path, manifest)


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"prediction_cycles": (40, 100, 200, 500)}, "prediction_cycles"),
        ({"variable_names": tuple(reversed(VARIABLE_NAMES))}, "variable_names"),
        ({"aggregation_version": "unapproved-mean-v2"}, "aggregation_version"),
    ),
)
def test_current_hybrid_loader_rejects_changed_feature_axis(
    tmp_path: Path,
    update: dict[str, object],
    message: str,
) -> None:
    manifest = _export(tmp_path)
    changed = _rewrite_registered_json(tmp_path, manifest, update)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, changed)


def test_current_hybrid_loader_rejects_legacy_hybrid_kind(tmp_path: Path) -> None:
    manifest = _export(tmp_path)
    legacy = _rewrite_kind(tmp_path, manifest, DeepArtifactKind.HYBRID)

    with pytest.raises(ValueError, match="artifact kind"):
        _load(tmp_path, legacy)
