from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from quanxin_life.application.deep_model_artifacts import (
    DeepArtifactFileRole,
    DeepModelArtifactManifest,
    export_cpmlp_artifact,
    export_hybrid_artifact,
    load_cpmlp_artifact,
    load_hybrid_artifact,
)
from quanxin_life.core import LifePrediction
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.curve_tensor import CurveTensor
from quanxin_life.models.cpmlp import CPMLPLifePredictor
from quanxin_life.models.hybrid_degradation import (
    HybridDegradationPredictor,
    TrajectoryTrainingSample,
)


def _cpmlp_curve(cell_id: str, scale: float) -> CurveTensor:
    values = tuple(
        (0.6 * scale, 0.3 * scale, 0.0) if cycle == 1 else (None, None, None)
        for cycle in range(3)
    )
    return CurveTensor(
        dataset_id="SAFE-HUST",
        cell_id=cell_id,
        cutoff_cycle=2,
        feature_version="curve-v1",
        cycle_indices=(0, 1, 2),
        voltage_grid_v=(3.0, 3.1, 3.2),
        values=values,
        observed_mask=(False, True, False),
    )


def _fitted_cpmlp() -> CPMLPLifePredictor:
    predictor = CPMLPLifePredictor(
        model_version="cpmlp-v1",
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
        cutoff_cycle=2,
        curve_hidden_dim=4,
        aggregation_hidden_dim=4,
        epochs=1,
    )
    label = LifePrediction(
        dataset_id="SAFE-HUST",
        cell_id="train-cell",
        cutoff_cycle=2,
        predicted_eol_cycle=100,
        observed_eol_cycle=100,
        right_censored=False,
        feature_version="curve-v1",
        split_version="split-v1",
        model_version="label-v1",
        data_version="safe-hust-v1",
    )
    split = SplitManifest(
        dataset_id="SAFE-HUST",
        train=("train-cell",),
        validation=("validation-cell",),
        calibration=("calibration-cell",),
        test=("test-cell",),
    )
    return predictor.fit(
        [label],
        training_curves={"train-cell": _cpmlp_curve("train-cell", 0.9)},
        split_manifest=split,
    )


def _fitted_hybrid() -> HybridDegradationPredictor:
    predictor = HybridDegradationPredictor(
        model_version="hybrid-v1",
        feature_version="early-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
        cutoff_cycle=2,
        prediction_cycles=(3, 4, 5),
        condition_feature_names=("temperature_c",),
        hidden_dim=4,
        epochs=1,
    )
    split = SplitManifest(
        dataset_id="SAFE-HUST",
        train=("train-cell",),
        validation=("validation-cell",),
        calibration=("calibration-cell",),
        test=("test-cell",),
    )
    sample = TrajectoryTrainingSample(
        dataset_id="SAFE-HUST",
        cell_id="train-cell",
        cutoff_cycle=2,
        observed_cycles=(0, 1, 2),
        observed_soh=(1.0, 0.99, 0.98),
        target_cycles=(3, 4, 5),
        target_soh=(0.97, 0.96, 0.95),
        condition_features={"temperature_c": 25.0},
        feature_version="early-v1",
        data_version="safe-hust-v1",
    )
    return predictor.fit([sample], split_manifest=split)


def test_cpmlp_safetensors_round_trip_preserves_prediction(tmp_path: Path) -> None:
    predictor = _fitted_cpmlp()
    curve = _cpmlp_curve("test-cell", 1.0)
    expected = predictor.predict(
        curve=curve,
        cutoff_cycle=2,
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
    )

    manifest = export_cpmlp_artifact(
        predictor,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    loaded = load_cpmlp_artifact(tmp_path, manifest)
    actual = loaded.predict(
        curve=curve,
        cutoff_cycle=2,
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
    )

    assert actual.predicted_eol_cycle == pytest.approx(expected.predicted_eol_cycle, abs=1e-7)


def test_hybrid_safetensors_round_trip_preserves_trajectory(tmp_path: Path) -> None:
    predictor = _fitted_hybrid()
    prediction_inputs = {
        "dataset_id": "SAFE-HUST",
        "cell_id": "test-cell",
        "observed_cycles": (0, 1, 2),
        "observed_soh": (1.0, 0.99, 0.98),
        "condition_features": {"temperature_c": 25.0},
        "cutoff_cycle": 2,
        "feature_version": "early-v1",
        "split_version": "split-v1",
        "data_version": "safe-hust-v1",
    }
    expected = predictor.predict(**prediction_inputs)

    manifest = export_hybrid_artifact(
        predictor,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    loaded = load_hybrid_artifact(tmp_path, manifest)
    actual = loaded.predict(**prediction_inputs)

    assert actual.predicted_soh == pytest.approx(expected.predicted_soh, abs=1e-7)


def test_deep_artifact_loader_rejects_changed_weights(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    weights = tmp_path / manifest.artifact_id / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        load_cpmlp_artifact(tmp_path, manifest)


def test_deep_artifact_loader_rejects_unknown_architecture(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    architecture = tmp_path / manifest.artifact_id / "architecture.json"
    payload = json.loads(architecture.read_text(encoding="utf-8"))
    payload["architecture"] = "unknown-network"
    architecture.write_text(json.dumps(payload), encoding="utf-8")

    files = tuple(
        item.model_copy(
            update={
                "size_bytes": architecture.stat().st_size,
                "sha256": hashlib.sha256(architecture.read_bytes()).hexdigest(),
            }
        )
        if item.role is DeepArtifactFileRole.ARCHITECTURE
        else item
        for item in manifest.files
    )
    manifest_payload = {
        **manifest.model_dump(mode="json", exclude={"manifest_sha256", "files"}),
        "files": [item.model_dump(mode="json") for item in files],
    }
    modified_manifest = DeepModelArtifactManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    (tmp_path / manifest.artifact_id / "manifest.json").write_text(
        json.dumps(modified_manifest.model_dump(mode="json")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="architecture"):
        load_cpmlp_artifact(tmp_path, modified_manifest)


def test_deep_artifact_loader_rejects_unlisted_files(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    (tmp_path / manifest.artifact_id / "unlisted.pkl").write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="unexpected file"):
        load_cpmlp_artifact(tmp_path, manifest)
