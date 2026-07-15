from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.application.model_artifacts import (
    ArtifactFormat,
    ArtifactKind,
    ModelArtifactManifest,
    ModelArtifactRegistry,
    load_verified_xgboost_life_predictor,
)
from quanxin_life.core import LifePrediction
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.models.xgboost import XGBoostLifePredictor


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    path: Path,
    *,
    sha256: str,
    size_bytes: int,
    artifact_id: str | None = None,
    artifact_kind: ArtifactKind = ArtifactKind.VARIANCE,
    artifact_format: ArtifactFormat = ArtifactFormat.VARIANCE_JSON,
) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        artifact_id=artifact_id or str(uuid4()),
        artifact_kind=artifact_kind,
        artifact_format=artifact_format,
        relative_path=path.as_posix(),
        sha256=sha256,
        size_bytes=size_bytes,
        model_version="variance-v1",
        data_version="matr-v1",
        feature_version="delta-q-v1",
        split_version="cell-split-v1",
        schema_version="model-artifact-manifest-v1",
        dataset_id="matr",
        cutoff_cycle=20,
        feature_names=("capacity_delta_ah",),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def test_registers_and_reads_verified_variance_json(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "artifact_type": "quanxin_life.variance_life_predictor.v1",
            "slope": -1.25,
            "intercept": 3.5,
        },
        sort_keys=True,
    ).encode()
    relative_path = Path("models/variance.json")
    digest = _write(tmp_path / relative_path, payload)
    manifest = _manifest(relative_path, sha256=digest, size_bytes=len(payload))
    registry = ModelArtifactRegistry(tmp_path)

    verified = registry.register(manifest)
    resolved = registry.resolve(
        manifest.artifact_id,
        model_version="variance-v1",
        data_version="matr-v1",
        feature_version="delta-q-v1",
        split_version="cell-split-v1",
    )

    assert verified == resolved
    assert verified.absolute_path == (tmp_path / relative_path).resolve()
    assert verified.manifest.sha256 == digest
    assert registry.read_json_object(manifest.artifact_id)["slope"] == -1.25


@pytest.mark.parametrize(
    ("filename", "artifact_format", "payload"),
    [
        ("xgboost.json", ArtifactFormat.XGBOOST_JSON, b'{"learner": {}}'),
        ("xgboost.ubj", ArtifactFormat.XGBOOST_UBJ, b"{Lopaque-ubj-payload"),
    ],
)
def test_accepts_declared_xgboost_native_formats(
    tmp_path: Path,
    filename: str,
    artifact_format: ArtifactFormat,
    payload: bytes,
) -> None:
    relative_path = Path("models") / filename
    digest = _write(tmp_path / relative_path, payload)
    manifest = _manifest(
        relative_path,
        sha256=digest,
        size_bytes=len(payload),
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=artifact_format,
    )

    verified = ModelArtifactRegistry(tmp_path).register(manifest)

    assert verified.manifest.artifact_format is artifact_format


def test_rejects_sha256_mismatch_and_detects_later_tampering(tmp_path: Path) -> None:
    relative_path = Path("models/variance.json")
    artifact_path = tmp_path / relative_path
    digest = _write(artifact_path, b'{"slope": 1.0}')
    manifest = _manifest(relative_path, sha256="0" * 64, size_bytes=artifact_path.stat().st_size)
    registry = ModelArtifactRegistry(tmp_path)

    with pytest.raises(ValueError, match="SHA-256"):
        registry.register(manifest)

    verified_manifest = manifest.model_copy(update={"sha256": digest})
    registry.register(verified_manifest)
    artifact_path.write_bytes(b'{"slope": 2.0}')

    with pytest.raises(ValueError, match="SHA-256"):
        registry.resolve(verified_manifest.artifact_id)


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".joblib", ".pt", ".pth"])
def test_rejects_executable_deserialization_artifact_suffixes(
    tmp_path: Path, suffix: str
) -> None:
    relative_path = Path("models") / f"unsafe{suffix}"
    digest = _write(tmp_path / relative_path, b"untrusted")
    with pytest.raises(ValueError, match=r"forbidden|format extension"):
        _manifest(relative_path, sha256=digest, size_bytes=len(b"untrusted"))


def test_rejects_path_escape_and_invalid_artifact_id(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"outside-{uuid4()}.json"
    digest = _write(outside, b"{}")
    with pytest.raises(ValueError, match=r"relative_path|root"):
        _manifest(Path("..") / outside.name, sha256=digest, size_bytes=outside.stat().st_size)

    relative_path = Path("models/variance.json")
    payload = b"{}"
    valid_digest = _write(tmp_path / relative_path, payload)
    manifest = _manifest(relative_path, sha256=valid_digest, size_bytes=len(payload))

    with pytest.raises(ValidationError, match="artifact_id"):
        ModelArtifactManifest.model_validate(
            {**manifest.model_dump(mode="json"), "artifact_id": "not-a-uuid"}
        )


def test_rejects_context_mismatch_duplicate_id_and_non_object_json(tmp_path: Path) -> None:
    relative_path = Path("models/variance.json")
    digest = _write(tmp_path / relative_path, b"[]")
    manifest = _manifest(relative_path, sha256=digest, size_bytes=len(b"[]"))
    registry = ModelArtifactRegistry(tmp_path)
    registry.register(manifest)

    with pytest.raises(ValueError, match="model_version"):
        registry.resolve(manifest.artifact_id, model_version="other-model")
    with pytest.raises(ValueError, match="already registered"):
        registry.register(manifest)
    with pytest.raises(ValueError, match="JSON object"):
        registry.read_json_object(manifest.artifact_id)


def test_refuses_binary_artifact_through_json_reader(tmp_path: Path) -> None:
    relative_path = Path("models/xgboost.ubj")
    digest = _write(tmp_path / relative_path, b"opaque-ubj")
    manifest = _manifest(
        relative_path,
        sha256=digest,
        size_bytes=len(b"opaque-ubj"),
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_UBJ,
    )
    registry = ModelArtifactRegistry(tmp_path)
    registry.register(manifest)

    with pytest.raises(ValueError, match="JSON"):
        registry.read_json_object(manifest.artifact_id)


def test_loads_xgboost_predictor_only_from_verified_native_artifact(tmp_path: Path) -> None:
    predictor = XGBoostLifePredictor(
        model_version="xgboost-v1",
        data_version="matr-v1",
        feature_version="delta-q-v1",
        split_version="cell-split-v1",
        cutoff_cycle=20,
        feature_names=("capacity_delta_ah",),
    )
    split = SplitManifest(
        dataset_id="matr",
        train=("train-a", "train-b"),
        validation=("val",),
        calibration=("cal",),
        test=("test",),
    )
    labels = tuple(
        LifePrediction(
            dataset_id="matr",
            cell_id=cell_id,
            cutoff_cycle=20,
            predicted_eol_cycle=eol,
            observed_eol_cycle=int(eol),
            right_censored=False,
            feature_version="delta-q-v1",
            split_version="cell-split-v1",
            model_version="label-v1",
            data_version="matr-v1",
        )
        for cell_id, eol in (("train-a", 100.0), ("train-b", 200.0))
    )
    predictor.fit(
        labels,
        training_features={
            "train-a": {"capacity_delta_ah": -0.1},
            "train-b": {"capacity_delta_ah": -0.2},
        },
        split_manifest=split,
    )
    relative_path = Path("models/xgboost.json")
    predictor.export_native_model(tmp_path / relative_path)
    payload = (tmp_path / relative_path).read_bytes()
    manifest = ModelArtifactManifest(
        artifact_id=str(uuid4()),
        artifact_kind=ArtifactKind.XGBOOST,
        artifact_format=ArtifactFormat.XGBOOST_JSON,
        relative_path=relative_path.as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        model_version="xgboost-v1",
        data_version="matr-v1",
        feature_version="delta-q-v1",
        split_version="cell-split-v1",
        schema_version="model-artifact-manifest-v1",
        dataset_id="matr",
        cutoff_cycle=20,
        feature_names=("capacity_delta_ah",),
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )
    registry = ModelArtifactRegistry(tmp_path)
    registry.register(manifest)

    loaded = load_verified_xgboost_life_predictor(registry, manifest.artifact_id)
    prediction = loaded.predict(
        dataset_id="matr",
        cell_id="test",
        features={"capacity_delta_ah": -0.15},
        cutoff_cycle=20,
        feature_version="delta-q-v1",
        split_version="cell-split-v1",
        data_version="matr-v1",
    )

    assert prediction.model_version == "xgboost-v1"
    assert loaded.model_artifact_id == manifest.artifact_id
    assert loaded.model_artifact_sha256 == manifest.sha256


@pytest.mark.parametrize(
    "payload",
    [
        b'{"slope": NaN}',
        b'{"slope": 1.0, "slope": 2.0}',
    ],
)
def test_rejects_ambiguous_or_nonstandard_json(tmp_path: Path, payload: bytes) -> None:
    relative_path = Path("models/variance.json")
    digest = _write(tmp_path / relative_path, payload)
    manifest = _manifest(relative_path, sha256=digest, size_bytes=len(payload))
    registry = ModelArtifactRegistry(tmp_path)
    registry.register(manifest)

    with pytest.raises(ValueError, match=r"JSON|duplicate"):
        registry.read_json_object(manifest.artifact_id)
