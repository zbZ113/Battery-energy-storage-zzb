from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from quanxin_life.core import SelectionMetricDirection
from quanxin_life.training.adapters.base import validate_upstream_artifact
from quanxin_life.training.adapters.registry import TrainingAdapterRegistry


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt"])
def test_unsafe_upstream_artifact_suffixes_are_rejected(tmp_path: Path, suffix: str) -> None:
    artifact = tmp_path / f"weights{suffix}"
    artifact.write_bytes(b"untrusted")

    with pytest.raises(ValueError, match=r"unsafe|approved"):
        validate_upstream_artifact(artifact, hashlib.sha256(b"untrusted").hexdigest())


@pytest.mark.parametrize("suffix", [".safetensors", ".json", ".parquet", ".ubj"])
def test_approved_upstream_artifacts_require_matching_sha256(
    tmp_path: Path, suffix: str
) -> None:
    payload = b"verified-bytes"
    artifact = tmp_path / f"artifact{suffix}"
    artifact.write_bytes(payload)

    verified = validate_upstream_artifact(artifact, hashlib.sha256(payload).hexdigest())
    assert verified.path == artifact.resolve()
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()

    with pytest.raises(ValueError, match="SHA-256"):
        validate_upstream_artifact(artifact, "0" * 64)


class _UnsafeAdapter:
    adapter_version = "unsafe-adapter-v1"
    selection_metric_name = "mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE

    def load_view(self, path: Path) -> object:
        return torch.load(path)


class _AliasedUnsafeAdapter:
    adapter_version = "unsafe-alias-adapter-v1"
    selection_metric_name = "mae"
    selection_metric_direction = SelectionMetricDirection.MINIMIZE

    def load_view(self, payload: bytes) -> object:
        import pickle as serializer

        return serializer.loads(payload)


def test_registry_blocks_adapter_source_that_calls_unsafe_deserializer() -> None:
    with pytest.raises(ValueError, match=r"torch\.load"):
        TrainingAdapterRegistry().register("unsafe", _UnsafeAdapter())

    with pytest.raises(ValueError, match=r"pickle\.loads"):
        TrainingAdapterRegistry().register("unsafe-alias", _AliasedUnsafeAdapter())
