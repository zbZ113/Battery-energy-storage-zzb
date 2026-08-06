from __future__ import annotations

import pytest
from pydantic import ValidationError

from quanxin_life.data.dataset_bundle import ArtifactFile, ArtifactManifest


def test_artifact_manifest_is_closed_and_sorted() -> None:
    manifest = ArtifactManifest(
        dataset_id="MATR",
        dataset_version="v1",
        files=(
            ArtifactFile(
                relative_path="metadata/source_index.json",
                size_bytes=10,
                sha256="a" * 64,
            ),
        ),
    )

    assert manifest.files[0].relative_path == "metadata/source_index.json"


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.json", "observations/payload.pkl", "unknown/file.bin"],
)
def test_artifact_manifest_rejects_unknown_or_unsafe_files(relative_path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactFile(
            relative_path=relative_path,
            size_bytes=10,
            sha256="a" * 64,
        )
