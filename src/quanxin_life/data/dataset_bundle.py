"""Closed-world manifests for immutable canonical dataset bundles."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256

_ROOT_FILES = frozenset({"dataset_manifest.json", "quality_report.json"})
_PAYLOAD_ROOTS = frozenset({"metadata", "observations", "targets"})
_PAYLOAD_SUFFIXES = frozenset({".json", ".parquet"})


class SourceArtifact(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: Sha256


class ArtifactFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_allowlisted(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("artifact path must remain inside the bundle")
        if value in _ROOT_FILES:
            return value
        if (
            len(path.parts) < 2
            or path.parts[0] not in _PAYLOAD_ROOTS
            or path.suffix.lower() not in _PAYLOAD_SUFFIXES
        ):
            raise ValueError("artifact path is outside the closed bundle schema")
        return value


class ArtifactManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["artifact-manifest-v1"] = "artifact-manifest-v1"
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    files: tuple[ArtifactFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def inventory_is_unique_and_sorted(self) -> ArtifactManifest:
        paths = [item.relative_path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("artifact files must be unique and sorted")
        return self


class CanonicalDatasetManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["canonical-dataset-v1"] = "canonical-dataset-v1"
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    source_artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)


class CanonicalQualityReport(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["canonical-quality-report-v1"] = (
        "canonical-quality-report-v1"
    )
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    record_count: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


__all__ = [
    "ArtifactFile",
    "ArtifactManifest",
    "CanonicalDatasetManifest",
    "CanonicalQualityReport",
    "SourceArtifact",
]
