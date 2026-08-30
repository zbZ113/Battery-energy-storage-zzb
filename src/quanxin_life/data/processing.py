"""Idempotent staging, verification, and atomic canonical bundle publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core import CanonicalTableType, DatasetBuildStatus
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.canonical import CanonicalNumericValue
from quanxin_life.data.dataset_bundle import (
    ArtifactFile,
    ArtifactManifest,
    CanonicalDatasetManifest,
    CanonicalQualityReport,
    SourceArtifact,
)


class RawDatasetFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_versioned_raw_input(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.startswith("data/raw/")
            or "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("raw dataset files must use versioned data/raw paths")
        return value


class DatasetBuildSpec(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    dataset_version: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    adapter_version: str = Field(min_length=1)
    raw_files: tuple[RawDatasetFile, ...] = Field(min_length=1)
    values: tuple[CanonicalNumericValue, ...] = ()


class DatasetBuildResult(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: DatasetBuildStatus
    output_root: str = Field(min_length=1)
    output_sha256: Sha256


class DatasetProcessor:
    """Publish source-bound canonical bundles without mutating raw inputs."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = Path(repository_root).resolve(strict=True)

    def output_path(self, spec: DatasetBuildSpec) -> Path:
        return (
            self.repository_root
            / "data"
            / "processed"
            / spec.dataset_id
            / spec.dataset_version
            / "canonical-v1"
        )

    def build(self, spec: DatasetBuildSpec) -> DatasetBuildResult:
        validated = DatasetBuildSpec.model_validate(spec.model_dump(mode="json"))
        actual_sources = self._verify_sources(validated)
        final = self.output_path(validated)
        if final.exists():
            manifest = self.verify(final)
            expected = tuple(
                SourceArtifact(relative_path=item.relative_path, sha256=actual)
                for item, actual in zip(validated.raw_files, actual_sources, strict=True)
            )
            if manifest.source_artifacts != expected:
                raise ValueError("changed raw SHA requires a new dataset version")
            return DatasetBuildResult(
                status=DatasetBuildStatus.SKIPPED_VALID,
                output_root=final.relative_to(self.repository_root).as_posix(),
                output_sha256=_sha256_file(final / "artifact_manifest.json"),
            )

        parent = final.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = parent / f".canonical-v1.staging-{uuid4().hex}"
        staging.mkdir()
        for directory in ("metadata", "observations", "targets"):
            (staging / directory).mkdir()

        source_artifacts = tuple(
            SourceArtifact(relative_path=item.relative_path, sha256=actual)
            for item, actual in zip(validated.raw_files, actual_sources, strict=True)
        )
        source_index = {
            "schema_version": "canonical-source-index-v1",
            "dataset_id": validated.dataset_id,
            "dataset_version": validated.dataset_version,
            "adapter_version": validated.adapter_version,
            "source_artifacts": [item.model_dump(mode="json") for item in source_artifacts],
        }
        _write_json(staging / "metadata" / "source_index.json", source_index)
        _write_canonical_values(staging, validated.values)
        dataset_manifest = CanonicalDatasetManifest(
            dataset_id=validated.dataset_id,
            dataset_version=validated.dataset_version,
            adapter_version=validated.adapter_version,
            source_artifacts=source_artifacts,
        )
        _write_json(
            staging / "dataset_manifest.json", dataset_manifest.model_dump(mode="json")
        )
        report = CanonicalQualityReport(
            dataset_id=validated.dataset_id,
            dataset_version=validated.dataset_version,
            record_count=len(validated.values),
            warnings=(
                ("SOURCE_INDEX_ONLY_PENDING_DATASET_ADAPTER",)
                if not validated.values
                else ()
            ),
        )
        _write_json(staging / "quality_report.json", report.model_dump(mode="json"))
        artifact_manifest = _build_artifact_manifest(
            staging,
            dataset_id=validated.dataset_id,
            dataset_version=validated.dataset_version,
        )
        _write_json(
            staging / "artifact_manifest.json",
            artifact_manifest.model_dump(mode="json"),
        )
        output_sha256 = _sha256_file(staging / "artifact_manifest.json")
        (staging / "COMMITTED").write_text(output_sha256 + "\n", encoding="ascii")
        self.verify(staging)
        os.replace(staging, final)
        return DatasetBuildResult(
            status=DatasetBuildStatus.BUILT,
            output_root=final.relative_to(self.repository_root).as_posix(),
            output_sha256=output_sha256,
        )

    def verify(self, bundle_root: Path) -> CanonicalDatasetManifest:
        root = Path(bundle_root).resolve(strict=True)
        if not root.is_relative_to(self.repository_root) or not root.is_dir():
            raise ValueError("canonical bundle must remain inside the repository")
        committed = root / "COMMITTED"
        artifact_path = root / "artifact_manifest.json"
        if not committed.is_file() or not artifact_path.is_file():
            raise ValueError("canonical bundle is not committed")
        artifact_manifest = ArtifactManifest.model_validate_json(artifact_path.read_bytes())
        if committed.read_text(encoding="ascii").strip() != _sha256_file(artifact_path):
            raise ValueError("canonical bundle commit marker SHA mismatch")
        expected = {item.relative_path for item in artifact_manifest.files}
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "COMMITTED"}
        }
        if actual != expected:
            raise ValueError("canonical bundle contains an unregistered file")
        for item in artifact_manifest.files:
            path = root.joinpath(*PurePosixPath(item.relative_path).parts)
            if path.stat().st_size != item.size_bytes or _sha256_file(path) != item.sha256:
                raise ValueError(f"canonical bundle artifact mismatch: {item.relative_path}")
        manifest = CanonicalDatasetManifest.model_validate_json(
            (root / "dataset_manifest.json").read_bytes()
        )
        if (
            manifest.dataset_id != artifact_manifest.dataset_id
            or manifest.dataset_version != artifact_manifest.dataset_version
        ):
            raise ValueError("canonical bundle manifests disagree")
        return manifest

    def _verify_sources(self, spec: DatasetBuildSpec) -> tuple[str, ...]:
        actual: list[str] = []
        for item in spec.raw_files:
            path = self.repository_root.joinpath(*PurePosixPath(item.relative_path).parts)
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self.repository_root) or not resolved.is_file():
                raise ValueError("raw dataset input escapes the repository")
            digest = _sha256_file(resolved)
            if digest != item.sha256 and not self.output_path(spec).exists():
                raise ValueError(f"raw dataset SHA mismatch: {item.relative_path}")
            actual.append(digest)
        return tuple(actual)


def _build_artifact_manifest(
    root: Path,
    *,
    dataset_id: str,
    dataset_version: str,
) -> ArtifactManifest:
    files = tuple(
        ArtifactFile(
            relative_path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256_file(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    return ArtifactManifest(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        files=files,
    )


def _write_canonical_values(
    root: Path,
    values: tuple[CanonicalNumericValue, ...],
) -> None:
    grouped: dict[CanonicalTableType, list[CanonicalNumericValue]] = {}
    for value in values:
        grouped.setdefault(value.table_type, []).append(value)
    observation_tables = {
        CanonicalTableType.CELL_CYCLE_TELEMETRY,
        CanonicalTableType.TRAJECTORY_OBSERVATIONS,
        CanonicalTableType.CONDITION_OBSERVATIONS,
    }
    for table_type, records in grouped.items():
        if table_type in observation_tables:
            directory = "observations"
        elif table_type is CanonicalTableType.METADATA:
            directory = "metadata"
        elif table_type is CanonicalTableType.TARGETS:
            directory = "targets"
        else:
            raise ValueError(f"canonical numeric values cannot populate {table_type.value}")
        _write_json(
            root / directory / f"{table_type.value}.json",
            {
                "schema_version": "canonical-values-v1",
                "table_type": table_type.value,
                "records": [record.model_dump(mode="json") for record in records],
            },
        )
def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DatasetBuildResult",
    "DatasetBuildSpec",
    "DatasetProcessor",
    "RawDatasetFile",
]
