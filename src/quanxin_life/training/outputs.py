"""Strict contract and byte-level verification for an A100 training run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

_REQUIRED_ROOT_FILES = frozenset(
    {
        "config_resolved.json",
        "environment.json",
        "metrics.csv",
        "metrics.json",
        "model_card.md",
        "run_manifest.json",
        "training_log.jsonl",
    }
)
_ALLOWED_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".md", ".png", ".safetensors", ".svg", ".ubj"}
)
_FORBIDDEN_SUFFIXES = frozenset(
    {".joblib", ".key", ".pem", ".pickle", ".pkl", ".pt", ".pth"}
)
_SECRET_PATTERN = re.compile(
    rb"(?i)['\"]?(?:api[_-]?key|app[_-]?secret)['\"]?\s*[:=]\s*"
    rb"['\"]?[A-Za-z0-9_\-]{8,}"
)


class A100TrainingRunManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["a100-training-run-v1"] = "a100-training-run-v1"
    run_id: str
    training_bundle_sha256: Sha256
    preflight_sha256: Sha256
    git_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    git_dirty: Literal[False]
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoffs: tuple[int, ...] = Field(min_length=1)
    models: tuple[str, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = Field(min_length=1)
    started_at: datetime
    finished_at: datetime

    @field_validator("run_id")
    @classmethod
    def run_id_is_uuid(cls, value: str) -> str:
        try:
            return str(UUID(value))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("run_id must be a UUID string") from exc

    @field_validator("started_at", "finished_at")
    @classmethod
    def time_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("training run timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def run_is_consistent(self) -> A100TrainingRunManifest:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if any(cutoff < 0 for cutoff in self.cutoffs) or any(
            current <= previous
            for previous, current in zip(self.cutoffs, self.cutoffs[1:], strict=False)
        ):
            raise ValueError("cutoffs must be unique, non-negative and increasing")
        if len(set(self.models)) != len(self.models) or any(
            not name.strip() for name in self.models
        ):
            raise ValueError("models must be unique and nonblank")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        return self


class TrainingOutputFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = Path(normalized)
        if path.is_absolute() or path.drive or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("relative_path must remain inside the output root")
        return path.as_posix()


class TrainingOutputIndex(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["a100-training-output-index-v1"] = (
        "a100-training-output-index-v1"
    )
    created_at: datetime
    run_manifest: A100TrainingRunManifest
    files: tuple[TrainingOutputFile, ...] = Field(min_length=1)
    output_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


def build_training_output_index(
    output_root: Path,
    *,
    created_at: datetime,
) -> TrainingOutputIndex:
    root = _validated_root(output_root)
    files = _collect_output_files(root)
    paths = {item.relative_path for item in files}
    missing = _REQUIRED_ROOT_FILES - paths
    if missing:
        raise ValueError(f"required output evidence is missing: {sorted(missing)}")
    if not any(
        path.startswith("plots/") and Path(path).suffix.lower() in {".png", ".svg"}
        for path in paths
    ):
        raise ValueError("required output plot is missing")
    if not any(
        path.startswith("artifacts/")
        and Path(path).suffix.lower() in {".json", ".safetensors", ".ubj"}
        for path in paths
    ):
        raise ValueError("required safe model artifact is missing")

    run_manifest = A100TrainingRunManifest.model_validate(
        _read_strict_json(root / "run_manifest.json")
    )
    output_hash = _output_hash(run_manifest=run_manifest, files=files)
    return TrainingOutputIndex(
        created_at=created_at,
        run_manifest=run_manifest,
        files=files,
        output_sha256=output_hash,
    )


def verify_training_output_index(output_root: Path, index: TrainingOutputIndex) -> None:
    actual = build_training_output_index(output_root, created_at=index.created_at)
    expected_files = {item.relative_path: item for item in index.files}
    actual_files = {item.relative_path: item for item in actual.files}
    if set(expected_files) != set(actual_files):
        raise ValueError("training output file inventory does not match the index")
    for relative_path, expected in expected_files.items():
        observed = actual_files[relative_path]
        if observed.size_bytes != expected.size_bytes:
            raise ValueError("training output file size does not match the index")
        if observed.sha256 != expected.sha256:
            raise ValueError("training output file SHA-256 does not match the index")
    if actual.run_manifest != index.run_manifest:
        raise ValueError("training run manifest does not match the index")
    if actual.output_sha256 != index.output_sha256:
        raise ValueError("training output SHA-256 does not match the index")


def load_training_output_index(path: Path) -> TrainingOutputIndex:
    """Load a strict external index before re-verifying the referenced directory."""

    return TrainingOutputIndex.model_validate(_read_strict_json(path))


def _collect_output_files(root: Path) -> tuple[TrainingOutputFile, ...]:
    files: list[TrainingOutputFile] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            if directory.is_symlink():
                raise ValueError("symbolic link is forbidden in training output")
            _validate_output_path(directory.relative_to(root), is_file=False)
        for name in file_names:
            path = current / name
            relative = path.relative_to(root)
            _validate_output_path(relative, is_file=True)
            if path.is_symlink():
                raise ValueError("symbolic link is forbidden in training output")
            if not path.is_file():
                raise ValueError("non-regular training output file is forbidden")
            _scan_text_for_secrets(path)
            files.append(
                TrainingOutputFile(
                    relative_path=relative.as_posix(),
                    size_bytes=path.stat().st_size,
                    sha256=_sha256_file(path),
                )
            )
    files.sort(key=lambda item: item.relative_path)
    return tuple(files)


def _validate_output_path(relative: Path, *, is_file: bool) -> None:
    lowered = tuple(part.lower() for part in relative.parts)
    if any(part.startswith(".env") for part in lowered) or any(
        marker in part for part in lowered for marker in ("credential", "secret", "token")
    ):
        raise ValueError("forbidden secret-bearing path in training output")
    if not is_file:
        return
    suffix = relative.suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES or suffix not in _ALLOWED_SUFFIXES:
        raise ValueError("forbidden training output file format")


def _scan_text_for_secrets(path: Path) -> None:
    if path.suffix.lower() in {".png", ".safetensors", ".ubj"}:
        return
    if path.stat().st_size > 16 * 1024 * 1024:
        return
    payload = path.read_bytes()
    if b"-----BEGIN PRIVATE KEY-----" in payload or _SECRET_PATTERN.search(payload):
        raise ValueError("forbidden secret material detected in training output")


def _output_hash(
    *,
    run_manifest: A100TrainingRunManifest,
    files: tuple[TrainingOutputFile, ...],
) -> str:
    return sha256_canonical(
        {
            "schema_version": "a100-training-output-index-v1",
            "run_manifest": run_manifest.model_dump(mode="json"),
            "files": [item.model_dump(mode="json") for item in files],
        }
    )


def _validated_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("training output root must exist") from exc
    if root.is_symlink() or not resolved.is_dir():
        raise ValueError("training output root must be a regular directory")
    return resolved


def _read_strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key is forbidden: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is forbidden: {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("training output JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("training output JSON must contain an object")
    for value in payload.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("training output JSON cannot contain non-finite values")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
