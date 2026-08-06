"""Train-only normalization and atomic publication for frozen model views."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import overload
from uuid import uuid4

from quanxin_life.core import (
    DatasetBuildStatus,
    TrainingReadableSplit,
    sha256_canonical,
)
from quanxin_life.core.schemas import Sha256
from quanxin_life.data.model_views.schemas import (
    ModelViewBuildResult,
    ModelViewConfig,
    ModelViewManifest,
    ModelViewRow,
    NormalizerStatistics,
)

_BUILDER_VERSION = "model-view-builder-v1"


def validate_split_isolation(rows: Sequence[ModelViewRow]) -> None:
    assignments: dict[str, TrainingReadableSplit] = {}
    for row in rows:
        previous = assignments.setdefault(row.entity_id, row.split)
        if previous is not row.split:
            raise ValueError(f"entity {row.entity_id} crosses splits")


def fit_normalizer(
    rows: Sequence[ModelViewRow],
    *,
    feature_names: tuple[str, ...],
) -> NormalizerStatistics:
    validate_split_isolation(rows)
    train_rows = [row for row in rows if row.split is TrainingReadableSplit.TRAIN]
    if not train_rows:
        raise ValueError("normalizer requires train rows")
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in feature_names:
        values = [
            value
            for row in train_rows
            if row.feature_mask.get(name) and (value := row.features.get(name)) is not None
        ]
        if not values:
            raise ValueError(f"training feature has no observed values: {name}")
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        means[name] = mean
        scales[name] = math.sqrt(variance) or 1.0
    training_ids = tuple(sorted({row.entity_id for row in train_rows}))
    return NormalizerStatistics(
        means=means,
        scales=scales,
        training_entity_ids_sha256=sha256_canonical(training_ids),
    )


def build_model_view(
    *,
    rows: Sequence[ModelViewRow],
    config: ModelViewConfig,
    output_root: Path,
    canonical_sha256: Sha256,
    split_sha256: Sha256,
    builder_code_sha256: Sha256,
    config_sha256: Sha256,
) -> ModelViewBuildResult:
    validated_rows = tuple(
        ModelViewRow.model_validate(row.model_dump(mode="json")) for row in rows
    )
    expected_features = set(config.feature_names)
    for row in validated_rows:
        if set(row.features) != expected_features:
            raise ValueError("row feature names must match the registered view config")
    expected_split_sha256 = sha256_canonical(
        {
            entity_id: split.value
            for entity_id, split in sorted(
                {row.entity_id: row.split for row in validated_rows}.items()
            )
        }
    )
    if split_sha256 != expected_split_sha256:
        raise ValueError("split SHA does not match entity assignments")
    normalizer = fit_normalizer(validated_rows, feature_names=config.feature_names)
    view_payload = {"rows": [row.model_dump(mode="json") for row in validated_rows]}
    normalization_payload = normalizer.model_dump(mode="json")
    manifest = ModelViewManifest(
        view_id=config.view_id,
        view_version=config.view_version,
        task_type=config.task_type,
        target_semantics=config.target_semantics,
        mask_semantics=config.mask_semantics,
        cutoff_cycle=config.cutoff_cycle,
        canonical_sha256=canonical_sha256,
        split_sha256=split_sha256,
        builder_version=_BUILDER_VERSION,
        builder_code_sha256=builder_code_sha256,
        config_sha256=config_sha256,
        normalization_sha256=sha256_canonical(normalization_payload),
        training_entity_ids_sha256=normalizer.training_entity_ids_sha256,
        row_count=len(validated_rows),
    )
    root = Path(output_root)
    if root.exists():
        existing = _verify_model_view(root, allow_legacy_commit=True)
        if existing != manifest or _sha256_file(root / "view.json") != _sha256_bytes(
            _json_bytes(view_payload)
        ):
            raise ValueError("changed model view context requires a new model view version")
        marker = root / "COMMITTED"
        if marker.read_text(encoding="ascii").strip() != _commit_digest(root):
            replacement = root / ".COMMITTED.tmp"
            replacement.write_text(_commit_digest(root) + "\n", encoding="ascii")
            os.replace(replacement, marker)
        return ModelViewBuildResult(
            status=DatasetBuildStatus.SKIPPED_VALID,
            output_sha256=_commit_digest(root),
            manifest=existing,
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.parent / f".{root.name}.staging-{uuid4().hex}"
    staging.mkdir()
    _write_json(staging / "view.json", view_payload)
    _write_json(staging / "normalization.json", normalization_payload)
    _write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
    digest = _commit_digest(staging)
    (staging / "COMMITTED").write_text(digest + "\n", encoding="ascii")
    verify_model_view(staging)
    os.replace(staging, root)
    return ModelViewBuildResult(
        status=DatasetBuildStatus.BUILT,
        output_sha256=digest,
        manifest=manifest,
    )


def verify_model_view(root: Path) -> ModelViewManifest:
    return _verify_model_view(root, allow_legacy_commit=False)


def _verify_model_view(
    root: Path,
    *,
    allow_legacy_commit: bool,
) -> ModelViewManifest:
    directory = Path(root).resolve(strict=True)
    manifest_path = directory / "manifest.json"
    manifest = ModelViewManifest.model_validate_json(manifest_path.read_bytes())
    expected = (
        {"view.json", "normalization.json", "manifest.json", "COMMITTED"}
        if manifest.schema_version == "model-view-manifest-v1"
        else {
            "manifest.json",
            "COMMITTED",
            *(artifact.relative_path for artifact in manifest.artifacts),
        }
    )
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise ValueError("model view inventory is not closed")
    if manifest.schema_version == "model-view-manifest-v2":
        for artifact in manifest.artifacts:
            path = directory / artifact.relative_path
            if path.stat().st_size != artifact.size_bytes or _sha256_file(path) != artifact.sha256:
                raise ValueError(
                    f"model view artifact SHA-256 mismatch: {artifact.relative_path}"
                )
        normalization_path = directory / "normalization.json"
        if _sha256_file(normalization_path) != manifest.normalization_sha256:
            raise ValueError("model view normalization SHA mismatch")
        if (directory / "COMMITTED").read_text(
            encoding="ascii"
        ).strip() != _commit_digest(directory):
            raise ValueError("model view commit marker mismatch")
        return manifest
    marker = (directory / "COMMITTED").read_text(encoding="ascii").strip()
    valid_markers = {_commit_digest(directory)}
    if allow_legacy_commit:
        valid_markers.add(_sha256_file(manifest_path))
    if marker not in valid_markers:
        raise ValueError("model view commit marker mismatch")
    normalizer = NormalizerStatistics.model_validate_json(
        (directory / "normalization.json").read_bytes()
    )
    if sha256_canonical(normalizer.model_dump(mode="json")) != manifest.normalization_sha256:
        raise ValueError("model view normalization SHA mismatch")
    if normalizer.training_entity_ids_sha256 != manifest.training_entity_ids_sha256:
        raise ValueError("model view training entity SHA mismatch")
    payload = json.loads((directory / "view.json").read_text(encoding="utf-8"))
    rows_payload = payload.get("rows")
    if not isinstance(rows_payload, list):
        raise ValueError("model view rows must be a list")
    rows = tuple(ModelViewRow.model_validate(row) for row in rows_payload)
    if len(rows) != manifest.row_count:
        raise ValueError("model view row count mismatch")
    feature_names = tuple(normalizer.means)
    if any(set(row.features) != set(feature_names) for row in rows):
        raise ValueError("model view feature names mismatch")
    if fit_normalizer(rows, feature_names=feature_names) != normalizer:
        raise ValueError("model view normalizer does not match train rows")
    computed_split_sha256 = sha256_canonical(
        {
            entity_id: split.value
            for entity_id, split in sorted(
                {row.entity_id: row.split for row in rows}.items()
            )
        }
    )
    if computed_split_sha256 != manifest.split_sha256:
        raise ValueError("model view split SHA mismatch")
    return manifest


class FrozenModelViewDataset(Sequence[ModelViewRow]):
    """Read-only training dataset backed solely by a frozen view JSON file."""

    def __init__(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError("frozen model view rows must be a list")
        self._rows = tuple(ModelViewRow.model_validate(row) for row in rows)

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> ModelViewRow: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ModelViewRow, ...]: ...

    def __getitem__(self, index: int | slice) -> ModelViewRow | tuple[ModelViewRow, ...]:
        return self._rows[index]

    def __iter__(self) -> Iterator[ModelViewRow]:
        return iter(self._rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_json_bytes(payload))


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _commit_digest(root: Path) -> str:
    return sha256_canonical(
        {
            path.name: _sha256_file(path)
            for path in sorted(root.iterdir())
            if path.is_file() and path.name != "COMMITTED"
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "FrozenModelViewDataset",
    "build_model_view",
    "fit_normalizer",
    "validate_split_isolation",
    "verify_model_view",
]
