"""Traceable tabular sources accompanying evaluated PNG figures."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from pydantic import ConfigDict, Field

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256

Scalar: TypeAlias = str | int | float | bool | None


class FigureKind(StrEnum):
    LOSS_VALIDATION_CURVE = "loss_validation_curve"
    BEST_EPOCH = "best_epoch"
    OBSERVED_PREDICTED = "observed_predicted"
    ERROR_DISTRIBUTION = "error_distribution"
    PER_CELL_WATERFALL = "per_cell_waterfall"
    REPRESENTATIVE_SOH_TRAJECTORY = "representative_soh_trajectory"
    HORIZON_ERROR = "horizon_error"
    SEEN_UNSEEN = "seen_unseen"
    COVERAGE_WIDTH = "coverage_width"
    ACCURACY_RESOURCE_PARETO = "accuracy_resource_pareto"


class FigureSourceArtifact(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: FigureKind
    png_path: Path
    source_path: Path
    manifest_path: Path
    row_count: int = Field(gt=0)
    png_sha256: Sha256
    source_sha256: Sha256
    manifest_sha256: Sha256


def materialize_figure_source(
    png_path: Path,
    *,
    kind: FigureKind,
    rows: Sequence[Mapping[str, Scalar]],
) -> FigureSourceArtifact:
    png = Path(png_path)
    if png.suffix.lower() != ".png" or not png.is_file():
        raise ValueError("an existing PNG figure is required")
    material = tuple(dict(row) for row in rows)
    if not material:
        raise ValueError("figure source requires at least one row")
    columns = tuple(material[0])
    if not columns or any(not column.strip() for column in columns):
        raise ValueError("figure source columns must be non-empty")
    if any(tuple(row) != columns for row in material):
        raise ValueError("all figure source rows must use identical ordered columns")
    for row in material:
        for value in row.values():
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("figure source values must be scalar")
            if isinstance(value, float) and (
                value != value or value in {float("inf"), -float("inf")}
            ):
                raise ValueError("figure source values must be finite")
    source = png.with_suffix(".csv")
    temporary = source.with_name(f".{source.name}.tmp")
    source.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(material)
        temporary.replace(source)
    finally:
        temporary.unlink(missing_ok=True)
    png_sha256 = _sha256_file(png)
    source_sha256 = _sha256_file(source)
    payload: dict[str, Scalar | list[str]] = {
        "schema_version": "figure-source-manifest-v1",
        "kind": kind.value,
        "png_file": png.name,
        "source_file": source.name,
        "row_count": len(material),
        "columns": list(columns),
        "png_sha256": png_sha256,
        "source_sha256": source_sha256,
    }
    manifest_sha256 = sha256_canonical(payload)
    manifest = png.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps({**payload, "manifest_sha256": manifest_sha256}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return FigureSourceArtifact(
        kind=kind,
        png_path=png,
        source_path=source,
        manifest_path=manifest,
        row_count=len(material),
        png_sha256=png_sha256,
        source_sha256=source_sha256,
        manifest_sha256=manifest_sha256,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["FigureKind", "FigureSourceArtifact", "materialize_figure_source"]
