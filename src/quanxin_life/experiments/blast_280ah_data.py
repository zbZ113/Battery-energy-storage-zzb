"""Build a verified 280 Ah capacity bundle for BLAST reference checking."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

from pydantic import Field

from quanxin_life.core import DatasetBuildStatus, canonical_json_bytes
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.lfp_280ah_dod import (
    DodCapacityCellSeries,
    DodCapacityDatasetLayout,
    load_dod_capacity_dataset_layout,
    load_dod_capacity_series,
)
from quanxin_life.data.manifest import (
    DatasetFileAuditManifest,
    load_dataset_file_audit_manifest,
    verify_audited_dataset_files,
)
from quanxin_life.data.source_catalog import SourceCatalog

BLAST_280AH_PROCESSOR_VERSION = "blast-280ah-dod-capacity-v1.0.0"
_SOURCE_CATALOG = "configs/data_sources.json"
_AUDIT_MANIFEST = "configs/data_manifests/lfp_280ah_dod_v1.json"
_CAPACITY_LAYOUT = "configs/data_layouts/lfp_280ah_dod_capacity_v1.json"
_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"


class Blast280AhBundleResult(ContractModel):
    status: DatasetBuildStatus
    output_dir: str = Field(min_length=1)
    bundle_sha256: Sha256
    cell_count: int = Field(gt=0)
    observation_count: int = Field(gt=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise ValueError(f"280Ah validation input must not be a symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"280Ah validation input is not a repository file: {relative_path}")
    return resolved


def _input_descriptors(audit: DatasetFileAuditManifest) -> list[dict[str, object]]:
    return [
        {
            "dataset_id": item.dataset_id,
            "relative_path": item.relative_path,
            "size_bytes": item.size_bytes,
            "sha256": item.sha256,
        }
        for item in audit.files
    ]


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_governance(
    root: Path,
) -> tuple[
    DatasetFileAuditManifest,
    DodCapacityDatasetLayout,
    list[dict[str, object]],
]:
    audit = load_dataset_file_audit_manifest(_resolve_file(root, _AUDIT_MANIFEST))
    if any(item.dataset_id != "LFP_280AH_DOD" for item in audit.files):
        raise ValueError("280Ah audit manifest contains another dataset")
    verify_audited_dataset_files(root, audit)
    catalog = SourceCatalog.load(_resolve_file(root, _SOURCE_CATALOG))
    source = catalog.require("LFP_280AH_DOD")
    if source.license_status != "CC BY 4.0":
        raise ValueError("280Ah validation requires verified CC BY 4.0 source metadata")
    if source.source_uri != "https://zenodo.org/records/14576042":
        raise ValueError("280Ah validation source URI is not the reviewed Zenodo record")
    catalog_hashes = dict(zip(source.artifact_paths, source.artifact_sha256, strict=True))
    for item in audit.files:
        if catalog_hashes.get(item.relative_path) != item.sha256:
            raise ValueError("280Ah audit manifest does not match the source catalog")
    layout_path = _resolve_file(root, _CAPACITY_LAYOUT)
    layout = load_dod_capacity_dataset_layout(layout_path)
    audited_paths = {item.relative_path for item in audit.files}
    if any(item.outer_relative_path not in audited_paths for item in layout.archives):
        raise ValueError("280Ah capacity layout references an unaudited outer archive")
    layout_descriptor = {
        "relative_path": _CAPACITY_LAYOUT,
        "size_bytes": layout_path.stat().st_size,
        "sha256": _sha256(layout_path),
    }
    return audit, layout, [layout_descriptor]


def _flatten(series: tuple[DodCapacityCellSeries, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in series:
        for point in cell.observations:
            rows.append(
                {
                    "dataset_id": cell.dataset_id,
                    "vendor": cell.vendor,
                    "cell_id": cell.cell_id,
                    "cycle_number": point.cycle_number,
                    "scheduled_efc": point.cycle_number
                    - cell.observations[0].cycle_number,
                    "elapsed_days": point.elapsed_days,
                    "discharge_capacity_ah": point.discharge_capacity_ah,
                    "nominal_soh": point.nominal_soh,
                    "relative_capacity_ratio": point.relative_capacity_ratio,
                    "mean_temperature_c": point.mean_temperature_c,
                    "sample_count": point.sample_count,
                    "dod_fraction": cell.dod_fraction,
                    "c_rate": cell.c_rate,
                    "reference_capacity_ah": cell.reference_capacity_ah,
                    "source_member": cell.source_member,
                    "nested_archive_sha256": cell.nested_archive_sha256,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("280Ah capacity bundle cannot write an empty CSV")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _existing_result(
    output: Path,
    *,
    input_files: list[dict[str, object]],
    layout_files: list[dict[str, object]],
) -> Blast280AhBundleResult | None:
    if not output.exists():
        return None
    manifest_path = output / "manifest.json"
    committed_path = output / "COMMITTED"
    series_path = output / "series.json"
    csv_path = output / "capacity_trajectories.csv"
    if not all(
        path.is_file()
        for path in (manifest_path, committed_path, series_path, csv_path)
    ):
        raise ValueError("existing 280Ah capacity bundle is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    bundle_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != bundle_sha:
        raise ValueError("existing 280Ah capacity COMMITTED hash is invalid")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing 280Ah capacity manifest is invalid") from exc
    if (
        manifest.get("processor_version") != BLAST_280AH_PROCESSOR_VERSION
        or manifest.get("input_files") != input_files
        or manifest.get("layout_files") != layout_files
    ):
        raise ValueError("existing 280Ah capacity bundle does not match current inputs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("existing 280Ah capacity artifact list is invalid")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise ValueError("existing 280Ah capacity artifact descriptor is invalid")
        path = output / item["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != item.get("size_bytes")
            or _sha256(path) != item.get("sha256")
        ):
            raise ValueError("existing 280Ah capacity artifact hash is invalid")
    return Blast280AhBundleResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_dir=str(output),
        bundle_sha256=bundle_sha,
        cell_count=int(manifest["cell_count"]),
        observation_count=int(manifest["observation_count"]),
    )


def prepare_blast_280ah_validation_bundle(
    repository_root: Path,
    *,
    output_dir: Path,
) -> Blast280AhBundleResult:
    """Publish cycle-level 100% DoD trajectories without modifying raw archives."""

    root = Path(repository_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output != root and root not in output.parents:
        raise ValueError("280Ah capacity output must remain inside the repository")
    audit, layout, layout_files = _validate_governance(root)
    input_files = _input_descriptors(audit)
    existing = _existing_result(
        output,
        input_files=input_files,
        layout_files=layout_files,
    )
    if existing is not None:
        return existing

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise ValueError("280Ah capacity temporary directory already exists")
    temporary.mkdir(parents=True)
    try:
        collected: list[DodCapacityCellSeries] = []
        for selection in layout.archives:
            outer = _resolve_file(root, selection.outer_relative_path)
            collected.extend(
                load_dod_capacity_series(
                    outer,
                    layout=layout,
                    vendor=selection.archive.vendor,
                    temporary_directory=(
                        temporary / f"nested-{selection.archive.vendor.lower()}"
                    ),
                )
            )
        series = tuple(sorted(collected, key=lambda item: item.cell_id))
        if not series:
            raise ValueError("280Ah capacity processing produced no cell series")
        rows = _flatten(series)
        temperature_missing_members = sorted(
            {
                member
                for item in series
                for member in item.temperature_missing_csv_members
            }
        )
        series_payload = {
            "schema_version": "blast-280ah-capacity-series-v1",
            "processor_version": BLAST_280AH_PROCESSOR_VERSION,
            "series": [item.model_dump(mode="json") for item in series],
        }
        series_path = temporary / "series.json"
        series_path.write_bytes(canonical_json_bytes(series_payload))
        csv_path = temporary / "capacity_trajectories.csv"
        _write_csv(csv_path, rows)
        manifest = {
            "schema_version": "blast-280ah-capacity-bundle-v1",
            "processor_version": BLAST_280AH_PROCESSOR_VERSION,
            "dataset_id": "LFP_280AH_DOD",
            "dataset_version": "Zenodo-14576042-v3",
            "source_uri": "https://zenodo.org/records/14576042",
            "license": "CC BY 4.0",
            "license_uri": _LICENSE_URI,
            "input_files": input_files,
            "layout_files": layout_files,
            "selection": {
                "dod_fraction": layout.selected_dod_fraction,
                "c_rate": layout.c_rate,
                "cell_ids": [item.cell_id for item in series],
                "normalization": layout.normalization,
                "temperature_missing_csv_members": temperature_missing_members,
            },
            "field_definitions": {
                "cycle_number": "Published source cycle number.",
                "scheduled_efc": "Cycle offset under the reviewed 100% DoD protocol.",
                "elapsed_days": (
                    "Elapsed source time from the first retained cycle; source timezone "
                    "is unresolved and no UTC claim is made."
                ),
                "discharge_capacity_ah": "Maximum source discharge capacity per cycle.",
                "nominal_soh": "Discharge capacity divided by 280 Ah nominal capacity.",
                "relative_capacity_ratio": (
                    "Discharge capacity divided by the first valid retained cycle capacity."
                ),
                "mean_temperature_c": "Mean of finite source temperature samples per cycle.",
            },
            "cell_count": len(series),
            "observation_count": len(rows),
            "warnings": [
                "EXTERNAL_TEST_ONLY",
                "FIRST_VALID_CYCLE_CAPACITY_NORMALIZATION",
                "SOURCE_TIMEZONE_UNRESOLVED",
                "NOT_A_15_TO_25_YEAR_VALIDATION_DATASET",
                *(
                    ["REVIEWED_CSV_MEMBERS_WITHOUT_TEMPERATURE_EXCLUDED"]
                    if temperature_missing_members
                    else []
                ),
            ],
            "artifacts": [
                _artifact(series_path, temporary),
                _artifact(csv_path, temporary),
            ],
        }
        manifest_bytes = canonical_json_bytes(manifest)
        bundle_sha = hashlib.sha256(manifest_bytes).hexdigest()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "COMMITTED").write_text(bundle_sha + "\n", encoding="ascii")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Blast280AhBundleResult(
        status=DatasetBuildStatus.BUILT,
        output_dir=str(output),
        bundle_sha256=bundle_sha,
        cell_count=len(series),
        observation_count=len(rows),
    )


__all__ = [
    "BLAST_280AH_PROCESSOR_VERSION",
    "Blast280AhBundleResult",
    "prepare_blast_280ah_validation_bundle",
]
