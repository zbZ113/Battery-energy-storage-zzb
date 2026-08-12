"""Build an immutable Naumann bundle for BLAST-Lite external validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field

from quanxin_life.core import DatasetBuildStatus, canonical_json_bytes
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.naumann_calendar import (
    load_naumann_calendar_capacity,
    load_naumann_calendar_layout,
)
from quanxin_life.data.adapters.naumann_cycle_mat import (
    load_naumann_cycle_layout,
    load_naumann_cycle_matrix,
    load_naumann_cycle_read_policy,
)
from quanxin_life.data.manifest import (
    AuditedDatasetFile,
    RawFileManifest,
    load_dataset_file_audit_manifest,
    verify_audited_dataset_files,
)
from quanxin_life.data.source_catalog import SourceCatalog, SourceCatalogEntry

BLAST_VALIDATION_PROCESSOR_VERSION = "blast-validation-naumann-v1.0.0"
_SOURCE_CATALOG = "configs/data_sources.json"
_AUDIT_MANIFEST = "configs/data_manifests/naumann_public_files_v2.json"
_CALENDAR_SOURCE = "data/raw/NAUMANN_CALENDAR/v1/DischargeCapacity.xlsx"
_CALENDAR_LAYOUT = "configs/data_layouts/naumann_calendar_capacity_v1.json"
_CYCLE_INPUTS = (
    (
        "data/raw/NAUMANN_CYCLE/v1/xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat",
        "configs/data_layouts/naumann_cycle_xdod_capacity_fec_v1.json",
        None,
    ),
    (
        "data/raw/NAUMANN_CYCLE/v1/xCyC_80DOD_40°C_Capacity_CC_CV_FEC.mat",
        "configs/data_layouts/naumann_cycle_xcyc_capacity_fec_v1.json",
        "configs/data_layouts/naumann_cycle_xcyc_read_policy_v1.json",
    ),
    (
        "data/raw/NAUMANN_CYCLE/v1/xDOD_1C1C_x°C_Capacity_CC_CV_FEC.mat",
        "configs/data_layouts/naumann_cycle_temperature_capacity_fec_v1.json",
        "configs/data_layouts/naumann_cycle_temperature_read_policy_v1.json",
    ),
    (
        "data/raw/NAUMANN_CYCLE/v1/xSOC_20DOD_1C1C_40°C_Capacity_CC_CV_FEC.mat",
        "configs/data_layouts/naumann_cycle_xsoc_capacity_fec_v1.json",
        None,
    ),
)
_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/"


class BlastValidationBundleResult(ContractModel):
    status: DatasetBuildStatus
    output_dir: str = Field(min_length=1)
    bundle_sha256: Sha256
    calendar_observation_count: int = Field(gt=0)
    cycle_observation_count: int = Field(gt=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise ValueError(f"BLAST validation input must not be a symlink: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"BLAST validation input is not a repository file: {relative_path}")
    return resolved


def _audited_file(
    files: tuple[AuditedDatasetFile, ...],
    relative_path: str,
) -> AuditedDatasetFile:
    match = next((item for item in files if item.relative_path == relative_path), None)
    if match is None:
        raise ValueError(f"BLAST validation source is not audited: {relative_path}")
    return match


def _raw_manifest(
    item: AuditedDatasetFile,
    source: SourceCatalogEntry,
) -> RawFileManifest:
    downloaded_at = source.downloaded_at
    if downloaded_at is None:
        raise ValueError("BLAST validation source has no audited download timestamp")
    return RawFileManifest(
        dataset_id=item.dataset_id,
        relative_path=item.relative_path,
        sha256=item.sha256,
        source_uri=source.source_uri,
        license_name=source.license_status,
        license_uri=_LICENSE_URI,
        paper_doi=source.paper_uri.removeprefix("https://doi.org/"),
        downloaded_at=downloaded_at,
    )


def _input_descriptor(item: AuditedDatasetFile) -> dict[str, object]:
    return {
        "dataset_id": item.dataset_id,
        "relative_path": item.relative_path,
        "size_bytes": item.size_bytes,
        "sha256": item.sha256,
    }


def _layout_descriptor(root: Path, relative_path: str) -> dict[str, object]:
    path = _resolve_file(root, relative_path)
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_observations(
    root: Path,
) -> tuple[
    dict[str, Any],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, str]],
]:
    audit = load_dataset_file_audit_manifest(_resolve_file(root, _AUDIT_MANIFEST))
    verify_audited_dataset_files(root, audit)
    catalog = SourceCatalog.load(_resolve_file(root, _SOURCE_CATALOG))
    calendar_source = catalog.require("NAUMANN_CALENDAR")
    cycle_source = catalog.require("NAUMANN_CYCLE")

    calendar_item = _audited_file(audit.files, _CALENDAR_SOURCE)
    calendar = load_naumann_calendar_capacity(
        _resolve_file(root, _CALENDAR_SOURCE),
        _raw_manifest(calendar_item, calendar_source),
        calendar_source,
        layout=load_naumann_calendar_layout(_resolve_file(root, _CALENDAR_LAYOUT)),
    )
    input_files = [_input_descriptor(calendar_item)]
    layout_files = [_layout_descriptor(root, _CALENDAR_LAYOUT)]
    cycle_rows: list[dict[str, Any]] = []
    excluded_conditions: list[dict[str, str]] = []
    for source_path, layout_path, policy_path in _CYCLE_INPUTS:
        item = _audited_file(audit.files, source_path)
        layout = load_naumann_cycle_layout(_resolve_file(root, layout_path))
        read_policy = (
            load_naumann_cycle_read_policy(_resolve_file(root, policy_path))
            if policy_path is not None
            else None
        )
        observations = load_naumann_cycle_matrix(
            _resolve_file(root, source_path),
            _raw_manifest(item, cycle_source),
            cycle_source,
            layout=layout,
            read_policy=read_policy,
        )
        cycle_rows.extend(item.model_dump(mode="json") for item in observations)
        input_files.append(_input_descriptor(item))
        layout_files.append(_layout_descriptor(root, layout_path))
        if policy_path is not None:
            layout_files.append(_layout_descriptor(root, policy_path))
        if read_policy is not None:
            for column_policy in read_policy.columns:
                if column_policy.exclusion_reason is None:
                    continue
                condition = next(
                    value
                    for value in layout.condition_columns
                    if value.column_index == column_policy.column_index
                )
                excluded_conditions.append(
                    {
                        "condition_id": condition.condition_id,
                        "reason": column_policy.exclusion_reason,
                        "source_file": source_path,
                    }
                )

    payload = {
        "schema_version": "blast-validation-observations-v1",
        "processor_version": BLAST_VALIDATION_PROCESSOR_VERSION,
        "calendar": [item.model_dump(mode="json") for item in calendar],
        "cycle": cycle_rows,
    }
    return payload, input_files, layout_files, excluded_conditions


def _existing_result(
    output_dir: Path,
    *,
    input_files: list[dict[str, object]],
    layout_files: list[dict[str, object]],
) -> BlastValidationBundleResult | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    observations_path = output_dir / "observations.json"
    committed_path = output_dir / "COMMITTED"
    if not all(path.is_file() for path in (manifest_path, observations_path, committed_path)):
        raise ValueError("existing BLAST validation bundle is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != manifest_sha:
        raise ValueError("existing BLAST validation COMMITTED hash is invalid")
    payload = json.loads(manifest_bytes)
    if (
        payload.get("processor_version") != BLAST_VALIDATION_PROCESSOR_VERSION
        or payload.get("input_files") != input_files
        or payload.get("layout_files") != layout_files
        or payload.get("observations_sha256") != _sha256(observations_path)
    ):
        raise ValueError("existing BLAST validation bundle does not match current inputs")
    counts = payload.get("observation_counts")
    if not isinstance(counts, dict):
        raise ValueError("existing BLAST validation bundle has invalid counts")
    return BlastValidationBundleResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_dir=str(output_dir),
        bundle_sha256=manifest_sha,
        calendar_observation_count=int(counts["calendar"]),
        cycle_observation_count=int(counts["cycle"]),
    )


def prepare_blast_validation_bundle(
    repository_root: Path,
    *,
    output_dir: Path,
) -> BlastValidationBundleResult:
    """Verify five Naumann files and atomically publish reusable validation rows."""

    root = Path(repository_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    observations, input_files, layout_files, excluded_conditions = _load_observations(root)
    existing = _existing_result(
        output,
        input_files=input_files,
        layout_files=layout_files,
    )
    if existing is not None:
        return existing

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise ValueError("BLAST validation temporary directory already exists")
    temporary.mkdir(parents=True)
    observations_bytes = canonical_json_bytes(observations)
    observations_path = temporary / "observations.json"
    observations_path.write_bytes(observations_bytes)
    counts = {
        "calendar": len(observations["calendar"]),
        "cycle": len(observations["cycle"]),
    }
    manifest = {
        "schema_version": "blast-validation-bundle-v1",
        "processor_version": BLAST_VALIDATION_PROCESSOR_VERSION,
        "license": "CC BY 4.0",
        "license_uri": _LICENSE_URI,
        "source_catalog": _SOURCE_CATALOG,
        "audit_manifest": _AUDIT_MANIFEST,
        "input_files": input_files,
        "layout_files": layout_files,
        "excluded_conditions": excluded_conditions,
        "observation_counts": counts,
        "observations_sha256": hashlib.sha256(observations_bytes).hexdigest(),
    }
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (temporary / "manifest.json").write_bytes(manifest_bytes)
    (temporary / "COMMITTED").write_text(manifest_sha + "\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return BlastValidationBundleResult(
        status=DatasetBuildStatus.BUILT,
        output_dir=str(output),
        bundle_sha256=manifest_sha,
        calendar_observation_count=counts["calendar"],
        cycle_observation_count=counts["cycle"],
    )


__all__ = [
    "BLAST_VALIDATION_PROCESSOR_VERSION",
    "BlastValidationBundleResult",
    "prepare_blast_validation_bundle",
]
