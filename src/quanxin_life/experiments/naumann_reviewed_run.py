"""Run a reviewed raw Naumann cycle matrix through the finite-pool GP pipeline."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from pydantic import Field, field_validator

from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.adapters.naumann_cycle_mat import (
    ReviewedAxisSelection,
    load_naumann_cycle_layout,
    load_naumann_cycle_matrix,
    select_reviewed_axis_observations,
)
from quanxin_life.data.manifest import (
    RawFileManifest,
    load_dataset_file_audit_manifest,
    verify_audited_dataset_files,
)
from quanxin_life.data.source_catalog import SourceCatalog
from quanxin_life.experiments.naumann_bridge import NaumannGpBridgeMapping
from quanxin_life.experiments.naumann_pipeline import (
    NaumannAcquisitionConfig,
    NaumannOperatingBounds,
    NaumannPipelineRequest,
    NaumannPipelineResult,
    NaumannSelectionAudit,
    run_naumann_gp_pipeline,
)


class ReviewedNaumannReplayConfig(ContractModel):
    """Only repository-relative, versioned inputs for one public-data replay."""

    config_version: str = Field(min_length=1)
    source_catalog_path: str = Field(min_length=1)
    audit_manifest_path: str = Field(min_length=1)
    source_file_path: str = Field(min_length=1)
    layout_path: str = Field(min_length=1)
    selection: ReviewedAxisSelection
    mapping: NaumannGpBridgeMapping
    operating_bounds: NaumannOperatingBounds
    acquisition_config: NaumannAcquisitionConfig
    initial_condition_ids: tuple[str, ...] = Field(min_length=2)
    query_budget: int = Field(gt=0)

    @field_validator(
        "source_catalog_path",
        "audit_manifest_path",
        "source_file_path",
        "layout_path",
    )
    @classmethod
    def paths_are_portable_and_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("reviewed replay paths must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("reviewed replay paths must remain below repository root")
        if not path.parts or ":" in path.parts[0]:
            raise ValueError("reviewed replay paths must not contain a drive prefix")
        return value


def load_reviewed_naumann_replay_config(path: Path) -> ReviewedNaumannReplayConfig:
    """Load a strict JSON configuration; executable formats are unsupported."""

    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("reviewed Naumann replay config must use a .json file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"reviewed Naumann replay config does not exist: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reviewed Naumann replay config must contain valid UTF-8 JSON") from exc
    return ReviewedNaumannReplayConfig.model_validate(payload)


def run_reviewed_naumann_cycle_replay(
    repository_root: Path,
    config: ReviewedNaumannReplayConfig,
    *,
    output_dir: Path,
) -> NaumannPipelineResult:
    """Verify raw public data, select direct rows, then run the governed replay."""

    root = Path(repository_root).resolve(strict=True)
    config = ReviewedNaumannReplayConfig.model_validate(config.model_dump(mode="json"))
    source_catalog_path = _resolve_repository_file(root, config.source_catalog_path)
    audit_manifest_path = _resolve_repository_file(root, config.audit_manifest_path)
    source_file_path = _resolve_repository_file(root, config.source_file_path)
    layout_path = _resolve_repository_file(root, config.layout_path)

    audit_manifest = load_dataset_file_audit_manifest(audit_manifest_path)
    if audit_manifest.source_catalog != config.source_catalog_path:
        raise ValueError("audit manifest source_catalog does not match replay config")
    verify_audited_dataset_files(root, audit_manifest)
    audited_file = next(
        (
            item
            for item in audit_manifest.files
            if item.relative_path == config.source_file_path
        ),
        None,
    )
    if audited_file is None:
        raise ValueError("reviewed replay source file is not declared in audit manifest")
    if audited_file.dataset_id != "NAUMANN_CYCLE":
        raise ValueError("reviewed cycle replay requires a NAUMANN_CYCLE source file")

    source = SourceCatalog.load(source_catalog_path).require("NAUMANN_CYCLE")
    raw_manifest = RawFileManifest(
        dataset_id=audited_file.dataset_id,
        relative_path=audited_file.relative_path,
        sha256=audited_file.sha256,
        source_uri=source.source_uri,
        license_name=source.license_status,
        paper_doi=source.paper_uri.removeprefix("https://doi.org/"),
    )
    layout = load_naumann_cycle_layout(layout_path)
    observations = load_naumann_cycle_matrix(
        source_file_path,
        raw_manifest,
        source,
        layout=layout,
    )
    selected = select_reviewed_axis_observations(
        observations,
        selection=config.selection,
    )
    selection_audit = NaumannSelectionAudit(
        selection_version=selected.selection_version,
        observation_axis=selected.observations[0].observation_axis,
        target_axis_value=selected.target_axis_value,
        max_absolute_deviation=config.selection.max_absolute_deviation,
        condition_ids=tuple(item.condition_id for item in selected.observations),
        selected_axis_values=tuple(
            item.observation_value for item in selected.observations
        ),
        absolute_deviations=selected.absolute_deviations,
    )
    request = NaumannPipelineRequest(
        pipeline_config_version=config.config_version,
        observations=selected.observations,
        mapping=config.mapping,
        selection_audit=selection_audit,
        operating_bounds=config.operating_bounds,
        acquisition_config=config.acquisition_config,
        initial_condition_ids=config.initial_condition_ids,
        query_budget=config.query_budget,
    )
    return run_naumann_gp_pipeline(request, output_dir=output_dir)


def _resolve_repository_file(root: Path, relative_path: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink():
        raise ValueError(f"reviewed replay input must not be a symbolic link: {relative_path}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"reviewed replay input is not a repository file: {relative_path}")
    return resolved
