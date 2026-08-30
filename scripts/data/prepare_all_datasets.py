"""Plan or publish source-bound canonical dataset bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.data.processing import (  # noqa: E402
    DatasetBuildSpec,
    DatasetProcessor,
    RawDatasetFile,
)
from quanxin_life.data.source_catalog import SourceCatalog, SourceCatalogEntry  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build"))
    parser.add_argument("--dataset")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    catalog = SourceCatalog.load(root / "configs" / "data_sources.json")
    processor = DatasetProcessor(root)

    if args.mode == "plan":
        for entry in catalog.entries:
            print(json.dumps(_plan_entry(entry, processor), sort_keys=True))
        return 0

    if not args.dataset:
        parser.error("build requires --dataset")
    entry = catalog.require(args.dataset)
    blocker = _blocker(entry)
    if blocker is not None:
        print(json.dumps({"dataset_id": entry.dataset_id, "status": blocker}, sort_keys=True))
        return 42
    result = processor.build(_build_spec(entry, root=root))
    print(
        json.dumps(
            {
                "dataset_id": entry.dataset_id,
                "output_root": result.output_root,
                "output_sha256": result.output_sha256,
                "status": result.status.value,
            },
            sort_keys=True,
        )
    )
    return 0


def _plan_entry(
    entry: SourceCatalogEntry,
    processor: DatasetProcessor,
) -> dict[str, object]:
    blocker = _blocker(entry)
    if blocker is not None:
        return {"dataset_id": entry.dataset_id, "status": blocker}
    output = (
        processor.repository_root
        / "data"
        / "processed"
        / entry.dataset_id
        / entry.version
        / "canonical-v1"
    )
    status = "READY"
    if output.exists():
        processor.verify(output)
        status = "SKIPPED_VALID"
    return {"dataset_id": entry.dataset_id, "status": status}


def _blocker(entry: SourceCatalogEntry) -> str | None:
    if entry.license_status.upper() == "UNVERIFIED":
        return "BLOCKED_LICENSE"
    if entry.dataset_id not in {"MATR", "NAUMANN_CYCLE", "NAUMANN_CALENDAR"}:
        return "BLOCKED_DATA_ADAPTER"
    return None


def _build_spec(entry: SourceCatalogEntry, *, root: Path) -> DatasetBuildSpec:
    if entry.dataset_id in {"NAUMANN_CYCLE", "NAUMANN_CALENDAR"}:
        return _build_naumann_spec(entry, root=root)
    return DatasetBuildSpec(
        dataset_id=entry.dataset_id,
        dataset_version=entry.version,
        adapter_version="source-index-canonical-v1",
        raw_files=tuple(
            RawDatasetFile(relative_path=path, sha256=sha256)
            for path, sha256 in zip(
                entry.artifact_paths, entry.artifact_sha256, strict=True
            )
        ),
    )


def _build_naumann_spec(entry: SourceCatalogEntry, *, root: Path) -> DatasetBuildSpec:
    from quanxin_life.data.adapters.naumann_bundle import (
        build_naumann_condition_bundle,
        to_naumann_build_spec,
    )
    from quanxin_life.data.adapters.naumann_calendar import (
        load_naumann_calendar_capacity,
        load_naumann_calendar_layout,
    )
    from quanxin_life.data.adapters.naumann_cycle_mat import (
        load_naumann_cycle_layout,
        load_naumann_cycle_matrix,
    )
    from quanxin_life.data.manifest import RawFileManifest

    if entry.downloaded_at is None:
        raise ValueError("Naumann catalog entry lacks download provenance")
    artifact_hashes = dict(zip(entry.artifact_paths, entry.artifact_sha256, strict=True))
    if entry.dataset_id == "NAUMANN_CYCLE":
        relative = (
            "data/raw/NAUMANN_CYCLE/v1/"
            "xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat"
        )
        layout = load_naumann_cycle_layout(
            root / "configs/data_layouts/naumann_cycle_xdod_capacity_fec_v1.json"
        )
        manifest = RawFileManifest(
            dataset_id=entry.dataset_id,
            relative_path=relative,
            sha256=artifact_hashes[relative],
            source_uri=entry.source_uri,
            license_name=entry.license_status,
            paper_doi="10.1016/j.jpowsour.2019.227666",
            downloaded_at=entry.downloaded_at,
        )
        observations = load_naumann_cycle_matrix(
            root / relative, manifest, entry, layout=layout
        )
    else:
        relative = "data/raw/NAUMANN_CALENDAR/v1/DischargeCapacity.xlsx"
        layout = load_naumann_calendar_layout(
            root / "configs/data_layouts/naumann_calendar_capacity_v1.json"
        )
        manifest = RawFileManifest(
            dataset_id=entry.dataset_id,
            relative_path=relative,
            sha256=artifact_hashes[relative],
            source_uri=entry.source_uri,
            license_name=entry.license_status,
            paper_doi="10.1016/j.est.2018.01.019",
            downloaded_at=entry.downloaded_at,
        )
        observations = load_naumann_calendar_capacity(
            root / relative, manifest, entry, layout=layout
        )
    bundle = build_naumann_condition_bundle(
        dataset_version=entry.version,
        observations=observations,
    )
    return to_naumann_build_spec(
        bundle,
        raw_files=(
            RawDatasetFile(relative_path=relative, sha256=artifact_hashes[relative]),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
