"""Create or verify the three reviewed MATR batches and their joint split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.core import sha256_canonical  # noqa: E402
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file  # noqa: E402
from quanxin_life.data.matr_multibatch import (  # noqa: E402
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    audit_matr_supervision_eligibility,
    combine_matr_batch_splits,
)
from quanxin_life.data.matr_pipeline import (  # noqa: E402
    MatrBatchConversionReport,
    MatrSupervisionArtifact,
    build_matr_split_evidence,
    build_matr_supervision_artifact,
    convert_matr_batch,
)
from quanxin_life.data.schemas import SplitManifest  # noqa: E402
from quanxin_life.data.storage import (  # noqa: E402
    ProcessedCellManifest,
    verify_cell_artifacts,
)


@dataclass(frozen=True)
class BatchSpec:
    batch_index: int
    batch_date: date
    raw_path: str
    raw_manifest: str
    processed_root: str
    conversion_report: str
    split_manifest: str
    split_evidence: str
    supervision_root: str
    supervision_report: str
    eligibility_report: str


SPECS = (
    BatchSpec(
        batch_index=1,
        batch_date=date(2017, 5, 12),
        raw_path="data/2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        raw_manifest="configs/data_manifests/matr_2017_05_12_batch_v1.json",
        processed_root="data/processed/MATR/2017-05-12-cutoff150",
        conversion_report="reports/data_quality/matr_2017_05_12_cutoff150.json",
        split_manifest="configs/data_splits/matr_2017_05_12_cell_split_v1.json",
        split_evidence="reports/data_quality/matr_2017_05_12_split_v1.json",
        supervision_root="data/processed/MATR/2017-05-12-supervision500",
        supervision_report="reports/data_quality/matr_2017_05_12_supervision500_v1.json",
        eligibility_report=(
            "reports/data_quality/matr_2017_05_12_trajectory500_eligibility_v1.json"
        ),
    ),
    BatchSpec(
        batch_index=2,
        batch_date=date(2017, 6, 30),
        raw_path="data/2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        raw_manifest="configs/data_manifests/matr_2017_06_30_batch_v1.json",
        processed_root="data/processed/MATR/2017-06-30-cutoff150",
        conversion_report="reports/data_quality/matr_2017_06_30_cutoff150.json",
        split_manifest="configs/data_splits/matr_2017_06_30_cell_split_v1.json",
        split_evidence="reports/data_quality/matr_2017_06_30_split_v1.json",
        supervision_root="data/processed/MATR/2017-06-30-supervision500-eligible",
        supervision_report="reports/data_quality/matr_2017_06_30_supervision500_v1.json",
        eligibility_report=(
            "reports/data_quality/matr_2017_06_30_trajectory500_eligibility_v1.json"
        ),
    ),
    BatchSpec(
        batch_index=3,
        batch_date=date(2018, 4, 12),
        raw_path="data/2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        raw_manifest="configs/data_manifests/matr_2018_04_12_batch_v1.json",
        processed_root="data/processed/MATR/2018-04-12-cutoff150",
        conversion_report="reports/data_quality/matr_2018_04_12_cutoff150.json",
        split_manifest="configs/data_splits/matr_2018_04_12_cell_split_v1.json",
        split_evidence="reports/data_quality/matr_2018_04_12_split_v1.json",
        supervision_root="data/processed/MATR/2018-04-12-supervision500",
        supervision_report="reports/data_quality/matr_2018_04_12_supervision500_v1.json",
        eligibility_report=(
            "reports/data_quality/matr_2018_04_12_trajectory500_eligibility_v1.json"
        ),
    ),
)

COMBINED_SPLIT = "configs/data_splits/matr_three_batch_cell_split_v1.json"
COMBINED_MANIFEST = "reports/data_quality/matr_three_batch_manifest_v1.json"
SPLIT_VERSION = "matr-three-batch-cell-split-v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "final"))
    parser.parse_args()
    components: list[MatrBatchArtifactReference] = []
    splits: list[SplitManifest] = []
    for spec in SPECS:
        component, split = _prepare_batch(spec)
        components.append(component)
        splits.append(split)

    combined = combine_matr_batch_splits(tuple(splits))
    combined_path = _path(COMBINED_SPLIT, must_exist=False)
    if combined_path.exists():
        existing_split = SplitManifest.model_validate_json(combined_path.read_bytes())
        if existing_split != combined:
            raise ValueError("existing combined MATR split differs from batch assignments")
    else:
        _write_json_atomic(combined_path, combined.model_dump(mode="json"))
    combined_sha256 = _sha256_file(combined_path)
    data_fingerprint = sha256_canonical(
        [component.raw_sha256 for component in components]
    )
    proposed = MatrThreeBatchManifest(
        data_version=f"matr-three-batch-{data_fingerprint[:16]}",
        split_version=SPLIT_VERSION,
        combined_split_manifest=COMBINED_SPLIT,
        combined_split_sha256=combined_sha256,
        batches=tuple(components),
        total_cell_count=sum(component.cell_count for component in components),
        scalar_label_count=sum(component.scalar_label_count for component in components),
        hybrid_eligible_count=sum(
            component.hybrid_eligible_count for component in components
        ),
        hybrid_excluded_count=sum(
            component.hybrid_excluded_count for component in components
        ),
        created_at=datetime.now(UTC),
    )
    manifest_path = _path(COMBINED_MANIFEST, must_exist=False)
    if manifest_path.exists():
        existing = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
        if existing.model_dump(exclude={"created_at"}) != proposed.model_dump(
            exclude={"created_at"}
        ):
            raise ValueError("existing three-batch manifest differs from verified inputs")
        manifest = existing
    else:
        _write_json_atomic(manifest_path, proposed.model_dump(mode="json"))
        manifest = proposed
    print(
        json.dumps(
            {
                "status": "MATR_THREE_BATCH_INPUTS_READY",
                "data_version": manifest.data_version,
                "total_cell_count": manifest.total_cell_count,
                "scalar_label_count": manifest.scalar_label_count,
                "hybrid_eligible_count": manifest.hybrid_eligible_count,
                "hybrid_excluded_count": manifest.hybrid_excluded_count,
                "combined_split_sha256": manifest.combined_split_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare_batch(
    spec: BatchSpec,
) -> tuple[MatrBatchArtifactReference, SplitManifest]:
    raw_path = _path(spec.raw_path)
    manifest_path = _path(spec.raw_manifest)
    raw_manifest = RawFileManifest.model_validate_json(manifest_path.read_bytes())
    verify_raw_file(raw_path, raw_manifest)
    processed_root = _path(spec.processed_root, must_exist=False)
    conversion_path = _path(spec.conversion_report, must_exist=False)
    if conversion_path.exists():
        conversion = MatrBatchConversionReport.model_validate_json(
            conversion_path.read_bytes()
        )
    else:
        conversion = convert_matr_batch(
            raw_path=raw_path,
            raw_manifest=raw_manifest,
            output_root=processed_root,
            batch_index=spec.batch_index,
            batch_date=spec.batch_date,
            time_unit="minutes",
            max_cycle_index=150,
        )
        _write_json_atomic(conversion_path, conversion.model_dump(mode="json"))
    observed_counts = _verify_conversion(
        conversion,
        spec=spec,
        raw_manifest=raw_manifest,
        processed_root=processed_root,
    )

    split_path = _path(spec.split_manifest, must_exist=False)
    if split_path.exists():
        split = SplitManifest.model_validate_json(split_path.read_bytes())
    else:
        evidence = build_matr_split_evidence(
            conversion,
            split_version=f"matr-b{spec.batch_index}-cell-split-v1",
            created_at=datetime.now(UTC),
        )
        split = evidence.split_manifest
        _write_json_atomic(split_path, split.model_dump(mode="json"))
        _write_json_atomic(
            _path(spec.split_evidence, must_exist=False),
            evidence.model_dump(mode="json"),
        )
    if set(split.all_cells) != {cell.cell_id for cell in conversion.cells}:
        raise ValueError("batch split does not exactly cover converted cells")

    proposed_eligibility = audit_matr_supervision_eligibility(
        batch_index=spec.batch_index,
        observed_cycle_counts=observed_counts,
        horizon_cycle=500,
        created_at=datetime.now(UTC),
    )
    eligibility_path = _path(spec.eligibility_report, must_exist=False)
    eligibility = _load_or_publish_eligibility(
        eligibility_path,
        proposed_eligibility,
    )
    supervision_root = _path(spec.supervision_root, must_exist=False)
    supervision_path = _path(spec.supervision_report, must_exist=False)
    if supervision_path.exists():
        supervision = MatrSupervisionArtifact.model_validate_json(
            supervision_path.read_bytes()
        )
    else:
        supervision = build_matr_supervision_artifact(
            raw_path=raw_path,
            raw_manifest=raw_manifest,
            conversion_report=conversion,
            output_root=supervision_root,
            horizon_cycle=500,
            selected_cell_ids=eligibility.eligible_cell_ids,
            created_at=datetime.now(UTC),
        )
        _write_json_atomic(supervision_path, supervision.model_dump(mode="json"))
    _verify_supervision(
        supervision,
        supervision_root=supervision_root,
        conversion=conversion,
        raw_manifest=raw_manifest,
        eligibility=eligibility,
    )
    return (
        MatrBatchArtifactReference(
            batch_index=spec.batch_index,
            batch_date=spec.batch_date,
            raw_relative_path=spec.raw_path,
            raw_manifest=spec.raw_manifest,
            raw_sha256=raw_manifest.sha256,
            processed_root=spec.processed_root,
            conversion_report=spec.conversion_report,
            conversion_report_sha256=_sha256_file(conversion_path),
            split_manifest=spec.split_manifest,
            split_manifest_sha256=_sha256_file(split_path),
            supervision_root=spec.supervision_root,
            supervision_report=spec.supervision_report,
            supervision_report_sha256=_sha256_file(supervision_path),
            eligibility_report=spec.eligibility_report,
            eligibility_report_sha256=_sha256_file(eligibility_path),
            cell_count=conversion.cell_count,
            scalar_label_count=sum(
                not cell.official_life_right_censored for cell in conversion.cells
            ),
            hybrid_eligible_count=len(eligibility.eligible_cell_ids),
            hybrid_excluded_count=len(eligibility.excluded),
        ),
        split,
    )


def _verify_conversion(
    conversion: MatrBatchConversionReport,
    *,
    spec: BatchSpec,
    raw_manifest: RawFileManifest,
    processed_root: Path,
) -> dict[str, int]:
    if (
        conversion.batch_index != spec.batch_index
        or conversion.batch_date != spec.batch_date
        or conversion.time_unit != "minutes"
        or conversion.max_cycle_index != 150
        or conversion.raw_sha256 != raw_manifest.sha256
    ):
        raise ValueError("MATR batch conversion context mismatch")
    observed: dict[str, int] = {}
    for cell in conversion.cells:
        manifest_path = processed_root / cell.manifest_relative_path
        processed = ProcessedCellManifest.model_validate_json(manifest_path.read_bytes())
        verified = verify_cell_artifacts(processed_root, processed)
        value = verified.metadata.ingestion_parameters.get("observed_cycle_count")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("processed MATR metadata lacks observed_cycle_count")
        observed[cell.cell_id] = value
    return observed


def _verify_supervision(
    artifact: MatrSupervisionArtifact,
    *,
    supervision_root: Path,
    conversion: MatrBatchConversionReport,
    raw_manifest: RawFileManifest,
    eligibility: MatrTrajectoryEligibilityAudit,
) -> None:
    if (
        artifact.raw_sha256 != raw_manifest.sha256
        or artifact.source_report_sha256
        != sha256_canonical(conversion.model_dump(mode="json"))
        or artifact.horizon_cycle != 500
        or tuple(cell.cell_id for cell in artifact.cells)
        != tuple(
            cell.cell_id
            for cell in conversion.cells
            if cell.cell_id in set(eligibility.eligible_cell_ids)
        )
    ):
        raise ValueError("MATR supervision context mismatch")
    parquet_path = supervision_root / artifact.parquet_relative_path
    if _sha256_file(parquet_path) != artifact.parquet_sha256:
        raise ValueError("MATR supervision Parquet SHA-256 mismatch")


def _load_or_publish_eligibility(
    path: Path,
    proposed: MatrTrajectoryEligibilityAudit,
) -> MatrTrajectoryEligibilityAudit:
    if path.exists():
        existing = MatrTrajectoryEligibilityAudit.model_validate_json(path.read_bytes())
        if existing.model_dump(exclude={"created_at"}) != proposed.model_dump(
            exclude={"created_at"}
        ):
            raise ValueError("existing trajectory eligibility audit differs from inputs")
        return existing
    _write_json_atomic(path, proposed.model_dump(mode="json"))
    return proposed


def _path(relative: str, *, must_exist: bool = True) -> Path:
    path = (REPO_ROOT / relative).resolve(strict=must_exist)
    if not path.is_relative_to(REPO_ROOT):
        raise ValueError("three-batch preparation path escapes the repository")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
