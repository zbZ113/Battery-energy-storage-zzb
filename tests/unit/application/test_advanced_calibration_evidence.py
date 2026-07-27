from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quanxin_life.application.advanced_calibration_evidence import (
    AdvancedCalibrationSourceRegistration,
    AdvancedRULCalibrationEvidence,
    AdvancedSOHCalibrationEvidence,
    RegisteredAdvancedCalibrationEvidenceResolver,
)
from quanxin_life.core import AdvancedModelTask, sha256_canonical
from quanxin_life.data.matr_multibatch import (
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    MatrTrajectoryExclusion,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrCellConversionEvidence,
    MatrSupervisionArtifact,
    MatrSupervisionCellEvidence,
)
from quanxin_life.data.schemas import SplitManifest

_NOW = datetime(2026, 7, 27, tzinfo=UTC)
_MANIFEST_PATH = Path("reports/data_quality/matr_three_batch_manifest_v1.json")


@dataclass(frozen=True)
class _Fixture:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    calibration_cells: tuple[str, ...]
    conversion_paths: tuple[Path, ...]
    split_paths: tuple[Path, ...]
    supervision_paths: tuple[Path, ...]
    parquet_paths: tuple[Path, ...]

    def resolver(self) -> RegisteredAdvancedCalibrationEvidenceResolver:
        return RegisteredAdvancedCalibrationEvidenceResolver(
            (
                AdvancedCalibrationSourceRegistration(
                    registration_id="matr-three-batch-final-v1",
                    evidence_root=self.root,
                    three_batch_manifest_sha256=self.manifest_sha256,
                ),
            )
        )


def test_resolves_rul_from_verified_official_calibration_labels(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    evidence = fixture.resolver().resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
    )

    assert isinstance(evidence, AdvancedRULCalibrationEvidence)
    assert tuple(cell.cell_id for cell in evidence.cells) == fixture.calibration_cells
    assert tuple(cell.observed_cycle for cell in evidence.cells) == (611, 612, 613)
    assert evidence.source_identity.registration_id == "matr-three-batch-final-v1"
    assert evidence.source_identity.data_version == "matr-three-batch-test-v1"
    assert evidence.source_identity.split_version == "matr-three-batch-split-v1"
    assert evidence.source_identity.three_batch_manifest_sha256 == fixture.manifest_sha256
    assert len(evidence.source_identity.source_identity_sha256) == 64
    assert "path" not in json.dumps(
        evidence.model_dump(mode="json"),
        sort_keys=True,
    ).lower()


@pytest.mark.parametrize(
    ("right_censored", "official_life_label", "match"),
    (
        (True, None, "non-censored official cycle life"),
        (False, 100, "greater than cutoff"),
    ),
)
def test_rejects_invalid_rul_calibration_labels(
    tmp_path: Path,
    right_censored: bool,
    official_life_label: int | None,
    match: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    payload = _read_json(fixture.conversion_paths[0])
    payload["cells"][0]["official_life_right_censored"] = right_censored
    payload["cells"][0]["official_life_label"] = official_life_label
    _replace_component_json(
        fixture,
        batch_index=1,
        field="conversion_report_sha256",
        target=fixture.conversion_paths[0],
        payload=payload,
    )

    with pytest.raises(ValueError, match=match):
        _resolver_for_current_manifest(fixture).resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=100,
        )


def test_resolves_soh_from_verified_finite_horizon_supervision(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)

    evidence = fixture.resolver().resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.SOH,
        cutoff_cycle=150,
    )

    assert isinstance(evidence, AdvancedSOHCalibrationEvidence)
    assert evidence.prediction_cycles == tuple(range(151, 501))
    assert tuple(cell.cell_id for cell in evidence.cells) == fixture.calibration_cells
    assert all(len(cell.observed_soh) == 350 for cell in evidence.cells)
    assert evidence.cells[0].observed_soh[0] == pytest.approx(0.999 - 151 / 10000)
    assert evidence.cells[0].observed_soh[-1] == pytest.approx(0.999 - 500 / 10000)
    assert all(
        cycle <= 500
        for cycle in evidence.prediction_cycles
    )


def test_resolves_verified_calibration_cell_source_and_cutoff_soh(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    resolver = fixture.resolver()
    rul = resolver.resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
    )
    soh = resolver.resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.SOH,
        cutoff_cycle=100,
    )
    cell_id = fixture.calibration_cells[0]

    rul_source = resolver.resolve_cell_source(
        "matr-three-batch-final-v1",
        source_identity=rul.source_identity,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
        cell_id=cell_id,
    )
    soh_source = resolver.resolve_cell_source(
        "matr-three-batch-final-v1",
        source_identity=soh.source_identity,
        task=AdvancedModelTask.SOH,
        cutoff_cycle=100,
        cell_id=cell_id,
    )

    assert rul_source.cell_evidence.cell_id == cell_id
    assert rul_source.processed_root == (
        fixture.root / "data/processed-1"
    ).resolve()
    assert rul_source.raw_sha256 == _digest("raw-1")
    assert rul_source.initial_soh is None
    assert soh_source.initial_soh == pytest.approx(0.989)


def test_cell_source_rejects_non_calibration_cell_and_changed_identity(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    resolver = fixture.resolver()
    evidence = resolver.resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.RUL,
        cutoff_cycle=100,
    )

    with pytest.raises(ValueError, match="calibration"):
        resolver.resolve_cell_source(
            "matr-three-batch-final-v1",
            source_identity=evidence.source_identity,
            task=AdvancedModelTask.RUL,
            cutoff_cycle=100,
            cell_id="MATR_b1c1",
        )

    with pytest.raises(ValueError, match="identity"):
        resolver.resolve_cell_source(
            "matr-three-batch-final-v1",
            source_identity=evidence.source_identity.model_copy(
                update={"source_identity_sha256": "f" * 64}
            ),
            task=AdvancedModelTask.RUL,
            cutoff_cycle=100,
            cell_id=fixture.calibration_cells[0],
        )


def test_rejects_supervision_replacement_after_initial_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_fixture(tmp_path)
    target = fixture.parquet_paths[0]
    real_read_bytes = Path.read_bytes

    def read_tampered_bytes(path: Path) -> bytes:
        if path == target:
            return b"tampered-at-consumption"
        return real_read_bytes(path)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        read_tampered_bytes,
    )

    with pytest.raises(ValueError, match="Parquet SHA-256"):
        fixture.resolver().resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
        )


def test_rejects_self_consistent_soh_with_wrong_reference_capacity(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    table = pq.read_table(fixture.parquet_paths[0])
    rows = table.to_pylist()
    target_cell = fixture.calibration_cells[0]
    target = next(
        row
        for row in rows
        if row["cell_id"] == target_cell and row["cycle_index"] == 100
    )
    target["reference_capacity_ah"] = 2.0
    target["discharge_capacity_ah"] = float(target["soh"]) * 2.0
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        fixture.parquet_paths[0],
    )
    _refresh_supervision_hashes(fixture, batch_index=1)

    with pytest.raises(ValueError, match="reference capacity"):
        _resolver_for_current_manifest(fixture).resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
        )


def test_rejects_eligibility_inventory_that_replaces_a_real_cell(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = _read_json(fixture.manifest_path)
    component = manifest["batches"][0]
    eligibility_path = fixture.root / component["eligibility_report"]
    eligibility = _read_json(eligibility_path)
    removed_cell = "MATR_b1c1"
    eligibility["eligible_cell_ids"].remove(removed_cell)
    eligibility["excluded"].append(
        MatrTrajectoryExclusion(
            cell_id="MATR_forged_exclusion",
            observed_cycle_count=400,
        ).model_dump(mode="json")
    )
    _write_json(eligibility_path, eligibility)
    component["eligibility_report_sha256"] = _sha256_file(
        eligibility_path
    )
    component["hybrid_eligible_count"] -= 1
    component["hybrid_excluded_count"] += 1
    manifest["hybrid_eligible_count"] -= 1
    manifest["hybrid_excluded_count"] += 1

    parquet_path = fixture.parquet_paths[0]
    table = pq.read_table(parquet_path)
    rows = [
        row
        for row in table.to_pylist()
        if row["cell_id"] != removed_cell
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        parquet_path,
    )
    supervision_path = fixture.supervision_paths[0]
    supervision = _read_json(supervision_path)
    supervision["cells"] = [
        cell
        for cell in supervision["cells"]
        if cell["cell_id"] != removed_cell
    ]
    supervision["cell_count"] -= 1
    supervision["row_count"] -= 500
    supervision["parquet_sha256"] = _sha256_file(parquet_path)
    _write_json(supervision_path, supervision)
    component["supervision_report_sha256"] = _sha256_file(
        supervision_path
    )
    _write_json(fixture.manifest_path, manifest)

    with pytest.raises(ValueError, match="inventory"):
        _resolver_for_current_manifest(fixture).resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
        )


def test_accepts_positive_capacity_derived_soh_without_inventing_upper_threshold(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    table = pq.read_table(fixture.parquet_paths[0])
    rows = table.to_pylist()
    non_calibration_cell = "MATR_b1c1"
    target = next(
        row
        for row in rows
        if row["cell_id"] == non_calibration_cell and row["cycle_index"] == 39
    )
    target["discharge_capacity_ah"] = 2.7
    target["reference_capacity_ah"] = 1.0
    target["soh"] = 2.7
    pq.write_table(
        pa.Table.from_pylist(rows, schema=table.schema),
        fixture.parquet_paths[0],
    )
    _refresh_supervision_hashes(fixture, batch_index=1)

    evidence = _resolver_for_current_manifest(fixture).resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.SOH,
        cutoff_cycle=150,
    )

    assert isinstance(evidence, AdvancedSOHCalibrationEvidence)
    assert tuple(cell.cell_id for cell in evidence.cells) == fixture.calibration_cells


def test_projects_soh_calibration_to_verified_eligible_cells(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    excluded_cell = fixture.calibration_cells[0]
    _exclude_supervision_cell(fixture, batch_index=1, cell_id=excluded_cell)

    evidence = _resolver_for_current_manifest(fixture).resolve(
        "matr-three-batch-final-v1",
        task=AdvancedModelTask.SOH,
        cutoff_cycle=150,
    )

    assert isinstance(evidence, AdvancedSOHCalibrationEvidence)
    assert tuple(cell.cell_id for cell in evidence.cells) == fixture.calibration_cells[1:]
    assert excluded_cell not in {cell.cell_id for cell in evidence.cells}


def test_rejects_manifest_sha_mismatch_and_duplicate_json_keys(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    wrong_registration = AdvancedCalibrationSourceRegistration(
        registration_id="wrong-sha",
        evidence_root=fixture.root,
        three_batch_manifest_sha256="0" * 64,
    )
    resolver = RegisteredAdvancedCalibrationEvidenceResolver((wrong_registration,))
    with pytest.raises(ValueError, match="SHA-256"):
        resolver.resolve(
            "wrong-sha",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
        )

    fixture.manifest_path.write_text(
        '{"schema_version":"matr-three-batch-manifest-v1",'
        '"schema_version":"matr-three-batch-manifest-v1"}',
        encoding="utf-8",
    )
    duplicate_sha = _sha256_file(fixture.manifest_path)
    duplicate_resolver = RegisteredAdvancedCalibrationEvidenceResolver(
        (
            AdvancedCalibrationSourceRegistration(
                registration_id="duplicate-json",
                evidence_root=fixture.root,
                three_batch_manifest_sha256=duplicate_sha,
            ),
        )
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        duplicate_resolver.resolve(
            "duplicate-json",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
        )


def test_rejects_path_and_symbolic_link_escape(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path / "evidence")
    manifest = _read_json(fixture.manifest_path)
    manifest["batches"][0]["conversion_report"] = "../../../../outside.json"
    _write_json(fixture.manifest_path, manifest)
    escaped = _resolver_for_current_manifest(fixture)
    with pytest.raises(ValueError, match="escapes"):
        escaped.resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
        )

    if not hasattr(Path, "symlink_to"):
        pytest.skip("symbolic links are unavailable")
    fixture = _build_fixture(tmp_path / "symlink-evidence")
    outside = tmp_path / "outside-conversion.json"
    outside.write_bytes(fixture.conversion_paths[0].read_bytes())
    fixture.conversion_paths[0].unlink()
    try:
        fixture.conversion_paths[0].symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable for this account")
    with pytest.raises(ValueError, match=r"symbolic link|escapes"):
        fixture.resolver().resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
        )


def test_rejects_component_and_combined_split_inventory_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _build_fixture(tmp_path)
    split_payload = _read_json(fixture.split_paths[0])
    split_payload["calibration"] = []
    split_payload["train"].append(fixture.calibration_cells[0])
    _replace_component_json(
        fixture,
        batch_index=1,
        field="split_manifest_sha256",
        target=fixture.split_paths[0],
        payload=split_payload,
    )

    with pytest.raises(ValueError, match=r"combined split|calibration"):
        _resolver_for_current_manifest(fixture).resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
        )


@pytest.mark.parametrize(
    ("mutator", "match"),
    (
        ("missing_cycle", "continuous"),
        ("duplicate_cycle", "unique"),
        ("non_finite", "finite"),
        ("inconsistent_soh", "capacity"),
    ),
)
def test_rejects_invalid_soh_parquet(
    tmp_path: Path,
    mutator: str,
    match: str,
) -> None:
    fixture = _build_fixture(tmp_path)
    table = pq.read_table(fixture.parquet_paths[0])
    rows = table.to_pylist()
    target_cell = fixture.calibration_cells[0]
    target_indexes = [
        index for index, row in enumerate(rows) if row["cell_id"] == target_cell
    ]
    if mutator == "missing_cycle":
        rows[target_indexes[199]]["cycle_index"] = 501
    elif mutator == "duplicate_cycle":
        rows[target_indexes[200]]["cycle_index"] = rows[target_indexes[199]][
            "cycle_index"
        ]
    elif mutator == "non_finite":
        rows[target_indexes[200]]["soh"] = float("nan")
    else:
        rows[target_indexes[200]]["soh"] = 0.4
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), fixture.parquet_paths[0])
    _refresh_supervision_hashes(fixture, batch_index=1)

    with pytest.raises(ValueError, match=match):
        _resolver_for_current_manifest(fixture).resolve(
            "matr-three-batch-final-v1",
            task=AdvancedModelTask.SOH,
            cutoff_cycle=100,
        )


def _build_fixture(root: Path) -> _Fixture:
    root.mkdir(parents=True, exist_ok=True)
    combined_parts: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    component_references: list[MatrBatchArtifactReference] = []
    conversion_paths: list[Path] = []
    split_paths: list[Path] = []
    supervision_paths: list[Path] = []
    parquet_paths: list[Path] = []
    calibration_cells: list[str] = []

    for batch_index, batch_date in (
        (1, date(2017, 5, 12)),
        (2, date(2017, 6, 30)),
        (3, date(2018, 4, 12)),
    ):
        cell_ids = tuple(f"MATR_b{batch_index}c{index}" for index in range(4))
        split = SplitManifest(
            dataset_id="MATR",
            seed=20260712,
            train=(cell_ids[1],),
            validation=(cell_ids[2],),
            calibration=(cell_ids[0],),
            test=(cell_ids[3],),
        )
        for partition in combined_parts:
            combined_parts[partition].extend(getattr(split, partition))
        calibration_cells.append(cell_ids[0])

        conversion = MatrBatchConversionReport(
            batch_index=batch_index,
            batch_date=batch_date,
            raw_relative_path=f"data/batch-{batch_index}.mat",
            raw_size_bytes=1024,
            raw_sha256=_digest(f"raw-{batch_index}"),
            source_uri=f"https://example.test/matr/{batch_index}",
            license_name="fixture",
            adapter_version="matr-fixture-v1",
            time_unit="minutes",
            max_cycle_index=150,
            cell_count=4,
            total_row_count=2400,
            quality_issue_counts={},
            cells=tuple(
                MatrCellConversionEvidence(
                    cell_id=cell_id,
                    raw_cell_id=f"b{batch_index}c{cell_index}",
                    official_life_label=610 + batch_index + cell_index,
                    official_life_right_censored=False,
                    protocol_id=f"protocol-{batch_index}",
                    reference_capacity_ah=1.0,
                    row_count=600,
                    cycle_count=150,
                    quality_issue_counts={},
                    manifest_relative_path=f"cells/{cell_id}/manifest.json",
                    parquet_sha256=_digest(f"cell-parquet-{cell_id}"),
                    metadata_sha256=_digest(f"cell-metadata-{cell_id}"),
                )
                for cell_index, cell_id in enumerate(cell_ids)
            ),
            created_at=_NOW,
        )
        conversion_path = root / f"reports/data_quality/conversion-{batch_index}.json"
        split_path = root / f"configs/data_splits/split-{batch_index}.json"
        eligibility_path = (
            root / f"reports/data_quality/eligibility-{batch_index}.json"
        )
        supervision_path = (
            root / f"reports/data_quality/supervision-{batch_index}.json"
        )
        (root / f"data/processed-{batch_index}").mkdir(parents=True)
        supervision_root = root / f"data/supervision-{batch_index}"
        parquet_path = supervision_root / "trajectories/labels.parquet"
        _write_json(conversion_path, conversion.model_dump(mode="json"))
        _write_json(split_path, split.model_dump(mode="json"))
        eligibility = MatrTrajectoryEligibilityAudit(
            batch_index=batch_index,
            horizon_cycle=500,
            eligible_cell_ids=cell_ids,
            excluded=(),
            created_at=_NOW,
        )
        _write_json(eligibility_path, eligibility.model_dump(mode="json"))
        _write_supervision_parquet(parquet_path, cell_ids)
        parquet_sha256 = _sha256_file(parquet_path)
        supervision = MatrSupervisionArtifact(
            source_report_sha256=sha256_canonical(
                conversion.model_dump(mode="json")
            ),
            raw_sha256=conversion.raw_sha256,
            horizon_cycle=500,
            cell_count=4,
            row_count=2000,
            parquet_relative_path="trajectories/labels.parquet",
            parquet_sha256=parquet_sha256,
            cells=tuple(
                MatrSupervisionCellEvidence(
                    cell_id=cell_id,
                    raw_cell_id=f"b{batch_index}c{cell_index}",
                    official_life_label=610 + batch_index + cell_index,
                    official_life_right_censored=False,
                    reference_capacity_ah=1.0,
                    observed_cycle_count=700,
                )
                for cell_index, cell_id in enumerate(cell_ids)
            ),
            created_at=_NOW,
        )
        _write_json(supervision_path, supervision.model_dump(mode="json"))
        component_references.append(
            MatrBatchArtifactReference(
                batch_index=batch_index,
                batch_date=batch_date,
                raw_relative_path=f"data/batch-{batch_index}.mat",
                raw_manifest=f"configs/data_manifests/raw-{batch_index}.json",
                raw_sha256=conversion.raw_sha256,
                processed_root=f"data/processed-{batch_index}",
                conversion_report=conversion_path.relative_to(root).as_posix(),
                conversion_report_sha256=_sha256_file(conversion_path),
                split_manifest=split_path.relative_to(root).as_posix(),
                split_manifest_sha256=_sha256_file(split_path),
                supervision_root=supervision_root.relative_to(root).as_posix(),
                supervision_report=supervision_path.relative_to(root).as_posix(),
                supervision_report_sha256=_sha256_file(supervision_path),
                eligibility_report=eligibility_path.relative_to(root).as_posix(),
                eligibility_report_sha256=_sha256_file(eligibility_path),
                cell_count=4,
                scalar_label_count=4,
                hybrid_eligible_count=4,
                hybrid_excluded_count=0,
            )
        )
        conversion_paths.append(conversion_path)
        split_paths.append(split_path)
        supervision_paths.append(supervision_path)
        parquet_paths.append(parquet_path)

    combined_split = SplitManifest(
        dataset_id="MATR",
        seed=20260712,
        train=tuple(combined_parts["train"]),
        validation=tuple(combined_parts["validation"]),
        calibration=tuple(combined_parts["calibration"]),
        test=tuple(combined_parts["test"]),
    )
    combined_path = root / "configs/data_splits/combined.json"
    _write_json(combined_path, combined_split.model_dump(mode="json"))
    manifest = MatrThreeBatchManifest(
        data_version="matr-three-batch-test-v1",
        split_version="matr-three-batch-split-v1",
        combined_split_manifest=combined_path.relative_to(root).as_posix(),
        combined_split_sha256=_sha256_file(combined_path),
        batches=tuple(component_references),
        total_cell_count=12,
        scalar_label_count=12,
        hybrid_eligible_count=12,
        hybrid_excluded_count=0,
        created_at=_NOW,
    )
    manifest_path = root / _MANIFEST_PATH
    _write_json(manifest_path, manifest.model_dump(mode="json"))
    return _Fixture(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        calibration_cells=tuple(calibration_cells),
        conversion_paths=tuple(conversion_paths),
        split_paths=tuple(split_paths),
        supervision_paths=tuple(supervision_paths),
        parquet_paths=tuple(parquet_paths),
    )


def _write_supervision_parquet(path: Path, cell_ids: tuple[str, ...]) -> None:
    rows = []
    for cell_index, cell_id in enumerate(cell_ids):
        reference_capacity = 1.0
        for cycle in range(1, 501):
            soh = 0.999 - cycle / 10000 - cell_index / 100000
            rows.append(
                {
                    "dataset_id": "MATR",
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "discharge_capacity_ah": soh * reference_capacity,
                    "reference_capacity_ah": reference_capacity,
                    "soh": soh,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _replace_component_json(
    fixture: _Fixture,
    *,
    batch_index: int,
    field: str,
    target: Path,
    payload: dict[str, Any],
) -> None:
    _write_json(target, payload)
    manifest = _read_json(fixture.manifest_path)
    manifest["batches"][batch_index - 1][field] = _sha256_file(target)
    if field == "conversion_report_sha256":
        conversion = MatrBatchConversionReport.model_validate(payload)
        scalar_count = sum(
            not cell.official_life_right_censored
            and cell.official_life_label is not None
            for cell in conversion.cells
        )
        previous_count = manifest["batches"][batch_index - 1]["scalar_label_count"]
        manifest["batches"][batch_index - 1]["scalar_label_count"] = scalar_count
        manifest["scalar_label_count"] += scalar_count - previous_count
        supervision_path = fixture.supervision_paths[batch_index - 1]
        supervision = _read_json(supervision_path)
        supervision["source_report_sha256"] = sha256_canonical(
            conversion.model_dump(mode="json")
        )
        by_id = {cell.cell_id: cell for cell in conversion.cells}
        for cell in supervision["cells"]:
            source = by_id[cell["cell_id"]]
            cell["official_life_label"] = source.official_life_label
            cell["official_life_right_censored"] = (
                source.official_life_right_censored
            )
        _write_json(supervision_path, supervision)
        manifest["batches"][batch_index - 1][
            "supervision_report_sha256"
        ] = _sha256_file(supervision_path)
    _write_json(fixture.manifest_path, manifest)


def _refresh_supervision_hashes(fixture: _Fixture, *, batch_index: int) -> None:
    supervision_path = fixture.supervision_paths[batch_index - 1]
    supervision = _read_json(supervision_path)
    supervision["parquet_sha256"] = _sha256_file(
        fixture.parquet_paths[batch_index - 1]
    )
    supervision["row_count"] = pq.read_metadata(
        fixture.parquet_paths[batch_index - 1]
    ).num_rows
    _write_json(supervision_path, supervision)
    manifest = _read_json(fixture.manifest_path)
    manifest["batches"][batch_index - 1][
        "supervision_report_sha256"
    ] = _sha256_file(supervision_path)
    _write_json(fixture.manifest_path, manifest)


def _exclude_supervision_cell(
    fixture: _Fixture,
    *,
    batch_index: int,
    cell_id: str,
) -> None:
    manifest = _read_json(fixture.manifest_path)
    component = manifest["batches"][batch_index - 1]
    eligibility_path = fixture.root / component["eligibility_report"]
    eligibility = _read_json(eligibility_path)
    eligibility["eligible_cell_ids"].remove(cell_id)
    eligibility["excluded"].append(
        MatrTrajectoryExclusion(
            cell_id=cell_id,
            observed_cycle_count=400,
        ).model_dump(mode="json")
    )
    _write_json(eligibility_path, eligibility)
    component["eligibility_report_sha256"] = _sha256_file(eligibility_path)
    component["hybrid_eligible_count"] -= 1
    component["hybrid_excluded_count"] += 1
    manifest["hybrid_eligible_count"] -= 1
    manifest["hybrid_excluded_count"] += 1

    parquet_path = fixture.parquet_paths[batch_index - 1]
    table = pq.read_table(parquet_path)
    rows = [row for row in table.to_pylist() if row["cell_id"] != cell_id]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), parquet_path)
    supervision_path = fixture.supervision_paths[batch_index - 1]
    supervision = _read_json(supervision_path)
    supervision["cells"] = [
        cell for cell in supervision["cells"] if cell["cell_id"] != cell_id
    ]
    supervision["cell_count"] -= 1
    supervision["row_count"] -= 500
    supervision["parquet_sha256"] = _sha256_file(parquet_path)
    _write_json(supervision_path, supervision)
    component["supervision_report_sha256"] = _sha256_file(supervision_path)
    _write_json(fixture.manifest_path, manifest)


def _resolver_for_current_manifest(
    fixture: _Fixture,
) -> RegisteredAdvancedCalibrationEvidenceResolver:
    return RegisteredAdvancedCalibrationEvidenceResolver(
        (
            AdvancedCalibrationSourceRegistration(
                registration_id="matr-three-batch-final-v1",
                evidence_root=fixture.root,
                three_batch_manifest_sha256=_sha256_file(fixture.manifest_path),
            ),
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
