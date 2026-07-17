from datetime import UTC, date, datetime

from quanxin_life.data.matr_multibatch import (
    MatrBatchArtifactReference,
    MatrThreeBatchManifest,
    audit_matr_supervision_eligibility,
    combine_matr_batch_splits,
)
from quanxin_life.data.schemas import SplitManifest


def test_supervision_eligibility_keeps_only_cells_with_real_cycle_500() -> None:
    audit = audit_matr_supervision_eligibility(
        batch_index=2,
        observed_cycle_counts={"MATR_b2c0": 170, "MATR_b2c1": 501},
        horizon_cycle=500,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )

    assert audit.eligible_cell_ids == ("MATR_b2c1",)
    assert audit.excluded[0].cell_id == "MATR_b2c0"
    assert audit.excluded[0].observed_cycle_count == 170
    assert audit.excluded[0].reason == "INSUFFICIENT_REAL_TRAJECTORY_500"


def _split(batch_index: int, cell_count: int) -> SplitManifest:
    cells = tuple(f"MATR_b{batch_index}c{index}" for index in range(cell_count))
    return SplitManifest(
        dataset_id="MATR",
        train=cells[: max(1, int(cell_count * 0.6))],
        validation=cells[max(1, int(cell_count * 0.6)) : max(2, int(cell_count * 0.75))],
        calibration=cells[max(2, int(cell_count * 0.75)) : max(3, int(cell_count * 0.85))],
        test=cells[max(3, int(cell_count * 0.85)) :],
    )


def test_combined_split_covers_each_cell_once_and_preserves_batches() -> None:
    combined = combine_matr_batch_splits(
        (_split(1, 46), _split(2, 48), _split(3, 46))
    )

    all_ids = combined.train + combined.validation + combined.calibration + combined.test
    assert len(all_ids) == 140
    assert len(set(all_ids)) == 140
    for partition in (
        combined.train,
        combined.validation,
        combined.calibration,
        combined.test,
    ):
        assert {cell_id.split("c", 1)[0] for cell_id in partition} == {
            "MATR_b1",
            "MATR_b2",
            "MATR_b3",
        }


def _component(
    batch_index: int,
    batch_date: date,
    cell_count: int,
    scalar_labels: int,
    hybrid_eligible: int,
) -> MatrBatchArtifactReference:
    return MatrBatchArtifactReference(
        batch_index=batch_index,
        batch_date=batch_date,
        raw_relative_path=f"data/{batch_date.isoformat()}_batch.mat",
        raw_sha256=str(batch_index) * 64,
        processed_root=f"data/processed/MATR/{batch_date.isoformat()}-cutoff150",
        conversion_report=f"reports/data_quality/batch{batch_index}-conversion.json",
        conversion_report_sha256="a" * 64,
        split_manifest=f"configs/data_splits/batch{batch_index}.json",
        split_manifest_sha256="b" * 64,
        supervision_root=f"data/processed/MATR/{batch_date.isoformat()}-supervision500",
        supervision_report=f"reports/data_quality/batch{batch_index}-supervision.json",
        supervision_report_sha256="c" * 64,
        eligibility_report=f"reports/data_quality/batch{batch_index}-eligibility.json",
        eligibility_report_sha256="d" * 64,
        cell_count=cell_count,
        scalar_label_count=scalar_labels,
        hybrid_eligible_count=hybrid_eligible,
        hybrid_excluded_count=cell_count - hybrid_eligible,
    )


def test_three_batch_manifest_binds_140_cells_and_task_specific_counts() -> None:
    manifest = MatrThreeBatchManifest(
        data_version="matr-three-batch-v1",
        split_version="matr-three-batch-cell-split-v1",
        combined_split_manifest="configs/data_splits/matr_three_batch_v1.json",
        combined_split_sha256="e" * 64,
        batches=(
            _component(1, date(2017, 5, 12), 46, 46, 46),
            _component(2, date(2017, 6, 30), 48, 48, 28),
            _component(3, date(2018, 4, 12), 46, 44, 46),
        ),
        total_cell_count=140,
        scalar_label_count=138,
        hybrid_eligible_count=120,
        hybrid_excluded_count=20,
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
    )

    assert tuple(batch.batch_index for batch in manifest.batches) == (1, 2, 3)
    assert manifest.scalar_label_count == 138
    assert manifest.hybrid_eligible_count == 120
