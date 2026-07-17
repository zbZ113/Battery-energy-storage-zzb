from datetime import UTC, datetime

from quanxin_life.data.matr_multibatch import audit_matr_supervision_eligibility


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
