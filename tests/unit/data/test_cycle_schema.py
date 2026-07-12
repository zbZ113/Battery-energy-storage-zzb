import pytest
from pydantic import ValidationError

from quanxin_life.data.schemas import (
    CycleRecord,
    DataQualityIssue,
    DataQualityReport,
    DataQualitySeverity,
)


def test_cycle_record_requires_physical_units_and_nonnegative_time() -> None:
    record = CycleRecord(
        dataset_id="MATR",
        cell_id="b1c0",
        cycle_index=1,
        sample_index=0,
        time_s=0.0,
        voltage_v=3.2,
        current_a=-1.1,
        temperature_c=30.0,
        discharge_capacity_ah=1.08,
        diagnostic=True,
    )

    assert record.voltage_v == 3.2
    assert record.discharge_capacity_ah == 1.08


def test_cycle_record_rejects_negative_elapsed_time() -> None:
    with pytest.raises(ValidationError):
        CycleRecord(
            dataset_id="MATR",
            cell_id="b1c0",
            cycle_index=1,
            sample_index=0,
            time_s=-0.1,
            voltage_v=3.2,
            current_a=0.0,
        )


def test_blocking_quality_issue_blocks_dataset() -> None:
    report = DataQualityReport(
        dataset_id="MATR",
        issues=(
            DataQualityIssue(
                code="UNIT_CONFLICT",
                severity=DataQualitySeverity.BLOCKING,
                message="voltage unit is ambiguous",
            ),
        ),
    )

    assert report.blocked is True
    assert report.quality_score == 0.0


def test_quality_score_is_one_without_issues() -> None:
    assert DataQualityReport(dataset_id="MATR").quality_score == 1.0
