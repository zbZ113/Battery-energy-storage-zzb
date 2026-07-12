from quanxin_life.data.schemas import CycleRecord, DataQualitySeverity
from quanxin_life.data.validation import validate_cycle_records


def _record(cycle: int, sample: int, time_s: float, **overrides: object) -> CycleRecord:
    values: dict[str, object] = {
        "dataset_id": "MATR",
        "cell_id": "MATR_b1c0",
        "cycle_index": cycle,
        "sample_index": sample,
        "time_s": time_s,
        "voltage_v": 3.2,
        "current_a": -1.0,
        "temperature_c": 30.0,
    }
    values.update(overrides)
    return CycleRecord.model_validate(values)


def test_empty_cell_is_blocked() -> None:
    report = validate_cycle_records(())

    assert report.blocked
    assert report.issues[0].code == "EMPTY_CELL"


def test_duplicate_sample_key_is_blocking() -> None:
    record = _record(1, 0, 0.0)

    report = validate_cycle_records((record, record))

    issue = next(issue for issue in report.issues if issue.code == "DUPLICATE_SAMPLE")
    assert issue.severity == DataQualitySeverity.BLOCKING


def test_time_must_increase_within_cycle() -> None:
    report = validate_cycle_records((_record(1, 0, 1.0), _record(1, 1, 0.5)))

    assert any(issue.code == "NON_MONOTONIC_TIME" for issue in report.issues)


def test_mixed_cells_are_blocked() -> None:
    report = validate_cycle_records(
        (_record(1, 0, 0.0), _record(1, 1, 1.0, cell_id="MATR_b1c1"))
    )

    assert any(issue.code == "MIXED_CELL" for issue in report.issues)
    assert report.blocked


def test_cycle_gap_and_missing_temperature_are_reported_once() -> None:
    report = validate_cycle_records(
        (
            _record(1, 0, 0.0, temperature_c=None),
            _record(3, 0, 0.0, temperature_c=None),
        )
    )

    codes = [issue.code for issue in report.issues]
    assert codes.count("MISSING_TEMPERATURE") == 1
    assert codes.count("CYCLE_GAP") == 1
