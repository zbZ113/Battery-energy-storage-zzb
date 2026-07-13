import hashlib
from pathlib import Path

import openpyxl
import pytest

from quanxin_life.data.adapters.naumann_calendar import (
    CalendarConditionColumn,
    NaumannCalendarLayout,
    load_naumann_calendar_capacity,
    load_naumann_calendar_layout,
)
from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalogEntry


def _write_calendar_workbook(path: Path, rows: list[list[object]]) -> None:
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "calendar_capacity"
    for row in rows:
        worksheet.append(row)
    workbook.save(path)


def _rows() -> list[list[object]]:
    return [
        ["Discharge capacity / Ah", "", ""],
        ["Storage time / h", "TP_25C_50SOC", "TP_40C_75SOC"],
        [0.0, 3.00, 2.99],
        [100.0, 2.98, 2.90],
        [200.0, 2.96, 2.80],
    ]


def _source(**updates: object) -> SourceCatalogEntry:
    values: dict[str, object] = {
        "dataset_id": "NAUMANN_CALENDAR",
        "version": "Mendeley-v1",
        "source_uri": "https://data.mendeley.com/datasets/kxh42bfgtj/1",
        "paper_uri": "https://doi.org/10.1016/j.est.2018.01.019",
        "license_status": "CC BY 4.0",
        "ingestion_mode": IngestionMode.TABULAR,
        "expected_suffixes": (".xlsx",),
    }
    values.update(updates)
    return SourceCatalogEntry.model_validate(values)


def _manifest(path: Path, **updates: object) -> RawFileManifest:
    values: dict[str, object] = {
        "dataset_id": "NAUMANN_CALENDAR",
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_uri": "https://data.mendeley.com/datasets/kxh42bfgtj/1",
        "license_name": "CC BY 4.0",
        "paper_doi": "10.1016/j.est.2018.01.019",
    }
    values.update(updates)
    return RawFileManifest.model_validate(values)


def _layout() -> NaumannCalendarLayout:
    return NaumannCalendarLayout(
        layout_version="naumann-calendar-fixture-v1",
        sheet_name="calendar_capacity",
        capacity_header_row=1,
        capacity_header_column=1,
        capacity_header="Discharge capacity / Ah",
        time_header_row=2,
        first_observation_row=3,
        time_column=1,
        time_header="Storage time / h",
        condition_columns=(
            CalendarConditionColumn(
                column=2,
                expected_header="TP_25C_50SOC",
                condition_id="T25_SOC50",
                temperature_c=25.0,
                mean_soc=0.50,
            ),
            CalendarConditionColumn(
                column=3,
                expected_header="TP_40C_75SOC",
                condition_id="T40_SOC75",
                temperature_c=40.0,
                mean_soc=0.75,
            ),
        ),
    )


def test_loads_reviewed_calendar_capacity_as_condition_level_observations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "DischargeCapacity.xlsx"
    _write_calendar_workbook(path, _rows())

    observations = load_naumann_calendar_capacity(
        path,
        _manifest(path),
        _source(),
        layout=_layout(),
    )

    assert len(observations) == 6
    first = observations[0]
    assert first.dataset_id == "NAUMANN_CALENDAR"
    assert first.condition_id == "T25_SOC50"
    assert first.storage_time_h == 0.0
    assert first.temperature_c == 25.0
    assert first.mean_soc == 0.50
    assert first.capacity_ah == 3.00
    assert first.source_sha256 == _manifest(path).sha256
    assert first.layout_version == "naumann-calendar-fixture-v1"
    assert first.dod is None
    assert first.charge_c_rate is None
    assert first.discharge_c_rate is None


def test_loads_versioned_calendar_layout_from_json_without_column_inference(
    tmp_path: Path,
) -> None:
    layout_path = tmp_path / "calendar-layout.json"
    layout_path.write_text(_layout().model_dump_json(), encoding="utf-8")

    loaded = load_naumann_calendar_layout(layout_path)

    assert loaded == _layout()


def test_rejects_calendar_header_mismatch_before_treating_columns_as_conditions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "DischargeCapacity.xlsx"
    rows = _rows()
    rows[1][1] = "unreviewed-header"
    _write_calendar_workbook(path, rows)

    with pytest.raises(ValueError, match="expected_header"):
        load_naumann_calendar_capacity(path, _manifest(path), _source(), layout=_layout())


def test_rejects_calendar_capacity_header_mismatch_before_reading_measurements(
    tmp_path: Path,
) -> None:
    path = tmp_path / "DischargeCapacity.xlsx"
    rows = _rows()
    rows[0][0] = "Resistance / mOhm"
    _write_calendar_workbook(path, rows)

    with pytest.raises(ValueError, match="capacity header"):
        load_naumann_calendar_capacity(path, _manifest(path), _source(), layout=_layout())


def test_rejects_nonmonotonic_storage_time_without_reordering_rows(tmp_path: Path) -> None:
    path = tmp_path / "DischargeCapacity.xlsx"
    rows = _rows()
    rows[4][0] = 50.0
    _write_calendar_workbook(path, rows)

    with pytest.raises(ValueError, match="storage time must strictly increase"):
        load_naumann_calendar_capacity(path, _manifest(path), _source(), layout=_layout())


def test_rejects_catalog_or_manifest_mismatch_before_opening_calendar_workbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "DischargeCapacity.xlsx"
    _write_calendar_workbook(path, _rows())

    def workbook_open_is_forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("calendar workbook must not open before provenance validation")

    monkeypatch.setattr(openpyxl, "load_workbook", workbook_open_is_forbidden)
    with pytest.raises(ValueError, match="source URI"):
        load_naumann_calendar_capacity(
            path,
            _manifest(path),
            _source(source_uri="https://example.invalid/calendar"),
            layout=_layout(),
        )
