from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from quanxin_life.data.adapters.lfp_280ah_dod import (
    DodArchiveLayout,
    DodCapacityArchiveSelection,
    DodCapacityDatasetLayout,
    DodTelemetryColumns,
    load_dod_capacity_series,
)

HEADER = (
    "数据序号,循环号,工步号,工步类型,时间,总时间,电流(A),电压(V),容量(Ah),"
    "充电容量(Ah),放电容量(Ah),能量(Wh),充电能量(Wh),放电能量(Wh),"
    "绝对时间,功率(W),temp1_1,\n"
)


def _nested_zip(members: dict[str, str]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, text in members.items():
            archive.writestr(name, text.encode("gb18030"))
    return payload.getvalue()


def _outer_zip(path: Path, member: str, nested: bytes) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, nested)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _layout(
    *,
    member: str,
    outer_sha256: str,
    temperature_optional_csv_members: tuple[str, ...] = (),
) -> DodCapacityDatasetLayout:
    return DodCapacityDatasetLayout(
        schema_version="lfp-280ah-dod-capacity-layout-v1",
        layout_version="fixture-v1",
        nominal_capacity_ah=280.0,
        selected_dod_fraction=1.0,
        c_rate=0.5,
        csv_encoding="gb18030",
        normalization="FIRST_VALID_CYCLE_CAPACITY",
        temperature_optional_csv_members=temperature_optional_csv_members,
        columns=DodTelemetryColumns(
            cycle_number="循环号",
            discharge_capacity_ah="放电容量(Ah)",
            absolute_time="绝对时间",
            temperature_c="temp1_1",
        ),
        archives=(
            DodCapacityArchiveSelection(
                outer_relative_path="data/raw/LFP_280AH_DOD/v3/CATL.zip",
                archive=DodArchiveLayout(
                    layout_version="fixture-v1",
                    vendor="CATL",
                    archive_sha256=outer_sha256,
                    expected_members=(member,),
                ),
                selected_members=(member,),
            ),
        ),
    )


def test_capacity_adapter_streams_reviewed_nested_csv_into_cell_series(
    tmp_path: Path,
) -> None:
    member = "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip"
    nested = _nested_zip(
        {
            "cell-0.5C-100%DOD-1.csv": HEADER
            + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,281,0,0,0,"
            "2024-01-01 00:00:00,0,24,\n"
            + "2,1,1,放电,00:00:01,00:00:01,-140,3.1,0,0,282,0,0,0,"
            "2024-01-01 00:00:01,0,26,\n",
            "cell-0.5C-100%DOD-2.csv": HEADER
            + "3,2,1,放电,00:00:00,05:00:00,-140,3.2,0,0,279,0,0,0,"
            "2024-01-01 05:00:00,0,25,\n"
            + "4,2,1,放电,00:00:01,05:00:01,-140,3.1,0,0,280,0,0,0,"
            "2024-01-01 05:00:01,0,27,\n",
        }
    )
    outer = tmp_path / "CATL.zip"
    sha256 = _outer_zip(outer, member, nested)

    series = load_dod_capacity_series(
        outer,
        layout=_layout(member=member, outer_sha256=sha256),
        vendor="CATL",
        temporary_directory=tmp_path / "work",
    )

    assert len(series) == 1
    cell = series[0]
    assert cell.cell_id == "CATL-2763"
    assert cell.dod_fraction == 1.0
    assert cell.reference_capacity_ah == 282.0
    assert cell.source_member == member
    assert [point.cycle_number for point in cell.observations] == [1, 2]
    assert cell.observations[0].relative_capacity_ratio == 1.0
    assert cell.observations[1].relative_capacity_ratio == pytest.approx(280 / 282)
    assert cell.observations[1].nominal_soh == 1.0
    assert cell.observations[1].elapsed_days == pytest.approx(5 * 3600 / 86400)
    assert cell.observations[0].mean_temperature_c == 25.0
    assert not (tmp_path / "work").exists()


def test_capacity_adapter_rejects_non_100_percent_dod_selection(tmp_path: Path) -> None:
    member = "CATL/CATL-0.5C-60% Depth of Discharge/3821.zip"
    nested = _nested_zip(
        {
            "cell-0.5C-60%DOD-1.csv": HEADER
            + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,168,0,0,0,"
            "2024-01-01 00:00:00,0,25,\n"
        }
    )
    outer = tmp_path / "CATL.zip"
    sha256 = _outer_zip(outer, member, nested)

    with pytest.raises(ValueError, match="100% DoD"):
        load_dod_capacity_series(
            outer,
            layout=_layout(member=member, outer_sha256=sha256),
            vendor="CATL",
            temporary_directory=tmp_path / "work",
        )


def test_capacity_adapter_rejects_unsafe_inner_member(tmp_path: Path) -> None:
    member = "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip"
    nested = _nested_zip(
        {
            "../escape.csv": HEADER
            + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,280,0,0,0,"
            "2024-01-01 00:00:00,0,25,\n"
        }
    )
    outer = tmp_path / "CATL.zip"
    sha256 = _outer_zip(outer, member, nested)

    with pytest.raises(ValueError, match="unsafe nested ZIP member"):
        load_dod_capacity_series(
            outer,
            layout=_layout(member=member, outer_sha256=sha256),
            vendor="CATL",
            temporary_directory=tmp_path / "work",
        )


def test_capacity_adapter_rejects_unreviewed_missing_temperature_column(
    tmp_path: Path,
) -> None:
    member = "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip"
    missing_temperature_header = HEADER.replace("temp1_1,", "")
    nested = _nested_zip(
        {
            "cell-0.5C-100%DOD-1.csv": missing_temperature_header
            + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,282,0,0,0,"
            "2024-01-01 00:00:00,0,\n"
        }
    )
    outer = tmp_path / "CATL.zip"
    sha256 = _outer_zip(outer, member, nested)

    with pytest.raises(ValueError, match="temp1_1"):
        load_dod_capacity_series(
            outer,
            layout=_layout(member=member, outer_sha256=sha256),
            vendor="CATL",
            temporary_directory=tmp_path / "work",
        )


def test_capacity_adapter_excludes_reviewed_members_without_temperature(
    tmp_path: Path,
) -> None:
    member = "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip"
    missing_member = "cell-0.5C-100%DOD-1.csv"
    missing_temperature_header = HEADER.replace("temp1_1,", "")
    nested = _nested_zip(
        {
            missing_member: missing_temperature_header
            + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,282,0,0,0,"
            "2024-01-01 00:00:00,0,\n",
            "cell-0.5C-100%DOD-2.csv": HEADER
            + "2,2,1,放电,00:00:00,05:00:00,-140,3.2,0,0,281,0,0,0,"
            "2024-01-01 05:00:00,0,25,\n",
            "cell-0.5C-100%DOD-3.csv": HEADER
            + "3,3,1,放电,00:00:00,10:00:00,-140,3.2,0,0,280,0,0,0,"
            "2024-01-01 10:00:00,0,26,\n",
        }
    )
    outer = tmp_path / "CATL.zip"
    sha256 = _outer_zip(outer, member, nested)

    series = load_dod_capacity_series(
        outer,
        layout=_layout(
            member=member,
            outer_sha256=sha256,
            temperature_optional_csv_members=(missing_member,),
        ),
        vendor="CATL",
        temporary_directory=tmp_path / "work",
    )

    assert [point.cycle_number for point in series[0].observations] == [2, 3]
    assert series[0].reference_capacity_ah == 281.0
    assert series[0].temperature_missing_csv_members == (missing_member,)
