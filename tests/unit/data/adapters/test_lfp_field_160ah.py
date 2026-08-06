from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from quanxin_life.data.adapters.lfp_field_160ah import FieldLayout, load_field_sources


def test_field_adapter_uses_system_identity_and_never_invents_soh_or_rul(
    tmp_path: Path,
) -> None:
    path = tmp_path / "field.zip"
    members = ("field_data/data_sys_1.csv", "field_data/data_sys_2.csv")
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "Timestamp,U_Battery,I_Battery,SOC_Battery\n")
        archive.writestr("__MACOSX/field_data/._data_sys_1.csv", b"metadata")
    layout = FieldLayout(
        layout_version="fixture-v1",
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_members=members,
        ignored_members=("__MACOSX/field_data/._data_sys_1.csv",),
    )

    records = load_field_sources(path, layout=layout)

    assert {record.system_id for record in records} == {"FIELD-SYSTEM-1", "FIELD-SYSTEM-2"}
    serialized = str([record.model_dump() for record in records]).lower()
    assert "soh" not in serialized
    assert "rul" not in serialized
    assert all(record.split_group == record.system_id for record in records)
