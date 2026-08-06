from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from quanxin_life.data.adapters.lfp_280ah_tempest import (
    TempestArchive,
    TempestLayout,
    load_tempest_index,
)


def _zip(path: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tempest_keeps_all_temperatures_under_one_cell_identity(tmp_path: Path) -> None:
    aging = tmp_path / "aging.zip"
    characterization = tmp_path / "characterization.zip"
    aging_member = "LIC_aging/LIC_aging_data.csv"
    characterization_members = (
        "capacity_hppc_entropy/15C_capacity.csv",
        "capacity_hppc_entropy/35C_capacity.csv",
    )
    aging_sha = _zip(aging, {aging_member: b"Time_s\n1\n"})
    characterization_sha = _zip(
        characterization, {member: b"Time_s\n1\n" for member in characterization_members}
    )
    layout = TempestLayout(
        layout_version="fixture-v1",
        cell_id="TEMPEST-LIC-280AH",
        archives=(
            TempestArchive(
                role="aging", archive_sha256=aging_sha, expected_members=(aging_member,)
            ),
            TempestArchive(
                role="characterization",
                archive_sha256=characterization_sha,
                expected_members=characterization_members,
            ),
        ),
    )

    index = load_tempest_index((aging, characterization), layout=layout)

    assert index.cell_id == "TEMPEST-LIC-280AH"
    assert index.temperatures_c == (15, 35)
    assert index.split_policy == "single_cell_no_temperature_split"
