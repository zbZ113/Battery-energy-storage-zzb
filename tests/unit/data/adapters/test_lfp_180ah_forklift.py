from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from quanxin_life.data.adapters.lfp_180ah_forklift import (
    ForkliftLayout,
    load_forklift_sources,
)


def test_forklift_preserves_cell_round_and_measurement_type(tmp_path: Path) -> None:
    path = tmp_path / "forklift.zip"
    root = "Lithium-ion battery degradation dataset based on a realistic forklift operation profile"
    members = (
        f"{root}/Cell1/Round01/RPT.csv",
        f"{root}/Cell1/Round01/Ageing.csv",
    )
    with zipfile.ZipFile(path, "w") as archive:
        for member in members:
            archive.writestr(member, "Cell,Round,Time,Current,Voltage\n1,1,0,0,3.2\n")
    layout = ForkliftLayout(
        layout_version="fixture-v1",
        archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_members=members,
    )

    records = load_forklift_sources(path, layout=layout)

    assert {record.measurement_type for record in records} == {"RPT", "AGEING"}
    assert {record.cell_id for record in records} == {"FORKLIFT-CELL-1"}
    assert {record.round_index for record in records} == {1}
