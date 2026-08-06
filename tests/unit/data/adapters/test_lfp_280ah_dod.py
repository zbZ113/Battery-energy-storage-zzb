from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from quanxin_life.data.adapters.lfp_280ah_dod import DodArchiveLayout, load_dod_sources


def _zip(path: Path, members: dict[str, bytes]) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dod_adapter_uses_vendor_cell_and_dod_from_reviewed_members(tmp_path: Path) -> None:
    path = tmp_path / "CATL.zip"
    member = "CATL/CATL-0.5C-20% Depth of Discharge/2899.zip"
    sha256 = _zip(path, {member: b"nested reviewed cell archive"})
    layout = DodArchiveLayout(
        layout_version="fixture-v1",
        vendor="CATL",
        archive_sha256=sha256,
        expected_members=(member,),
    )

    records = load_dod_sources(path, layout=layout)

    assert records[0].cell_id == "CATL-2899"
    assert records[0].dod_fraction == 0.2
    assert records[0].task_role in {"calibration", "external_test"}
    assert "eol" not in str(records[0].model_dump()).lower()


def test_zip_adapter_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    sha256 = _zip(path, {"../escape.zip": b"bad"})
    layout = DodArchiveLayout(
        layout_version="fixture-v1",
        vendor="CATL",
        archive_sha256=sha256,
        expected_members=("../escape.zip",),
    )

    with pytest.raises(ValueError, match="unsafe ZIP member"):
        load_dod_sources(path, layout=layout)
