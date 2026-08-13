from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FONT_SHA256 = "cd42dca9abc49fc97b6e5426afd8bcf87b6b002ace89bb1d03e9b2f4ecfa32d5"
LICENSE_SHA256 = "33b361c36fd26e0e7d8033c4520bd3acc00de78c64e6e2af42a32b6d7e3db7c9"
UPSTREAM_SHA256 = "d1437797cc993243f596990a927a75d88413b62f8c07a79d4b7a5406e7f01af4"


def test_feishu_soh_plot_assets_are_present_and_provenanced_in_wheel(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        members = {
            "quanxin_life/integrations/feishu/assets/QuanxinSohSans-Regular.ttf",
            "quanxin_life/integrations/feishu/assets/OFL.txt",
            "quanxin_life/integrations/feishu/assets/UPSTREAM.json",
        }
        assert members <= set(archive.namelist())
        font = archive.read(
            "quanxin_life/integrations/feishu/assets/QuanxinSohSans-Regular.ttf"
        )
        license_text = archive.read(
            "quanxin_life/integrations/feishu/assets/OFL.txt"
        )
        upstream_bytes = archive.read(
            "quanxin_life/integrations/feishu/assets/UPSTREAM.json"
        )

    assert hashlib.sha256(font).hexdigest() == FONT_SHA256
    assert hashlib.sha256(license_text).hexdigest() == LICENSE_SHA256
    assert hashlib.sha256(upstream_bytes).hexdigest() == UPSTREAM_SHA256
    assert b"SIL Open Font License" in license_text
    upstream = json.loads(upstream_bytes)
    assert upstream["output_sha256"] == FONT_SHA256
    assert upstream["license"] == "SIL Open Font License 1.1"
