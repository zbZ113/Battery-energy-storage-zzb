from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_converter_is_blocked_until_field_layout_is_approved(tmp_path: Path) -> None:
    output = tmp_path / "converted"
    result = subprocess.run(
        [
            sys.executable,
            "quarantine/hust/convert.py",
            "--layout",
            "configs/data_layouts/hust_mendeley_v2_layout_v1.json",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42
    assert '"status": "BLOCKED_REVIEW"' in result.stdout
    assert not output.exists()


def test_converter_source_never_uses_unrestricted_pickle_loading() -> None:
    text = Path("quarantine/hust/convert.py").read_text(encoding="utf-8")
    assert "pickle.load(" not in text
    assert "pickle.loads(" not in text
    assert "joblib.load(" not in text
    assert "torch.load(" not in text
