"""Verify every committed model view below a root directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.data.model_views.builder import verify_model_view  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("data/model_views"))
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    project = args.project_root.resolve(strict=True)
    root = args.root if args.root.is_absolute() else project / args.root
    if not root.is_dir():
        raise ValueError("model view root does not exist")
    artifact_directories = sorted(
        {path.parent for path in root.rglob("*") if path.is_file()}
    )
    if not artifact_directories:
        raise ValueError("model view root contains no artifacts")
    for directory in artifact_directories:
        manifest = verify_model_view(directory)
        print(json.dumps({"status": "VERIFIED", "view_id": manifest.view_id}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
