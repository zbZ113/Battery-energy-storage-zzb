"""Verify one committed canonical dataset bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.data.processing import DatasetProcessor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    manifest = DatasetProcessor(root).verify(args.bundle)
    print(
        json.dumps(
            {
                "dataset_id": manifest.dataset_id,
                "dataset_version": manifest.dataset_version,
                "status": "VERIFIED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
