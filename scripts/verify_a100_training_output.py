"""Verify downloaded A100 run bytes against an external output index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.outputs import (  # noqa: E402
    load_training_output_index,
    verify_training_output_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("output_index", type=Path)
    args = parser.parse_args()
    index = load_training_output_index(args.output_index)
    verify_training_output_index(args.output_root, index)
    print(
        f"verified run_id={index.run_manifest.run_id} "
        f"output_sha256={index.output_sha256} files={len(index.files)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
