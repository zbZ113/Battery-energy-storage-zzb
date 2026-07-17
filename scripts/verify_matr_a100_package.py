"""Verify the external and internal MATR A100 package hashes before extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.a100_package import (  # noqa: E402
    MatrA100ArchiveIndex,
    verify_matr_a100_archive,
    verify_matr_a100_archive_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    index = MatrA100ArchiveIndex.model_validate_json(args.index.read_bytes())
    verify_matr_a100_archive_index(args.archive, index)
    manifest = verify_matr_a100_archive(args.archive)
    if manifest.package_sha256 != index.package_sha256:
        raise ValueError("internal package digest does not match the external index")
    if manifest.source_commit != index.source_commit:
        raise ValueError("internal source revision does not match the external index")
    print(
        json.dumps(
            {
                "status": "MATR_A100_PACKAGE_VERIFIED",
                "archive_sha256": index.archive_sha256,
                "package_sha256": manifest.package_sha256,
                "source_commit": manifest.source_commit,
                "file_count": len(manifest.files),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
