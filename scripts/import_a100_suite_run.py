"""Verify and register one completed three-batch MATR A100 experiment suite."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.application.a100_suite_import import (  # noqa: E402
    A100SuiteRunImporter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("config_path", type=Path)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--transfer-archive", required=True, type=Path)
    parser.add_argument("--transfer-sha256", required=True)
    args = parser.parse_args()

    record = A100SuiteRunImporter(args.registry_root).import_run(
        args.output_root,
        config_path=args.config_path,
        expected_source_commit=args.expected_source_commit,
        transfer_archive=args.transfer_archive,
        transfer_sha256=args.transfer_sha256,
        imported_at=datetime.now(UTC),
    )
    print(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
