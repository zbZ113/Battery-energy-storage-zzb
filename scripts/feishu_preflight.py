"""Run secret-free Feishu/Aily configuration completeness checks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.integrations.feishu.preflight import run_feishu_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Check the local fake sandbox, which needs no real credentials.",
    )
    args = parser.parse_args()
    report = run_feishu_preflight(os.environ, production=not args.sandbox)
    print(
        json.dumps(
            {
                "mode": report.mode,
                "ready": report.ready,
                "configured": list(report.configured),
                "missing": list(report.missing),
                "invalid": list(report.invalid),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
