"""Write a secret-safe A100 capability report without starting training."""

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

from quanxin_life.training.preflight import collect_local_training_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = collect_local_training_preflight(repo_root=REPO_ROOT)
    payload = json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
