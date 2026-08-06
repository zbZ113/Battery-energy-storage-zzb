"""Fail-closed entry point for the governed A100 offline package."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("plan", "final"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required_locks = (
        Path("requirements/a100-linux-py311.lock"),
        Path("requirements/data-prep-py311.lock"),
        Path("requirements/blast-cpu-py311.lock"),
    )
    incomplete = [
        path.as_posix()
        for path in required_locks
        if "BLOCKED_DEPENDENCY" in path.read_text(encoding="utf-8")
    ]
    if incomplete:
        print({"status": "BLOCKED_DEPENDENCY", "incomplete_locks": incomplete})
        return 42
    if args.mode == "plan":
        print({"status": "READY", "output": str(args.output)})
        return 0
    raise RuntimeError("final package builder requires completed hashed locks")


if __name__ == "__main__":
    raise SystemExit(main())
