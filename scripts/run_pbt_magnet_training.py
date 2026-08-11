#!/usr/bin/env python3
"""Run the closed-world PBT/MAGNet A100 training lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quanxin_life.training.pbt_magnet_runner import run_stage  # noqa: E402


def _models(value: str) -> tuple[str, ...]:
    models = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not models or len(models) != len(set(models)) or any(
        model not in {"pbt", "magnet"} for model in models
    ):
        raise argparse.ArgumentTypeError("models must be unique values from pbt,magnet")
    return models


def _seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be unique non-negative integers")
    return seeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_models, required=True)
    parser.add_argument(
        "--stage",
        choices=("smoke", "selection", "freeze-selection", "final", "collect"),
        required=True,
    )
    parser.add_argument("--seeds", type=_seeds, default=(38,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument(
        "--allow-partial-matrix",
        action="store_true",
        help="allow a local CPU tiny-fixture final run; formal A100 runs must not use this",
    )
    parser.add_argument(
        "--recovery-seed42",
        action="store_true",
        help="run only seed 42 in an isolated recovery namespace",
    )
    args = parser.parse_args()
    try:
        result = run_stage(
            project_root=args.project_root,
            models=args.models,
            stage=args.stage,
            seeds=args.seeds,
            device=args.device,
            enforce_formal_matrix=not args.allow_partial_matrix,
            recovery_seed42=args.recovery_seed42,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 42
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
