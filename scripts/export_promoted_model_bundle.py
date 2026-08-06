#!/usr/bin/env python3
"""Export a SHA-verified offline bundle without activating a model route."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quanxin_life.training.promotion_bundle import (  # noqa: E402
    PromotionBundleManifest,
    export_promoted_model_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        "--best-checkpoint",
        dest="source_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        "--destination",
        dest="output_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--split-version", required=True)
    args = parser.parse_args()
    manifest: PromotionBundleManifest = export_promoted_model_bundle(
        args.source_root,
        args.output_root,
        task=cast(
            Literal[
                "RUL",
                "SOH",
                "CYCLE_LIFE",
                "FIELD_MONITORING",
                "CONDITION_DEGRADATION",
                "PARTIAL_CHARGE_FEATURE",
            ],
            args.task,
        ),
        family=args.family,
        version=args.version,
        source_commit=args.source_commit,
        data_version=args.data_version,
        split_version=args.split_version,
    )
    print(
        json.dumps(
            {
                "status": manifest.activation_status,
                "bundle_sha256": manifest.bundle_sha256,
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
