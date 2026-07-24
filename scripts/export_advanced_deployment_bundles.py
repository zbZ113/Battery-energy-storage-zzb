#!/usr/bin/env python3
"""Export inactive, SHA-verified deployment bundles from Advanced Final."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quanxin_life.application.advanced_deployment_bundles import (  # noqa: E402
    export_advanced_deployment_bundles,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export representative Advanced Final deployment bundles."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--promotion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-promotion-manifest-sha256", required=True)
    parser.add_argument("--resume-artifact-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    index = export_advanced_deployment_bundles(
        args.project_root,
        args.result_root,
        args.promotion_root,
        args.output_root,
        expected_promotion_manifest_sha256=(
            args.expected_promotion_manifest_sha256
        ),
        resume_artifact_root=args.resume_artifact_root,
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED_NOT_ACTIVATED",
                "routes": len(index.routes),
                "artifacts": len(index.artifacts),
                "manifest_sha256": index.manifest_sha256,
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
