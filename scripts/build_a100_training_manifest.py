"""Build and immediately re-verify a safe A100 training bundle manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.bundle import (  # noqa: E402
    build_training_bundle_manifest,
    verify_training_bundle_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--split-version", required=True)
    args = parser.parse_args()

    bundle_root = args.bundle_root.resolve(strict=True)
    output = args.output_manifest.resolve(strict=False)
    if output.is_relative_to(bundle_root):
        parser.error("output_manifest must be outside bundle_root to avoid a circular manifest")
    manifest = build_training_bundle_manifest(
        bundle_root,
        created_at=datetime.now(UTC),
        data_version=args.data_version,
        split_version=args.split_version,
    )
    verify_training_bundle_manifest(bundle_root, manifest)
    payload = json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
