"""Register one inactive Advanced deployment bundle in a managed local registry."""

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

from quanxin_life.application.advanced_deployment_registry import (  # noqa: E402
    AdvancedDeploymentBundleRegistry,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()

    record = AdvancedDeploymentBundleRegistry(args.registry_root).register(
        args.bundle_root,
        expected_manifest_sha256=args.expected_manifest_sha256,
        registered_at=datetime.now(UTC),
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
