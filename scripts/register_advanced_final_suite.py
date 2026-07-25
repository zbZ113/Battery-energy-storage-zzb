"""Register the exact external Advanced Final suite without copying its 15 GB tree."""

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

from quanxin_life.application.advanced_final_suite_import import (  # noqa: E402
    AdvancedFinalSuiteImporter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("transfer_archive", type=Path)
    parser.add_argument("registry_root", type=Path)
    parser.add_argument("--expected-output-sha256", required=True)
    parser.add_argument("--expected-output-index-file-sha256", required=True)
    parser.add_argument("--transfer-sha256", required=True)
    parser.add_argument("--local-input-bundle-sha256", required=True)
    args = parser.parse_args()

    importer = AdvancedFinalSuiteImporter(
        args.registry_root,
        evidence_root=args.evidence_root,
    )
    record = importer.register(
        args.output_root,
        expected_output_sha256=args.expected_output_sha256,
        expected_output_index_file_sha256=(
            args.expected_output_index_file_sha256
        ),
        transfer_archive=args.transfer_archive,
        transfer_sha256=args.transfer_sha256,
        local_reconstructed_input_bundle_sha256=(
            args.local_input_bundle_sha256
        ),
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
