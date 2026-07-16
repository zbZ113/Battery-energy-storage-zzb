"""Verify versioned public-dataset file sizes and SHA-256 values without parsing payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quanxin_life.data.manifest import (
    load_dataset_file_audit_manifest,
    verify_audited_dataset_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = load_dataset_file_audit_manifest(arguments.manifest)
    digests = verify_audited_dataset_files(arguments.repository_root, manifest)
    print(
        json.dumps(
            {
                "manifest_version": manifest.manifest_version,
                "verified_file_count": len(digests),
                "verified_total_size_bytes": manifest.total_size_bytes,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
