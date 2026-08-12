"""Run the fixed BLAST 250 Ah reference model over observed 280 Ah cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from quanxin_life.experiments.blast_280ah_validation import (
    run_lfp_280ah_reference_validation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the pinned BLAST source and route manifest.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("data/processed/LFP_280AH_DOD/Zenodo-14576042-v3/capacity-v1"),
        help="Verified 280 Ah capacity bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/blast_280ah_v1/validation-v1"),
        help="Immutable observed-range validation result directory.",
    )
    return parser


def _source_revision(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    head = completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN_HEAD"
    files = [
        repository_root
        / "src"
        / "quanxin_life"
        / "experiments"
        / "blast_280ah_validation.py",
        repository_root
        / "src"
        / "quanxin_life"
        / "scenarios"
        / "manifests"
        / "blast_lite_routes_v1.json",
        *sorted(
            (
                repository_root / "src" / "quanxin_life" / "_vendor" / "blast_lite"
            ).glob("*.py")
        ),
    ]
    digest = hashlib.sha256()
    for path in files:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return f"{head}+blast-280ah-source-sha256:{digest.hexdigest()}"


def main() -> int:
    args = _parser().parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    bundle_dir = args.bundle_dir
    if not bundle_dir.is_absolute():
        bundle_dir = repository_root / bundle_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    result = run_lfp_280ah_reference_validation(
        bundle_dir,
        output_dir=output_dir,
        code_revision=_source_revision(repository_root),
    )
    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
