"""Prepare the verified 280 Ah capacity bundle for BLAST reference checking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quanxin_life.experiments.blast_280ah_data import (
    prepare_blast_280ah_validation_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing configs/ and data/raw/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/LFP_280AH_DOD/Zenodo-14576042-v3/capacity-v1"),
        help="Immutable output bundle directory.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repository_root / output_dir
    result = prepare_blast_280ah_validation_bundle(
        repository_root,
        output_dir=output_dir,
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
