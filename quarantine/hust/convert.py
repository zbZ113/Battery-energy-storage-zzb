"""Fail-closed HUST conversion gate pending an approved semantic layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.layout.read_text(encoding="utf-8"))
    if payload.get("review_status") != "APPROVED":
        print(
            json.dumps(
                {
                    "layout_version": payload.get("layout_version"),
                    "status": "BLOCKED_REVIEW",
                    "unresolved_reasons": payload.get("unresolved_reasons", []),
                },
                sort_keys=True,
            )
        )
        return 42
    print(
        json.dumps(
            {
                "layout_version": payload.get("layout_version"),
                "status": "BLOCKED_CONVERTER_NOT_REVIEWED",
            },
            sort_keys=True,
        )
    )
    return 43


if __name__ == "__main__":
    raise SystemExit(main())
