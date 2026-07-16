"""Record explicit human approval for a candidate frozen HUST inventory."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from quanxin_life.data.hust_archive import (
    approve_hust_inventory,
    load_hust_frozen_inventory,
)


def _json_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.casefold() != ".json":
        raise argparse.ArgumentTypeError("candidate and output paths must end in .json")
    return path


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("HUST approved inventory output cannot be a symbolic link")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=_json_path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--confirm-inventory-sha256", required=True)
    parser.add_argument("--output", type=_json_path, required=True)
    arguments = parser.parse_args()

    if arguments.output.resolve(strict=False) == arguments.candidate.resolve(strict=False):
        raise ValueError("approved inventory output must differ from candidate input")
    candidate = load_hust_frozen_inventory(arguments.candidate)
    if arguments.confirm_inventory_sha256 != candidate.inventory_sha256:
        raise ValueError("confirmed inventory SHA-256 does not match candidate")
    approved = approve_hust_inventory(
        candidate,
        approved_by=arguments.approved_by,
        approved_at=datetime.now(UTC),
    )
    _write_atomic_json(arguments.output, approved.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "approved_at": approved.approved_at.isoformat()
                if approved.approved_at is not None
                else None,
                "approved_by": approved.approved_by,
                "inventory_sha256": approved.inventory_sha256,
                "inventory_version": approved.inventory_version,
                "output": str(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
