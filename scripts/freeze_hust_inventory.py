"""Freeze an initial HUST opaque-byte audit into reviewed inventory evidence."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from quanxin_life.data.hust_archive import (
    freeze_hust_inventory,
    load_hust_archive_audit,
)


def _json_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.casefold() != ".json":
        raise argparse.ArgumentTypeError("audit and output paths must end in .json")
    return path


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("HUST inventory output cannot be a symbolic link")
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
    parser.add_argument("--audit", type=_json_path, required=True)
    parser.add_argument("--inventory-version", required=True)
    parser.add_argument("--output", type=_json_path, required=True)
    arguments = parser.parse_args()

    if arguments.output.resolve(strict=False) == arguments.audit.resolve(strict=False):
        raise ValueError("HUST inventory output must differ from the audit input")
    audit = load_hust_archive_audit(arguments.audit)
    inventory = freeze_hust_inventory(
        audit,
        inventory_version=arguments.inventory_version,
    )
    _write_atomic_json(arguments.output, inventory.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "archive_sha256": inventory.archive_sha256,
                "inventory_sha256": inventory.inventory_sha256,
                "inventory_version": inventory.inventory_version,
                "member_count": inventory.member_count,
                "output": str(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
