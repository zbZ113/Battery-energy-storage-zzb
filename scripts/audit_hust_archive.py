"""Audit the official HUST ZIP as opaque bytes and write a JSON evidence file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from quanxin_life.data.hust_archive import audit_hust_archive
from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.source_catalog import SourceCatalog


def _json_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.casefold() != ".json":
        raise argparse.ArgumentTypeError("output and manifest paths must end in .json")
    return path


def _write_atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("HUST audit output cannot be a symbolic link")
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


def _reject_output_input_collision(output: Path, inputs: tuple[Path, ...]) -> None:
    resolved_output = output.resolve(strict=False)
    for input_path in inputs:
        if resolved_output == input_path.resolve(strict=False):
            raise ValueError("HUST audit output must differ from every input path")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=_json_path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=_json_path, required=True)
    arguments = parser.parse_args()

    _reject_output_input_collision(
        arguments.output,
        (arguments.archive, arguments.manifest, arguments.catalog),
    )

    manifest = RawFileManifest.model_validate_json(
        arguments.manifest.read_text(encoding="utf-8")
    )
    source = SourceCatalog.load(arguments.catalog).require("HUST")
    audit = audit_hust_archive(arguments.archive, manifest, source)
    payload = audit.model_dump(mode="json")
    _write_atomic_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "archive_sha256": audit.archive_sha256,
                "member_count": audit.member_count,
                "output": str(arguments.output),
                "ready_for_conversion": audit.ready_for_conversion,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
