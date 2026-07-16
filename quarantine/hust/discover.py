"""Inspect one approved HUST pickle stream without executing pickle objects.

This single-purpose quarantine program performs only pickle opcode inspection.
It intentionally does not call pickle.load or pickle.loads and cannot emit
canonical battery records.  A later, separately reviewed converter is required
after field semantics and a digest-pinned image have been approved.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickletools
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_GLOBAL_REFERENCES = 4096
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError(f"invalid bounded JSON input: {path}")
    payload = json.loads(
        path.read_bytes(),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return payload


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 2
        or path.parts[0] != "our_data"
        or path.parts[1] in {"", ".", ".."}
        or path.suffix != ".pkl"
        or "\\" in value
    ):
        raise ValueError("job member_path must be our_data/<name>.pkl")
    return value


def _verified_inputs() -> tuple[Path, str, str]:
    archive = Path(os.environ["HUST_ARCHIVE"])
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("archive must be a regular non-symlinked file")
    manifest = _load_json(Path(os.environ["HUST_RAW_MANIFEST"]))
    inventory = _load_json(Path(os.environ["HUST_APPROVED_INVENTORY"]))
    audit = _load_json(Path(os.environ["HUST_APPROVED_AUDIT"]))
    job = _load_json(Path(os.environ["HUST_JOB"]))

    if inventory.get("review_status") != "APPROVED":
        raise ValueError("review_status != \"APPROVED\"")
    inventory_without_digest = dict(inventory)
    claimed_inventory_sha256 = inventory_without_digest.pop("inventory_sha256", None)
    actual_inventory_sha256 = _canonical_sha256(inventory_without_digest)
    if claimed_inventory_sha256 != actual_inventory_sha256:
        raise ValueError("approved inventory digest mismatch")
    confirm_inventory_sha256 = job.get("confirm_inventory_sha256")
    if confirm_inventory_sha256 != claimed_inventory_sha256:
        raise ValueError("job does not confirm approved inventory digest")
    if audit.get("ready_for_conversion") is not True:
        raise ValueError("approved archive audit is not ready_for_conversion")
    if audit.get("inventory_version") != inventory.get("inventory_version"):
        raise ValueError("audit and inventory versions differ")

    expected_archive_sha256 = inventory.get("archive_sha256")
    if manifest.get("sha256") != expected_archive_sha256:
        raise ValueError("raw manifest and inventory archive digests differ")
    if _sha256_file(archive) != expected_archive_sha256:
        raise ValueError("archive bytes do not match approved inventory")

    member_path = _safe_member_path(str(job.get("member_path", "")))
    expected_member = next(
        (item for item in inventory.get("members", []) if item.get("path") == member_path),
        None,
    )
    if expected_member is None:
        raise ValueError("job member is absent from approved inventory")
    member_sha256 = str(expected_member.get("sha256", ""))
    return archive, member_path, member_sha256


def _inspect_member(archive_path: Path, member_path: str, member_sha256: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path, "r") as archive:
        info = archive.getinfo(member_path)
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError("member exceeds opcode-inspection size limit")
        payload = archive.read(info)
    if len(payload) != info.file_size:
        raise ValueError("member size differs from ZIP metadata")
    if hashlib.sha256(payload).hexdigest() != member_sha256:
        raise ValueError("member bytes do not match approved inventory")

    opcode_counts: dict[str, int] = {}
    global_references: list[str] = []
    protocol = 0
    for opcode, argument, _position in pickletools.genops(payload):
        opcode_counts[opcode.name] = opcode_counts.get(opcode.name, 0) + 1
        if opcode.name == "PROTO" and isinstance(argument, int):
            protocol = max(protocol, argument)
        if opcode.name == "GLOBAL" and isinstance(argument, str):
            if len(global_references) >= MAX_GLOBAL_REFERENCES:
                raise ValueError("pickle global-reference count exceeds limit")
            global_references.append(argument)

    return {
        "mode": "OPCODE_INSPECTION_ONLY",
        "member_path": member_path,
        "member_sha256": member_sha256,
        "member_size_bytes": len(payload),
        "pickle_protocol": protocol,
        "opcode_counts": dict(sorted(opcode_counts.items())),
        "global_references": sorted(set(global_references)),
        "limitations": [
            "pickle objects were not executed or deserialized",
            "field names, units, battery semantics and cell identity remain unverified",
            "this report cannot be used as canonical training data",
        ],
    }


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink() or path.exists():
        raise ValueError("quarantine output path must be new and non-symlinked")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    if len(encoded) > MAX_OUTPUT_BYTES:
        raise ValueError("opcode report exceeds output limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=".hust-", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    archive, member_path, member_sha256 = _verified_inputs()
    report = _inspect_member(archive, member_path, member_sha256)
    _write_output(Path(os.environ["HUST_OUTPUT"]), report)
    print(json.dumps({"status": "OPCODE_INSPECTION_COMPLETE", "member": member_path}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
