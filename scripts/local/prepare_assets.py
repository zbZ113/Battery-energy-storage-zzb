"""Verify and install the reviewed Runtime V7 assets for local use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "server-results" / "deployment-handoff"
DEMO_ARCHIVE = HANDOFF / "quanxin-competition-demo-ready-2026.07.28-1.tgz"
PROVENANCE_PACKAGE = HANDOFF / "deployment-registry-provenance-v2"
RUNTIME_DELTA = HANDOFF / "deployment-registry-runtime-v2-delta"
OLD_REGISTRY_ID = "2d9c882d293e9cd25ce83bc6c4c6c3f78a567e1a285cf8865ff411be8898a9b4"
REGISTRY_ID = "c31f62e68faa66e56b16d21ebdd3067d5dea0c8408bb3ad6baa73e05a42824be"
CUTOFFS = (20, 50, 100, 150)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
        or "\\" in value
    ):
        raise ValueError("manifest entry must use a bounded relative path")
    return Path(*pure.parts)


def verify_manifest(root: Path, manifest: Path) -> None:
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("manifest must be a regular file")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError("manifest line is invalid")
        expected, raw_relative = pieces
        relative = _relative_path(raw_relative.strip())
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"manifest file is missing: {relative.as_posix()}")
        if sha256_file(candidate) != expected.lower():
            raise ValueError(
                f"manifest SHA-256 mismatch: {relative.as_posix()}"
            )


def verify_archive_sidecar(archive: Path) -> None:
    sidecar = archive.with_name(archive.name + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("archive SHA-256 sidecar is missing")
    line = sidecar.read_text(encoding="utf-8").strip()
    pieces = line.split(maxsplit=1)
    if len(pieces) != 2 or pieces[1].strip() != archive.name:
        raise ValueError("archive SHA-256 sidecar is invalid")
    if sha256_file(archive) != pieces[0].lower():
        raise ValueError("archive SHA-256 mismatch")


def _install_bytes(destination: Path, payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"asset destination is not a regular file: {destination}")
        if sha256_file(destination) != digest:
            raise ValueError(f"existing asset differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def _install_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"asset source is not a regular file: {source}")
    if destination.exists():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != source.stat().st_size
            or sha256_file(destination) != sha256_file(source)
        ):
            raise ValueError(f"existing asset differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def install_demo_package(
    archive: Path,
    *,
    target: Path,
    release_root: str = "2026.07.28-1",
) -> None:
    destinations = {
        "calibration-evidence": target / "calibration",
        "config": target / "config",
        "demo-target": target / "demo-target",
    }
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or any(part in {"", ".", ".."} or ":" in part for part in pure.parts)
                or member.issym()
                or member.islnk()
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
            if len(pure.parts) < 2 or pure.parts[0] != release_root:
                continue
            section = pure.parts[1]
            destination_root = destinations.get(section)
            if destination_root is None:
                continue
            relative_parts = pure.parts[2:]
            destination = destination_root.joinpath(*relative_parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"unsupported archive member: {member.name}")
            stream = bundle.extractfile(member)
            if stream is None:
                raise ValueError(f"archive member cannot be read: {member.name}")
            _install_bytes(destination, stream.read())


def verify_demo_registration(csv_path: Path, registration_path: Path) -> None:
    try:
        registration = json.loads(registration_path.read_text(encoding="utf-8"))
        expected = registration["metadata"]["source_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("demo registration is invalid") from exc
    if expected != sha256_file(csv_path):
        raise ValueError("registration source SHA-256 does not match observed CSV")


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON asset is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON asset must be an object: {path}")
    return payload


def install_registry(target: Path) -> None:
    verify_manifest(
        PROVENANCE_PACKAGE,
        PROVENANCE_PACKAGE / "MANIFEST.sha256",
    )
    verify_manifest(RUNTIME_DELTA, RUNTIME_DELTA / "MANIFEST.sha256")

    delta = _json_object(RUNTIME_DELTA / "DELTA.json")
    mappings = delta.get("mappings")
    if not isinstance(mappings, list) or len(mappings) != 15:
        raise ValueError("runtime delta must contain 15 artifact mappings")

    payload_root = RUNTIME_DELTA / "payload"
    index = _json_object(payload_root / "deployment_bundle_index.json")
    record = _json_object(payload_root / "registry-record.json")
    if (
        index.get("schema_version") != "advanced-deployment-bundle-index-v2"
        or index.get("manifest_sha256") != REGISTRY_ID
        or record.get("registry_id") != REGISTRY_ID
    ):
        raise ValueError("runtime v2 registry identity is invalid")

    registry_root = target / "registry"
    bundle_root = registry_root / "bundles" / REGISTRY_ID[:16]
    for source in sorted(payload_root.rglob("*")):
        if not source.is_file() or source.name == "registry-record.json":
            continue
        relative = source.relative_to(payload_root)
        _install_file(source, bundle_root / relative)
    _install_file(
        payload_root / "registry-record.json",
        registry_root / "records" / f"{REGISTRY_ID}.json",
    )

    old_bundle = (
        PROVENANCE_PACKAGE
        / "registry"
        / "bundles"
        / OLD_REGISTRY_ID[:16]
        / "artifacts"
    )
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError("runtime delta mapping is invalid")
        old_relative = _relative_path(str(mapping["old_weight_relative_path"]))
        new_relative = _relative_path(str(mapping["new_weight_relative_path"]))
        source = old_bundle / old_relative
        destination = bundle_root / "artifacts" / new_relative
        expected_hash = str(mapping["weight_sha256"])
        expected_size = int(mapping["weight_size_bytes"])
        if (
            source.stat().st_size != expected_size
            or sha256_file(source) != expected_hash
        ):
            raise ValueError("runtime delta model.safetensors source is invalid")
        if destination.exists():
            if (
                destination.stat().st_size != expected_size
                or sha256_file(destination) != expected_hash
            ):
                raise ValueError("installed model.safetensors differs")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if (
            destination.stat().st_size != expected_size
            or sha256_file(destination) != expected_hash
        ):
            raise ValueError("installed model.safetensors failed verification")


def install_policy(target: Path) -> None:
    policy = (
        b'{"schema_version":"advanced-agent-policy-v1",'
        b'"conformal_alpha":0.1}\n'
    )
    _install_bytes(target / "policies" / "advanced-agent.json", policy)


def prepare(target: Path) -> None:
    verify_archive_sidecar(DEMO_ARCHIVE)
    install_demo_package(DEMO_ARCHIVE, target=target)
    for cutoff in CUTOFFS:
        verify_demo_registration(
            target / "demo-target" / f"MATR_b3c34-cutoff-{cutoff}.csv",
            target
            / "demo-target"
            / f"MATR_b3c34-cutoff-{cutoff}.registration.json",
        )
    install_registry(target)
    install_policy(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        with tempfile.TemporaryDirectory(
            prefix="quanxin-assets-",
            dir=target.parent,
        ) as temporary:
            prepare(Path(temporary))
    else:
        prepare(target)
    print(f"registry_id={REGISTRY_ID}")
    print("model_artifacts=15")
    print("demo_cutoffs=20,50,100,150")
    print(f"dry_run={str(args.dry_run).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
