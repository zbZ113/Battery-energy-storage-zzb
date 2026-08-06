"""Audit frozen training upstream commits and root license evidence without network I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_COMMITS = {
    "pbt": "a2df9d36db3f57ab2c5686638952ad7715ea0646",
    "diting": "b67f48373c591ca62c030fca257ee94910c974ce",
    "batterymformer": "febe174032ad4861fa057b9af23f5bcee8a8fb77",
    "magnet": "aafb90c551d20748251a35fd51a34eae2539aaca",
    "battgp": "6d5e1db3337f0de5f3a533acbc09eddccc178e9d",
    "blast": "b093495b47dc40dd96dba865d91f553619501e94",
    "smart_feature": "dc6beea547960cf44d1721734dba93bdcab19a1f",
}
_DIRECTORIES = {
    "battgp": "BattGP",
    "batterymformer": "BatteryMFormer",
    "blast": "BLAST-Lite",
    "diting": "DITING",
    "magnet": "MAGNet",
    "pbt": "PBT",
    "smart_feature": "Smart-Feature-Identification",
}


class UpstreamAuditRecord(TypedDict):
    schema_version: str
    model_family: str
    relative_path: str
    commit: str
    license_status: str
    license_file: str | None
    license_sha256: str | None


def audit_training_upstreams(
    repository_root: Path,
    *,
    expected_commits: dict[str, str] | None = None,
) -> tuple[UpstreamAuditRecord, ...]:
    root = Path(repository_root).resolve(strict=True)
    expected = expected_commits or _DEFAULT_COMMITS
    if set(expected) != set(_DIRECTORIES):
        raise ValueError("training upstream commit registry must contain every reviewed adapter")
    records = tuple(
        _audit_repository(
            root=root,
            model_family=model_family,
            directory_name=_DIRECTORIES[model_family],
            expected_commit=expected[model_family],
        )
        for model_family in sorted(_DIRECTORIES)
    )
    return records


def _audit_repository(
    *,
    root: Path,
    model_family: str,
    directory_name: str,
    expected_commit: str,
) -> UpstreamAuditRecord:
    if not _is_sha1(expected_commit):
        raise ValueError("expected upstream commit must be a lowercase Git SHA-1")
    upstream_root = (root / "research" / "upstream").resolve(strict=True)
    repository = (upstream_root / directory_name).resolve(strict=True)
    if not repository.is_relative_to(upstream_root) or repository.is_symlink():
        raise ValueError("training upstream repository escapes the reviewed root")
    actual_commit = _git_head(repository)
    if actual_commit != expected_commit:
        raise ValueError(
            f"{model_family} upstream commit drift: expected {expected_commit}, "
            f"found {actual_commit}"
        )
    license_paths = tuple(
        path
        for path in sorted(repository.iterdir())
        if path.is_file() and path.name.upper().startswith(("LICENSE", "COPYING"))
    )
    if len(license_paths) > 1:
        raise ValueError(f"{model_family} has ambiguous root license files")
    license_path = license_paths[0] if license_paths else None
    if license_path is None:
        status = "RESEARCH_ONLY_LICENSE_UNVERIFIED"
        license_file = None
        license_sha256 = None
    else:
        status = "VERIFIED_LICENSE_PRESENT"
        license_file = license_path.name
        license_sha256 = _sha256_file(license_path)
    return UpstreamAuditRecord(
        schema_version="training-upstream-audit-v1",
        model_family=model_family,
        relative_path=repository.relative_to(root).as_posix(),
        commit=actual_commit,
        license_status=status,
        license_file=license_file,
        license_sha256=license_sha256,
    )


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if not _is_sha1(value):
        raise ValueError("training upstream HEAD is not a lowercase Git SHA-1")
    return value


def _is_sha1(value: str) -> bool:
    return len(value) == 40 and value == value.lower() and all(
        character in "0123456789abcdef" for character in value
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    for record in audit_training_upstreams(args.project_root):
        print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
