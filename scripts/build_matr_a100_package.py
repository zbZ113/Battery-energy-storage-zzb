"""Build the clean, hash-bound MATR A100 ZIP64 training package."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.training.a100_package import (  # noqa: E402
    build_matr_a100_archive,
    build_matr_a100_archive_index,
    filter_a100_source_paths,
    verify_matr_a100_archive,
    verify_matr_a100_archive_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_archive", type=Path)
    parser.add_argument("--output-index", type=Path)
    args = parser.parse_args()
    _require_clean_git(REPO_ROOT)
    tracked = filter_a100_source_paths(_tracked_files(REPO_ROOT))
    source_commit = _git_output(REPO_ROOT, ["git", "rev-parse", "HEAD"])
    output = args.output_archive.resolve(strict=False)
    index_path = (
        args.output_index.resolve(strict=False)
        if args.output_index is not None
        else output.with_suffix(".sha256.json")
    )
    if index_path.exists() or index_path.is_symlink():
        parser.error("output index must be a new path")
    manifest = build_matr_a100_archive(
        project_root=REPO_ROOT,
        output_archive=output,
        tracked_files=tracked,
        source_commit=source_commit,
        raw_relative_path="data/2018-04-12_batchdata_updated_struct_errorcorrect.mat",
        raw_manifest_relative_path=(
            "configs/data_manifests/matr_2018_04_12_batch_v1.json"
        ),
        processed_relative_paths=(
            "data/processed/MATR/2018-04-12-cutoff150",
            "data/processed/MATR/2018-04-12-supervision500",
        ),
        created_at=datetime.now(UTC),
    )
    verified = verify_matr_a100_archive(output)
    if verified != manifest:
        raise ValueError("new A100 package failed immediate internal verification")
    index = build_matr_a100_archive_index(output, manifest)
    verify_matr_a100_archive_index(output, index)
    _write_json_atomic(index_path, index.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "status": "MATR_A100_PACKAGE_READY",
                "archive": str(output),
                "index": str(index_path),
                "archive_sha256": index.archive_sha256,
                "package_sha256": index.package_sha256,
                "size_bytes": index.size_bytes,
                "source_commit": index.source_commit,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _require_clean_git(root: Path) -> None:
    status = _git_output(root, ["git", "status", "--porcelain"])
    if status:
        raise ValueError("A100 package requires a clean Git worktree")


def _tracked_files(root: Path) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    paths = tuple(
        item.decode("utf-8") for item in result.split(b"\0") if item
    )
    if not paths:
        raise ValueError("Git tracked-file inventory is empty")
    return paths


def _git_output(root: Path, command: list[str]) -> str:
    return subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
