"""Synchronize the battery-model research mirror from reviewed public sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

SourceKind = Literal["git", "pdf", "archive", "manual"]
ALLOWED_KINDS = frozenset({"git", "pdf", "archive", "manual"})
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ResearchSource:
    slug: str
    kind: SourceKind
    title: str
    url: str
    destination: str | None
    rank: int | None
    notes: str | None


def load_catalog(path: Path) -> tuple[ResearchSource, ...]:
    """Load a strict source catalog and reject ambiguous local destinations."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "battery-research-v1":
        raise ValueError("unsupported research catalog schema")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")

    sources: list[ResearchSource] = []
    slugs: set[str] = set()
    destinations: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError("each source must be an object")
        slug = _required_text(raw, "slug")
        if slug in slugs:
            raise ValueError(f"duplicate slug: {slug}")
        slugs.add(slug)
        kind = _required_text(raw, "kind")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported source kind: {kind}")
        destination_value = raw.get("destination")
        destination = None
        if destination_value is not None:
            if not isinstance(destination_value, str):
                raise ValueError("destination must be text")
            destination = _safe_destination(destination_value)
            if destination in destinations:
                raise ValueError(f"duplicate destination: {destination}")
            destinations.add(destination)
        if kind != "manual" and destination is None:
            raise ValueError("non-manual source requires a destination")
        rank_value = raw.get("rank")
        if rank_value is not None and (not isinstance(rank_value, int) or rank_value < 1):
            raise ValueError("rank must be a positive integer")
        notes_value = raw.get("notes")
        if notes_value is not None and not isinstance(notes_value, str):
            raise ValueError("notes must be text")
        sources.append(
            ResearchSource(
                slug=slug,
                kind=kind,  # type: ignore[arg-type]
                title=_required_text(raw, "title", default=slug),
                url=_required_text(raw, "url"),
                destination=destination,
                rank=rank_value,
                notes=notes_value,
            )
        )
    return tuple(sources)


def build_clone_command(url: str, destination: Path) -> tuple[str, ...]:
    """Return a deliberately complete clone command with no history filters."""

    return ("git", "clone", url, str(destination))


def validate_pdf_bytes(payload: bytes) -> None:
    """Reject HTML/login/error bodies saved under a PDF filename."""

    if not payload.startswith(b"%PDF-"):
        raise ValueError("downloaded payload is not a PDF")


def synchronize(
    sources: tuple[ResearchSource, ...],
    *,
    research_root: Path,
    selected_slugs: frozenset[str] | None = None,
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    research_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for source in sources:
        if selected_slugs is not None and source.slug not in selected_slugs:
            continue
        try:
            result = _sync_one(source, research_root)
        except Exception as exc:
            result = _base_result(source, status="FAILED")
            result["error"] = f"{type(exc).__name__}: {exc}"
            results.append(result)
            _write_state(research_root, results)
            if not continue_on_error:
                raise
        else:
            results.append(result)
            _write_state(research_root, results)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return results


def _sync_one(source: ResearchSource, research_root: Path) -> dict[str, Any]:
    if source.kind == "manual":
        result = _base_result(source, status="MANUAL_REQUIRED")
        result["notes"] = source.notes
        return result
    assert source.destination is not None
    destination = research_root / source.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.kind == "git":
        return _sync_git(source, destination)
    return _sync_file(source, destination)


def _sync_git(source: ResearchSource, destination: Path) -> dict[str, Any]:
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise ValueError(f"existing destination is not a Git repository: {destination}")
        remote = _run_git(destination, "remote", "get-url", "origin").strip()
        if _normalize_git_url(remote) != _normalize_git_url(source.url):
            raise ValueError(f"origin does not match catalog for {source.slug}")
        _run_git(destination, "fetch", "--all", "--tags", "--prune")
        action = "updated"
    else:
        subprocess.run(build_clone_command(source.url, destination), check=True)
        action = "cloned"
    _run_git(destination, "fsck", "--full")
    result = _base_result(source, status="READY")
    result.update(
        {
            "action": action,
            "head": _run_git(destination, "rev-parse", "HEAD").strip(),
            "default_branch": _run_git(destination, "branch", "--show-current").strip(),
            "commit_count": int(_run_git(destination, "rev-list", "--all", "--count").strip()),
            "tag_count": len(_nonempty_lines(_run_git(destination, "tag", "--list"))),
            "remote_branch_count": len(
                [
                    line
                    for line in _nonempty_lines(_run_git(destination, "branch", "--remotes"))
                    if not line.endswith("/HEAD -> origin/main")
                    and not line.endswith("/HEAD -> origin/master")
                ]
            ),
        }
    )
    return result


def _sync_file(source: ResearchSource, destination: Path) -> dict[str, Any]:
    if not destination.exists():
        _download_atomic(source.url, destination, kind=source.kind)
        action = "downloaded"
    else:
        action = "verified-existing"
    if source.kind == "pdf":
        with destination.open("rb") as handle:
            validate_pdf_bytes(handle.read(8))
    result = _base_result(source, status="READY")
    result.update(
        {
            "action": action,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256_file(destination),
        }
    )
    return result


def _download_atomic(url: str, destination: Path, *, kind: SourceKind) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "QuanxinLifeResearchMirror/1.0 (+academic study)"},
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
            request,
            timeout=120,
        ) as response:
            while chunk := response.read(CHUNK_SIZE):
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        publish_downloaded_file(temporary, destination, kind=kind)
    finally:
        temporary.unlink(missing_ok=True)


def publish_downloaded_file(
    temporary: Path,
    destination: Path,
    *,
    kind: SourceKind,
) -> None:
    """Validate downloaded bytes before atomically publishing the target path."""

    if temporary.stat().st_size == 0:
        raise ValueError("downloaded file is empty")
    if kind == "pdf":
        with temporary.open("rb") as handle:
            validate_pdf_bytes(handle.read(8))
    temporary.replace(destination)


def _write_state(research_root: Path, results: list[dict[str, Any]]) -> None:
    state_directory = research_root / ".state"
    state_directory.mkdir(parents=True, exist_ok=True)
    target = state_directory / "sync-status.json"
    payload = {
        "schema_version": "battery-research-sync-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "results": results,
    }
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    ).stdout


def _base_result(source: ResearchSource, *, status: str) -> dict[str, Any]:
    return {
        "slug": source.slug,
        "kind": source.kind,
        "title": source.title,
        "url": source.url,
        "destination": source.destination,
        "rank": source.rank,
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
    }


def _safe_destination(value: str) -> str:
    normalized = value.replace("\\", "/")
    posix_path = PurePosixPath(normalized)
    windows_path = PureWindowsPath(normalized)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError("destination must remain inside the research root")
    return posix_path.as_posix()


def _required_text(raw: dict[str, Any], key: str, *, default: str | None = None) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be nonblank text")
    return value.strip()


def _normalize_git_url(url: str) -> str:
    return url.removesuffix(".git").rstrip("/").lower()


def _nonempty_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("research/catalog.json"))
    parser.add_argument("--research-root", type=Path, default=Path("research"))
    parser.add_argument("--slug", action="append", default=[])
    parser.add_argument("--continue-on-error", action="store_true")
    arguments = parser.parse_args()
    sources = load_catalog(arguments.catalog)
    synchronize(
        sources,
        research_root=arguments.research_root,
        selected_slugs=frozenset(arguments.slug) if arguments.slug else None,
        continue_on_error=arguments.continue_on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
