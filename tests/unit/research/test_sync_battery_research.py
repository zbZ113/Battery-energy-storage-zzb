from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.sync_battery_research import (
    build_clone_command,
    load_catalog,
    publish_downloaded_file,
    validate_pdf_bytes,
)


def _write_catalog(path: Path, sources: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"schema_version": "battery-research-v1", "sources": sources}))


def test_catalog_rejects_duplicate_slugs(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    source = {
        "slug": "model-a",
        "kind": "git",
        "url": "https://github.com/example/model-a.git",
        "destination": "upstream/model-a",
    }
    _write_catalog(catalog, [source, source])

    with pytest.raises(ValueError, match="duplicate slug"):
        load_catalog(catalog)


@pytest.mark.parametrize("destination", ["../escape", "/absolute", "C:/absolute"])
def test_catalog_rejects_destination_outside_research_root(
    tmp_path: Path,
    destination: str,
) -> None:
    catalog = tmp_path / "catalog.json"
    _write_catalog(
        catalog,
        [
            {
                "slug": "model-a",
                "kind": "git",
                "url": "https://github.com/example/model-a.git",
                "destination": destination,
            }
        ],
    )

    with pytest.raises(ValueError, match="destination"):
        load_catalog(catalog)


def test_clone_command_keeps_complete_history() -> None:
    command = build_clone_command(
        "https://github.com/example/model-a.git",
        Path("research/upstream/model-a"),
    )

    assert command[:2] == ("git", "clone")
    assert "--depth" not in command
    assert "--filter" not in command
    assert "--single-branch" not in command


def test_pdf_validator_rejects_html_error_page() -> None:
    with pytest.raises(ValueError, match="PDF"):
        validate_pdf_bytes(b"<html>access denied</html>")


def test_invalid_pdf_is_not_published(tmp_path: Path) -> None:
    temporary = tmp_path / "payload.tmp"
    destination = tmp_path / "paper.pdf"
    temporary.write_bytes(b"<html>access denied</html>")

    with pytest.raises(ValueError, match="PDF"):
        publish_downloaded_file(temporary, destination, kind="pdf")

    assert not destination.exists()
    assert temporary.exists()
