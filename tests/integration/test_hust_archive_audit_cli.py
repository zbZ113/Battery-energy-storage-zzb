from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def _write_archive(path: Path) -> bytes:
    payload = b"opaque pickle bytes; never deserialize"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("our_data/", b"")
        archive.writestr("our_data/1-1.pkl", payload)
    return payload


def _write_manifest(path: Path, archive: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "dataset_id": "HUST",
                "relative_path": archive.name,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "source_uri": "https://data.mendeley.com/datasets/nsc7hnsg4s/2",
                "license_name": "CC BY 4.0",
                "license_uri": "https://creativecommons.org/licenses/by/4.0/",
                "paper_doi": "10.1039/D2EE01676A",
                "downloaded_at": datetime(2026, 7, 13, tzinfo=UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def _run_script(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd() / "src")
    return subprocess.run(
        [sys.executable, script, *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return _run_script("scripts/audit_hust_archive.py", *arguments)


def test_hust_archive_audit_cli_writes_traceable_json_without_extracting(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "our_data.zip"
    member_payload = _write_archive(archive)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, archive)
    output = tmp_path / "audit.json"

    completed = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    audit = json.loads(output.read_text(encoding="utf-8"))
    assert audit["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert audit["member_count"] == 1
    assert audit["members"][0]["sha256"] == hashlib.sha256(member_payload).hexdigest()
    assert audit["ready_for_conversion"] is False
    assert not (tmp_path / "our_data").exists()


def test_hust_archive_cli_freezes_then_enforces_exact_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "our_data.zip"
    _write_archive(archive)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, archive)
    initial_audit = tmp_path / "audit-unfrozen.json"
    candidate = tmp_path / "inventory-candidate.json"
    inventory = tmp_path / "inventory-approved.json"
    verified_audit = tmp_path / "audit-frozen.json"

    first = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--output",
        str(initial_audit),
    )
    frozen = _run_script(
        "scripts/freeze_hust_inventory.py",
        "--audit",
        str(initial_audit),
        "--inventory-version",
        "hust-fixture-inventory-v1",
        "--output",
        str(candidate),
    )
    candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
    rejected_approval = _run_script(
        "scripts/approve_hust_inventory.py",
        "--candidate",
        str(candidate),
        "--approved-by",
        "integration-test-reviewer",
        "--confirm-inventory-sha256",
        "0" * 64,
        "--output",
        str(inventory),
    )
    approved = _run_script(
        "scripts/approve_hust_inventory.py",
        "--candidate",
        str(candidate),
        "--approved-by",
        "integration-test-reviewer",
        "--confirm-inventory-sha256",
        candidate_payload["inventory_sha256"],
        "--output",
        str(inventory),
    )
    second = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--inventory",
        str(inventory),
        "--output",
        str(verified_audit),
    )

    assert first.returncode == 0, first.stderr
    assert frozen.returncode == 0, frozen.stderr
    assert rejected_approval.returncode != 0
    assert "does not match" in rejected_approval.stderr
    assert approved.returncode == 0, approved.stderr
    assert second.returncode == 0, second.stderr
    inventory_payload = json.loads(inventory.read_text(encoding="utf-8"))
    audit_payload = json.loads(verified_audit.read_text(encoding="utf-8"))
    assert inventory_payload["inventory_version"] == "hust-fixture-inventory-v1"
    assert inventory_payload["review_status"] == "APPROVED"
    assert inventory_payload["member_count"] == 1
    assert audit_payload["ready_for_conversion"] is True
    assert audit_payload["inventory_version"] == "hust-fixture-inventory-v1"
    assert audit_payload["warnings"] == []
    assert not (tmp_path / "our_data").exists()


def test_hust_archive_audit_cli_refuses_non_json_output(tmp_path: Path) -> None:
    archive = tmp_path / "our_data.zip"
    _write_archive(archive)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, archive)

    completed = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--output",
        str(tmp_path / "audit.pkl"),
    )

    assert completed.returncode != 0
    assert ".json" in completed.stderr


def test_hust_archive_audit_cli_refuses_to_overwrite_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "our_data.zip"
    _write_archive(archive)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, archive)
    original_manifest = manifest.read_bytes()

    completed = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--output",
        str(manifest),
    )

    assert completed.returncode != 0
    assert "must differ" in completed.stderr
    assert manifest.read_bytes() == original_manifest


def test_hust_archive_audit_cli_keeps_existing_output_when_audit_fails(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "our_data.zip"
    _write_archive(archive)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, archive)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "audit.json"
    original_output = b'{"status":"reviewed-old-output"}\n'
    output.write_bytes(original_output)

    completed = _run_cli(
        "--archive",
        str(archive),
        "--manifest",
        str(manifest),
        "--catalog",
        "configs/data_sources.json",
        "--output",
        str(output),
    )

    assert completed.returncode != 0
    assert output.read_bytes() == original_output
