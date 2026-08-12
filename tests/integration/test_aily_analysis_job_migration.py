from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_0021_adds_only_aily_job_origin_and_idempotency_metadata() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0021_aily_analysis_job_origins.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "0021"' in migration
    assert 'down_revision: str | None = "0020"' in migration
    assert '"job_origin"' in migration
    assert '"job_request_sha256"' in migration
    assert "FEISHU" in migration
    assert "AILY" in migration
    assert "trajectory" not in migration.casefold()
    assert "soh" not in migration.casefold()
