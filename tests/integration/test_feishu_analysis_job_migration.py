from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_0017_extends_receipts_with_sanitized_analysis_job_state() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0017_feishu_analysis_jobs.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "0017"' in migration
    assert 'down_revision: str | None = "0016"' in migration
    for column in (
        "job_id",
        "job_status",
        "job_stage",
        "task_type",
        "run_id",
        "message_id",
        "file_key",
        "file_name",
        "chat_id",
        "sender_id",
        "receive_id_type",
        "event_time",
        "job_claim_token",
        "job_attempt_count",
        "job_lease_expires_at",
        "job_task_id",
        "job_last_error_code",
        "record_batch_id",
        "cell_reference",
        "input_file_sha256",
        "validation_result_id",
        "analysis_result_id",
        "report_result_id",
        "report_file_key",
        "bitable_record_id",
        "job_created_at",
        "job_updated_at",
        "job_completed_at",
    ):
        assert f'"{column}"' in migration


def test_0017_does_not_create_a_second_job_or_result_table() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0017_feishu_analysis_jobs.py"
    ).read_text(encoding="utf-8")

    assert "create_table" not in migration
    assert "tool_results" not in migration
