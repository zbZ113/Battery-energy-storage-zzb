from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_NAME = "uq_feishu_receipts_proactive_source_task"


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _insert_job(
    connection: object,
    *,
    suffix: str,
    event_type: str,
    origin: str,
    task_type: str = "make_engineering_recommendation",
) -> None:
    connection.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO feishu_event_receipts "
            "(id,event_id,event_type,payload_sha256,status,attempt_count,received_at,"
            "job_id,job_origin,job_request_sha256,source_job_id,job_status,job_stage,"
            "task_type,run_id,job_attempt_count) VALUES "
            "(:id,:event_id,:event_type,:payload_sha,'PROCESSED',1,CURRENT_TIMESTAMP,"
            ":job_id,:origin,:request_sha,'root-job','PENDING','RECEIVED',:task_type,"
            ":run_id,0)"
        ),
        {
            "id": f"receipt-{suffix}",
            "event_id": f"event-{suffix}",
            "event_type": event_type,
            "payload_sha": sha256(f"payload:{suffix}".encode()).hexdigest(),
            "job_id": f"job-{suffix}",
            "origin": origin,
            "request_sha": sha256(f"request:{suffix}".encode()).hexdigest(),
            "task_type": task_type,
            "run_id": f"run-{suffix}",
        },
    )


def test_0029_enforces_one_proactive_task_per_root_without_blocking_aily(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'proactive-uniqueness.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0028")
    engine = create_engine(database_url)
    try:
        command.upgrade(config, "0029")
        indexes = {
            item["name"]: item
            for item in inspect(engine).get_indexes("feishu_event_receipts")
        }
        assert indexes[INDEX_NAME]["unique"] == 1

        with engine.begin() as connection:
            _insert_job(
                connection,
                suffix="derived-a",
                event_type="feishu.analysis_job.derived_v1",
                origin="FEISHU",
            )
            _insert_job(
                connection,
                suffix="aily-a",
                event_type="aily.analysis_task.create_v2",
                origin="AILY",
            )
            _insert_job(
                connection,
                suffix="derived-soh",
                event_type="feishu.analysis_job.derived_v1",
                origin="FEISHU",
                task_type="predict_soh_trajectory",
            )

        with engine.begin() as connection, pytest.raises(IntegrityError):
            _insert_job(
                connection,
                suffix="derived-b",
                event_type="feishu.analysis_job.derived_v1",
                origin="FEISHU",
            )
    finally:
        engine.dispose()
