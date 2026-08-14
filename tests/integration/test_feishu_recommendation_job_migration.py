from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECOMMENDATION_COLUMNS = {
    "recommendation_ruleset_id",
    "recommendation_ruleset_version",
    "recommendation_ruleset_sha256",
    "recommendation_upstream_result_ids_json",
    "recommendation_upstream_result_ids_sha256",
}


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0028_adds_complete_recommendation_input_checkpoints(tmp_path: Path) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'recommendation-job.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0027")
    engine = create_engine(database_url)
    try:
        assert RECOMMENDATION_COLUMNS.isdisjoint(
            {column["name"] for column in inspect(engine).get_columns("feishu_event_receipts")}
        )

        command.upgrade(config, "0028")

        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert set(columns) >= RECOMMENDATION_COLUMNS
        assert columns["recommendation_ruleset_id"]["type"].length == 200
        assert columns["recommendation_ruleset_version"]["type"].length == 200
        assert columns["recommendation_ruleset_sha256"]["type"].length == 64
        assert columns["recommendation_upstream_result_ids_sha256"]["type"].length == 64

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,recommendation_ruleset_id) "
                    "VALUES ('partial-rules','partial-rules-event','type',:sha,"
                    "'PROCESSED',1,CURRENT_TIMESTAMP,'FEISHU',0,'rules-v1')"
                ),
                {"sha": "a" * 64},
            )

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,"
                    "recommendation_upstream_result_ids_json) VALUES "
                    "('partial-results','partial-results-event','type',:sha,"
                    "'PROCESSED',1,CURRENT_TIMESTAMP,'FEISHU',0,'[]')"
                ),
                {"sha": "b" * 64},
            )
    finally:
        engine.dispose()
