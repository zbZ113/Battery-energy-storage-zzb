from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0025_adds_prepared_input_result_slot_and_empty_downgrade_is_safe(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'project-result-slots.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0024")
    engine = create_engine(database_url)
    try:
        command.upgrade(config, "0025")
        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "prepared_input_result_id" in columns
        assert columns["prepared_input_result_id"]["type"].length == 64
        assert columns["prepared_input_result_id"]["nullable"] is True
        command.upgrade(config, "head")
        command.check(config)
        command.downgrade(config, "0024")
        columns_after = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "prepared_input_result_id" not in columns_after
    finally:
        engine.dispose()


def test_0025_refuses_to_discard_prepared_input_evidence(tmp_path: Path) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'project-result-slots-evidence.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0025")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,prepared_input_result_id) "
                    "VALUES ('prepared-id','prepared-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,:result_id)"
                ),
                {"sha": "a" * 64, "result_id": "b" * 64},
            )

        with pytest.raises(RuntimeError, match="discard prepared input evidence"):
            command.downgrade(config, "0024")

        columns = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "prepared_input_result_id" in columns
    finally:
        engine.dispose()
