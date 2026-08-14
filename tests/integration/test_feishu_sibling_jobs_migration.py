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


def test_0026_adds_sibling_job_provenance_and_empty_downgrade_is_safe(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'feishu-sibling-jobs.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0025")
    engine = create_engine(database_url)
    try:
        command.upgrade(config, "0026")
        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert columns["source_job_id"]["type"].length == 64
        assert columns["default_scenario_profile_id"]["type"].length == 200
        assert columns["default_scenario_profile_version"]["type"].length == 100
        assert columns["default_scenario_profile_sha256"]["type"].length == 64
        assert "ix_feishu_receipts_source_job_id" in {
            item["name"]
            for item in inspect(engine).get_indexes("feishu_event_receipts")
        }
        command.check(config)
        command.downgrade(config, "0025")
        remaining = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "source_job_id" not in remaining
    finally:
        engine.dispose()


def test_0026_refuses_to_discard_sibling_job_evidence(tmp_path: Path) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'feishu-sibling-evidence.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0026")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,source_job_id) "
                    "VALUES ('sibling-id','sibling-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,:source_job_id)"
                ),
                {
                    "sha": "a" * 64,
                    "source_job_id": "3a3c972b-a23e-42c3-af76-e39038806f13",
                },
            )

        with pytest.raises(RuntimeError, match="discard sibling job evidence"):
            command.downgrade(config, "0025")
    finally:
        engine.dispose()
