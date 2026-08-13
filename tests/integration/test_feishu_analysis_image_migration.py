from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from quanxin_life.persistence.models import FeishuEventReceipt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0024_adds_a_generic_analysis_image_checkpoint() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0024_feishu_analysis_image_checkpoint.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "0024"' in migration
    assert 'down_revision: str | None = "0023"' in migration
    assert '"analysis_image_key"' in migration
    assert '"analysis_image_renderer_version"' in migration
    assert '"analysis_image_sha256"' in migration
    assert "ck_feishu_event_receipt_analysis_image_provenance" in migration
    assert "predicted_soh" not in migration
    assert "natural_year" not in migration


def test_orm_declares_the_analysis_image_provenance_constraint() -> None:
    names = {constraint.name for constraint in FeishuEventReceipt.__table__.constraints}
    assert "ck_feishu_event_receipt_analysis_image_provenance" in names


def test_0024_upgrades_real_schema_and_refuses_to_discard_image_evidence(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'analysis-image-checkpoint.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0023")
    engine = create_engine(database_url)
    try:
        before = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "analysis_image_key" not in before

        command.upgrade(config, "0024")

        after = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert {
            "analysis_image_key",
            "analysis_image_renderer_version",
            "analysis_image_sha256",
        } <= after
        columns = {
            column["name"]: column for column in inspect(engine).get_columns(
                "feishu_event_receipts"
            )
        }
        assert columns["analysis_image_key"]["type"].length == 200
        assert columns["analysis_image_renderer_version"]["type"].length == 100
        assert columns["analysis_image_sha256"]["type"].length == 64
        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_image_key) "
                    "VALUES ('image-id','image-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'img-audited')"
                ),
                {"sha": "a" * 64},
            )

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_image_key,"
                    "analysis_image_renderer_version,analysis_image_sha256) "
                    "VALUES ('bad-sha','bad-sha-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'img-audited','renderer','bad')"
                ),
                {"sha": "b" * 64},
            )

        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_image_key,"
                    "analysis_image_renderer_version,analysis_image_sha256) "
                    "VALUES ('image-id','image-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'img-audited','renderer',:image_sha)"
                ),
                {"sha": "a" * 64, "image_sha": "c" * 64},
            )

        with pytest.raises(RuntimeError, match="discard analysis image evidence"):
            command.downgrade(config, "0023")
    finally:
        engine.dispose()


def test_0024_empty_downgrade_removes_all_analysis_image_provenance_columns(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'analysis-image-empty-downgrade.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0024")
    engine = create_engine(database_url)
    try:
        command.downgrade(config, "0023")
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert {
            "analysis_image_key",
            "analysis_image_renderer_version",
            "analysis_image_sha256",
        }.isdisjoint(columns)
    finally:
        engine.dispose()
