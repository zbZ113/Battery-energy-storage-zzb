from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from quanxin_life.persistence.models import FeishuEventReceipt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    PROJECT_ROOT
    / "migrations/versions/0027_feishu_bitable_curve_checkpoint.py"
)
CURVE_COLUMNS = {
    "bitable_curve_file_token",
    "bitable_curve_source_result_id",
    "bitable_curve_renderer_version",
    "bitable_curve_sha256",
    "bitable_curve_template",
}


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0027_declares_complete_bitable_curve_provenance() -> None:
    migration = MIGRATION.read_text(encoding="utf-8")

    assert 'revision: str = "0027"' in migration
    assert 'down_revision: str | None = "0026"' in migration
    assert set(migration.split('"')) >= CURVE_COLUMNS
    assert "ck_feishu_event_receipt_bitable_curve_provenance" in migration
    assert "analysis_result_id IS NOT NULL" in migration
    assert "bitable_curve_source_result_id = analysis_result_id" in migration
    assert "predicted_soh" not in migration
    assert "predicted_cycle" not in migration


def test_orm_declares_the_bitable_curve_provenance_constraint() -> None:
    names = {constraint.name for constraint in FeishuEventReceipt.__table__.constraints}
    assert "ck_feishu_event_receipt_bitable_curve_provenance" in names


def test_0027_upgrades_schema_and_rejects_partial_or_mismatched_evidence(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'bitable-curve.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0026")
    engine = create_engine(database_url)
    try:
        before = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert CURVE_COLUMNS.isdisjoint(before)

        command.upgrade(config, "0027")

        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert set(columns) >= CURVE_COLUMNS
        assert columns["bitable_curve_file_token"]["type"].length == 200
        assert columns["bitable_curve_source_result_id"]["type"].length == 64
        assert columns["bitable_curve_renderer_version"]["type"].length == 100
        assert columns["bitable_curve_sha256"]["type"].length == 64
        assert columns["bitable_curve_template"]["type"].length == 100

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_result_id,"
                    "bitable_curve_file_token) VALUES "
                    "('partial','partial-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'result-a','file-a')"
                ),
                {"sha": "a" * 64},
            )

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_result_id,"
                    "bitable_curve_file_token,bitable_curve_source_result_id,"
                    "bitable_curve_renderer_version,bitable_curve_sha256,"
                    "bitable_curve_template) VALUES "
                    "('mismatch','mismatch-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'result-a','file-a','result-b',"
                    "'renderer-v1',:curve_sha,'FINITE_SOH_CURVE')"
                ),
                {"sha": "b" * 64, "curve_sha": "c" * 64},
            )

        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,"
                    "bitable_curve_file_token,bitable_curve_source_result_id,"
                    "bitable_curve_renderer_version,bitable_curve_sha256,"
                    "bitable_curve_template) VALUES "
                    "('missing-analysis','missing-analysis-event','type',:sha,"
                    "'PROCESSED',1,CURRENT_TIMESTAMP,'FEISHU',0,'file-a','result-a',"
                    "'renderer-v1',:curve_sha,'FINITE_SOH_CURVE')"
                ),
                {"sha": "f" * 64, "curve_sha": "a" * 64},
            )

        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,analysis_result_id,"
                    "bitable_curve_file_token,bitable_curve_source_result_id,"
                    "bitable_curve_renderer_version,bitable_curve_sha256,"
                    "bitable_curve_template) VALUES "
                    "('complete','complete-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'result-a','file-a','result-a',"
                    "'renderer-v1',:curve_sha,'FINITE_SOH_CURVE')"
                ),
                {"sha": "d" * 64, "curve_sha": "e" * 64},
            )

        with pytest.raises(RuntimeError, match="discard Bitable curve evidence"):
            command.downgrade(config, "0026")
    finally:
        engine.dispose()


def test_0027_empty_downgrade_removes_curve_columns(tmp_path: Path) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'bitable-curve-empty.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0027")
    engine = create_engine(database_url)
    try:
        command.downgrade(config, "0026")
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert CURVE_COLUMNS.isdisjoint(columns)
    finally:
        engine.dispose()
