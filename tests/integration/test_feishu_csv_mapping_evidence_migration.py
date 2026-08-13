from __future__ import annotations

import json
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


def _database_url(tmp_path: Path, name: str) -> str:
    return f"sqlite+pysqlite:///{(tmp_path / name).as_posix()}"


def test_orm_declares_the_same_csv_mapping_evidence_constraints() -> None:
    constraint_names = {
        constraint.name for constraint in FeishuEventReceipt.__table__.constraints
    }

    assert "ck_feishu_event_receipt_csv_mapping_status" in constraint_names
    assert (
        "ck_feishu_event_receipt_csv_mapping_evidence_contract"
        in constraint_names
    )


def test_0023_adds_sanitized_csv_mapping_evidence_contract(tmp_path: Path) -> None:
    database_url = _database_url(tmp_path, "csv-mapping-evidence.sqlite3")
    config = _config(database_url)
    command.upgrade(config, "0022")
    engine = create_engine(database_url)
    try:
        before = {
            column["name"]
            for column in inspect(engine).get_columns("feishu_event_receipts")
        }
        assert "csv_mapping_status" not in before

        command.upgrade(config, "0023")

        inspector = inspect(engine)
        columns = {
            column["name"]
            for column in inspector.get_columns("feishu_event_receipts")
        }
        assert {
            "csv_mapping_status",
            "csv_mapping_evidence_json",
            "csv_mapping_evidence_sha256",
        } <= columns
        checks = " ".join(
            " ".join(str(item["sqltext"]).lower().split())
            for item in inspector.get_check_constraints("feishu_event_receipts")
        )
        assert "csv_mapping_status in ('mapped', 'rejected')" in checks
        assert "length(csv_mapping_evidence_sha256) = 64" in checks
        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,csv_mapping_status) "
                    "VALUES ('partial','partial-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'MAPPED')"
                ),
                {"sha": "a" * 64},
            )
        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,csv_mapping_status,"
                    "csv_mapping_evidence_json,csv_mapping_evidence_sha256) VALUES "
                    "('null-hash','null-hash-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'MAPPED','{}',NULL)"
                ),
                {"sha": "a" * 64},
            )
        with engine.begin() as connection, pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,csv_mapping_status,"
                    "csv_mapping_evidence_json,csv_mapping_evidence_sha256) VALUES "
                    "('invalid','invalid-event','type',:sha,'PROCESSED',1,"
                    "CURRENT_TIMESTAMP,'FEISHU',0,'UNKNOWN','{}',:sha)"
                ),
                {"sha": "a" * 64},
            )
        command.check(config)
    finally:
        engine.dispose()


def test_0023_downgrade_refuses_to_discard_mapping_evidence(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path, "csv-mapping-downgrade.sqlite3")
    config = _config(database_url)
    command.upgrade(config, "0023")
    engine = create_engine(database_url)
    evidence = {"code": "UNREVIEWED_LAYOUT", "raw_sha256": "a" * 64}
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_event_receipts "
                    "(id,event_id,event_type,payload_sha256,status,attempt_count,"
                    "received_at,job_origin,job_attempt_count,csv_mapping_status,"
                    "csv_mapping_evidence_json,csv_mapping_evidence_sha256) VALUES "
                    "('id','event','type',:sha,'PROCESSED',1,CURRENT_TIMESTAMP,"
                    "'FEISHU',0,'REJECTED',:evidence,:evidence_sha)"
                ),
                {
                    "sha": "a" * 64,
                    "evidence": json.dumps(evidence, separators=(",", ":")),
                    "evidence_sha": "b" * 64,
                },
            )

        with pytest.raises(RuntimeError, match="discard CSV mapping evidence"):
            command.downgrade(config, "0022")
    finally:
        engine.dispose()
