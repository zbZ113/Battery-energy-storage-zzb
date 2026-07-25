from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import String, create_engine, inspect, text
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.exc import IntegrityError

from quanxin_life.persistence import models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_COLUMNS = {
    "id",
    "binding_schema_version",
    "content_batch_id",
    "project_id",
    "dataset_id",
    "source_manifest_sha256",
    "registration_sha256",
    "content_dataset_id",
    "dataset_schema_version",
    "cell_id",
    "cutoff_cycle",
    "data_version",
    "split_version",
    "feature_version",
    "created_by_user_id",
    "created_at",
}


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _database_url(tmp_path: Path, filename: str) -> str:
    database_path = tmp_path / filename
    return f"sqlite+pysqlite:///{database_path.as_posix()}"


def _normalized_check_sql(inspector: Inspector) -> str:
    constraints = inspector.get_check_constraints("record_batch_bindings")
    return " ".join(
        " ".join(str(constraint["sqltext"]).lower().split())
        for constraint in constraints
    )


def test_record_batch_binding_migration_matches_the_frozen_contract(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path, "record-batch-binding.sqlite3")
    config = _config(database_url)
    command.upgrade(config, "0010")
    engine = create_engine(database_url)
    try:
        assert "record_batch_bindings" not in inspect(engine).get_table_names()

        command.upgrade(config, "head")
        inspector = inspect(engine)
        assert "record_batch_bindings" in inspector.get_table_names()

        columns = {
            column["name"]: column
            for column in inspector.get_columns("record_batch_bindings")
        }
        assert columns.keys() >= REQUIRED_COLUMNS
        assert all(columns[name]["nullable"] is False for name in REQUIRED_COLUMNS)
        assert columns["id"]["primary_key"] == 1
        assert isinstance(columns["content_batch_id"]["type"], String)
        assert columns["content_batch_id"]["type"].length >= 78

        foreign_keys = {
            (
                tuple(constraint["constrained_columns"]),
                constraint["referred_table"],
                tuple(constraint["referred_columns"]),
            )
            for constraint in inspector.get_foreign_keys("record_batch_bindings")
        }
        assert (("project_id",), "projects", ("id",)) in foreign_keys
        assert (("dataset_id",), "datasets", ("id",)) in foreign_keys
        assert (("created_by_user_id",), "users", ("id",)) in foreign_keys

        unique_columns = {
            tuple(constraint["column_names"])
            for constraint in inspector.get_unique_constraints(
                "record_batch_bindings"
            )
        }
        assert ("dataset_id", "content_batch_id") in unique_columns
        assert ("content_batch_id",) not in unique_columns

        index_columns = {
            tuple(index["column_names"])
            for index in inspector.get_indexes("record_batch_bindings")
        }
        assert any(
            columns[:2] == ("project_id", "dataset_id")
            for columns in index_columns
        )

        check_sql = _normalized_check_sql(inspector)
        assert "cutoff_cycle > 0" in check_sql
        assert "length(source_manifest_sha256) = 64" in check_sql
        assert "length(registration_sha256) = 64" in check_sql
        assert "binding_schema_version = 'record-batch-binding-v1'" in check_sql

        command.check(config)
        command.downgrade(config, "0010")
        assert "record_batch_bindings" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()


def test_content_batch_identifier_is_unique_only_within_one_dataset(
    tmp_path: Path,
) -> None:
    database_url = _database_url(tmp_path, "record-batch-binding-unique.sqlite3")
    config = _config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "record_batch_bindings" in inspector.get_table_names()

        now = datetime(2026, 7, 25, tzinfo=UTC)
        content_batch_id = "canonical-csv-" + "a" * 64
        binding_values = {
            "binding_schema_version": "record-batch-binding-v1",
            "content_batch_id": content_batch_id,
            "source_manifest_sha256": "b" * 64,
            "registration_sha256": "c" * 64,
            "content_dataset_id": "source-dataset",
            "dataset_schema_version": "cycle-record-v1",
            "cell_id": "cell-001",
            "cutoff_cycle": 20,
            "data_version": "data-v1",
            "split_version": "split-v1",
            "feature_version": "feature-v1",
            "created_by_user_id": "binding-user",
            "created_at": now,
        }
        with engine.begin() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, username, credential_hash, must_change_credential, role, "
                    "status, created_at, updated_at) VALUES "
                    "('binding-user', 'binding@example.test', 'hash', 0, 'ADMIN', "
                    "'ACTIVE', :now, :now)"
                ),
                {"now": now},
            )
            for project_id, dataset_id in (
                ("project-a", "dataset-a"),
                ("project-b", "dataset-b"),
            ):
                connection.execute(
                    text(
                        "INSERT INTO projects "
                        "(id, owner_user_id, name, status, created_at, updated_at) "
                        "VALUES (:project_id, 'binding-user', :project_id, 'ACTIVE', "
                        ":now, :now)"
                    ),
                    {"project_id": project_id, "now": now},
                )
                connection.execute(
                    text(
                        "INSERT INTO datasets "
                        "(id, project_id, name, data_version, schema_version, status, "
                        "created_at) VALUES (:dataset_id, :project_id, :dataset_id, "
                        "'data-v1', 'cycle-record-v1', 'DRAFT', :now)"
                    ),
                    {
                        "dataset_id": dataset_id,
                        "project_id": project_id,
                        "now": now,
                    },
                )

        insert_sql = text(
            "INSERT INTO record_batch_bindings "
            "(id, binding_schema_version, content_batch_id, project_id, dataset_id, "
            "source_manifest_sha256, registration_sha256, content_dataset_id, "
            "dataset_schema_version, cell_id, cutoff_cycle, data_version, "
            "split_version, feature_version, created_by_user_id, created_at) VALUES "
            "(:id, :binding_schema_version, :content_batch_id, :project_id, "
            ":dataset_id, :source_manifest_sha256, :registration_sha256, "
            ":content_dataset_id, :dataset_schema_version, :cell_id, :cutoff_cycle, "
            ":data_version, :split_version, :feature_version, :created_by_user_id, "
            ":created_at)"
        )
        with engine.begin() as connection:
            connection.execute(
                insert_sql,
                binding_values
                | {
                    "id": "00000000-0000-4000-8000-000000000001",
                    "project_id": "project-a",
                    "dataset_id": "dataset-a",
                },
            )
            connection.execute(
                insert_sql,
                binding_values
                | {
                    "id": "00000000-0000-4000-8000-000000000002",
                    "project_id": "project-b",
                    "dataset_id": "dataset-b",
                },
            )

        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                insert_sql,
                binding_values
                | {
                    "id": "00000000-0000-4000-8000-000000000003",
                    "project_id": "project-a",
                    "dataset_id": "dataset-a",
                },
            )
    finally:
        engine.dispose()


def test_record_batch_binding_model_is_declared_in_base_metadata() -> None:
    assert hasattr(models, "RecordBatchBinding")
    table = models.RecordBatchBinding.__table__

    assert table is models.Base.metadata.tables["record_batch_bindings"]
    assert set(table.columns.keys()) >= REQUIRED_COLUMNS
    assert table.c.id.primary_key is True
    assert isinstance(table.c.id.type, String)
    assert isinstance(table.c.content_batch_id.type, String)
    assert table.c.content_batch_id.type.length >= 78
    assert isinstance(table.c.created_at.type, models.UTCDateTime)
    assert table.c.created_at.type.timezone is True
