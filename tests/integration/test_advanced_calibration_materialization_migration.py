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


def test_0014_creates_materialization_tables_and_round_trips(tmp_path: Path) -> None:
    database_path = tmp_path / "advanced-calibration.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)

    command.upgrade(config, "0013")
    command.upgrade(config, "0014")

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert {
            "advanced_calibration_materializations",
            "advanced_calibration_sample_bindings",
        } <= set(inspector.get_table_names())
        assert {
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "status",
            "artifact_id",
            "artifact_manifest_sha256",
            "normalization_statistics_sha256",
            "decision_event_id",
            "ledger_sequence_number",
            "ledger_head_sha256",
            "source_registration_id",
            "source_identity_sha256",
            "sample_manifest_sha256",
            "sample_count",
            "idempotency_key_sha256",
            "request_sha256",
            "created_by_user_id",
            "created_at",
            "started_at",
            "completed_at",
            "failure_code",
        } <= {
            column["name"]
            for column in inspector.get_columns(
                "advanced_calibration_materializations"
            )
        }

        command.downgrade(config, "0013")
        downgraded_tables = set(inspect(engine).get_table_names())
        assert "advanced_calibration_materializations" not in downgraded_tables
        assert "advanced_calibration_sample_bindings" not in downgraded_tables
    finally:
        engine.dispose()


def test_0014_downgrade_rejects_persisted_audit_rows(tmp_path: Path) -> None:
    database_path = tmp_path / "advanced-calibration-audit.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "0014")
    engine = create_engine(database_url)
    timestamp = "2026-07-27 00:00:00+00:00"

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, username, credential_hash, must_change_credential, role, "
                    "status, created_at, updated_at) VALUES "
                    "('calibration-admin', 'admin@example.test', 'hash', 0, "
                    "'ADMIN', 'ACTIVE', :now, :now)"
                ),
                {"now": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, owner_user_id, name, status, created_at, updated_at) VALUES "
                    "('calibration-project', 'calibration-admin', 'calibration', "
                    "'ACTIVE', :now, :now)"
                ),
                {"now": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO model_artifacts "
                    "(id, project_id, artifact_format, object_uri, sha256, status, "
                    "created_at) VALUES ('calibration-artifact', "
                    "'calibration-project', 'safetensors-bundle', "
                    "'verified-model-artifact://calibration/bundle', :digest, "
                    "'VERIFIED', :now)"
                ),
                {"digest": "a" * 64, "now": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO model_route_activation_events "
                    "(id, project_id, stream_sequence, task, cutoff_cycle, "
                    "route_role, decision_type, artifact_id, artifact_sha256, "
                    "manifest_sha256, deployment_bundle_manifest_sha256, "
                    "route_provenance_sha256, rollback_target_event_id, "
                    "previous_event_sha256, event_sha256, actor_user_id, reason, "
                    "idempotency_key_sha256, request_sha256, created_at) VALUES "
                    "('calibration-route-event', 'calibration-project', 1, 'RUL', "
                    "20, 'DEFAULT', 'ACTIVATE', 'calibration-artifact', :a, :b, "
                    ":c, :d, NULL, :e, :f, 'calibration-admin', 'approved', :g, "
                    ":h, :now)"
                ),
                {
                    "a": "a" * 64,
                    "b": "b" * 64,
                    "c": "c" * 64,
                    "d": "d" * 64,
                    "e": "0" * 64,
                    "f": "f" * 64,
                    "g": "1" * 64,
                    "h": "2" * 64,
                    "now": timestamp,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO advanced_calibration_materializations "
                    "(id, project_id, task, cutoff_cycle, route_role, status, "
                    "data_version, split_version, feature_version, artifact_id, "
                    "artifact_manifest_sha256, normalization_statistics_sha256, "
                    "decision_event_id, ledger_sequence_number, ledger_head_sha256, "
                    "source_registration_id, source_identity_sha256, "
                    "sample_manifest_sha256, sample_count, idempotency_key_sha256, "
                    "request_sha256, created_by_user_id, created_at, started_at, "
                    "completed_at, failure_code) VALUES "
                    "('materialization-1', 'calibration-project', 'RUL', 20, "
                    "'DEFAULT', 'PENDING', 'data-v1', 'split-v1', 'feature-v1', "
                    "'calibration-artifact', :a, :b, 'calibration-route-event', "
                    "1, :c, 'matr-source-v1', :d, NULL, 0, :e, :f, "
                    "'calibration-admin', :now, NULL, NULL, NULL)"
                ),
                {
                    "a": "3" * 64,
                    "b": "4" * 64,
                    "c": "5" * 64,
                    "d": "6" * 64,
                    "e": "7" * 64,
                    "f": "8" * 64,
                    "now": timestamp,
                },
            )

        with pytest.raises(RuntimeError, match="would discard audited materializations"):
            command.downgrade(config, "0013")
    finally:
        engine.dispose()
