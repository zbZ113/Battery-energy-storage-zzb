from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _migration_0015_module() -> ModuleType:
    path = PROJECT_ROOT / "migrations/versions/0015_advanced_calibration_claims.py"
    spec = importlib.util.spec_from_file_location("migration_0015_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_0015_uses_native_alter_for_postgresql_and_recreate_for_sqlite() -> None:
    migration = _migration_0015_module()

    assert migration._batch_recreate_mode("postgresql") == "auto"
    assert migration._batch_recreate_mode("sqlite") == "always"


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_0015_adds_actor_session_and_fenced_claim_contract(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "advanced-calibration-claims.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "0015")
    engine = create_engine(database_url)

    try:
        inspector = inspect(engine)
        columns = {
            column["name"]: column
            for column in inspector.get_columns(
                "advanced_calibration_materializations"
            )
        }
        assert {
            "created_by_session_id",
            "created_by_role",
            "claim_token_sha256",
            "claim_attempt",
            "claim_lease_expires_at",
        } <= set(columns)
        assert columns["created_by_session_id"]["nullable"] is False
        assert columns["created_by_role"]["nullable"] is False
        assert columns["claim_token_sha256"]["nullable"] is True
        assert columns["claim_attempt"]["nullable"] is False
        assert columns["claim_lease_expires_at"]["nullable"] is True

        foreign_keys = inspector.get_foreign_keys(
            "advanced_calibration_materializations"
        )
        assert any(
            item["constrained_columns"] == ["created_by_session_id"]
            and item["referred_table"] == "sessions"
            for item in foreign_keys
        )
        checks = " ".join(
            str(item["sqltext"])
            for item in inspector.get_check_constraints(
                "advanced_calibration_materializations"
            )
        )
        assert "created_by_role = 'ADMIN'" in checks
        assert "claim_attempt = 0" in checks
        assert "claim_attempt > 0" in checks
        assert "claim_lease_expires_at IS NOT NULL" in checks
        assert "length(claim_token_sha256) = 64" in checks
    finally:
        engine.dispose()


def test_0015_state_constraint_rejects_unfenced_running_row(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "advanced-calibration-state.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "0015")
    engine = create_engine(database_url)
    now = "2026-07-27 00:00:00+00:00"

    try:
        with engine.begin() as connection:
            _seed_dependencies(connection, now=now)
            with pytest.raises(IntegrityError):
                connection.execute(
                    text(
                        _materialization_insert_sql(
                            status="RUNNING",
                            started_at=":now",
                            claim_token_sha256="NULL",
                            claim_attempt="1",
                            claim_lease_expires_at="NULL",
                        )
                    ),
                    _materialization_parameters(now=now),
                )
    finally:
        engine.dispose()


def test_0015_round_trips_when_empty_and_rejects_audited_downgrade(
    tmp_path: Path,
) -> None:
    empty_path = tmp_path / "advanced-calibration-empty.sqlite3"
    empty_url = f"sqlite+pysqlite:///{empty_path.as_posix()}"
    empty_config = _config(empty_url)
    command.upgrade(empty_config, "0015")
    command.downgrade(empty_config, "0014")
    empty_engine = create_engine(empty_url)
    try:
        columns = {
            column["name"]
            for column in inspect(empty_engine).get_columns(
                "advanced_calibration_materializations"
            )
        }
        assert "created_by_session_id" not in columns
        assert "claim_token_sha256" not in columns
    finally:
        empty_engine.dispose()

    audited_path = tmp_path / "advanced-calibration-audited.sqlite3"
    audited_url = f"sqlite+pysqlite:///{audited_path.as_posix()}"
    audited_config = _config(audited_url)
    command.upgrade(audited_config, "0015")
    audited_engine = create_engine(audited_url)
    now = "2026-07-27 00:00:00+00:00"
    try:
        with audited_engine.begin() as connection:
            _seed_dependencies(connection, now=now)
            connection.execute(
                text(
                    _materialization_insert_sql(
                        status="PENDING",
                        started_at="NULL",
                        claim_token_sha256="NULL",
                        claim_attempt="0",
                        claim_lease_expires_at="NULL",
                    )
                ),
                _materialization_parameters(now=now),
            )
        with pytest.raises(RuntimeError, match="would discard fenced claim evidence"):
            command.downgrade(audited_config, "0014")
    finally:
        audited_engine.dispose()


def _seed_dependencies(connection: Connection, *, now: str) -> None:
    connection.execute(
        text(
            "INSERT INTO users "
            "(id, username, credential_hash, must_change_credential, role, "
            "status, created_at, updated_at) VALUES "
            "('calibration-admin', 'admin@example.test', 'hash', 0, "
            "'ADMIN', 'ACTIVE', :now, :now)"
        ),
        {"now": now},
    )
    connection.execute(
        text(
            "INSERT INTO sessions "
            "(id, user_id, token_hash, status, created_at, expires_at, revoked_at) "
            "VALUES ('calibration-session', 'calibration-admin', :token, 'ACTIVE', "
            ":now, :expires, NULL)"
        ),
        {"token": "9" * 64, "now": now, "expires": "2026-07-28 00:00:00+00:00"},
    )
    connection.execute(
        text(
            "INSERT INTO projects "
            "(id, owner_user_id, name, status, created_at, updated_at) VALUES "
            "('calibration-project', 'calibration-admin', 'calibration', "
            "'ACTIVE', :now, :now)"
        ),
        {"now": now},
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
        {"digest": "a" * 64, "now": now},
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
            "now": now,
        },
    )


def _materialization_insert_sql(
    *,
    status: str,
    started_at: str,
    claim_token_sha256: str,
    claim_attempt: str,
    claim_lease_expires_at: str,
) -> str:
    return (
        "INSERT INTO advanced_calibration_materializations "
        "(id, project_id, task, cutoff_cycle, route_role, status, "
        "data_version, split_version, feature_version, artifact_id, "
        "artifact_manifest_sha256, normalization_statistics_sha256, "
        "decision_event_id, ledger_sequence_number, ledger_head_sha256, "
        "source_registration_id, source_identity_sha256, "
        "sample_manifest_sha256, sample_count, idempotency_key_sha256, "
        "request_sha256, created_by_user_id, created_by_session_id, "
        "created_by_role, created_at, started_at, completed_at, failure_code, "
        "claim_token_sha256, claim_attempt, claim_lease_expires_at) VALUES "
        "('materialization-1', 'calibration-project', 'RUL', 20, 'DEFAULT', "
        f"'{status}', 'data-v1', 'split-v1', 'feature-v1', "
        "'calibration-artifact', :a, :b, 'calibration-route-event', 1, :c, "
        "'matr-source-v1', :d, NULL, 0, :e, :f, 'calibration-admin', "
        "'calibration-session', 'ADMIN', :now, "
        f"{started_at}, NULL, NULL, {claim_token_sha256}, {claim_attempt}, "
        f"{claim_lease_expires_at})"
    )


def _materialization_parameters(*, now: str) -> dict[str, str]:
    return {
        "a": "3" * 64,
        "b": "4" * 64,
        "c": "5" * 64,
        "d": "6" * 64,
        "e": "7" * 64,
        "f": "8" * 64,
        "now": now,
    }
