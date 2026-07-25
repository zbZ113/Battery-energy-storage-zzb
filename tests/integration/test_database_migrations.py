from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_TABLES = {
    "users",
    "sessions",
    "projects",
    "user_project_roles",
    "datasets",
    "dataset_files",
    "cell_splits",
    "agent_runs",
    "agent_run_dispatches",
    "agent_steps",
    "agent_events",
    "tool_results",
    "provenance_records",
    "model_artifacts",
    "model_manifests",
    "calibration_cohorts",
    "decision_policies",
    "approval_requests",
    "approval_actions",
    "reports",
    "report_exports",
    "knowledge_documents",
    "knowledge_chunks",
    "feishu_bindings",
    "feishu_event_receipts",
    "experiment_suites",
    "experiment_runs",
    "model_route_activation_events",
}


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_initial_migration_upgrades_empty_sqlite_and_downgrades_to_base(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "migration.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)

    command.upgrade(config, "0001")

    engine = create_engine(database_url)
    try:
        initial_inspector = inspect(engine)
        assert "agent_run_dispatches" not in initial_inspector.get_table_names()
        assert "created_by_user_id" not in {
            column["name"] for column in initial_inspector.get_columns("agent_runs")
        }

        command.upgrade(config, "head")

        upgraded_tables = set(inspect(engine).get_table_names())
        assert upgraded_tables >= EXPECTED_TABLES

        upgraded_inspector = inspect(engine)
        assert {
            "created_by_user_id",
            "idempotency_key_hash",
            "request_hash",
            "plan_json",
            "execution_plan_hash",
        } <= {column["name"] for column in upgraded_inspector.get_columns("agent_runs")}
        assert {"depends_on_json", "failure_policy"} <= {
            column["name"] for column in upgraded_inspector.get_columns("agent_steps")
        }
        assert {
            "attempts",
            "claim_token",
            "lease_expires_at",
            "last_error_code",
        } <= {column["name"] for column in upgraded_inspector.get_columns("agent_steps")}
        assert ("created_by_user_id", "idempotency_key_hash") in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints("agent_runs")
        }
        assert ("run_id", "step_id") in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints("approval_requests")
        }
        assert ("approval_request_id",) in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints("approval_actions")
        }
        assert ("run_id",) in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints("agent_run_dispatches")
        }
        assert ("agent_step_id",) in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints("tool_results")
        }
        assert {
            "project_id",
            "import_id",
            "dataset_id",
            "target",
            "source_commit",
            "output_sha256",
            "evidence_uri",
            "created_by_user_id",
        } <= {
            column["name"]
            for column in upgraded_inspector.get_columns("experiment_suites")
        }
        assert ("project_id", "import_id") in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints(
                "experiment_suites"
            )
        }
        assert ("suite_id", "cutoff_cycle", "model_name", "seed") in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints(
                "experiment_runs"
            )
        }
        assert {
            "project_id",
            "stream_sequence",
            "task",
            "cutoff_cycle",
            "route_role",
            "decision_type",
            "artifact_id",
            "previous_event_sha256",
            "event_sha256",
            "idempotency_key_sha256",
            "request_sha256",
        } <= {
            column["name"]
            for column in upgraded_inspector.get_columns(
                "model_route_activation_events"
            )
        }
        activation_uniques = {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints(
                "model_route_activation_events"
            )
        }
        assert (
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "stream_sequence",
        ) in activation_uniques
        assert ("project_id", "idempotency_key_sha256") in activation_uniques
        assert {
            "created_by_user_id",
            "object_size_bytes",
            "object_content_type",
            "idempotency_key_hash",
            "request_hash",
        } <= {
            column["name"]
            for column in upgraded_inspector.get_columns("knowledge_documents")
        }
        assert ("created_by_user_id", "idempotency_key_hash") in {
            tuple(constraint["column_names"])
            for constraint in upgraded_inspector.get_unique_constraints(
                "knowledge_documents"
            )
        }
        assert {"text_size_bytes", "text_content_type", "section_label"} <= {
            column["name"]
            for column in upgraded_inspector.get_columns("knowledge_chunks")
        }

        # The declared metadata must also round-trip through the lightweight
        # SQLite migration target without producing type drift.
        command.check(config)

        command.downgrade(config, "0001")

        revision_one_inspector = inspect(engine)
        assert "agent_run_dispatches" not in revision_one_inspector.get_table_names()
        assert "created_by_user_id" not in {
            column["name"] for column in revision_one_inspector.get_columns("agent_runs")
        }
        assert "depends_on_json" not in {
            column["name"] for column in revision_one_inspector.get_columns("agent_steps")
        }
        assert "claim_token" not in {
            column["name"] for column in revision_one_inspector.get_columns("agent_steps")
        }
        assert "object_size_bytes" not in {
            column["name"]
            for column in revision_one_inspector.get_columns("knowledge_documents")
        }
        assert "idempotency_key_hash" not in {
            column["name"]
            for column in revision_one_inspector.get_columns("knowledge_documents")
        }
        assert ("agent_step_id",) not in {
            tuple(constraint["column_names"])
            for constraint in revision_one_inspector.get_unique_constraints("tool_results")
        }

        command.downgrade(config, "base")

        downgraded_tables = set(inspect(engine).get_table_names())
        assert EXPECTED_TABLES.isdisjoint(downgraded_tables)
    finally:
        engine.dispose()


def test_knowledge_metadata_migration_rejects_unverified_legacy_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy-knowledge.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "0004")
    engine = create_engine(database_url)
    timestamp = "2026-07-16 00:00:00+00:00"
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, username, credential_hash, must_change_credential, role, "
                    "status, created_at, updated_at) VALUES "
                    "('legacy-user', 'legacy@example.test', 'legacy-hash', 0, "
                    "'ADMIN', 'ACTIVE', :now, :now)"
                ),
                {"now": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, owner_user_id, name, status, created_at, updated_at) VALUES "
                    "('legacy-project', 'legacy-user', 'legacy', 'ACTIVE', :now, :now)"
                ),
                {"now": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO knowledge_documents "
                    "(id, project_id, title, source_uri, source_sha256, license_name, "
                    "document_version, review_status, reviewer_user_id, reviewed_at, "
                    "object_uri, created_at) VALUES "
                    "('legacy-document', 'legacy-project', 'legacy', "
                    "'https://example.test/legacy', :digest, 'unknown', 'v1', "
                    "'PENDING', NULL, NULL, 'minio://legacy/object', :now)"
                ),
                {"digest": "a" * 64, "now": timestamp},
            )

        with pytest.raises(RuntimeError, match="requires empty knowledge tables"):
            command.upgrade(config, "head")
    finally:
        engine.dispose()
