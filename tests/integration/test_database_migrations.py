from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

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
        assert {"created_by_user_id", "object_size_bytes", "object_content_type"} <= {
            column["name"]
            for column in upgraded_inspector.get_columns("knowledge_documents")
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
        assert ("agent_step_id",) not in {
            tuple(constraint["column_names"])
            for constraint in revision_one_inspector.get_unique_constraints("tool_results")
        }

        command.downgrade(config, "base")

        downgraded_tables = set(inspect(engine).get_table_names())
        assert EXPECTED_TABLES.isdisjoint(downgraded_tables)
    finally:
        engine.dispose()
