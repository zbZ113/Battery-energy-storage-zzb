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

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        upgraded_tables = set(inspect(engine).get_table_names())
        assert upgraded_tables >= EXPECTED_TABLES

        # The declared metadata must also round-trip through the lightweight
        # SQLite migration target without producing type drift.
        command.check(config)

        command.downgrade(config, "base")

        downgraded_tables = set(inspect(engine).get_table_names())
        assert EXPECTED_TABLES.isdisjoint(downgraded_tables)
    finally:
        engine.dispose()
