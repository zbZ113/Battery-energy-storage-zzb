from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import String, create_engine, inspect

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_project_tool_result_binding_migration_matches_frozen_contract(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite+pysqlite:///{(tmp_path / 'project-results.sqlite3').as_posix()}"
    config = _config(database_url)
    command.upgrade(config, "0011")
    engine = create_engine(database_url)
    try:
        assert "project_tool_result_bindings" not in inspect(engine).get_table_names()

        command.upgrade(config, "head")
        inspector = inspect(engine)
        columns = {
            column["name"]: column
            for column in inspector.get_columns("project_tool_result_bindings")
        }
        required = {
            "result_id",
            "binding_schema_version",
            "project_id",
            "actor_user_id",
            "actor_session_id",
            "actor_role",
            "invocation_source",
            "agent_run_id",
            "tool_name",
            "input_hash",
            "result_sha256",
            "binding_sha256",
            "created_at",
        }
        assert columns.keys() >= required
        assert columns["result_id"]["primary_key"] == 1
        assert isinstance(columns["result_sha256"]["type"], String)
        assert columns["result_sha256"]["type"].length == 64
        assert columns["binding_sha256"]["type"].length == 64
        assert columns["agent_run_id"]["nullable"] is True
        assert all(
            columns[name]["nullable"] is False
            for name in required - {"agent_run_id"}
        )

        foreign_keys = {
            (
                tuple(item["constrained_columns"]),
                item["referred_table"],
                tuple(item["referred_columns"]),
            )
            for item in inspector.get_foreign_keys("project_tool_result_bindings")
        }
        assert (("result_id",), "tool_results", ("id",)) in foreign_keys
        assert (("project_id",), "projects", ("id",)) in foreign_keys
        assert (("actor_user_id",), "users", ("id",)) in foreign_keys
        assert (("actor_session_id",), "sessions", ("id",)) in foreign_keys
        assert (("agent_run_id",), "agent_runs", ("id",)) in foreign_keys

        checks = " ".join(
            " ".join(str(item["sqltext"]).lower().split())
            for item in inspector.get_check_constraints(
                "project_tool_result_bindings"
            )
        )
        assert "project-tool-result-binding-v1" in checks
        assert "length(result_sha256) = 64" in checks
        assert "length(binding_sha256) = 64" in checks
        assert "'http'" in checks and "'agent'" in checks

        command.check(config)
        command.downgrade(config, "0011")
        assert "project_tool_result_bindings" not in inspect(engine).get_table_names()
    finally:
        engine.dispose()
