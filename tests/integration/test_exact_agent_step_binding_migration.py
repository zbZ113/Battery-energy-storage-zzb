import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AGENT_STEP_FREEZE_COLUMNS = {
    "resolved_input_json",
    "resolved_input_hash",
    "execution_snapshot_sha256",
    "dependency_evidence_sha256",
    "execution_claim_sha256",
}
APPROVAL_FREEZE_COLUMNS = {
    "agent_step_id",
    "execution_snapshot_sha256",
}
BINDING_V2_COLUMNS = {
    "agent_step_id",
    "step_id",
    "plan_hash",
    "claim_token_sha256",
    "claim_attempt",
    "claim_lease_expires_at",
    "execution_snapshot_sha256",
    "dependency_evidence_sha256",
    "approval_required",
    "approval_request_id",
    "approval_action_id",
    "approval_evidence_sha256",
}


def _migration_0013_module() -> ModuleType:
    path = PROJECT_ROOT / "migrations/versions/0013_exact_agent_step_bindings.py"
    spec = importlib.util.spec_from_file_location("migration_0013_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_0013_uses_native_alter_for_postgresql_and_recreate_for_sqlite() -> None:
    migration = _migration_0013_module()

    assert migration._batch_recreate_mode("postgresql") == "auto"
    assert migration._batch_recreate_mode("sqlite") == "always"


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _columns(inspector, table_name: str) -> dict[str, dict[str, object]]:
    return {
        column["name"]: column
        for column in inspector.get_columns(table_name)
    }


def _checks(inspector, table_name: str) -> str:
    return " ".join(
        " ".join(str(item["sqltext"]).split()).casefold()
        for item in inspector.get_check_constraints(table_name)
    )


def _revision(engine) -> str:
    with engine.connect() as connection:
        return connection.scalar(text("SELECT version_num FROM alembic_version"))


def _foreign_keys(
    inspector,
    table_name: str,
) -> set[tuple[tuple[str, ...], str, tuple[str, ...]]]:
    return {
        (
            tuple(item["constrained_columns"]),
            item["referred_table"],
            tuple(item["referred_columns"]),
        )
        for item in inspector.get_foreign_keys(table_name)
    }


def test_0013_upgrades_0012_to_exact_agent_step_binding_contract(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'exact-agent-step-binding.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0012")
    engine = create_engine(database_url)
    try:
        revision_0012 = inspect(engine)
        assert AGENT_STEP_FREEZE_COLUMNS.isdisjoint(
            _columns(revision_0012, "agent_steps")
        )
        assert APPROVAL_FREEZE_COLUMNS.isdisjoint(
            _columns(revision_0012, "approval_requests")
        )
        assert BINDING_V2_COLUMNS.isdisjoint(
            _columns(revision_0012, "project_tool_result_bindings")
        )

        command.upgrade(config, "head")
        inspector = inspect(engine)

        step_columns = _columns(inspector, "agent_steps")
        assert step_columns.keys() >= AGENT_STEP_FREEZE_COLUMNS
        assert isinstance(step_columns["resolved_input_json"]["type"], JSON)
        for name in AGENT_STEP_FREEZE_COLUMNS:
            assert step_columns[name]["nullable"] is True
        for name in AGENT_STEP_FREEZE_COLUMNS - {"resolved_input_json"}:
            assert isinstance(step_columns[name]["type"], String)
            assert step_columns[name]["type"].length == 64

        step_checks = _checks(inspector, "agent_steps")
        for name in AGENT_STEP_FREEZE_COLUMNS - {"resolved_input_json"}:
            assert name in step_checks
            assert f"length({name}) = 64" in step_checks

        approval_columns = _columns(inspector, "approval_requests")
        assert approval_columns.keys() >= APPROVAL_FREEZE_COLUMNS
        assert approval_columns["agent_step_id"]["nullable"] is True
        assert isinstance(approval_columns["agent_step_id"]["type"], String)
        assert approval_columns["agent_step_id"]["type"].length == 64
        assert approval_columns["execution_snapshot_sha256"]["nullable"] is True
        assert isinstance(
            approval_columns["execution_snapshot_sha256"]["type"], String
        )
        assert approval_columns["execution_snapshot_sha256"]["type"].length == 64
        approval_foreign_keys = _foreign_keys(inspector, "approval_requests")
        assert (("agent_step_id",), "agent_steps", ("id",)) in approval_foreign_keys
        approval_checks = _checks(inspector, "approval_requests")
        assert "execution_snapshot_sha256" in approval_checks
        assert "length(execution_snapshot_sha256) = 64" in approval_checks

        binding_columns = _columns(inspector, "project_tool_result_bindings")
        assert binding_columns.keys() >= BINDING_V2_COLUMNS
        for name in BINDING_V2_COLUMNS:
            assert binding_columns[name]["nullable"] is True
        for name in {
            "agent_step_id",
            "plan_hash",
            "claim_token_sha256",
            "execution_snapshot_sha256",
            "dependency_evidence_sha256",
            "approval_request_id",
            "approval_action_id",
            "approval_evidence_sha256",
        }:
            assert isinstance(binding_columns[name]["type"], String)
            assert binding_columns[name]["type"].length == 64
        assert isinstance(binding_columns["step_id"]["type"], String)
        assert binding_columns["step_id"]["type"].length == 200
        assert isinstance(binding_columns["claim_attempt"]["type"], Integer)
        assert isinstance(
            binding_columns["claim_lease_expires_at"]["type"], DateTime
        )
        assert isinstance(binding_columns["approval_required"]["type"], Boolean)

        binding_foreign_keys = _foreign_keys(
            inspector,
            "project_tool_result_bindings",
        )
        assert (("agent_step_id",), "agent_steps", ("id",)) in binding_foreign_keys
        assert (
            ("approval_request_id",),
            "approval_requests",
            ("id",),
        ) in binding_foreign_keys
        assert (
            ("approval_action_id",),
            "approval_actions",
            ("id",),
        ) in binding_foreign_keys

        binding_uniques = {
            tuple(item["column_names"])
            for item in inspector.get_unique_constraints(
                "project_tool_result_bindings"
            )
        }
        assert ("agent_step_id",) in binding_uniques
        assert ("agent_run_id", "step_id") in binding_uniques

        binding_indexes = {
            tuple(item["column_names"])
            for item in inspector.get_indexes("project_tool_result_bindings")
            if not item["unique"]
        }
        assert any("agent_step_id" in columns for columns in binding_indexes)

        binding_checks = _checks(inspector, "project_tool_result_bindings")
        for fragment in (
            "project-tool-result-binding-v1",
            "project-tool-result-binding-v2",
            "invocation_source = 'http'",
            "invocation_source = 'agent'",
            "agent_run_id",
            "agent_step_id",
            "step_id",
            "plan_hash",
            "claim_token_sha256",
            "claim_attempt",
            "claim_lease_expires_at",
            "execution_snapshot_sha256",
            "dependency_evidence_sha256",
            "approval_required",
            "approval_request_id",
            "approval_action_id",
            "approval_evidence_sha256",
        ):
            assert fragment in binding_checks
        assert "claim_attempt > 0" in binding_checks
        for name in {
            "plan_hash",
            "claim_token_sha256",
            "execution_snapshot_sha256",
            "dependency_evidence_sha256",
            "approval_evidence_sha256",
        }:
            assert f"length({name}) = 64" in binding_checks

        command.check(config)

        command.downgrade(config, "0012")
        downgraded = inspect(engine)
        assert AGENT_STEP_FREEZE_COLUMNS.isdisjoint(
            _columns(downgraded, "agent_steps")
        )
        assert APPROVAL_FREEZE_COLUMNS.isdisjoint(
            _columns(downgraded, "approval_requests")
        )
        assert BINDING_V2_COLUMNS.isdisjoint(
            _columns(downgraded, "project_tool_result_bindings")
        )
    finally:
        engine.dispose()


def test_0013_empty_database_can_downgrade_to_base(tmp_path: Path) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'exact-agent-step-empty-downgrade.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        assert "project_tool_result_bindings" in inspect(engine).get_table_names()

        command.downgrade(config, "base")

        assert inspect(engine).get_table_names() == ["alembic_version"]
    finally:
        engine.dispose()


def test_0013_preserves_http_v1_binding_across_upgrade_and_downgrade(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'http-v1-binding-roundtrip.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0012")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO project_tool_result_bindings "
                    "(result_id,binding_schema_version,project_id,actor_user_id,"
                    "actor_session_id,actor_role,invocation_source,agent_run_id,"
                    "tool_name,input_hash,result_sha256,binding_sha256,created_at) "
                    "VALUES (:result_id,'project-tool-result-binding-v1',:project_id,"
                    ":user_id,:session_id,'MEMBER','HTTP',NULL,"
                    "'predict_cycle_life',:digest,:digest,:digest,:created_at)"
                ),
                {
                    "result_id": "http-result",
                    "project_id": "http-project",
                    "user_id": "http-user",
                    "session_id": "http-session",
                    "digest": "c" * 64,
                    "created_at": "2026-07-26 00:00:00",
                },
            )

        command.upgrade(config, "0013")
        with engine.connect() as connection:
            upgraded = connection.execute(
                text(
                    "SELECT binding_schema_version,invocation_source,agent_run_id,"
                    "agent_step_id,step_id,plan_hash,claim_token_sha256 "
                    "FROM project_tool_result_bindings WHERE result_id='http-result'"
                )
            ).one()
        assert tuple(upgraded) == (
            "project-tool-result-binding-v1",
            "HTTP",
            None,
            None,
            None,
            None,
            None,
        )

        command.downgrade(config, "0012")
        with engine.connect() as connection:
            downgraded = connection.execute(
                text(
                    "SELECT binding_schema_version,invocation_source,agent_run_id "
                    "FROM project_tool_result_bindings WHERE result_id='http-result'"
                )
            ).one()
        assert tuple(downgraded) == (
            "project-tool-result-binding-v1",
            "HTTP",
            None,
        )
    finally:
        engine.dispose()


def test_0013_rejects_legacy_agent_v1_rows_before_schema_changes(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'legacy-agent-binding.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0012")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO project_tool_result_bindings "
                    "(result_id,binding_schema_version,project_id,actor_user_id,"
                    "actor_session_id,actor_role,invocation_source,agent_run_id,"
                    "tool_name,input_hash,result_sha256,binding_sha256,created_at) "
                    "VALUES (:result_id,'project-tool-result-binding-v1',:project_id,"
                    ":user_id,:session_id,'MEMBER','AGENT',:run_id,"
                    "'predict_cycle_life',:digest,:digest,:digest,:created_at)"
                ),
                {
                    "result_id": "legacy-result",
                    "project_id": "legacy-project",
                    "user_id": "legacy-user",
                    "session_id": "legacy-session",
                    "run_id": "legacy-run",
                    "digest": "a" * 64,
                    "created_at": "2026-07-26 00:00:00",
                },
            )

        with pytest.raises(RuntimeError, match="legacy AGENT"):
            command.upgrade(config, "head")

        assert _revision(engine) == "0012"
        inspector = inspect(engine)
        assert AGENT_STEP_FREEZE_COLUMNS.isdisjoint(
            _columns(inspector, "agent_steps")
        )
        assert BINDING_V2_COLUMNS.isdisjoint(
            _columns(inspector, "project_tool_result_bindings")
        )
    finally:
        engine.dispose()


def test_0013_rejects_downgrade_when_v2_agent_evidence_exists(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'v2-agent-binding.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "head")
    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO project_tool_result_bindings "
                    "(result_id,binding_schema_version,project_id,actor_user_id,"
                    "actor_session_id,actor_role,invocation_source,agent_run_id,"
                    "agent_step_id,step_id,plan_hash,claim_token_sha256,claim_attempt,"
                    "claim_lease_expires_at,execution_snapshot_sha256,"
                    "dependency_evidence_sha256,approval_required,approval_request_id,"
                    "approval_action_id,approval_evidence_sha256,tool_name,input_hash,"
                    "result_sha256,binding_sha256,created_at) VALUES "
                    "(:result_id,'project-tool-result-binding-v2',:project_id,:user_id,"
                    ":session_id,'MEMBER','AGENT',:run_id,:agent_step_id,'predict',"
                    ":digest,:digest,1,:lease,:digest,:digest,0,NULL,NULL,:digest,"
                    "'predict_cycle_life',:digest,:digest,:digest,:created_at)"
                ),
                {
                    "result_id": "v2-result",
                    "project_id": "v2-project",
                    "user_id": "v2-user",
                    "session_id": "v2-session",
                    "run_id": "v2-run",
                    "agent_step_id": "v2-step",
                    "digest": "b" * 64,
                    "lease": "2026-07-26 01:00:00",
                    "created_at": "2026-07-26 00:00:00",
                },
            )

        with pytest.raises(RuntimeError, match="v2 Agent"):
            command.downgrade(config, "0012")

        assert _revision(engine) == "0013"
        assert _columns(
            inspect(engine), "project_tool_result_bindings"
        ).keys() >= BINDING_V2_COLUMNS
    finally:
        engine.dispose()
