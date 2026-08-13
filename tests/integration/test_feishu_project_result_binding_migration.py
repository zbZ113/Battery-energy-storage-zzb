from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _migration_0022_module() -> ModuleType:
    path = PROJECT_ROOT / "migrations/versions/0022_feishu_project_result_bindings.py"
    spec = importlib.util.spec_from_file_location("migration_0022_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_0022_uses_native_alter_for_postgresql_and_recreate_for_sqlite() -> None:
    migration = _migration_0022_module()

    assert migration._batch_recreate_mode("postgresql") == "auto"
    assert migration._batch_recreate_mode("sqlite") == "always"


def test_0022_declares_explicit_legacy_and_feishu_source_contracts() -> None:
    migration = _migration_0022_module()

    legacy = " ".join(migration._legacy_source_contract().split())
    current = " ".join(migration._source_contract().split())

    assert "project-tool-result-binding-v3" not in legacy
    assert "feishu_binding_id" not in legacy
    assert "project-tool-result-binding-v3" in current
    assert "actor_role IN ('ADMIN', 'MEMBER')" in current


def _config(database_url: str) -> Config:
    config = Config(PROJECT_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _checks(inspector: object) -> str:
    return " ".join(
        " ".join(str(item["sqltext"]).lower().split())
        for item in inspector.get_check_constraints(  # type: ignore[attr-defined]
            "project_tool_result_bindings"
        )
    )


def test_0022_adds_strict_feishu_project_result_binding_evidence(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'feishu-project-bindings.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0021")
    engine = create_engine(database_url)
    try:
        before = {
            column["name"]: column
            for column in inspect(engine).get_columns("project_tool_result_bindings")
        }
        assert "feishu_binding_id" not in before
        assert before["actor_session_id"]["nullable"] is False

        command.upgrade(config, "head")

        inspector = inspect(engine)
        columns = {
            column["name"]: column
            for column in inspector.get_columns("project_tool_result_bindings")
        }
        assert columns["actor_session_id"]["nullable"] is True
        assert columns["feishu_binding_id"]["nullable"] is True
        foreign_keys = {
            (
                tuple(item["constrained_columns"]),
                item["referred_table"],
                tuple(item["referred_columns"]),
            )
            for item in inspector.get_foreign_keys("project_tool_result_bindings")
        }
        assert (("feishu_binding_id",), "feishu_bindings", ("id",)) in foreign_keys
        checks = _checks(inspector)
        assert "project-tool-result-binding-v3" in checks
        assert "'feishu'" in checks
        assert "actor_session_id is null" in checks
        assert "feishu_binding_id is not null" in checks
        assert "actor_role in ('admin', 'member')" in checks
        command.check(config)
    finally:
        engine.dispose()


def test_0022_rejects_v3_rows_outside_the_exact_feishu_contract(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'strict-feishu-contract.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0022")
    engine = create_engine(database_url)
    now = "2026-08-12 00:00:00+00:00"
    digest = "a" * 64
    try:
        with engine.begin() as connection:
            _seed_feishu_binding_dependencies(connection, now=now, digest=digest)

        invalid_rows = (
            {"result_id": "judge-result", "actor_role": "JUDGE"},
            {
                "result_id": "session-result",
                "actor_role": "MEMBER",
                "actor_session_id": "binding-session",
            },
            {
                "result_id": "missing-binding-result",
                "actor_role": "MEMBER",
                "feishu_binding_id": None,
            },
            {
                "result_id": "agent-field-result",
                "actor_role": "MEMBER",
                "plan_hash": digest,
            },
        )
        for row in invalid_rows:
            with engine.begin() as connection:
                _insert_tool_result(
                    connection,
                    result_id=row["result_id"],
                    now=now,
                    digest=digest,
                )
            with engine.begin() as connection, pytest.raises(IntegrityError):
                _insert_feishu_project_binding(
                    connection,
                    result_id=row["result_id"],
                    actor_role=row["actor_role"],
                    actor_session_id=row.get("actor_session_id"),
                    feishu_binding_id=row.get("feishu_binding_id", "feishu-binding"),
                    plan_hash=row.get("plan_hash"),
                    now=now,
                    digest=digest,
                )
    finally:
        engine.dispose()


def _seed_feishu_binding_dependencies(connection, *, now: str, digest: str) -> None:
    connection.execute(
        text(
            "INSERT INTO users "
            "(id,username,credential_hash,must_change_credential,role,status,"
            "created_at,updated_at) VALUES "
            "('binding-user','binding@example.test','hash',0,'ADMIN','ACTIVE',"
            ":now,:now)"
        ),
        {"now": now},
    )
    connection.execute(
        text(
            "INSERT INTO sessions "
            "(id,user_id,token_hash,status,created_at,expires_at,revoked_at) "
            "VALUES ('binding-session','binding-user',:digest,'ACTIVE',:now,:now,NULL)"
        ),
        {"digest": digest, "now": now},
    )
    connection.execute(
        text(
            "INSERT INTO projects "
            "(id,owner_user_id,name,status,created_at,updated_at) VALUES "
            "('binding-project','binding-user','binding','ACTIVE',:now,:now)"
        ),
        {"now": now},
    )
    connection.execute(
        text(
            "INSERT INTO feishu_bindings "
            "(id,project_id,chat_id,bitable_app_token,bitable_table_id,"
            "user_open_id_map_json,binding_version,status,created_at) VALUES "
            "('feishu-binding','binding-project','oc-approved',NULL,NULL,"
            "'{\"binding-user\":\"ou-approved\"}','feishu-binding-v1','ACTIVE',:now)"
        ),
        {"now": now},
    )


def _insert_tool_result(connection, *, result_id: str, now: str, digest: str) -> None:
    connection.execute(
        text(
            "INSERT INTO tool_results "
            "(id,run_id,agent_step_id,tool_name,tool_version,model_version,"
            "data_version,feature_version,input_hash,values_json,uncertainty_json,"
            "warnings_json,created_at) VALUES "
            "(:result_id,NULL,NULL,'predict_cycle_life','v1','model-v1','data-v1',"
            "'feature-v1',:digest,'{}',NULL,'[]',:now)"
        ),
        {"result_id": result_id, "digest": digest, "now": now},
    )


def _insert_feishu_project_binding(
    connection,
    *,
    result_id: str,
    actor_role: str,
    actor_session_id: str | None,
    feishu_binding_id: str | None,
    plan_hash: str | None,
    now: str,
    digest: str,
) -> None:
    connection.execute(
        text(
            "INSERT INTO project_tool_result_bindings "
            "(result_id,binding_schema_version,project_id,actor_user_id,"
            "actor_session_id,actor_role,invocation_source,feishu_binding_id,"
            "agent_run_id,agent_step_id,step_id,plan_hash,claim_token_sha256,"
            "claim_attempt,claim_lease_expires_at,execution_snapshot_sha256,"
            "dependency_evidence_sha256,approval_required,approval_request_id,"
            "approval_action_id,approval_evidence_sha256,tool_name,input_hash,"
            "result_sha256,binding_sha256,created_at) VALUES "
            "(:result_id,'project-tool-result-binding-v3','binding-project',"
            "'binding-user',:actor_session_id,:actor_role,'FEISHU',"
            ":feishu_binding_id,NULL,NULL,NULL,:plan_hash,NULL,NULL,NULL,NULL,NULL,"
            "NULL,NULL,NULL,NULL,'predict_cycle_life',:digest,:digest,:digest,:now)"
        ),
        {
            "result_id": result_id,
            "actor_session_id": actor_session_id,
            "actor_role": actor_role,
            "feishu_binding_id": feishu_binding_id,
            "plan_hash": plan_hash,
            "digest": digest,
            "now": now,
        },
    )


def test_0022_preserves_http_bindings_and_refuses_to_drop_feishu_evidence(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'feishu-project-binding-downgrade.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0021")
    engine = create_engine(database_url)
    now = "2026-08-12 00:00:00+00:00"
    digest = "a" * 64
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id,username,credential_hash,must_change_credential,role,status,"
                    "created_at,updated_at) VALUES "
                    "('binding-user','binding@example.test','hash',0,'ADMIN','ACTIVE',"
                    ":now,:now)"
                ),
                {"now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO sessions "
                    "(id,user_id,token_hash,status,created_at,expires_at,revoked_at) "
                    "VALUES ('binding-session','binding-user',:digest,'ACTIVE',:now,"
                    ":now,NULL)"
                ),
                {"digest": digest, "now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id,owner_user_id,name,status,created_at,updated_at) VALUES "
                    "('binding-project','binding-user','binding','ACTIVE',:now,:now)"
                ),
                {"now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO tool_results "
                    "(id,run_id,agent_step_id,tool_name,tool_version,model_version,"
                    "data_version,feature_version,input_hash,values_json,"
                    "uncertainty_json,warnings_json,created_at) VALUES "
                    "('http-result',NULL,NULL,'validate_battery_data','v1',NULL,'data-v1',"
                    "'feature-v1',:digest,'{}',NULL,'[]',:now)"
                ),
                {"digest": digest, "now": now},
            )
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
                    "('http-result','project-tool-result-binding-v1','binding-project',"
                    "'binding-user','binding-session','ADMIN','HTTP',NULL,NULL,NULL,NULL,"
                    "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'validate_battery_data',"
                    ":digest,:digest,:digest,:now)"
                ),
                {"digest": digest, "now": now},
            )

        command.upgrade(config, "head")
        with engine.connect() as connection:
            http_row = connection.execute(
                text(
                    "SELECT binding_schema_version,invocation_source,actor_session_id,"
                    "feishu_binding_id FROM project_tool_result_bindings "
                    "WHERE result_id='http-result'"
                )
            ).one()
        assert tuple(http_row) == (
            "project-tool-result-binding-v1",
            "HTTP",
            "binding-session",
            None,
        )

        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feishu_bindings "
                    "(id,project_id,chat_id,bitable_app_token,bitable_table_id,"
                    "user_open_id_map_json,binding_version,status,created_at) VALUES "
                    "('feishu-binding','binding-project','oc-approved',NULL,NULL,"
                    "'{\"binding-user\":\"ou-approved\"}','feishu-binding-v1',"
                    "'ACTIVE',:now)"
                ),
                {"now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO tool_results "
                    "(id,run_id,agent_step_id,tool_name,tool_version,model_version,"
                    "data_version,feature_version,input_hash,values_json,"
                    "uncertainty_json,warnings_json,created_at) VALUES "
                    "('feishu-result',NULL,NULL,'predict_cycle_life','v1','model-v1',"
                    "'data-v1','feature-v1',:digest,'{}',NULL,'[]',:now)"
                ),
                {"digest": digest, "now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO project_tool_result_bindings "
                    "(result_id,binding_schema_version,project_id,actor_user_id,"
                    "actor_session_id,actor_role,invocation_source,feishu_binding_id,"
                    "agent_run_id,agent_step_id,step_id,plan_hash,claim_token_sha256,"
                    "claim_attempt,claim_lease_expires_at,execution_snapshot_sha256,"
                    "dependency_evidence_sha256,approval_required,approval_request_id,"
                    "approval_action_id,approval_evidence_sha256,tool_name,input_hash,"
                    "result_sha256,binding_sha256,created_at) VALUES "
                    "('feishu-result','project-tool-result-binding-v3','binding-project',"
                    "'binding-user',NULL,'ADMIN','FEISHU','feishu-binding',NULL,NULL,NULL,"
                    "NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,'predict_cycle_life',"
                    ":digest,:digest,:digest,:now)"
                ),
                {"digest": digest, "now": now},
            )

        with pytest.raises(RuntimeError, match="Feishu evidence"):
            command.downgrade(config, "0021")
    finally:
        engine.dispose()


def test_0022_preserves_v2_agent_binding_across_upgrade_and_downgrade(
    tmp_path: Path,
) -> None:
    database_url = (
        "sqlite+pysqlite:///"
        f"{(tmp_path / 'v2-agent-binding-roundtrip.sqlite3').as_posix()}"
    )
    config = _config(database_url)
    command.upgrade(config, "0021")
    engine = create_engine(database_url)
    digest = "b" * 64
    created_at = "2026-08-12 00:00:00+00:00"
    lease = "2026-08-12 01:00:00+00:00"
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
                    "('v2-result','project-tool-result-binding-v2','v2-project',"
                    "'v2-user','v2-session','MEMBER','AGENT','v2-run','v2-step',"
                    "'predict',:digest,:digest,1,:lease,:digest,:digest,0,NULL,NULL,"
                    ":digest,'predict_cycle_life',:digest,:digest,:digest,:created_at)"
                ),
                {"digest": digest, "lease": lease, "created_at": created_at},
            )

        command.upgrade(config, "0022")
        with engine.connect() as connection:
            upgraded = connection.execute(
                text(
                    "SELECT binding_schema_version,invocation_source,actor_session_id,"
                    "agent_run_id,agent_step_id,step_id,feishu_binding_id "
                    "FROM project_tool_result_bindings WHERE result_id='v2-result'"
                )
            ).one()
        assert tuple(upgraded) == (
            "project-tool-result-binding-v2",
            "AGENT",
            "v2-session",
            "v2-run",
            "v2-step",
            "predict",
            None,
        )

        command.downgrade(config, "0021")
        inspector = inspect(engine)
        assert "feishu_binding_id" not in {
            column["name"]
            for column in inspector.get_columns("project_tool_result_bindings")
        }
        checks = _checks(inspector)
        assert "project-tool-result-binding-v2" in checks
        assert "project-tool-result-binding-v3" not in checks
        with engine.connect() as connection:
            downgraded = connection.execute(
                text(
                    "SELECT binding_schema_version,invocation_source,actor_session_id,"
                    "agent_run_id,agent_step_id,step_id FROM "
                    "project_tool_result_bindings WHERE result_id='v2-result'"
                )
            ).one()
        assert tuple(downgraded) == (
            "project-tool-result-binding-v2",
            "AGENT",
            "v2-session",
            "v2-run",
            "v2-step",
            "predict",
        )
    finally:
        engine.dispose()
