from sqlalchemy import JSON, String, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from quanxin_life.persistence.models import Base

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


def _unique_column_sets(table_name: str) -> set[tuple[str, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def test_metadata_contains_the_minimum_persistence_tables() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLES


def test_cell_splits_are_unique_per_dataset_and_cell() -> None:
    assert ("dataset_id", "cell_id") in _unique_column_sets("cell_splits")


def test_idempotency_and_evidence_constraints_are_declared() -> None:
    assert ("dataset_id", "sha256") in _unique_column_sets("dataset_files")
    assert ("created_by_user_id", "idempotency_key_hash") in _unique_column_sets(
        "agent_runs"
    )
    assert ("run_id",) in _unique_column_sets("agent_run_dispatches")
    assert ("run_id", "step_id") in _unique_column_sets("agent_steps")
    assert ("agent_step_id",) in _unique_column_sets("tool_results")
    assert ("run_id", "step_id") in _unique_column_sets("approval_requests")
    assert ("approval_request_id",) in _unique_column_sets("approval_actions")
    assert ("event_id",) in _unique_column_sets("feishu_event_receipts")


def test_agent_run_control_plane_columns_are_strictly_declared() -> None:
    agent_runs = Base.metadata.tables["agent_runs"]
    assert agent_runs.c.created_by_user_id.nullable is False
    assert {
        foreign_key.target_fullname
        for foreign_key in agent_runs.c.created_by_user_id.foreign_keys
    } == {"users.id"}
    for column_name in ("idempotency_key_hash", "request_hash"):
        column = agent_runs.c[column_name]
        assert isinstance(column.type, String)
        assert column.type.length == 64
        assert column.nullable is False
    assert isinstance(agent_runs.c.plan_json.type, JSON)
    assert agent_runs.c.plan_json.nullable is False
    assert isinstance(agent_runs.c.execution_plan_hash.type, String)
    assert agent_runs.c.execution_plan_hash.type.length == 64
    assert agent_runs.c.execution_plan_hash.nullable is True

    agent_steps = Base.metadata.tables["agent_steps"]
    assert isinstance(agent_steps.c.depends_on_json.type, JSON)
    assert agent_steps.c.depends_on_json.nullable is False
    assert isinstance(agent_steps.c.failure_policy.type, String)
    assert agent_steps.c.failure_policy.nullable is False
    assert agent_steps.c.attempts.nullable is False
    assert isinstance(agent_steps.c.claim_token.type, String)
    assert agent_steps.c.claim_token.type.length == 64
    assert agent_steps.c.claim_token.nullable is True
    assert agent_steps.c.lease_expires_at.nullable is True
    assert agent_steps.c.last_error_code.nullable is True


def test_agent_run_dispatch_is_a_durable_one_row_per_run_outbox() -> None:
    dispatches = Base.metadata.tables["agent_run_dispatches"]
    assert {
        foreign_key.target_fullname for foreign_key in dispatches.c.run_id.foreign_keys
    } == {"agent_runs.id"}
    assert dispatches.c.run_id.nullable is False
    assert isinstance(dispatches.c.plan_hash.type, String)
    assert dispatches.c.plan_hash.type.length == 64
    assert dispatches.c.plan_hash.nullable is False
    assert dispatches.c.status.nullable is False
    assert dispatches.c.task_id.nullable is True
    assert dispatches.c.attempts.nullable is False
    assert dispatches.c.last_error_code.nullable is True
    assert dispatches.c.created_at.nullable is False
    assert dispatches.c.updated_at.nullable is False


def test_structured_payloads_use_json_and_timestamps_require_timezones() -> None:
    tool_results = Base.metadata.tables["tool_results"]
    assert isinstance(tool_results.c.values_json.type, JSON)
    assert isinstance(tool_results.c.uncertainty_json.type, JSON)

    timestamp_columns = [
        column
        for table in Base.metadata.tables.values()
        for column in table.columns
        if column.name.endswith("_at")
    ]
    assert timestamp_columns
    assert all(getattr(column.type, "timezone", False) for column in timestamp_columns)


def test_schema_does_not_offer_columns_for_secrets_raw_files_or_full_prompts() -> None:
    forbidden_names = {
        "api_key",
        "password",
        "secret",
        "raw_content",
        "file_content",
        "full_prompt",
    }
    actual_names = {
        column.name for table in Base.metadata.tables.values() for column in table.columns
    }
    assert actual_names.isdisjoint(forbidden_names)


def test_users_store_only_a_credential_hash_and_first_login_flag() -> None:
    users = Base.metadata.tables["users"]

    assert "credential_hash" in users.c
    assert users.c.credential_hash.nullable is False
    assert "must_change_credential" in users.c
    assert users.c.must_change_credential.nullable is False


def test_all_tables_compile_for_the_postgresql_dialect() -> None:
    compiled = {
        table.name: str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in Base.metadata.sorted_tables
    }

    assert set(compiled) == EXPECTED_TABLES
    assert all(statement.startswith("\nCREATE TABLE") for statement in compiled.values())
