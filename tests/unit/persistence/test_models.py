from sqlalchemy import JSON, CheckConstraint, String, UniqueConstraint
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
    "record_batch_bindings",
    "project_tool_result_bindings",
    "cell_splits",
    "agent_runs",
    "agent_run_dispatches",
    "agent_steps",
    "agent_events",
    "tool_results",
    "provenance_records",
    "global_tool_result_bindings",
    "model_artifacts",
    "model_manifests",
    "model_route_activation_events",
    "model_route_activation_stream_heads",
    "calibration_cohorts",
    "advanced_calibration_materializations",
    "advanced_calibration_sample_bindings",
    "decision_policies",
    "approval_requests",
    "approval_actions",
    "reports",
    "report_exports",
    "knowledge_documents",
    "knowledge_chunks",
    "feishu_bindings",
    "feishu_event_receipts",
    "feishu_scenario_contexts",
    "experiment_suites",
    "experiment_runs",
}


def _unique_column_sets(table_name: str) -> set[tuple[str, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def _named_check_constraints(table_name: str) -> dict[str, str]:
    table = Base.metadata.tables[table_name]
    return {
        str(constraint.name): " ".join(str(constraint.sqltext).split())
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }


def _named_index_column_sets(table_name: str) -> dict[str, tuple[str, ...]]:
    table = Base.metadata.tables[table_name]
    return {
        index.name: tuple(column.name for column in index.columns)
        for index in table.indexes
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
    assert ("job_request_sha256",) in _unique_column_sets("feishu_event_receipts")
    assert ("created_by_user_id", "idempotency_key_hash") in _unique_column_sets(
        "knowledge_documents"
    )
    assert ("project_id", "import_id") in _unique_column_sets("experiment_suites")
    assert ("suite_id", "run_id") in _unique_column_sets("experiment_runs")
    assert (
        "suite_id",
        "cutoff_cycle",
        "model_name",
        "seed",
    ) in _unique_column_sets("experiment_runs")


def test_feishu_analysis_job_origin_matches_the_0021_migration_contract() -> None:
    receipts = Base.metadata.tables["feishu_event_receipts"]
    checks = _named_check_constraints("feishu_event_receipts")

    assert receipts.c.job_origin.nullable is False
    assert receipts.c.job_origin.server_default is not None
    assert str(receipts.c.job_origin.server_default.arg) == "FEISHU"
    assert checks["ck_feishu_event_receipt_job_origin"] == (
        "job_origin IN ('FEISHU', 'AILY')"
    )


def test_model_route_activation_ledger_declares_stream_identity_and_indexes() -> None:
    unique_columns = _unique_column_sets("model_route_activation_events")
    assert (
        "project_id",
        "task",
        "cutoff_cycle",
        "route_role",
        "stream_sequence",
    ) in unique_columns
    assert ("project_id", "idempotency_key_sha256") in unique_columns
    assert ("event_sha256",) in unique_columns

    indexes = _named_index_column_sets("model_route_activation_events")
    assert indexes["ix_model_route_activation_stream"] == (
        "project_id",
        "task",
        "cutoff_cycle",
        "route_role",
        "stream_sequence",
    )
    assert indexes["ix_model_route_activation_artifact"] == ("artifact_id",)


def test_model_route_activation_ledger_declares_domain_checks() -> None:
    checks = _named_check_constraints("model_route_activation_events")

    assert checks["ck_model_route_activation_positive_coordinates"] == (
        "stream_sequence > 0 AND cutoff_cycle > 0"
    )
    assert checks["ck_model_route_activation_decision_type"] == (
        "decision_type IN ('ACTIVATE', 'ROLLBACK')"
    )
    assert checks["ck_model_route_activation_rollback_target"] == (
        "(decision_type = 'ACTIVATE' AND rollback_target_event_id IS NULL) "
        "OR (decision_type = 'ROLLBACK' AND rollback_target_event_id IS NOT NULL)"
    )
    assert checks["ck_model_route_activation_task_role"] == (
        "(task = 'RUL' AND route_role IN ('DEFAULT', 'POINT_ACCURACY', 'COVERAGE')) "
        "OR (task = 'SOH' AND route_role IN ('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))"
    )


def test_model_route_activation_ledger_is_append_only_evidence() -> None:
    events = Base.metadata.tables["model_route_activation_events"]

    foreign_keys = {
        column.name: {
            (foreign_key.target_fullname, foreign_key.ondelete)
            for foreign_key in column.foreign_keys
        }
        for column in events.columns
        if column.foreign_keys
    }
    assert foreign_keys == {
        "project_id": {("projects.id", None)},
        "artifact_id": {("model_artifacts.id", None)},
        "rollback_target_event_id": {
            ("model_route_activation_events.id", None)
        },
        "actor_user_id": {("users.id", None)},
    }
    assert events.c.rollback_target_event_id.nullable is True

    required_hash_columns = {
        "artifact_sha256",
        "manifest_sha256",
        "deployment_bundle_manifest_sha256",
        "route_provenance_sha256",
        "previous_event_sha256",
        "event_sha256",
        "idempotency_key_sha256",
        "request_sha256",
    }
    for column_name in required_hash_columns:
        column = events.c[column_name]
        assert isinstance(column.type, String)
        assert column.type.length == 64
        assert column.nullable is False

    assert all(
        column.nullable is False
        for column in events.columns
        if column.name != "rollback_target_event_id"
    )
    assert events.c.created_at.type.timezone is True
    assert {
        "updated_at",
        "deleted_at",
        "is_active",
        "status",
        "current_artifact_id",
    }.isdisjoint(events.c.keys())


def test_model_route_stream_head_is_only_an_integrity_anchor() -> None:
    heads = Base.metadata.tables["model_route_activation_stream_heads"]

    assert tuple(column.name for column in heads.primary_key.columns) == (
        "project_id",
        "task",
        "cutoff_cycle",
        "route_role",
    )
    assert {
        foreign_key.target_fullname
        for foreign_key in heads.c.head_event_id.foreign_keys
    } == {"model_route_activation_events.id"}
    assert heads.c.head_sequence.nullable is False
    assert isinstance(heads.c.head_event_sha256.type, String)
    assert heads.c.head_event_sha256.type.length == 64
    assert heads.c.head_event_sha256.nullable is False
    assert {
        "artifact_id",
        "artifact_sha256",
        "manifest_sha256",
        "model_version",
        "status",
        "current_artifact_id",
    }.isdisjoint(heads.c.keys())


def test_advanced_calibration_materialization_declares_exact_route_identity() -> None:
    materializations = Base.metadata.tables["advanced_calibration_materializations"]

    assert (
        "project_id",
        "task",
        "cutoff_cycle",
        "route_role",
        "decision_event_id",
        "source_registration_id",
    ) in _unique_column_sets("advanced_calibration_materializations")
    assert ("project_id", "idempotency_key_sha256") in _unique_column_sets(
        "advanced_calibration_materializations"
    )
    assert _named_index_column_sets("advanced_calibration_materializations")[
        "ix_advanced_calibration_project_status"
    ] == ("project_id", "status")

    required_hashes = {
        "artifact_manifest_sha256",
        "normalization_statistics_sha256",
        "ledger_head_sha256",
        "source_identity_sha256",
        "sample_manifest_sha256",
        "idempotency_key_sha256",
        "request_sha256",
        "claim_token_sha256",
    }
    for column_name in required_hashes:
        column = materializations.c[column_name]
        assert isinstance(column.type, String)
        assert column.type.length == 64
    assert materializations.c.sample_manifest_sha256.nullable is True
    assert materializations.c.claim_token_sha256.nullable is True
    assert materializations.c.claim_attempt.nullable is False
    assert materializations.c.claim_lease_expires_at.nullable is True
    assert materializations.c.created_by_session_id.nullable is False
    assert materializations.c.created_by_role.nullable is False

    foreign_keys = {
        column.name: {
            foreign_key.target_fullname for foreign_key in column.foreign_keys
        }
        for column in materializations.columns
        if column.foreign_keys
    }
    assert foreign_keys == {
        "project_id": {"projects.id"},
        "artifact_id": {"model_artifacts.id"},
        "decision_event_id": {"model_route_activation_events.id"},
        "created_by_user_id": {"users.id"},
        "created_by_session_id": {"sessions.id"},
    }


def test_advanced_calibration_materialization_declares_state_checks() -> None:
    checks = _named_check_constraints("advanced_calibration_materializations")

    assert checks["ck_advanced_calibration_coordinates"] == (
        "cutoff_cycle IN (20, 50, 100, 150) AND "
        "ledger_sequence_number > 0 AND sample_count >= 0"
    )
    assert checks["ck_advanced_calibration_status"] == (
        "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED', 'STALE')"
    )
    assert checks["ck_advanced_calibration_task_role"] == (
        "(task = 'RUL' AND ((cutoff_cycle = 20 AND route_role = 'DEFAULT') OR "
        "(cutoff_cycle IN (50, 100, 150) AND route_role = 'COVERAGE'))) OR "
        "(task = 'SOH' AND route_role IN ('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))"
    )
    assert checks["ck_advanced_calibration_state_payload"] == (
        "(status = 'PENDING' AND started_at IS NULL AND completed_at IS NULL AND "
        "sample_count = 0 AND sample_manifest_sha256 IS NULL AND failure_code IS NULL "
        "AND claim_token_sha256 IS NULL AND claim_attempt = 0 AND "
        "claim_lease_expires_at IS NULL) "
        "OR (status = 'RUNNING' AND started_at IS NOT NULL AND completed_at IS NULL "
        "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
        "failure_code IS NULL AND claim_token_sha256 IS NOT NULL AND "
        "claim_attempt > 0 AND claim_lease_expires_at IS NOT NULL) OR "
        "(status = 'READY' AND started_at IS NOT NULL AND "
        "completed_at IS NOT NULL AND sample_count > 0 AND "
        "sample_manifest_sha256 IS NOT NULL AND failure_code IS NULL AND "
        "claim_token_sha256 IS NULL AND claim_attempt > 0 AND "
        "claim_lease_expires_at IS NULL) OR "
        "(status = 'FAILED' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
        "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
        "failure_code IS NOT NULL AND claim_token_sha256 IS NULL AND "
        "claim_attempt > 0 AND claim_lease_expires_at IS NULL) OR "
        "(status = 'STALE' AND started_at IS NOT NULL "
        "AND completed_at IS NOT NULL AND sample_count > 0 AND "
        "sample_manifest_sha256 IS NOT NULL AND failure_code IS NOT NULL AND "
        "claim_token_sha256 IS NULL AND claim_attempt > 0 AND "
        "claim_lease_expires_at IS NULL)"
    )
    assert checks["ck_advanced_calibration_admin_actor"] == (
        "created_by_role = 'ADMIN'"
    )


def test_advanced_calibration_sample_bindings_are_ordered_and_unique() -> None:
    bindings = Base.metadata.tables["advanced_calibration_sample_bindings"]

    unique_columns = _unique_column_sets("advanced_calibration_sample_bindings")
    assert ("materialization_id", "ordinal") in unique_columns
    assert ("materialization_id", "cell_id") in unique_columns
    assert ("materialization_id", "result_id") in unique_columns
    assert _named_check_constraints("advanced_calibration_sample_bindings") == {
        "ck_advanced_calibration_sample_ordinal": "ordinal >= 0",
        "ck_advanced_calibration_sample_sha_length": "length(sample_sha256) = 64",
    }
    assert {
        foreign_key.target_fullname
        for foreign_key in bindings.c.materialization_id.foreign_keys
    } == {"advanced_calibration_materializations.id"}
    assert {
        foreign_key.target_fullname
        for foreign_key in bindings.c.result_id.foreign_keys
    } == {"tool_results.id"}


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
