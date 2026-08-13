"""Bind project ToolResults to durable Feishu identities.

Revision ID: 0022
Revises: 0021
"""

from collections.abc import Sequence
from typing import Literal

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _batch_recreate_mode(dialect_name: str) -> Literal["always", "auto"]:
    return "always" if dialect_name == "sqlite" else "auto"


def _source_contract() -> str:
    agent_fields = (
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
    )
    null_agent_fields = " AND ".join(f"{name} IS NULL" for name in agent_fields)
    return (
        "(binding_schema_version = 'project-tool-result-binding-v1' AND "
        "invocation_source = 'HTTP' AND actor_session_id IS NOT NULL AND "
        "feishu_binding_id IS NULL AND agent_run_id IS NULL AND "
        f"{null_agent_fields}) OR "
        "(binding_schema_version = 'project-tool-result-binding-v2' AND "
        "invocation_source = 'AGENT' AND actor_session_id IS NOT NULL AND "
        "feishu_binding_id IS NULL AND agent_run_id IS NOT NULL AND "
        "agent_step_id IS NOT NULL AND step_id IS NOT NULL AND "
        "plan_hash IS NOT NULL AND claim_token_sha256 IS NOT NULL AND "
        "claim_attempt IS NOT NULL AND claim_attempt > 0 AND "
        "claim_lease_expires_at IS NOT NULL AND "
        "execution_snapshot_sha256 IS NOT NULL AND "
        "dependency_evidence_sha256 IS NOT NULL AND "
        "approval_required IS NOT NULL AND approval_evidence_sha256 IS NOT NULL AND "
        "((approval_required IS FALSE AND approval_request_id IS NULL AND "
        "approval_action_id IS NULL) OR (approval_required IS TRUE AND "
        "approval_request_id IS NOT NULL AND approval_action_id IS NOT NULL))) OR "
        "(binding_schema_version = 'project-tool-result-binding-v3' AND "
        "invocation_source = 'FEISHU' AND "
        "actor_role IN ('ADMIN', 'MEMBER') AND actor_session_id IS NULL AND "
        "feishu_binding_id IS NOT NULL AND agent_run_id IS NULL AND "
        f"{null_agent_fields})"
    )


def _legacy_source_contract() -> str:
    agent_fields = (
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
    )
    null_agent_fields = " AND ".join(f"{name} IS NULL" for name in agent_fields)
    return (
        "(binding_schema_version = 'project-tool-result-binding-v1' AND "
        "invocation_source = 'HTTP' AND actor_session_id IS NOT NULL AND "
        "agent_run_id IS NULL AND "
        f"{null_agent_fields}) OR "
        "(binding_schema_version = 'project-tool-result-binding-v2' AND "
        "invocation_source = 'AGENT' AND actor_session_id IS NOT NULL AND "
        "agent_run_id IS NOT NULL AND agent_step_id IS NOT NULL AND "
        "step_id IS NOT NULL AND plan_hash IS NOT NULL AND "
        "claim_token_sha256 IS NOT NULL AND claim_attempt IS NOT NULL AND "
        "claim_attempt > 0 AND claim_lease_expires_at IS NOT NULL AND "
        "execution_snapshot_sha256 IS NOT NULL AND "
        "dependency_evidence_sha256 IS NOT NULL AND "
        "approval_required IS NOT NULL AND approval_evidence_sha256 IS NOT NULL AND "
        "((approval_required IS FALSE AND approval_request_id IS NULL AND "
        "approval_action_id IS NULL) OR (approval_required IS TRUE AND "
        "approval_request_id IS NOT NULL AND approval_action_id IS NOT NULL)))"
    )


def upgrade() -> None:
    bind = op.get_bind()
    recreate = _batch_recreate_mode(bind.dialect.name)
    with op.batch_alter_table(
        "project_tool_result_bindings", recreate=recreate
    ) as batch:
        batch.drop_constraint(
            "ck_project_tool_result_binding_exact_agent_contract", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_invocation_source", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_schema_version", type_="check"
        )
        batch.alter_column(
            "actor_session_id",
            existing_type=sa.String(length=64),
            nullable=True,
        )
        batch.add_column(
            sa.Column("feishu_binding_id", sa.String(length=64), nullable=True)
        )
        batch.create_foreign_key(
            "fk_project_binding_feishu_binding",
            "feishu_bindings",
            ["feishu_binding_id"],
            ["id"],
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_schema_version",
            "binding_schema_version IN "
            "('project-tool-result-binding-v1', 'project-tool-result-binding-v2', "
            "'project-tool-result-binding-v3')",
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_invocation_source",
            "invocation_source IN ('HTTP', 'AGENT', 'FEISHU')",
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_exact_agent_contract",
            _source_contract(),
        )


def downgrade() -> None:
    bind = op.get_bind()
    feishu_rows = bind.execute(
        sa.text(
            "SELECT count(*) FROM project_tool_result_bindings "
            "WHERE binding_schema_version = 'project-tool-result-binding-v3' "
            "OR invocation_source = 'FEISHU' OR feishu_binding_id IS NOT NULL"
        )
    ).scalar_one()
    if feishu_rows:
        raise RuntimeError(
            "0022 downgrade would discard Feishu evidence; retain revision 0022"
        )

    recreate = _batch_recreate_mode(bind.dialect.name)
    with op.batch_alter_table(
        "project_tool_result_bindings", recreate=recreate
    ) as batch:
        batch.drop_constraint(
            "ck_project_tool_result_binding_exact_agent_contract", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_invocation_source", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_schema_version", type_="check"
        )
        batch.drop_constraint(
            "fk_project_binding_feishu_binding", type_="foreignkey"
        )
        batch.drop_column("feishu_binding_id")
        batch.alter_column(
            "actor_session_id",
            existing_type=sa.String(length=64),
            nullable=False,
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_schema_version",
            "binding_schema_version IN "
            "('project-tool-result-binding-v1', 'project-tool-result-binding-v2')",
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_invocation_source",
            "invocation_source IN ('HTTP', 'AGENT')",
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_exact_agent_contract",
            _legacy_source_contract(),
        )
