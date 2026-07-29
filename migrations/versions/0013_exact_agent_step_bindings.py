"""Bind project Agent results to exact fenced step evidence.

Revision ID: 0013
Revises: 0012
"""

from collections.abc import Sequence
from typing import Literal

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STEP_HASH_COLUMNS = (
    "resolved_input_hash",
    "execution_snapshot_sha256",
    "dependency_evidence_sha256",
    "execution_claim_sha256",
)
_BINDING_AGENT_COLUMNS = (
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


def _batch_recreate_mode(
    dialect_name: str,
) -> Literal["always", "auto"]:
    """Recreate only where SQLite requires it; preserve referenced PostgreSQL tables."""

    return "always" if dialect_name == "sqlite" else "auto"


def upgrade() -> None:
    """Add frozen step snapshots and exact Agent result evidence."""

    bind = op.get_bind()
    recreate = _batch_recreate_mode(bind.dialect.name)
    legacy_agent_rows = bind.execute(
        sa.text(
            "SELECT count(*) FROM project_tool_result_bindings "
            "WHERE invocation_source = 'AGENT'"
        )
    ).scalar_one()
    if legacy_agent_rows:
        raise RuntimeError(
            "0013 cannot infer exact evidence for legacy AGENT v1 bindings; "
            "archive or explicitly replay them before upgrading"
        )

    with op.batch_alter_table("agent_steps", recreate=recreate) as batch:
        batch.add_column(sa.Column("resolved_input_json", sa.JSON(), nullable=True))
        for name in _STEP_HASH_COLUMNS:
            batch.add_column(sa.Column(name, sa.String(length=64), nullable=True))
            batch.create_check_constraint(
                f"ck_agent_step_{name}_length",
                f"{name} IS NULL OR length({name}) = 64",
            )

    with op.batch_alter_table("approval_requests", recreate=recreate) as batch:
        batch.add_column(sa.Column("agent_step_id", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("execution_snapshot_sha256", sa.String(length=64), nullable=True)
        )
        batch.create_foreign_key(
            "fk_approval_request_agent_step",
            "agent_steps",
            ["agent_step_id"],
            ["id"],
        )
        batch.create_check_constraint(
            "ck_approval_request_execution_snapshot_length",
            "execution_snapshot_sha256 IS NULL OR "
            "length(execution_snapshot_sha256) = 64",
        )

    with op.batch_alter_table(
        "project_tool_result_bindings", recreate=recreate
    ) as batch:
        batch.drop_constraint(
            "ck_project_tool_result_binding_schema_version",
            type_="check",
        )
        batch.add_column(sa.Column("agent_step_id", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("step_id", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("plan_hash", sa.String(length=64), nullable=True))
        batch.add_column(
            sa.Column("claim_token_sha256", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("claim_attempt", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("claim_lease_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("execution_snapshot_sha256", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("dependency_evidence_sha256", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("approval_required", sa.Boolean(), nullable=True))
        batch.add_column(
            sa.Column("approval_request_id", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("approval_action_id", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("approval_evidence_sha256", sa.String(length=64), nullable=True)
        )
        batch.create_foreign_key(
            "fk_project_binding_agent_step",
            "agent_steps",
            ["agent_step_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_project_binding_approval_request",
            "approval_requests",
            ["approval_request_id"],
            ["id"],
        )
        batch.create_foreign_key(
            "fk_project_binding_approval_action",
            "approval_actions",
            ["approval_action_id"],
            ["id"],
        )
        batch.create_unique_constraint(
            "uq_project_tool_result_binding_agent_step",
            ["agent_step_id"],
        )
        batch.create_unique_constraint(
            "uq_project_tool_result_binding_run_step",
            ["agent_run_id", "step_id"],
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_schema_version",
            "binding_schema_version IN "
            "('project-tool-result-binding-v1', 'project-tool-result-binding-v2')",
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_agent_hash_lengths",
            "(plan_hash IS NULL OR length(plan_hash) = 64) AND "
            "(claim_token_sha256 IS NULL OR length(claim_token_sha256) = 64) AND "
            "(execution_snapshot_sha256 IS NULL OR "
            "length(execution_snapshot_sha256) = 64) AND "
            "(dependency_evidence_sha256 IS NULL OR "
            "length(dependency_evidence_sha256) = 64) AND "
            "(approval_evidence_sha256 IS NULL OR "
            "length(approval_evidence_sha256) = 64)",
        )
        null_agent_fields = " AND ".join(
            f"{name} IS NULL" for name in ("agent_run_id", *_BINDING_AGENT_COLUMNS)
        )
        batch.create_check_constraint(
            "ck_project_tool_result_binding_exact_agent_contract",
            "(binding_schema_version = 'project-tool-result-binding-v1' AND "
            f"invocation_source = 'HTTP' AND {null_agent_fields}) OR "
            "(binding_schema_version = 'project-tool-result-binding-v2' AND "
            "invocation_source = 'AGENT' AND agent_run_id IS NOT NULL AND "
            "agent_step_id IS NOT NULL AND step_id IS NOT NULL AND "
            "plan_hash IS NOT NULL AND claim_token_sha256 IS NOT NULL AND "
            "claim_attempt IS NOT NULL AND claim_attempt > 0 AND "
            "claim_lease_expires_at IS NOT NULL AND "
            "execution_snapshot_sha256 IS NOT NULL AND "
            "dependency_evidence_sha256 IS NOT NULL AND "
            "approval_required IS NOT NULL AND approval_evidence_sha256 IS NOT NULL AND "
            "((approval_required IS FALSE AND approval_request_id IS NULL AND "
            "approval_action_id IS NULL) OR (approval_required IS TRUE AND "
            "approval_request_id IS NOT NULL AND approval_action_id IS NOT NULL)))",
        )

    op.create_index(
        "ix_project_tool_result_bindings_agent_step",
        "project_tool_result_bindings",
        ["agent_step_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove exact-step evidence columns."""

    bind = op.get_bind()
    recreate = _batch_recreate_mode(bind.dialect.name)
    v2_agent_rows = bind.execute(
        sa.text(
            "SELECT count(*) FROM project_tool_result_bindings "
            "WHERE binding_schema_version = 'project-tool-result-binding-v2'"
        )
    ).scalar_one()
    if v2_agent_rows:
        raise RuntimeError(
            "0013 downgrade would discard v2 Agent evidence; remove no audited rows "
            "and retain revision 0013"
        )

    op.drop_index(
        "ix_project_tool_result_bindings_agent_step",
        table_name="project_tool_result_bindings",
    )
    with op.batch_alter_table(
        "project_tool_result_bindings", recreate=recreate
    ) as batch:
        batch.drop_constraint(
            "ck_project_tool_result_binding_exact_agent_contract", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_agent_hash_lengths", type_="check"
        )
        batch.drop_constraint(
            "ck_project_tool_result_binding_schema_version", type_="check"
        )
        batch.drop_constraint(
            "uq_project_tool_result_binding_run_step", type_="unique"
        )
        batch.drop_constraint(
            "uq_project_tool_result_binding_agent_step", type_="unique"
        )
        batch.drop_constraint(
            "fk_project_binding_approval_action", type_="foreignkey"
        )
        batch.drop_constraint(
            "fk_project_binding_approval_request", type_="foreignkey"
        )
        batch.drop_constraint("fk_project_binding_agent_step", type_="foreignkey")
        for name in reversed(_BINDING_AGENT_COLUMNS):
            batch.drop_column(name)
        batch.create_check_constraint(
            "ck_project_tool_result_binding_schema_version",
            "binding_schema_version = 'project-tool-result-binding-v1'",
        )

    with op.batch_alter_table("approval_requests", recreate=recreate) as batch:
        batch.drop_constraint(
            "ck_approval_request_execution_snapshot_length", type_="check"
        )
        batch.drop_constraint("fk_approval_request_agent_step", type_="foreignkey")
        batch.drop_column("execution_snapshot_sha256")
        batch.drop_column("agent_step_id")

    with op.batch_alter_table("agent_steps", recreate=recreate) as batch:
        for name in reversed(_STEP_HASH_COLUMNS):
            batch.drop_constraint(f"ck_agent_step_{name}_length", type_="check")
            batch.drop_column(name)
        batch.drop_column("resolved_input_json")
