"""Persist project ownership and integrity evidence for ToolResult records.

Revision ID: 0012
Revises: 0011
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create append-only project bindings over persisted ToolResult rows."""

    op.create_table(
        "project_tool_result_bindings",
        sa.Column("result_id", sa.String(length=64), nullable=False),
        sa.Column("binding_schema_version", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.String(length=64), nullable=False),
        sa.Column("actor_session_id", sa.String(length=64), nullable=False),
        sa.Column("actor_role", sa.String(length=32), nullable=False),
        sa.Column("invocation_source", sa.String(length=32), nullable=False),
        sa.Column("agent_run_id", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=200), nullable=False),
        sa.Column("input_hash", sa.String(length=64), nullable=False),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("binding_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "binding_schema_version = 'project-tool-result-binding-v1'",
            name="ck_project_tool_result_binding_schema_version",
        ),
        sa.CheckConstraint(
            "actor_role IN ('ADMIN', 'MEMBER', 'JUDGE')",
            name="ck_project_tool_result_binding_actor_role",
        ),
        sa.CheckConstraint(
            "invocation_source IN ('HTTP', 'AGENT')",
            name="ck_project_tool_result_binding_invocation_source",
        ),
        sa.CheckConstraint(
            "length(input_hash) = 64 AND length(result_sha256) = 64 AND "
            "length(binding_sha256) = 64",
            name="ck_project_tool_result_binding_hash_lengths",
        ),
        sa.ForeignKeyConstraint(["result_id"], ["tool_results.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["actor_session_id"], ["sessions.id"]),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("result_id"),
    )
    op.create_index(
        "ix_project_tool_result_bindings_project_created",
        "project_tool_result_bindings",
        ["project_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove project bindings while leaving their ToolResult rows intact."""

    op.drop_index(
        "ix_project_tool_result_bindings_project_created",
        table_name="project_tool_result_bindings",
    )
    op.drop_table("project_tool_result_bindings")
