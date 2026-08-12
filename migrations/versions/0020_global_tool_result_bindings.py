"""Add integrity bindings for process-shared global ToolResults.

Revision ID: 0020
Revises: 0019
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "global_tool_result_bindings",
        sa.Column("result_id", sa.String(length=64), nullable=False),
        sa.Column("binding_schema_version", sa.String(length=64), nullable=False),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("binding_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "binding_schema_version = 'global-tool-result-binding-v1'",
            name="ck_global_tool_result_binding_schema_version",
        ),
        sa.CheckConstraint(
            "length(result_sha256) = 64 AND length(binding_sha256) = 64",
            name="ck_global_tool_result_binding_hash_lengths",
        ),
        sa.ForeignKeyConstraint(
            ["result_id"],
            ["tool_results.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("result_id"),
    )


def downgrade() -> None:
    op.drop_table("global_tool_result_bindings")
