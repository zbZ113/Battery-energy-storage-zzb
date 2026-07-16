"""Require at most one durable ToolResult for each Agent step.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Prevent duplicate successful evidence under at-least-once task delivery."""

    with op.batch_alter_table("tool_results") as batch_op:
        batch_op.create_unique_constraint(
            "uq_tool_result_agent_step_id",
            ["agent_step_id"],
        )


def downgrade() -> None:
    """Remove the Agent-step evidence uniqueness constraint."""

    with op.batch_alter_table("tool_results") as batch_op:
        batch_op.drop_constraint("uq_tool_result_agent_step_id", type_="unique")
