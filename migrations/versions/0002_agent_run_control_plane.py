"""Add the durable Agent run control plane.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add idempotent Agent runs, dependency-aware steps, and an outbox."""

    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.add_column(sa.Column("created_by_user_id", sa.String(length=64), nullable=False))
        batch_op.add_column(
            sa.Column("idempotency_key_hash", sa.String(length=64), nullable=False)
        )
        batch_op.add_column(sa.Column("request_hash", sa.String(length=64), nullable=False))
        batch_op.add_column(sa.Column("plan_json", sa.JSON(), nullable=False))
        batch_op.add_column(
            sa.Column("execution_plan_hash", sa.String(length=64), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_agent_runs_created_by_user_id_users",
            "users",
            ["created_by_user_id"],
            ["id"],
        )
        batch_op.create_unique_constraint(
            "uq_agent_run_user_idempotency_key",
            ["created_by_user_id", "idempotency_key_hash"],
        )

    with op.batch_alter_table("agent_steps") as batch_op:
        batch_op.add_column(sa.Column("depends_on_json", sa.JSON(), nullable=False))
        batch_op.add_column(sa.Column("failure_policy", sa.String(length=32), nullable=False))

    with op.batch_alter_table("approval_requests") as batch_op:
        batch_op.create_unique_constraint(
            "uq_approval_request_run_step",
            ["run_id", "step_id"],
        )

    with op.batch_alter_table("approval_actions") as batch_op:
        batch_op.create_unique_constraint(
            "uq_approval_action_request_id",
            ["approval_request_id"],
        )

    op.create_table(
        "agent_run_dispatches",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("plan_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=200), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error_code", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", name="uq_agent_run_dispatch_run_id"),
    )
    op.create_index(
        "ix_agent_run_dispatches_status_created",
        "agent_run_dispatches",
        ["status", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove the Agent run control-plane additions."""

    op.drop_index(
        "ix_agent_run_dispatches_status_created",
        table_name="agent_run_dispatches",
    )
    op.drop_table("agent_run_dispatches")

    with op.batch_alter_table("approval_actions") as batch_op:
        batch_op.drop_constraint("uq_approval_action_request_id", type_="unique")

    with op.batch_alter_table("approval_requests") as batch_op:
        batch_op.drop_constraint("uq_approval_request_run_step", type_="unique")

    with op.batch_alter_table("agent_steps") as batch_op:
        batch_op.drop_column("failure_policy")
        batch_op.drop_column("depends_on_json")

    with op.batch_alter_table("agent_runs") as batch_op:
        batch_op.drop_constraint("uq_agent_run_user_idempotency_key", type_="unique")
        batch_op.drop_constraint(
            "fk_agent_runs_created_by_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("execution_plan_hash")
        batch_op.drop_column("plan_json")
        batch_op.drop_column("request_hash")
        batch_op.drop_column("idempotency_key_hash")
        batch_op.drop_column("created_by_user_id")
