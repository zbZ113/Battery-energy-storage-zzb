"""Add recoverable execution leases to Agent steps.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Support safe claim, retry and recovery under at-least-once delivery."""

    with op.batch_alter_table("agent_steps") as batch_op:
        batch_op.add_column(
            sa.Column(
                "attempts",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("claim_token", sa.String(length=64), nullable=True))
        batch_op.add_column(
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("last_error_code", sa.String(length=100), nullable=True)
        )
    with op.batch_alter_table("agent_steps") as batch_op:
        batch_op.alter_column("attempts", server_default=None)


def downgrade() -> None:
    """Remove step execution lease state."""

    with op.batch_alter_table("agent_steps") as batch_op:
        batch_op.drop_column("last_error_code")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("claim_token")
        batch_op.drop_column("attempts")
