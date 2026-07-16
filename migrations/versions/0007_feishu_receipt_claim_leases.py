"""Add ownership leases to Feishu event receipts.

Revision ID: 0007
Revises: 0006
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add attempt ownership and crash-recovery metadata."""

    with op.batch_alter_table("feishu_event_receipts") as batch_op:
        batch_op.add_column(sa.Column("claim_token", sa.String(length=64)))
        batch_op.add_column(
            sa.Column(
                "attempt_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True)))
        batch_op.add_column(sa.Column("failed_at", sa.DateTime(timezone=True)))


def downgrade() -> None:
    """Remove Feishu receipt lease metadata."""

    with op.batch_alter_table("feishu_event_receipts") as batch_op:
        batch_op.drop_column("failed_at")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("attempt_count")
        batch_op.drop_column("claim_token")
