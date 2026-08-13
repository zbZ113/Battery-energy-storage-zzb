"""Persist per-stage project result slots for crash-safe Feishu recovery.

Revision ID: 0025
Revises: 0024
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0025"
down_revision: str | None = "0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("prepared_input_result_id", sa.String(length=64)))


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts "
            "WHERE prepared_input_result_id IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0025 downgrade would discard prepared input evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_column("prepared_input_result_id")
