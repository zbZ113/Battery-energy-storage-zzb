"""Add replay checkpoints for external Feishu delivery references.

Revision ID: 0019
Revises: 0018
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: str | None = "0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("scenario_image_key", sa.String(length=200)))
        batch.add_column(sa.Column("result_card_message_id", sa.String(length=200)))
        batch.add_column(sa.Column("report_message_id", sa.String(length=200)))
        batch.add_column(sa.Column("report_card_message_id", sa.String(length=200)))


def downgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        for column in (
            "report_card_message_id",
            "report_message_id",
            "result_card_message_id",
            "scenario_image_key",
        ):
            batch.drop_column(column)
