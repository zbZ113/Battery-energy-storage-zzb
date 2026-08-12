"""Add durable scenario contexts referenced by Feishu analysis jobs.

Revision ID: 0018
Revises: 0017
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: str | None = "0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "feishu_scenario_contexts",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("schema_version", sa.String(length=100), nullable=False),
        sa.Column("task_type", sa.String(length=100), nullable=False),
        sa.Column("data_batch_id", sa.String(length=200), nullable=False),
        sa.Column("route_id", sa.String(length=200), nullable=False),
        sa.Column("verified_context_json", sa.JSON(), nullable=False),
        sa.Column("analysis_input_json", sa.JSON(), nullable=False),
        sa.Column("input_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_by_reference", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_feishu_scenario_contexts_task_created",
        "feishu_scenario_contexts",
        ["task_type", "created_at"],
    )
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("scenario_context_id", sa.String(length=64)))
        batch.create_index(
            "ix_feishu_receipts_scenario_context_id",
            ["scenario_context_id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_index("ix_feishu_receipts_scenario_context_id")
        batch.drop_column("scenario_context_id")
    op.drop_index(
        "ix_feishu_scenario_contexts_task_created",
        table_name="feishu_scenario_contexts",
    )
    op.drop_table("feishu_scenario_contexts")
