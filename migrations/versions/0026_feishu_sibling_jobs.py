"""Persist Feishu sibling-job lineage and reviewed default-scenario provenance.

Revision ID: 0026
Revises: 0025
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: str | None = "0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("source_job_id", sa.String(length=64)))
        batch.add_column(
            sa.Column("default_scenario_profile_id", sa.String(length=200))
        )
        batch.add_column(
            sa.Column("default_scenario_profile_version", sa.String(length=100))
        )
        batch.add_column(
            sa.Column("default_scenario_profile_sha256", sa.String(length=64))
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_source_job_distinct",
            "source_job_id IS NULL OR source_job_id <> job_id",
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_default_scenario_profile",
            "(default_scenario_profile_id IS NULL AND "
            "default_scenario_profile_version IS NULL AND "
            "default_scenario_profile_sha256 IS NULL) OR "
            "(default_scenario_profile_id IS NOT NULL AND "
            "default_scenario_profile_version IS NOT NULL AND "
            "default_scenario_profile_sha256 IS NOT NULL AND "
            "length(default_scenario_profile_sha256) = 64)",
        )
        batch.create_index(
            "ix_feishu_receipts_source_job_id",
            ["source_job_id", "task_type"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts "
            "WHERE source_job_id IS NOT NULL OR "
            "default_scenario_profile_id IS NOT NULL OR "
            "default_scenario_profile_version IS NOT NULL OR "
            "default_scenario_profile_sha256 IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0026 downgrade would discard sibling job evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_index("ix_feishu_receipts_source_job_id")
        batch.drop_constraint(
            "ck_feishu_event_receipt_default_scenario_profile",
            type_="check",
        )
        batch.drop_constraint(
            "ck_feishu_event_receipt_source_job_distinct",
            type_="check",
        )
        batch.drop_column("default_scenario_profile_sha256")
        batch.drop_column("default_scenario_profile_version")
        batch.drop_column("default_scenario_profile_id")
        batch.drop_column("source_job_id")
