"""Persist audited Feishu Bitable curve attachment checkpoints.

Revision ID: 0027
Revises: 0026
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: str | None = "0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(
            sa.Column("bitable_curve_file_token", sa.String(length=200))
        )
        batch.add_column(
            sa.Column("bitable_curve_source_result_id", sa.String(length=64))
        )
        batch.add_column(
            sa.Column("bitable_curve_renderer_version", sa.String(length=100))
        )
        batch.add_column(
            sa.Column("bitable_curve_sha256", sa.String(length=64))
        )
        batch.add_column(
            sa.Column("bitable_curve_template", sa.String(length=100))
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_bitable_curve_provenance",
            "(bitable_curve_file_token IS NULL AND "
            "bitable_curve_source_result_id IS NULL AND "
            "bitable_curve_renderer_version IS NULL AND "
            "bitable_curve_sha256 IS NULL AND "
            "bitable_curve_template IS NULL) OR "
            "(bitable_curve_file_token IS NOT NULL AND "
            "bitable_curve_source_result_id IS NOT NULL AND "
            "bitable_curve_renderer_version IS NOT NULL AND "
            "bitable_curve_sha256 IS NOT NULL AND "
            "bitable_curve_template IS NOT NULL AND "
            "length(bitable_curve_sha256) = 64 AND "
            "analysis_result_id IS NOT NULL AND "
            "bitable_curve_source_result_id = analysis_result_id)",
        )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts WHERE "
            "bitable_curve_file_token IS NOT NULL OR "
            "bitable_curve_source_result_id IS NOT NULL OR "
            "bitable_curve_renderer_version IS NOT NULL OR "
            "bitable_curve_sha256 IS NOT NULL OR "
            "bitable_curve_template IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0027 downgrade would discard Bitable curve evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_constraint(
            "ck_feishu_event_receipt_bitable_curve_provenance",
            type_="check",
        )
        batch.drop_column("bitable_curve_template")
        batch.drop_column("bitable_curve_sha256")
        batch.drop_column("bitable_curve_renderer_version")
        batch.drop_column("bitable_curve_source_result_id")
        batch.drop_column("bitable_curve_file_token")
