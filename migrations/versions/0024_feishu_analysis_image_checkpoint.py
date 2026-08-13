"""Add a generic audited analysis image checkpoint.

Revision ID: 0024
Revises: 0023
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0024"
down_revision: str | None = "0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("analysis_image_key", sa.String(length=200)))
        batch.add_column(
            sa.Column("analysis_image_renderer_version", sa.String(length=100))
        )
        batch.add_column(sa.Column("analysis_image_sha256", sa.String(length=64)))
        batch.create_check_constraint(
            "ck_feishu_event_receipt_analysis_image_provenance",
            "(analysis_image_key IS NULL AND "
            "analysis_image_renderer_version IS NULL AND "
            "analysis_image_sha256 IS NULL) OR "
            "(analysis_image_key IS NOT NULL AND "
            "analysis_image_renderer_version IS NOT NULL AND "
            "analysis_image_sha256 IS NOT NULL AND "
            "length(analysis_image_sha256) = 64)",
        )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts "
            "WHERE analysis_image_key IS NOT NULL OR "
            "analysis_image_renderer_version IS NOT NULL OR "
            "analysis_image_sha256 IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0024 downgrade would discard analysis image evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_constraint(
            "ck_feishu_event_receipt_analysis_image_provenance",
            type_="check",
        )
        batch.drop_column("analysis_image_sha256")
        batch.drop_column("analysis_image_renderer_version")
        batch.drop_column("analysis_image_key")
