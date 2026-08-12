"""Add Aily origin and idempotency metadata to shared analysis jobs.

Revision ID: 0021
Revises: 0020
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(
            sa.Column(
                "job_origin",
                sa.String(length=32),
                nullable=False,
                server_default="FEISHU",
            )
        )
        batch.add_column(sa.Column("job_request_sha256", sa.String(length=64)))
        batch.create_check_constraint(
            "ck_feishu_event_receipt_job_origin",
            "job_origin IN ('FEISHU', 'AILY')",
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_job_request_sha256_length",
            "job_request_sha256 IS NULL OR length(job_request_sha256) = 64",
        )
        batch.create_unique_constraint(
            "uq_feishu_event_receipt_job_request_sha256",
            ["job_request_sha256"],
        )


def downgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_constraint(
            "uq_feishu_event_receipt_job_request_sha256",
            type_="unique",
        )
        batch.drop_constraint(
            "ck_feishu_event_receipt_job_request_sha256_length",
            type_="check",
        )
        batch.drop_constraint(
            "ck_feishu_event_receipt_job_origin",
            type_="check",
        )
        batch.drop_column("job_request_sha256")
        batch.drop_column("job_origin")
