"""Add immutable completed-run report export metadata.

Revision ID: 0016
Revises: 0015
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("reports") as batch:
        batch.add_column(sa.Column("failure_code", sa.String(length=100), nullable=True))
        batch.add_column(
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.add_column(
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.create_unique_constraint("uq_reports_run_id", ["run_id"])
        batch.create_check_constraint(
            "ck_reports_status",
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED')",
        )
    op.execute("UPDATE reports SET updated_at = created_at WHERE updated_at IS NULL")
    with op.batch_alter_table("reports") as batch:
        batch.alter_column("updated_at", nullable=False)

    with op.batch_alter_table("report_exports") as batch:
        batch.add_column(sa.Column("filename", sa.String(length=255), nullable=True))
        batch.add_column(sa.Column("media_type", sa.String(length=200), nullable=True))
        batch.add_column(sa.Column("size_bytes", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("failure_code", sa.String(length=100), nullable=True))
        batch.add_column(
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch.create_unique_constraint(
            "uq_report_export_format", ["report_id", "export_format"]
        )
        batch.create_check_constraint(
            "ck_report_exports_status",
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED', 'EXPIRED')",
        )


def downgrade() -> None:
    with op.batch_alter_table("report_exports") as batch:
        batch.drop_constraint("ck_report_exports_status", type_="check")
        batch.drop_constraint("uq_report_export_format", type_="unique")
        batch.drop_column("expires_at")
        batch.drop_column("failure_code")
        batch.drop_column("size_bytes")
        batch.drop_column("media_type")
        batch.drop_column("filename")
    with op.batch_alter_table("reports") as batch:
        batch.drop_constraint("ck_reports_status", type_="check")
        batch.drop_constraint("uq_reports_run_id", type_="unique")
        batch.drop_column("completed_at")
        batch.drop_column("updated_at")
        batch.drop_column("failure_code")
