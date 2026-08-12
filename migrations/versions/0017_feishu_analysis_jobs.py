"""Extend Feishu receipts with durable sanitized analysis-job state.

Revision ID: 0017
Revises: 0016
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("job_id", sa.String(length=64)))
        batch.add_column(sa.Column("job_status", sa.String(length=32)))
        batch.add_column(sa.Column("job_stage", sa.String(length=32)))
        batch.add_column(sa.Column("task_type", sa.String(length=100)))
        batch.add_column(sa.Column("run_id", sa.String(length=64)))
        batch.add_column(sa.Column("message_id", sa.String(length=200)))
        batch.add_column(sa.Column("file_key", sa.String(length=200)))
        batch.add_column(sa.Column("file_name", sa.String(length=255)))
        batch.add_column(sa.Column("chat_id", sa.String(length=200)))
        batch.add_column(sa.Column("sender_id", sa.String(length=200)))
        batch.add_column(sa.Column("receive_id_type", sa.String(length=32)))
        batch.add_column(sa.Column("event_time", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("job_claim_token", sa.String(length=64)))
        batch.add_column(
            sa.Column(
                "job_attempt_count",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch.add_column(sa.Column("job_lease_expires_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("job_task_id", sa.String(length=200)))
        batch.add_column(sa.Column("job_last_error_code", sa.String(length=100)))
        batch.add_column(sa.Column("record_batch_id", sa.String(length=100)))
        batch.add_column(sa.Column("cell_reference", sa.String(length=200)))
        batch.add_column(sa.Column("input_file_sha256", sa.String(length=64)))
        batch.add_column(sa.Column("validation_result_id", sa.String(length=64)))
        batch.add_column(sa.Column("analysis_result_id", sa.String(length=64)))
        batch.add_column(sa.Column("report_result_id", sa.String(length=64)))
        batch.add_column(sa.Column("report_file_key", sa.String(length=200)))
        batch.add_column(sa.Column("bitable_record_id", sa.String(length=200)))
        batch.add_column(sa.Column("job_created_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("job_updated_at", sa.DateTime(timezone=True)))
        batch.add_column(sa.Column("job_completed_at", sa.DateTime(timezone=True)))
        batch.create_unique_constraint("uq_feishu_event_receipt_job_id", ["job_id"])
        batch.create_index(
            "ix_feishu_receipts_job_status_updated",
            ["job_status", "job_updated_at"],
        )


def downgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_index("ix_feishu_receipts_job_status_updated")
        batch.drop_constraint("uq_feishu_event_receipt_job_id", type_="unique")
        for column in (
            "job_completed_at",
            "job_updated_at",
            "job_created_at",
            "bitable_record_id",
            "report_file_key",
            "report_result_id",
            "analysis_result_id",
            "validation_result_id",
            "input_file_sha256",
            "cell_reference",
            "record_batch_id",
            "job_last_error_code",
            "job_task_id",
            "job_lease_expires_at",
            "job_attempt_count",
            "job_claim_token",
            "event_time",
            "receive_id_type",
            "sender_id",
            "chat_id",
            "file_name",
            "file_key",
            "message_id",
            "run_id",
            "task_type",
            "job_stage",
            "job_status",
            "job_id",
        ):
            batch.drop_column(column)
