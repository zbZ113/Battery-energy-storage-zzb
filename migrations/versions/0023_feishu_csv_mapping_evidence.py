"""Persist sanitized CSV mapping evidence on the durable Feishu job.

Revision ID: 0023
Revises: 0022
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: str | None = "0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("csv_mapping_status", sa.String(length=32)))
        batch.add_column(sa.Column("csv_mapping_evidence_json", sa.JSON()))
        batch.add_column(
            sa.Column("csv_mapping_evidence_sha256", sa.String(length=64))
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_csv_mapping_status",
            "csv_mapping_status IS NULL OR csv_mapping_status IN ('MAPPED', 'REJECTED')",
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_csv_mapping_evidence_contract",
            "(csv_mapping_status IS NULL AND csv_mapping_evidence_json IS NULL AND "
            "csv_mapping_evidence_sha256 IS NULL) OR "
            "(csv_mapping_status IS NOT NULL AND csv_mapping_evidence_json IS NOT NULL "
            "AND csv_mapping_evidence_sha256 IS NOT NULL "
            "AND length(csv_mapping_evidence_sha256) = 64)",
        )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts "
            "WHERE csv_mapping_status IS NOT NULL OR "
            "csv_mapping_evidence_json IS NOT NULL OR "
            "csv_mapping_evidence_sha256 IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0023 downgrade would discard CSV mapping evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_constraint(
            "ck_feishu_event_receipt_csv_mapping_evidence_contract",
            type_="check",
        )
        batch.drop_constraint(
            "ck_feishu_event_receipt_csv_mapping_status",
            type_="check",
        )
        batch.drop_column("csv_mapping_evidence_sha256")
        batch.drop_column("csv_mapping_evidence_json")
        batch.drop_column("csv_mapping_status")
