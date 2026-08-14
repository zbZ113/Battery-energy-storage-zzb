"""Persist reviewed recommendation rules and same-root result inputs.

Revision ID: 0028
Revises: 0027
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0028"
down_revision: str | None = "0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.add_column(sa.Column("recommendation_ruleset_id", sa.String(length=200)))
        batch.add_column(
            sa.Column("recommendation_ruleset_version", sa.String(length=200))
        )
        batch.add_column(
            sa.Column("recommendation_ruleset_sha256", sa.String(length=64))
        )
        batch.add_column(
            sa.Column("recommendation_upstream_result_ids_json", sa.JSON())
        )
        batch.add_column(
            sa.Column(
                "recommendation_upstream_result_ids_sha256",
                sa.String(length=64),
            )
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_recommendation_ruleset",
            "(recommendation_ruleset_id IS NULL AND "
            "recommendation_ruleset_version IS NULL AND "
            "recommendation_ruleset_sha256 IS NULL) OR "
            "(recommendation_ruleset_id IS NOT NULL AND "
            "recommendation_ruleset_version IS NOT NULL AND "
            "recommendation_ruleset_sha256 IS NOT NULL AND "
            "length(recommendation_ruleset_sha256) = 64)",
        )
        batch.create_check_constraint(
            "ck_feishu_event_receipt_recommendation_upstream_results",
            "(recommendation_upstream_result_ids_json IS NULL AND "
            "recommendation_upstream_result_ids_sha256 IS NULL) OR "
            "(recommendation_upstream_result_ids_json IS NOT NULL AND "
            "recommendation_upstream_result_ids_sha256 IS NOT NULL AND "
            "length(recommendation_upstream_result_ids_sha256) = 64)",
        )


def downgrade() -> None:
    bind = op.get_bind()
    count = bind.execute(
        sa.text(
            "SELECT count(*) FROM feishu_event_receipts WHERE "
            "recommendation_ruleset_id IS NOT NULL OR "
            "recommendation_ruleset_version IS NOT NULL OR "
            "recommendation_ruleset_sha256 IS NOT NULL OR "
            "recommendation_upstream_result_ids_json IS NOT NULL OR "
            "recommendation_upstream_result_ids_sha256 IS NOT NULL"
        )
    ).scalar_one()
    if count:
        raise RuntimeError("0028 downgrade would discard recommendation evidence")
    with op.batch_alter_table("feishu_event_receipts") as batch:
        batch.drop_constraint(
            "ck_feishu_event_receipt_recommendation_upstream_results",
            type_="check",
        )
        batch.drop_constraint(
            "ck_feishu_event_receipt_recommendation_ruleset",
            type_="check",
        )
        batch.drop_column("recommendation_upstream_result_ids_sha256")
        batch.drop_column("recommendation_upstream_result_ids_json")
        batch.drop_column("recommendation_ruleset_sha256")
        batch.drop_column("recommendation_ruleset_version")
        batch.drop_column("recommendation_ruleset_id")
