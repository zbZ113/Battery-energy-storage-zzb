"""Enforce one proactive Feishu-derived task of each type per root upload.

Revision ID: 0029
Revises: 0028
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0029"
down_revision: str | None = "0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_feishu_receipts_proactive_source_task"
_PREDICATE = (
    "source_job_id IS NOT NULL AND job_origin = 'FEISHU' AND "
    "event_type = 'feishu.analysis_job.derived_v1'"
)


def upgrade() -> None:
    duplicate = op.get_bind().execute(
        sa.text(
            "SELECT source_job_id, task_type FROM feishu_event_receipts WHERE "
            + _PREDICATE
            + " GROUP BY source_job_id, task_type HAVING count(*) > 1 LIMIT 1"
        )
    ).first()
    if duplicate is not None:
        raise RuntimeError(
            "0029 cannot enforce proactive sibling uniqueness while duplicates exist"
        )
    op.create_index(
        _INDEX_NAME,
        "feishu_event_receipts",
        ["source_job_id", "task_type"],
        unique=True,
        postgresql_where=sa.text(_PREDICATE),
        sqlite_where=sa.text(_PREDICATE),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="feishu_event_receipts")
