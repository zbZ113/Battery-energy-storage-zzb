"""Add project-scoped bindings for verified canonical record batches.

Revision ID: 0011
Revises: 0010
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create opaque project bindings without changing content-addressed storage."""

    op.create_table(
        "record_batch_bindings",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("binding_schema_version", sa.String(length=64), nullable=False),
        sa.Column("content_batch_id", sa.String(length=78), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=64), nullable=False),
        sa.Column("source_manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("registration_sha256", sa.String(length=64), nullable=False),
        sa.Column("content_dataset_id", sa.String(length=200), nullable=False),
        sa.Column("dataset_schema_version", sa.String(length=100), nullable=False),
        sa.Column("cell_id", sa.String(length=200), nullable=False),
        sa.Column("cutoff_cycle", sa.Integer(), nullable=False),
        sa.Column("data_version", sa.String(length=100), nullable=False),
        sa.Column("split_version", sa.String(length=100), nullable=False),
        sa.Column("feature_version", sa.String(length=100), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "cutoff_cycle > 0",
            name="ck_record_batch_binding_cutoff_positive",
        ),
        sa.CheckConstraint(
            "length(source_manifest_sha256) = 64 AND "
            "length(registration_sha256) = 64",
            name="ck_record_batch_binding_sha256_lengths",
        ),
        sa.CheckConstraint(
            "binding_schema_version = 'record-batch-binding-v1'",
            name="ck_record_batch_binding_schema_version",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id",
            "content_batch_id",
            name="uq_record_batch_binding_dataset_content",
        ),
    )
    op.create_index(
        "ix_record_batch_bindings_project_dataset",
        "record_batch_bindings",
        ["project_id", "dataset_id"],
        unique=False,
    )


def downgrade() -> None:
    """Remove project bindings without touching verified content bytes."""

    op.drop_index(
        "ix_record_batch_bindings_project_dataset",
        table_name="record_batch_bindings",
    )
    op.drop_table("record_batch_bindings")
