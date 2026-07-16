"""Make knowledge uploads idempotent per uploader.

Revision ID: 0006
Revises: 0005
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add request fingerprints and an uploader-scoped idempotency key."""

    connection = op.get_bind()
    document_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM knowledge_documents")
    ).scalar_one()
    if document_count:
        raise RuntimeError(
            "0006 requires empty knowledge_documents; export and re-ingest legacy "
            "documents with a real Idempotency-Key"
        )

    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.add_column(
            sa.Column("idempotency_key_hash", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(sa.Column("request_hash", sa.String(length=64), nullable=True))
    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.alter_column("idempotency_key_hash", nullable=False)
        batch_op.alter_column("request_hash", nullable=False)
        batch_op.create_unique_constraint(
            "uq_knowledge_uploader_idempotency_key",
            ["created_by_user_id", "idempotency_key_hash"],
        )
        batch_op.drop_constraint("uq_knowledge_source_sha256", type_="unique")
        batch_op.create_unique_constraint(
            "uq_knowledge_project_source_sha256",
            ["project_id", "source_sha256"],
        )


def downgrade() -> None:
    """Remove knowledge upload idempotency fields."""

    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.drop_constraint(
            "uq_knowledge_project_source_sha256",
            type_="unique",
        )
        batch_op.create_unique_constraint(
            "uq_knowledge_source_sha256",
            ["source_sha256"],
        )
        batch_op.drop_constraint(
            "uq_knowledge_uploader_idempotency_key",
            type_="unique",
        )
        batch_op.drop_column("request_hash")
        batch_op.drop_column("idempotency_key_hash")
