"""Bind knowledge objects to size, media type and section metadata.

Revision ID: 0005
Revises: 0004
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add metadata required for verified MinIO reads and page-aware chunks."""

    connection = op.get_bind()
    document_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM knowledge_documents")
    ).scalar_one()
    chunk_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM knowledge_chunks")
    ).scalar_one()
    if document_count or chunk_count:
        raise RuntimeError(
            "0005 requires empty knowledge tables; export and re-ingest legacy "
            "documents with verified uploader and object metadata"
        )

    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.add_column(
            sa.Column("created_by_user_id", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "object_size_bytes",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "object_content_type",
                sa.String(length=200),
                nullable=False,
                server_default="application/octet-stream",
            )
        )
    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.alter_column("created_by_user_id", nullable=False)
        batch_op.alter_column("object_size_bytes", server_default=None)
        batch_op.alter_column("object_content_type", server_default=None)
        batch_op.create_foreign_key(
            "fk_knowledge_documents_created_by_user_id_users",
            "users",
            ["created_by_user_id"],
            ["id"],
        )

    with op.batch_alter_table("knowledge_chunks") as batch_op:
        batch_op.add_column(
            sa.Column(
                "text_size_bytes",
                sa.Integer(),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "text_content_type",
                sa.String(length=200),
                nullable=False,
                server_default="text/plain; charset=utf-8",
            )
        )
        batch_op.add_column(sa.Column("section_label", sa.String(length=500), nullable=True))
    with op.batch_alter_table("knowledge_chunks") as batch_op:
        batch_op.alter_column("text_size_bytes", server_default=None)
        batch_op.alter_column("text_content_type", server_default=None)


def downgrade() -> None:
    """Remove verified object metadata."""

    with op.batch_alter_table("knowledge_chunks") as batch_op:
        batch_op.drop_column("section_label")
        batch_op.drop_column("text_content_type")
        batch_op.drop_column("text_size_bytes")
    with op.batch_alter_table("knowledge_documents") as batch_op:
        batch_op.drop_constraint(
            "fk_knowledge_documents_created_by_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("object_content_type")
        batch_op.drop_column("object_size_bytes")
        batch_op.drop_column("created_by_user_id")
