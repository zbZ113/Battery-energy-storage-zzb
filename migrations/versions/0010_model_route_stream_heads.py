"""Add integrity anchors for Advanced model route event streams.

Revision ID: 0010
Revises: 0009
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create and backfill route stream heads without storing active artifacts."""

    op.create_table(
        "model_route_activation_stream_heads",
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("cutoff_cycle", sa.Integer(), nullable=False),
        sa.Column("route_role", sa.String(length=32), nullable=False),
        sa.Column("head_event_id", sa.String(length=64), nullable=False),
        sa.Column("head_sequence", sa.Integer(), nullable=False),
        sa.Column("head_event_sha256", sa.String(length=64), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "head_sequence > 0 AND cutoff_cycle > 0",
            name="ck_model_route_stream_head_positive_coordinates",
        ),
        sa.CheckConstraint(
            "(task = 'RUL' AND route_role IN "
            "('DEFAULT', 'POINT_ACCURACY', 'COVERAGE')) OR "
            "(task = 'SOH' AND route_role IN "
            "('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_model_route_stream_head_task_role",
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(
            ["head_event_id"],
            ["model_route_activation_events.id"],
        ),
        sa.PrimaryKeyConstraint(
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            name="pk_model_route_activation_stream_heads",
        ),
    )
    op.execute(
        sa.text(
            "INSERT INTO model_route_activation_stream_heads "
            "(project_id, task, cutoff_cycle, route_role, head_event_id, "
            "head_sequence, head_event_sha256, updated_at) "
            "SELECT event.project_id, event.task, event.cutoff_cycle, "
            "event.route_role, event.id, event.stream_sequence, "
            "event.event_sha256, event.created_at "
            "FROM model_route_activation_events AS event "
            "WHERE NOT EXISTS ("
            "SELECT 1 FROM model_route_activation_events AS newer "
            "WHERE newer.project_id = event.project_id "
            "AND newer.task = event.task "
            "AND newer.cutoff_cycle = event.cutoff_cycle "
            "AND newer.route_role = event.route_role "
            "AND newer.stream_sequence > event.stream_sequence)"
        )
    )


def downgrade() -> None:
    """Remove stream heads without touching immutable decision events."""

    op.drop_table("model_route_activation_stream_heads")
