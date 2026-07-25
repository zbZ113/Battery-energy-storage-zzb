"""Add the append-only Advanced model route activation ledger.

Revision ID: 0009
Revises: 0008
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the immutable decision event stream without an active projection."""

    op.create_table(
        "model_route_activation_events",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("stream_sequence", sa.Integer(), nullable=False),
        sa.Column("task", sa.String(length=16), nullable=False),
        sa.Column("cutoff_cycle", sa.Integer(), nullable=False),
        sa.Column("route_role", sa.String(length=32), nullable=False),
        sa.Column("decision_type", sa.String(length=16), nullable=False),
        sa.Column("artifact_id", sa.String(length=64), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "deployment_bundle_manifest_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("route_provenance_sha256", sa.String(length=64), nullable=False),
        sa.Column("rollback_target_event_id", sa.String(length=64), nullable=True),
        sa.Column("previous_event_sha256", sa.String(length=64), nullable=False),
        sa.Column("event_sha256", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("idempotency_key_sha256", sa.String(length=64), nullable=False),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "stream_sequence > 0 AND cutoff_cycle > 0",
            name="ck_model_route_activation_positive_coordinates",
        ),
        sa.CheckConstraint(
            "decision_type IN ('ACTIVATE', 'ROLLBACK')",
            name="ck_model_route_activation_decision_type",
        ),
        sa.CheckConstraint(
            "(decision_type = 'ACTIVATE' AND rollback_target_event_id IS NULL) "
            "OR (decision_type = 'ROLLBACK' AND rollback_target_event_id IS NOT NULL)",
            name="ck_model_route_activation_rollback_target",
        ),
        sa.CheckConstraint(
            "(task = 'RUL' AND route_role IN "
            "('DEFAULT', 'POINT_ACCURACY', 'COVERAGE')) OR "
            "(task = 'SOH' AND route_role IN "
            "('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_model_route_activation_task_role",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["artifact_id"], ["model_artifacts.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(
            ["rollback_target_event_id"],
            ["model_route_activation_events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "stream_sequence",
            name="uq_model_route_activation_stream_sequence",
        ),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key_sha256",
            name="uq_model_route_activation_idempotency",
        ),
        sa.UniqueConstraint(
            "event_sha256",
            name="uq_model_route_activation_event_sha",
        ),
    )
    op.create_index(
        "ix_model_route_activation_stream",
        "model_route_activation_events",
        ["project_id", "task", "cutoff_cycle", "route_role", "stream_sequence"],
    )
    op.create_index(
        "ix_model_route_activation_artifact",
        "model_route_activation_events",
        ["artifact_id"],
    )


def downgrade() -> None:
    """Remove the decision stream without touching registered candidates."""

    op.drop_index(
        "ix_model_route_activation_artifact",
        table_name="model_route_activation_events",
    )
    op.drop_index(
        "ix_model_route_activation_stream",
        table_name="model_route_activation_events",
    )
    op.drop_table("model_route_activation_events")
