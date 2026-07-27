"""Persist route-specific Advanced calibration materializations.

Revision ID: 0014
Revises: 0013
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create immutable calibration cohort identities and sample bindings."""

    op.create_table(
        "advanced_calibration_materializations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(length=64),
            sa.ForeignKey("projects.id"),
            nullable=False,
        ),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("cutoff_cycle", sa.Integer(), nullable=False),
        sa.Column("route_role", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("data_version", sa.String(length=200), nullable=False),
        sa.Column("split_version", sa.String(length=200), nullable=False),
        sa.Column("feature_version", sa.String(length=200), nullable=False),
        sa.Column(
            "artifact_id",
            sa.String(length=64),
            sa.ForeignKey("model_artifacts.id"),
            nullable=False,
        ),
        sa.Column(
            "artifact_manifest_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "normalization_statistics_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "decision_event_id",
            sa.String(length=64),
            sa.ForeignKey("model_route_activation_events.id"),
            nullable=False,
        ),
        sa.Column("ledger_sequence_number", sa.Integer(), nullable=False),
        sa.Column("ledger_head_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_registration_id", sa.String(length=200), nullable=False),
        sa.Column("source_identity_sha256", sa.String(length=64), nullable=False),
        sa.Column("sample_manifest_sha256", sa.String(length=64), nullable=True),
        sa.Column("sample_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "idempotency_key_sha256",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column("request_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_by_user_id",
            sa.String(length=64),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_code", sa.String(length=100), nullable=True),
        sa.UniqueConstraint(
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "decision_event_id",
            "source_registration_id",
            name="uq_advanced_calibration_exact_route",
        ),
        sa.UniqueConstraint(
            "project_id",
            "idempotency_key_sha256",
            name="uq_advanced_calibration_idempotency",
        ),
        sa.CheckConstraint(
            "cutoff_cycle IN (20, 50, 100, 150) AND "
            "ledger_sequence_number > 0 AND sample_count >= 0",
            name="ck_advanced_calibration_coordinates",
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED', 'STALE')",
            name="ck_advanced_calibration_status",
        ),
        sa.CheckConstraint(
            "(task = 'RUL' AND ((cutoff_cycle = 20 AND route_role = 'DEFAULT') OR "
            "(cutoff_cycle IN (50, 100, 150) AND route_role = 'COVERAGE'))) OR "
            "(task = 'SOH' AND route_role IN ('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_advanced_calibration_task_role",
        ),
        sa.CheckConstraint(
            "(status = 'PENDING' AND started_at IS NULL AND completed_at IS NULL AND "
            "sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
            "failure_code IS NULL) OR "
            "(status = 'RUNNING' AND started_at IS NOT NULL AND completed_at IS NULL "
            "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
            "failure_code IS NULL) OR "
            "(status = 'READY' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND sample_count > 0 AND sample_manifest_sha256 IS NOT NULL AND "
            "failure_code IS NULL) OR "
            "(status = 'FAILED' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
            "failure_code IS NOT NULL) OR "
            "(status = 'STALE' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND sample_count > 0 AND sample_manifest_sha256 IS NOT NULL AND "
            "failure_code IS NOT NULL)",
            name="ck_advanced_calibration_state_payload",
        ),
        sa.CheckConstraint(
            "length(artifact_manifest_sha256) = 64 AND "
            "length(normalization_statistics_sha256) = 64 AND "
            "length(ledger_head_sha256) = 64 AND "
            "length(source_identity_sha256) = 64 AND "
            "(sample_manifest_sha256 IS NULL OR "
            "length(sample_manifest_sha256) = 64) AND "
            "length(idempotency_key_sha256) = 64 AND "
            "length(request_sha256) = 64",
            name="ck_advanced_calibration_hash_lengths",
        ),
    )
    op.create_index(
        "ix_advanced_calibration_project_status",
        "advanced_calibration_materializations",
        ["project_id", "status"],
        unique=False,
    )
    op.create_table(
        "advanced_calibration_sample_bindings",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "materialization_id",
            sa.String(length=64),
            sa.ForeignKey(
                "advanced_calibration_materializations.id",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("cell_id", sa.String(length=200), nullable=False),
        sa.Column(
            "result_id",
            sa.String(length=64),
            sa.ForeignKey("tool_results.id"),
            nullable=False,
        ),
        sa.Column("sample_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "materialization_id",
            "ordinal",
            name="uq_advanced_calibration_sample_ordinal",
        ),
        sa.UniqueConstraint(
            "materialization_id",
            "cell_id",
            name="uq_advanced_calibration_sample_cell",
        ),
        sa.UniqueConstraint(
            "materialization_id",
            "result_id",
            name="uq_advanced_calibration_sample_result",
        ),
        sa.CheckConstraint(
            "ordinal >= 0",
            name="ck_advanced_calibration_sample_ordinal",
        ),
        sa.CheckConstraint(
            "length(sample_sha256) = 64",
            name="ck_advanced_calibration_sample_sha_length",
        ),
    )


def downgrade() -> None:
    """Drop empty materialization tables without discarding audit evidence."""

    connection = op.get_bind()
    materialization_rows = connection.execute(
        sa.text("SELECT count(*) FROM advanced_calibration_materializations")
    ).scalar_one()
    sample_rows = connection.execute(
        sa.text("SELECT count(*) FROM advanced_calibration_sample_bindings")
    ).scalar_one()
    if materialization_rows or sample_rows:
        raise RuntimeError(
            "0014 downgrade would discard audited materializations; "
            "retain revision 0014"
        )

    op.drop_table("advanced_calibration_sample_bindings")
    op.drop_index(
        "ix_advanced_calibration_project_status",
        table_name="advanced_calibration_materializations",
    )
    op.drop_table("advanced_calibration_materializations")
