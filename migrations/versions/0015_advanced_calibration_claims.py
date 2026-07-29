"""Bind Advanced calibration work to an ADMIN session and fenced claim.

Revision ID: 0015
Revises: 0014
"""

from collections.abc import Sequence
from typing import Literal

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_OLD_STATE_CHECK = (
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
    "failure_code IS NOT NULL)"
)
_FENCED_STATE_CHECK = (
    "(status = 'PENDING' AND started_at IS NULL AND completed_at IS NULL AND "
    "sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
    "failure_code IS NULL AND claim_token_sha256 IS NULL AND "
    "claim_attempt = 0 AND claim_lease_expires_at IS NULL) OR "
    "(status = 'RUNNING' AND started_at IS NOT NULL AND completed_at IS NULL "
    "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
    "failure_code IS NULL AND claim_token_sha256 IS NOT NULL AND "
    "claim_attempt > 0 AND claim_lease_expires_at IS NOT NULL) OR "
    "(status = 'READY' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
    "AND sample_count > 0 AND sample_manifest_sha256 IS NOT NULL AND "
    "failure_code IS NULL AND claim_token_sha256 IS NULL AND "
    "claim_attempt > 0 AND claim_lease_expires_at IS NULL) OR "
    "(status = 'FAILED' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
    "AND sample_count = 0 AND sample_manifest_sha256 IS NULL AND "
    "failure_code IS NOT NULL AND claim_token_sha256 IS NULL AND "
    "claim_attempt > 0 AND claim_lease_expires_at IS NULL) OR "
    "(status = 'STALE' AND started_at IS NOT NULL AND completed_at IS NOT NULL "
    "AND sample_count > 0 AND sample_manifest_sha256 IS NOT NULL AND "
    "failure_code IS NOT NULL AND claim_token_sha256 IS NULL AND "
    "claim_attempt > 0 AND claim_lease_expires_at IS NULL)"
)


def _batch_recreate_mode(
    dialect_name: str,
) -> Literal["always", "auto"]:
    """Recreate only where SQLite requires it; preserve PostgreSQL dependencies."""

    return "always" if dialect_name == "sqlite" else "auto"


def upgrade() -> None:
    """Add the immutable actor session and reclaimable worker lease."""

    bind = op.get_bind()
    recreate = _batch_recreate_mode(bind.dialect.name)
    existing_rows = bind.execute(
        sa.text("SELECT count(*) FROM advanced_calibration_materializations")
    ).scalar_one()
    if existing_rows:
        raise RuntimeError(
            "0015 cannot infer ADMIN session and claim evidence for existing "
            "Advanced calibration materializations"
        )

    with op.batch_alter_table(
        "advanced_calibration_materializations",
        recreate=recreate,
    ) as batch:
        batch.drop_constraint(
            "ck_advanced_calibration_state_payload",
            type_="check",
        )
        batch.add_column(
            sa.Column("created_by_session_id", sa.String(length=64), nullable=False)
        )
        batch.add_column(
            sa.Column("created_by_role", sa.String(length=32), nullable=False)
        )
        batch.add_column(
            sa.Column("claim_token_sha256", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column(
                "claim_attempt",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch.add_column(
            sa.Column(
                "claim_lease_expires_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch.create_foreign_key(
            "fk_advanced_calibration_created_by_session",
            "sessions",
            ["created_by_session_id"],
            ["id"],
        )
        batch.create_check_constraint(
            "ck_advanced_calibration_admin_actor",
            "created_by_role = 'ADMIN'",
        )
        batch.create_check_constraint(
            "ck_advanced_calibration_claim_hash_length",
            "claim_token_sha256 IS NULL OR length(claim_token_sha256) = 64",
        )
        batch.create_check_constraint(
            "ck_advanced_calibration_state_payload",
            _FENCED_STATE_CHECK,
        )


def downgrade() -> None:
    """Remove claim evidence only when no audited materialization exists."""

    bind = op.get_bind()
    recreate = _batch_recreate_mode(bind.dialect.name)
    existing_rows = bind.execute(
        sa.text("SELECT count(*) FROM advanced_calibration_materializations")
    ).scalar_one()
    if existing_rows:
        raise RuntimeError(
            "0015 downgrade would discard fenced claim evidence; retain revision 0015"
        )

    with op.batch_alter_table(
        "advanced_calibration_materializations",
        recreate=recreate,
    ) as batch:
        batch.drop_constraint(
            "ck_advanced_calibration_state_payload",
            type_="check",
        )
        batch.drop_constraint(
            "ck_advanced_calibration_claim_hash_length",
            type_="check",
        )
        batch.drop_constraint(
            "ck_advanced_calibration_admin_actor",
            type_="check",
        )
        batch.drop_constraint(
            "fk_advanced_calibration_created_by_session",
            type_="foreignkey",
        )
        batch.drop_column("claim_lease_expires_at")
        batch.drop_column("claim_attempt")
        batch.drop_column("claim_token_sha256")
        batch.drop_column("created_by_role")
        batch.drop_column("created_by_session_id")
        batch.create_check_constraint(
            "ck_advanced_calibration_state_payload",
            _OLD_STATE_CHECK,
        )
