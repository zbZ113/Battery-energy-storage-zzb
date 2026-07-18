"""Add project-scoped A100 experiment suite and task registry.

Revision ID: 0008
Revises: 0007
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create metadata-only experiment registry tables."""

    op.create_table(
        "experiment_suites",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=64), nullable=False),
        sa.Column("import_id", sa.String(length=64), nullable=False),
        sa.Column("dataset_id", sa.String(length=100), nullable=False),
        sa.Column("target", sa.String(length=100), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("source_commit", sa.String(length=64), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("data_version", sa.String(length=200), nullable=False),
        sa.Column("split_version", sa.String(length=200), nullable=False),
        sa.Column("feature_version", sa.String(length=200), nullable=False),
        sa.Column("output_sha256", sa.String(length=64), nullable=False),
        sa.Column("transfer_sha256", sa.String(length=64), nullable=False),
        sa.Column("task_count", sa.Integer(), nullable=False),
        sa.Column("file_count", sa.Integer(), nullable=False),
        sa.Column("formal_performance_claim", sa.Boolean(), nullable=False),
        sa.Column("evidence_uri", sa.Text(), nullable=False),
        sa.Column("created_by_user_id", sa.String(length=64), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "import_id",
            name="uq_experiment_suite_project_import",
        ),
    )
    op.create_index(
        "ix_experiment_suites_project_dataset_mode",
        "experiment_suites",
        ["project_id", "dataset_id", "mode"],
    )
    op.create_table(
        "experiment_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("suite_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=200), nullable=False),
        sa.Column("dataset_id", sa.String(length=100), nullable=False),
        sa.Column("target", sa.String(length=100), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("cutoff_cycle", sa.Integer(), nullable=False),
        sa.Column("seed", sa.Integer(), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("input_bundle_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_commit", sa.String(length=64), nullable=False),
        sa.Column("data_version", sa.String(length=200), nullable=False),
        sa.Column("split_version", sa.String(length=200), nullable=False),
        sa.Column("feature_version", sa.String(length=200), nullable=False),
        sa.Column("context_sha256", sa.String(length=64), nullable=False),
        sa.Column("task_relative_root", sa.Text(), nullable=False),
        sa.Column("evidence_uri", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["suite_id"],
            ["experiment_suites.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "suite_id",
            "run_id",
            name="uq_experiment_run_suite_run_id",
        ),
        sa.UniqueConstraint(
            "suite_id",
            "cutoff_cycle",
            "model_name",
            "seed",
            name="uq_experiment_run_suite_matrix",
        ),
    )
    op.create_index(
        "ix_experiment_runs_suite_model_cutoff_seed",
        "experiment_runs",
        ["suite_id", "model_name", "cutoff_cycle", "seed"],
    )


def downgrade() -> None:
    """Remove the experiment registry without touching imported bytes."""

    op.drop_index(
        "ix_experiment_runs_suite_model_cutoff_seed",
        table_name="experiment_runs",
    )
    op.drop_table("experiment_runs")
    op.drop_index(
        "ix_experiment_suites_project_dataset_mode",
        table_name="experiment_suites",
    )
    op.drop_table("experiment_suites")
