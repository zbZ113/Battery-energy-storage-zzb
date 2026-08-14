"""SQLAlchemy persistence schema for structured, auditable product state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    DateTime as SQLAlchemyDateTime,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from quanxin_life.core.enums import (
    AgentDispatchStatus,
    AgentRunStatus,
    ApprovalStatus,
    KnowledgeReviewStatus,
    UserRole,
)

EMBEDDING_TYPE = Vector(1536).with_variant(JSON(), "sqlite")


def utc_now() -> datetime:
    """Return an aware UTC timestamp for application-side column defaults."""

    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist only aware timestamps and normalize values to UTC."""

    impl = SQLAlchemyDateTime
    cache_ok = True
    timezone = True

    def __init__(self, *, timezone: bool = True) -> None:
        if not timezone:
            raise ValueError("UTCDateTime requires timezone=True")
        super().__init__(timezone=True)

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include a timezone")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        del dialect
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


DateTime = UTCDateTime


class Base(DeclarativeBase):
    """Shared declarative base; importing it never opens a database connection."""


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    credential_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    must_change_credential: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role: Mapped[str] = mapped_column(String(32), default=UserRole.MEMBER.value, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class SessionRecord(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user_id", "user_id"),
        Index("ix_sessions_expires_at", "expires_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (Index("ix_projects_owner_user_id", "owner_user_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class UserProjectRole(Base):
    __tablename__ = "user_project_roles"
    __table_args__ = (UniqueConstraint("user_id", "project_id", name="uq_user_project_role"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = (Index("ix_datasets_project_id_status", "project_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    data_version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DatasetFile(Base):
    __tablename__ = "dataset_files"
    __table_args__ = (
        UniqueConstraint("dataset_id", "sha256", name="uq_dataset_file_sha256"),
        Index("ix_dataset_files_dataset_id", "dataset_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    object_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(200))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    source_uri: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class RecordBatchBinding(Base):
    __tablename__ = "record_batch_bindings"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "content_batch_id",
            name="uq_record_batch_binding_dataset_content",
        ),
        Index(
            "ix_record_batch_bindings_project_dataset",
            "project_id",
            "dataset_id",
        ),
        CheckConstraint(
            "cutoff_cycle > 0",
            name="ck_record_batch_binding_cutoff_positive",
        ),
        CheckConstraint(
            "length(source_manifest_sha256) = 64 AND "
            "length(registration_sha256) = 64",
            name="ck_record_batch_binding_sha256_lengths",
        ),
        CheckConstraint(
            "binding_schema_version = 'record-batch-binding-v1'",
            name="ck_record_batch_binding_schema_version",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    binding_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    content_batch_id: Mapped[str] = mapped_column(String(78), nullable=False)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    source_manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    registration_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    content_dataset_id: Mapped[str] = mapped_column(String(200), nullable=False)
    dataset_schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    cell_id: Mapped[str] = mapped_column(String(200), nullable=False)
    cutoff_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    data_version: Mapped[str] = mapped_column(String(100), nullable=False)
    split_version: Mapped[str] = mapped_column(String(100), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class CellSplit(Base):
    __tablename__ = "cell_splits"
    __table_args__ = (
        UniqueConstraint("dataset_id", "cell_id", name="uq_cell_split_dataset_cell"),
        Index("ix_cell_splits_dataset_split", "dataset_id", "split_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    cell_id: Mapped[str] = mapped_column(String(200), nullable=False)
    split_name: Mapped[str] = mapped_column(String(32), nullable=False)
    split_version: Mapped[str] = mapped_column(String(100), nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint(
            "created_by_user_id",
            "idempotency_key_hash",
            name="uq_agent_run_user_idempotency_key",
        ),
        Index("ix_agent_runs_project_created", "project_id", "created_at"),
        Index("ix_agent_runs_status", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    session_id: Mapped[str | None] = mapped_column(ForeignKey("sessions.id"))
    created_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    idempotency_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=AgentRunStatus.PLANNING.value, nullable=False
    )
    planning_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    intent_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    plan_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    plan_hash: Mapped[str | None] = mapped_column(String(64))
    execution_plan_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentRunDispatch(Base):
    __tablename__ = "agent_run_dispatches"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_agent_run_dispatch_run_id"),
        Index("ix_agent_run_dispatches_status_created", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=AgentDispatchStatus.PENDING.value, nullable=False
    )
    task_id: Mapped[str | None] = mapped_column(String(200))
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class AgentStep(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (
        UniqueConstraint("run_id", "step_id", name="uq_agent_step_run_step"),
        Index("ix_agent_steps_run_ordinal", "run_id", "ordinal"),
        Index("ix_agent_steps_status", "status"),
        CheckConstraint(
            "resolved_input_hash IS NULL OR length(resolved_input_hash) = 64",
            name="ck_agent_step_resolved_input_hash_length",
        ),
        CheckConstraint(
            "execution_snapshot_sha256 IS NULL OR "
            "length(execution_snapshot_sha256) = 64",
            name="ck_agent_step_execution_snapshot_sha256_length",
        ),
        CheckConstraint(
            "dependency_evidence_sha256 IS NULL OR "
            "length(dependency_evidence_sha256) = 64",
            name="ck_agent_step_dependency_evidence_sha256_length",
        ),
        CheckConstraint(
            "execution_claim_sha256 IS NULL OR length(execution_claim_sha256) = 64",
            name="ck_agent_step_execution_claim_sha256_length",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    step_id: Mapped[str] = mapped_column(String(200), nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_refs_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    depends_on_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    failure_policy: Mapped[str] = mapped_column(String(32), nullable=False)
    requires_human_approval: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    claim_token: Mapped[str | None] = mapped_column(String(64))
    execution_claim_sha256: Mapped[str | None] = mapped_column(String(64))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_input_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    resolved_input_hash: Mapped[str | None] = mapped_column(String(64))
    execution_snapshot_sha256: Mapped[str | None] = mapped_column(String(64))
    dependency_evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    last_error_code: Mapped[str | None] = mapped_column(String(100))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AgentEvent(Base):
    __tablename__ = "agent_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_agent_event_run_sequence"),
        Index("ix_agent_events_run_created", "run_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ToolResultRecord(Base):
    __tablename__ = "tool_results"
    __table_args__ = (
        UniqueConstraint("agent_step_id", name="uq_tool_result_agent_step_id"),
        Index("ix_tool_results_run_created", "run_id", "created_at"),
        Index("ix_tool_results_step_id", "agent_step_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"))
    agent_step_id: Mapped[str | None] = mapped_column(ForeignKey("agent_steps.id"))
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    tool_version: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(100))
    data_version: Mapped[str | None] = mapped_column(String(100))
    feature_version: Mapped[str | None] = mapped_column(String(100))
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    values_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    uncertainty_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    warnings_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ProvenanceRecordRow(Base):
    __tablename__ = "provenance_records"
    __table_args__ = (Index("ix_provenance_records_result_id", "tool_result_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tool_result_id: Mapped[str] = mapped_column(
        ForeignKey("tool_results.id", ondelete="CASCADE"), nullable=False
    )
    source_id: Mapped[str] = mapped_column(String(200), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GlobalToolResultBindingRecord(Base):
    __tablename__ = "global_tool_result_bindings"
    __table_args__ = (
        CheckConstraint(
            "binding_schema_version = 'global-tool-result-binding-v1'",
            name="ck_global_tool_result_binding_schema_version",
        ),
        CheckConstraint(
            "length(result_sha256) = 64 AND length(binding_sha256) = 64",
            name="ck_global_tool_result_binding_hash_lengths",
        ),
    )

    result_id: Mapped[str] = mapped_column(
        ForeignKey("tool_results.id", ondelete="CASCADE"), primary_key=True
    )
    binding_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    result_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    binding_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ProjectToolResultBindingRecord(Base):
    __tablename__ = "project_tool_result_bindings"
    __table_args__ = (
        Index(
            "ix_project_tool_result_bindings_project_created",
            "project_id",
            "created_at",
        ),
        CheckConstraint(
            "binding_schema_version IN "
            "('project-tool-result-binding-v1', 'project-tool-result-binding-v2', "
            "'project-tool-result-binding-v3')",
            name="ck_project_tool_result_binding_schema_version",
        ),
        CheckConstraint(
            "actor_role IN ('ADMIN', 'MEMBER', 'JUDGE')",
            name="ck_project_tool_result_binding_actor_role",
        ),
        CheckConstraint(
            "invocation_source IN ('HTTP', 'AGENT', 'FEISHU')",
            name="ck_project_tool_result_binding_invocation_source",
        ),
        CheckConstraint(
            "length(input_hash) = 64 AND length(result_sha256) = 64 AND "
            "length(binding_sha256) = 64",
            name="ck_project_tool_result_binding_hash_lengths",
        ),
        CheckConstraint(
            "(plan_hash IS NULL OR length(plan_hash) = 64) AND "
            "(claim_token_sha256 IS NULL OR length(claim_token_sha256) = 64) AND "
            "(execution_snapshot_sha256 IS NULL OR "
            "length(execution_snapshot_sha256) = 64) AND "
            "(dependency_evidence_sha256 IS NULL OR "
            "length(dependency_evidence_sha256) = 64) AND "
            "(approval_evidence_sha256 IS NULL OR "
            "length(approval_evidence_sha256) = 64)",
            name="ck_project_tool_result_binding_agent_hash_lengths",
        ),
        CheckConstraint(
            "(binding_schema_version = 'project-tool-result-binding-v1' AND "
            "invocation_source = 'HTTP' AND actor_session_id IS NOT NULL AND "
            "feishu_binding_id IS NULL AND agent_run_id IS NULL AND "
            "agent_step_id IS NULL AND step_id IS NULL AND plan_hash IS NULL AND "
            "claim_token_sha256 IS NULL AND claim_attempt IS NULL AND "
            "claim_lease_expires_at IS NULL AND execution_snapshot_sha256 IS NULL AND "
            "dependency_evidence_sha256 IS NULL AND approval_required IS NULL AND "
            "approval_request_id IS NULL AND approval_action_id IS NULL AND "
            "approval_evidence_sha256 IS NULL) OR "
            "(binding_schema_version = 'project-tool-result-binding-v2' AND "
            "invocation_source = 'AGENT' AND actor_session_id IS NOT NULL AND "
            "feishu_binding_id IS NULL AND agent_run_id IS NOT NULL AND "
            "agent_step_id IS NOT NULL AND step_id IS NOT NULL AND "
            "plan_hash IS NOT NULL AND claim_token_sha256 IS NOT NULL AND "
            "claim_attempt IS NOT NULL AND claim_attempt > 0 AND "
            "claim_lease_expires_at IS NOT NULL AND "
            "execution_snapshot_sha256 IS NOT NULL AND "
            "dependency_evidence_sha256 IS NOT NULL AND "
            "approval_required IS NOT NULL AND approval_evidence_sha256 IS NOT NULL AND "
            "((approval_required IS FALSE AND approval_request_id IS NULL AND "
            "approval_action_id IS NULL) OR (approval_required IS TRUE AND "
            "approval_request_id IS NOT NULL AND approval_action_id IS NOT NULL))) OR "
            "(binding_schema_version = 'project-tool-result-binding-v3' AND "
            "invocation_source = 'FEISHU' AND "
            "actor_role IN ('ADMIN', 'MEMBER') AND actor_session_id IS NULL AND "
            "feishu_binding_id IS NOT NULL AND agent_run_id IS NULL AND "
            "agent_step_id IS NULL AND step_id IS NULL AND plan_hash IS NULL AND "
            "claim_token_sha256 IS NULL AND claim_attempt IS NULL AND "
            "claim_lease_expires_at IS NULL AND execution_snapshot_sha256 IS NULL AND "
            "dependency_evidence_sha256 IS NULL AND approval_required IS NULL AND "
            "approval_request_id IS NULL AND approval_action_id IS NULL AND "
            "approval_evidence_sha256 IS NULL)",
            name="ck_project_tool_result_binding_exact_agent_contract",
        ),
        UniqueConstraint(
            "agent_step_id", name="uq_project_tool_result_binding_agent_step"
        ),
        UniqueConstraint(
            "agent_run_id",
            "step_id",
            name="uq_project_tool_result_binding_run_step",
        ),
        Index("ix_project_tool_result_bindings_agent_step", "agent_step_id"),
    )

    result_id: Mapped[str] = mapped_column(
        ForeignKey("tool_results.id"), primary_key=True
    )
    binding_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id"), nullable=False
    )
    actor_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    actor_session_id: Mapped[str | None] = mapped_column(
        ForeignKey("sessions.id"), nullable=True
    )
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    invocation_source: Mapped[str] = mapped_column(String(32), nullable=False)
    feishu_binding_id: Mapped[str | None] = mapped_column(
        ForeignKey("feishu_bindings.id")
    )
    agent_run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"))
    agent_step_id: Mapped[str | None] = mapped_column(ForeignKey("agent_steps.id"))
    step_id: Mapped[str | None] = mapped_column(String(200))
    plan_hash: Mapped[str | None] = mapped_column(String(64))
    claim_token_sha256: Mapped[str | None] = mapped_column(String(64))
    claim_attempt: Mapped[int | None] = mapped_column(Integer)
    claim_lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    execution_snapshot_sha256: Mapped[str | None] = mapped_column(String(64))
    dependency_evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    approval_required: Mapped[bool | None] = mapped_column(Boolean)
    approval_request_id: Mapped[str | None] = mapped_column(
        ForeignKey("approval_requests.id")
    )
    approval_action_id: Mapped[str | None] = mapped_column(
        ForeignKey("approval_actions.id")
    )
    approval_evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(200), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    result_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    binding_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        UniqueConstraint("sha256", name="uq_model_artifact_sha256"),
        Index("ix_model_artifacts_project_id", "project_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    artifact_format: Mapped[str] = mapped_column(String(50), nullable=False)
    object_uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelManifest(Base):
    __tablename__ = "model_manifests"
    __table_args__ = (UniqueConstraint("artifact_id", name="uq_model_manifest_artifact"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("model_artifacts.id", ondelete="CASCADE"), nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    manifest_uri: Mapped[str] = mapped_column(Text, nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ModelRouteActivationEvent(Base):
    __tablename__ = "model_route_activation_events"
    __table_args__ = (
        CheckConstraint(
            "stream_sequence > 0 AND cutoff_cycle > 0",
            name="ck_model_route_activation_positive_coordinates",
        ),
        CheckConstraint(
            "decision_type IN ('ACTIVATE', 'ROLLBACK')",
            name="ck_model_route_activation_decision_type",
        ),
        CheckConstraint(
            "(decision_type = 'ACTIVATE' AND rollback_target_event_id IS NULL) "
            "OR (decision_type = 'ROLLBACK' AND rollback_target_event_id IS NOT NULL)",
            name="ck_model_route_activation_rollback_target",
        ),
        CheckConstraint(
            "(task = 'RUL' AND route_role IN "
            "('DEFAULT', 'POINT_ACCURACY', 'COVERAGE')) OR "
            "(task = 'SOH' AND route_role IN "
            "('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_model_route_activation_task_role",
        ),
        UniqueConstraint(
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "stream_sequence",
            name="uq_model_route_activation_stream_sequence",
        ),
        UniqueConstraint(
            "project_id",
            "idempotency_key_sha256",
            name="uq_model_route_activation_idempotency",
        ),
        UniqueConstraint("event_sha256", name="uq_model_route_activation_event_sha"),
        Index(
            "ix_model_route_activation_stream",
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "stream_sequence",
        ),
        Index("ix_model_route_activation_artifact", "artifact_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    stream_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    task: Mapped[str] = mapped_column(String(16), nullable=False)
    cutoff_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    route_role: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_type: Mapped[str] = mapped_column(String(16), nullable=False)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("model_artifacts.id"), nullable=False
    )
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    deployment_bundle_manifest_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    route_provenance_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    rollback_target_event_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_route_activation_events.id")
    )
    previous_event_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    event_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    idempotency_key_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ModelRouteActivationStreamHead(Base):
    __tablename__ = "model_route_activation_stream_heads"
    __table_args__ = (
        CheckConstraint(
            "head_sequence > 0 AND cutoff_cycle > 0",
            name="ck_model_route_stream_head_positive_coordinates",
        ),
        CheckConstraint(
            "(task = 'RUL' AND route_role IN "
            "('DEFAULT', 'POINT_ACCURACY', 'COVERAGE')) OR "
            "(task = 'SOH' AND route_role IN "
            "('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_model_route_stream_head_task_role",
        ),
    )

    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id"),
        primary_key=True,
    )
    task: Mapped[str] = mapped_column(String(16), primary_key=True)
    cutoff_cycle: Mapped[int] = mapped_column(Integer, primary_key=True)
    route_role: Mapped[str] = mapped_column(String(32), primary_key=True)
    head_event_id: Mapped[str] = mapped_column(
        ForeignKey("model_route_activation_events.id"),
        nullable=False,
    )
    head_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    head_event_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


class ExperimentSuite(Base):
    __tablename__ = "experiment_suites"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "import_id",
            name="uq_experiment_suite_project_import",
        ),
        Index(
            "ix_experiment_suites_project_dataset_mode",
            "project_id",
            "dataset_id",
            "mode",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    import_id: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    target: Mapped[str] = mapped_column(String(100), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    source_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    data_version: Mapped[str] = mapped_column(String(200), nullable=False)
    split_version: Mapped[str] = mapped_column(String(200), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(200), nullable=False)
    output_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    transfer_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    task_count: Mapped[int] = mapped_column(Integer, nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False)
    formal_performance_claim: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evidence_uri: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    __table_args__ = (
        UniqueConstraint(
            "suite_id",
            "run_id",
            name="uq_experiment_run_suite_run_id",
        ),
        UniqueConstraint(
            "suite_id",
            "cutoff_cycle",
            "model_name",
            "seed",
            name="uq_experiment_run_suite_matrix",
        ),
        Index(
            "ix_experiment_runs_suite_model_cutoff_seed",
            "suite_id",
            "model_name",
            "cutoff_cycle",
            "seed",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    suite_id: Mapped[str] = mapped_column(
        ForeignKey("experiment_suites.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str] = mapped_column(String(200), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(100), nullable=False)
    target: Mapped[str] = mapped_column(String(100), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    cutoff_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    input_bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    data_version: Mapped[str] = mapped_column(String(200), nullable=False)
    split_version: Mapped[str] = mapped_column(String(200), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(200), nullable=False)
    context_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    task_relative_root: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_uri: Mapped[str] = mapped_column(Text, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class CalibrationCohort(Base):
    __tablename__ = "calibration_cohorts"
    __table_args__ = (
        UniqueConstraint("dataset_id", "cohort_version", name="uq_calibration_cohort_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    cohort_version: Mapped[str] = mapped_column(String(100), nullable=False)
    split_version: Mapped[str] = mapped_column(String(100), nullable=False)
    cell_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    manifest_uri: Mapped[str | None] = mapped_column(Text)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class AdvancedCalibrationMaterialization(Base):
    __tablename__ = "advanced_calibration_materializations"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "task",
            "cutoff_cycle",
            "route_role",
            "decision_event_id",
            "source_registration_id",
            name="uq_advanced_calibration_exact_route",
        ),
        UniqueConstraint(
            "project_id",
            "idempotency_key_sha256",
            name="uq_advanced_calibration_idempotency",
        ),
        CheckConstraint(
            "cutoff_cycle IN (20, 50, 100, 150) AND "
            "ledger_sequence_number > 0 AND sample_count >= 0",
            name="ck_advanced_calibration_coordinates",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED', 'STALE')",
            name="ck_advanced_calibration_status",
        ),
        CheckConstraint(
            "(task = 'RUL' AND ((cutoff_cycle = 20 AND route_role = 'DEFAULT') OR "
            "(cutoff_cycle IN (50, 100, 150) AND route_role = 'COVERAGE'))) OR "
            "(task = 'SOH' AND route_role IN ('MEAN_ACCURACY', 'TAIL_EFFICIENCY'))",
            name="ck_advanced_calibration_task_role",
        ),
        CheckConstraint(
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
            "claim_attempt > 0 AND claim_lease_expires_at IS NULL)",
            name="ck_advanced_calibration_state_payload",
        ),
        CheckConstraint(
            "created_by_role = 'ADMIN'",
            name="ck_advanced_calibration_admin_actor",
        ),
        CheckConstraint(
            "claim_token_sha256 IS NULL OR length(claim_token_sha256) = 64",
            name="ck_advanced_calibration_claim_hash_length",
        ),
        CheckConstraint(
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
        Index(
            "ix_advanced_calibration_project_status",
            "project_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id"), nullable=False
    )
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    cutoff_cycle: Mapped[int] = mapped_column(Integer, nullable=False)
    route_role: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    data_version: Mapped[str] = mapped_column(String(200), nullable=False)
    split_version: Mapped[str] = mapped_column(String(200), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(200), nullable=False)
    artifact_id: Mapped[str] = mapped_column(
        ForeignKey("model_artifacts.id"), nullable=False
    )
    artifact_manifest_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    normalization_statistics_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    decision_event_id: Mapped[str] = mapped_column(
        ForeignKey("model_route_activation_events.id"), nullable=False
    )
    ledger_sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    ledger_head_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_registration_id: Mapped[str] = mapped_column(
        String(200), nullable=False
    )
    source_identity_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    sample_manifest_sha256: Mapped[str | None] = mapped_column(String(64))
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    idempotency_key_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    request_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by_user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id"), nullable=False
    )
    created_by_session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id"), nullable=False
    )
    created_by_role: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    failure_code: Mapped[str | None] = mapped_column(String(100))
    claim_token_sha256: Mapped[str | None] = mapped_column(String(64))
    claim_attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    claim_lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class AdvancedCalibrationSampleBinding(Base):
    __tablename__ = "advanced_calibration_sample_bindings"
    __table_args__ = (
        UniqueConstraint(
            "materialization_id",
            "ordinal",
            name="uq_advanced_calibration_sample_ordinal",
        ),
        UniqueConstraint(
            "materialization_id",
            "cell_id",
            name="uq_advanced_calibration_sample_cell",
        ),
        UniqueConstraint(
            "materialization_id",
            "result_id",
            name="uq_advanced_calibration_sample_result",
        ),
        CheckConstraint(
            "ordinal >= 0",
            name="ck_advanced_calibration_sample_ordinal",
        ),
        CheckConstraint(
            "length(sample_sha256) = 64",
            name="ck_advanced_calibration_sample_sha_length",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    materialization_id: Mapped[str] = mapped_column(
        ForeignKey(
            "advanced_calibration_materializations.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    cell_id: Mapped[str] = mapped_column(String(200), nullable=False)
    result_id: Mapped[str] = mapped_column(
        ForeignKey("tool_results.id"), nullable=False
    )
    sample_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, nullable=False
    )


class DecisionPolicy(Base):
    __tablename__ = "decision_policies"
    __table_args__ = (
        UniqueConstraint("project_id", "policy_version", name="uq_decision_policy_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    approved_by_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ApprovalRequestRow(Base):
    __tablename__ = "approval_requests"
    __table_args__ = (
        UniqueConstraint("run_id", "step_id", name="uq_approval_request_run_step"),
        Index("ix_approval_requests_run_status", "run_id", "status"),
        CheckConstraint(
            "execution_snapshot_sha256 IS NULL OR "
            "length(execution_snapshot_sha256) = 64",
            name="ck_approval_request_execution_snapshot_length",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), nullable=False)
    approval_kind: Mapped[str] = mapped_column(String(50), nullable=False)
    source_plan_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_step_id: Mapped[str | None] = mapped_column(ForeignKey("agent_steps.id"))
    execution_snapshot_sha256: Mapped[str | None] = mapped_column(String(64))
    step_id: Mapped[str] = mapped_column(String(200), nullable=False)
    impact_scope: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), default=ApprovalStatus.PENDING.value, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ApprovalAction(Base):
    __tablename__ = "approval_actions"
    __table_args__ = (
        UniqueConstraint("approval_request_id", name="uq_approval_action_request_id"),
        Index("ix_approval_actions_request_id", "approval_request_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    approval_request_id: Mapped[str] = mapped_column(
        ForeignKey("approval_requests.id", ondelete="CASCADE"), nullable=False
    )
    actor_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    acted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Report(Base):
    __tablename__ = "reports"
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_reports_run_id"),
        Index("ix_reports_project_created", "project_id", "created_at"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED')",
            name="ck_reports_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    template_version: Mapped[str] = mapped_column(String(100), nullable=False)
    result_ids_json: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    object_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    failure_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ReportExport(Base):
    __tablename__ = "report_exports"
    __table_args__ = (
        UniqueConstraint("report_id", "export_format", name="uq_report_export_format"),
        Index("ix_report_exports_report_id", "report_id"),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'READY', 'FAILED', 'EXPIRED')",
            name="ck_report_exports_status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    report_id: Mapped[str] = mapped_column(
        ForeignKey("reports.id", ondelete="CASCADE"), nullable=False
    )
    export_format: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    object_uri: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    filename: Mapped[str | None] = mapped_column(String(255))
    media_type: Mapped[str | None] = mapped_column(String(200))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    failure_code: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "source_sha256",
            name="uq_knowledge_project_source_sha256",
        ),
        UniqueConstraint(
            "created_by_user_id",
            "idempotency_key_hash",
            name="uq_knowledge_uploader_idempotency_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False)
    created_by_user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    idempotency_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    license_name: Mapped[str] = mapped_column(String(200), nullable=False)
    document_version: Mapped[str] = mapped_column(String(100), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(32), default=KnowledgeReviewStatus.PENDING.value, nullable=False
    )
    reviewer_user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    object_uri: Mapped[str] = mapped_column(Text, nullable=False)
    object_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    object_content_type: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_knowledge_chunk_index"),
        Index("ix_knowledge_chunks_document_id", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text_object_uri: Mapped[str] = mapped_column(Text, nullable=False)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    text_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content_type: Mapped[str] = mapped_column(String(200), nullable=False)
    section_label: Mapped[str | None] = mapped_column(String(500))
    embedding_model_version: Mapped[str | None] = mapped_column(String(100))
    embedding: Mapped[list[float] | None] = mapped_column(EMBEDDING_TYPE)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class FeishuBindingRow(Base):
    __tablename__ = "feishu_bindings"
    __table_args__ = (
        UniqueConstraint("project_id", "binding_version", name="uq_feishu_binding_version"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    chat_id: Mapped[str | None] = mapped_column(String(200))
    bitable_app_token: Mapped[str | None] = mapped_column(String(200))
    bitable_table_id: Mapped[str | None] = mapped_column(String(200))
    user_open_id_map_json: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)
    binding_version: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class FeishuEventReceipt(Base):
    __tablename__ = "feishu_event_receipts"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_feishu_event_receipt_event_id"),
        UniqueConstraint("job_id", name="uq_feishu_event_receipt_job_id"),
        UniqueConstraint(
            "job_request_sha256",
            name="uq_feishu_event_receipt_job_request_sha256",
        ),
        CheckConstraint(
            "job_origin IN ('FEISHU', 'AILY')",
            name="ck_feishu_event_receipt_job_origin",
        ),
        CheckConstraint(
            "job_request_sha256 IS NULL OR length(job_request_sha256) = 64",
            name="ck_feishu_event_receipt_job_request_sha256_length",
        ),
        CheckConstraint(
            "csv_mapping_status IS NULL OR "
            "csv_mapping_status IN ('MAPPED', 'REJECTED')",
            name="ck_feishu_event_receipt_csv_mapping_status",
        ),
        CheckConstraint(
            "(csv_mapping_status IS NULL AND csv_mapping_evidence_json IS NULL AND "
            "csv_mapping_evidence_sha256 IS NULL) OR "
            "(csv_mapping_status IS NOT NULL AND csv_mapping_evidence_json IS NOT NULL "
            "AND csv_mapping_evidence_sha256 IS NOT NULL "
            "AND length(csv_mapping_evidence_sha256) = 64)",
            name="ck_feishu_event_receipt_csv_mapping_evidence_contract",
        ),
        CheckConstraint(
            "(analysis_image_key IS NULL AND "
            "analysis_image_renderer_version IS NULL AND "
            "analysis_image_sha256 IS NULL) OR "
            "(analysis_image_key IS NOT NULL AND "
            "analysis_image_renderer_version IS NOT NULL AND "
            "analysis_image_sha256 IS NOT NULL AND "
            "length(analysis_image_sha256) = 64)",
            name="ck_feishu_event_receipt_analysis_image_provenance",
        ),
        CheckConstraint(
            "(bitable_curve_file_token IS NULL AND "
            "bitable_curve_source_result_id IS NULL AND "
            "bitable_curve_renderer_version IS NULL AND "
            "bitable_curve_sha256 IS NULL AND "
            "bitable_curve_template IS NULL) OR "
            "(bitable_curve_file_token IS NOT NULL AND "
            "bitable_curve_source_result_id IS NOT NULL AND "
            "bitable_curve_renderer_version IS NOT NULL AND "
            "bitable_curve_sha256 IS NOT NULL AND "
            "bitable_curve_template IS NOT NULL AND "
            "length(bitable_curve_sha256) = 64 AND "
            "analysis_result_id IS NOT NULL AND "
            "bitable_curve_source_result_id = analysis_result_id)",
            name="ck_feishu_event_receipt_bitable_curve_provenance",
        ),
        CheckConstraint(
            "source_job_id IS NULL OR source_job_id <> job_id",
            name="ck_feishu_event_receipt_source_job_distinct",
        ),
        CheckConstraint(
            "(default_scenario_profile_id IS NULL AND "
            "default_scenario_profile_version IS NULL AND "
            "default_scenario_profile_sha256 IS NULL) OR "
            "(default_scenario_profile_id IS NOT NULL AND "
            "default_scenario_profile_version IS NOT NULL AND "
            "default_scenario_profile_sha256 IS NOT NULL AND "
            "length(default_scenario_profile_sha256) = 64)",
            name="ck_feishu_event_receipt_default_scenario_profile",
        ),
        Index("ix_feishu_receipts_received_at", "received_at"),
        Index("ix_feishu_receipts_job_status_updated", "job_status", "job_updated_at"),
        Index("ix_feishu_receipts_scenario_context_id", "scenario_context_id"),
        Index("ix_feishu_receipts_source_job_id", "source_job_id", "task_type"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(200), nullable=False)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    claim_token: Mapped[str | None] = mapped_column(String(64))
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    received_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), default=utc_now, nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    failed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    job_id: Mapped[str | None] = mapped_column(String(64))
    job_origin: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="FEISHU",
    )
    job_request_sha256: Mapped[str | None] = mapped_column(String(64))
    source_job_id: Mapped[str | None] = mapped_column(String(64))
    job_status: Mapped[str | None] = mapped_column(String(32))
    job_stage: Mapped[str | None] = mapped_column(String(32))
    task_type: Mapped[str | None] = mapped_column(String(100))
    run_id: Mapped[str | None] = mapped_column(String(64))
    message_id: Mapped[str | None] = mapped_column(String(200))
    file_key: Mapped[str | None] = mapped_column(String(200))
    file_name: Mapped[str | None] = mapped_column(String(255))
    chat_id: Mapped[str | None] = mapped_column(String(200))
    sender_id: Mapped[str | None] = mapped_column(String(200))
    receive_id_type: Mapped[str | None] = mapped_column(String(32))
    event_time: Mapped[datetime | None] = mapped_column(UTCDateTime())
    scenario_context_id: Mapped[str | None] = mapped_column(String(64))
    default_scenario_profile_id: Mapped[str | None] = mapped_column(String(200))
    default_scenario_profile_version: Mapped[str | None] = mapped_column(String(100))
    default_scenario_profile_sha256: Mapped[str | None] = mapped_column(String(64))
    job_claim_token: Mapped[str | None] = mapped_column(String(64))
    job_attempt_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    job_lease_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    job_task_id: Mapped[str | None] = mapped_column(String(200))
    job_last_error_code: Mapped[str | None] = mapped_column(String(100))
    record_batch_id: Mapped[str | None] = mapped_column(String(100))
    cell_reference: Mapped[str | None] = mapped_column(String(200))
    input_file_sha256: Mapped[str | None] = mapped_column(String(64))
    csv_mapping_status: Mapped[str | None] = mapped_column(String(32))
    csv_mapping_evidence_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    csv_mapping_evidence_sha256: Mapped[str | None] = mapped_column(String(64))
    validation_result_id: Mapped[str | None] = mapped_column(String(64))
    prepared_input_result_id: Mapped[str | None] = mapped_column(String(64))
    analysis_result_id: Mapped[str | None] = mapped_column(String(64))
    report_result_id: Mapped[str | None] = mapped_column(String(64))
    scenario_image_key: Mapped[str | None] = mapped_column(String(200))
    analysis_image_key: Mapped[str | None] = mapped_column(String(200))
    analysis_image_renderer_version: Mapped[str | None] = mapped_column(String(100))
    analysis_image_sha256: Mapped[str | None] = mapped_column(String(64))
    bitable_curve_file_token: Mapped[str | None] = mapped_column(String(200))
    bitable_curve_source_result_id: Mapped[str | None] = mapped_column(String(64))
    bitable_curve_renderer_version: Mapped[str | None] = mapped_column(String(100))
    bitable_curve_sha256: Mapped[str | None] = mapped_column(String(64))
    bitable_curve_template: Mapped[str | None] = mapped_column(String(100))
    result_card_message_id: Mapped[str | None] = mapped_column(String(200))
    report_file_key: Mapped[str | None] = mapped_column(String(200))
    report_message_id: Mapped[str | None] = mapped_column(String(200))
    report_card_message_id: Mapped[str | None] = mapped_column(String(200))
    bitable_record_id: Mapped[str | None] = mapped_column(String(200))
    job_created_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    job_updated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    job_completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class FeishuScenarioContextRow(Base):
    __tablename__ = "feishu_scenario_contexts"
    __table_args__ = (
        Index("ix_feishu_scenario_contexts_task_created", "task_type", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(100), nullable=False)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    data_batch_id: Mapped[str] = mapped_column(String(200), nullable=False)
    route_id: Mapped[str] = mapped_column(String(200), nullable=False)
    verified_context_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    analysis_input_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    input_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_by_reference: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


__all__ = ["Base", "User", "utc_now"]
