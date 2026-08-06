"""Transactional project bindings for persisted, integrity-checked ToolResults."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4, uuid5

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from quanxin_life.audit.numeric_firewall import (
    AuditLedgerError,
    DuplicateAuditResultError,
    InvalidAuditResultError,
)
from quanxin_life.audit.project_ledger import (
    MaterializedProjectResult,
    ProjectContextValidator,
    ProjectMaterializationCommit,
    ProjectMaterializationReceipt,
    ProjectToolResultBinding,
)
from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    AdvancedModelTask,
    AgentDispatchStatus,
    AgentRunStatus,
    AgentStepStatus,
    ProjectStatus,
    ProvenanceRecord,
    SessionStatus,
    SourceKind,
    ToolResult,
    UserRole,
    UserStatus,
)
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import (
    AdvancedCalibrationMaterialization,
    AdvancedCalibrationSampleBinding,
    AgentEvent,
    AgentRun,
    AgentRunDispatch,
    AgentStep,
    ApprovalAction,
    ApprovalRequestRow,
    ModelManifest,
    ModelRouteActivationEvent,
    ModelRouteActivationStreamHead,
    Project,
    ProjectToolResultBindingRecord,
    ProvenanceRecordRow,
    SessionRecord,
    ToolResultRecord,
    User,
    UserProjectRole,
)

if TYPE_CHECKING:
    from quanxin_life.application.agent_run_invocation import (
        AgentRunInvocationValidator,
        VerifiedAgentRunInvocationGrant,
    )
    from quanxin_life.application.invocation_context import (
        VerifiedProjectInvocationContext,
    )

PROJECT_RESULT_BINDING_SCHEMA_VERSION = "project-tool-result-binding-v1"
AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION = "project-tool-result-binding-v2"
ADVANCED_CALIBRATION_SAMPLE_MANIFEST_SCHEMA_VERSION = (
    "advanced-calibration-sample-manifest-v1"
)
ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION = (
    "advanced-calibration-sample-producer-v1"
)
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _ConcurrentMaterializationCommit(RuntimeError):
    """Exit a stale SQLite snapshot before re-reading an exact committed retry."""


class SqlProjectAuditLedger:
    """Persist ToolResult bytes and project ownership in one SQL transaction."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        context_validator: ProjectContextValidator,
        clock: Clock = _utc_now,
    ) -> None:
        if not callable(getattr(context_validator, "revalidate", None)):
            raise TypeError("context_validator must revalidate project contexts")
        self._session_factory = session_factory
        self._context_validator = context_validator
        self._clock = clock

    @property
    def context_validator(self) -> ProjectContextValidator:
        return self._context_validator

    def register_result(
        self,
        context: VerifiedProjectInvocationContext,
        result: ToolResult,
    ) -> ToolResult:
        """Atomically append one normalized result, provenance and project binding."""

        verified = self._require_context(context)
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        if verified.invocation_source is ProjectInvocationSource.AGENT:
            raise AuditLedgerError(
                "AGENT project results require an exact Agent step commit"
            )
        normalized = self._normalized_result(result)
        created_at = self._utc(self._clock())
        result_sha256 = sha256_canonical(normalized.model_dump(mode="json"))
        binding_values = self._binding_values(
            verified,
            normalized,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        try:
            with session_scope(self._session_factory) as session:
                if (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                ):
                    raise DuplicateAuditResultError(
                        f"duplicate ToolResult result_id: {normalized.result_id}"
                    )
                tool_result = ToolResultRecord(
                    id=normalized.result_id,
                    run_id=verified.agent_run_id,
                    agent_step_id=None,
                    tool_name=normalized.tool_name,
                    tool_version=normalized.tool_version,
                    model_version=normalized.model_version,
                    data_version=normalized.data_version,
                    feature_version=normalized.feature_version,
                    input_hash=normalized.input_hash,
                    values_json=dict(normalized.values),
                    uncertainty_json=(
                        dict(normalized.uncertainty)
                        if normalized.uncertainty is not None
                        else None
                    ),
                    warnings_json=list(normalized.warnings),
                    created_at=normalized.created_at,
                )
                session.add(tool_result)
                # These tables use scalar foreign keys without ORM relationships, so
                # SQLAlchemy cannot infer that the ToolResult parent must be inserted first.
                session.flush((tool_result,))
                for item in normalized.provenance:
                    session.add(
                        ProvenanceRecordRow(
                            id=str(uuid4()),
                            tool_result_id=normalized.result_id,
                            source_id=item.source_id,
                            source_kind=item.source_kind.value,
                            uri=item.uri,
                            sha256=item.sha256,
                            description=item.description,
                            created_at=item.created_at,
                        )
                    )
                session.add(ProjectToolResultBindingRecord(**binding_values))
                session.flush()
        except DuplicateAuditResultError:
            raise
        except IntegrityError as exc:
            with session_scope(self._session_factory) as session:
                duplicate_exists = (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                )
            if duplicate_exists:
                raise DuplicateAuditResultError(
                    f"duplicate ToolResult result_id: {normalized.result_id}"
                ) from exc
            raise AuditLedgerError(
                "project ToolResult persistence integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            raise AuditLedgerError("project ToolResult persistence failed") from exc
        return ToolResult.model_validate(normalized.model_dump(mode="json"))

    def commit_advanced_calibration_materialization(
        self,
        commit: ProjectMaterializationCommit,
    ) -> ProjectMaterializationReceipt:
        """Atomically persist one fenced Advanced calibration cohort."""

        normalized_samples = self._normalized_materialization_samples(commit)
        now = self._utc(self._clock())
        claim_lease = self._utc(commit.claim_lease_expires_at)
        claim_token_sha256 = sha256_canonical(
            {"claim_token": commit.claim_token}
        )
        try:
            with session_scope(self._session_factory) as session:
                project = session.scalar(
                    select(Project)
                    .where(Project.id == commit.project_id)
                    .with_for_update()
                )
                head = session.scalar(
                    select(ModelRouteActivationStreamHead)
                    .where(
                        ModelRouteActivationStreamHead.project_id
                        == commit.project_id,
                        ModelRouteActivationStreamHead.task
                        == commit.task.value,
                        ModelRouteActivationStreamHead.cutoff_cycle
                        == commit.cutoff_cycle,
                        ModelRouteActivationStreamHead.route_role
                        == commit.route_role.value,
                    )
                    .with_for_update()
                )
                materialization = session.scalar(
                    select(AdvancedCalibrationMaterialization)
                    .where(
                        AdvancedCalibrationMaterialization.id
                        == commit.materialization_id
                    )
                    .with_for_update()
                )
                self._require_materialization_scope(
                    session,
                    project=project,
                    head=head,
                    materialization=materialization,
                    commit=commit,
                    now=now,
                )
                assert materialization is not None
                if (
                    materialization.status
                    == AdvancedCalibrationMaterializationStatus.READY.value
                ):
                    return self._ready_materialization_receipt(
                        session,
                        materialization=materialization,
                        commit=commit,
                        normalized_samples=normalized_samples,
                    )
                if (
                    materialization.status
                    != AdvancedCalibrationMaterializationStatus.RUNNING.value
                ):
                    raise AuditLedgerError(
                        "Advanced calibration materialization is not RUNNING"
                    )
                if (
                    materialization.claim_token_sha256
                    != claim_token_sha256
                    or materialization.claim_attempt != commit.claim_attempt
                    or materialization.claim_lease_expires_at is None
                    or self._database_utc(
                        materialization.claim_lease_expires_at
                    )
                    != claim_lease
                    or claim_lease <= now
                ):
                    raise AuditLedgerError(
                        "Advanced calibration materialization claim is stale"
                    )
                existing_bindings = tuple(
                    session.scalars(
                        select(AdvancedCalibrationSampleBinding).where(
                            AdvancedCalibrationSampleBinding.materialization_id
                            == commit.materialization_id
                        )
                    )
                )
                if existing_bindings:
                    raise _ConcurrentMaterializationCommit

                sample_entries: list[dict[str, object]] = []
                for sample in normalized_samples:
                    self._require_materialization_sample_identity(
                        commit,
                        sample,
                    )
                    result = sample.result
                    result_sha256 = sha256_canonical(
                        result.model_dump(mode="json")
                    )
                    self._add_materialized_result(
                        session,
                        materialization=materialization,
                        sample=sample,
                        result_sha256=result_sha256,
                        created_at=now,
                    )
                    sample_entries.append(
                        {
                            "ordinal": sample.ordinal,
                            "cell_id": sample.cell_id,
                            "result_id": result.result_id,
                            "sample_sha256": result_sha256,
                        }
                    )
                manifest_sha256 = self._materialization_manifest_sha256(
                    commit,
                    tuple(sample_entries),
                )
                materialization.status = (
                    AdvancedCalibrationMaterializationStatus.READY.value
                )
                materialization.sample_count = len(normalized_samples)
                materialization.sample_manifest_sha256 = manifest_sha256
                materialization.completed_at = now
                materialization.failure_code = None
                materialization.claim_token_sha256 = None
                materialization.claim_lease_expires_at = None
                session.flush()
                receipt = ProjectMaterializationReceipt(
                    materialization_id=commit.materialization_id,
                    sample_count=len(normalized_samples),
                    sample_manifest_sha256=manifest_sha256,
                    result_ids=tuple(
                        sample.result.result_id
                        for sample in normalized_samples
                    ),
                )
        except _ConcurrentMaterializationCommit as exc:
            recovered = self._recover_ready_materialization(
                commit,
                normalized_samples,
            )
            if recovered is not None:
                return recovered
            raise AuditLedgerError(
                "Advanced calibration materialization contains partial samples"
            ) from exc
        except AuditLedgerError:
            raise
        except IntegrityError as exc:
            recovered = self._recover_ready_materialization(
                commit,
                normalized_samples,
            )
            if recovered is not None:
                return recovered
            raise AuditLedgerError(
                "Advanced calibration materialization integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            recovered = self._recover_ready_materialization(
                commit,
                normalized_samples,
            )
            if recovered is not None:
                return recovered
            raise AuditLedgerError(
                "Advanced calibration materialization persistence failed"
            ) from exc
        return receipt

    def _normalized_materialization_samples(
        self,
        commit: ProjectMaterializationCommit,
    ) -> tuple[MaterializedProjectResult, ...]:
        try:
            UUID(commit.materialization_id)
            UUID(commit.project_id)
            UUID(commit.artifact_id)
            UUID(commit.decision_event_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise AuditLedgerError(
                "Advanced calibration materialization identity is invalid"
            ) from exc
        if (
            not commit.claim_token
            or commit.claim_attempt <= 0
            or commit.ledger_sequence_number <= 0
            or not commit.samples
        ):
            raise AuditLedgerError(
                "Advanced calibration materialization claim or samples are invalid"
            )
        normalized = tuple(
            MaterializedProjectResult(
                ordinal=sample.ordinal,
                cell_id=sample.cell_id,
                result=self._normalized_result(sample.result),
            )
            for sample in commit.samples
        )
        ordinals = tuple(sample.ordinal for sample in normalized)
        cell_ids = tuple(sample.cell_id for sample in normalized)
        result_ids = tuple(sample.result.result_id for sample in normalized)
        if (
            ordinals != tuple(range(len(normalized)))
            or any(not cell_id for cell_id in cell_ids)
            or len(set(cell_ids)) != len(cell_ids)
            or len(set(result_ids)) != len(result_ids)
        ):
            raise AuditLedgerError(
                "Advanced calibration samples are not one ordered unique cohort"
            )
        return normalized

    def _require_materialization_scope(
        self,
        session: Session,
        *,
        project: Project | None,
        head: ModelRouteActivationStreamHead | None,
        materialization: AdvancedCalibrationMaterialization | None,
        commit: ProjectMaterializationCommit,
        now: datetime,
    ) -> None:
        if (
            project is None
            or project.status != ProjectStatus.ACTIVE.value
            or materialization is None
            or materialization.project_id != commit.project_id
        ):
            raise AuditLedgerError(
                "Advanced calibration materialization project is inactive"
            )
        actor = session.get(User, materialization.created_by_user_id)
        actor_session = session.get(
            SessionRecord,
            materialization.created_by_session_id,
        )
        if (
            actor is None
            or actor.status != UserStatus.ACTIVE.value
            or actor.must_change_credential
            or actor.role != UserRole.ADMIN.value
            or materialization.created_by_role != UserRole.ADMIN.value
            or actor_session is None
            or actor_session.user_id != actor.id
            or actor_session.status != SessionStatus.ACTIVE.value
            or self._database_utc(actor_session.expires_at) <= now
        ):
            raise AuditLedgerError(
                "Advanced calibration materialization ADMIN session is stale"
            )
        self._require_materialization_identity(materialization, commit)
        if (
            head is None
            or head.head_event_id != commit.decision_event_id
            or head.head_sequence != commit.ledger_sequence_number
            or head.head_event_sha256 != commit.ledger_head_sha256
        ):
            raise AuditLedgerError(
                "Advanced calibration active route changed before commit"
            )
        event = session.get(
            ModelRouteActivationEvent,
            commit.decision_event_id,
        )
        manifest = session.scalar(
            select(ModelManifest).where(
                ModelManifest.artifact_id == commit.artifact_id
            )
        )
        if (
            event is None
            or event.project_id != commit.project_id
            or event.task != commit.task.value
            or event.cutoff_cycle != commit.cutoff_cycle
            or event.route_role != commit.route_role.value
            or event.stream_sequence != commit.ledger_sequence_number
            or event.event_sha256 != commit.ledger_head_sha256
            or event.artifact_id != commit.artifact_id
            or event.manifest_sha256
            != commit.artifact_manifest_sha256
            or manifest is None
            or manifest.manifest_sha256
            != commit.artifact_manifest_sha256
            or manifest.model_version != commit.model_version
        ):
            raise AuditLedgerError(
                "Advanced calibration active route identity is invalid"
            )

    @staticmethod
    def _require_materialization_identity(
        materialization: AdvancedCalibrationMaterialization,
        commit: ProjectMaterializationCommit,
    ) -> None:
        expected = {
            "project_id": commit.project_id,
            "task": commit.task.value,
            "cutoff_cycle": commit.cutoff_cycle,
            "route_role": commit.route_role.value,
            "data_version": commit.data_version,
            "split_version": commit.split_version,
            "feature_version": commit.feature_version,
            "artifact_id": commit.artifact_id,
            "artifact_manifest_sha256": (
                commit.artifact_manifest_sha256
            ),
            "normalization_statistics_sha256": (
                commit.normalization_statistics_sha256
            ),
            "decision_event_id": commit.decision_event_id,
            "ledger_sequence_number": commit.ledger_sequence_number,
            "ledger_head_sha256": commit.ledger_head_sha256,
            "source_registration_id": commit.source_registration_id,
            "source_identity_sha256": commit.source_identity_sha256,
            "request_sha256": commit.request_sha256,
        }
        if any(
            getattr(materialization, name) != value
            for name, value in expected.items()
        ):
            raise AuditLedgerError(
                "Advanced calibration materialization frozen identity changed"
            )

    @staticmethod
    def _require_materialization_sample_identity(
        commit: ProjectMaterializationCommit,
        sample: MaterializedProjectResult,
    ) -> None:
        result = sample.result
        from quanxin_life.tools.advanced_conformal import (
            validate_advanced_calibration_sample_result,
        )

        try:
            validate_advanced_calibration_sample_result(
                result,
                task=commit.task,
            )
        except (TypeError, ValueError) as exc:
            raise AuditLedgerError(
                "Advanced calibration sample contract is invalid"
            ) from exc
        values = result.values
        if (
            set(values) != {"artifact_type", "artifact"}
            or not isinstance(values["artifact"], dict)
        ):
            raise AuditLedgerError(
                "Advanced calibration sample identity envelope is invalid"
            )
        artifact = values["artifact"]
        expected = {
            "materialization_id": commit.materialization_id,
            "source_registration_id": commit.source_registration_id,
            "source_identity_sha256": commit.source_identity_sha256,
            "split_partition": "calibration",
            "cell_id": sample.cell_id,
            "task": commit.task.value,
            "route_role": commit.route_role.value,
            "artifact_id": commit.artifact_id,
            "artifact_manifest_sha256": (
                commit.artifact_manifest_sha256
            ),
            "model_version": commit.model_version,
            "dataset_id": "MATR",
            "cutoff_cycle": commit.cutoff_cycle,
            "data_version": commit.data_version,
            "feature_version": commit.feature_version,
            "split_version": commit.split_version,
            "normalization_statistics_sha256": (
                commit.normalization_statistics_sha256
            ),
            "decision_event_id": commit.decision_event_id,
            "ledger_sequence_number": commit.ledger_sequence_number,
            "ledger_head_sha256": commit.ledger_head_sha256,
        }
        if (
            result.tool_version
            != ADVANCED_CALIBRATION_SAMPLE_PRODUCER_VERSION
            or result.model_version != commit.model_version
            or result.data_version != commit.data_version
            or result.feature_version != commit.feature_version
            or any(artifact.get(name) != value for name, value in expected.items())
        ):
            raise AuditLedgerError(
                "Advanced calibration sample identity does not match its materialization"
            )
        if commit.task is AdvancedModelTask.RUL:
            expected_tool = "predict_cycle_life"
            expected_type = (
                "quanxin_life.advanced_rul_calibration_sample.v1"
            )
        else:
            expected_tool = "predict_soh_trajectory"
            expected_type = (
                "quanxin_life.advanced_soh_calibration_sample.v1"
            )
        kinds = {item.source_kind for item in result.provenance}
        if (
            result.tool_name != expected_tool
            or values["artifact_type"] != expected_type
            or SourceKind.OBSERVED not in kinds
            or SourceKind.PREDICTED not in kinds
            or not any(
                item.source_kind is SourceKind.OBSERVED
                and item.sha256 == commit.source_identity_sha256
                for item in result.provenance
            )
            or not any(
                item.source_kind is SourceKind.PREDICTED
                and item.sha256 == commit.artifact_manifest_sha256
                for item in result.provenance
            )
        ):
            raise AuditLedgerError(
                "Advanced calibration sample provenance identity is invalid"
            )

    def _add_materialized_result(
        self,
        session: Session,
        *,
        materialization: AdvancedCalibrationMaterialization,
        sample: MaterializedProjectResult,
        result_sha256: str,
        created_at: datetime,
    ) -> None:
        result = sample.result
        tool_result = ToolResultRecord(
            id=result.result_id,
            run_id=None,
            agent_step_id=None,
            tool_name=result.tool_name,
            tool_version=result.tool_version,
            model_version=result.model_version,
            data_version=result.data_version,
            feature_version=result.feature_version,
            input_hash=result.input_hash,
            values_json=dict(result.values),
            uncertainty_json=(
                dict(result.uncertainty)
                if result.uncertainty is not None
                else None
            ),
            warnings_json=list(result.warnings),
            created_at=result.created_at,
        )
        session.add(tool_result)
        # These tables use scalar foreign keys without ORM relationships, so
        # SQLAlchemy cannot infer that the ToolResult parent must be inserted first.
        session.flush((tool_result,))
        for item in result.provenance:
            session.add(
                ProvenanceRecordRow(
                    id=str(uuid4()),
                    tool_result_id=result.result_id,
                    source_id=item.source_id,
                    source_kind=item.source_kind.value,
                    uri=item.uri,
                    sha256=item.sha256,
                    description=item.description,
                    created_at=item.created_at,
                )
            )
        binding_payload = self._binding_hash_payload(
            result_id=result.result_id,
            project_id=materialization.project_id,
            actor_user_id=materialization.created_by_user_id,
            actor_session_id=materialization.created_by_session_id,
            actor_role=materialization.created_by_role,
            invocation_source="HTTP",
            agent_run_id=None,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        session.add(
            ProjectToolResultBindingRecord(
                **(
                    binding_payload
                    | {
                        "binding_sha256": sha256_canonical(
                            binding_payload
                        ),
                        "created_at": created_at,
                    }
                )
            )
        )
        sample_binding_payload = {
            "materialization_id": materialization.id,
            "ordinal": sample.ordinal,
            "cell_id": sample.cell_id,
            "result_id": result.result_id,
            "sample_sha256": result_sha256,
        }
        session.add(
            AdvancedCalibrationSampleBinding(
                id=str(
                    uuid5(
                        UUID(materialization.id),
                        sha256_canonical(sample_binding_payload),
                    )
                ),
                **sample_binding_payload,
                created_at=created_at,
            )
        )

    def _ready_materialization_receipt(
        self,
        session: Session,
        *,
        materialization: AdvancedCalibrationMaterialization,
        commit: ProjectMaterializationCommit,
        normalized_samples: tuple[MaterializedProjectResult, ...],
    ) -> ProjectMaterializationReceipt:
        bindings = tuple(
            session.scalars(
                select(AdvancedCalibrationSampleBinding)
                .where(
                    AdvancedCalibrationSampleBinding.materialization_id
                    == materialization.id
                )
                .order_by(AdvancedCalibrationSampleBinding.ordinal)
            )
        )
        if len(bindings) != len(normalized_samples):
            raise AuditLedgerError(
                "Advanced calibration READY content is incomplete"
            )
        entries: list[dict[str, object]] = []
        for expected, sample_binding in zip(
            normalized_samples,
            bindings,
            strict=True,
        ):
            result_row = session.get(
                ToolResultRecord,
                sample_binding.result_id,
            )
            binding_row = session.get(
                ProjectToolResultBindingRecord,
                sample_binding.result_id,
            )
            provenance_rows = tuple(
                session.scalars(
                    select(ProvenanceRecordRow).where(
                        ProvenanceRecordRow.tool_result_id
                        == sample_binding.result_id
                    )
                )
            )
            if (
                result_row is None
                or binding_row is None
                or result_row.run_id is not None
                or result_row.agent_step_id is not None
                or binding_row.project_id != materialization.project_id
                or binding_row.actor_user_id
                != materialization.created_by_user_id
                or binding_row.actor_session_id
                != materialization.created_by_session_id
                or binding_row.actor_role
                != materialization.created_by_role
                or sample_binding.ordinal != expected.ordinal
                or sample_binding.cell_id != expected.cell_id
                or sample_binding.result_id != expected.result.result_id
            ):
                raise AuditLedgerError(
                    "Advanced calibration READY content is invalid"
                )
            result = self._result_from_rows(result_row, provenance_rows)
            self._verify_binding(binding_row, result)
            result_sha256 = sha256_canonical(result.model_dump(mode="json"))
            if (
                result != expected.result
                or sample_binding.sample_sha256 != result_sha256
            ):
                raise AuditLedgerError(
                    "Advanced calibration READY content conflicts with retry"
                )
            entries.append(
                {
                    "ordinal": sample_binding.ordinal,
                    "cell_id": sample_binding.cell_id,
                    "result_id": sample_binding.result_id,
                    "sample_sha256": sample_binding.sample_sha256,
                }
            )
        manifest_sha256 = self._materialization_manifest_sha256(
            commit,
            tuple(entries),
        )
        if (
            materialization.sample_count != len(bindings)
            or materialization.sample_manifest_sha256 != manifest_sha256
        ):
            raise AuditLedgerError(
                "Advanced calibration READY manifest is invalid"
            )
        return ProjectMaterializationReceipt(
            materialization_id=materialization.id,
            sample_count=len(bindings),
            sample_manifest_sha256=manifest_sha256,
            result_ids=tuple(item.result_id for item in bindings),
        )

    @staticmethod
    def _materialization_manifest_sha256(
        commit: ProjectMaterializationCommit,
        entries: tuple[dict[str, object], ...],
    ) -> str:
        payload = {
            "schema_version": (
                ADVANCED_CALIBRATION_SAMPLE_MANIFEST_SCHEMA_VERSION
            ),
            "materialization_id": commit.materialization_id,
            "project_id": commit.project_id,
            "task": commit.task.value,
            "cutoff_cycle": commit.cutoff_cycle,
            "route_role": commit.route_role.value,
            "data_version": commit.data_version,
            "split_version": commit.split_version,
            "feature_version": commit.feature_version,
            "artifact_id": commit.artifact_id,
            "artifact_manifest_sha256": (
                commit.artifact_manifest_sha256
            ),
            "model_version": commit.model_version,
            "normalization_statistics_sha256": (
                commit.normalization_statistics_sha256
            ),
            "decision_event_id": commit.decision_event_id,
            "ledger_sequence_number": commit.ledger_sequence_number,
            "ledger_head_sha256": commit.ledger_head_sha256,
            "source_registration_id": commit.source_registration_id,
            "source_identity_sha256": commit.source_identity_sha256,
            "request_sha256": commit.request_sha256,
            "samples": list(entries),
        }
        return sha256_canonical(payload)

    def _recover_ready_materialization(
        self,
        commit: ProjectMaterializationCommit,
        normalized_samples: tuple[MaterializedProjectResult, ...],
    ) -> ProjectMaterializationReceipt | None:
        try:
            with self._session_factory() as session:
                materialization = session.get(
                    AdvancedCalibrationMaterialization,
                    commit.materialization_id,
                )
                if (
                    materialization is None
                    or materialization.status
                    != AdvancedCalibrationMaterializationStatus.READY.value
                ):
                    return None
                self._require_materialization_identity(
                    materialization,
                    commit,
                )
                return self._ready_materialization_receipt(
                    session,
                    materialization=materialization,
                    commit=commit,
                    normalized_samples=normalized_samples,
                )
        except (AuditLedgerError, SQLAlchemyError):
            return None

    @classmethod
    def _require_live_agent_scope(
        cls,
        session: Session,
        *,
        grant: VerifiedAgentRunInvocationGrant,
        run: AgentRun | None,
        dispatch: AgentRunDispatch | None,
        step: AgentStep | None,
        now: datetime,
    ) -> None:
        context = grant.project_context
        actor = session.get(User, context.actor_user_id)
        actor_session = session.get(SessionRecord, context.actor_session_id)
        project = session.get(Project, context.project_id)
        membership = session.scalar(
            select(UserProjectRole).where(
                UserProjectRole.user_id == context.actor_user_id,
                UserProjectRole.project_id == context.project_id,
            )
        )
        visible = project is not None and (
            project.owner_user_id == context.actor_user_id or membership is not None
        )
        if (
            actor is None
            or actor.status != UserStatus.ACTIVE.value
            or actor.must_change_credential
            or actor.role != context.actor_role.value
            or actor_session is None
            or actor_session.user_id != context.actor_user_id
            or actor_session.status != SessionStatus.ACTIVE.value
            or cls._database_utc(actor_session.expires_at) <= now
            or project is None
            or project.status != ProjectStatus.ACTIVE.value
            or not visible
            or run is None
            or run.status != AgentRunStatus.RUNNING.value
            or run.project_id != context.project_id
            or run.created_by_user_id != context.actor_user_id
            or run.session_id != context.actor_session_id
            or run.plan_hash != grant.plan_hash
            or dispatch is None
            or dispatch.status != AgentDispatchStatus.DISPATCHED.value
            or dispatch.plan_hash != grant.plan_hash
            or step is None
            or step.run_id != grant.agent_run_id
            or step.step_id != grant.step_id
            or step.tool_name
            != next(iter(grant.allowed_tool_names)).value
            or step.status != AgentStepStatus.RUNNING.value
            or step.attempts != grant.claim_attempt
            or not step.claim_token
            or sha256_canonical({"claim_token": step.claim_token})
            != grant._claim_token_sha256
            or step.execution_claim_sha256 != grant._claim_token_sha256
            or step.lease_expires_at is None
            or cls._database_utc(step.lease_expires_at) <= now
            or cls._database_utc(step.lease_expires_at)
            != grant.claim_lease_expires_at
            or step.resolved_input_hash != grant.input_hash
            or step.execution_snapshot_sha256
            not in {None, grant.execution_snapshot_sha256}
            or step.dependency_evidence_sha256
            not in {None, grant.dependency_evidence_sha256}
        ):
            raise AuditLedgerError("Agent step execution claim is stale")
        if grant.approval_required:
            request = session.get(ApprovalRequestRow, grant.approval_request_id)
            action = session.get(ApprovalAction, grant.approval_action_id)
            if (
                request is None
                or action is None
                or request.run_id != grant.agent_run_id
                or request.step_id != grant.step_id
                or request.source_plan_hash != grant.plan_hash
                or request.status != "APPROVED"
                or request.agent_step_id != grant.agent_step_row_id
                or request.execution_snapshot_sha256
                != grant.execution_snapshot_sha256
                or action.approval_request_id != request.id
                or action.action != "APPROVED"
                or cls._database_utc(action.acted_at)
                >= cls._database_utc(request.expires_at)
            ):
                raise AuditLedgerError("Agent step approval evidence is stale")

    @staticmethod
    def _database_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def commit_agent_step_result(
        self,
        *,
        grant: VerifiedAgentRunInvocationGrant,
        result: ToolResult,
        grant_validator: AgentRunInvocationValidator,
    ) -> ToolResult:
        """Atomically fence, bind and complete one exact Agent step result."""

        verified = grant_validator.revalidate(grant)
        normalized = self._normalized_result(result)
        now = self._utc(self._clock())
        if (
            normalized.tool_name
            != next(iter(verified.allowed_tool_names)).value
            or normalized.input_hash != verified.input_hash
        ):
            raise AuditLedgerError(
                "Agent ToolResult does not match the frozen step invocation"
            )
        try:
            with session_scope(self._session_factory) as session:
                run = session.scalar(
                    select(AgentRun)
                    .where(AgentRun.id == verified.agent_run_id)
                    .with_for_update()
                )
                dispatch = session.scalar(
                    select(AgentRunDispatch)
                    .where(AgentRunDispatch.run_id == verified.agent_run_id)
                    .with_for_update()
                )
                step = session.scalar(
                    select(AgentStep)
                    .where(AgentStep.id == verified.agent_step_row_id)
                    .with_for_update()
                )
                self._require_live_agent_scope(
                    session,
                    grant=verified,
                    run=run,
                    dispatch=dispatch,
                    step=step,
                    now=now,
                )
                try:
                    grant_validator.revalidate_in_session(session, verified)
                except RuntimeError as exc:
                    raise AuditLedgerError(
                        "Agent step frozen input, approval, or dependency evidence is stale"
                    ) from exc
                if (
                    session.get(ToolResultRecord, normalized.result_id) is not None
                    or session.get(
                        ProjectToolResultBindingRecord, normalized.result_id
                    )
                    is not None
                ):
                    raise DuplicateAuditResultError(
                        f"duplicate ToolResult result_id: {normalized.result_id}"
                    )
                assert run is not None and step is not None
                tool_result = ToolResultRecord(
                    id=normalized.result_id,
                    run_id=verified.agent_run_id,
                    agent_step_id=verified.agent_step_row_id,
                    tool_name=normalized.tool_name,
                    tool_version=normalized.tool_version,
                    model_version=normalized.model_version,
                    data_version=normalized.data_version,
                    feature_version=normalized.feature_version,
                    input_hash=normalized.input_hash,
                    values_json=dict(normalized.values),
                    uncertainty_json=(
                        dict(normalized.uncertainty)
                        if normalized.uncertainty is not None
                        else None
                    ),
                    warnings_json=list(normalized.warnings),
                    created_at=normalized.created_at,
                )
                session.add(tool_result)
                # These tables use scalar foreign keys without ORM relationships, so
                # SQLAlchemy cannot infer that the ToolResult parent must be inserted first.
                session.flush((tool_result,))
                for item in normalized.provenance:
                    session.add(
                        ProvenanceRecordRow(
                            id=str(uuid4()),
                            tool_result_id=normalized.result_id,
                            source_id=item.source_id,
                            source_kind=item.source_kind.value,
                            uri=item.uri,
                            sha256=item.sha256,
                            description=item.description,
                            created_at=item.created_at,
                        )
                    )
                result_sha256 = sha256_canonical(
                    normalized.model_dump(mode="json")
                )
                binding_values = self._agent_binding_values(
                    verified,
                    normalized,
                    result_sha256=result_sha256,
                    created_at=now,
                )
                session.add(ProjectToolResultBindingRecord(**binding_values))
                if step.execution_snapshot_sha256 is None:
                    step.execution_snapshot_sha256 = (
                        verified.execution_snapshot_sha256
                    )
                if step.dependency_evidence_sha256 is None:
                    step.dependency_evidence_sha256 = (
                        verified.dependency_evidence_sha256
                    )
                step.status = AgentStepStatus.COMPLETED.value
                step.claim_token = None
                step.lease_expires_at = None
                step.last_error_code = None
                step.completed_at = now
                run.updated_at = now
                sequence = session.scalar(
                    select(func.max(AgentEvent.sequence)).where(
                        AgentEvent.run_id == verified.agent_run_id
                    )
                )
                session.add(
                    AgentEvent(
                        id=str(uuid4()),
                        run_id=verified.agent_run_id,
                        sequence=(sequence or 0) + 1,
                        event_type="STEP_COMPLETED",
                        payload_json={
                            "step_id": verified.step_id,
                            "tool_name": normalized.tool_name,
                            "result_id": normalized.result_id,
                        },
                        created_at=now,
                    )
                )
                session.flush()
        except (DuplicateAuditResultError, AuditLedgerError):
            raise
        except IntegrityError as exc:
            raise AuditLedgerError(
                "Agent step result persistence integrity check failed"
            ) from exc
        except SQLAlchemyError as exc:
            raise AuditLedgerError("Agent step result persistence failed") from exc
        return ToolResult.model_validate(normalized.model_dump(mode="json"))

    def resolve_registered_result(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ToolResult:
        """Resolve one detached result only inside its live project scope."""

        verified = self._require_context(context)
        result, _binding = self._resolve_rows(verified, result_id)
        return result

    def resolve_binding(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> ProjectToolResultBinding:
        """Resolve immutable non-numeric ownership metadata after full verification."""

        verified = self._require_context(context)
        _result, binding = self._resolve_rows(verified, result_id)
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        try:
            actor_role = UserRole(binding.actor_role)
            invocation_source = ProjectInvocationSource(binding.invocation_source)
        except ValueError as exc:  # pragma: no cover - checked in _resolve_rows
            raise AuditLedgerError("project ToolResult binding integrity check failed") from exc
        return ProjectToolResultBinding(
            result_id=binding.result_id,
            project_id=binding.project_id,
            actor_user_id=binding.actor_user_id,
            actor_session_id=binding.actor_session_id,
            actor_role=actor_role,
            invocation_source=invocation_source,
            agent_run_id=binding.agent_run_id,
            tool_name=binding.tool_name,
            input_hash=binding.input_hash,
            agent_step_id=binding.agent_step_id,
            step_id=binding.step_id,
            plan_hash=binding.plan_hash,
            claim_attempt=binding.claim_attempt,
            execution_snapshot_sha256=binding.execution_snapshot_sha256,
            approval_request_id=binding.approval_request_id,
        )

    def _resolve_rows(
        self,
        context: VerifiedProjectInvocationContext,
        result_id: str,
    ) -> tuple[ToolResult, ProjectToolResultBindingRecord]:
        normalized_id = self._result_id(result_id)
        try:
            with session_scope(self._session_factory) as session:
                binding = session.scalar(
                    select(ProjectToolResultBindingRecord).where(
                        ProjectToolResultBindingRecord.result_id == normalized_id,
                        ProjectToolResultBindingRecord.project_id == context.project_id,
                    )
                )
                result_row = session.get(ToolResultRecord, normalized_id)
                if binding is None or result_row is None:
                    raise ValueError("ToolResult is not registered for this project")
                if result_row.run_id != binding.agent_run_id:
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                provenance_rows = tuple(
                    session.scalars(
                        select(ProvenanceRecordRow).where(
                            ProvenanceRecordRow.tool_result_id == normalized_id
                        )
                    ).all()
                )
                actor_session = session.get(SessionRecord, binding.actor_session_id)
                if (
                    actor_session is None
                    or actor_session.user_id != binding.actor_user_id
                ):
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                if binding.invocation_source == "AGENT":
                    agent_run = session.get(AgentRun, binding.agent_run_id)
                    agent_step = session.get(AgentStep, binding.agent_step_id)
                    if (
                        agent_run is None
                        or agent_run.project_id != binding.project_id
                        or agent_run.created_by_user_id != binding.actor_user_id
                        or agent_run.session_id != binding.actor_session_id
                        or agent_step is None
                        or agent_step.run_id != binding.agent_run_id
                        or agent_step.step_id != binding.step_id
                        or result_row.agent_step_id != binding.agent_step_id
                        or agent_step.execution_claim_sha256
                        != binding.claim_token_sha256
                        or agent_step.resolved_input_hash != binding.input_hash
                        or agent_step.execution_snapshot_sha256
                        != binding.execution_snapshot_sha256
                        or agent_step.dependency_evidence_sha256
                        != binding.dependency_evidence_sha256
                    ):
                        raise AuditLedgerError(
                            "project ToolResult binding integrity check failed"
                        )
                elif result_row.agent_step_id is not None:
                    raise AuditLedgerError(
                        "project ToolResult binding integrity check failed"
                    )
                result = self._result_from_rows(result_row, provenance_rows)
                self._verify_binding(binding, result)
                detached_binding = ProjectToolResultBindingRecord(
                    **{
                        column.name: getattr(binding, column.name)
                        for column in ProjectToolResultBindingRecord.__table__.columns
                    }
                )
        except (ValueError, AuditLedgerError):
            raise
        except (SQLAlchemyError, TypeError) as exc:
            raise AuditLedgerError(
                "project ToolResult persistence could not be verified"
            ) from exc
        return result, detached_binding

    @classmethod
    def verify_persisted_binding(
        cls,
        binding: ProjectToolResultBindingRecord,
        result: ToolResult,
    ) -> None:
        """Verify a detached binding for an internal completed-run reader."""

        cls._verify_binding(binding, result)

    @classmethod
    def rebuild_persisted_result(
        cls,
        row: ToolResultRecord,
        provenance_rows: tuple[ProvenanceRecordRow, ...],
    ) -> ToolResult:
        """Rebuild a stored ToolResult with the ledger's canonical ordering."""

        return cls._result_from_rows(row, provenance_rows)

    @classmethod
    def _verify_binding(
        cls,
        binding: ProjectToolResultBindingRecord,
        result: ToolResult,
    ) -> None:
        from quanxin_life.application.invocation_context import (
            ProjectInvocationSource,
        )

        try:
            actor_role = UserRole(binding.actor_role)
            invocation_source = ProjectInvocationSource(binding.invocation_source)
        except ValueError as exc:
            raise AuditLedgerError(
                "project ToolResult binding integrity check failed"
            ) from exc
        if (
            binding.tool_name != result.tool_name
            or binding.input_hash != result.input_hash
            or binding.result_sha256
            != sha256_canonical(result.model_dump(mode="json"))
        ):
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        if binding.binding_schema_version == PROJECT_RESULT_BINDING_SCHEMA_VERSION:
            if (
                invocation_source is not ProjectInvocationSource.HTTP
                or binding.agent_run_id is not None
                or any(
                    value is not None
                    for value in (
                        binding.agent_step_id,
                        binding.step_id,
                        binding.plan_hash,
                        binding.claim_token_sha256,
                        binding.claim_attempt,
                        binding.claim_lease_expires_at,
                        binding.execution_snapshot_sha256,
                        binding.dependency_evidence_sha256,
                        binding.approval_required,
                        binding.approval_request_id,
                        binding.approval_action_id,
                        binding.approval_evidence_sha256,
                    )
                )
            ):
                raise AuditLedgerError(
                    "project ToolResult binding integrity check failed"
                )
            payload = cls._binding_hash_payload(
                result_id=binding.result_id,
                project_id=binding.project_id,
                actor_user_id=binding.actor_user_id,
                actor_session_id=binding.actor_session_id,
                actor_role=actor_role.value,
                invocation_source=invocation_source.value,
                agent_run_id=binding.agent_run_id,
                tool_name=binding.tool_name,
                input_hash=binding.input_hash,
                result_sha256=binding.result_sha256,
                created_at=binding.created_at,
            )
        elif (
            binding.binding_schema_version
            == AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION
        ):
            if (
                invocation_source is not ProjectInvocationSource.AGENT
                or not binding.agent_run_id
                or not binding.agent_step_id
                or not binding.step_id
                or not binding.plan_hash
                or not binding.claim_token_sha256
                or binding.claim_attempt is None
                or binding.claim_lease_expires_at is None
                or not binding.execution_snapshot_sha256
                or not binding.dependency_evidence_sha256
                or binding.approval_required is None
                or not binding.approval_evidence_sha256
            ):
                raise AuditLedgerError(
                    "project ToolResult binding integrity check failed"
                )
            payload = cls._agent_binding_hash_payload(
                result_id=binding.result_id,
                project_id=binding.project_id,
                actor_user_id=binding.actor_user_id,
                actor_session_id=binding.actor_session_id,
                actor_role=actor_role.value,
                agent_run_id=binding.agent_run_id,
                agent_step_id=binding.agent_step_id,
                step_id=binding.step_id,
                plan_hash=binding.plan_hash,
                claim_token_sha256=binding.claim_token_sha256,
                claim_attempt=binding.claim_attempt,
                claim_lease_expires_at=binding.claim_lease_expires_at,
                execution_snapshot_sha256=binding.execution_snapshot_sha256,
                dependency_evidence_sha256=binding.dependency_evidence_sha256,
                approval_required=binding.approval_required,
                approval_request_id=binding.approval_request_id,
                approval_action_id=binding.approval_action_id,
                approval_evidence_sha256=binding.approval_evidence_sha256,
                tool_name=binding.tool_name,
                input_hash=binding.input_hash,
                result_sha256=binding.result_sha256,
                created_at=binding.created_at,
            )
        else:
            raise AuditLedgerError("project ToolResult binding integrity check failed")
        expected_hash = sha256_canonical(payload)
        if expected_hash != binding.binding_sha256:
            raise AuditLedgerError("project ToolResult binding integrity check failed")

    @classmethod
    def _agent_binding_values(
        cls,
        grant: VerifiedAgentRunInvocationGrant,
        result: ToolResult,
        *,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        context = grant.project_context
        payload = cls._agent_binding_hash_payload(
            result_id=result.result_id,
            project_id=context.project_id,
            actor_user_id=context.actor_user_id,
            actor_session_id=context.actor_session_id,
            actor_role=context.actor_role.value,
            agent_run_id=grant.agent_run_id,
            agent_step_id=grant.agent_step_row_id,
            step_id=grant.step_id,
            plan_hash=grant.plan_hash,
            claim_token_sha256=grant._claim_token_sha256,
            claim_attempt=grant.claim_attempt,
            claim_lease_expires_at=grant.claim_lease_expires_at,
            execution_snapshot_sha256=grant.execution_snapshot_sha256,
            dependency_evidence_sha256=grant.dependency_evidence_sha256,
            approval_required=grant.approval_required,
            approval_request_id=grant.approval_request_id,
            approval_action_id=grant.approval_action_id,
            approval_evidence_sha256=grant.approval_evidence_sha256,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        return payload | {
            "binding_sha256": sha256_canonical(payload),
            "claim_lease_expires_at": grant.claim_lease_expires_at,
            "created_at": created_at,
        }

    @staticmethod
    def _agent_binding_hash_payload(
        *,
        result_id: str,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
        actor_role: str,
        agent_run_id: str,
        agent_step_id: str,
        step_id: str,
        plan_hash: str,
        claim_token_sha256: str,
        claim_attempt: int,
        claim_lease_expires_at: datetime,
        execution_snapshot_sha256: str,
        dependency_evidence_sha256: str,
        approval_required: bool,
        approval_request_id: str | None,
        approval_action_id: str | None,
        approval_evidence_sha256: str,
        tool_name: str,
        input_hash: str,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        return {
            "result_id": result_id,
            "binding_schema_version": AGENT_PROJECT_RESULT_BINDING_SCHEMA_VERSION,
            "project_id": project_id,
            "actor_user_id": actor_user_id,
            "actor_session_id": actor_session_id,
            "actor_role": actor_role,
            "invocation_source": "AGENT",
            "agent_run_id": agent_run_id,
            "agent_step_id": agent_step_id,
            "step_id": step_id,
            "plan_hash": plan_hash,
            "claim_token_sha256": claim_token_sha256,
            "claim_attempt": claim_attempt,
            "claim_lease_expires_at": claim_lease_expires_at.astimezone(UTC).isoformat(),
            "execution_snapshot_sha256": execution_snapshot_sha256,
            "dependency_evidence_sha256": dependency_evidence_sha256,
            "approval_required": approval_required,
            "approval_request_id": approval_request_id,
            "approval_action_id": approval_action_id,
            "approval_evidence_sha256": approval_evidence_sha256,
            "tool_name": tool_name,
            "input_hash": input_hash,
            "result_sha256": result_sha256,
            "created_at": created_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def _binding_values(
        cls,
        context: VerifiedProjectInvocationContext,
        result: ToolResult,
        *,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        payload = cls._binding_hash_payload(
            result_id=result.result_id,
            project_id=context.project_id,
            actor_user_id=context.actor_user_id,
            actor_session_id=context.actor_session_id,
            actor_role=context.actor_role.value,
            invocation_source=context.invocation_source.value,
            agent_run_id=context.agent_run_id,
            tool_name=result.tool_name,
            input_hash=result.input_hash,
            result_sha256=result_sha256,
            created_at=created_at,
        )
        return payload | {
            "binding_sha256": sha256_canonical(payload),
            "created_at": created_at,
        }

    @staticmethod
    def _binding_hash_payload(
        *,
        result_id: str,
        project_id: str,
        actor_user_id: str,
        actor_session_id: str,
        actor_role: str,
        invocation_source: str,
        agent_run_id: str | None,
        tool_name: str,
        input_hash: str,
        result_sha256: str,
        created_at: datetime,
    ) -> dict[str, object]:
        return {
            "result_id": result_id,
            "binding_schema_version": PROJECT_RESULT_BINDING_SCHEMA_VERSION,
            "project_id": project_id,
            "actor_user_id": actor_user_id,
            "actor_session_id": actor_session_id,
            "actor_role": actor_role,
            "invocation_source": invocation_source,
            "agent_run_id": agent_run_id,
            "tool_name": tool_name,
            "input_hash": input_hash,
            "result_sha256": result_sha256,
            "created_at": created_at.astimezone(UTC).isoformat(),
        }

    @classmethod
    def _result_from_rows(
        cls,
        row: ToolResultRecord,
        provenance_rows: tuple[ProvenanceRecordRow, ...],
    ) -> ToolResult:
        try:
            result = ToolResult(
                result_id=row.id,
                tool_name=row.tool_name,
                tool_version=row.tool_version,
                model_version=row.model_version,
                data_version=row.data_version,
                feature_version=row.feature_version,
                input_hash=row.input_hash,
                values=row.values_json,
                uncertainty=row.uncertainty_json,
                warnings=row.warnings_json,
                provenance=[
                    ProvenanceRecord(
                        source_id=item.source_id,
                        source_kind=SourceKind(item.source_kind),
                        uri=item.uri,
                        sha256=item.sha256,
                        description=item.description,
                        created_at=cls._utc(item.created_at),
                    )
                    for item in provenance_rows
                ],
                created_at=cls._utc(row.created_at),
            )
        except (TypeError, ValueError) as exc:
            raise AuditLedgerError("project ToolResult integrity check failed") from exc
        return cls._normalized_result(result)

    @staticmethod
    def _normalized_result(result: ToolResult) -> ToolResult:
        try:
            validated = ToolResult.model_validate(result.model_dump(mode="json"))
        except (AttributeError, TypeError, ValueError) as exc:
            raise InvalidAuditResultError(
                "ToolResult does not satisfy the public audit contract"
            ) from exc
        payload = validated.model_dump(mode="json")
        payload["provenance"] = sorted(
            payload["provenance"],
            key=lambda item: (
                item["source_id"],
                item["source_kind"],
                item["uri"],
                item["sha256"],
                item["description"],
                item["created_at"],
            ),
        )
        return ToolResult.model_validate(payload)

    def _require_context(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        from quanxin_life.application.invocation_context import (
            VerifiedProjectInvocationContext,
        )

        if not isinstance(context, VerifiedProjectInvocationContext):
            raise TypeError("verified project invocation context is required")
        return self._context_validator.revalidate(context)

    @staticmethod
    def _result_id(value: str) -> str:
        try:
            UUID(value)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("result_id must be a UUID string") from exc
        return value

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise AuditLedgerError("project ToolResult timestamp must include a timezone")
        return value.astimezone(UTC)


__all__ = ["PROJECT_RESULT_BINDING_SCHEMA_VERSION", "SqlProjectAuditLedger"]
