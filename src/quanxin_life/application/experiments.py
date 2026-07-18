"""Project-scoped registration and query of verified A100 experiment metadata."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator
from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from quanxin_life.application.a100_suite_import import (
    ImportedA100SuiteRecord,
    ImportedA100TaskRecord,
    RegisteredA100Suite,
)
from quanxin_life.application.projects import ProjectService
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import PredictionTarget, ProjectStatus, UserRole
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import ExperimentRun, ExperimentSuite, Project


class ExperimentAccessError(RuntimeError):
    """Raised when a role cannot mutate the experiment registry."""


class ExperimentNotFoundError(RuntimeError):
    """Used for absent and invisible projects or experiment suites."""


class ExperimentSourceError(RuntimeError):
    """Raised when the verified A100 source cannot support registration."""


class ExperimentStateError(RuntimeError):
    """Raised when persisted experiment metadata violates its source context."""


class A100SuiteCatalogSource(Protocol):
    """Small read-only port implemented by the safe A100 suite importer."""

    def resolve(self, import_id: str) -> RegisteredA100Suite: ...

    def list_tasks(self, import_id: str) -> tuple[ImportedA100TaskRecord, ...]: ...


class ExperimentSuiteRecord(ContractModel):
    """Structured suite identity; numeric training metrics remain in evidence files."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    import_id: Sha256
    dataset_id: str = Field(min_length=1)
    target: PredictionTarget
    mode: Literal["smoke", "final"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    output_sha256: Sha256
    transfer_sha256: Sha256
    task_count: int = Field(gt=0)
    file_count: int = Field(gt=0)
    formal_performance_claim: bool
    evidence_uri: str = Field(pattern=r"^a100-suite://[0-9a-f]{64}$")
    created_by_user_id: str = Field(min_length=1)
    imported_at: datetime
    registered_at: datetime

    @field_validator("imported_at", "registered_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return _utc(value)


class ExperimentRunRecord(ContractModel):
    """One task identity linked to immutable A100 evidence, without metric copies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_run_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    import_id: Sha256
    run_id: str = Field(min_length=1, max_length=200)
    dataset_id: str = Field(min_length=1)
    target: PredictionTarget
    model_name: str = Field(min_length=1, max_length=100)
    cutoff_cycle: int = Field(gt=0)
    seed: int = Field(gt=0)
    config_sha256: Sha256
    input_bundle_sha256: Sha256
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    context_sha256: Sha256
    task_relative_root: str = Field(min_length=1)
    evidence_uri: str = Field(min_length=1)
    completed_at: datetime
    registered_at: datetime

    @field_validator("completed_at", "registered_at")
    @classmethod
    def normalize_time(cls, value: datetime) -> datetime:
        return _utc(value)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("experiment timestamp must include a timezone")
    return value.astimezone(UTC)


def _suite_evidence_uri(import_id: str) -> str:
    return f"a100-suite://{import_id}"


def _task_evidence_uri(import_id: str, relative_root: str) -> str:
    return f"{_suite_evidence_uri(import_id)}/{relative_root}"


def _suite_record(row: ExperimentSuite) -> ExperimentSuiteRecord:
    try:
        target = PredictionTarget(row.target)
    except ValueError as exc:
        raise ExperimentStateError("experiment target is unsupported") from exc
    if row.mode == "smoke":
        mode: Literal["smoke", "final"] = "smoke"
    elif row.mode == "final":
        mode = "final"
    else:
        raise ExperimentStateError("experiment mode is unsupported")
    return ExperimentSuiteRecord(
        experiment_id=row.id,
        project_id=row.project_id,
        import_id=row.import_id,
        dataset_id=row.dataset_id,
        target=target,
        mode=mode,
        source_commit=row.source_commit,
        config_sha256=row.config_sha256,
        input_bundle_sha256=row.input_bundle_sha256,
        data_version=row.data_version,
        split_version=row.split_version,
        feature_version=row.feature_version,
        output_sha256=row.output_sha256,
        transfer_sha256=row.transfer_sha256,
        task_count=row.task_count,
        file_count=row.file_count,
        formal_performance_claim=row.formal_performance_claim,
        evidence_uri=row.evidence_uri,
        created_by_user_id=row.created_by_user_id,
        imported_at=row.imported_at,
        registered_at=row.registered_at,
    )


def _run_record(
    row: ExperimentRun,
    *,
    project_id: str,
    import_id: str,
) -> ExperimentRunRecord:
    try:
        target = PredictionTarget(row.target)
    except ValueError as exc:
        raise ExperimentStateError("experiment run target is unsupported") from exc
    return ExperimentRunRecord(
        experiment_run_id=row.id,
        experiment_id=row.suite_id,
        project_id=project_id,
        import_id=import_id,
        run_id=row.run_id,
        dataset_id=row.dataset_id,
        target=target,
        model_name=row.model_name,
        cutoff_cycle=row.cutoff_cycle,
        seed=row.seed,
        config_sha256=row.config_sha256,
        input_bundle_sha256=row.input_bundle_sha256,
        source_commit=row.source_commit,
        data_version=row.data_version,
        split_version=row.split_version,
        feature_version=row.feature_version,
        context_sha256=row.context_sha256,
        task_relative_root=row.task_relative_root,
        evidence_uri=row.evidence_uri,
        completed_at=row.completed_at,
        registered_at=row.registered_at,
    )


class ExperimentRegistryService:
    """Register trusted suite identities and query them under project authorization."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        source: A100SuiteCatalogSource,
    ) -> None:
        self._session_factory = session_factory
        self._source = source

    def register_suite(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str,
        import_id: str,
        registered_at: datetime,
    ) -> ExperimentSuiteRecord:
        self._require_operator(principal)
        normalized_project_id = self._identifier(project_id)
        normalized_import_id = self._identifier(import_id)
        timestamp = _utc(registered_at)
        self._assert_active_project(principal, normalized_project_id)
        suite, tasks = self._load_source(normalized_import_id)

        with session_scope(self._session_factory) as session:
            if session.scalar(
                self._visible_active_project_statement(
                    principal, normalized_project_id
                )
            ) is None:
                raise ExperimentNotFoundError("project was not found")
            existing = session.scalar(
                select(ExperimentSuite).where(
                    ExperimentSuite.project_id == normalized_project_id,
                    ExperimentSuite.import_id == suite.import_id,
                )
            )
            if existing is not None:
                self._assert_persisted_context(session, existing, suite, tasks)
                return _suite_record(existing)

            row = ExperimentSuite(
                id=str(uuid4()),
                project_id=normalized_project_id,
                import_id=suite.import_id,
                dataset_id=suite.dataset_id,
                target=suite.target,
                mode=suite.mode,
                source_commit=suite.source_commit,
                config_sha256=suite.config_sha256,
                input_bundle_sha256=suite.input_bundle_sha256,
                data_version=suite.data_version,
                split_version=suite.split_version,
                feature_version=suite.feature_version,
                output_sha256=suite.output_sha256,
                transfer_sha256=suite.transfer_sha256,
                task_count=suite.task_count,
                file_count=suite.file_count,
                formal_performance_claim=suite.formal_performance_claim,
                evidence_uri=_suite_evidence_uri(suite.import_id),
                created_by_user_id=principal.user_id,
                imported_at=suite.imported_at,
                registered_at=timestamp,
            )
            session.add(row)
            session.flush()
            session.add_all(
                self._task_rows(row.id, suite, tasks, registered_at=timestamp)
            )
            session.flush()
            return _suite_record(row)

    def list_suites(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str | None = None,
        dataset_id: str | None = None,
        mode: Literal["smoke", "final"] | None = None,
    ) -> tuple[ExperimentSuiteRecord, ...]:
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        statement = select(ExperimentSuite).where(
            ExperimentSuite.project_id.in_(visible_project_ids)
        )
        if project_id is not None:
            statement = statement.where(
                ExperimentSuite.project_id == self._identifier(project_id)
            )
        if dataset_id is not None:
            statement = statement.where(
                ExperimentSuite.dataset_id == self._identifier(dataset_id)
            )
        if mode is not None:
            statement = statement.where(ExperimentSuite.mode == mode)
        statement = statement.order_by(
            ExperimentSuite.registered_at, ExperimentSuite.id
        )
        with session_scope(self._session_factory) as session:
            return tuple(_suite_record(row) for row in session.scalars(statement))

    def get_suite(
        self,
        principal: AuthPrincipal,
        experiment_id: str,
    ) -> ExperimentSuiteRecord:
        normalized_id = self._identifier(experiment_id)
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        with session_scope(self._session_factory) as session:
            row = session.scalar(
                select(ExperimentSuite).where(
                    ExperimentSuite.id == normalized_id,
                    ExperimentSuite.project_id.in_(visible_project_ids),
                )
            )
            if row is None:
                raise ExperimentNotFoundError("experiment was not found")
            return _suite_record(row)

    def list_runs(
        self,
        principal: AuthPrincipal,
        *,
        project_id: str | None = None,
        experiment_id: str | None = None,
        dataset_id: str | None = None,
        model_name: str | None = None,
        cutoff_cycle: int | None = None,
        seed: int | None = None,
    ) -> tuple[ExperimentRunRecord, ...]:
        visible_project_ids = ProjectService.visible_projects_statement(
            principal
        ).with_only_columns(Project.id)
        statement = (
            select(
                ExperimentRun,
                ExperimentSuite.project_id,
                ExperimentSuite.import_id,
            )
            .join(ExperimentSuite, ExperimentSuite.id == ExperimentRun.suite_id)
            .where(ExperimentSuite.project_id.in_(visible_project_ids))
        )
        if project_id is not None:
            statement = statement.where(
                ExperimentSuite.project_id == self._identifier(project_id)
            )
        if experiment_id is not None:
            statement = statement.where(
                ExperimentRun.suite_id == self._identifier(experiment_id)
            )
        if dataset_id is not None:
            statement = statement.where(
                ExperimentRun.dataset_id == self._identifier(dataset_id)
            )
        if model_name is not None:
            statement = statement.where(
                ExperimentRun.model_name == self._identifier(model_name)
            )
        if cutoff_cycle is not None:
            if cutoff_cycle <= 0:
                raise ValueError("cutoff_cycle must be positive")
            statement = statement.where(ExperimentRun.cutoff_cycle == cutoff_cycle)
        if seed is not None:
            if seed <= 0:
                raise ValueError("seed must be positive")
            statement = statement.where(ExperimentRun.seed == seed)
        statement = statement.order_by(
            ExperimentRun.cutoff_cycle,
            ExperimentRun.model_name,
            ExperimentRun.seed,
            ExperimentRun.id,
        )
        with session_scope(self._session_factory) as session:
            return tuple(
                _run_record(row, project_id=row_project_id, import_id=import_id)
                for row, row_project_id, import_id in session.execute(statement)
            )

    def _assert_active_project(
        self,
        principal: AuthPrincipal,
        project_id: str,
    ) -> None:
        with session_scope(self._session_factory) as session:
            if session.scalar(
                self._visible_active_project_statement(principal, project_id)
            ) is None:
                raise ExperimentNotFoundError("project was not found")

    @staticmethod
    def _visible_active_project_statement(
        principal: AuthPrincipal,
        project_id: str,
    ) -> Select[tuple[Project]]:
        return ProjectService.visible_projects_statement(principal).where(
            Project.id == project_id,
            Project.status == ProjectStatus.ACTIVE.value,
        )

    def _load_source(
        self,
        import_id: str,
    ) -> tuple[ImportedA100SuiteRecord, tuple[ImportedA100TaskRecord, ...]]:
        try:
            registered = self._source.resolve(import_id)
            tasks = self._source.list_tasks(import_id)
        except (KeyError, ValueError) as exc:
            raise ExperimentSourceError(
                "verified A100 suite source is unavailable"
            ) from exc
        suite = registered.record
        keys = {
            (task.cutoff_cycle, task.model_name, task.seed) for task in tasks
        }
        run_ids = {task.run_id for task in tasks}
        if (
            len(tasks) != suite.task_count
            or len(keys) != len(tasks)
            or len(run_ids) != len(tasks)
        ):
            raise ExperimentSourceError(
                "verified A100 suite task catalog is incomplete or duplicated"
            )
        for task in tasks:
            if (
                task.dataset_id != suite.dataset_id
                or task.target != suite.target
                or task.source_commit != suite.source_commit
                or task.config_sha256 != suite.config_sha256
                or task.input_bundle_sha256 != suite.input_bundle_sha256
                or task.data_version != suite.data_version
                or task.split_version != suite.split_version
                or task.feature_version != suite.feature_version
            ):
                raise ExperimentSourceError(
                    "verified A100 task context does not match its suite"
                )
        return suite, tasks

    @staticmethod
    def _task_rows(
        suite_id: str,
        suite: ImportedA100SuiteRecord,
        tasks: Sequence[ImportedA100TaskRecord],
        *,
        registered_at: datetime,
    ) -> tuple[ExperimentRun, ...]:
        return tuple(
            ExperimentRun(
                id=str(uuid4()),
                suite_id=suite_id,
                run_id=task.run_id,
                dataset_id=task.dataset_id,
                target=task.target,
                model_name=task.model_name,
                cutoff_cycle=task.cutoff_cycle,
                seed=task.seed,
                config_sha256=task.config_sha256,
                input_bundle_sha256=task.input_bundle_sha256,
                source_commit=task.source_commit,
                data_version=task.data_version,
                split_version=task.split_version,
                feature_version=task.feature_version,
                context_sha256=task.context_sha256,
                task_relative_root=task.task_relative_root,
                evidence_uri=_task_evidence_uri(
                    suite.import_id, task.task_relative_root
                ),
                completed_at=task.completed_at,
                registered_at=registered_at,
            )
            for task in tasks
        )

    @staticmethod
    def _assert_persisted_context(
        session: Session,
        row: ExperimentSuite,
        suite: ImportedA100SuiteRecord,
        tasks: Sequence[ImportedA100TaskRecord],
    ) -> None:
        expected = {
            "import_id": suite.import_id,
            "dataset_id": suite.dataset_id,
            "target": suite.target,
            "mode": suite.mode,
            "source_commit": suite.source_commit,
            "config_sha256": suite.config_sha256,
            "input_bundle_sha256": suite.input_bundle_sha256,
            "data_version": suite.data_version,
            "split_version": suite.split_version,
            "feature_version": suite.feature_version,
            "output_sha256": suite.output_sha256,
            "transfer_sha256": suite.transfer_sha256,
            "task_count": suite.task_count,
            "file_count": suite.file_count,
            "formal_performance_claim": suite.formal_performance_claim,
            "evidence_uri": _suite_evidence_uri(suite.import_id),
        }
        if any(getattr(row, name) != value for name, value in expected.items()):
            raise ExperimentStateError(
                "persisted experiment context does not match its source"
            )
        persisted_count = session.scalar(
            select(func.count())
            .select_from(ExperimentRun)
            .where(ExperimentRun.suite_id == row.id)
        )
        persisted_tasks = tuple(
            session.scalars(
                select(ExperimentRun).where(ExperimentRun.suite_id == row.id)
            )
        )
        expected_tasks = {
            (
                task.run_id,
                task.dataset_id,
                task.target,
                task.model_name,
                task.cutoff_cycle,
                task.seed,
                task.config_sha256,
                task.input_bundle_sha256,
                task.source_commit,
                task.data_version,
                task.split_version,
                task.feature_version,
                task.context_sha256,
                task.task_relative_root,
                _task_evidence_uri(suite.import_id, task.task_relative_root),
                task.completed_at,
            )
            for task in tasks
        }
        actual_tasks = {
            (
                task.run_id,
                task.dataset_id,
                task.target,
                task.model_name,
                task.cutoff_cycle,
                task.seed,
                task.config_sha256,
                task.input_bundle_sha256,
                task.source_commit,
                task.data_version,
                task.split_version,
                task.feature_version,
                task.context_sha256,
                task.task_relative_root,
                task.evidence_uri,
                task.completed_at,
            )
            for task in persisted_tasks
        }
        if (
            persisted_count != len(tasks)
            or len(persisted_tasks) != len(tasks)
            or actual_tasks != expected_tasks
        ):
            raise ExperimentStateError("persisted experiment task matrix is incomplete")

    @staticmethod
    def _require_operator(principal: AuthPrincipal) -> None:
        if principal.role not in {UserRole.ADMIN, UserRole.MEMBER}:
            raise ExperimentAccessError(
                "role is not allowed to register experiments"
            )

    @staticmethod
    def _identifier(value: str) -> str:
        normalized = value.strip() if isinstance(value, str) else ""
        if not normalized:
            raise ExperimentNotFoundError("experiment or project was not found")
        return normalized


__all__ = [
    "A100SuiteCatalogSource",
    "ExperimentAccessError",
    "ExperimentNotFoundError",
    "ExperimentRegistryService",
    "ExperimentRunRecord",
    "ExperimentSourceError",
    "ExperimentStateError",
    "ExperimentSuiteRecord",
]
