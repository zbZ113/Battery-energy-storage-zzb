"""Durable, integrity-checked scenario inputs referenced by Feishu and Aily."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypeAlias
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from quanxin_life.core import sha256_canonical
from quanxin_life.persistence.database import SessionFactory
from quanxin_life.persistence.models import FeishuScenarioContextRow
from quanxin_life.scenarios import VerifiedScenarioContext
from quanxin_life.tools.blast_scenarios import (
    CompareOperationScenariosToolInput,
    ProjectStorageLifetimeToolInput,
)

from .workflow import FeishuAnalysisTask

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_SCHEMA_VERSION = "feishu-scenario-context-v1"
ScenarioAnalysisInput: TypeAlias = (
    CompareOperationScenariosToolInput | ProjectStorageLifetimeToolInput
)
_SCENARIO_TASKS = frozenset(
    {
        FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
        FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME,
    }
)


@dataclass(frozen=True, slots=True)
class FeishuScenarioContextRecord:
    scenario_context_id: str
    task: FeishuAnalysisTask
    data_batch_id: str
    route_id: str
    input_sha256: str
    created_by_reference: str
    created_at: datetime


class SqlAlchemyFeishuScenarioContextStore:
    """Persist typed scenario requests without storing any simulated output."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._session_factory = session_factory

    def create(
        self,
        *,
        task: FeishuAnalysisTask,
        data_batch_id: str,
        verified_context: VerifiedScenarioContext,
        analysis_input: ScenarioAnalysisInput,
        created_by_reference: str,
        created_at: datetime,
    ) -> FeishuScenarioContextRecord:
        checked_task = _scenario_task(task)
        checked_batch_id = _reference(data_batch_id, field_name="data_batch_id")
        checked_creator = _reference(
            created_by_reference,
            field_name="created_by_reference",
        )
        checked_created_at = _utc(created_at)
        try:
            context, typed_input = _validated_payload(
                task=checked_task,
                verified_context=verified_context,
                analysis_input=analysis_input,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("scenario context payload is invalid") from exc
        context_id = _canonical_uuid(context.scenario_context_id, field_name="scenario_context_id")
        payload = _integrity_payload(
            task=checked_task,
            data_batch_id=checked_batch_id,
            route_id=typed_input.route_id,
            verified_context=context,
            analysis_input=typed_input,
            created_by_reference=checked_creator,
            created_at=checked_created_at,
        )
        input_sha256 = sha256_canonical(payload)
        session = self._session_factory()
        try:
            existing = session.scalar(
                select(FeishuScenarioContextRow).where(
                    FeishuScenarioContextRow.id == context_id
                )
            )
            if existing is not None:
                raise ValueError("scenario context already exists")
            session.add(
                FeishuScenarioContextRow(
                    id=context_id,
                    schema_version=_SCHEMA_VERSION,
                    task_type=checked_task.value,
                    data_batch_id=checked_batch_id,
                    route_id=typed_input.route_id,
                    verified_context_json=context.model_dump(mode="json"),
                    analysis_input_json=typed_input.model_dump(mode="json"),
                    input_sha256=input_sha256,
                    created_by_reference=checked_creator,
                    created_at=checked_created_at,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
        return FeishuScenarioContextRecord(
            scenario_context_id=context_id,
            task=checked_task,
            data_batch_id=checked_batch_id,
            route_id=typed_input.route_id,
            input_sha256=input_sha256,
            created_by_reference=checked_creator,
            created_at=checked_created_at,
        )

    def create_or_resolve(
        self,
        *,
        task: FeishuAnalysisTask,
        data_batch_id: str,
        verified_context: VerifiedScenarioContext,
        analysis_input: ScenarioAnalysisInput,
        created_by_reference: str,
        created_at: datetime,
    ) -> FeishuScenarioContextRecord:
        concurrent_conflict: IntegrityError | None = None
        try:
            return self.create(
                task=task,
                data_batch_id=data_batch_id,
                verified_context=verified_context,
                analysis_input=analysis_input,
                created_by_reference=created_by_reference,
                created_at=created_at,
            )
        except ValueError as exc:
            if str(exc) != "scenario context already exists":
                raise
        except IntegrityError as exc:
            concurrent_conflict = exc
        try:
            record = self.get(verified_context.scenario_context_id)
        except (LookupError, RuntimeError, TypeError, ValueError) as exc:
            if concurrent_conflict is not None:
                raise concurrent_conflict from exc
            raise
        checked_task = _scenario_task(task)
        checked_batch_id = _reference(data_batch_id, field_name="data_batch_id")
        checked_creator = _reference(
            created_by_reference,
            field_name="created_by_reference",
        )
        context, typed_input = _validated_payload(
            task=checked_task,
            verified_context=verified_context,
            analysis_input=analysis_input,
        )
        if (
            record.task is not checked_task
            or record.data_batch_id != checked_batch_id
            or record.route_id != typed_input.route_id
            or record.created_by_reference != checked_creator
            or self.resolve_scenario_context(record.scenario_context_id) != context
            or self.resolve_analysis_input(
                scenario_context_id=record.scenario_context_id,
                task=checked_task,
                run_id=record.scenario_context_id,
            )
            != typed_input.model_dump(mode="json")
        ):
            raise ValueError("scenario context already exists with conflicting content")
        return record


    def get(self, scenario_context_id: str) -> FeishuScenarioContextRecord:
        row, _context, _analysis_input = self._resolve_row(scenario_context_id)
        return FeishuScenarioContextRecord(
            scenario_context_id=row.id,
            task=FeishuAnalysisTask(row.task_type),
            data_batch_id=row.data_batch_id,
            route_id=row.route_id,
            input_sha256=row.input_sha256,
            created_by_reference=row.created_by_reference,
            created_at=_database_utc(row.created_at),
        )

    def resolve_scenario_context(
        self,
        scenario_context_id: str,
    ) -> VerifiedScenarioContext:
        _row, context, _analysis_input = self._resolve_row(scenario_context_id)
        return VerifiedScenarioContext.model_validate(context.model_dump(mode="json"))

    def resolve_analysis_input(
        self,
        *,
        scenario_context_id: str,
        task: FeishuAnalysisTask,
        run_id: str,
    ) -> dict[str, object]:
        row, _context, analysis_input = self._resolve_row(scenario_context_id)
        checked_task = _scenario_task(task)
        if row.task_type != checked_task.value:
            raise ValueError("scenario context task does not match the requested analysis")
        checked_run_id = _canonical_uuid(run_id, field_name="run_id")
        rebound = analysis_input.model_copy(update={"run_id": checked_run_id})
        typed = _analysis_input_model(checked_task).model_validate(
            rebound.model_dump(mode="json")
        )
        return typed.model_dump(mode="json")

    def resolve_data_batch_id(self, scenario_context_id: str) -> str:
        row, _context, _analysis_input = self._resolve_row(scenario_context_id)
        return row.data_batch_id

    def _resolve_row(
        self,
        scenario_context_id: str,
    ) -> tuple[
        FeishuScenarioContextRow,
        VerifiedScenarioContext,
        ScenarioAnalysisInput,
    ]:
        checked_id = _canonical_uuid(
            scenario_context_id,
            field_name="scenario_context_id",
        )
        session = self._session_factory()
        try:
            row = session.scalar(
                select(FeishuScenarioContextRow).where(
                    FeishuScenarioContextRow.id == checked_id
                )
            )
            if row is None:
                raise ValueError("scenario context was not found")
            detached = FeishuScenarioContextRow(
                id=row.id,
                schema_version=row.schema_version,
                task_type=row.task_type,
                data_batch_id=row.data_batch_id,
                route_id=row.route_id,
                verified_context_json=dict(row.verified_context_json),
                analysis_input_json=dict(row.analysis_input_json),
                input_sha256=row.input_sha256,
                created_by_reference=row.created_by_reference,
                created_at=_database_utc(row.created_at),
            )
        finally:
            session.close()
        try:
            if detached.schema_version != _SCHEMA_VERSION:
                raise ValueError("unsupported scenario context schema")
            task = _scenario_task(FeishuAnalysisTask(detached.task_type))
            context = VerifiedScenarioContext.model_validate(
                detached.verified_context_json
            )
            input_model = _analysis_input_model(task)
            analysis_input = input_model.model_validate(detached.analysis_input_json)
            context, analysis_input = _validated_payload(
                task=task,
                verified_context=context,
                analysis_input=analysis_input,
            )
            payload = _integrity_payload(
                task=task,
                data_batch_id=detached.data_batch_id,
                route_id=detached.route_id,
                verified_context=context,
                analysis_input=analysis_input,
                created_by_reference=detached.created_by_reference,
                created_at=detached.created_at,
            )
            if sha256_canonical(payload) != detached.input_sha256:
                raise ValueError("scenario context integrity hash does not match")
        except (TypeError, ValueError) as exc:
            raise ValueError("scenario context integrity check failed") from exc
        return detached, context, analysis_input


def _validated_payload(
    *,
    task: FeishuAnalysisTask,
    verified_context: VerifiedScenarioContext,
    analysis_input: ScenarioAnalysisInput,
) -> tuple[VerifiedScenarioContext, ScenarioAnalysisInput]:
    context = VerifiedScenarioContext.model_validate(
        verified_context.model_dump(mode="json")
    )
    input_model = _analysis_input_model(task)
    typed_input = input_model.model_validate(analysis_input.model_dump(mode="json"))
    if context.scenario_context_id != typed_input.scenario_context_id:
        raise ValueError("scenario context identity does not match its analysis input")
    if typed_input.run_id != typed_input.scenario_context_id:
        raise ValueError("scenario context template run_id must equal its context identity")
    if context.cell != typed_input.cell:
        raise ValueError("scenario context cell does not match its analysis input")
    return context, typed_input


def _analysis_input_model(
    task: FeishuAnalysisTask,
) -> type[CompareOperationScenariosToolInput] | type[ProjectStorageLifetimeToolInput]:
    if task is FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS:
        return CompareOperationScenariosToolInput
    if task is FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME:
        return ProjectStorageLifetimeToolInput
    raise ValueError("scenario context task is not supported")


def _integrity_payload(
    *,
    task: FeishuAnalysisTask,
    data_batch_id: str,
    route_id: str,
    verified_context: VerifiedScenarioContext,
    analysis_input: ScenarioAnalysisInput,
    created_by_reference: str,
    created_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "task_type": task.value,
        "data_batch_id": data_batch_id,
        "route_id": route_id,
        "verified_context": verified_context.model_dump(mode="json"),
        "analysis_input": analysis_input.model_dump(mode="json"),
        "created_by_reference": created_by_reference,
        "created_at": created_at.isoformat(),
    }


def _scenario_task(task: FeishuAnalysisTask) -> FeishuAnalysisTask:
    if task not in _SCENARIO_TASKS:
        raise ValueError("scenario context task is not supported")
    return task


def _canonical_uuid(value: str, *, field_name: str) -> str:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field_name} must be canonical")
    return value


def _reference(value: str, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe machine reference")
    return normalized


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("scenario context timestamps must include a timezone")
    return value.astimezone(UTC)


def _database_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "FeishuScenarioContextRecord",
    "ScenarioAnalysisInput",
    "SqlAlchemyFeishuScenarioContextStore",
]
