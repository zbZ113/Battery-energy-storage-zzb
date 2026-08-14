"""Live project authorization for Feishu engineering recheck actions."""

from __future__ import annotations

import re
from typing import Protocol

from quanxin_life.core import Decision, ToolResult
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorizer
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.recommendation_presentation import (
    engineering_recommendation_presentation,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import FeishuBindingRow

_SAFE_REFERENCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class RecheckJobStore(Protocol):
    def get(self, job_id: str) -> FeishuAnalysisJobRecord: ...

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool: ...


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class FeishuContextResolver(Protocol):
    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> object: ...


class ProjectBoundRecheckActionAuthorizationVerifier:
    """Authorize only a live RECHECK_REQUIRED recommendation and named owner."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        job_store: RecheckJobStore,
        result_resolver: RegisteredResultResolver,
        result_authorizer: AuditedResultAuthorizer,
        context_service: FeishuContextResolver,
        expected_permission_reference: str,
    ) -> None:
        self._session_factory = session_factory
        self._job_store = job_store
        self._result_resolver = result_resolver
        self._result_authorizer = result_authorizer
        self._context_service = context_service
        self._expected_permission_reference = _safe_reference(
            expected_permission_reference,
            field_name="expected_permission_reference",
        )

    def verify_recheck_action(
        self,
        *,
        source_run_id: str,
        source_result_id: str,
        action_type: Decision,
        responsibility_reference: str,
        permission_reference: str,
    ) -> bool:
        if (
            action_type is not Decision.RECHECK
            or permission_reference != self._expected_permission_reference
        ):
            return False
        try:
            job = self._job_store.get(source_run_id)
            if (
                job.job_origin is not FeishuAnalysisJobOrigin.FEISHU
                or job.event_type != "feishu.analysis_job.derived_v1"
                or job.job_status is not FeishuAnalysisJobStatus.SUCCEEDED
                or job.task_type
                is not FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION
                or job.source_job_id is None
                or job.analysis_result_id != source_result_id
                or not job.chat_id
                or not job.sender_id
                or not self._job_store.is_result_bound_to_run(
                    run_id=source_run_id,
                    result_id=source_result_id,
                )
            ):
                return False
            result = ToolResult.model_validate(
                self._result_resolver.resolve_registered_result(
                    source_result_id
                ).model_dump(mode="json")
            )
            authorization = self._result_authorizer.authorize(result)
            presentation = engineering_recommendation_presentation(result)
            if not authorization.allowed or presentation.outcome != "RECHECK_REQUIRED":
                return False
            context = self._context_service.resolve_feishu(
                chat_id=job.chat_id,
                sender_open_id=job.sender_id,
            )
            responsible_context = self._context_service.resolve_feishu(
                chat_id=job.chat_id,
                sender_open_id=_safe_reference(
                    responsibility_reference,
                    field_name="responsibility_reference",
                ),
            )
            binding_id = getattr(context, "feishu_binding_id", None)
            project_id = getattr(context, "project_id", None)
            if (
                not isinstance(binding_id, str)
                or not isinstance(project_id, str)
                or getattr(responsible_context, "feishu_binding_id", None)
                != binding_id
                or getattr(responsible_context, "project_id", None) != project_id
            ):
                return False
            with session_scope(self._session_factory) as session:
                binding = session.get(FeishuBindingRow, binding_id)
                return bool(
                    binding is not None
                    and binding.project_id == project_id
                    and binding.chat_id == job.chat_id
                    and binding.status == "ACTIVE"
                    and responsibility_reference
                    in binding.user_open_id_map_json.values()
                )
        except (AttributeError, LookupError, RuntimeError, TypeError, ValueError):
            return False


def _safe_reference(value: object, *, field_name: str) -> str:
    normalized = value.strip() if isinstance(value, str) else ""
    if _SAFE_REFERENCE.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a safe reference")
    return normalized


__all__ = ["ProjectBoundRecheckActionAuthorizationVerifier"]
