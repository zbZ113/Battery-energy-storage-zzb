from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from uuid import uuid4

from sqlalchemy import create_engine

from quanxin_life.application.feishu_recheck_authorization import (
    ProjectBoundRecheckActionAuthorizationVerifier,
)
from quanxin_life.core import Decision, EvidenceLevel, ToolResult
from quanxin_life.integrations.feishu.cards import AuditedResultAuthorization
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import session_scope
from quanxin_life.persistence.models import FeishuBindingRow
from tests.unit.integrations.feishu.test_analysis_bitable import (
    _recommendation_result,
)


class _Jobs:
    def __init__(
        self,
        run_id: str,
        result_id: str,
        *,
        bound: bool = True,
        event_type: str = "feishu.analysis_job.derived_v1",
    ) -> None:
        self.run_id = run_id
        self.result_id = result_id
        self.bound = bound
        self.event_type = event_type

    def get(self, run_id: str) -> FeishuAnalysisJobRecord:
        assert run_id == self.run_id
        return cast(
            FeishuAnalysisJobRecord,
            SimpleNamespace(
                job_id=run_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU,
                event_type=self.event_type,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED,
                task_type=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
                source_job_id=str(uuid4()),
                analysis_result_id=self.result_id,
                chat_id="oc-reviewed",
                sender_id="ou-requester",
            ),
        )

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool:
        return self.bound and run_id == self.run_id and result_id == self.result_id


class _Resolver:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        assert result_id == self.result.result_id
        return self.result


class _Authorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        assert result.tool_name == "make_engineering_recommendation"
        return AuditedResultAuthorization(
            allowed=True,
            route_id="reviewed-release-gate",
            activation_status="REVIEWED_RULESET",
            evidence_level=EvidenceLevel.DOMAIN_KNOWLEDGE,
            supported_domain="same-root-reviewed-results",
        )


class _Contexts:
    def __init__(
        self,
        *,
        binding_id: str,
        project_id: str,
        owner_active: bool = True,
    ) -> None:
        self.binding_id = binding_id
        self.project_id = project_id
        self.owner_active = owner_active

    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> object:
        assert chat_id == "oc-reviewed"
        if sender_open_id == "ou-battery-owner" and not self.owner_active:
            raise ValueError("responsible user is no longer active")
        if sender_open_id not in {"ou-requester", "ou-battery-owner"}:
            raise ValueError("Feishu identity is not authorized")
        return SimpleNamespace(
            feishu_binding_id=self.binding_id,
            project_id=self.project_id,
        )


def _verifier(
    *,
    bound: bool = True,
    event_type: str = "feishu.analysis_job.derived_v1",
    owner_active: bool = True,
) -> tuple[
    ProjectBoundRecheckActionAuthorizationVerifier,
    str,
    str,
]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    binding_id = str(uuid4())
    project_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add(
            FeishuBindingRow(
                id=binding_id,
                project_id=project_id,
                chat_id="oc-reviewed",
                bitable_app_token=None,
                bitable_table_id=None,
                user_open_id_map_json={"owner-user-id": "ou-battery-owner"},
                binding_version="reviewed-binding-v1",
                status="ACTIVE",
            )
        )
    recommendation = _recommendation_result()
    run_id = str(uuid4())
    return (
        ProjectBoundRecheckActionAuthorizationVerifier(
            session_factory=sessions,
            job_store=_Jobs(
                run_id,
                recommendation.result_id,
                bound=bound,
                event_type=event_type,
            ),
            result_resolver=_Resolver(recommendation),
            result_authorizer=_Authorizer(),
            context_service=_Contexts(
                binding_id=binding_id,
                project_id=project_id,
                owner_active=owner_active,
            ),
            expected_permission_reference="permission-reviewed-v1",
        ),
        run_id,
        recommendation.result_id,
    )


def test_authorizes_only_recheck_result_with_live_owner_and_permission() -> None:
    verifier, run_id, result_id = _verifier()

    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-battery-owner",
        permission_reference="permission-reviewed-v1",
    ) is True

    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-unknown-owner",
        permission_reference="permission-reviewed-v1",
    ) is False
    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-battery-owner",
        permission_reference="permission-substituted-v2",
    ) is False


def test_rejects_result_without_exact_run_binding() -> None:
    verifier, run_id, result_id = _verifier(bound=False)

    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-battery-owner",
        permission_reference="permission-reviewed-v1",
    ) is False


def test_rejects_non_derived_or_inactive_responsible_user() -> None:
    verifier, run_id, result_id = _verifier(event_type="aily.analysis_task.create_v2")
    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-battery-owner",
        permission_reference="permission-reviewed-v1",
    ) is False

    verifier, run_id, result_id = _verifier(owner_active=False)
    assert verifier.verify_recheck_action(
        source_run_id=run_id,
        source_result_id=result_id,
        action_type=Decision.RECHECK,
        responsibility_reference="ou-battery-owner",
        permission_reference="permission-reviewed-v1",
    ) is False
