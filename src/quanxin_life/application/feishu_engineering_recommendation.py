"""Same-root audited engineering recommendation execution for Feishu jobs."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import uuid4

from sqlalchemy import or_, select

from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.audit import ProjectResultLedger
from quanxin_life.core import ToolResult, sha256_canonical
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobRecord,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence.database import SessionFactory, session_scope
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.reporting.contracts import (
    AUDITED_REPORT_TOOL_VERSION,
    RECOMMENDATION_REPORT_RENDERER_VERSION,
)
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.engineering_recommendation import (
    ENGINEERING_RECOMMENDATION_TOOL_VERSION,
    EngineeringRecommendationToolInput,
    VerifiedEngineeringRecommendationRuleset,
    VerifiedEngineeringRecommendationRulesetResolver,
    execute_engineering_recommendation_tool,
)

Clock = Callable[[], datetime]
_TERMINAL_STATUSES = frozenset(
    {
        FeishuAnalysisJobStatus.SUCCEEDED.value,
        FeishuAnalysisJobStatus.REJECTED.value,
        FeishuAnalysisJobStatus.FAILED.value,
    }
)
_RECOMMENDATION_LABELS = {
    "ADOPTABLE": "建议采用",
    "RECHECK_REQUIRED": "建议复检",
    "UNRESOLVED": "暂无法判断",
}


class RegisteredResultResolver(Protocol):
    def resolve_registered_result(self, result_id: str) -> ToolResult: ...


class FeishuProjectContextResolver(Protocol):
    def resolve_feishu(
        self,
        *,
        chat_id: str,
        sender_open_id: str,
    ) -> VerifiedProjectInvocationContext: ...


class FeishuResultSlotCommitter(Protocol):
    def commit_feishu_result_slot(
        self,
        *,
        context: VerifiedProjectInvocationContext,
        job_id: str,
        claim_token: str,
        slot: str,
        expected_task: str,
        run_id: str,
        chat_id: str,
        sender_id: str,
        result: ToolResult,
    ) -> ToolResult: ...


class RootBoundFeishuRecommendationResultResolver:
    """Resolve only frozen results owned by one root or its direct children."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        job: FeishuAnalysisJobRecord,
        delegated_resolver: RegisteredResultResolver,
    ) -> None:
        if job.task_type is not FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION:
            raise ValueError("root-bound recommendation resolver requires a recommendation job")
        if job.source_job_id is None or not job.recommendation_upstream_result_ids:
            raise ValueError("recommendation job has no frozen root evidence")
        self._session_factory = session_factory
        self._job = job
        self._delegated_resolver = delegated_resolver
        self._allowed_ids = frozenset(job.recommendation_upstream_result_ids)

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        if result_id not in self._allowed_ids:
            raise ValueError("ToolResult is not frozen for this recommendation job")
        with session_scope(self._session_factory) as session:
            bindings = tuple(
                session.scalars(
                    select(FeishuEventReceipt).where(
                        or_(
                            FeishuEventReceipt.validation_result_id == result_id,
                            FeishuEventReceipt.analysis_result_id == result_id,
                        )
                    )
                ).all()
            )
        if len(bindings) != 1:
            raise ValueError("ToolResult is not uniquely bound to the root analysis family")
        member = bindings[0]
        expected_tool_name = self._expected_tool_name(member, result_id=result_id)
        identity = (
            self._job.record_batch_id,
            self._job.cell_reference,
            self._job.input_file_sha256,
        )
        if (
            member.record_batch_id,
            member.cell_reference,
            member.input_file_sha256,
        ) != identity:
            raise ValueError("ToolResult data identity does not match the root analysis family")
        result = ToolResult.model_validate(
            self._delegated_resolver.resolve_registered_result(result_id).model_dump(
                mode="json"
            )
        )
        if result.result_id != result_id or result.tool_name != expected_tool_name:
            raise ValueError("ToolResult contract does not match its root analysis task")
        return result

    def _expected_tool_name(
        self,
        member: FeishuEventReceipt,
        *,
        result_id: str,
    ) -> str:
        root_job_id = self._job.source_job_id
        if member.job_status not in _TERMINAL_STATUSES:
            raise ValueError("ToolResult family member is not terminal")
        if member.job_id == root_job_id and member.source_job_id is None:
            if member.validation_result_id == result_id:
                return StandardToolName.VALIDATE_BATTERY_DATA.value
            if (
                member.job_status == FeishuAnalysisJobStatus.SUCCEEDED.value
                and member.analysis_result_id == result_id
            ):
                return StandardToolName.PREDICT_CYCLE_LIFE.value
        elif (
            member.source_job_id == root_job_id
            and member.job_status == FeishuAnalysisJobStatus.SUCCEEDED.value
            and member.analysis_result_id == result_id
            and member.task_type
            in {
                FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY.value,
                FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
                FeishuAnalysisTask.PROJECT_STORAGE_LIFETIME.value,
            }
        ):
            assert member.task_type is not None
            return member.task_type
        raise ValueError("ToolResult does not belong to the root analysis family")


class FeishuEngineeringRecommendationExecutor:
    """Create and atomically checkpoint recommendation and report ToolResults."""

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        context_service: FeishuProjectContextResolver,
        project_ledger: ProjectResultLedger,
        result_resolver: RegisteredResultResolver,
        ruleset_resolver: VerifiedEngineeringRecommendationRulesetResolver,
        clock: Clock,
    ) -> None:
        if not callable(getattr(project_ledger, "commit_feishu_result_slot", None)):
            raise TypeError("recommendation execution requires atomic project result slots")
        self._session_factory = session_factory
        self._context_service = context_service
        self._project_ledger = project_ledger
        self._result_resolver = result_resolver
        self._ruleset_resolver = ruleset_resolver
        self._clock = clock

    def execute(
        self,
        job: FeishuAnalysisJobRecord,
        *,
        claim_token: str,
    ) -> tuple[ToolResult, ToolResult]:
        ruleset, input_value = self._validated_inputs(job)
        if not job.chat_id or not job.sender_id:
            raise ValueError("recommendation job has no Feishu project identity")
        context = self._context_service.resolve_feishu(
            chat_id=job.chat_id,
            sender_open_id=job.sender_id,
        )
        committer = cast(FeishuResultSlotCommitter, self._project_ledger)
        if job.analysis_result_id is None:
            analysis = execute_engineering_recommendation_tool(
                input_value,
                result_resolver=RootBoundFeishuRecommendationResultResolver(
                    session_factory=self._session_factory,
                    job=job,
                    delegated_resolver=self._result_resolver,
                ),
                ruleset_resolver=_FrozenRulesetResolver(ruleset),
                clock=self._clock,
            )
            analysis = committer.commit_feishu_result_slot(
                context=context,
                job_id=job.job_id,
                claim_token=claim_token,
                slot="ANALYSIS",
                expected_task=job.task_type.value,
                run_id=job.run_id,
                chat_id=job.chat_id,
                sender_id=job.sender_id,
                result=analysis,
            )
        else:
            analysis = self._project_ledger.resolve_registered_result(
                context,
                job.analysis_result_id,
            )
        _validate_analysis_result(analysis, input_value=input_value, ruleset=ruleset)

        if job.report_result_id is None:
            report = build_engineering_recommendation_report_result(
                analysis,
                clock=self._clock,
            )
            report = committer.commit_feishu_result_slot(
                context=context,
                job_id=job.job_id,
                claim_token=claim_token,
                slot="REPORT",
                expected_task=job.task_type.value,
                run_id=job.run_id,
                chat_id=job.chat_id,
                sender_id=job.sender_id,
                result=report,
            )
        else:
            report = self._project_ledger.resolve_registered_result(
                context,
                job.report_result_id,
            )
        _validate_report_result(report, analysis_result_id=analysis.result_id)
        return analysis, report

    def _validated_inputs(
        self,
        job: FeishuAnalysisJobRecord,
    ) -> tuple[
        VerifiedEngineeringRecommendationRuleset,
        EngineeringRecommendationToolInput,
    ]:
        if (
            job.task_type is not FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION
            or job.source_job_id is None
            or job.recommendation_ruleset_id is None
            or job.recommendation_ruleset_version is None
            or job.recommendation_ruleset_sha256 is None
            or not job.recommendation_upstream_result_ids
        ):
            raise ValueError("recommendation job input checkpoint is incomplete")
        ruleset = VerifiedEngineeringRecommendationRuleset.model_validate(
            self._ruleset_resolver.resolve_verified_engineering_recommendation_ruleset(
                job.recommendation_ruleset_id
            ).model_dump(mode="json")
        )
        if (
            ruleset.ruleset_id != job.recommendation_ruleset_id
            or ruleset.ruleset_version != job.recommendation_ruleset_version
            or ruleset.ruleset_manifest_sha256 != job.recommendation_ruleset_sha256
        ):
            raise ValueError("reviewed recommendation ruleset changed after staging")
        return ruleset, EngineeringRecommendationToolInput(
            result_ids=job.recommendation_upstream_result_ids,
            ruleset_id=job.recommendation_ruleset_id,
        )


class _FrozenRulesetResolver:
    def __init__(self, ruleset: VerifiedEngineeringRecommendationRuleset) -> None:
        self._ruleset = ruleset

    def resolve_verified_engineering_recommendation_ruleset(
        self,
        ruleset_id: str,
    ) -> VerifiedEngineeringRecommendationRuleset:
        if ruleset_id != self._ruleset.ruleset_id:
            raise ValueError("frozen recommendation ruleset identity does not match")
        return VerifiedEngineeringRecommendationRuleset.model_validate(
            self._ruleset.model_dump(mode="json")
        )


def build_engineering_recommendation_report_result(
    analysis: ToolResult,
    *,
    clock: Clock,
) -> ToolResult:
    checked = ToolResult.model_validate(analysis.model_dump(mode="json"))
    if (
        checked.tool_name != StandardToolName.MAKE_ENGINEERING_RECOMMENDATION.value
        or checked.tool_version != ENGINEERING_RECOMMENDATION_TOOL_VERSION
    ):
        raise ValueError("recommendation report requires an audited recommendation result")
    created_at = _utc_timestamp(clock())
    result_id = str(uuid4())
    return ToolResult(
        result_id=result_id,
        tool_name=StandardToolName.GENERATE_AUDITED_REPORT.value,
        tool_version=AUDITED_REPORT_TOOL_VERSION,
        model_version=RECOMMENDATION_REPORT_RENDERER_VERSION,
        data_version=checked.data_version,
        feature_version=checked.feature_version,
        input_hash=sha256_canonical(
            {
                "schema_version": "engineering-recommendation-report-input-v1",
                "analysis_result_id": checked.result_id,
                "analysis_input_hash": checked.input_hash,
            }
        ),
        values={
            "report_id": result_id,
            "rendering_version": RECOMMENDATION_REPORT_RENDERER_VERSION,
            "markdown": _render_recommendation_markdown(checked),
            "upstream_result_ids": [checked.result_id],
        },
        warnings=list(checked.warnings),
        provenance=list(checked.provenance),
        created_at=created_at,
    )


def _validate_analysis_result(
    result: ToolResult,
    *,
    input_value: EngineeringRecommendationToolInput,
    ruleset: VerifiedEngineeringRecommendationRuleset,
) -> None:
    checked = ToolResult.model_validate(result.model_dump(mode="json"))
    if (
        checked.tool_name != StandardToolName.MAKE_ENGINEERING_RECOMMENDATION.value
        or checked.tool_version != ENGINEERING_RECOMMENDATION_TOOL_VERSION
        or checked.input_hash != sha256_canonical(input_value.model_dump(mode="json"))
        or checked.values.get("ruleset_id") != ruleset.ruleset_id
        or checked.values.get("ruleset_version") != ruleset.ruleset_version
        or checked.values.get("ruleset_manifest_sha256")
        != ruleset.ruleset_manifest_sha256
        or checked.values.get("authorized_upstream_result_ids")
        != list(input_value.result_ids)
    ):
        raise ValueError("persisted recommendation ToolResult contract is invalid")


def _validate_report_result(result: ToolResult, *, analysis_result_id: str) -> None:
    checked = ToolResult.model_validate(result.model_dump(mode="json"))
    if (
        checked.tool_name != StandardToolName.GENERATE_AUDITED_REPORT.value
        or checked.tool_version != AUDITED_REPORT_TOOL_VERSION
        or checked.values.get("upstream_result_ids") != [analysis_result_id]
    ):
        raise ValueError("persisted recommendation report contract is invalid")


def _render_recommendation_markdown(result: ToolResult) -> str:
    recommendation = result.values.get("recommendation")
    if not isinstance(recommendation, str):
        raise ValueError("recommendation result has no controlled outcome")
    label = _RECOMMENDATION_LABELS.get(recommendation)
    if label is None:
        raise ValueError("recommendation result has an invalid controlled outcome")
    ruleset_id = _markdown_text(result.values.get("ruleset_id"))
    ruleset_version = _markdown_text(result.values.get("ruleset_version"))
    reason_codes = _text_list(result.values.get("reason_codes"), "reason_codes")
    resolution_issues = _text_list(
        result.values.get("resolution_issues"),
        "resolution_issues",
    )
    threshold_evidence = result.values.get("threshold_evidence")
    if not isinstance(threshold_evidence, list):
        raise ValueError("recommendation threshold evidence is invalid")
    lines = [
        "# 工程综合建议审计报告",
        "",
        f"- 综合建议: {label}",
        f"- 规则集: {ruleset_id}",
        f"- 规则集版本: {ruleset_version}",
        f"- 推荐结果 ID: {result.result_id}",
        "",
        "## 原因与缺口",
        "",
    ]
    messages = [*reason_codes, *resolution_issues]
    lines.extend(f"- {_markdown_text(item)}" for item in messages)
    if not messages:
        lines.append("- 已登记证据满足全部受审规则。")
    lines.extend(("", "## 受审规则证据", ""))
    if not threshold_evidence:
        lines.append("当前没有可用于阈值核验的完整数值证据。")
        return "\n".join(lines) + "\n"
    lines.extend(
        (
            "| 规则 | 来源结果 | JSON 路径 | 比较符 | 审核阈值 | 实际值 | 通过 |",
            "| --- | --- | --- | --- | ---: | ---: | --- |",
        )
    )
    for item in threshold_evidence:
        if not isinstance(item, Mapping):
            raise ValueError("recommendation threshold evidence entry is invalid")
        threshold = _finite_number(item.get("threshold"), field_name="threshold")
        actual = _finite_number(item.get("actual_value"), field_name="actual_value")
        passed = item.get("comparison_passed")
        if not isinstance(passed, bool):
            raise ValueError("recommendation comparison outcome is invalid")
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_text(item.get("rule_id")),
                    _markdown_text(item.get("result_id")),
                    _markdown_text(item.get("value_path")),
                    _markdown_text(item.get("comparator")),
                    json.dumps(threshold, ensure_ascii=True, allow_nan=False),
                    json.dumps(actual, ensure_ascii=True, allow_nan=False),
                    "是" if passed else "否",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _markdown_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("recommendation report text evidence is invalid")
    normalized = value.strip().replace("|", "\\|").replace("\r", " ").replace("\n", " ")
    if not normalized or any(ord(character) < 32 for character in normalized):
        raise ValueError("recommendation report text evidence is unsafe")
    return normalized


def _text_list(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"recommendation {field_name} is invalid")
    return tuple(_markdown_text(item) for item in value)


def _finite_number(value: object, *, field_name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"recommendation {field_name} is not numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"recommendation {field_name} is not finite")
    return value


def _utc_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recommendation execution clock must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "RECOMMENDATION_REPORT_RENDERER_VERSION",
    "FeishuEngineeringRecommendationExecutor",
    "RootBoundFeishuRecommendationResultResolver",
    "build_engineering_recommendation_report_result",
]
