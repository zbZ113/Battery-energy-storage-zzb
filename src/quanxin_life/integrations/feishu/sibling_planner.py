"""Plan independent Feishu analyses after one upload passes data validation."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from quanxin_life.application.engineering_recommendation_rules import (
    ReviewedEngineeringRecommendationRulesetRegistry,
)
from quanxin_life.application.ingestion import VerifiedEarlyCycleBatchStore

from .default_scenarios import ReviewedDefaultScenarioRegistry
from .jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobRecord,
    SqlAlchemyFeishuSiblingJobService,
)
from .scenario_contexts import SqlAlchemyFeishuScenarioContextStore
from .workflow import FeishuAnalysisTask


class ProactiveFeishuSiblingPlanner:
    """Create SOH and reviewed-scenario siblings without coupling their outcomes."""

    def __init__(
        self,
        *,
        batch_store: VerifiedEarlyCycleBatchStore,
        context_store: SqlAlchemyFeishuScenarioContextStore,
        sibling_jobs: SqlAlchemyFeishuSiblingJobService,
        default_scenarios: ReviewedDefaultScenarioRegistry | None,
        clock: Callable[[], datetime],
        recommendation_rulesets: (
            ReviewedEngineeringRecommendationRulesetRegistry | None
        ) = None,
    ) -> None:
        self._batch_store = batch_store
        self._context_store = context_store
        self._sibling_jobs = sibling_jobs
        self._default_scenarios = default_scenarios
        self._clock = clock
        self._recommendation_rulesets = recommendation_rulesets

    def plan_validated_siblings(self, *, job: FeishuAnalysisJobRecord) -> None:
        if (
            job.job_origin is not FeishuAnalysisJobOrigin.FEISHU
            or job.event_type != "im.message.receive_v1"
            or job.task_type is not FeishuAnalysisTask.PREDICT_CYCLE_LIFE
            or job.source_job_id is not None
            or job.record_batch_id is None
            or job.validation_result_id is None
        ):
            raise ValueError(
                "proactive siblings require a validated original Feishu file event"
            )
        batch = self._batch_store.resolve_verified_early_cycle_batch(
            job.record_batch_id
        )
        self._stage(
            source_job_id=job.job_id,
            task=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY,
        )

        profile = (
            self._default_scenarios.resolve(batch)
            if self._default_scenarios is not None
            else None
        )
        if profile is None:
            self._stage(
                source_job_id=job.job_id,
                task=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS,
            )
            self._stage_recommendation(source_job_id=job.job_id)
            return
        assert self._default_scenarios is not None
        template = self._default_scenarios.create_context_template(
            batch=batch,
            profile=profile,
            source_job_id=job.job_id,
        )
        self._context_store.create_or_resolve(
            task=template.task,
            data_batch_id=batch.record_batch_id,
            verified_context=template.verified_context,
            analysis_input=template.analysis_input,
            created_by_reference=template.created_by_reference,
            created_at=self._clock(),
        )
        self._stage(
            source_job_id=job.job_id,
            task=template.task,
            scenario_context_id=template.scenario_context_id,
            default_scenario_profile_id=template.profile_id,
            default_scenario_profile_version=template.profile_version,
            default_scenario_profile_sha256=template.profile_sha256,
        )
        self._stage_recommendation(source_job_id=job.job_id)

    def _stage_recommendation(self, *, source_job_id: str) -> None:
        registry = self._recommendation_rulesets
        if registry is None:
            return
        ruleset = registry.resolve_verified_engineering_recommendation_ruleset(
            registry.default_ruleset_id
        )
        self._sibling_jobs.stage_sibling(
            source_job_id=source_job_id,
            task=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION,
            recommendation_ruleset_id=ruleset.ruleset_id,
            recommendation_ruleset_version=ruleset.ruleset_version,
            recommendation_ruleset_sha256=ruleset.ruleset_manifest_sha256,
        )

    def _stage(
        self,
        *,
        source_job_id: str,
        task: FeishuAnalysisTask,
        scenario_context_id: str | None = None,
        default_scenario_profile_id: str | None = None,
        default_scenario_profile_version: str | None = None,
        default_scenario_profile_sha256: str | None = None,
    ) -> None:
        self._sibling_jobs.stage_sibling(
            source_job_id=source_job_id,
            task=task,
            scenario_context_id=scenario_context_id,
            default_scenario_profile_id=default_scenario_profile_id,
            default_scenario_profile_version=default_scenario_profile_version,
            default_scenario_profile_sha256=default_scenario_profile_sha256,
        )


__all__ = ["ProactiveFeishuSiblingPlanner"]
