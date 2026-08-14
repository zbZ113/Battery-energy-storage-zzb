from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine

from quanxin_life.application.feishu_engineering_recommendation import (
    FeishuEngineeringRecommendationExecutor,
    RootBoundFeishuRecommendationResultResolver,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStage,
    FeishuAnalysisJobStatus,
    SqlAlchemyFeishuJobStore,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.database import session_scope
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.reporting.audited_artifacts import ReportArtifactFormat
from quanxin_life.tools import StandardToolName
from quanxin_life.tools.engineering_recommendation import (
    EngineeringRecommendationRule,
    VerifiedEngineeringRecommendationRuleset,
)

NOW = datetime(2026, 8, 14, 8, 0, tzinfo=UTC)
ROOT_JOB_ID = "3a3c972b-a23e-42c3-af76-e39038806f21"
SOH_JOB_ID = "3a3c972b-a23e-42c3-af76-e39038806f22"
SCENARIO_JOB_ID = "3a3c972b-a23e-42c3-af76-e39038806f23"
RECOMMENDATION_JOB_ID = "3a3c972b-a23e-42c3-af76-e39038806f24"


class _ResultResolver:
    def __init__(self, results: tuple[ToolResult, ...]) -> None:
        self.results = {result.result_id: result for result in results}

    def resolve_registered_result(self, result_id: str) -> ToolResult:
        return self.results[result_id]


class _RulesetResolver:
    def __init__(self, ruleset: VerifiedEngineeringRecommendationRuleset) -> None:
        self.ruleset = ruleset

    def resolve_verified_engineering_recommendation_ruleset(
        self,
        ruleset_id: str,
    ) -> VerifiedEngineeringRecommendationRuleset:
        if ruleset_id != self.ruleset.ruleset_id:
            raise ValueError("ruleset unavailable")
        return self.ruleset


class _ContextService:
    def resolve_feishu(self, *, chat_id: str, sender_open_id: str) -> object:
        assert chat_id == "oc-reviewed"
        assert sender_open_id == "ou-reviewed"
        return SimpleNamespace(project_id="project-reviewed")


class _ProjectLedger:
    def __init__(self) -> None:
        self.results: dict[str, ToolResult] = {}
        self.committed_slots: list[str] = []

    def resolve_registered_result(self, _context: object, result_id: str) -> ToolResult:
        return self.results[result_id]

    def commit_feishu_result_slot(self, **kwargs: object) -> ToolResult:
        result = kwargs["result"]
        assert isinstance(result, ToolResult)
        slot = kwargs["slot"]
        assert isinstance(slot, str)
        self.results[result.result_id] = result
        self.committed_slots.append(slot)
        return result


def _result(tool: StandardToolName, version: str) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=tool.value,
        tool_version=version,
        model_version=f"{tool.value}-model-v1",
        data_version="matr-v1",
        feature_version="reviewed-feature-v1",
        input_hash="a" * 64,
        values={"score": 1.0},
        provenance=[
            ProvenanceRecord(
                source_id="reviewed-source",
                source_kind=SourceKind.OBSERVED,
                uri="artifact://reviewed-source",
                sha256="b" * 64,
                description="Reviewed test evidence.",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def _ruleset(results: tuple[ToolResult, ...]) -> VerifiedEngineeringRecommendationRuleset:
    rules = tuple(
        EngineeringRecommendationRule(
            rule_id=f"allow-{index}",
            result_tool_name=StandardToolName(result.tool_name),
            result_tool_version=result.tool_version,
            value_path="values.score",
            comparator="GTE",
            threshold=0.0,
            recheck_reason_code=f"RECHECK_{index}",
        )
        for index, result in enumerate(results)
    )
    return VerifiedEngineeringRecommendationRuleset(
        ruleset_id="reviewed-release-gate",
        ruleset_version="reviewed-release-gate-v1",
        ruleset_manifest_sha256="e" * 64,
        unresolved_reason_code="EVIDENCE_UNRESOLVED",
        rules=rules,
        provenance=results[0].provenance,
    )


def _fixture() -> tuple[object, SqlAlchemyFeishuJobStore, tuple[ToolResult, ...]]:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = create_session_factory(engine)
    results = (
        _result(StandardToolName.VALIDATE_BATTERY_DATA, "validation-v1"),
        _result(StandardToolName.PREDICT_CYCLE_LIFE, "rul-v1"),
        _result(StandardToolName.PREDICT_SOH_TRAJECTORY, "soh-v1"),
        _result(StandardToolName.COMPARE_OPERATION_SCENARIOS, "scenario-v1"),
    )
    upstream_ids = tuple(sorted(result.result_id for result in results))
    upstream_sha256 = sha256_canonical(
        {
            "schema_version": "feishu-recommendation-upstream-results-v1",
            "result_ids": list(upstream_ids),
        }
    )
    common = {
        "status": "PROCESSED",
        "attempt_count": 1,
        "received_at": NOW,
        "processed_at": NOW,
        "job_origin": FeishuAnalysisJobOrigin.FEISHU.value,
        "chat_id": "oc-reviewed",
        "sender_id": "ou-reviewed",
        "receive_id_type": "chat_id",
        "event_time": NOW,
        "record_batch_id": "canonical-csv-" + "c" * 64,
        "cell_reference": "MATR_b3c34",
        "input_file_sha256": "d" * 64,
        "job_attempt_count": 1,
        "job_created_at": NOW,
        "job_updated_at": NOW,
    }
    with session_scope(sessions) as session:
        session.add_all(
            (
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="root-event",
                    event_type="im.message.receive_v1",
                    payload_sha256="1" * 64,
                    job_id=ROOT_JOB_ID,
                    job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                    job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                    task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                    run_id=ROOT_JOB_ID,
                    validation_result_id=results[0].result_id,
                    analysis_result_id=results[1].result_id,
                    job_completed_at=NOW,
                    **common,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="soh-event",
                    event_type="feishu.analysis_job.derived_v1",
                    payload_sha256="2" * 64,
                    job_id=SOH_JOB_ID,
                    source_job_id=ROOT_JOB_ID,
                    job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                    job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                    task_type=FeishuAnalysisTask.PREDICT_SOH_TRAJECTORY.value,
                    run_id=SOH_JOB_ID,
                    analysis_result_id=results[2].result_id,
                    job_completed_at=NOW,
                    **common,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="scenario-event",
                    event_type="feishu.analysis_job.derived_v1",
                    payload_sha256="3" * 64,
                    job_id=SCENARIO_JOB_ID,
                    source_job_id=ROOT_JOB_ID,
                    job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                    job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                    task_type=FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
                    run_id=SCENARIO_JOB_ID,
                    analysis_result_id=results[3].result_id,
                    job_completed_at=NOW,
                    **common,
                ),
                FeishuEventReceipt(
                    id=str(uuid4()),
                    event_id="recommendation-event",
                    event_type="feishu.analysis_job.derived_v1",
                    payload_sha256="4" * 64,
                    job_id=RECOMMENDATION_JOB_ID,
                    source_job_id=ROOT_JOB_ID,
                    job_status=FeishuAnalysisJobStatus.PENDING.value,
                    job_stage=FeishuAnalysisJobStage.RECEIVED.value,
                    task_type=FeishuAnalysisTask.MAKE_ENGINEERING_RECOMMENDATION.value,
                    run_id=RECOMMENDATION_JOB_ID,
                    recommendation_ruleset_id="reviewed-release-gate",
                    recommendation_ruleset_version="reviewed-release-gate-v1",
                    recommendation_ruleset_sha256="e" * 64,
                    recommendation_upstream_result_ids_json=list(upstream_ids),
                    recommendation_upstream_result_ids_sha256=upstream_sha256,
                    **common,
                ),
            )
        )
    return sessions, SqlAlchemyFeishuJobStore(sessions), results


def test_executor_uses_only_frozen_same_root_evidence_and_builds_audited_report() -> None:
    sessions, jobs, results = _fixture()
    claim = jobs.claim(job_id=RECOMMENDATION_JOB_ID, claimed_at=NOW)
    assert claim.claim_token is not None
    job = jobs.get(RECOMMENDATION_JOB_ID)
    ledger = _ProjectLedger()
    executor = FeishuEngineeringRecommendationExecutor(
        session_factory=sessions,
        context_service=_ContextService(),
        project_ledger=ledger,
        result_resolver=_ResultResolver(results),
        ruleset_resolver=_RulesetResolver(_ruleset(results)),
        clock=lambda: NOW,
    )

    analysis, report = executor.execute(job, claim_token=claim.claim_token)

    assert analysis.values["recommendation"] == "ADOPTABLE"
    assert analysis.values["authorized_upstream_result_ids"] == sorted(
        result.result_id for result in results
    )
    assert report.values["upstream_result_ids"] == [analysis.result_id]
    assert "工程综合建议审计报告" in report.values["markdown"]
    assert ledger.committed_slots == ["ANALYSIS", "REPORT"]
    artifact = AuditedReportArtifactExporter(AuditLedger((report,))).export(
        report.result_id,
        ReportArtifactFormat.MARKDOWN,
    )
    assert "工程综合建议审计报告" in artifact.payload.decode("utf-8")


def test_root_bound_resolver_rejects_a_real_result_from_another_root() -> None:
    sessions, jobs, results = _fixture()
    external = _result(StandardToolName.PREDICT_CYCLE_LIFE, "rul-v1")
    other_root_id = str(uuid4())
    with session_scope(sessions) as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="other-root-event",
                event_type="im.message.receive_v1",
                payload_sha256="9" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=other_root_id,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                job_stage=FeishuAnalysisJobStage.SUCCEEDED.value,
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=other_root_id,
                chat_id="oc-reviewed",
                sender_id="ou-reviewed",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id="canonical-csv-" + "c" * 64,
                cell_reference="MATR_b3c34",
                input_file_sha256="d" * 64,
                analysis_result_id=external.result_id,
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
    job = jobs.get(RECOMMENDATION_JOB_ID)
    assert job.recommendation_upstream_result_ids is not None
    forged = replace(
        job,
        recommendation_upstream_result_ids=(
            *job.recommendation_upstream_result_ids,
            external.result_id,
        ),
    )
    resolver = RootBoundFeishuRecommendationResultResolver(
        session_factory=sessions,
        job=forged,
        delegated_resolver=_ResultResolver((*results, external)),
    )

    with pytest.raises(ValueError, match="root analysis family"):
        resolver.resolve_registered_result(external.result_id)
