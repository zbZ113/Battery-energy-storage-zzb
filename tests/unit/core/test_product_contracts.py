from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from quanxin_life.core import (
    AgentFailurePolicy,
    AgentIntent,
    AgentPlan,
    AgentPlanStep,
    AgentRole,
    AgentRunState,
    AgentRunStatus,
    ApprovalKind,
    ApprovalRequest,
    EvidenceLevel,
    FeishuBinding,
    KnowledgeDocumentManifest,
    KnowledgeReviewStatus,
    LlmProviderConfig,
    ScenarioLifetimeRequest,
    ScenarioLifetimeResult,
    UserRole,
)

NOW = datetime(2026, 7, 15, 12, tzinfo=UTC)


def _intent() -> AgentIntent:
    return AgentIntent(
        intent_id=str(uuid4()),
        project_id="quanxin-demo",
        goal="诊断这枚电芯的早期寿命风险",
        dataset_ids=("hust-safe-v1",),
        requested_outputs=("cycle_life", "audited_report"),
        created_at=NOW,
    )


def _steps() -> tuple[AgentPlanStep, ...]:
    return (
        AgentPlanStep(
            step_id="validate",
            role=AgentRole.DATA_QUALITY,
            tool_name="validate_battery_data",
            input_references={"record_batch_id": "batch:hust-cell-001"},
            failure_policy=AgentFailurePolicy.STOP,
        ),
        AgentPlanStep(
            step_id="predict",
            role=AgentRole.LIFETIME,
            tool_name="predict_cycle_life",
            depends_on=("validate",),
            input_references={"feature_result_id": "step:validate.result_id"},
            failure_policy=AgentFailurePolicy.REPLAN,
        ),
    )


def test_llm_provider_config_excludes_secrets_and_normalizes_capability_time() -> None:
    config = LlmProviderConfig(
        config_version="llm-provider-v1",
        provider_id="primary-openai-compatible",
        base_url="https://llm.example.test/v1",
        primary_model="reasoning-model",
        economy_model="economy-model",
        embedding_model="embedding-model",
        timeout_seconds=30,
        monthly_budget_cny=100,
        supports_json_schema=True,
        supports_tool_calling=True,
        supports_streaming=True,
        supports_embeddings=False,
        capability_checked_at=datetime(2026, 7, 15, 20, tzinfo=UTC),
    )

    assert config.capability_checked_at is not None
    assert config.capability_checked_at.tzinfo is UTC
    assert "api_key" not in config.model_dump(mode="json")
    assert config.embedding_model == "embedding-model"
    with pytest.raises(ValidationError):
        LlmProviderConfig.model_validate(
            {
                **config.model_dump(mode="json"),
                "api_key": "must-never-be-persisted",
            }
        )


def test_optional_review_and_capability_times_accept_explicit_none() -> None:
    config = LlmProviderConfig(
        config_version="llm-provider-v1",
        provider_id="unprobed-provider",
        base_url="https://llm.example.test/v1",
        primary_model="unknown-capability-model",
        timeout_seconds=30,
        monthly_budget_cny=100,
        capability_checked_at=None,
    )
    manifest = KnowledgeDocumentManifest(
        document_id="pending-paper",
        title="Pending review paper",
        source_uri="https://example.test/pending.pdf",
        source_sha256="d" * 64,
        license_name="review-required",
        document_version="v1",
        review_status=KnowledgeReviewStatus.PENDING,
        reviewed_at=None,
    )

    assert config.capability_checked_at is None
    assert manifest.reviewed_at is None


def test_intent_requires_unique_requested_outputs() -> None:
    intent = _intent()

    assert intent.requested_outputs == ("cycle_life", "audited_report")
    with pytest.raises(ValidationError, match="requested_outputs must be unique"):
        AgentIntent(
            intent_id=str(uuid4()),
            project_id="quanxin-demo",
            goal="诊断",
            requested_outputs=("report", "report"),
            created_at=NOW,
        )


def test_agent_plan_builds_and_verifies_a_canonical_hash() -> None:
    intent = _intent()
    plan = AgentPlan.build(
        plan_version="agent-plan-v1",
        intent_id=intent.intent_id,
        steps=_steps(),
        created_at=NOW,
    )

    assert len(plan.plan_hash) == 64
    assert plan.steps[1].depends_on == ("validate",)
    payload = plan.model_dump(mode="json")
    payload["steps"][1]["tool_name"] = "generate_audited_report"
    with pytest.raises(ValidationError, match="plan_hash"):
        AgentPlan.model_validate(payload)


def test_agent_plan_rejects_forward_dependencies_and_more_than_twelve_steps() -> None:
    first = AgentPlanStep(
        step_id="first",
        role=AgentRole.DATA_QUALITY,
        tool_name="validate_battery_data",
        depends_on=("later",),
    )
    later = AgentPlanStep(
        step_id="later",
        role=AgentRole.LIFETIME,
        tool_name="predict_cycle_life",
    )

    with pytest.raises(ValidationError, match="previously declared"):
        AgentPlan.build(
            plan_version="agent-plan-v1",
            intent_id=str(uuid4()),
            steps=(first, later),
            created_at=NOW,
        )

    too_many = tuple(
        AgentPlanStep(
            step_id=f"step-{index}",
            role=AgentRole.DATA_QUALITY,
            tool_name="validate_battery_data",
        )
        for index in range(13)
    )
    with pytest.raises(ValidationError):
        AgentPlan.build(
            plan_version="agent-plan-v1",
            intent_id=str(uuid4()),
            steps=too_many,
            created_at=NOW,
        )


def test_agent_run_state_rejects_completed_and_pending_overlap() -> None:
    plan = AgentPlan.build(
        plan_version="agent-plan-v1",
        intent_id=str(uuid4()),
        steps=_steps(),
        created_at=NOW,
    )

    with pytest.raises(ValidationError, match="cannot also await approval"):
        AgentRunState(
            run_id=str(uuid4()),
            intent_id=plan.intent_id,
            plan_hash=plan.plan_hash,
            status=AgentRunStatus.AWAITING_APPROVAL,
            completed_step_ids=("validate",),
            pending_approval_step_ids=("validate",),
            result_ids=(str(uuid4()),),
            updated_at=NOW,
        )


def test_approval_request_requires_a_future_expiry() -> None:
    with pytest.raises(ValidationError, match="expires_at must be later"):
        ApprovalRequest(
            approval_id=str(uuid4()),
            run_id=str(uuid4()),
            approval_kind=ApprovalKind.FORMAL_DECISION,
            source_plan_hash="a" * 64,
            step_id="make-decision",
            impact_scope="签发正式批次决策",
            created_at=NOW,
            expires_at=NOW,
        )


def test_approved_knowledge_manifest_requires_reviewer_and_review_time() -> None:
    base = {
        "document_id": "paper-hust-001",
        "title": "HUST battery dataset paper",
        "source_uri": "https://example.test/hust-paper.pdf",
        "source_sha256": "b" * 64,
        "license_name": "review-required",
        "document_version": "v1",
        "review_status": KnowledgeReviewStatus.APPROVED,
    }

    with pytest.raises(ValidationError, match="approved knowledge documents"):
        KnowledgeDocumentManifest.model_validate(base)
    manifest = KnowledgeDocumentManifest.model_validate(
        {**base, "reviewer_id": "admin-001", "reviewed_at": NOW}
    )
    assert manifest.review_status is KnowledgeReviewStatus.APPROVED


def test_scenario_lifetime_contract_preserves_source_and_limitations() -> None:
    source_result_id = str(uuid4())
    request = ScenarioLifetimeRequest(
        lifetime_result_id=source_result_id,
        operation_policy_version="storage-policy-v1",
        equivalent_cycles_per_day=0.8,
    )
    result = ScenarioLifetimeResult(
        source_lifetime_result_id=source_result_id,
        operation_policy_version=request.operation_policy_version,
        equivalent_cycles_per_day=request.equivalent_cycles_per_day,
        days_per_year=request.days_per_year,
        scenario_years=8.5,
        limitations=("情景换算, 不代表长期实测验证",),
    )

    assert result.source_lifetime_result_id == request.lifetime_result_id
    assert result.limitations
    with pytest.raises(ValidationError, match="MODEL_INFERENCE"):
        ScenarioLifetimeResult(
            **{
                **result.model_dump(mode="json"),
                "evidence_level": EvidenceLevel.DATA_DIRECT,
            }
        )


def test_feishu_binding_requires_a_complete_target_and_nonblank_mapping() -> None:
    binding = FeishuBinding(
        binding_id=str(uuid4()),
        project_id="quanxin-demo",
        chat_id="oc_test_group",
        user_open_id_map={"admin-001": "ou_test_user"},
        binding_version="feishu-binding-v1",
    )

    assert binding.chat_id == "oc_test_group"
    with pytest.raises(ValidationError, match="chat or a complete Bitable target"):
        FeishuBinding(
            binding_id=str(uuid4()),
            project_id="quanxin-demo",
            binding_version="feishu-binding-v1",
        )


def test_public_roles_are_strict_enums() -> None:
    assert UserRole.JUDGE.value == "JUDGE"
    assert AgentRole.SUPERVISOR.value == "supervisor"
    with pytest.raises(ValueError):
        UserRole("PUBLIC")


def test_all_product_timestamps_reject_naive_values() -> None:
    with pytest.raises(ValidationError):
        ApprovalRequest(
            approval_id=str(uuid4()),
            run_id=str(uuid4()),
            approval_kind=ApprovalKind.EXTERNAL_WRITE,
            source_plan_hash="c" * 64,
            step_id="write-feishu",
            impact_scope="写入飞书多维表格",
            created_at=datetime(2026, 7, 15, 12),
            expires_at=datetime(2026, 7, 15, 12) + timedelta(hours=1),
        )
