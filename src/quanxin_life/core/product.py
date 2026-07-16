"""Service-independent contracts for the complete multi-agent product.

These models contain configuration, identifiers and workflow state only.  They
do not execute models, calculate battery values, hold secrets, or grant tool
permissions.  Numeric engineering outputs remain inside audited ``ToolResult``
artifacts produced by registered domain tools.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Self

from pydantic import Field, HttpUrl, field_validator, model_validator

from quanxin_life.core.enums import (
    AgentFailurePolicy,
    AgentPlanningMode,
    AgentRole,
    AgentRunStatus,
    ApprovalKind,
    ApprovalStatus,
    EvidenceLevel,
    KnowledgeReviewStatus,
)
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import (
    ContractModel,
    JsonMapping,
    Sha256,
    _json_mapping,
    _utc_datetime,
    _uuid_string,
)


def _nonblank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be blank")
    return normalized


def _optional_utc_datetime(value: datetime | None) -> datetime | None:
    return _utc_datetime(value) if value is not None else None


def _unique_nonblank(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(_nonblank(value) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must be unique")
    return normalized


class LlmProviderConfig(ContractModel):
    """Persistable, secret-free configuration for an OpenAI-compatible provider."""

    config_version: str = Field(min_length=1)
    provider_id: str = Field(min_length=1)
    base_url: HttpUrl
    primary_model: str = Field(min_length=1)
    economy_model: str | None = Field(default=None, min_length=1)
    embedding_model: str | None = Field(default=None, min_length=1)
    timeout_seconds: float = Field(gt=0, le=120, allow_inf_nan=False)
    monthly_budget_cny: float = Field(gt=0, le=100, allow_inf_nan=False)
    supports_json_schema: bool | None = None
    supports_tool_calling: bool | None = None
    supports_streaming: bool | None = None
    supports_embeddings: bool | None = None
    capability_checked_at: datetime | None = None
    capability_warnings: tuple[str, ...] = ()

    _capability_checked_at_utc = field_validator("capability_checked_at")(
        _optional_utc_datetime
    )

    @field_validator(
        "config_version", "provider_id", "primary_model", "economy_model", "embedding_model"
    )
    @classmethod
    def text_fields_are_not_blank(cls, value: str | None) -> str | None:
        return _nonblank(value) if value is not None else None

    @field_validator("capability_warnings")
    @classmethod
    def warnings_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field_name="capability_warnings")


class AgentIntent(ContractModel):
    """A structured user objective produced without executable tool inputs."""

    intent_id: str
    project_id: str = Field(min_length=1)
    goal: str = Field(min_length=2, max_length=4_000)
    dataset_ids: tuple[str, ...] = ()
    requested_outputs: tuple[str, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _intent_id_uuid = field_validator("intent_id")(_uuid_string)
    _created_at_utc = field_validator("created_at")(_utc_datetime)

    @field_validator("project_id", "goal")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("dataset_ids", "requested_outputs")
    @classmethod
    def list_fields_are_unique(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "values")
        return _unique_nonblank(value, field_name=field_name)


class AgentPlanStep(ContractModel):
    """One planner-produced request that still requires deterministic validation."""

    step_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z][A-Za-z0-9_-]*$",
    )
    role: AgentRole
    tool_name: str = Field(min_length=1)
    depends_on: tuple[str, ...] = ()
    input_references: dict[str, str] = Field(default_factory=dict)
    requires_approval: bool = False
    failure_policy: AgentFailurePolicy = AgentFailurePolicy.STOP

    @field_validator("step_id", "tool_name")
    @classmethod
    def identifiers_are_not_blank(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("depends_on")
    @classmethod
    def dependencies_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field_name="depends_on")

    @field_validator("input_references")
    @classmethod
    def input_references_are_json_safe(cls, value: dict[str, str]) -> dict[str, str]:
        _json_mapping(value)
        return {_nonblank(key): _nonblank(item) for key, item in value.items()}

    @model_validator(mode="after")
    def step_cannot_depend_on_itself(self) -> Self:
        if self.step_id in self.depends_on:
            raise ValueError("an Agent plan step cannot depend on itself")
        return self


class AgentPlan(ContractModel):
    """A bounded, dependency-ordered plan whose hash is verified on every load."""

    plan_version: str = Field(min_length=1)
    intent_id: str
    steps: tuple[AgentPlanStep, ...] = Field(min_length=1, max_length=12)
    planning_mode: AgentPlanningMode = AgentPlanningMode.LLM
    plan_hash: Sha256
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _intent_id_uuid = field_validator("intent_id")(_uuid_string)
    _created_at_utc = field_validator("created_at")(_utc_datetime)

    @field_validator("plan_version")
    @classmethod
    def version_is_not_blank(cls, value: str) -> str:
        return _nonblank(value)

    @staticmethod
    def calculate_hash(
        *,
        plan_version: str,
        intent_id: str,
        steps: tuple[AgentPlanStep, ...],
        planning_mode: AgentPlanningMode,
    ) -> str:
        return sha256_canonical(
            {
                "plan_version": plan_version,
                "intent_id": intent_id,
                "planning_mode": planning_mode.value,
                "steps": [step.model_dump(mode="json") for step in steps],
            }
        )

    @classmethod
    def build(
        cls,
        *,
        plan_version: str,
        intent_id: str,
        steps: tuple[AgentPlanStep, ...],
        planning_mode: AgentPlanningMode = AgentPlanningMode.LLM,
        created_at: datetime | None = None,
    ) -> Self:
        normalized_version = _nonblank(plan_version)
        plan_hash = cls.calculate_hash(
            plan_version=normalized_version,
            intent_id=intent_id,
            steps=steps,
            planning_mode=planning_mode,
        )
        return cls(
            plan_version=normalized_version,
            intent_id=intent_id,
            steps=steps,
            planning_mode=planning_mode,
            plan_hash=plan_hash,
            created_at=created_at or datetime.now(UTC),
        )

    @model_validator(mode="after")
    def dependencies_and_hash_are_valid(self) -> Self:
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError("Agent plan step_id values must be unique")
            unknown = set(step.depends_on) - seen
            if unknown:
                raise ValueError("Agent plan dependencies must reference previously declared steps")
            seen.add(step.step_id)
        expected = self.calculate_hash(
            plan_version=self.plan_version,
            intent_id=self.intent_id,
            steps=self.steps,
            planning_mode=self.planning_mode,
        )
        if self.plan_hash != expected:
            raise ValueError("plan_hash does not match the canonical Agent plan")
        return self


class AgentRunState(ContractModel):
    """Persistable structured state shared between the planner and safe executor."""

    run_id: str
    intent_id: str
    plan_hash: Sha256
    status: AgentRunStatus
    completed_step_ids: tuple[str, ...] = ()
    pending_approval_step_ids: tuple[str, ...] = ()
    result_ids: tuple[str, ...] = ()
    replan_count: int = Field(default=0, ge=0, le=2)
    warnings: tuple[str, ...] = ()
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _run_id_uuid = field_validator("run_id")(_uuid_string)
    _intent_id_uuid = field_validator("intent_id")(_uuid_string)
    _updated_at_utc = field_validator("updated_at")(_utc_datetime)

    @field_validator("completed_step_ids", "pending_approval_step_ids", "warnings")
    @classmethod
    def sequence_values_are_unique(cls, value: tuple[str, ...], info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "values")
        return _unique_nonblank(value, field_name=field_name)

    @field_validator("result_ids")
    @classmethod
    def result_ids_are_unique_uuids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(_uuid_string(item) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("result_ids must be unique")
        return normalized

    @model_validator(mode="after")
    def completed_steps_do_not_await_approval(self) -> Self:
        if set(self.completed_step_ids) & set(self.pending_approval_step_ids):
            raise ValueError("a completed step cannot also await approval")
        return self


class ApprovalRequest(ContractModel):
    """A time-bounded human gate for one externally consequential step."""

    approval_id: str
    run_id: str
    approval_kind: ApprovalKind
    source_plan_hash: Sha256
    step_id: str = Field(min_length=1)
    impact_scope: str = Field(min_length=2, max_length=2_000)
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime

    _approval_id_uuid = field_validator("approval_id")(_uuid_string)
    _run_id_uuid = field_validator("run_id")(_uuid_string)
    _created_at_utc = field_validator("created_at")(_utc_datetime)
    _expires_at_utc = field_validator("expires_at")(_utc_datetime)

    @field_validator("step_id", "impact_scope")
    @classmethod
    def text_fields_are_not_blank(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def expiry_is_after_creation(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at")
        return self


class KnowledgeDocumentManifest(ContractModel):
    """Review and provenance manifest for one knowledge-base source document."""

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    source_sha256: Sha256
    license_name: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    review_status: KnowledgeReviewStatus = KnowledgeReviewStatus.PENDING
    reviewer_id: str | None = Field(default=None, min_length=1)
    reviewed_at: datetime | None = None
    object_uri: str | None = Field(default=None, min_length=1)

    _reviewed_at_utc = field_validator("reviewed_at")(_optional_utc_datetime)

    @field_validator(
        "document_id",
        "title",
        "source_uri",
        "license_name",
        "document_version",
        "reviewer_id",
        "object_uri",
    )
    @classmethod
    def text_fields_are_not_blank(cls, value: str | None) -> str | None:
        return _nonblank(value) if value is not None else None

    @model_validator(mode="after")
    def approved_documents_have_a_review_record(self) -> Self:
        if self.review_status is KnowledgeReviewStatus.APPROVED and (
            self.reviewer_id is None or self.reviewed_at is None
        ):
            raise ValueError("approved knowledge documents require reviewer_id and reviewed_at")
        return self


class ScenarioLifetimeRequest(ContractModel):
    """Inputs for an audited cycle-to-years scenario conversion tool."""

    lifetime_result_id: str
    operation_policy_version: str = Field(min_length=1)
    equivalent_cycles_per_day: float = Field(gt=0, allow_inf_nan=False)
    days_per_year: float = Field(default=365.25, gt=0, allow_inf_nan=False)

    _lifetime_result_id_uuid = field_validator("lifetime_result_id")(_uuid_string)

    @field_validator("operation_policy_version")
    @classmethod
    def policy_version_is_not_blank(cls, value: str) -> str:
        return _nonblank(value)


class ScenarioLifetimeResult(ContractModel):
    """Tool-produced scenario value, never a long-term observed-life claim."""

    source_lifetime_result_id: str
    operation_policy_version: str = Field(min_length=1)
    equivalent_cycles_per_day: float = Field(gt=0, allow_inf_nan=False)
    days_per_year: float = Field(gt=0, allow_inf_nan=False)
    scenario_years: float = Field(ge=0, allow_inf_nan=False)
    limitations: tuple[str, ...] = Field(min_length=1)
    evidence_level: EvidenceLevel = EvidenceLevel.MODEL_INFERENCE

    _source_lifetime_result_id_uuid = field_validator("source_lifetime_result_id")(
        _uuid_string
    )

    @field_validator("operation_policy_version")
    @classmethod
    def policy_version_is_not_blank(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("limitations")
    @classmethod
    def limitations_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _unique_nonblank(value, field_name="limitations")

    @field_validator("evidence_level")
    @classmethod
    def evidence_is_model_inference(cls, value: EvidenceLevel) -> EvidenceLevel:
        if value is not EvidenceLevel.MODEL_INFERENCE:
            raise ValueError("scenario lifetime evidence_level must be MODEL_INFERENCE")
        return value


class FeishuBinding(ContractModel):
    """Secret-free mapping from one project to approved Feishu destinations."""

    binding_id: str
    project_id: str = Field(min_length=1)
    chat_id: str | None = Field(default=None, min_length=1)
    bitable_app_token: str | None = Field(default=None, min_length=1)
    bitable_table_id: str | None = Field(default=None, min_length=1)
    user_open_id_map: dict[str, str] = Field(default_factory=dict)
    binding_version: str = Field(min_length=1)

    _binding_id_uuid = field_validator("binding_id")(_uuid_string)

    @field_validator(
        "project_id", "chat_id", "bitable_app_token", "bitable_table_id", "binding_version"
    )
    @classmethod
    def text_fields_are_not_blank(cls, value: str | None) -> str | None:
        return _nonblank(value) if value is not None else None

    @field_validator("user_open_id_map")
    @classmethod
    def user_mapping_is_json_safe_and_nonblank(cls, value: dict[str, str]) -> dict[str, str]:
        mapping: JsonMapping = value
        _json_mapping(mapping)
        return {_nonblank(key): _nonblank(item) for key, item in value.items()}

    @model_validator(mode="after")
    def binding_has_a_complete_destination(self) -> Self:
        has_bitable = self.bitable_app_token is not None and self.bitable_table_id is not None
        has_partial_bitable = (self.bitable_app_token is None) != (self.bitable_table_id is None)
        if has_partial_bitable:
            raise ValueError("Feishu Bitable app token and table id must be configured together")
        if self.chat_id is None and not has_bitable:
            raise ValueError("Feishu binding requires a chat or a complete Bitable target")
        return self
