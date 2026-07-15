"""Trusted Gaussian-process recommendations for the next ageing experiment.

The public boundary deliberately accepts only a server-owned context ID.  Raw
Naumann observations, candidate operating conditions, safety decisions, costs,
and acquisition settings are resolved from a reviewed server-side context.
Consequently neither a client nor an LLM can fabricate an experiment outcome,
relax an operating limit, or silently alter the ranking policy.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.experiments.gp_active import (
    GP_MODEL_VERSION,
    AcquisitionConfig,
    AcquisitionMetadata,
    AcquisitionStrategy,
    BatchExperimentRecommendation,
    CandidateAssessment,
    ExperimentCandidate,
    ExperimentObservation,
    GaussianProcessExperimentRecommender,
    OperatingBounds,
    OperatingCondition,
)
from quanxin_life.tools.registry import (
    RegisteredTool,
    StandardToolName,
    ToolDefinition,
    ToolRegistry,
)

NEXT_EXPERIMENT_RECOMMENDATION_TOOL_VERSION = "next-experiment-recommendation-tool-v1"
EXPERIMENT_RECOMMENDATION_EVIDENCE_TYPE = "quanxin_life.experiment_recommendation.v1"
HUMAN_SAFETY_APPROVAL_WARNING = "RECOMMENDATION_REQUIRES_HUMAN_SAFETY_APPROVAL"
PUBLIC_LAB_DATA_WARNING = "PUBLIC_LABORATORY_DATA_NOT_INDUSTRIAL_DEPLOYMENT_EVIDENCE"
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class _TrustedRecommendationModel(ContractModel):
    """Strict resolver-only context; never exposed as a public tool input."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class TrustedOperatingCondition(_TrustedRecommendationModel):
    """A condition reviewed by the server-side experiment-design service."""

    temperature_c: float = Field(allow_inf_nan=False)
    mean_soc: float = Field(allow_inf_nan=False)
    dod: float = Field(allow_inf_nan=False)
    charge_c_rate: float = Field(allow_inf_nan=False)
    discharge_c_rate: float = Field(allow_inf_nan=False)

    def to_numerical(self) -> OperatingCondition:
        return OperatingCondition(**self.model_dump(mode="python"))

    def condition_key(self) -> tuple[float, float, float, float, float]:
        return self.to_numerical().as_tuple()


class TrustedOperatingBounds(_TrustedRecommendationModel):
    """Reviewed physical and equipment boundaries for one candidate catalog."""

    temperature_c: tuple[float, float]
    mean_soc: tuple[float, float]
    dod: tuple[float, float]
    charge_c_rate: tuple[float, float]
    discharge_c_rate: tuple[float, float]

    def to_numerical(self) -> OperatingBounds:
        return OperatingBounds(**self.model_dump(mode="python"))


class TrustedAcquisitionConfig(_TrustedRecommendationModel):
    """Reviewed GP acquisition constants, unavailable to public callers."""

    time_normalizer_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost_normalizer: float = Field(gt=0, allow_inf_nan=False)
    duplicate_penalty_weight: float = Field(ge=0, allow_inf_nan=False)
    similarity_length_scale: float = Field(gt=0, allow_inf_nan=False)

    def to_numerical(self) -> AcquisitionConfig:
        return AcquisitionConfig(**self.model_dump(mode="python"))


class TrustedExperimentObservation(_TrustedRecommendationModel):
    """One reviewed, actually observed condition-level source measurement."""

    observation_id: str = Field(min_length=1)
    condition: TrustedOperatingCondition
    observed_target: float = Field(allow_inf_nan=False)
    duration_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost: float = Field(gt=0, allow_inf_nan=False)
    source_observation_id: str = Field(min_length=1)
    source_record_hash: Sha256

    def to_numerical(self, *, target_name: str) -> ExperimentObservation:
        return ExperimentObservation(
            observation_id=self.observation_id,
            condition=self.condition.to_numerical(),
            target_name=target_name,
            observed_target=self.observed_target,
            duration_hours=self.duration_hours,
            equipment_cost=self.equipment_cost,
        )


class TrustedExperimentCandidate(_TrustedRecommendationModel):
    """One catalog candidate with explicit safety and equipment declarations."""

    candidate_id: str = Field(min_length=1)
    condition: TrustedOperatingCondition
    duration_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost: float = Field(gt=0, allow_inf_nan=False)
    safety_approved: bool | None = None
    equipment_available: bool | None = None

    def to_numerical(self) -> ExperimentCandidate:
        return ExperimentCandidate(
            candidate_id=self.candidate_id,
            condition=self.condition.to_numerical(),
            duration_hours=self.duration_hours,
            equipment_cost=self.equipment_cost,
            safety_approved=self.safety_approved,
            equipment_available=self.equipment_available,
        )


class VerifiedExperimentRecommendationContext(_TrustedRecommendationModel):
    """Server-owned observations, candidate catalog and numerical GP policy.

    This is intentionally the only place where raw condition-level Naumann
    numbers may be assembled for the recommendation service.  A context is
    created by an audited data/review workflow; it cannot be submitted through
    API, MCP, Agent or UI tool input.
    """

    recommendation_context_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    source_manifest_hash: Sha256
    mapping_version: str = Field(min_length=1)
    candidate_catalog_version: str = Field(min_length=1)
    operating_bounds: TrustedOperatingBounds
    acquisition_config: TrustedAcquisitionConfig
    target_name: str = Field(min_length=1)
    strategy: AcquisitionStrategy = AcquisitionStrategy.COST_AWARE_EIVR
    batch_size: int = Field(gt=0)
    observations: tuple[TrustedExperimentObservation, ...] = Field(min_length=2)
    reference_conditions: tuple[TrustedOperatingCondition, ...] = Field(min_length=1)
    candidates: tuple[TrustedExperimentCandidate, ...] = Field(min_length=1)
    provenance: tuple[ProvenanceRecord, ...] = Field(min_length=1)

    @field_validator(
        "recommendation_context_id",
        "dataset_id",
        "data_version",
        "feature_version",
        "mapping_version",
        "candidate_catalog_version",
        "target_name",
    )
    @classmethod
    def require_nonblank_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("recommendation context identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_reviewed_cohort_and_supported_policy(
        self,
    ) -> VerifiedExperimentRecommendationContext:
        if self.strategy not in (
            AcquisitionStrategy.COST_AWARE_EIVR,
            AcquisitionStrategy.MAX_VARIANCE,
        ):
            raise ValueError("recommendation context strategy is not supported by GP ranking")
        if not any(item.source_kind is SourceKind.OBSERVED for item in self.provenance):
            raise ValueError("recommendation context provenance must include an OBSERVED source")

        observation_ids = [item.observation_id for item in self.observations]
        source_observation_ids = [item.source_observation_id for item in self.observations]
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("recommendation context observation_id values must be unique")
        if len(source_observation_ids) != len(set(source_observation_ids)):
            raise ValueError("recommendation context source_observation_id values must be unique")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("recommendation context candidate_id values must be unique")

        bounds = self.operating_bounds.to_numerical()
        for observation in self.observations:
            violations = bounds.violations(observation.condition.to_numerical())
            if violations:
                raise ValueError(
                    "reviewed observed condition violates operating bounds: "
                    + ", ".join(violations)
                )
        reference_keys = [item.condition_key() for item in self.reference_conditions]
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("recommendation reference_conditions must be unique")
        for reference in self.reference_conditions:
            violations = bounds.violations(reference.to_numerical())
            if violations:
                raise ValueError(
                    "recommendation reference condition violates operating bounds: "
                    + ", ".join(violations)
                )
        return self


class VerifiedExperimentRecommendationContextResolver(Protocol):
    """Server-side lookup for an approved experiment-design context."""

    def resolve_verified_experiment_recommendation_context(
        self, recommendation_context_id: str
    ) -> VerifiedExperimentRecommendationContext: ...


class RecommendNextExperimentToolInput(ContractModel):
    """Public input contains no measured values, candidates or policy values."""

    recommendation_context_id: str = Field(min_length=1)

    @field_validator("recommendation_context_id")
    @classmethod
    def require_nonblank_context_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("recommendation_context_id must not be blank")
        return normalized


def _execution_timestamp(clock: Clock) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _resolve_verified_context(
    resolver: VerifiedExperimentRecommendationContextResolver,
    recommendation_context_id: str,
) -> VerifiedExperimentRecommendationContext:
    resolved = resolver.resolve_verified_experiment_recommendation_context(
        recommendation_context_id
    )
    try:
        context = VerifiedExperimentRecommendationContext.model_validate(
            resolved.model_dump(mode="json")
        )
    except ValidationError as exc:
        message = str(exc.errors(include_url=False)[0]["msg"])
        raise ValueError(
            f"trusted experiment recommendation context violates its contract: {message}"
        ) from exc
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("trusted experiment recommendation context is invalid") from exc
    if context.recommendation_context_id != recommendation_context_id:
        raise ValueError("trusted recommendation resolver returned a mismatched context identifier")
    return context


def _metadata_payload(metadata: AcquisitionMetadata | None) -> dict[str, object] | None:
    if metadata is None:
        return None
    return {
        "strategy": metadata.strategy.value,
        "reference_condition_count": metadata.reference_condition_count,
        "reference_total_variance_before": metadata.reference_total_variance_before,
        "reference_total_variance_after": metadata.reference_total_variance_after,
        "expected_variance_reduction": metadata.expected_variance_reduction,
    }


def _assessment_payload(assessment: CandidateAssessment) -> dict[str, object]:
    """Serialize a numerical assessment without dropping rejected candidates."""

    return {
        "candidate_id": assessment.candidate_id,
        "accepted": assessment.accepted,
        "predicted_mean": assessment.predicted_mean,
        "predicted_std": assessment.predicted_std,
        "normalized_cost": assessment.normalized_cost,
        "duplicate_penalty": assessment.duplicate_penalty,
        "acquisition_score": assessment.acquisition_score,
        "rejection_reasons": list(assessment.rejection_reasons),
        "constraint_details": list(assessment.constraint_details),
        "acquisition_metadata": _metadata_payload(assessment.acquisition_metadata),
    }


def _recommendation_artifact(
    context: VerifiedExperimentRecommendationContext,
    recommendation: BatchExperimentRecommendation,
) -> dict[str, object]:
    """Create a JSON-only numerical record for the audit ledger and report layer."""

    context_policy = {
        "operating_bounds": context.operating_bounds.model_dump(mode="json"),
        "acquisition_config": context.acquisition_config.model_dump(mode="json"),
        "strategy": context.strategy.value,
        "batch_size": context.batch_size,
        "target_name": context.target_name,
        "reference_conditions": [
            item.model_dump(mode="json") for item in context.reference_conditions
        ],
    }
    return {
        "recommendation_context_id": context.recommendation_context_id,
        "dataset_id": context.dataset_id,
        "source_manifest_hash": context.source_manifest_hash,
        "mapping_version": context.mapping_version,
        "candidate_catalog_version": context.candidate_catalog_version,
        "gp_model_version": recommendation.model_version,
        "target_name": recommendation.target_name,
        "strategy": recommendation.strategy.value,
        "batch_size": context.batch_size,
        "context_policy": context_policy,
        "context_policy_hash": sha256_canonical(context_policy),
        "observation_ids": [item.observation_id for item in context.observations],
        "source_observation_ids": [item.source_observation_id for item in context.observations],
        "observation_record_hashes": [item.source_record_hash for item in context.observations],
        "selected": [_assessment_payload(item) for item in recommendation.selected],
        "rejected": [_assessment_payload(item) for item in recommendation.rejected],
    }


def execute_recommend_next_experiment_tool(
    input_value: RecommendNextExperimentToolInput,
    *,
    resolver: VerifiedExperimentRecommendationContextResolver,
    clock: Clock = _utc_now,
) -> ToolResult:
    """Fit a GP only on reviewed observations and recommend a safe trial batch."""

    validated_input = RecommendNextExperimentToolInput.model_validate(
        input_value.model_dump(mode="json")
    )
    context = _resolve_verified_context(resolver, validated_input.recommendation_context_id)
    recommender = GaussianProcessExperimentRecommender(
        operating_bounds=context.operating_bounds.to_numerical(),
        acquisition_config=context.acquisition_config.to_numerical(),
    ).fit(
        tuple(item.to_numerical(target_name=context.target_name) for item in context.observations)
    )
    recommendation = recommender.recommend_batch(
        tuple(item.to_numerical() for item in context.candidates),
        batch_size=context.batch_size,
        reference_conditions=tuple(
            item.to_numerical() for item in context.reference_conditions
        ),
        strategy=context.strategy,
    )
    artifact = _recommendation_artifact(context, recommendation)
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.RECOMMEND_NEXT_EXPERIMENT.value,
        tool_version=NEXT_EXPERIMENT_RECOMMENDATION_TOOL_VERSION,
        model_version=GP_MODEL_VERSION,
        data_version=context.data_version,
        feature_version=context.feature_version,
        input_hash=sha256_canonical(validated_input.model_dump(mode="json")),
        values={"artifact_type": EXPERIMENT_RECOMMENDATION_EVIDENCE_TYPE, "artifact": artifact},
        uncertainty={
            "uncertainty_type": "gaussian_process_posterior_standard_deviation",
            "selected_candidate_ids": [item.candidate_id for item in recommendation.selected],
        },
        warnings=[HUMAN_SAFETY_APPROVAL_WARNING, PUBLIC_LAB_DATA_WARNING],
        provenance=list(context.provenance),
        created_at=_execution_timestamp(clock),
    )


def register_recommend_next_experiment_tool(
    registry: ToolRegistry,
    *,
    resolver: VerifiedExperimentRecommendationContextResolver,
    clock: Clock = _utc_now,
) -> RegisteredTool[RecommendNextExperimentToolInput]:
    """Register the experiment service only with its reviewed context resolver."""

    return registry.register(
        ToolDefinition(
            tool_name=StandardToolName.RECOMMEND_NEXT_EXPERIMENT,
            tool_version=NEXT_EXPERIMENT_RECOMMENDATION_TOOL_VERSION,
            input_model=RecommendNextExperimentToolInput,
            executor=lambda input_value: execute_recommend_next_experiment_tool(
                input_value,
                resolver=resolver,
                clock=clock,
            ),
        )
    )
