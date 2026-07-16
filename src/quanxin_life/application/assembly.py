"""Explicit assembly for the complete competition tool application.

This module only wires already implemented tools to caller-owned dependencies.
Importing it performs no model training, I/O, network access, or service start.
"""

from __future__ import annotations

from dataclasses import dataclass

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.model_artifacts import ModelArtifactRegistry
from quanxin_life.audit import AuditLedger
from quanxin_life.models import HybridDegradationPredictor
from quanxin_life.online import IndividualTrajectoryCalibrator
from quanxin_life.tools.audited_report import register_generate_audited_report_tool
from quanxin_life.tools.batch_decision import (
    VerifiedBatchDecisionPolicyResolver,
    register_batch_decision_tool,
)
from quanxin_life.tools.battery_evidence import (
    HybridBatteryEvidenceBackend,
    VerifiedKnowledgeScopeResolver,
    register_retrieve_battery_evidence_tool,
)
from quanxin_life.tools.conformal_calibration import (
    VerifiedNormalizedCalibrationCohortResolver,
    VerifiedPredictionDifficultyScaleResolver,
    register_calibrate_prediction_interval_tool,
)
from quanxin_life.tools.cycle_life_prediction import (
    CycleLifePredictor,
    register_predict_cycle_life_tool,
)
from quanxin_life.tools.data_quality import register_validate_battery_data_tool
from quanxin_life.tools.early_cycle_features import (
    VerifiedEarlyCycleBatchResolver,
    register_extract_early_cycle_features_tool,
)
from quanxin_life.tools.next_experiment_recommendation import (
    VerifiedExperimentRecommendationContextResolver,
    register_recommend_next_experiment_tool,
)
from quanxin_life.tools.observed_soh_ingestion import (
    VerifiedMeasurementResolver,
    register_ingest_newly_observed_soh_tool,
)
from quanxin_life.tools.online_update import register_update_cell_parameters_tool
from quanxin_life.tools.physics_check import (
    PhysicsValidator,
    register_check_operating_condition_tool,
)
from quanxin_life.tools.registry import ToolRegistry
from quanxin_life.tools.scenario_lifetime import register_scenario_lifetime_tool
from quanxin_life.tools.split_audit import register_audit_dataset_split_tool
from quanxin_life.tools.target_domain_adaptation import (
    VerifiedAdaptationCohortResolver,
    register_adapt_to_target_domain_tool,
)
from quanxin_life.tools.trajectory_prediction import register_predict_soh_trajectory_tool


@dataclass(frozen=True, slots=True)
class CompetitionToolDependencies:
    """All caller-owned dependencies needed by the competition tools."""

    audit_ledger: AuditLedger
    early_cycle_batch_resolver: VerifiedEarlyCycleBatchResolver
    cycle_life_predictor: CycleLifePredictor
    hybrid_degradation_predictor: HybridDegradationPredictor
    normalized_calibration_cohort_resolver: VerifiedNormalizedCalibrationCohortResolver
    prediction_difficulty_scale_resolver: VerifiedPredictionDifficultyScaleResolver | None
    adaptation_cohort_resolver: VerifiedAdaptationCohortResolver
    measurement_resolver: VerifiedMeasurementResolver
    individual_trajectory_calibrator: IndividualTrajectoryCalibrator | None
    physics_validator: PhysicsValidator
    experiment_recommendation_context_resolver: (
        VerifiedExperimentRecommendationContextResolver
    )
    batch_decision_policy_resolver: VerifiedBatchDecisionPolicyResolver
    knowledge_scope_resolver: VerifiedKnowledgeScopeResolver
    battery_evidence_backend: HybridBatteryEvidenceBackend
    model_artifact_registry: ModelArtifactRegistry | None = None

    def __post_init__(self) -> None:
        required = (
            self.audit_ledger,
            self.early_cycle_batch_resolver,
            self.cycle_life_predictor,
            self.hybrid_degradation_predictor,
            self.normalized_calibration_cohort_resolver,
            self.adaptation_cohort_resolver,
            self.measurement_resolver,
            self.physics_validator,
            self.experiment_recommendation_context_resolver,
            self.batch_decision_policy_resolver,
            self.knowledge_scope_resolver,
            self.battery_evidence_backend,
        )
        if any(dependency is None for dependency in required):
            raise TypeError("Required competition tool dependencies must not be None")


def create_competition_tool_registry(
    dependencies: CompetitionToolDependencies,
) -> ToolRegistry:
    """Register every standard tool exactly once in one explicit context."""

    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)
    register_audit_dataset_split_tool(registry)
    register_extract_early_cycle_features_tool(
        registry,
        resolver=dependencies.early_cycle_batch_resolver,
    )
    register_ingest_newly_observed_soh_tool(
        registry,
        resolver=dependencies.measurement_resolver,
    )
    register_predict_cycle_life_tool(
        registry,
        predictor=dependencies.cycle_life_predictor,
        audit_ledger=dependencies.audit_ledger,
        model_artifact_registry=dependencies.model_artifact_registry,
    )
    register_scenario_lifetime_tool(
        registry,
        audit_ledger=dependencies.audit_ledger,
    )
    register_predict_soh_trajectory_tool(
        registry,
        predictor=dependencies.hybrid_degradation_predictor,
        audit_ledger=dependencies.audit_ledger,
    )
    register_calibrate_prediction_interval_tool(
        registry,
        resolver=dependencies.normalized_calibration_cohort_resolver,
        audit_ledger=dependencies.audit_ledger,
        difficulty_scale_resolver=dependencies.prediction_difficulty_scale_resolver,
    )
    register_adapt_to_target_domain_tool(
        registry,
        audit_ledger=dependencies.audit_ledger,
        resolver=dependencies.adaptation_cohort_resolver,
    )
    register_update_cell_parameters_tool(
        registry,
        audit_ledger=dependencies.audit_ledger,
        calibrator=dependencies.individual_trajectory_calibrator,
    )
    register_check_operating_condition_tool(
        registry,
        validator=dependencies.physics_validator,
    )
    register_recommend_next_experiment_tool(
        registry,
        resolver=dependencies.experiment_recommendation_context_resolver,
    )
    register_batch_decision_tool(
        registry,
        audit_ledger=dependencies.audit_ledger,
        policy_resolver=dependencies.batch_decision_policy_resolver,
    )
    register_retrieve_battery_evidence_tool(
        registry,
        resolver=dependencies.knowledge_scope_resolver,
        backend=dependencies.battery_evidence_backend,
    )
    register_generate_audited_report_tool(
        registry,
        audit_ledger=dependencies.audit_ledger,
    )
    return registry


def create_competition_tool_invocation_service(
    dependencies: CompetitionToolDependencies,
) -> ToolInvocationService:
    """Wrap the complete competition registry in the shared invocation service."""

    registry = create_competition_tool_registry(dependencies)
    return ToolInvocationService(
        registry=registry,
        audit_ledger=dependencies.audit_ledger,
    )
