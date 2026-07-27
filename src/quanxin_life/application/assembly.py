"""Explicit assembly for the complete competition tool application.

This module only wires already implemented tools to caller-owned dependencies.
Importing it performs no model training, I/O, network access, or service start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.agent_run_invocation import AgentRunInvocationValidator
from quanxin_life.application.model_artifacts import ModelArtifactRegistry
from quanxin_life.audit import AuditLedger, ProjectResultLedger
from quanxin_life.audit.project_ledger import ProjectContextValidator
from quanxin_life.models import HybridDegradationPredictor
from quanxin_life.online import IndividualTrajectoryCalibrator
from quanxin_life.tools.advanced_conformal import (
    register_project_advanced_split_conformal_tool,
)
from quanxin_life.tools.advanced_cycle_life_prediction import (
    AdvancedRULInferenceService,
    register_project_predict_advanced_rul_tool,
)
from quanxin_life.tools.advanced_input import (
    ProjectEarlyCycleBatchResolver,
    register_project_prepare_advanced_input_tool,
)
from quanxin_life.tools.advanced_project_report import (
    register_project_generate_advanced_cell_report_tool,
)
from quanxin_life.tools.advanced_soh_prediction import (
    AdvancedSOHInferenceService,
    register_project_predict_advanced_soh_tool,
)
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
from quanxin_life.tools.trajectory_prediction import (
    register_predict_soh_trajectory_tool,
)

if TYPE_CHECKING:
    from quanxin_life.agents.execution_adapter import (
        AgentExecutionContextResolver,
    )
    from quanxin_life.api.advanced_calibration import (
        AdvancedCalibrationHttpAdapter,
        AdvancedCalibrationQueue,
    )
    from quanxin_life.api.auth import AuthHttpAdapter
    from quanxin_life.application.advanced_agent_execution_context import (
        AdvancedAgentExecutionContextResolver,
        BoundRecordBatchResolver,
    )
    from quanxin_life.application.advanced_calibration_evidence import (
        AdvancedCalibrationEvidenceResolver,
    )
    from quanxin_life.application.advanced_calibration_jobs import (
        AdvancedCalibrationMaterializationService,
        AdvancedCalibrationMaterializationWorker,
    )
    from quanxin_life.application.advanced_calibration_materialization import (
        AdvancedCalibrationCellInputResolver,
        AdvancedCalibrationRuntimeResolver,
    )
    from quanxin_life.application.invocation_context import (
        ProjectInvocationContextService,
    )
    from quanxin_life.audit.project_ledger import (
        AtomicProjectResultMaterializer,
    )
    from quanxin_life.persistence.database import SessionFactory


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


@dataclass(frozen=True, slots=True)
class ProjectPredictionToolDependencies:
    """Trusted dependencies for project-scoped RUL, SOH and Conformal tools."""

    project_audit_ledger: ProjectResultLedger
    project_context_validator: ProjectContextValidator
    agent_run_invocation_resolver: AgentRunInvocationValidator
    advanced_input_batch_resolver: ProjectEarlyCycleBatchResolver
    advanced_rul_inference_service: AdvancedRULInferenceService
    advanced_soh_inference_service: AdvancedSOHInferenceService

    def __post_init__(self) -> None:
        required = (
            self.project_audit_ledger,
            self.project_context_validator,
            self.agent_run_invocation_resolver,
            self.advanced_input_batch_resolver,
            self.advanced_rul_inference_service,
            self.advanced_soh_inference_service,
        )
        if any(dependency is None for dependency in required):
            raise TypeError("Required project prediction dependencies must not be None")


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationAssemblyDependencies:
    """Caller-owned dependencies for the trusted calibration vertical slice."""

    session_factory: SessionFactory
    context_service: ProjectInvocationContextService
    evidence_resolver: AdvancedCalibrationEvidenceResolver
    runtime_resolver: AdvancedCalibrationRuntimeResolver
    cell_input_resolver: AdvancedCalibrationCellInputResolver
    project_materializer: AtomicProjectResultMaterializer
    queue: AdvancedCalibrationQueue
    auth_adapter: AuthHttpAdapter
    agent_context_delegate: AgentExecutionContextResolver
    target_record_batch_resolver: BoundRecordBatchResolver

    def __post_init__(self) -> None:
        if any(
            dependency is None
            for dependency in (
                self.session_factory,
                self.context_service,
                self.evidence_resolver,
                self.runtime_resolver,
                self.cell_input_resolver,
                self.project_materializer,
                self.queue,
                self.auth_adapter,
                self.agent_context_delegate,
                self.target_record_batch_resolver,
            )
        ):
            raise TypeError(
                "Required Advanced calibration dependencies must not be None"
            )


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationComponents:
    """Explicitly assembled services without starting workers or loading models."""

    materialization_service: AdvancedCalibrationMaterializationService
    worker: AdvancedCalibrationMaterializationWorker
    http_adapter: AdvancedCalibrationHttpAdapter
    agent_context_resolver: AdvancedAgentExecutionContextResolver


def create_advanced_calibration_components(
    dependencies: AdvancedCalibrationAssemblyDependencies,
) -> AdvancedCalibrationComponents:
    """Wire the trusted calibration API, worker and Agent context boundary."""

    from quanxin_life.api.advanced_calibration import (
        create_advanced_calibration_http_adapter,
    )
    from quanxin_life.application.advanced_agent_execution_context import (
        AdvancedAgentExecutionContextResolver,
    )
    from quanxin_life.application.advanced_calibration_jobs import (
        AdvancedCalibrationMaterializationService,
        AdvancedCalibrationMaterializationWorker,
    )
    from quanxin_life.application.advanced_calibration_materialization import (
        AdvancedCalibrationSampleProducer,
        VerifiedAdvancedCalibrationCellPredictor,
    )
    from quanxin_life.application.advanced_calibration_preparation import (
        ActiveAdvancedCalibrationPreparationResolver,
    )

    preparation_resolver = ActiveAdvancedCalibrationPreparationResolver(
        runtime_resolver=dependencies.runtime_resolver,
        evidence_resolver=dependencies.evidence_resolver,
    )
    predictor = VerifiedAdvancedCalibrationCellPredictor(
        input_resolver=dependencies.cell_input_resolver,
    )
    producer = AdvancedCalibrationSampleProducer(
        evidence_resolver=dependencies.evidence_resolver,
        runtime_resolver=dependencies.runtime_resolver,
        predictor=predictor,
    )
    service = AdvancedCalibrationMaterializationService(
        dependencies.session_factory,
        preparation_resolver=preparation_resolver,
    )
    worker = AdvancedCalibrationMaterializationWorker(
        dependencies.session_factory,
        context_service=dependencies.context_service,
        producer=producer,
        materializer=dependencies.project_materializer,
    )
    agent_context_resolver = AdvancedAgentExecutionContextResolver(
        dependencies.session_factory,
        context_service=dependencies.context_service,
        record_batch_resolver=dependencies.target_record_batch_resolver,
        runtime_resolver=dependencies.runtime_resolver,
        delegate=dependencies.agent_context_delegate,
    )
    http_adapter = create_advanced_calibration_http_adapter(
        service,
        queue=dependencies.queue,
        context_service=dependencies.context_service,
        auth_adapter=dependencies.auth_adapter,
    )
    return AdvancedCalibrationComponents(
        materialization_service=service,
        worker=worker,
        http_adapter=http_adapter,
        agent_context_resolver=agent_context_resolver,
    )


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


def create_project_prediction_tool_registry(
    dependencies: ProjectPredictionToolDependencies,
) -> ToolRegistry:
    """Register only project-isolated formal prediction tools."""

    registry = ToolRegistry(
        project_context_validator=dependencies.project_context_validator
    )
    register_project_prepare_advanced_input_tool(
        registry,
        batch_resolver=dependencies.advanced_input_batch_resolver,
    )
    register_project_predict_advanced_rul_tool(
        registry,
        inference_service=dependencies.advanced_rul_inference_service,
        project_audit_ledger=dependencies.project_audit_ledger,
    )
    register_project_predict_advanced_soh_tool(
        registry,
        inference_service=dependencies.advanced_soh_inference_service,
        project_audit_ledger=dependencies.project_audit_ledger,
    )
    register_project_advanced_split_conformal_tool(
        registry,
        project_audit_ledger=dependencies.project_audit_ledger,
    )
    register_project_generate_advanced_cell_report_tool(
        registry,
        project_audit_ledger=dependencies.project_audit_ledger,
    )
    return registry


def create_project_prediction_tool_invocation_service(
    dependencies: ProjectPredictionToolDependencies,
) -> ToolInvocationService:
    """Assemble formal project prediction execution without GLOBAL exposure."""

    return ToolInvocationService(
        registry=create_project_prediction_tool_registry(dependencies),
        project_audit_ledger=dependencies.project_audit_ledger,
        agent_run_invocation_validator=dependencies.agent_run_invocation_resolver,
    )
