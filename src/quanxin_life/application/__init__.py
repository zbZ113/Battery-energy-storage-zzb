"""Lazy, side-effect-free exports for deployable application contexts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "AdvancedModelArtifactCatalogBatchRecord": "model_artifact_catalog",
    "AdvancedCalibrationAssemblyDependencies": "assembly",
    "AdvancedCalibrationComponents": "assembly",
    "ArtifactFormat": "model_artifacts",
    "ArtifactKind": "model_artifacts",
    "BINDING_SCHEMA_VERSION": "record_batch_bindings",
    "CANONICAL_CYCLE_CSV_FIELDS": "ingestion",
    "CanonicalCsvBatchRegistration": "ingestion",
    "ClassicModelArtifactCatalogSource": "model_artifact_catalog",
    "CompetitionToolDependencies": "assembly",
    "FileSystemVerifiedEarlyCycleBatchStore": "ingestion",
    "FeishuAilyAssemblyConfig": "feishu_aily_assembly",
    "FeishuAilyComponents": "feishu_aily_assembly",
    "FeishuCsvRegistrationResolver": "feishu_aily_assembly",
    "InMemoryVerifiedEarlyCycleBatchStore": "ingestion",
    "LifetimeDecisionWorkflowRequest": "lifetime_workflow",
    "LifetimeDecisionWorkflowResult": "lifetime_workflow",
    "LifetimeDecisionWorkflowStatus": "lifetime_workflow",
    "ModelArtifactCatalogRecord": "model_artifact_catalog",
    "ModelArtifactCatalogService": "model_artifact_catalog",
    "ModelArtifactManifest": "model_artifacts",
    "ModelArtifactPolicy": "lifetime_workflow",
    "ModelArtifactRegistry": "model_artifacts",
    "ProjectInvocationAccessError": "invocation_context",
    "ProjectInvocationContextService": "invocation_context",
    "ProjectInvocationNotFoundError": "invocation_context",
    "ProjectInvocationSource": "invocation_context",
    "ProjectPredictionToolDependencies": "assembly",
    "RegisteredFeishuCsvRegistrationResolver": "feishu_aily_assembly",
    "RecordBatchBindingAccessError": "record_batch_bindings",
    "RecordBatchBindingNotFoundError": "record_batch_bindings",
    "RecordBatchBindingRecord": "record_batch_bindings",
    "RecordBatchBindingService": "record_batch_bindings",
    "RecordBatchBindingStateError": "record_batch_bindings",
    "VerifiedEarlyCycleBatchStore": "ingestion",
    "VerifiedModelArtifact": "model_artifacts",
    "VerifiedModelArtifactMetadata": "model_artifact_catalog",
    "VerifiedModelArtifactRegistration": "model_artifact_catalog",
    "VerifiedProjectInvocationContext": "invocation_context",
    "create_competition_tool_invocation_service": "assembly",
    "create_competition_tool_registry": "assembly",
    "create_advanced_calibration_components": "assembly",
    "create_feishu_aily_components": "feishu_aily_assembly",
    "create_project_prediction_tool_invocation_service": "assembly",
    "create_project_prediction_tool_registry": "assembly",
    "load_verified_xgboost_life_predictor": "model_artifacts",
    "run_lifetime_decision_workflow": "lifetime_workflow",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
