"""Contracts for trusted GP-based next-experiment recommendations."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.tools.registry import StandardToolName, ToolRegistry

if TYPE_CHECKING:
    from quanxin_life.tools.next_experiment_recommendation import (
        VerifiedExperimentRecommendationContext,
    )


def _provenance() -> tuple[ProvenanceRecord, ...]:
    return (
        ProvenanceRecord(
            source_id="naumann-cycle-reviewed-fixture",
            source_kind=SourceKind.OBSERVED,
            uri="trusted-store://naumann/cycle/fixture",
            sha256=sha256_canonical({"fixture": "naumann-cycle"}),
            description="Reviewed public Naumann condition-level measurement fixture",
            created_at=datetime(2026, 7, 15, tzinfo=UTC),
        ),
    )


def _context() -> VerifiedExperimentRecommendationContext:
    from quanxin_life.tools.next_experiment_recommendation import (
        VerifiedExperimentRecommendationContext,
    )

    return VerifiedExperimentRecommendationContext.model_validate(
        {
            "recommendation_context_id": "naumann-cycle-capacity-loss-v1",
            "dataset_id": "NAUMANN_CYCLE",
            "data_version": "naumann-cycle-v1",
            "feature_version": "naumann-condition-v1",
            "source_manifest_hash": sha256_canonical({"fixture": "naumann-cycle-manifest"}),
            "mapping_version": "naumann-gp-map-v1",
            "candidate_catalog_version": "naumann-candidates-v1",
            "operating_bounds": {
                "temperature_c": [15.0, 45.0],
                "mean_soc": [0.2, 0.8],
                "dod": [0.2, 0.8],
                "charge_c_rate": [0.2, 1.5],
                "discharge_c_rate": [0.2, 1.5],
            },
            "acquisition_config": {
                "time_normalizer_hours": 200.0,
                "equipment_cost_normalizer": 20.0,
                "duplicate_penalty_weight": 0.1,
                "similarity_length_scale": 0.25,
            },
            "target_name": "relative_capacity_loss_rate_per_efc",
            "strategy": "cost_aware_eivr",
            "batch_size": 1,
            "observations": [
                {
                    "observation_id": "naumann-observation-01",
                    "condition": {
                        "temperature_c": 20.0,
                        "mean_soc": 0.3,
                        "dod": 0.3,
                        "charge_c_rate": 0.3,
                        "discharge_c_rate": 0.3,
                    },
                    "observed_target": 0.01,
                    "duration_hours": 100.0,
                    "equipment_cost": 5.0,
                    "source_observation_id": "NAUMANN_CYCLE:condition-01:100",
                    "source_record_hash": sha256_canonical({"source": "observation-01"}),
                },
                {
                    "observation_id": "naumann-observation-02",
                    "condition": {
                        "temperature_c": 25.0,
                        "mean_soc": 0.4,
                        "dod": 0.4,
                        "charge_c_rate": 0.5,
                        "discharge_c_rate": 0.5,
                    },
                    "observed_target": 0.02,
                    "duration_hours": 120.0,
                    "equipment_cost": 6.0,
                    "source_observation_id": "NAUMANN_CYCLE:condition-02:200",
                    "source_record_hash": sha256_canonical({"source": "observation-02"}),
                },
                {
                    "observation_id": "naumann-observation-03",
                    "condition": {
                        "temperature_c": 35.0,
                        "mean_soc": 0.6,
                        "dod": 0.6,
                        "charge_c_rate": 1.0,
                        "discharge_c_rate": 1.0,
                    },
                    "observed_target": 0.06,
                    "duration_hours": 150.0,
                    "equipment_cost": 8.0,
                    "source_observation_id": "NAUMANN_CYCLE:condition-03:300",
                    "source_record_hash": sha256_canonical({"source": "observation-03"}),
                },
            ],
            "reference_conditions": [
                {
                    "temperature_c": 25.0,
                    "mean_soc": 0.5,
                    "dod": 0.5,
                    "charge_c_rate": 0.8,
                    "discharge_c_rate": 0.8,
                },
                {
                    "temperature_c": 40.0,
                    "mean_soc": 0.7,
                    "dod": 0.4,
                    "charge_c_rate": 1.2,
                    "discharge_c_rate": 1.2,
                },
            ],
            "candidates": [
                {
                    "candidate_id": "candidate-safe",
                    "condition": {
                        "temperature_c": 30.0,
                        "mean_soc": 0.5,
                        "dod": 0.5,
                        "charge_c_rate": 0.8,
                        "discharge_c_rate": 0.8,
                    },
                    "duration_hours": 140.0,
                    "equipment_cost": 7.0,
                    "safety_approved": True,
                    "equipment_available": True,
                },
                {
                    "candidate_id": "candidate-unapproved",
                    "condition": {
                        "temperature_c": 42.0,
                        "mean_soc": 0.7,
                        "dod": 0.4,
                        "charge_c_rate": 1.2,
                        "discharge_c_rate": 1.2,
                    },
                    "duration_hours": 180.0,
                    "equipment_cost": 9.0,
                    "safety_approved": False,
                    "equipment_available": True,
                },
            ],
            "provenance": [item.model_dump(mode="json") for item in _provenance()],
        }
    )


class _RecommendationContextResolver:
    def __init__(self, context: VerifiedExperimentRecommendationContext) -> None:
        self.context = context

    def resolve_verified_experiment_recommendation_context(
        self, recommendation_context_id: str
    ) -> VerifiedExperimentRecommendationContext:
        if recommendation_context_id != self.context.recommendation_context_id:
            raise ValueError("trusted experiment recommendation context was not found")
        return self.context


class _MismatchedRecommendationContextResolver:
    def __init__(self, context: VerifiedExperimentRecommendationContext) -> None:
        self.context = context

    def resolve_verified_experiment_recommendation_context(
        self, recommendation_context_id: str
    ) -> VerifiedExperimentRecommendationContext:
        return self.context


def test_next_experiment_tool_returns_auditable_batch_from_trusted_context() -> None:
    from quanxin_life.tools.next_experiment_recommendation import (
        EXPERIMENT_RECOMMENDATION_EVIDENCE_TYPE,
        RecommendNextExperimentToolInput,
        register_recommend_next_experiment_tool,
    )

    context = _context()
    registry = ToolRegistry()
    register_recommend_next_experiment_tool(
        registry,
        resolver=_RecommendationContextResolver(context),
    )

    result = registry.execute(
        StandardToolName.RECOMMEND_NEXT_EXPERIMENT,
        RecommendNextExperimentToolInput(
            recommendation_context_id=context.recommendation_context_id
        ),
    )

    assert result.tool_name == StandardToolName.RECOMMEND_NEXT_EXPERIMENT.value
    assert result.tool_version == "next-experiment-recommendation-tool-v1"
    assert result.data_version == "naumann-cycle-v1"
    assert result.feature_version == "naumann-condition-v1"
    assert result.values["artifact_type"] == EXPERIMENT_RECOMMENDATION_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert artifact["recommendation_context_id"] == context.recommendation_context_id
    assert artifact["selected"][0]["candidate_id"] == "candidate-safe"
    assert artifact["rejected"][0]["candidate_id"] == "candidate-unapproved"
    assert artifact["rejected"][0]["rejection_reasons"] == ["SAFETY_NOT_APPROVED"]
    assert artifact["observation_record_hashes"] == [
        sha256_canonical({"source": "observation-01"}),
        sha256_canonical({"source": "observation-02"}),
        sha256_canonical({"source": "observation-03"}),
    ]
    assert "RECOMMENDATION_REQUIRES_HUMAN_SAFETY_APPROVAL" in result.warnings


def test_next_experiment_public_input_rejects_raw_observations_and_candidates() -> None:
    from quanxin_life.tools.next_experiment_recommendation import (
        RecommendNextExperimentToolInput,
    )

    with pytest.raises(ValueError, match="Extra inputs"):
        RecommendNextExperimentToolInput.model_validate(
            {
                "recommendation_context_id": "naumann-cycle-capacity-loss-v1",
                "observations": [{"observed_target": 0.02}],
                "candidates": [{"candidate_id": "unsafe-client-candidate"}],
                "operating_bounds": {"temperature_c": [0.0, 100.0]},
            }
        )


def test_next_experiment_context_must_hold_observed_provenance_and_exact_identity() -> None:
    from quanxin_life.tools.next_experiment_recommendation import (
        RecommendNextExperimentToolInput,
        execute_recommend_next_experiment_tool,
    )

    context = _context()
    missing_observed_provenance = context.model_copy(
        update={
            "provenance": (
                context.provenance[0].model_copy(update={"source_kind": SourceKind.PREDICTED}),
            )
        }
    )
    with pytest.raises(ValueError, match="OBSERVED"):
        execute_recommend_next_experiment_tool(
            RecommendNextExperimentToolInput(
                recommendation_context_id=context.recommendation_context_id
            ),
            resolver=_RecommendationContextResolver(missing_observed_provenance),
        )

    mismatched_context = context.model_copy(update={"recommendation_context_id": "other"})
    with pytest.raises(ValueError, match="mismatched"):
        execute_recommend_next_experiment_tool(
            RecommendNextExperimentToolInput(
                recommendation_context_id=context.recommendation_context_id
            ),
            resolver=_MismatchedRecommendationContextResolver(mismatched_context),
        )
