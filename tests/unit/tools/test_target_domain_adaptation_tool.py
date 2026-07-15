"""Contracts for ledger-bound CORAL target-domain adaptation evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from quanxin_life.audit import AuditLedger
from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.tools.early_cycle_features import (
    EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
    EARLY_CYCLE_FEATURE_TOOL_VERSION,
    EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
)
from quanxin_life.tools.registry import StandardToolName

if TYPE_CHECKING:
    from quanxin_life.tools.target_domain_adaptation import VerifiedAdaptationCohort


def _provenance(source_id: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_id=source_id,
        source_kind=SourceKind.OBSERVED,
        uri=f"trusted-store://adaptation/{source_id}",
        sha256=sha256_canonical({"fixture": source_id}),
        description=f"Trusted observed early-cycle evidence for {source_id}",
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _feature_result(
    *,
    dataset_id: str,
    cell_id: str,
    split_version: str,
    data_version: str,
    capacity_drop: float,
    resistance_growth: float,
    cutoff_cycle: int = 20,
) -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name=StandardToolName.EXTRACT_EARLY_CYCLE_FEATURES.value,
        tool_version=EARLY_CYCLE_FEATURE_TOOL_VERSION,
        model_version=EARLY_CYCLE_FEATURE_TOOL_MODEL_VERSION,
        data_version=data_version,
        feature_version="early-cycle-v1",
        input_hash=sha256_canonical({"fixture": f"{dataset_id}:{cell_id}"}),
        values={
            "artifact_type": EARLY_CYCLE_TRAJECTORY_EVIDENCE_TYPE,
            "artifact": {
                "record_batch_id": f"batch-{dataset_id}-{cell_id}",
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "cutoff_cycle": cutoff_cycle,
                "observed_cycles": [0, cutoff_cycle],
                "observed_soh": [1.0, 0.99],
                "reference_capacity_ah": 1.0,
                "reference_capacity_method": "fixture",
                "feature_values": {
                    "capacity_drop": capacity_drop,
                    "resistance_growth": resistance_growth,
                },
                "condition_features": {
                    "capacity_drop": capacity_drop,
                    "resistance_growth": resistance_growth,
                },
                "source_cycles": [0, cutoff_cycle],
                "feature_warnings": [],
                "source_manifest_hash": sha256_canonical(
                    {"fixture": f"manifest:{dataset_id}:{cell_id}"}
                ),
                "feature_version": "early-cycle-v1",
                "split_version": split_version,
                "data_version": data_version,
            },
        },
        provenance=[_provenance(f"{dataset_id}-{cell_id}")],
        created_at=datetime(2026, 7, 15, tzinfo=UTC),
    )


def _source_split() -> SplitManifest:
    return SplitManifest(
        dataset_id="MATR",
        train=("source-a", "source-b"),
        validation=("source-validation",),
        calibration=("source-calibration",),
        test=("source-test",),
    )


def _target_split() -> SplitManifest:
    return SplitManifest(
        dataset_id="HUST",
        train=("target-a", "target-b"),
        validation=("target-validation",),
        calibration=("target-calibration",),
        test=("target-test",),
    )


def _results() -> tuple[ToolResult, ...]:
    return (
        _feature_result(
            dataset_id="MATR",
            cell_id="source-a",
            split_version="matr-split-v1",
            data_version="matr-v1",
            capacity_drop=0.01,
            resistance_growth=0.10,
        ),
        _feature_result(
            dataset_id="MATR",
            cell_id="source-b",
            split_version="matr-split-v1",
            data_version="matr-v1",
            capacity_drop=0.03,
            resistance_growth=0.30,
        ),
        _feature_result(
            dataset_id="HUST",
            cell_id="target-a",
            split_version="hust-split-v1",
            data_version="hust-v1",
            capacity_drop=0.10,
            resistance_growth=0.80,
        ),
        _feature_result(
            dataset_id="HUST",
            cell_id="target-b",
            split_version="hust-split-v1",
            data_version="hust-v1",
            capacity_drop=0.30,
            resistance_growth=1.20,
        ),
    )


def _cohort(results: tuple[ToolResult, ...]) -> VerifiedAdaptationCohort:
    from quanxin_life.tools.target_domain_adaptation import VerifiedAdaptationCohort

    return VerifiedAdaptationCohort(
        adaptation_cohort_id="matr-to-hust-train-v1",
        source_feature_result_ids=tuple(result.result_id for result in results[:2]),
        target_feature_result_ids=tuple(result.result_id for result in results[2:]),
        source_split_manifest=_source_split(),
        target_split_manifest=_target_split(),
        adapter_version="coral-v1",
        feature_names=("capacity_drop", "resistance_growth"),
    )


class _CohortResolver:
    def __init__(self, cohort: VerifiedAdaptationCohort) -> None:
        self.cohort = cohort

    def resolve_verified_adaptation_cohort(
        self, adaptation_cohort_id: str
    ) -> VerifiedAdaptationCohort:
        if adaptation_cohort_id != self.cohort.adaptation_cohort_id:
            raise ValueError("trusted adaptation cohort was not found")
        return self.cohort


def test_target_domain_adaptation_aligns_only_registered_train_feature_evidence() -> None:
    from quanxin_life.tools.target_domain_adaptation import (
        ADAPTATION_EVIDENCE_TYPE,
        AdaptToTargetDomainToolInput,
        execute_adapt_to_target_domain_tool,
    )

    results = _results()
    result = execute_adapt_to_target_domain_tool(
        AdaptToTargetDomainToolInput(adaptation_cohort_id="matr-to-hust-train-v1"),
        audit_ledger=AuditLedger(results),
        resolver=_CohortResolver(_cohort(results)),
        clock=lambda: datetime(2026, 7, 15, 12, tzinfo=UTC),
    )

    assert result.tool_name == StandardToolName.ADAPT_TO_TARGET_DOMAIN.value
    assert result.tool_version == "target-domain-adaptation-tool-v1"
    assert result.values["artifact_type"] == ADAPTATION_EVIDENCE_TYPE
    artifact = result.values["artifact"]
    assert artifact["source_dataset_id"] == "MATR"
    assert artifact["target_dataset_id"] == "HUST"
    assert artifact["source_feature_result_ids"] == [item.result_id for item in results[:2]]
    assert artifact["target_feature_result_ids"] == [item.result_id for item in results[2:]]
    aligned = artifact["adapted_source_features"]
    assert artifact["adapted_source_features_sha256"] == sha256_canonical(aligned)
    assert sum(row["capacity_drop"] for row in aligned.values()) / len(aligned) == pytest.approx(
        0.20
    )
    resistance_mean = sum(row["resistance_growth"] for row in aligned.values()) / len(aligned)
    assert resistance_mean == pytest.approx(1.0)
    assert result.uncertainty is None
    assert result.created_at == datetime(2026, 7, 15, 12, tzinfo=UTC)


def test_target_domain_adaptation_rejects_caller_features_and_target_test_leakage() -> None:
    from quanxin_life.tools.target_domain_adaptation import (
        AdaptToTargetDomainToolInput,
        execute_adapt_to_target_domain_tool,
    )

    with pytest.raises(ValueError, match="Extra inputs"):
        AdaptToTargetDomainToolInput.model_validate(
            {
                "adaptation_cohort_id": "matr-to-hust-train-v1",
                "source_features": {"source-a": {"capacity_drop": 1.0}},
                "target_features": {"target-test": {"capacity_drop": 2.0}},
            }
        )

    results = _results()
    target_test_result = _feature_result(
        dataset_id="HUST",
        cell_id="target-test",
        split_version="hust-split-v1",
        data_version="hust-v1",
        capacity_drop=0.40,
        resistance_growth=1.40,
    )
    unsafe_cohort = _cohort(results).model_copy(
        update={
            "target_feature_result_ids": (
                results[2].result_id,
                target_test_result.result_id,
            )
        }
    )
    with pytest.raises(ValueError, match=r"split_manifest\.train"):
        execute_adapt_to_target_domain_tool(
            AdaptToTargetDomainToolInput(adaptation_cohort_id=unsafe_cohort.adaptation_cohort_id),
            audit_ledger=AuditLedger((*results, target_test_result)),
            resolver=_CohortResolver(unsafe_cohort),
        )


def test_target_domain_adaptation_rejects_unregistered_feature_evidence() -> None:
    from quanxin_life.tools.target_domain_adaptation import (
        AdaptToTargetDomainToolInput,
        execute_adapt_to_target_domain_tool,
    )

    results = _results()
    with pytest.raises(ValueError, match="not registered"):
        execute_adapt_to_target_domain_tool(
            AdaptToTargetDomainToolInput(adaptation_cohort_id="matr-to-hust-train-v1"),
            audit_ledger=AuditLedger(results[:-1]),
            resolver=_CohortResolver(_cohort(results)),
        )


def test_target_domain_adaptation_rejects_cross_domain_cutoff_mismatch() -> None:
    from quanxin_life.tools.target_domain_adaptation import (
        AdaptToTargetDomainToolInput,
        execute_adapt_to_target_domain_tool,
    )

    results = _results()
    incompatible_target = _feature_result(
        dataset_id="HUST",
        cell_id="target-b",
        split_version="hust-split-v1",
        data_version="hust-v1",
        capacity_drop=0.30,
        resistance_growth=1.20,
        cutoff_cycle=50,
    )
    cohort = _cohort(results).model_copy(
        update={
            "target_feature_result_ids": (results[2].result_id, incompatible_target.result_id)
        }
    )

    with pytest.raises(ValueError, match="cutoff_cycle"):
        execute_adapt_to_target_domain_tool(
            AdaptToTargetDomainToolInput(adaptation_cohort_id=cohort.adaptation_cohort_id),
            audit_ledger=AuditLedger((results[0], results[1], results[2], incompatible_target)),
            resolver=_CohortResolver(cohort),
        )


def test_adaptation_cohort_rejects_shared_source_and_target_train_cell_ids() -> None:
    from quanxin_life.tools.target_domain_adaptation import VerifiedAdaptationCohort

    results = _results()
    with pytest.raises(ValueError, match="cell_id"):
        VerifiedAdaptationCohort(
            adaptation_cohort_id="unsafe-shared-cell-v1",
            source_feature_result_ids=(results[0].result_id, results[1].result_id),
            target_feature_result_ids=(results[2].result_id, results[3].result_id),
            source_split_manifest=_source_split(),
            target_split_manifest=SplitManifest(
                dataset_id="HUST",
                train=("source-a", "target-b"),
                validation=("target-validation",),
                calibration=("target-calibration",),
                test=("target-test",),
            ),
            adapter_version="coral-v1",
            feature_names=("capacity_drop", "resistance_growth"),
        )
