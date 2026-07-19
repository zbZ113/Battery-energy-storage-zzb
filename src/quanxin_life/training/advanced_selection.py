"""Validation-only staged selection for advanced MATR model candidates."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Literal, cast

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.training.advanced_config import (
    AdvancedCandidate,
    AdvancedFamily,
    AdvancedSelectionEvidence,
    AdvancedSelectionManifest,
)

_CUTOFFS = (20, 50, 100, 150)
_SEEDS = (38, 39, 40)
_RECHECK_GRID = frozenset((cutoff, seed) for cutoff in _CUTOFFS for seed in _SEEDS)


class AdvancedValidationEvidence(ContractModel):
    """One validation-only result; held-out metric fields are structurally forbidden."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: AdvancedFamily
    candidate_id: str = Field(min_length=1)
    candidate_config_sha256: Sha256
    stage: Literal["selection_stage1", "selection_stage2", "selection_recheck"]
    cutoff_cycle: Literal[20, 50, 100, 150]
    seed: Literal[38, 39, 40]
    status: Literal["completed", "failed"]
    validation_mae: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    validation_rmse: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    monotonic_violation_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )

    @model_validator(mode="after")
    def result_is_complete_or_failed(self) -> AdvancedValidationEvidence:
        metrics = (self.validation_mae, self.validation_rmse)
        if self.status == "completed" and any(metric is None for metric in metrics):
            raise ValueError("completed validation evidence requires MAE and RMSE")
        if self.status == "failed" and any(metric is not None for metric in metrics):
            raise ValueError("failed validation evidence cannot contain numeric metrics")
        if self.status == "failed" and self.monotonic_violation_rate is not None:
            raise ValueError("failed validation evidence cannot contain monotonic metrics")
        return self


class BaselineValidationEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: Literal["xgboost", "current_hybrid"]
    cutoff_cycle: Literal[20, 50, 100, 150]
    seed: Literal[38, 39, 40]
    validation_mae: float = Field(ge=0.0, allow_inf_nan=False)
    validation_rmse: float = Field(ge=0.0, allow_inf_nan=False)
    monotonic_violation_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )


class ProvisionalFinalEligibility(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    family: Literal["cyclepatch_batlinet", "hybridpatch_v2"]
    candidate_id: str = Field(min_length=1)
    baseline: Literal["xgboost", "current_hybrid"]
    eligible_for_final: bool
    reason_codes: tuple[str, ...]


class AdvancedSelectionTrace(ContractModel):
    """Complete validation evidence and derived decisions for every selection stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage1_evidence: tuple[AdvancedValidationEvidence, ...] = Field(min_length=1)
    stage1_survivors: dict[AdvancedFamily, tuple[str, ...]]
    stage2_evidence: tuple[AdvancedValidationEvidence, ...] = Field(min_length=1)
    stage2_finalists: dict[AdvancedFamily, tuple[str, ...]]
    recheck_evidence: tuple[AdvancedValidationEvidence, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def decisions_follow_recorded_evidence(self) -> AdvancedSelectionTrace:
        expected_survivors = select_stage1_candidates(self.stage1_evidence)
        if self.stage1_survivors != expected_survivors:
            raise ValueError("stage1 survivors do not match stage1 evidence")
        _validate_stage2_derivation(
            self.stage1_evidence,
            self.stage1_survivors,
            self.stage2_evidence,
        )
        expected_finalists = select_stage2_finalists(self.stage2_evidence)
        if self.stage2_finalists != expected_finalists:
            raise ValueError("stage2 finalists do not match stage2 evidence")
        if any(item.stage != "selection_recheck" for item in self.recheck_evidence):
            raise ValueError("recheck trace requires only selection_recheck evidence")
        return self


class AdvancedModelSelectionManifest(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-model-selection-v1"] = "advanced-model-selection-v1"
    selection: AdvancedSelectionManifest
    selection_trace: AdvancedSelectionTrace
    baseline_evidence: tuple[BaselineValidationEvidence, ...] = Field(min_length=24, max_length=24)
    provisional_eligibility: tuple[ProvisionalFinalEligibility, ProvisionalFinalEligibility]
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def manifest_hash_is_valid(self) -> AdvancedModelSelectionManifest:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if sha256_canonical(payload) != self.manifest_sha256:
            raise ValueError("manifest_sha256 does not match model selection contents")
        if {item.family for item in self.provisional_eligibility} != {
            "cyclepatch_batlinet",
            "hybridpatch_v2",
        }:
            raise ValueError("model selection requires both provisional eligibility decisions")
        _validate_baselines(self.baseline_evidence)
        return self


def select_stage1_candidates(
    evidence: tuple[AdvancedValidationEvidence, ...],
) -> dict[AdvancedFamily, tuple[str, ...]]:
    """Eliminate failed candidates, then keep ceil(50%) per family."""

    return _rank_stage(evidence, expected_stage="selection_stage1", keep="half")


def select_stage2_finalists(
    evidence: tuple[AdvancedValidationEvidence, ...],
) -> dict[AdvancedFamily, tuple[str, ...]]:
    """Keep the best two completed candidates per family."""

    return _rank_stage(evidence, expected_stage="selection_stage2", keep="two")


def _rank_stage(
    evidence: tuple[AdvancedValidationEvidence, ...],
    *,
    expected_stage: Literal["selection_stage1", "selection_stage2"],
    keep: Literal["half", "two"],
) -> dict[AdvancedFamily, tuple[str, ...]]:
    if not evidence or any(item.stage != expected_stage for item in evidence):
        raise ValueError(f"{expected_stage} requires only matching validation evidence")
    keys = [(item.family, item.candidate_id) for item in evidence]
    if len(keys) != len(set(keys)):
        raise ValueError("selection stage candidate evidence must be unique")
    grouped: dict[AdvancedFamily, list[AdvancedValidationEvidence]] = defaultdict(list)
    for item in evidence:
        if item.status == "completed":
            grouped[item.family].append(item)
    result: dict[AdvancedFamily, tuple[str, ...]] = {}
    for family, rows in grouped.items():
        rows.sort(key=lambda item: (cast(float, item.validation_mae), item.candidate_id))
        count = math.ceil(len(rows) / 2) if keep == "half" else min(2, len(rows))
        result[family] = tuple(item.candidate_id for item in rows[:count])
    return result


def build_advanced_model_selection_manifest(
    *,
    stage1_evidence: tuple[AdvancedValidationEvidence, ...],
    stage2_evidence: tuple[AdvancedValidationEvidence, ...],
    finalists: tuple[AdvancedCandidate, ...],
    recheck_evidence: tuple[AdvancedValidationEvidence, ...],
    baselines: tuple[BaselineValidationEvidence, ...],
    input_bundle_sha256: str,
    data_sha256: str,
    split_sha256: str,
    feature_sha256: str,
    normalization_sha256: str,
    source_commit: str,
    validation_cell_sha256: str,
) -> AdvancedModelSelectionManifest:
    """Select one candidate per family using a complete validation recheck grid."""

    stage1_survivors = select_stage1_candidates(stage1_evidence)
    _validate_stage2_derivation(stage1_evidence, stage1_survivors, stage2_evidence)
    stage2_finalists = select_stage2_finalists(stage2_evidence)
    if not finalists or any(item.stage != "selection_recheck" for item in recheck_evidence):
        raise ValueError("recheck selection requires selection_recheck evidence")
    by_candidate = {candidate.candidate_id: candidate for candidate in finalists}
    if len(by_candidate) != len(finalists):
        raise ValueError("finalist candidate identifiers must be unique")
    finalist_keys = {(candidate.family, candidate.candidate_id) for candidate in finalists}
    expected_finalist_keys = {
        (family, candidate_id)
        for family, candidate_ids in stage2_finalists.items()
        for candidate_id in candidate_ids
    }
    if finalist_keys != expected_finalist_keys:
        raise ValueError("finalists must exactly match the stage2 top candidates")
    stage2_by_key = {(row.family, row.candidate_id): row for row in stage2_evidence}
    for candidate in finalists:
        stage2_row = stage2_by_key[(candidate.family, candidate.candidate_id)]
        if stage2_row.candidate_config_sha256 != candidate.config_sha256:
            raise ValueError("finalist configuration does not match stage2 evidence")
    evidence_by_candidate: dict[str, list[AdvancedValidationEvidence]] = defaultdict(list)
    for row in recheck_evidence:
        candidate_for_row = by_candidate.get(row.candidate_id)
        if (
            candidate_for_row is None
            or row.family != candidate_for_row.family
            or row.candidate_config_sha256 != candidate_for_row.config_sha256
        ):
            raise ValueError("recheck evidence does not match a finalist configuration")
        evidence_by_candidate[row.candidate_id].append(row)
    for candidate in finalists:
        rows = evidence_by_candidate[candidate.candidate_id]
        axes = {(row.cutoff_cycle, row.seed) for row in rows}
        if len(rows) != 12 or axes != _RECHECK_GRID:
            raise ValueError("every finalist requires a unique complete 12-run recheck grid")

    grouped: dict[AdvancedFamily, list[AdvancedCandidate]] = defaultdict(list)
    for candidate in finalists:
        grouped[candidate.family].append(candidate)
    required = {
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    }
    if set(grouped) != required:
        raise ValueError("recheck finalists must cover all four advanced families")
    selected: list[AdvancedCandidate] = []
    selected_rows: list[AdvancedValidationEvidence] = []
    for family in sorted(grouped):
        candidate = min(
            grouped[family],
            key=lambda item: _recheck_rank(evidence_by_candidate[item.candidate_id]),
        )
        rows = evidence_by_candidate[candidate.candidate_id]
        if any(row.status != "completed" for row in rows):
            raise ValueError("selected candidate must have 12 successful recheck runs")
        selected.append(candidate)
        selected_rows.extend(rows)
    converted = tuple(
        AdvancedSelectionEvidence(
            family=row.family,
            candidate_id=row.candidate_id,
            candidate_config_sha256=row.candidate_config_sha256,
            cutoff_cycle=row.cutoff_cycle,
            seed=row.seed,
            validation_mae=cast(float, row.validation_mae),
        )
        for row in sorted(
            selected_rows,
            key=lambda item: (item.family, item.candidate_id, item.cutoff_cycle, item.seed),
        )
    )
    selection_payload = {
        "schema_version": "advanced-selection-manifest-v1",
        "input_bundle_sha256": input_bundle_sha256,
        "data_sha256": data_sha256,
        "split_sha256": split_sha256,
        "feature_sha256": feature_sha256,
        "normalization_sha256": normalization_sha256,
        "source_commit": source_commit,
        "validation_cell_sha256": validation_cell_sha256,
        "validation_results": [row.model_dump(mode="json") for row in converted],
        "selected_candidates": [candidate.model_dump(mode="json") for candidate in selected],
    }
    selection = AdvancedSelectionManifest.model_validate(
        {
            **selection_payload,
            "manifest_sha256": sha256_canonical(selection_payload),
        }
    )
    sorted_baselines = tuple(
        sorted(baselines, key=lambda item: (item.baseline, item.cutoff_cycle, item.seed))
    )
    baseline_groups = _validate_baselines(sorted_baselines)
    selected_by_family = {candidate.family: candidate for candidate in selected}
    selected_evidence = {
        family: evidence_by_candidate[candidate.candidate_id]
        for family, candidate in selected_by_family.items()
    }
    provisional_eligibility = (
        _batlinet_eligibility(
            selected_by_family["cyclepatch_batlinet"].candidate_id,
            selected_evidence["cyclepatch_batlinet"],
            baseline_groups["xgboost"],
        ),
        _hybrid_eligibility(
            selected_by_family["hybridpatch_v2"].candidate_id,
            selected_evidence["hybridpatch_v2"],
            baseline_groups["current_hybrid"],
        ),
    )
    selection_trace = AdvancedSelectionTrace(
        stage1_evidence=_sorted_validation_evidence(stage1_evidence),
        stage1_survivors=stage1_survivors,
        stage2_evidence=_sorted_validation_evidence(stage2_evidence),
        stage2_finalists=stage2_finalists,
        recheck_evidence=_sorted_validation_evidence(recheck_evidence),
    )
    payload = {
        "schema_version": "advanced-model-selection-v1",
        "selection": selection.model_dump(mode="json"),
        "selection_trace": selection_trace.model_dump(mode="json"),
        "baseline_evidence": [item.model_dump(mode="json") for item in sorted_baselines],
        "provisional_eligibility": [
            item.model_dump(mode="json") for item in provisional_eligibility
        ],
    }
    return AdvancedModelSelectionManifest.model_validate(
        {**payload, "manifest_sha256": sha256_canonical(payload)}
    )


def _validate_stage2_derivation(
    stage1_evidence: tuple[AdvancedValidationEvidence, ...],
    stage1_survivors: dict[AdvancedFamily, tuple[str, ...]],
    stage2_evidence: tuple[AdvancedValidationEvidence, ...],
) -> None:
    if not stage2_evidence or any(item.stage != "selection_stage2" for item in stage2_evidence):
        raise ValueError("stage2 requires only selection_stage2 evidence")
    stage1_by_key = {(row.family, row.candidate_id): row for row in stage1_evidence}
    survivor_keys = {
        (family, candidate_id)
        for family, candidate_ids in stage1_survivors.items()
        for candidate_id in candidate_ids
    }
    stage2_keys = {(row.family, row.candidate_id) for row in stage2_evidence}
    if stage2_keys != survivor_keys:
        raise ValueError("stage2 candidates must exactly match the stage1 survivors")
    for row in stage2_evidence:
        stage1_row = stage1_by_key[(row.family, row.candidate_id)]
        if row.candidate_config_sha256 != stage1_row.candidate_config_sha256:
            raise ValueError("stage2 candidate configuration differs from stage1 evidence")


def _sorted_validation_evidence(
    evidence: tuple[AdvancedValidationEvidence, ...],
) -> tuple[AdvancedValidationEvidence, ...]:
    return tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.family,
                item.candidate_id,
                item.cutoff_cycle,
                item.seed,
            ),
        )
    )


def _recheck_rank(rows: list[AdvancedValidationEvidence]) -> tuple[object, ...]:
    completed = [row for row in rows if row.status == "completed"]
    failed_count = len(rows) - len(completed)
    if not completed:
        return (failed_count, math.inf, math.inf, math.inf, rows[0].candidate_id)
    maes = [cast(float, row.validation_mae) for row in completed]
    cutoff_means = [
        statistics.fmean(
            cast(float, row.validation_mae) for row in completed if row.cutoff_cycle == cutoff
        )
        for cutoff in _CUTOFFS
        if any(row.cutoff_cycle == cutoff for row in completed)
    ]
    degradation = max(cutoff_means) - min(cutoff_means) if cutoff_means else math.inf
    stability = statistics.pstdev(maes) if len(maes) > 1 else 0.0
    return (failed_count, statistics.fmean(maes), degradation, stability, rows[0].candidate_id)


def _validate_baselines(
    baselines: tuple[BaselineValidationEvidence, ...],
) -> dict[str, list[BaselineValidationEvidence]]:
    grouped: dict[str, list[BaselineValidationEvidence]] = defaultdict(list)
    for row in baselines:
        grouped[row.baseline].append(row)
    if set(grouped) != {"xgboost", "current_hybrid"}:
        raise ValueError("eligibility checks require XGBoost and current Hybrid baselines")
    for rows in grouped.values():
        axes = {(row.cutoff_cycle, row.seed) for row in rows}
        if len(rows) != 12 or axes != _RECHECK_GRID:
            raise ValueError("every eligibility baseline requires a complete 12-run grid")
    return grouped


def _mean(rows: Iterable[object], field: str) -> float:
    return statistics.fmean(float(getattr(row, field)) for row in rows)


def _batlinet_eligibility(
    candidate_id: str,
    candidate: list[AdvancedValidationEvidence],
    baseline: list[BaselineValidationEvidence],
) -> ProvisionalFinalEligibility:
    reasons: list[str] = []
    if _mean(candidate, "validation_mae") > _mean(baseline, "validation_mae") * 0.98:
        reasons.append("MEAN_MAE_IMPROVEMENT_BELOW_2_PERCENT")
    for cutoff in _CUTOFFS:
        candidate_cutoff = [row for row in candidate if row.cutoff_cycle == cutoff]
        baseline_cutoff = [row for row in baseline if row.cutoff_cycle == cutoff]
        if (
            _mean(candidate_cutoff, "validation_mae")
            > _mean(baseline_cutoff, "validation_mae") * 1.10
        ):
            reasons.append(f"CUTOFF_{cutoff}_MAE_DEGRADATION_ABOVE_10_PERCENT")
    return ProvisionalFinalEligibility(
        family="cyclepatch_batlinet",
        candidate_id=candidate_id,
        baseline="xgboost",
        eligible_for_final=not reasons,
        reason_codes=tuple(reasons),
    )


def _hybrid_eligibility(
    candidate_id: str,
    candidate: list[AdvancedValidationEvidence],
    baseline: list[BaselineValidationEvidence],
) -> ProvisionalFinalEligibility:
    reasons: list[str] = []
    if _mean(candidate, "validation_mae") > _mean(baseline, "validation_mae") * 0.98:
        reasons.append("MEAN_MAE_IMPROVEMENT_BELOW_2_PERCENT")
    if _mean(candidate, "validation_rmse") > _mean(baseline, "validation_rmse"):
        reasons.append("RMSE_WORSE_THAN_CURRENT_HYBRID")
    if any(row.monotonic_violation_rate != 0.0 for row in candidate):
        reasons.append("MONOTONIC_VIOLATION_NONZERO")
    return ProvisionalFinalEligibility(
        family="hybridpatch_v2",
        candidate_id=candidate_id,
        baseline="current_hybrid",
        eligible_for_final=not reasons,
        reason_codes=tuple(reasons),
    )
