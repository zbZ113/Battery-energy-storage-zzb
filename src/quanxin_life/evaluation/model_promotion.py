"""Deterministic promotion recommendations from already verified model evidence."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel


class _PromotionEvidenceProtocol(Protocol):
    family: str
    candidate_id: str
    cutoff_cycle: int


class RulPromotionEvidence(ContractModel):
    """Minimal point-error and Split Conformal evidence for one RUL candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    mae_cycle: float = Field(ge=0, allow_inf_nan=False)
    p90_absolute_error_cycle: float = Field(ge=0, allow_inf_nan=False)
    split_picp: float = Field(ge=0, le=1, allow_inf_nan=False)
    split_mpiw_cycle: float = Field(ge=0, allow_inf_nan=False)


class SohPromotionEvidence(ContractModel):
    """Mean, tail, monotonicity and resource evidence for one SOH candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    mae_soh: float = Field(ge=0, allow_inf_nan=False)
    rmse_soh: float = Field(ge=0, allow_inf_nan=False)
    p90_cell_mae_soh: float = Field(ge=0, allow_inf_nan=False)
    monotonic_violation_rate_percent: float = Field(
        ge=0,
        le=100,
        allow_inf_nan=False,
    )
    training_time_seconds: float = Field(ge=0, allow_inf_nan=False)
    peak_gpu_memory_mib: float = Field(ge=0, allow_inf_nan=False)


class ModelPromotionRecommendation(ContractModel):
    """A non-activating recommendation that still requires manual approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: Literal["RUL", "SOH"]
    family: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    cutoff_cycle: int = Field(ge=0)
    role: Literal[
        "DEFAULT",
        "POINT_ACCURACY",
        "COVERAGE",
        "MEAN_ACCURACY",
        "TAIL_EFFICIENCY",
    ]
    disposition: Literal["CONDITIONAL"] = "CONDITIONAL"
    reason_codes: tuple[str, ...] = Field(min_length=1)
    warnings: tuple[str, ...] = ()


class PromotionGateEvidence(ContractModel):
    """Boolean evidence ledger for the final, non-activating promotion gate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    complete: bool
    leakage_free: bool
    validation_only_selection: bool
    five_seed_complete: bool
    test_evaluated_once: bool
    per_cell_metrics: bool
    trajectory_metrics: bool
    calibration_coverage: bool
    ood_boundaries: bool
    safety_artifacts: bool
    license_verified: bool
    manual_approval: bool = False

    @model_validator(mode="after")
    def all_required_evidence_is_present(self) -> PromotionGateEvidence:
        required = (
            "complete",
            "leakage_free",
            "validation_only_selection",
            "five_seed_complete",
            "test_evaluated_once",
            "per_cell_metrics",
            "trajectory_metrics",
            "calibration_coverage",
            "ood_boundaries",
            "safety_artifacts",
            "license_verified",
        )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"promotion evidence is incomplete: {', '.join(missing)}")
        return self


class PromotionGateResult(ContractModel):
    """A verified result that remains inactive until a separate product action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_status: Literal["VERIFIED"] = "VERIFIED"
    activation_status: Literal["VERIFIED_NOT_ACTIVATED"] = "VERIFIED_NOT_ACTIVATED"
    manual_approval_recorded: bool = False
    steps: tuple[str, ...] = (
        "COMPLETE_EVIDENCE",
        "NO_LEAKAGE",
        "VALIDATION_ONLY_SELECTION",
        "FIVE_SEEDS_COMPLETE",
        "TEST_EVALUATED_ONCE",
        "CELL_AND_TRAJECTORY_EVIDENCE",
        "CALIBRATION_COVERAGE",
        "OOD_BOUNDARIES",
        "SAFETY_ARTIFACTS",
        "LICENSE_VERIFIED",
    )


def verify_promotion_gate(evidence: PromotionGateEvidence) -> PromotionGateResult:
    """Validate promotion evidence and return an explicitly inactive result."""

    checked = PromotionGateEvidence.model_validate(evidence.model_dump())
    steps = PromotionGateResult.model_fields["steps"].default
    if checked.manual_approval:
        steps = (*steps, "MANUAL_APPROVAL_RECORDED")
    return PromotionGateResult(
        manual_approval_recorded=checked.manual_approval,
        steps=steps,
    )


def recommend_rul_routes(
    candidates: Sequence[RulPromotionEvidence],
    *,
    target_coverage: float,
) -> tuple[ModelPromotionRecommendation, ...]:
    """Choose a default using a declared Split coverage gate, then point error."""

    cohort = tuple(candidates)
    _validate_candidate_cohort(cohort)
    if not math.isfinite(target_coverage) or not 0 < target_coverage < 1:
        raise ValueError("target coverage must be finite and in (0, 1)")
    point_candidate = min(
        cohort,
        key=lambda item: (
            item.mae_cycle,
            item.p90_absolute_error_cycle,
            item.split_mpiw_cycle,
            item.family,
            item.candidate_id,
        ),
    )
    coverage_eligible = tuple(
        candidate for candidate in cohort if candidate.split_picp >= target_coverage
    )
    if coverage_eligible:
        coverage_candidate = min(
            coverage_eligible,
            key=lambda item: (
                item.split_mpiw_cycle,
                item.mae_cycle,
                item.p90_absolute_error_cycle,
                item.family,
                item.candidate_id,
            ),
        )
    else:
        coverage_candidate = min(
            cohort,
            key=lambda item: (
                -item.split_picp,
                item.split_mpiw_cycle,
                item.mae_cycle,
                item.p90_absolute_error_cycle,
                item.family,
                item.candidate_id,
            ),
        )
    common_warnings = (
        *(("NO_RUL_CANDIDATE_REACHED_TARGET_COVERAGE",) if not coverage_eligible else ()),
        "MODEL_ACTIVATION_REQUIRES_MANUAL_APPROVAL",
    )
    if point_candidate == coverage_candidate:
        return (
            ModelPromotionRecommendation(
                task="RUL",
                family=point_candidate.family,
                candidate_id=point_candidate.candidate_id,
                cutoff_cycle=point_candidate.cutoff_cycle,
                role="DEFAULT",
                reason_codes=(
                    "RUL_POINT_AND_COVERAGE_OBJECTIVES_ALIGNED",
                    (
                        "RUL_TARGET_COVERAGE_GATE_PASSED"
                        if coverage_eligible
                        else "RUL_COVERAGE_GATE_DEGRADED"
                    ),
                ),
                warnings=common_warnings,
            ),
        )
    return (
        ModelPromotionRecommendation(
            task="RUL",
            family=point_candidate.family,
            candidate_id=point_candidate.candidate_id,
            cutoff_cycle=point_candidate.cutoff_cycle,
            role="POINT_ACCURACY",
            reason_codes=("RUL_LOWEST_MEAN_ABSOLUTE_ERROR",),
            warnings=common_warnings,
        ),
        ModelPromotionRecommendation(
            task="RUL",
            family=coverage_candidate.family,
            candidate_id=coverage_candidate.candidate_id,
            cutoff_cycle=coverage_candidate.cutoff_cycle,
            role="COVERAGE",
            reason_codes=(
                (
                    "RUL_TARGET_COVERAGE_GATE_PASSED"
                    if coverage_eligible
                    else "RUL_HIGHEST_AVAILABLE_EMPIRICAL_COVERAGE"
                ),
                "RUL_INTERVAL_WIDTH_TIE_BREAK",
            ),
            warnings=common_warnings,
        ),
    )


def recommend_soh_routes(
    candidates: Sequence[SohPromotionEvidence],
) -> tuple[ModelPromotionRecommendation, ...]:
    """Retain dual routes whenever mean and conservative tail evidence conflict."""

    cohort = tuple(candidates)
    _validate_candidate_cohort(cohort)
    minimum_violation = min(item.monotonic_violation_rate_percent for item in cohort)
    monotonic_pool = tuple(
        item for item in cohort if item.monotonic_violation_rate_percent == minimum_violation
    )
    accuracy = min(
        monotonic_pool,
        key=lambda item: (
            item.mae_soh,
            item.rmse_soh,
            item.p90_cell_mae_soh,
            item.family,
            item.candidate_id,
        ),
    )
    conservative = min(
        monotonic_pool,
        key=lambda item: (
            item.rmse_soh,
            item.p90_cell_mae_soh,
            item.training_time_seconds,
            item.peak_gpu_memory_mib,
            item.family,
            item.candidate_id,
        ),
    )
    common_warnings = (
        "MODEL_ACTIVATION_REQUIRES_MANUAL_APPROVAL",
        "SOH_ROUTE_VALID_ONLY_FOR_DECLARED_MATR_EVALUATION",
    )
    if accuracy == conservative and _dominates_error_metrics(accuracy, cohort):
        return (
            ModelPromotionRecommendation(
                task="SOH",
                family=accuracy.family,
                candidate_id=accuracy.candidate_id,
                cutoff_cycle=accuracy.cutoff_cycle,
                role="DEFAULT",
                reason_codes=("SOH_SINGLE_MODEL_DOMINATES_ERROR_METRICS",),
                warnings=common_warnings,
            ),
        )
    return (
        ModelPromotionRecommendation(
            task="SOH",
            family=accuracy.family,
            candidate_id=accuracy.candidate_id,
            cutoff_cycle=accuracy.cutoff_cycle,
            role="MEAN_ACCURACY",
            reason_codes=("SOH_DUAL_ROUTE_REQUIRED", "SOH_LOWEST_MEAN_ABSOLUTE_ERROR"),
            warnings=(*common_warnings, "SOH_TAIL_RISK_REQUIRES_ROUTE_POLICY"),
        ),
        ModelPromotionRecommendation(
            task="SOH",
            family=conservative.family,
            candidate_id=conservative.candidate_id,
            cutoff_cycle=conservative.cutoff_cycle,
            role="TAIL_EFFICIENCY",
            reason_codes=("SOH_DUAL_ROUTE_REQUIRED", "SOH_LOWER_RMSE_AND_TAIL_RISK"),
            warnings=common_warnings,
        ),
    )


def _validate_candidate_cohort(
    candidates: Sequence[_PromotionEvidenceProtocol],
) -> None:
    if len(candidates) < 2:
        raise ValueError("promotion requires at least two candidates")
    cutoffs = {candidate.cutoff_cycle for candidate in candidates}
    if len(cutoffs) != 1:
        raise ValueError("promotion candidates must share one cutoff_cycle")
    coordinates = {
        (candidate.family, candidate.candidate_id)
        for candidate in candidates
    }
    if len(coordinates) != len(candidates):
        raise ValueError("promotion candidates must have unique family/candidate coordinates")


def _dominates_error_metrics(
    candidate: SohPromotionEvidence,
    cohort: Sequence[SohPromotionEvidence],
) -> bool:
    return all(
        candidate.mae_soh <= other.mae_soh
        and candidate.rmse_soh <= other.rmse_soh
        and candidate.p90_cell_mae_soh <= other.p90_cell_mae_soh
        for other in cohort
    )


__all__ = [
    "ModelPromotionRecommendation",
    "PromotionGateEvidence",
    "PromotionGateResult",
    "RulPromotionEvidence",
    "SohPromotionEvidence",
    "recommend_rul_routes",
    "recommend_soh_routes",
    "verify_promotion_gate",
]
