"""Deterministic batch triage from calibrated lifetime intervals.

This module never estimates SOH, RUL, a prediction interval, or a business
threshold.  It applies a versioned policy to an already-calibrated interval
and a data-quality report.  The future Tool Registry wrapper is responsible
for attaching the resulting outcome to a ``ToolResult`` provenance chain.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import ConfigDict, Field, field_validator

from quanxin_life.core import Decision, NormalizedPredictionInterval, PredictionInterval
from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.schemas import DataQualityReport

LifetimeInterval = PredictionInterval | NormalizedPredictionInterval


class _DecisionModel(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class BatchDecisionPolicy(_DecisionModel):
    """A human-approved, versioned EOL80 requirement for one triage scenario."""

    policy_version: str = Field(min_length=1)
    required_eol_cycle: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("policy_version")
    @classmethod
    def require_nonblank_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("policy_version must not be blank")
        return value


class BatchDecisionOutcome(_DecisionModel):
    """A deterministic, auditable triage outcome without invented evidence."""

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    decision: Decision
    reason_codes: tuple[str, ...] = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    required_eol_cycle: float = Field(gt=0, allow_inf_nan=False)
    interval_lower_eol_cycle: float = Field(ge=0, allow_inf_nan=False)
    interval_upper_eol_cycle: float = Field(ge=0, allow_inf_nan=False)
    target_domain_calibrated: bool
    quality_report_blocked: bool
    decided_at: datetime

    @field_validator("decided_at")
    @classmethod
    def normalize_decision_time_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decided_at must include a timezone")
        return value.astimezone(UTC)

    @field_validator("reason_codes")
    @classmethod
    def require_nonblank_unique_reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not code.strip() for code in value) or len(set(value)) != len(value):
            raise ValueError("reason_codes must be nonblank and unique")
        return value


def make_batch_decision(
    *,
    prediction_interval: LifetimeInterval,
    quality_report: DataQualityReport,
    target_domain_calibrated: bool,
    policy: BatchDecisionPolicy,
    decided_at: datetime,
) -> BatchDecisionOutcome:
    """Classify one cell as admit, recheck, downgrade, or reject.

    Priority is conservative: blocking quality defects reject; an uncalibrated
    target domain forces recheck; otherwise only an interval entirely above or
    below the supplied requirement is admitted or downgraded.  Any interval
    touching the requirement remains uncertain and is rechecked.
    """

    if not isinstance(target_domain_calibrated, bool):
        raise TypeError("target_domain_calibrated must be a bool")
    if quality_report.dataset_id != prediction_interval.dataset_id:
        raise ValueError("quality_report dataset_id must match prediction interval dataset_id")
    if not math.isfinite(policy.required_eol_cycle):
        raise ValueError("required_eol_cycle must be finite")

    lower = prediction_interval.lower_eol_cycle
    upper = prediction_interval.upper_eol_cycle
    if quality_report.blocked:
        return _outcome(
            prediction_interval=prediction_interval,
            policy=policy,
            decision=Decision.REJECT,
            reason_codes=("DATA_QUALITY_BLOCKING",),
            target_domain_calibrated=target_domain_calibrated,
            quality_report_blocked=True,
            decided_at=decided_at,
        )
    if not target_domain_calibrated:
        return _outcome(
            prediction_interval=prediction_interval,
            policy=policy,
            decision=Decision.RECHECK,
            reason_codes=("TARGET_DOMAIN_UNCALIBRATED",),
            target_domain_calibrated=False,
            quality_report_blocked=False,
            decided_at=decided_at,
        )
    if lower > policy.required_eol_cycle:
        decision = Decision.ADMIT
        reasons = ("LOWER_BOUND_PASSES",)
    elif upper < policy.required_eol_cycle:
        decision = Decision.DOWNGRADE
        reasons = ("UPPER_BOUND_BELOW_REQUIREMENT",)
    else:
        decision = Decision.RECHECK
        reasons = ("INTERVAL_CROSSES_REQUIREMENT",)
    return _outcome(
        prediction_interval=prediction_interval,
        policy=policy,
        decision=decision,
        reason_codes=reasons,
        target_domain_calibrated=True,
        quality_report_blocked=False,
        decided_at=decided_at,
    )


def _outcome(
    *,
    prediction_interval: LifetimeInterval,
    policy: BatchDecisionPolicy,
    decision: Decision,
    reason_codes: tuple[str, ...],
    target_domain_calibrated: bool,
    quality_report_blocked: bool,
    decided_at: datetime,
) -> BatchDecisionOutcome:
    return BatchDecisionOutcome(
        dataset_id=prediction_interval.dataset_id,
        cell_id=prediction_interval.cell_id,
        decision=decision,
        reason_codes=reason_codes,
        policy_version=policy.policy_version,
        required_eol_cycle=policy.required_eol_cycle,
        interval_lower_eol_cycle=prediction_interval.lower_eol_cycle,
        interval_upper_eol_cycle=prediction_interval.upper_eol_cycle,
        target_domain_calibrated=target_domain_calibrated,
        quality_report_blocked=quality_report_blocked,
        decided_at=decided_at,
    )
