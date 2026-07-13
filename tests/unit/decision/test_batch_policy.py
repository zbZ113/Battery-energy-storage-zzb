from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quanxin_life.core import (
    ConformalCalibration,
    PredictionInterval,
)
from quanxin_life.data.schemas import DataQualityIssue, DataQualityReport, DataQualitySeverity


def _interval(*, lower: float, point: float, upper: float) -> PredictionInterval:
    calibration = ConformalCalibration(
        alpha=0.10,
        residual_quantile_cycle=25.0,
        calibration_cell_count=4,
        feature_version="early-cycle-v1",
        split_version="matr-split-v1",
        model_version="xgboost-v1",
        data_version="matr-v1",
    )
    return PredictionInterval(
        dataset_id="MATR",
        cell_id="cell-a",
        cutoff_cycle=100,
        point_prediction_cycle=point,
        lower_eol_cycle=lower,
        upper_eol_cycle=upper,
        calibration=calibration,
    )


def _quality(*, blocking: bool = False) -> DataQualityReport:
    issues: tuple[DataQualityIssue, ...] = ()
    if blocking:
        issues = (
            DataQualityIssue(
                code="REQUIRED_FIELD_MISSING",
                severity=DataQualitySeverity.BLOCKING,
                message="Required diagnostic capacity is unavailable",
                cell_id="cell-a",
            ),
        )
    return DataQualityReport(dataset_id="MATR", issues=issues)


@pytest.mark.parametrize(
    ("interval", "target_domain_calibrated", "expected_decision", "reason_code"),
    [
        (_interval(lower=340.0, point=380.0, upper=420.0), True, "ADMIT", "LOWER_BOUND_PASSES"),
        (
            _interval(lower=180.0, point=220.0, upper=260.0),
            True,
            "DOWNGRADE",
            "UPPER_BOUND_BELOW_REQUIREMENT",
        ),
        (
            _interval(lower=260.0, point=300.0, upper=340.0),
            True,
            "RECHECK",
            "INTERVAL_CROSSES_REQUIREMENT",
        ),
        (
            _interval(lower=340.0, point=380.0, upper=420.0),
            False,
            "RECHECK",
            "TARGET_DOMAIN_UNCALIBRATED",
        ),
    ],
)
def test_batch_decision_uses_conformal_interval_and_explicit_domain_status(
    interval: PredictionInterval,
    target_domain_calibrated: bool,
    expected_decision: str,
    reason_code: str,
) -> None:
    from quanxin_life.decision.batch_policy import BatchDecisionPolicy, make_batch_decision

    decision = make_batch_decision(
        prediction_interval=interval,
        quality_report=_quality(),
        target_domain_calibrated=target_domain_calibrated,
        policy=BatchDecisionPolicy(
            policy_version="batch-policy-v1",
            required_eol_cycle=300.0,
        ),
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert decision.decision.value == expected_decision
    assert reason_code in decision.reason_codes
    assert decision.interval_lower_eol_cycle == interval.lower_eol_cycle
    assert decision.interval_upper_eol_cycle == interval.upper_eol_cycle


def test_blocking_data_quality_rejects_without_overriding_the_evidence_chain() -> None:
    from quanxin_life.core import Decision
    from quanxin_life.decision.batch_policy import BatchDecisionPolicy, make_batch_decision

    outcome = make_batch_decision(
        prediction_interval=_interval(lower=340.0, point=380.0, upper=420.0),
        quality_report=_quality(blocking=True),
        target_domain_calibrated=True,
        policy=BatchDecisionPolicy(
            policy_version="batch-policy-v1",
            required_eol_cycle=300.0,
        ),
        decided_at=datetime(2026, 7, 13, tzinfo=UTC),
    )

    assert outcome.decision is Decision.REJECT
    assert outcome.reason_codes == ("DATA_QUALITY_BLOCKING",)


def test_batch_decision_rejects_dataset_mismatch_and_implicit_domain_status() -> None:
    from quanxin_life.decision.batch_policy import BatchDecisionPolicy, make_batch_decision

    with pytest.raises(ValueError, match="dataset_id"):
        make_batch_decision(
            prediction_interval=_interval(lower=340.0, point=380.0, upper=420.0),
            quality_report=DataQualityReport(dataset_id="HUST"),
            target_domain_calibrated=True,
            policy=BatchDecisionPolicy(
                policy_version="batch-policy-v1",
                required_eol_cycle=300.0,
            ),
            decided_at=datetime(2026, 7, 13, tzinfo=UTC),
        )

    with pytest.raises(TypeError, match="bool"):
        make_batch_decision(
            prediction_interval=_interval(lower=340.0, point=380.0, upper=420.0),
            quality_report=_quality(),
            target_domain_calibrated=None,  # type: ignore[arg-type]
            policy=BatchDecisionPolicy(
                policy_version="batch-policy-v1",
                required_eol_cycle=300.0,
            ),
            decided_at=datetime(2026, 7, 13, tzinfo=UTC),
        )
