from __future__ import annotations

import pytest

from deploy.advanced_agent_context import resolve_advanced_context_reference
from deploy.competition_inputs import AdvancedAgentRuntimePolicy
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask


@pytest.mark.parametrize(
    ("cutoff_cycle", "reference", "expected"),
    (
        (20, "context.rul_point_route_role", AdvancedModelRouteRole.DEFAULT),
        (20, "context.rul_coverage_route_role", AdvancedModelRouteRole.DEFAULT),
        (
            100,
            "context.rul_point_route_role",
            AdvancedModelRouteRole.POINT_ACCURACY,
        ),
        (
            100,
            "context.rul_coverage_route_role",
            AdvancedModelRouteRole.COVERAGE,
        ),
        (
            100,
            "context.soh_route_role",
            AdvancedModelRouteRole.MEAN_ACCURACY,
        ),
        (100, "context.rul_task", AdvancedModelTask.RUL),
        (100, "context.soh_task", AdvancedModelTask.SOH),
        (100, "context.conformal_calibrate_operation", "calibrate"),
        (100, "context.conformal_issue_operation", "issue"),
        (100, "context.conformal_alpha", 0.1),
    ),
)
def test_server_context_references_are_derived_from_policy_and_cutoff(
    cutoff_cycle: int,
    reference: str,
    expected: object,
) -> None:
    policy = AdvancedAgentRuntimePolicy(
        schema_version="advanced-agent-policy-v1",
        conformal_alpha=0.1,
    )

    assert (
        resolve_advanced_context_reference(
            reference,
            cutoff_cycle=cutoff_cycle,
            policy=policy,
        )
        == expected
    )


def test_server_context_references_reject_unknown_or_unsupported_cutoff() -> None:
    policy = AdvancedAgentRuntimePolicy(
        schema_version="advanced-agent-policy-v1",
        conformal_alpha=0.1,
    )

    with pytest.raises(KeyError):
        resolve_advanced_context_reference(
            "context.caller_supplied_value",
            cutoff_cycle=100,
            policy=policy,
        )
    with pytest.raises(ValueError, match="cutoff"):
        resolve_advanced_context_reference(
            "context.rul_task",
            cutoff_cycle=75,
            policy=policy,
        )
