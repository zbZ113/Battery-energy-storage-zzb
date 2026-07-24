from __future__ import annotations


def test_rul_route_merges_roles_when_one_candidate_wins_both_objectives() -> None:
    from quanxin_life.evaluation.model_promotion import (
        RulPromotionEvidence,
        recommend_rul_routes,
    )

    direct = RulPromotionEvidence(
        family="direct",
        candidate_id="direct-v1",
        cutoff_cycle=20,
        mae_cycle=90.0,
        p90_absolute_error_cycle=180.0,
        split_picp=0.92,
        split_mpiw_cycle=340.0,
    )
    lower_mae_but_undercovered = RulPromotionEvidence(
        family="batlinet",
        candidate_id="batlinet-v1",
        cutoff_cycle=20,
        mae_cycle=100.0,
        p90_absolute_error_cycle=170.0,
        split_picp=0.85,
        split_mpiw_cycle=320.0,
    )

    routes = recommend_rul_routes(
        (direct, lower_mae_but_undercovered),
        target_coverage=0.90,
    )

    assert len(routes) == 1
    assert routes[0].family == "direct"
    assert routes[0].role == "DEFAULT"
    assert "RUL_POINT_AND_COVERAGE_OBJECTIVES_ALIGNED" in routes[0].reason_codes


def test_rul_route_keeps_separate_point_and_coverage_candidates() -> None:
    from quanxin_life.evaluation.model_promotion import (
        RulPromotionEvidence,
        recommend_rul_routes,
    )

    routes = recommend_rul_routes(
        (
            RulPromotionEvidence(
                family="direct",
                candidate_id="direct-v1",
                cutoff_cycle=50,
                mae_cycle=90.0,
                p90_absolute_error_cycle=180.0,
                split_picp=0.82,
                split_mpiw_cycle=340.0,
            ),
            RulPromotionEvidence(
                family="batlinet",
                candidate_id="batlinet-v1",
                cutoff_cycle=50,
                mae_cycle=95.0,
                p90_absolute_error_cycle=170.0,
                split_picp=0.93,
                split_mpiw_cycle=430.0,
            ),
        ),
        target_coverage=0.90,
    )

    assert [(route.family, route.role) for route in routes] == [
        ("direct", "POINT_ACCURACY"),
        ("batlinet", "COVERAGE"),
    ]
    assert all(route.disposition == "CONDITIONAL" for route in routes)


def test_rul_route_degrades_explicitly_when_no_candidate_meets_coverage() -> None:
    from quanxin_life.evaluation.model_promotion import (
        RulPromotionEvidence,
        recommend_rul_routes,
    )

    routes = recommend_rul_routes(
        (
            RulPromotionEvidence(
                family="direct",
                candidate_id="direct-v1",
                cutoff_cycle=100,
                mae_cycle=105.0,
                p90_absolute_error_cycle=210.0,
                split_picp=0.88,
                split_mpiw_cycle=390.0,
            ),
            RulPromotionEvidence(
                family="batlinet",
                candidate_id="batlinet-v1",
                cutoff_cycle=100,
                mae_cycle=103.0,
                p90_absolute_error_cycle=205.0,
                split_picp=0.88,
                split_mpiw_cycle=400.0,
            ),
        ),
        target_coverage=0.90,
    )

    assert [(route.family, route.role) for route in routes] == [
        ("batlinet", "POINT_ACCURACY"),
        ("direct", "COVERAGE"),
    ]
    assert all(
        "NO_RUL_CANDIDATE_REACHED_TARGET_COVERAGE" in route.warnings
        for route in routes
    )


def test_soh_conflicting_mean_and_tail_metrics_produce_dual_route() -> None:
    from quanxin_life.evaluation.model_promotion import (
        SohPromotionEvidence,
        recommend_soh_routes,
    )

    routes = recommend_soh_routes(
        (
            SohPromotionEvidence(
                family="hybridpatch_v2",
                candidate_id="patch-v2",
                cutoff_cycle=20,
                mae_soh=0.012,
                rmse_soh=0.027,
                p90_cell_mae_soh=0.029,
                monotonic_violation_rate_percent=0.0,
                training_time_seconds=450.0,
                peak_gpu_memory_mib=6150.0,
            ),
            SohPromotionEvidence(
                family="current_hybrid",
                candidate_id="current-v1",
                cutoff_cycle=20,
                mae_soh=0.016,
                rmse_soh=0.025,
                p90_cell_mae_soh=0.024,
                monotonic_violation_rate_percent=0.0,
                training_time_seconds=12.0,
                peak_gpu_memory_mib=106.0,
            ),
        )
    )

    assert [(route.family, route.role) for route in routes] == [
        ("hybridpatch_v2", "MEAN_ACCURACY"),
        ("current_hybrid", "TAIL_EFFICIENCY"),
    ]
    assert all(route.disposition == "CONDITIONAL" for route in routes)
    assert all("SOH_DUAL_ROUTE_REQUIRED" in route.reason_codes for route in routes)


def test_soh_dominating_candidate_can_be_single_default() -> None:
    from quanxin_life.evaluation.model_promotion import (
        SohPromotionEvidence,
        recommend_soh_routes,
    )

    routes = recommend_soh_routes(
        (
            SohPromotionEvidence(
                family="dominant",
                candidate_id="dominant-v1",
                cutoff_cycle=50,
                mae_soh=0.010,
                rmse_soh=0.020,
                p90_cell_mae_soh=0.025,
                monotonic_violation_rate_percent=0.0,
                training_time_seconds=100.0,
                peak_gpu_memory_mib=1000.0,
            ),
            SohPromotionEvidence(
                family="other",
                candidate_id="other-v1",
                cutoff_cycle=50,
                mae_soh=0.020,
                rmse_soh=0.030,
                p90_cell_mae_soh=0.035,
                monotonic_violation_rate_percent=0.0,
                training_time_seconds=10.0,
                peak_gpu_memory_mib=100.0,
            ),
        )
    )

    assert len(routes) == 1
    assert routes[0].family == "dominant"
    assert routes[0].role == "DEFAULT"
    assert "SOH_SINGLE_MODEL_DOMINATES_ERROR_METRICS" in routes[0].reason_codes
