import pytest

from quanxin_life.evaluation.multi_seed import (
    FailedSeedRun,
    PairedObservation,
    SeedMetric,
    aggregate_seed_metrics,
    paired_bootstrap,
)


def test_aggregate_uses_seed_metrics_and_preserves_failed_seed() -> None:
    cohort = "a" * 64
    aggregate = aggregate_seed_metrics(
        (
            SeedMetric(run_id="r1", seed=1, metric_name="mae", value=1.0, cohort_sha256=cohort),
            SeedMetric(run_id="r2", seed=2, metric_name="mae", value=3.0, cohort_sha256=cohort),
        ),
        failed_runs=(
            FailedSeedRun(
                run_id="r3",
                seed=3,
                failure_stage="test",
                error_code="OOM",
                last_epoch=10,
                context_sha256="b" * 64,
            ),
        ),
        expected_seeds=(1, 2, 3),
    )

    assert aggregate.mean == 2.0
    assert aggregate.sample_std == pytest.approx(2**0.5)
    assert aggregate.valid_seed_count == 2
    assert aggregate.failed_seed_count == 1
    assert aggregate.complete_seed_matrix is False
    assert aggregate.failed_runs[0].error_code == "OOM"


def test_aggregate_rejects_different_cell_cohorts_and_missing_seed_evidence() -> None:
    with pytest.raises(ValueError, match="cohort"):
        aggregate_seed_metrics(
            (
                SeedMetric(
                    run_id="r1", seed=1, metric_name="mae", value=1.0,
                    cohort_sha256="a" * 64,
                ),
                SeedMetric(
                    run_id="r2", seed=2, metric_name="mae", value=2.0,
                    cohort_sha256="b" * 64,
                ),
            ),
            expected_seeds=(1, 2),
        )
    with pytest.raises(ValueError, match="evidence"):
        aggregate_seed_metrics(
            (SeedMetric(
                run_id="r1", seed=1, metric_name="mae", value=1.0,
                cohort_sha256="a" * 64,
            ),),
            expected_seeds=(1, 2),
        )


def test_bootstrap_requires_exact_seed_cell_pairs_and_is_factual() -> None:
    result = paired_bootstrap(
        tuple(
            PairedObservation(seed=seed, cell_id=cell, left_value=value, right_value=value)
            for seed in (1, 2)
            for cell, value in (("c1", 1.0), ("c2", 2.0))
        ),
        resamples=200,
        random_seed=7,
    )
    assert result.effect == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0
    assert result.conclusion == "NO_DEMONSTRATED_SIGNIFICANT_ADVANTAGE"


def test_bootstrap_clusters_by_cell_instead_of_treating_seeds_as_new_cells() -> None:
    one_seed = tuple(
        PairedObservation(seed=1, cell_id=cell, left_value=value, right_value=0.0)
        for cell, value in (("c1", 1.0), ("c2", 3.0))
    )
    five_seeds = tuple(
        PairedObservation(seed=seed, cell_id=cell, left_value=value, right_value=0.0)
        for seed in (1, 2, 3, 4, 5)
        for cell, value in (("c1", 1.0), ("c2", 3.0))
    )
    first = paired_bootstrap(one_seed, resamples=200, random_seed=4)
    second = paired_bootstrap(five_seeds, resamples=200, random_seed=4)
    assert second.effect == first.effect == 2.0
    assert second.lower == first.lower
    assert second.upper == first.upper
