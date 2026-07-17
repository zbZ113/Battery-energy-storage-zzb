from pathlib import Path

from quanxin_life.training.suite import (
    MatrRunConfig,
    MatrThreeBatchRunConfig,
    build_run_matrix,
)


def test_matr_final_config_expands_four_cutoffs_five_models_five_seeds() -> None:
    config = MatrRunConfig.model_validate_json(
        Path("configs/training/matr_final.json").read_bytes()
    )
    matrix = build_run_matrix(config.suite)

    assert len(matrix) == 100
    assert {item.cutoff_cycle for item in matrix} == {20, 50, 100, 150}
    assert {item.model_name for item in matrix} == {
        "cpmlp",
        "dummy",
        "hybrid",
        "variance",
        "xgboost",
    }
    assert len({item.seed for item in matrix}) == 5
    assert next(model for model in config.suite.models if model.name == "cpmlp").max_epochs == 300
    assert next(model for model in config.suite.models if model.name == "hybrid").max_epochs == 500


def test_smoke_config_uses_one_seed_and_short_real_model_runs() -> None:
    config = MatrRunConfig.model_validate_json(
        Path("configs/training/matr_smoke.json").read_bytes()
    )
    matrix = build_run_matrix(config.suite)

    assert len(matrix) == 20
    assert config.paths.conversion_report.endswith("matr_2018_04_12_cutoff150.json")
    assert {item.seed for item in matrix} == {20260712}
    assert max(model.max_epochs for model in config.suite.models) <= 10


def test_three_batch_configs_preserve_the_full_approved_model_matrix() -> None:
    smoke = MatrThreeBatchRunConfig.model_validate_json(
        Path("configs/training/matr_three_batch_smoke.json").read_bytes()
    )
    final = MatrThreeBatchRunConfig.model_validate_json(
        Path("configs/training/matr_three_batch_final.json").read_bytes()
    )

    assert len(build_run_matrix(smoke.suite)) == 20
    assert len(build_run_matrix(final.suite)) == 100
    assert final.suite.data_version == "matr-three-batch-4b78769df9ddd562"
    assert final.suite.split_version == "matr-three-batch-cell-split-v1"
    assert final.paths.three_batch_manifest.endswith("matr_three_batch_manifest_v1.json")
