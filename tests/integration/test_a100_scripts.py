from pathlib import Path


def test_a100_preflight_binds_only_physical_gpu1() -> None:
    script = Path("scripts/a100/preflight.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in script
    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "a100_device_guard.py --physical-index 1" in script


def test_dataset_shell_is_one_command_and_never_parallelizes_models() -> None:
    script = Path("scripts/a100/train_dataset.sh").read_text(encoding="utf-8")

    assert "CUDA_VISIBLE_DEVICES=1" in script
    assert "MLFLOW_ALLOW_FILE_STORE=true" in script
    assert "MPLBACKEND=Agg" in script
    assert "PYTHONUNBUFFERED=1" in script
    assert "unset DISPLAY" in script
    assert "scripts/prepare_matr_training_data.py" in script
    assert script.index("scripts/prepare_matr_training_data.py") < script.index(
        "scripts/a100/preflight.sh"
    )
    assert 'python scripts/run_training_suite.py "${DATASET}" "${MODE}"' in script
    training = script.rindex(
        'python scripts/run_training_suite.py "${DATASET}" "${MODE}"'
    )
    assert script.index("MLFLOW_ALLOW_FILE_STORE=true") < training
    assert script.index("MPLBACKEND=Agg") < training
    assert script.index("PYTHONUNBUFFERED=1") < training
    assert script.index("unset DISPLAY") < training
    assert "xargs -P" not in script
    assert " wait" not in script


def test_three_batch_dataset_shell_prepares_all_batches_before_training() -> None:
    script = Path("scripts/a100/train_dataset.sh").read_text(encoding="utf-8")

    assert "matr-three-batch" in script
    assert "scripts/prepare_matr_three_batch_data.py" in script
    assert 'python scripts/run_training_suite.py "${DATASET}" "${MODE}"' in script
    assert script.index("scripts/prepare_matr_three_batch_data.py") < script.index(
        "scripts/a100/preflight.sh"
    )


def test_a100_environment_is_hash_locked_and_does_not_install_cuda_toolkit() -> None:
    script = Path("scripts/a100/create_env.sh").read_text(encoding="utf-8")
    lock = Path("requirements/a100-linux-py311.lock").read_text(encoding="utf-8")

    assert "python=3.11.13" in script
    assert "torch==2.12.0" in script
    assert "--require-hashes -r requirements/a100-linux-py311.lock" in script
    assert "pip check" in script
    assert "cuda-toolkit" not in script.lower()
    assert "--hash=sha256:" in lock
    assert "xgboost==2.1.4" in lock
    assert "\ntorch==" not in lock


def test_a100_package_commands_bind_clean_git_and_verify_before_extracting() -> None:
    build = Path("scripts/build_matr_a100_package.py").read_text(encoding="utf-8")
    verify = Path("scripts/verify_matr_a100_package.py").read_text(encoding="utf-8")

    assert '"git", "status", "--porcelain"' in build
    assert '"git", "ls-files", "-z"' in build
    assert "build_matr_a100_archive_index" in build
    assert "verify_matr_a100_archive_index" in verify
    assert "verify_matr_a100_archive" in verify
    assert "extractall" not in verify


def test_three_batch_preparation_declares_each_reviewed_batch_and_real_horizon() -> None:
    script = Path("scripts/prepare_matr_three_batch_data.py").read_text(encoding="utf-8")

    assert "2017-05-12" in script
    assert "2017-06-30" in script
    assert "2018-04-12" in script
    assert "horizon_cycle=500" in script
    assert "max_cycle_index=150" in script
    assert "selected_cell_ids=eligibility.eligible_cell_ids" in script
