from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(
    directory: Path,
    *,
    schema_version: str,
    names: tuple[str, ...],
) -> None:
    payload = {
        "schema_version": schema_version,
        "source_prediction_manifest": {
            "source_commit": "a" * 40,
            "data_version": "data-v1",
            "split_version": "split-v1",
            "input_bundle_sha256": "b" * 64,
            "local_reconstructed_input_bundle_sha256": "c" * 64,
            "input_bundle_hashes_match": False,
        },
        "outputs": {
            name: {
                "size_bytes": (directory / name).stat().st_size,
                "sha256": _sha256(directory / name),
            }
            for name in names
        },
    }
    (directory / "metric_closure_manifest.json").write_text(
        json.dumps(payload) + "\n",
        encoding="utf-8",
    )


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    rul_dir = root / "rul"
    soh_dir = root / "soh"
    conformal_dir = root / "conformal"
    rul_dir.mkdir()
    soh_dir.mkdir()
    conformal_dir.mkdir()
    rul_rows = []
    soh_rows = []
    conformal_rows = []
    run_rows = []
    for cutoff in (20, 50):
        for family, candidate_id, mae, picp in (
            ("cyclepatch_direct", "direct-v1", 90.0, 0.92 if cutoff == 20 else 0.82),
            ("cyclepatch_batlinet", "bat-v1", 95.0, 0.85 if cutoff == 20 else 0.93),
        ):
            rul_rows.append(
                {
                    "family": family,
                    "candidate_id": candidate_id,
                    "cutoff_cycle": cutoff,
                    "cell_count": 27,
                    "seed_count": 2,
                    "mae_cycle": mae,
                    "rmse_cycle": mae + 20.0,
                    "mape_percent": 12.0,
                    "r2": 0.8,
                    "accuracy_at_15_percent": 60.0,
                    "absolute_error_p90_cycle": 180.0,
                    "absolute_error_p95_cycle": 200.0,
                    "absolute_error_p100_cycle": 240.0,
                }
            )
            for method, width in (("split", 350.0), ("normalized", 500.0)):
                conformal_rows.append(
                    {
                        "method": method,
                        "method_role": "baseline" if method == "split" else "candidate",
                        "family": family,
                        "candidate_id": candidate_id,
                        "cutoff_cycle": cutoff,
                        "target_coverage": 0.9,
                        "calibration_cell_count": 12,
                        "test_cell_count": 27,
                        "picp": picp if method == "split" else picp - 0.05,
                        "mpiw_cycle": width,
                        "warnings": "SMALL_CALIBRATION_COHORT",
                    }
                )
            for seed in (38, 39):
                run_rows.append(
                    {
                        "family": family,
                        "candidate_id": candidate_id,
                        "cutoff_cycle": cutoff,
                        "seed": seed,
                        "status": "EARLY_STOPPED",
                        "best_epoch": 10 + seed,
                        "best_validation_metric": 1.0 + (seed - 38),
                        "training_time_seconds": 100.0,
                        "peak_gpu_memory_mib": 1000.0,
                    }
                )
        for family, candidate_id, mae, rmse, p90, seconds, memory in (
            ("hybridpatch_v2", "patch-v2", 0.012, 0.027, 0.030, 450.0, 6000.0),
            ("current_hybrid", "current-v1", 0.016, 0.025, 0.024, 12.0, 106.0),
        ):
            soh_rows.append(
                {
                    "family": family,
                    "candidate_id": candidate_id,
                    "cutoff_cycle": cutoff,
                    "cell_count": 25,
                    "seed_count": 2,
                    "mae_soh": mae,
                    "rmse_soh": rmse,
                    "accuracy_at_1_soh_percentage_points": 50.0,
                    "accuracy_at_2_soh_percentage_points": 80.0,
                    "accuracy_at_5_soh_percentage_points": 95.0,
                    "cell_mae_p90_soh": p90,
                    "cell_mae_p95_soh": p90 + 0.005,
                    "cell_mae_p100_soh": p90 + 0.01,
                    "monotonic_violation_rate_percent": 0.0,
                }
            )
            for seed in (38, 39):
                run_rows.append(
                    {
                        "family": family,
                        "candidate_id": candidate_id,
                        "cutoff_cycle": cutoff,
                        "seed": seed,
                        "status": "COMPLETED" if family == "current_hybrid" else "EARLY_STOPPED",
                        "best_epoch": 20 + seed,
                        "best_validation_metric": 0.1 + (seed - 38),
                        "training_time_seconds": seconds,
                        "peak_gpu_memory_mib": memory,
                    }
                )

    pd.DataFrame(rul_rows).to_csv(rul_dir / "rul_metric_closure.csv", index=False)
    pd.DataFrame(
        [
            {
                "left_family": "cyclepatch_direct",
                "left_candidate_id": "direct-v1",
                "right_family": "cyclepatch_batlinet",
                "right_candidate_id": "bat-v1",
                "cutoff_cycle": cutoff,
                "mean_absolute_error_delta_cycle": -5.0,
                "delta_ci_lower": -10.0,
                "delta_ci_upper": 2.0,
                "left_cell_win_rate_percent": 55.0,
            }
            for cutoff in (20, 50)
        ]
    ).to_csv(rul_dir / "rul_model_comparisons.csv", index=False)
    _write_manifest(
        rul_dir,
        schema_version="advanced-metric-closure-manifest-v1",
        names=("rul_metric_closure.csv", "rul_model_comparisons.csv"),
    )

    pd.DataFrame(soh_rows).to_csv(soh_dir / "soh_metric_closure.csv", index=False)
    pd.DataFrame(
        [
            {
                "left_family": "hybridpatch_v2",
                "left_candidate_id": "patch-v2",
                "right_family": "current_hybrid",
                "right_candidate_id": "current-v1",
                "cutoff_cycle": cutoff,
                "mean_cell_mae_delta_soh": -0.004,
                "delta_ci_lower": -0.006,
                "delta_ci_upper": -0.002,
                "left_cell_win_rate_percent": 72.0,
            }
            for cutoff in (20, 50)
        ]
    ).to_csv(soh_dir / "soh_model_comparisons.csv", index=False)
    _write_manifest(
        soh_dir,
        schema_version="advanced-soh-metric-closure-manifest-v1",
        names=("soh_metric_closure.csv", "soh_model_comparisons.csv"),
    )

    conformal_path = conformal_dir / "rul_conformal_summary.csv"
    pd.DataFrame(conformal_rows).to_csv(conformal_path, index=False)
    (conformal_dir / "conformal_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "advanced-rul-conformal-manifest-v1",
                "calibration_source_manifest": {
                    "source_commit": "a" * 40,
                    "data_version": "data-v1",
                    "split_version": "split-v1",
                    "input_bundle_sha256": "b" * 64,
                    "local_reconstructed_input_bundle_sha256": "c" * 64,
                    "input_bundle_hashes_match": False,
                },
                "outputs": {
                    conformal_path.name: {
                        "size_bytes": conformal_path.stat().st_size,
                        "sha256": _sha256(conformal_path),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_metrics = root / "advanced_final_run_metrics.csv"
    pd.DataFrame(run_rows).to_csv(run_metrics, index=False)
    evidence_report = root / "advanced_final_evidence_report.json"
    evidence_report.write_text(
        json.dumps(
            {
                "schema_version": "advanced-final-evidence-report-v1",
                "run_count": len(run_rows),
                "status_counts": {"COMPLETED": 4, "EARLY_STOPPED": len(run_rows) - 4},
                "evidence_counts": {
                    "training_log_jsonl": len(run_rows),
                    "metrics_epoch_csv": len(run_rows),
                    "metrics_validation_csv": len(run_rows),
                    "metrics_test_json": len(run_rows),
                    "mlflow_directory_in_export": 0,
                    "safetensors_files": 100,
                },
                "missing_counts": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return rul_dir, soh_dir, conformal_dir, run_metrics, evidence_report


def test_promotion_report_writes_routed_model_cards_and_hash_manifest(
    tmp_path: Path,
) -> None:
    rul_dir, soh_dir, conformal_dir, run_metrics, evidence_report = _write_inputs(
        tmp_path
    )
    output_dir = tmp_path / "promotion"
    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_advanced_model_promotion_report.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(rul_dir),
            str(soh_dir),
            str(conformal_dir),
            str(run_metrics),
            str(evidence_report),
            str(output_dir),
            "--expected-cutoffs",
            "20,50",
            "--expected-seeds",
            "38,39",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected_root = {
        "promotion_report.md",
        "promotion_report.json",
        "promotion_decisions.csv",
        "refusal_policy.json",
        "source_evidence.json",
        "artifact_manifest.json",
        "model_cards",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_root
    expected_cards = {
        "cyclepatch_direct.md",
        "cyclepatch_batlinet.md",
        "current_hybrid.md",
        "hybridpatch_v2.md",
    }
    assert {path.name for path in (output_dir / "model_cards").iterdir()} == expected_cards

    decisions = pd.read_csv(output_dir / "promotion_decisions.csv")
    assert len(decisions) == 7
    assert set(decisions["disposition"]) == {"CONDITIONAL"}
    rul_20 = decisions.loc[
        (decisions["task"] == "RUL") & (decisions["cutoff_cycle"] == 20)
    ]
    assert list(rul_20[["family", "role"]].itertuples(index=False, name=None)) == [
        ("cyclepatch_direct", "DEFAULT")
    ]
    rul_50 = decisions.loc[
        (decisions["task"] == "RUL") & (decisions["cutoff_cycle"] == 50)
    ]
    assert set(rul_50[["family", "role"]].itertuples(index=False, name=None)) == {
        ("cyclepatch_direct", "POINT_ACCURACY"),
        ("cyclepatch_batlinet", "COVERAGE"),
    }
    soh = decisions.loc[decisions["task"] == "SOH"]
    assert set(soh["role"]) == {"MEAN_ACCURACY", "TAIL_EFFICIENCY"}
    assert set(decisions["representative_seed"]) == {38}
    assert set(decisions["representative_seed_rule"]) == {
        "MINIMUM_BEST_VALIDATION_METRIC"
    }

    report = json.loads((output_dir / "promotion_report.json").read_text("utf-8"))
    assert report["schema_version"] == "advanced-model-promotion-report-v1"
    assert report["activation_status"] == "NOT_ACTIVATED"
    assert report["unconditional_champion_declared"] is False
    assert report["inference_latency_evidence_available"] is False
    assert report["mlflow_export_delivered"] is False
    assert report["source_provenance"]["input_bundle_hashes_match"] is False
    assert "TEST_SPLIT_CONSUMED_FOR_ONE_TIME_PROMOTION" in report["warnings"]

    refusal = json.loads((output_dir / "refusal_policy.json").read_text("utf-8"))
    reason_codes = {item["reason_code"] for item in refusal["conditions"]}
    assert {
        "MODEL_ARTIFACT_SHA256_MISMATCH",
        "UNSUPPORTED_CUTOFF",
        "TARGET_DOMAIN_UNCALIBRATED",
        "SOH_HORIZON_EXCEEDS_CYCLE_500",
        "FORMAL_TOOL_RESULT_CONTEXT_MISSING",
        "PROMOTION_TEST_EVIDENCE_REUSED_FOR_RETUNING",
    } <= reason_codes

    manifest = json.loads((output_dir / "artifact_manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == "advanced-promotion-artifact-manifest-v1"
    assert len(manifest["files"]) == 9
    for item in manifest["files"]:
        path = output_dir / Path(item["relative_path"])
        assert path.stat().st_size == item["size_bytes"]
        assert _sha256(path) == item["sha256"]
