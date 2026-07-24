from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_soh_metric_closure_script_writes_traceable_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "soh_trajectory_predictions.parquet"
    output_dir = tmp_path / "closure"
    rows: list[dict[str, object]] = []
    for family, offset in (("patch", 0.01), ("current", 0.02)):
        for seed in (38, 39):
            for cell_index, cell_id in enumerate(("a", "b", "c", "d")):
                for cycle, true_soh in ((21, 1.0), (22, 0.98), (71, 0.9)):
                    error = offset + 0.001 * cell_index
                    rows.append(
                        {
                            "run_id": f"{family}-c20-s{seed}",
                            "family": family,
                            "candidate_id": f"{family}-candidate",
                            "cutoff_cycle": 20,
                            "seed": seed,
                            "cell_id": cell_id,
                            "batch_date": "2017-05-12" if cell_index < 2 else "2017-06-30",
                            "cycle": cycle,
                            "is_observed_history": False,
                            "is_forecast_target": True,
                            "is_model_prediction": True,
                            "observed_soh": None,
                            "true_soh": true_soh,
                            "predicted_soh": true_soh - error,
                            "error_signed_soh": -error,
                            "error_abs_soh": error,
                        }
                    )
    pd.DataFrame(rows).to_parquet(input_path, index=False)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_manifest_path = tmp_path / "prediction_export_manifest.json"
    source_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "advanced-prediction-export-v1",
                "source_commit": "a" * 40,
                "data_version": "data-v1",
                "split_version": "split-v1",
                "input_bundle_sha256": "b" * 64,
                "local_reconstructed_input_bundle_sha256": "c" * 64,
                "input_bundle_hashes_match": False,
                "files": {"soh_trajectory_predictions.parquet": input_sha256},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "analyze_advanced_soh_predictions.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(input_path),
            str(output_dir),
            "--bootstrap-resamples",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "soh_metric_closure.json",
        "soh_metric_closure.csv",
        "soh_model_comparisons.csv",
        "soh_group_metrics.csv",
        "soh_failure_cells.csv",
        "metric_closure_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    manifest = json.loads(
        (output_dir / "metric_closure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "advanced-soh-metric-closure-manifest-v1"
    assert manifest["input"]["row_count"] == 48
    assert manifest["input"]["forecast_row_count"] == 48
    assert manifest["input"]["sha256"] == input_sha256
    assert manifest["source_prediction_manifest"]["source_commit"] == "a" * 40
    assert manifest["summary_count"] == 2
    assert manifest["comparison_count"] == 1
    assert set(manifest["outputs"]) == expected - {"metric_closure_manifest.json"}

    summaries = pd.read_csv(output_dir / "soh_metric_closure.csv")
    patch = summaries.loc[summaries["family"] == "patch"].iloc[0]
    current = summaries.loc[summaries["family"] == "current"].iloc[0]
    assert patch["mae_soh"] < current["mae_soh"]
    assert "accuracy_at_2_soh_percentage_points" in summaries.columns

    comparisons = pd.read_csv(output_dir / "soh_model_comparisons.csv")
    assert comparisons.iloc[0]["mean_cell_mae_delta_soh"] < 0

    groups = pd.read_csv(output_dir / "soh_group_metrics.csv")
    assert set(groups["group_type"]) == {"batch", "forecast_horizon"}

    failures = pd.read_csv(output_dir / "soh_failure_cells.csv")
    assert len(failures) == 8
    assert failures["error_rank"].min() == 1
