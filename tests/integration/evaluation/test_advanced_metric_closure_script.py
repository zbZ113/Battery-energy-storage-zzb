from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_advanced_metric_closure_script_writes_traceable_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "rul_predictions.parquet"
    output_dir = tmp_path / "closure"
    rows: list[dict[str, object]] = []
    actual_by_cell = {"a": 100.0, "b": 200.0, "c": 300.0, "d": 400.0}
    for family, offset in (("direct", 5.0), ("batlinet", 20.0)):
        for seed in (38, 39):
            for index, (cell_id, actual) in enumerate(actual_by_cell.items()):
                signed_offset = offset + index + (seed - 38)
                rows.append(
                    {
                        "run_id": f"{family}-c100-s{seed}",
                        "family": family,
                        "candidate_id": f"{family}-candidate",
                        "cutoff_cycle": 100,
                        "seed": seed,
                        "cell_id": cell_id,
                        "batch_date": "2017-05-12" if index < 2 else "2017-06-30",
                        "true_cycle_life": actual,
                        "predicted_cycle_life": actual + signed_offset,
                        "error_signed_cycles": signed_offset,
                        "error_abs_cycles": abs(signed_offset),
                        "absolute_percentage_error": 100.0
                        * abs(signed_offset)
                        / actual,
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
                "files": {"rul_predictions.parquet": input_sha256},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    script = Path(__file__).resolve().parents[3] / "scripts" / "analyze_advanced_predictions.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(input_path),
            str(output_dir),
            "--bootstrap-resamples",
            "100",
            "--bootstrap-seed",
            "20260712",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "rul_metric_closure.json",
        "rul_metric_closure.csv",
        "rul_model_comparisons.csv",
        "rul_group_metrics.csv",
        "rul_failure_cells.csv",
        "metric_closure_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected

    manifest = json.loads(
        (output_dir / "metric_closure_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == "advanced-metric-closure-manifest-v1"
    assert manifest["input"]["row_count"] == 16
    assert manifest["input"]["sha256"] == input_sha256
    assert manifest["source_prediction_manifest"] == {
        "path": str(source_manifest_path.resolve()),
        "sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "schema_version": "advanced-prediction-export-v1",
        "source_commit": "a" * 40,
        "data_version": "data-v1",
        "split_version": "split-v1",
        "input_bundle_sha256": "b" * 64,
        "local_reconstructed_input_bundle_sha256": "c" * 64,
        "input_bundle_hashes_match": False,
    }
    assert manifest["summary_count"] == 2
    assert manifest["comparison_count"] == 1
    assert set(manifest["outputs"]) == expected - {"metric_closure_manifest.json"}
    for name, item in manifest["outputs"].items():
        path = output_dir / name
        assert item["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert item["size_bytes"] == path.stat().st_size

    summaries = pd.read_csv(output_dir / "rul_metric_closure.csv")
    direct = summaries.loc[summaries["family"] == "direct"].iloc[0]
    batlinet = summaries.loc[summaries["family"] == "batlinet"].iloc[0]
    assert direct["mae_cycle"] < batlinet["mae_cycle"]
    assert "accuracy_at_15_percent" in summaries.columns

    comparisons = pd.read_csv(output_dir / "rul_model_comparisons.csv")
    assert comparisons.iloc[0]["mean_absolute_error_delta_cycle"] < 0

    failures = pd.read_csv(output_dir / "rul_failure_cells.csv")
    assert len(failures) == 8
    assert failures["error_rank"].min() == 1
