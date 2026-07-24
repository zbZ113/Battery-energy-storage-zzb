from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_rows(
    *,
    partition: str,
    actual_by_cell: dict[str, float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    batch_dates = ("2017-05-12", "2017-06-30")
    for family, base_offset in (("direct", 4.0), ("batlinet", 12.0)):
        for seed in (38, 39):
            for index, (cell_id, actual) in enumerate(actual_by_cell.items()):
                signed_offset = base_offset + index * 3.0 + (seed - 38) * 2.0
                if partition == "test" and index == len(actual_by_cell) - 1:
                    signed_offset += 20.0
                rows.append(
                    {
                        "run_id": f"{family}-c20-s{seed}",
                        "family": family,
                        "candidate_id": f"{family}-candidate",
                        "cutoff_cycle": 20,
                        "seed": seed,
                        "cell_id": cell_id,
                        "batch_date": batch_dates[index % 2],
                        "true_cycle_life": actual,
                        "predicted_cycle_life": actual + signed_offset,
                        "error_signed_cycles": signed_offset,
                        "error_abs_cycles": abs(signed_offset),
                        "absolute_percentage_error": abs(signed_offset) / actual * 100.0,
                    }
                )
    return rows


def _write_manifest(
    path: Path,
    *,
    schema_version: str,
    prediction_path: Path,
    partition: str | None,
    cell_ids: tuple[str, ...],
) -> None:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "source_commit": "a" * 40,
        "data_version": "data-v1",
        "split_version": "split-v1",
        "config_sha256": "d" * 64,
        "input_bundle_sha256": "b" * 64,
        "local_reconstructed_input_bundle_sha256": "c" * 64,
        "input_bundle_hashes_match": False,
        "files": {prediction_path.name: _sha256(prediction_path)},
    }
    if partition is not None:
        payload.update(
            {
                "partition": partition,
                "run_count": 4,
                "row_count": 16,
                "calibration_cell_count": len(cell_ids),
                "calibration_cell_ids": list(cell_ids),
            }
        )
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_advanced_conformal_script_writes_traceable_interval_outputs(
    tmp_path: Path,
) -> None:
    calibration_cells = ("c1", "c2", "c3", "c4")
    test_cells = ("t1", "t2", "t3", "t4")
    calibration_path = tmp_path / "calibration_predictions.parquet"
    test_path = tmp_path / "test_predictions.parquet"
    pd.DataFrame(
        _prediction_rows(
            partition="calibration",
            actual_by_cell=dict.fromkeys(calibration_cells, 200.0),
        )
    ).to_parquet(calibration_path, index=False)
    pd.DataFrame(
        _prediction_rows(
            partition="test",
            actual_by_cell={"t1": 100.0, "t2": 200.0, "t3": 300.0, "t4": 400.0},
        )
    ).to_parquet(test_path, index=False)

    calibration_manifest = tmp_path / "calibration_prediction_export_manifest.json"
    test_manifest = tmp_path / "prediction_export_manifest.json"
    _write_manifest(
        calibration_manifest,
        schema_version="advanced-calibration-prediction-export-v1",
        prediction_path=calibration_path,
        partition="calibration",
        cell_ids=calibration_cells,
    )
    _write_manifest(
        test_manifest,
        schema_version="advanced-prediction-export-v1",
        prediction_path=test_path,
        partition=None,
        cell_ids=test_cells,
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "dataset_id": "MATR",
                "train": ["train-1"],
                "validation": ["validation-1"],
                "calibration": list(calibration_cells),
                "test": list(test_cells),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "conformal"
    script = Path(__file__).resolve().parents[3] / "scripts" / "analyze_advanced_conformal.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(calibration_path),
            str(test_path),
            str(split_path),
            str(output_dir),
            "--calibration-manifest",
            str(calibration_manifest),
            "--test-manifest",
            str(test_manifest),
            "--expected-seeds",
            "38,39",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {
        "rul_conformal_report.json",
        "rul_conformal_summary.csv",
        "rul_conformal_group_coverage.csv",
        "rul_conformal_intervals.csv",
        "conformal_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected

    summary = pd.read_csv(output_dir / "rul_conformal_summary.csv")
    assert len(summary) == 12
    assert set(summary["method"]) == {"split", "normalized"}
    assert set(summary["method_role"]) == {"baseline", "candidate"}
    assert "is_primary_method" not in summary.columns
    assert set(summary["target_coverage"]) == {0.8, 0.9, 0.95}
    assert set(summary["calibration_cell_count"]) == {4}
    assert summary["warnings"].str.contains("SMALL_CALIBRATION_COHORT").all()

    intervals = pd.read_csv(output_dir / "rul_conformal_intervals.csv")
    assert len(intervals) == 48
    assert set(intervals["cell_id"]) == set(test_cells)
    assert intervals["lower_cycle"].le(intervals["point_prediction_cycle"]).all()
    assert intervals["upper_cycle"].ge(intervals["point_prediction_cycle"]).all()

    groups = pd.read_csv(output_dir / "rul_conformal_group_coverage.csv")
    assert set(groups["group_type"]) == {"batch", "life_quartile"}

    report = json.loads((output_dir / "rul_conformal_report.json").read_text("utf-8"))
    assert report["methods"] == {
        "baseline": "split",
        "candidate": "normalized",
        "promotion_status": "pending",
        "normalized_scale": {
            "version": "five-seed-sample-standard-deviation-v1",
            "definition": "sample standard deviation across registered seed predictions",
            "uses_labels": False,
        },
    }

    manifest = json.loads((output_dir / "conformal_manifest.json").read_text("utf-8"))
    assert manifest["schema_version"] == "advanced-rul-conformal-manifest-v1"
    assert manifest["calibration_input"]["sha256"] == _sha256(calibration_path)
    assert manifest["test_input"]["sha256"] == _sha256(test_path)
    assert manifest["scale"]["version"] == "five-seed-sample-standard-deviation-v1"
    assert manifest["scale"]["uses_labels"] is False
    assert set(manifest["outputs"]) == expected - {"conformal_manifest.json"}
    for name, evidence in manifest["outputs"].items():
        assert evidence["sha256"] == _sha256(output_dir / name)
