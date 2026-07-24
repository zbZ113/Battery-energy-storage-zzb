"""Close advanced RUL metrics from verified per-cell prediction exports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.evaluation import (  # noqa: E402
    RulPredictionPoint,
    compare_rul_models,
    summarize_rul_predictions,
)

_REQUIRED_COLUMNS = frozenset(
    {
        "run_id",
        "family",
        "candidate_id",
        "cutoff_cycle",
        "seed",
        "cell_id",
        "batch_date",
        "true_cycle_life",
        "predicted_cycle_life",
        "error_abs_cycles",
        "absolute_percentage_error",
    }
)
_SUMMARY_NAME = "rul_metric_closure.csv"
_SUMMARY_JSON_NAME = "rul_metric_closure.json"
_COMPARISON_NAME = "rul_model_comparisons.csv"
_GROUP_NAME = "rul_group_metrics.csv"
_FAILURE_NAME = "rul_failure_cells.csv"
_MANIFEST_NAME = "metric_closure_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_712)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    manifest = build_metric_closure(
        args.predictions,
        args.output_dir,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
        source_manifest_path=args.source_manifest,
    )
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


def build_metric_closure(
    predictions_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
    source_manifest_path: Path | None = None,
) -> dict[str, object]:
    input_path = predictions_path.resolve(strict=True)
    input_sha256 = _sha256_file(input_path)
    source_manifest = _load_source_prediction_manifest(
        source_manifest_path or input_path.with_name("prediction_export_manifest.json"),
        prediction_name=input_path.name,
        prediction_sha256=input_sha256,
    )
    frame = pd.read_parquet(input_path)
    _validate_frame(frame)
    frame = frame.copy()
    frame["life_quartile"] = _life_quartiles(frame)

    cohort_points: dict[tuple[str, str, int], tuple[RulPredictionPoint, ...]] = {}
    summaries = []
    for key, cohort_frame in frame.groupby(
        ["family", "candidate_id", "cutoff_cycle"],
        sort=True,
        dropna=False,
    ):
        family, candidate_id, cutoff_cycle = key
        points = _points(cohort_frame)
        cohort_key = (str(family), str(candidate_id), int(cutoff_cycle))
        cohort_points[cohort_key] = points
        summaries.append(
            summarize_rul_predictions(
                points,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        )

    comparisons = []
    cutoffs = sorted({key[2] for key in cohort_points})
    for cutoff in cutoffs:
        keys = sorted(
            (key for key in cohort_points if key[2] == cutoff),
            key=_comparison_sort_key,
        )
        for left_key, right_key in itertools.combinations(keys, 2):
            comparisons.append(
                compare_rul_models(
                    cohort_points[left_key],
                    cohort_points[right_key],
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    confidence_level=confidence_level,
                )
            )

    summary_rows = [_summary_row(summary) for summary in summaries]
    comparison_rows = [_comparison_row(item) for item in comparisons]
    group_rows = _group_metric_rows(frame)
    failure_rows = _failure_rows(frame)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "schema_version": "advanced-rul-metric-closure-v1",
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "sampling_unit": "cell_id",
            "seed_aggregation": "mean_of_seed_metrics",
        },
        "summaries": [item.model_dump(mode="json") for item in summaries],
        "comparisons": [item.model_dump(mode="json") for item in comparisons],
        "group_metrics": group_rows,
        "failure_cells": failure_rows,
    }
    _write_json(output_dir / _SUMMARY_JSON_NAME, json_payload)
    _write_csv(output_dir / _SUMMARY_NAME, summary_rows)
    _write_csv(output_dir / _COMPARISON_NAME, comparison_rows)
    _write_csv(output_dir / _GROUP_NAME, group_rows)
    _write_csv(output_dir / _FAILURE_NAME, failure_rows)

    output_names = (
        _SUMMARY_JSON_NAME,
        _SUMMARY_NAME,
        _COMPARISON_NAME,
        _GROUP_NAME,
        _FAILURE_NAME,
    )
    outputs = {
        name: {
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": _sha256_file(output_dir / name),
        }
        for name in output_names
    }
    manifest: dict[str, object] = {
        "schema_version": "advanced-metric-closure-manifest-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "input": {
            "path": str(input_path),
            "row_count": len(frame),
            "size_bytes": input_path.stat().st_size,
            "sha256": input_sha256,
        },
        "source_prediction_manifest": source_manifest,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence_level,
        "summary_count": len(summary_rows),
        "comparison_count": len(comparison_rows),
        "group_metric_count": len(group_rows),
        "failure_cell_count": len(failure_rows),
        "outputs": outputs,
    }
    _write_json(output_dir / _MANIFEST_NAME, manifest)
    return manifest


def _load_source_prediction_manifest(
    path: Path,
    *,
    prediction_name: str,
    prediction_sha256: str,
) -> dict[str, object]:
    manifest_path = path.resolve(strict=True)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prediction export manifest must be valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("prediction export manifest must contain a JSON object")
    required = (
        "schema_version",
        "source_commit",
        "data_version",
        "split_version",
        "input_bundle_sha256",
        "local_reconstructed_input_bundle_sha256",
        "input_bundle_hashes_match",
        "files",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"prediction export manifest is missing fields: {missing}")
    if payload["schema_version"] != "advanced-prediction-export-v1":
        raise ValueError("prediction export manifest schema_version is unsupported")
    files = payload["files"]
    if not isinstance(files, dict) or files.get(prediction_name) != prediction_sha256:
        raise ValueError("prediction parquet SHA-256 does not match its export manifest")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": payload["schema_version"],
        "source_commit": payload["source_commit"],
        "data_version": payload["data_version"],
        "split_version": payload["split_version"],
        "input_bundle_sha256": payload["input_bundle_sha256"],
        "local_reconstructed_input_bundle_sha256": payload[
            "local_reconstructed_input_bundle_sha256"
        ],
        "input_bundle_hashes_match": payload["input_bundle_hashes_match"],
    }


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"RUL prediction export is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("RUL prediction export must not be empty")
    if frame[list(_REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("RUL prediction export contains null required values")
    duplicate = frame.duplicated(
        subset=["family", "candidate_id", "cutoff_cycle", "seed", "cell_id"]
    )
    if duplicate.any():
        raise ValueError("RUL prediction export contains duplicate model/seed/cell rows")
    true_counts = frame.groupby("cell_id")["true_cycle_life"].nunique()
    if (true_counts != 1).any():
        raise ValueError("RUL prediction export changes true cycle life across runs")
    batch_counts = frame.groupby("cell_id")["batch_date"].nunique()
    if (batch_counts != 1).any():
        raise ValueError("RUL prediction export changes batch_date across runs")


def _points(frame: pd.DataFrame) -> tuple[RulPredictionPoint, ...]:
    return tuple(
        RulPredictionPoint(
            run_id=str(row.run_id),
            family=str(row.family),
            candidate_id=str(row.candidate_id),
            cutoff_cycle=int(row.cutoff_cycle),
            seed=int(row.seed),
            cell_id=str(row.cell_id),
            batch_date=str(row.batch_date),
            true_cycle_life=float(row.true_cycle_life),
            predicted_cycle_life=float(row.predicted_cycle_life),
        )
        for row in frame.itertuples(index=False)
    )


def _life_quartiles(frame: pd.DataFrame) -> pd.Series:
    cell_life = (
        frame[["cell_id", "true_cycle_life"]]
        .drop_duplicates()
        .sort_values(["true_cycle_life", "cell_id"])
        .reset_index(drop=True)
    )
    labels = ("Q1_short", "Q2", "Q3", "Q4_long")
    cell_life["life_quartile"] = pd.qcut(
        cell_life.index.to_series().rank(method="first"),
        q=4,
        labels=labels,
    ).astype(str)
    mapping = cell_life.set_index("cell_id")["life_quartile"]
    result = frame["cell_id"].map(mapping)
    if result.isnull().any():
        raise ValueError("failed to assign a life quartile to every cell")
    return result


def _group_metric_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_keys = ["family", "candidate_id", "cutoff_cycle"]
    for group_type, column in (("batch", "batch_date"), ("life_quartile", "life_quartile")):
        grouped = frame.groupby([*base_keys, column], sort=True, dropna=False)
        for key, cohort_frame in grouped:
            family, candidate_id, cutoff_cycle, group_value = key
            summary = summarize_rul_predictions(
                _points(cohort_frame),
                bootstrap_resamples=1,
                bootstrap_seed=0,
            )
            rows.append(
                {
                    "family": str(family),
                    "candidate_id": str(candidate_id),
                    "cutoff_cycle": int(cutoff_cycle),
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "cell_count": summary.cell_count,
                    "seed_count": summary.seed_count,
                    "mae_cycle": summary.mae_cycle,
                    "rmse_cycle": summary.rmse_cycle,
                    "mape_percent": summary.mape_percent,
                    "r2": summary.r2,
                    **{
                        f"accuracy_at_{name}_percent": value
                        for name, value in summary.accuracy_at_tolerance_percent.items()
                    },
                }
            )
    return rows


def _failure_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    grouped = (
        frame.groupby(
            ["family", "candidate_id", "cutoff_cycle", "cell_id"],
            sort=True,
            as_index=False,
        )
        .agg(
            batch_date=("batch_date", "first"),
            true_cycle_life=("true_cycle_life", "first"),
            seed_count=("seed", "nunique"),
            predicted_cycle_life_mean=("predicted_cycle_life", "mean"),
            absolute_error_cycle_mean=("error_abs_cycles", "mean"),
            absolute_error_cycle_max=("error_abs_cycles", "max"),
            absolute_percentage_error_mean=("absolute_percentage_error", "mean"),
            absolute_percentage_error_max=("absolute_percentage_error", "max"),
        )
    )
    grouped["is_over_20_percent"] = grouped["absolute_percentage_error_mean"] > 20.0
    grouped["error_rank"] = (
        grouped.groupby(["family", "candidate_id", "cutoff_cycle"])[
            "absolute_percentage_error_mean"
        ]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return grouped.to_dict(orient="records")


def _summary_row(summary: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "family": summary.family,
        "candidate_id": summary.candidate_id,
        "cutoff_cycle": summary.cutoff_cycle,
        "cell_count": summary.cell_count,
        "seed_count": summary.seed_count,
        "mae_cycle": summary.mae_cycle,
        "rmse_cycle": summary.rmse_cycle,
        "mape_percent": summary.mape_percent,
        "r2": summary.r2,
    }
    row.update(
        {
            f"accuracy_at_{name}_percent": value
            for name, value in summary.accuracy_at_tolerance_percent.items()
        }
    )
    row.update(
        {
            f"absolute_error_p{name}_cycle": value
            for name, value in summary.absolute_error_quantiles_cycle.items()
        }
    )
    for name, interval in summary.bootstrap_intervals.items():
        row[f"{name}_ci_lower"] = interval.lower
        row[f"{name}_ci_upper"] = interval.upper
    return row


def _comparison_row(comparison: Any) -> dict[str, object]:
    return {
        "left_family": comparison.left_family,
        "left_candidate_id": comparison.left_candidate_id,
        "right_family": comparison.right_family,
        "right_candidate_id": comparison.right_candidate_id,
        "cutoff_cycle": comparison.cutoff_cycle,
        "cell_count": comparison.cell_count,
        "seed_count": comparison.seed_count,
        "mean_absolute_error_delta_cycle": comparison.mean_absolute_error_delta_cycle,
        "delta_ci_lower": comparison.bootstrap_interval.lower,
        "delta_ci_upper": comparison.bootstrap_interval.upper,
        "left_cell_win_rate_percent": comparison.left_cell_win_rate_percent,
    }


def _comparison_sort_key(key: tuple[str, str, int]) -> tuple[int, str, str]:
    family, candidate_id, _ = key
    if "direct" in family:
        priority = 0
    elif "batlinet" in family:
        priority = 1
    else:
        priority = 2
    return priority, family, candidate_id


def _write_json(path: Path, payload: object) -> None:
    data = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _write_atomic(path, data)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to publish an empty metric closure file: {path.name}")
    data = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_atomic(path, data)


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
