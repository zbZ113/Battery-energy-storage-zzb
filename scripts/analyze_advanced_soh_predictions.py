"""Close advanced SOH trajectory metrics from verified prediction exports."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
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
    SohPredictionPoint,
    compare_soh_models,
    summarize_soh_predictions,
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
        "cycle",
        "is_model_prediction",
        "true_soh",
        "predicted_soh",
        "error_signed_soh",
        "error_abs_soh",
    }
)
_SUMMARY_NAME = "soh_metric_closure.csv"
_SUMMARY_JSON_NAME = "soh_metric_closure.json"
_COMPARISON_NAME = "soh_model_comparisons.csv"
_GROUP_NAME = "soh_group_metrics.csv"
_FAILURE_NAME = "soh_failure_cells.csv"
_MANIFEST_NAME = "metric_closure_manifest.json"
_HORIZON_BINS = (0, 50, 100, 200, 350, math.inf)
_HORIZON_LABELS = ("1-50", "51-100", "101-200", "201-350", "351+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_712)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()
    manifest = build_soh_metric_closure(
        args.predictions,
        args.output_dir,
        source_manifest_path=args.source_manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
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


def build_soh_metric_closure(
    predictions_path: Path,
    output_dir: Path,
    *,
    source_manifest_path: Path | None,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, object]:
    input_path = predictions_path.resolve(strict=True)
    input_sha256 = _sha256_file(input_path)
    source_manifest = _load_source_prediction_manifest(
        source_manifest_path or input_path.with_name("prediction_export_manifest.json"),
        prediction_name=input_path.name,
        prediction_sha256=input_sha256,
    )
    raw_frame = pd.read_parquet(input_path)
    _validate_frame(raw_frame)
    frame = raw_frame.loc[raw_frame["is_model_prediction"].astype(bool)].copy()
    if frame.empty:
        raise ValueError("SOH prediction export contains no model forecast rows")
    frame["forecast_horizon"] = frame["cycle"] - frame["cutoff_cycle"]
    if (frame["forecast_horizon"] <= 0).any():
        raise ValueError("SOH model forecast rows must occur after cutoff_cycle")
    frame["forecast_horizon_group"] = pd.cut(
        frame["forecast_horizon"],
        bins=_HORIZON_BINS,
        labels=_HORIZON_LABELS,
        include_lowest=True,
    ).astype(str)

    summaries = []
    comparisons = []
    for cutoff in sorted(int(value) for value in frame["cutoff_cycle"].unique()):
        cutoff_frame = frame.loc[frame["cutoff_cycle"] == cutoff]
        cohort_points: dict[tuple[str, str, int], tuple[SohPredictionPoint, ...]] = {}
        for key, cohort_frame in cutoff_frame.groupby(
            ["family", "candidate_id", "cutoff_cycle"],
            sort=True,
            dropna=False,
        ):
            family, candidate_id, cutoff_cycle = key
            points = _points(cohort_frame)
            cohort_points[(str(family), str(candidate_id), int(cutoff_cycle))] = points
            summaries.append(
                summarize_soh_predictions(
                    points,
                    bootstrap_resamples=bootstrap_resamples,
                    bootstrap_seed=bootstrap_seed,
                    confidence_level=confidence_level,
                )
            )
        keys = sorted(cohort_points, key=_comparison_sort_key)
        for left_key, right_key in itertools.combinations(keys, 2):
            comparisons.append(
                compare_soh_models(
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
    payload = {
        "schema_version": "advanced-soh-metric-closure-v1",
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
    _write_json(output_dir / _SUMMARY_JSON_NAME, payload)
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
        "schema_version": "advanced-soh-metric-closure-manifest-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "input": {
            "path": str(input_path),
            "row_count": len(raw_frame),
            "forecast_row_count": len(frame),
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


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"SOH prediction export is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("SOH prediction export must not be empty")
    forecast = frame.loc[frame["is_model_prediction"].astype(bool)]
    required_forecast = [
        "run_id",
        "family",
        "candidate_id",
        "cutoff_cycle",
        "seed",
        "cell_id",
        "batch_date",
        "cycle",
        "true_soh",
        "predicted_soh",
        "error_signed_soh",
        "error_abs_soh",
    ]
    if forecast[required_forecast].isnull().any().any():
        raise ValueError("SOH model forecast rows contain null required values")
    duplicate = forecast.duplicated(
        subset=["family", "candidate_id", "cutoff_cycle", "seed", "cell_id", "cycle"]
    )
    if duplicate.any():
        raise ValueError("SOH export contains duplicate model/seed/cell/cycle rows")
    actual_counts = forecast.groupby(["cell_id", "cycle"])["true_soh"].nunique()
    if (actual_counts != 1).any():
        raise ValueError("SOH export changes true SOH across runs")
    batch_counts = forecast.groupby("cell_id")["batch_date"].nunique()
    if (batch_counts != 1).any():
        raise ValueError("SOH export changes batch_date across runs")


def _points(frame: pd.DataFrame) -> tuple[SohPredictionPoint, ...]:
    return tuple(
        SohPredictionPoint(
            run_id=str(row.run_id),
            family=str(row.family),
            candidate_id=str(row.candidate_id),
            cutoff_cycle=int(row.cutoff_cycle),
            seed=int(row.seed),
            cell_id=str(row.cell_id),
            batch_date=str(row.batch_date),
            cycle=int(row.cycle),
            true_soh=float(row.true_soh),
            predicted_soh=float(row.predicted_soh),
        )
        for row in frame.itertuples(index=False)
    )


def _group_metric_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    base_keys = ["family", "candidate_id", "cutoff_cycle"]
    for group_type, column in (
        ("batch", "batch_date"),
        ("forecast_horizon", "forecast_horizon_group"),
    ):
        for key, group_frame in frame.groupby([*base_keys, column], sort=True, observed=True):
            family, candidate_id, cutoff_cycle, group_value = key
            seed_rows = []
            for _seed, seed_frame in group_frame.groupby("seed", sort=True):
                errors = seed_frame["predicted_soh"] - seed_frame["true_soh"]
                absolute_errors = errors.abs()
                seed_rows.append(
                    {
                        "mae_soh": float(absolute_errors.mean()),
                        "rmse_soh": float((errors.pow(2).mean()) ** 0.5),
                        "mean_signed_error_soh": float(errors.mean()),
                        "accuracy_at_1_soh_percentage_points": float(
                            (absolute_errors * 100.0 <= 1.0 + 1e-9).mean() * 100.0
                        ),
                        "accuracy_at_2_soh_percentage_points": float(
                            (absolute_errors * 100.0 <= 2.0 + 1e-9).mean() * 100.0
                        ),
                        "accuracy_at_5_soh_percentage_points": float(
                            (absolute_errors * 100.0 <= 5.0 + 1e-9).mean() * 100.0
                        ),
                    }
                )
            aggregated = pd.DataFrame(seed_rows).mean().to_dict()
            rows.append(
                {
                    "family": str(family),
                    "candidate_id": str(candidate_id),
                    "cutoff_cycle": int(cutoff_cycle),
                    "group_type": group_type,
                    "group_value": str(group_value),
                    "cell_count": int(group_frame["cell_id"].nunique()),
                    "seed_count": int(group_frame["seed"].nunique()),
                    "forecast_point_count": len(group_frame),
                    **{name: float(value) for name, value in aggregated.items()},
                }
            )
    return rows


def _failure_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    enriched = frame.copy()
    enriched["squared_error_soh"] = enriched["error_signed_soh"].pow(2)
    keys = ["family", "candidate_id", "cutoff_cycle", "cell_id"]
    grouped = (
        enriched.groupby(keys, sort=True, as_index=False)
        .agg(
            batch_date=("batch_date", "first"),
            seed_count=("seed", "nunique"),
            forecast_point_count=("cycle", "size"),
            forecast_cycle_min=("cycle", "min"),
            forecast_cycle_max=("cycle", "max"),
            mae_soh=("error_abs_soh", "mean"),
            squared_error_soh_mean=("squared_error_soh", "mean"),
            max_absolute_error_soh=("error_abs_soh", "max"),
            mean_signed_error_soh=("error_signed_soh", "mean"),
        )
    )
    grouped["rmse_soh"] = grouped["squared_error_soh_mean"].pow(0.5)
    final_rows = enriched.loc[
        enriched["cycle"]
        == enriched.groupby([*keys, "seed"])["cycle"].transform("max")
    ]
    final_error = (
        final_rows.groupby(keys, sort=True, as_index=False)
        .agg(final_cycle_absolute_error_soh=("error_abs_soh", "mean"))
    )
    grouped = grouped.merge(final_error, on=keys, how="left", validate="one_to_one")
    grouped["has_point_error_over_5pp"] = grouped["max_absolute_error_soh"] > 0.05
    grouped["error_rank"] = (
        grouped.groupby(["family", "candidate_id", "cutoff_cycle"])["mae_soh"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return grouped.drop(columns=["squared_error_soh_mean"]).to_dict(orient="records")


def _summary_row(summary: Any) -> dict[str, object]:
    row: dict[str, object] = {
        "family": summary.family,
        "candidate_id": summary.candidate_id,
        "cutoff_cycle": summary.cutoff_cycle,
        "cell_count": summary.cell_count,
        "seed_count": summary.seed_count,
        "forecast_point_count_per_seed": summary.forecast_point_count_per_seed,
        "forecast_horizon_cycle_min": summary.forecast_horizon_cycle_min,
        "forecast_horizon_cycle_max": summary.forecast_horizon_cycle_max,
        "mae_soh": summary.mae_soh,
        "rmse_soh": summary.rmse_soh,
        "mean_signed_error_soh": summary.mean_signed_error_soh,
        "monotonic_violation_rate_percent": summary.monotonic_violation_rate_percent,
    }
    row.update(
        {
            f"accuracy_at_{name}_soh_percentage_points": value
            for name, value in summary.accuracy_at_soh_point_tolerance_percent.items()
        }
    )
    row.update(
        {
            f"cell_mae_p{name}_soh": value
            for name, value in summary.cell_mae_quantiles_soh.items()
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
        "comparison_point_count": comparison.comparison_point_count,
        "left_only_point_count": comparison.left_only_point_count,
        "right_only_point_count": comparison.right_only_point_count,
        "mean_cell_mae_delta_soh": comparison.mean_cell_mae_delta_soh,
        "delta_ci_lower": comparison.bootstrap_interval.lower,
        "delta_ci_upper": comparison.bootstrap_interval.upper,
        "left_cell_win_rate_percent": comparison.left_cell_win_rate_percent,
    }


def _comparison_sort_key(key: tuple[str, str, int]) -> tuple[int, str, str]:
    family, candidate_id, _ = key
    if "patch" in family:
        priority = 0
    elif "current" in family:
        priority = 1
    else:
        priority = 2
    return priority, family, candidate_id


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
        raise ValueError(f"refusing to publish an empty SOH closure file: {path.name}")
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
