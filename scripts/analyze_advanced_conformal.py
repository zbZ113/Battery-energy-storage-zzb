"""Evaluate advanced RUL Split and Normalized Conformal intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.core import CycleLifePrediction, PredictionTarget  # noqa: E402
from quanxin_life.data.schemas import SplitManifest  # noqa: E402
from quanxin_life.uncertainty import (  # noqa: E402
    ScaledCycleLifePrediction,
    calibrate_cycle_life_conformal,
    calibrate_normalized_cycle_life_conformal,
    evaluate_cycle_life_interval_coverage,
    evaluate_normalized_cycle_life_interval_coverage,
    make_cycle_life_interval,
    make_normalized_cycle_life_interval,
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
    }
)
_SUMMARY_NAME = "rul_conformal_summary.csv"
_GROUP_NAME = "rul_conformal_group_coverage.csv"
_INTERVAL_NAME = "rul_conformal_intervals.csv"
_REPORT_NAME = "rul_conformal_report.json"
_MANIFEST_NAME = "conformal_manifest.json"
_SCALE_VERSION = "five-seed-sample-standard-deviation-v1"
_FEATURE_VERSION = "advanced-cyclepatch-v1"
_COVERAGE_BOUNDARY_WARNING = "COVERAGE_VALID_ONLY_FOR_DECLARED_CALIBRATION_COHORT"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_predictions", type=Path)
    parser.add_argument("test_predictions", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--expected-seeds", default="38,39,40,41,42")
    parser.add_argument(
        "--target-coverages",
        default="0.80,0.90,0.95",
        help="Comma-separated coverage targets in (0, 1).",
    )
    args = parser.parse_args()
    manifest = build_conformal_analysis(
        args.calibration_predictions,
        args.test_predictions,
        args.split_manifest,
        args.output_dir,
        calibration_manifest_path=args.calibration_manifest,
        test_manifest_path=args.test_manifest,
        expected_seeds=_parse_int_values(args.expected_seeds, name="expected seeds"),
        target_coverages=_parse_float_values(
            args.target_coverages,
            name="target coverages",
        ),
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


def build_conformal_analysis(
    calibration_predictions_path: Path,
    test_predictions_path: Path,
    split_manifest_path: Path,
    output_dir: Path,
    *,
    calibration_manifest_path: Path | None,
    test_manifest_path: Path | None,
    expected_seeds: tuple[int, ...],
    target_coverages: tuple[float, ...],
) -> dict[str, object]:
    calibration_path = calibration_predictions_path.resolve(strict=True)
    test_path = test_predictions_path.resolve(strict=True)
    split_path = split_manifest_path.resolve(strict=True)
    calibration_sha256 = _sha256_file(calibration_path)
    test_sha256 = _sha256_file(test_path)
    calibration_manifest = _load_source_manifest(
        calibration_manifest_path
        or calibration_path.with_name("calibration_prediction_export_manifest.json"),
        expected_schema="advanced-calibration-prediction-export-v1",
        prediction_name=calibration_path.name,
        prediction_sha256=calibration_sha256,
        expected_partition="calibration",
    )
    test_manifest = _load_source_manifest(
        test_manifest_path or test_path.with_name("prediction_export_manifest.json"),
        expected_schema="advanced-prediction-export-v1",
        prediction_name=test_path.name,
        prediction_sha256=test_sha256,
        expected_partition=None,
    )
    _require_shared_provenance(calibration_manifest, test_manifest)

    split_manifest = SplitManifest.model_validate_json(split_path.read_bytes())
    calibration_frame = pd.read_parquet(calibration_path)
    test_frame = pd.read_parquet(test_path)
    _validate_prediction_frame(
        calibration_frame,
        partition="calibration",
        split_manifest=split_manifest,
        expected_seeds=expected_seeds,
    )
    _validate_prediction_frame(
        test_frame,
        partition="test",
        split_manifest=split_manifest,
        expected_seeds=expected_seeds,
    )
    _validate_coordinate_match(calibration_frame, test_frame)
    test_frame = test_frame.copy()
    test_frame["life_quartile"] = _life_quartiles(test_frame)

    summary_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    coordinate_columns = ["family", "candidate_id", "cutoff_cycle"]
    for coordinate, calibration_group in calibration_frame.groupby(
        coordinate_columns,
        sort=True,
        dropna=False,
    ):
        family, candidate_id, cutoff_cycle = coordinate
        cutoff_value = int(cast(int, cutoff_cycle))
        test_group = test_frame.loc[
            (test_frame["family"] == family)
            & (test_frame["candidate_id"] == candidate_id)
            & (test_frame["cutoff_cycle"] == cutoff_cycle)
        ]
        model_version = f"{family}:{candidate_id}:seed-ensemble-v1"
        calibration_scaled = _ensemble_predictions(
            calibration_group,
            model_version=model_version,
            split_version=str(calibration_manifest["split_version"]),
            data_version=str(calibration_manifest["data_version"]),
        )
        test_scaled = _ensemble_predictions(
            test_group,
            model_version=model_version,
            split_version=str(test_manifest["split_version"]),
            data_version=str(test_manifest["data_version"]),
        )
        test_metadata = _test_metadata(test_group)
        for target_coverage in target_coverages:
            alpha = 1.0 - target_coverage
            split_calibration = calibrate_cycle_life_conformal(
                tuple(item.prediction for item in calibration_scaled),
                split_manifest=split_manifest,
                alpha=alpha,
            )
            split_intervals = tuple(
                make_cycle_life_interval(item.prediction, split_calibration)
                for item in test_scaled
            )
            split_coverage = evaluate_cycle_life_interval_coverage(
                split_intervals,
                split_manifest=split_manifest,
            )
            split_warnings = _warnings(split_coverage.warnings)
            summary_rows.append(
                {
                    "method": "split",
                    "method_role": "baseline",
                    "family": str(family),
                    "candidate_id": str(candidate_id),
                    "cutoff_cycle": cutoff_value,
                    "target": PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value,
                    "target_coverage": target_coverage,
                    "alpha": alpha,
                    "calibration_cell_count": split_coverage.calibration_cell_count,
                    "test_cell_count": split_coverage.evaluated_cell_count,
                    "calibration_quantile": split_calibration.residual_quantile_cycle,
                    "calibration_quantile_unit": "cycles",
                    "picp": split_coverage.picp,
                    "mpiw_cycle": split_coverage.mpiw_cycle,
                    "scale_version": None,
                    "warnings": split_warnings,
                }
            )
            interval_rows.extend(
                _split_interval_rows(
                    split_intervals,
                    family=str(family),
                    candidate_id=str(candidate_id),
                    target_coverage=target_coverage,
                    metadata=test_metadata,
                )
            )
            group_rows.extend(
                _split_group_rows(
                    split_intervals,
                    split_manifest=split_manifest,
                    family=str(family),
                    candidate_id=str(candidate_id),
                    target_coverage=target_coverage,
                    metadata=test_metadata,
                )
            )

            normalized_calibration = calibrate_normalized_cycle_life_conformal(
                calibration_scaled,
                split_manifest=split_manifest,
                alpha=alpha,
            )
            normalized_intervals = tuple(
                make_normalized_cycle_life_interval(item, normalized_calibration)
                for item in test_scaled
            )
            normalized_coverage = evaluate_normalized_cycle_life_interval_coverage(
                normalized_intervals,
                split_manifest=split_manifest,
            )
            normalized_warnings = _warnings(normalized_coverage.warnings)
            summary_rows.append(
                {
                    "method": "normalized",
                    "method_role": "candidate",
                    "family": str(family),
                    "candidate_id": str(candidate_id),
                    "cutoff_cycle": cutoff_value,
                    "target": PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value,
                    "target_coverage": target_coverage,
                    "alpha": alpha,
                    "calibration_cell_count": normalized_coverage.calibration_cell_count,
                    "test_cell_count": normalized_coverage.evaluated_cell_count,
                    "calibration_quantile": (
                        normalized_calibration.normalized_score_quantile
                    ),
                    "calibration_quantile_unit": "dimensionless",
                    "picp": normalized_coverage.picp,
                    "mpiw_cycle": normalized_coverage.mpiw_cycle,
                    "scale_version": _SCALE_VERSION,
                    "warnings": normalized_warnings,
                }
            )
            interval_rows.extend(
                _normalized_interval_rows(
                    normalized_intervals,
                    family=str(family),
                    candidate_id=str(candidate_id),
                    target_coverage=target_coverage,
                    metadata=test_metadata,
                )
            )
            group_rows.extend(
                _normalized_group_rows(
                    normalized_intervals,
                    split_manifest=split_manifest,
                    family=str(family),
                    candidate_id=str(candidate_id),
                    target_coverage=target_coverage,
                    metadata=test_metadata,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "advanced-rul-conformal-report-v1",
        "target": PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value,
        "methods": {
            "baseline": "split",
            "candidate": "normalized",
            "promotion_status": "pending",
            "normalized_scale": {
                "version": _SCALE_VERSION,
                "definition": "sample standard deviation across registered seed predictions",
                "uses_labels": False,
            },
        },
        "coverage_targets": list(target_coverages),
        "calibration_cell_count": len(split_manifest.calibration),
        "test_cell_count": len(split_manifest.test),
        "warnings": [
            "SMALL_CALIBRATION_COHORT",
            _COVERAGE_BOUNDARY_WARNING,
            "NO_CROSS_DOMAIN_COVERAGE_CLAIM",
        ],
        "summaries": summary_rows,
        "group_coverage": group_rows,
    }
    _write_json(output_dir / _REPORT_NAME, report)
    _write_csv(output_dir / _SUMMARY_NAME, summary_rows)
    _write_csv(output_dir / _GROUP_NAME, group_rows)
    _write_csv(output_dir / _INTERVAL_NAME, interval_rows)

    output_names = (_REPORT_NAME, _SUMMARY_NAME, _GROUP_NAME, _INTERVAL_NAME)
    outputs = {
        name: {
            "size_bytes": (output_dir / name).stat().st_size,
            "sha256": _sha256_file(output_dir / name),
        }
        for name in output_names
    }
    manifest: dict[str, object] = {
        "schema_version": "advanced-rul-conformal-manifest-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "calibration_input": _input_evidence(calibration_path, calibration_sha256),
        "test_input": _input_evidence(test_path, test_sha256),
        "split_manifest": _input_evidence(split_path, _sha256_file(split_path)),
        "calibration_source_manifest": calibration_manifest,
        "test_source_manifest": test_manifest,
        "expected_seeds": list(expected_seeds),
        "target_coverages": list(target_coverages),
        "scale": {
            "version": _SCALE_VERSION,
            "definition": "sample standard deviation across registered seed predictions",
            "uses_labels": False,
        },
        "summary_count": len(summary_rows),
        "group_coverage_count": len(group_rows),
        "interval_count": len(interval_rows),
        "outputs": outputs,
    }
    _write_json(output_dir / _MANIFEST_NAME, manifest)
    return manifest


def _ensemble_predictions(
    frame: pd.DataFrame,
    *,
    model_version: str,
    split_version: str,
    data_version: str,
) -> tuple[ScaledCycleLifePrediction, ...]:
    scaled: list[ScaledCycleLifePrediction] = []
    for cell_id, cell_frame in frame.groupby("cell_id", sort=True):
        predictions = [float(value) for value in cell_frame["predicted_cycle_life"]]
        scale = statistics.stdev(predictions)
        if not scale > 0:
            raise ValueError(
                f"model-produced seed dispersion is not positive for cell {cell_id}"
            )
        observed = float(cell_frame["true_cycle_life"].iloc[0])
        if not observed.is_integer():
            raise ValueError("MATR official cycle life must be an integer cycle")
        cutoff_cycle = int(cell_frame["cutoff_cycle"].iloc[0])
        prediction = CycleLifePrediction(
            dataset_id="MATR",
            cell_id=str(cell_id),
            cutoff_cycle=cutoff_cycle,
            target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
            predicted_cycle=statistics.fmean(predictions),
            observed_cycle=int(observed),
            right_censored=False,
            feature_version=_FEATURE_VERSION,
            split_version=split_version,
            model_version=model_version,
            data_version=data_version,
        )
        scaled.append(
            ScaledCycleLifePrediction(
                prediction=prediction,
                difficulty_scale_cycle=scale,
                scale_version=_SCALE_VERSION,
            )
        )
    return tuple(scaled)


def _test_metadata(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in frame[["cell_id", "batch_date", "life_quartile"]].drop_duplicates().itertuples(
        index=False
    ):
        cell_id = str(row.cell_id)
        if cell_id in result:
            raise ValueError("test metadata contains duplicate cell rows")
        result[cell_id] = {
            "batch_date": str(row.batch_date),
            "life_quartile": str(row.life_quartile),
        }
    return result


def _split_interval_rows(
    intervals: tuple[Any, ...],
    *,
    family: str,
    candidate_id: str,
    target_coverage: float,
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for interval in intervals:
        prediction = interval.prediction
        assert prediction.observed_cycle is not None
        rows.append(
            _interval_row(
                method="split",
                family=family,
                candidate_id=candidate_id,
                target_coverage=target_coverage,
                prediction=prediction,
                lower_cycle=interval.lower_cycle,
                upper_cycle=interval.upper_cycle,
                difficulty_scale_cycle=None,
                metadata=metadata[prediction.cell_id],
            )
        )
    return rows


def _normalized_interval_rows(
    intervals: tuple[Any, ...],
    *,
    family: str,
    candidate_id: str,
    target_coverage: float,
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for interval in intervals:
        prediction = interval.prediction
        assert prediction.observed_cycle is not None
        rows.append(
            _interval_row(
                method="normalized",
                family=family,
                candidate_id=candidate_id,
                target_coverage=target_coverage,
                prediction=prediction,
                lower_cycle=interval.lower_cycle,
                upper_cycle=interval.upper_cycle,
                difficulty_scale_cycle=interval.difficulty_scale_cycle,
                metadata=metadata[prediction.cell_id],
            )
        )
    return rows


def _interval_row(
    *,
    method: str,
    family: str,
    candidate_id: str,
    target_coverage: float,
    prediction: CycleLifePrediction,
    lower_cycle: float,
    upper_cycle: float,
    difficulty_scale_cycle: float | None,
    metadata: dict[str, str],
) -> dict[str, object]:
    assert prediction.observed_cycle is not None
    return {
        "method": method,
        "family": family,
        "candidate_id": candidate_id,
        "cutoff_cycle": prediction.cutoff_cycle,
        "target_coverage": target_coverage,
        "cell_id": prediction.cell_id,
        "batch_date": metadata["batch_date"],
        "life_quartile": metadata["life_quartile"],
        "point_prediction_cycle": prediction.predicted_cycle,
        "observed_cycle": prediction.observed_cycle,
        "lower_cycle": lower_cycle,
        "upper_cycle": upper_cycle,
        "interval_width_cycle": upper_cycle - lower_cycle,
        "covered": lower_cycle <= prediction.observed_cycle <= upper_cycle,
        "difficulty_scale_cycle": difficulty_scale_cycle,
        "scale_version": _SCALE_VERSION if method == "normalized" else None,
    }


def _split_group_rows(
    intervals: tuple[Any, ...],
    *,
    split_manifest: SplitManifest,
    family: str,
    candidate_id: str,
    target_coverage: float,
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    return _group_rows(
        method="split",
        intervals=intervals,
        split_manifest=split_manifest,
        family=family,
        candidate_id=candidate_id,
        target_coverage=target_coverage,
        metadata=metadata,
        evaluator=evaluate_cycle_life_interval_coverage,
    )


def _normalized_group_rows(
    intervals: tuple[Any, ...],
    *,
    split_manifest: SplitManifest,
    family: str,
    candidate_id: str,
    target_coverage: float,
    metadata: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    return _group_rows(
        method="normalized",
        intervals=intervals,
        split_manifest=split_manifest,
        family=family,
        candidate_id=candidate_id,
        target_coverage=target_coverage,
        metadata=metadata,
        evaluator=evaluate_normalized_cycle_life_interval_coverage,
    )


def _group_rows(
    *,
    method: str,
    intervals: tuple[Any, ...],
    split_manifest: SplitManifest,
    family: str,
    candidate_id: str,
    target_coverage: float,
    metadata: dict[str, dict[str, str]],
    evaluator: Any,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_type in ("batch_date", "life_quartile"):
        values = sorted({item[group_type] for item in metadata.values()})
        for group_value in values:
            selected = tuple(
                interval
                for interval in intervals
                if metadata[interval.prediction.cell_id][group_type] == group_value
            )
            coverage = evaluator(selected, split_manifest=split_manifest)
            rows.append(
                {
                    "method": method,
                    "family": family,
                    "candidate_id": candidate_id,
                    "cutoff_cycle": selected[0].prediction.cutoff_cycle,
                    "target_coverage": target_coverage,
                    "group_type": "batch" if group_type == "batch_date" else group_type,
                    "group_value": group_value,
                    "cell_count": coverage.evaluated_cell_count,
                    "picp": coverage.picp,
                    "mpiw_cycle": coverage.mpiw_cycle,
                    "warnings": _warnings(coverage.warnings),
                }
            )
    return rows


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    partition: str,
    split_manifest: SplitManifest,
    expected_seeds: tuple[int, ...],
) -> None:
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{partition} prediction export is missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{partition} prediction export must not be empty")
    if frame[list(_REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError(f"{partition} prediction export contains null required values")
    duplicates = frame.duplicated(
        subset=["family", "candidate_id", "cutoff_cycle", "seed", "cell_id"]
    )
    if duplicates.any():
        raise ValueError(f"{partition} prediction export contains duplicate rows")
    expected_cells = set(getattr(split_manifest, partition))
    actual_cells = set(frame["cell_id"].astype(str))
    if actual_cells != expected_cells:
        raise ValueError(f"{partition} cells do not exactly match the split manifest")
    expected_seed_set = set(expected_seeds)
    group_columns = ["family", "candidate_id", "cutoff_cycle"]
    for coordinate, group in frame.groupby(group_columns, sort=True, dropna=False):
        if set(int(value) for value in group["seed"].unique()) != expected_seed_set:
            raise ValueError(f"coordinate {coordinate} does not contain expected seeds")
        cell_seed_counts = group.groupby("cell_id")["seed"].nunique()
        if (cell_seed_counts != len(expected_seeds)).any():
            raise ValueError(f"coordinate {coordinate} is missing seed/cell predictions")
    true_counts = frame.groupby("cell_id")["true_cycle_life"].nunique()
    if (true_counts != 1).any():
        raise ValueError(f"{partition} true cycle life changes across runs")
    batch_counts = frame.groupby("cell_id")["batch_date"].nunique()
    if (batch_counts != 1).any():
        raise ValueError(f"{partition} batch date changes across runs")


def _validate_coordinate_match(
    calibration_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> None:
    columns = ["family", "candidate_id", "cutoff_cycle"]
    calibration_coordinates = {
        tuple(row)
        for row in calibration_frame[columns].drop_duplicates().itertuples(
            index=False,
            name=None,
        )
    }
    test_coordinates = {
        tuple(row)
        for row in test_frame[columns].drop_duplicates().itertuples(
            index=False,
            name=None,
        )
    }
    if calibration_coordinates != test_coordinates:
        raise ValueError("calibration and test model coordinates do not match")


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
        raise ValueError("failed to assign life quartiles")
    return result


def _load_source_manifest(
    path: Path,
    *,
    expected_schema: str,
    prediction_name: str,
    prediction_sha256: str,
    expected_partition: str | None,
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
        "config_sha256",
        "input_bundle_sha256",
        "local_reconstructed_input_bundle_sha256",
        "input_bundle_hashes_match",
        "files",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"prediction export manifest is missing fields: {missing}")
    if payload["schema_version"] != expected_schema:
        raise ValueError("prediction export manifest schema_version is unsupported")
    if expected_partition is not None and payload.get("partition") != expected_partition:
        raise ValueError("prediction export manifest partition is invalid")
    files = payload["files"]
    if not isinstance(files, dict) or files.get(prediction_name) != prediction_sha256:
        raise ValueError("prediction parquet SHA-256 does not match its export manifest")
    return {
        "path": str(manifest_path),
        "sha256": _sha256_file(manifest_path),
        "schema_version": payload["schema_version"],
        "partition": payload.get("partition"),
        "source_commit": payload["source_commit"],
        "data_version": payload["data_version"],
        "split_version": payload["split_version"],
        "config_sha256": payload["config_sha256"],
        "input_bundle_sha256": payload["input_bundle_sha256"],
        "local_reconstructed_input_bundle_sha256": payload[
            "local_reconstructed_input_bundle_sha256"
        ],
        "input_bundle_hashes_match": payload["input_bundle_hashes_match"],
    }


def _require_shared_provenance(
    calibration_manifest: dict[str, object],
    test_manifest: dict[str, object],
) -> None:
    for field in (
        "source_commit",
        "data_version",
        "split_version",
        "config_sha256",
        "input_bundle_sha256",
        "local_reconstructed_input_bundle_sha256",
        "input_bundle_hashes_match",
    ):
        if calibration_manifest[field] != test_manifest[field]:
            raise ValueError(f"calibration and test provenance differ for {field}")


def _parse_int_values(value: str, *, name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must contain unique values")
    return tuple(sorted(parsed))


def _parse_float_values(value: str, *, name: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated numbers") from exc
    if not parsed or len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must contain unique values")
    if any(not 0 < item < 1 for item in parsed):
        raise ValueError(f"{name} must be in (0, 1)")
    return tuple(sorted(parsed))


def _warnings(method_warnings: tuple[str, ...]) -> str:
    values = (*method_warnings, _COVERAGE_BOUNDARY_WARNING)
    return ";".join(dict.fromkeys(values))


def _input_evidence(path: Path, sha256: str) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256,
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
        raise ValueError(f"refusing to publish an empty Conformal file: {path.name}")
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
