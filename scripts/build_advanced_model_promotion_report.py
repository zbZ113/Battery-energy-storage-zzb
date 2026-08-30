# ruff: noqa: RUF001
"""Build a traceable advanced-model promotion report without activating models."""

from __future__ import annotations

import argparse
import hashlib
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
    RulPromotionEvidence,
    SohPromotionEvidence,
    recommend_rul_routes,
    recommend_soh_routes,
)

_RUL_SUMMARY = "rul_metric_closure.csv"
_RUL_COMPARISON = "rul_model_comparisons.csv"
_SOH_SUMMARY = "soh_metric_closure.csv"
_SOH_COMPARISON = "soh_model_comparisons.csv"
_CONFORMAL_SUMMARY = "rul_conformal_summary.csv"
_REPORT_JSON = "promotion_report.json"
_REPORT_MARKDOWN = "promotion_report.md"
_DECISIONS_CSV = "promotion_decisions.csv"
_REFUSAL_POLICY = "refusal_policy.json"
_SOURCE_EVIDENCE = "source_evidence.json"
_ARTIFACT_MANIFEST = "artifact_manifest.json"
_TARGET_COVERAGE = 0.90
_MODEL_CARD_NAMES = {
    "cyclepatch_direct": "cyclepatch_direct.md",
    "cyclepatch_batlinet": "cyclepatch_batlinet.md",
    "current_hybrid": "current_hybrid.md",
    "hybridpatch_v2": "hybridpatch_v2.md",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rul_closure_dir", type=Path)
    parser.add_argument("soh_closure_dir", type=Path)
    parser.add_argument("conformal_dir", type=Path)
    parser.add_argument("run_metrics", type=Path)
    parser.add_argument("evidence_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-cutoffs", default="20,50,100,150")
    parser.add_argument("--expected-seeds", default="38,39,40,41,42")
    args = parser.parse_args()
    manifest = build_promotion_report(
        args.rul_closure_dir,
        args.soh_closure_dir,
        args.conformal_dir,
        args.run_metrics,
        args.evidence_report,
        args.output_dir,
        expected_cutoffs=_parse_int_values(
            args.expected_cutoffs,
            name="expected cutoffs",
        ),
        expected_seeds=_parse_int_values(args.expected_seeds, name="expected seeds"),
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


def build_promotion_report(
    rul_closure_dir: Path,
    soh_closure_dir: Path,
    conformal_dir: Path,
    run_metrics_path: Path,
    evidence_report_path: Path,
    output_dir: Path,
    *,
    expected_cutoffs: tuple[int, ...],
    expected_seeds: tuple[int, ...],
) -> dict[str, object]:
    rul_root = rul_closure_dir.resolve(strict=True)
    soh_root = soh_closure_dir.resolve(strict=True)
    conformal_root = conformal_dir.resolve(strict=True)
    run_path = run_metrics_path.resolve(strict=True)
    evidence_path = evidence_report_path.resolve(strict=True)
    output_root = output_dir.resolve()

    rul_manifest, rul_inputs = _load_closure_inputs(
        rul_root,
        manifest_name="metric_closure_manifest.json",
        schema_version="advanced-metric-closure-manifest-v1",
        names=(_RUL_SUMMARY, _RUL_COMPARISON),
    )
    soh_manifest, soh_inputs = _load_closure_inputs(
        soh_root,
        manifest_name="metric_closure_manifest.json",
        schema_version="advanced-soh-metric-closure-manifest-v1",
        names=(_SOH_SUMMARY, _SOH_COMPARISON),
    )
    conformal_manifest, conformal_inputs = _load_closure_inputs(
        conformal_root,
        manifest_name="conformal_manifest.json",
        schema_version="advanced-rul-conformal-manifest-v1",
        names=(_CONFORMAL_SUMMARY,),
    )
    source_provenance = _shared_source_provenance(
        rul_manifest["source_prediction_manifest"],
        soh_manifest["source_prediction_manifest"],
        conformal_manifest["calibration_source_manifest"],
    )

    rul_summary = pd.read_csv(rul_root / _RUL_SUMMARY)
    rul_comparison = pd.read_csv(rul_root / _RUL_COMPARISON)
    soh_summary = pd.read_csv(soh_root / _SOH_SUMMARY)
    soh_comparison = pd.read_csv(soh_root / _SOH_COMPARISON)
    conformal_summary = pd.read_csv(conformal_root / _CONFORMAL_SUMMARY)
    run_metrics = pd.read_csv(run_path)
    evidence_report = _load_json_object(evidence_path)
    _validate_evidence_report(evidence_report, run_count=len(run_metrics))
    _validate_inputs(
        rul_summary=rul_summary,
        rul_comparison=rul_comparison,
        soh_summary=soh_summary,
        soh_comparison=soh_comparison,
        conformal_summary=conformal_summary,
        run_metrics=run_metrics,
        expected_cutoffs=expected_cutoffs,
        expected_seeds=expected_seeds,
    )
    representative_runs = _representative_runs(run_metrics)
    efficiency = _efficiency_rows(run_metrics)

    decisions: list[dict[str, object]] = []
    for cutoff in expected_cutoffs:
        decisions.extend(
            _rul_decisions(
                cutoff=cutoff,
                summary=rul_summary,
                comparison=rul_comparison,
                conformal=conformal_summary,
                representatives=representative_runs,
                efficiency=efficiency,
            )
        )
        decisions.extend(
            _soh_decisions(
                cutoff=cutoff,
                summary=soh_summary,
                comparison=soh_comparison,
                representatives=representative_runs,
                efficiency=efficiency,
            )
        )

    mlflow_delivered = (
        _as_int(evidence_report["evidence_counts"]["mlflow_directory_in_export"]) > 0
    )
    inference_latency_available = any(
        column.startswith("inference_") for column in run_metrics.columns
    )
    source_evidence = {
        "schema_version": "advanced-promotion-source-evidence-v1",
        "source_provenance": source_provenance,
        "inputs": {
            **rul_inputs,
            **soh_inputs,
            **conformal_inputs,
            "advanced_final_run_metrics.csv": _file_evidence(run_path),
            "advanced_final_evidence_report.json": _file_evidence(evidence_path),
        },
        "manifests": {
            "rul_metric_closure": _file_evidence(
                rul_root / "metric_closure_manifest.json"
            ),
            "soh_metric_closure": _file_evidence(
                soh_root / "metric_closure_manifest.json"
            ),
            "rul_conformal": _file_evidence(
                conformal_root / "conformal_manifest.json"
            ),
        },
        "run_count": len(run_metrics),
        "expected_cutoffs": list(expected_cutoffs),
        "expected_seeds": list(expected_seeds),
    }
    refusal_policy = _refusal_policy(expected_cutoffs)
    report = {
        "schema_version": "advanced-model-promotion-report-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "activation_status": "NOT_ACTIVATED",
        "unconditional_champion_declared": False,
        "representative_seed_rule": "MINIMUM_BEST_VALIDATION_METRIC",
        "representative_seed_uses_test_metrics": False,
        "rul_routing_policy": {
            "point_role": "POINT_ACCURACY",
            "coverage_role": "COVERAGE",
            "coverage_method": "split",
            "target_coverage": _TARGET_COVERAGE,
            "normalized_method_status": "CANDIDATE_NOT_PROMOTED",
        },
        "soh_routing_policy": {
            "default_when_single_route_required": "current_hybrid",
            "enhanced_mean_accuracy_role": "hybridpatch_v2",
            "route_status": "CONDITIONAL_DUAL_ROUTE",
        },
        "source_provenance": source_provenance,
        "inference_latency_evidence_available": inference_latency_available,
        "mlflow_export_delivered": mlflow_delivered,
        "decisions": decisions,
        "warnings": [
            "MATR_INTERNAL_EVIDENCE_ONLY",
            "MODEL_ACTIVATION_REQUIRES_MANUAL_APPROVAL",
            "INPUT_BUNDLE_BYTES_DIFFER_FROM_LOCAL_RECONSTRUCTION",
            "SMALL_CALIBRATION_COHORT",
            "NO_CROSS_DOMAIN_COVERAGE_CLAIM",
            "TEST_SPLIT_CONSUMED_FOR_ONE_TIME_PROMOTION",
            "MLFLOW_EXPORT_NOT_DELIVERED",
            "INFERENCE_LATENCY_NOT_MEASURED",
        ],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    cards_root = output_root / "model_cards"
    cards_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / _SOURCE_EVIDENCE, source_evidence)
    _write_json(output_root / _REFUSAL_POLICY, refusal_policy)
    _write_json(output_root / _REPORT_JSON, report)
    _write_csv(output_root / _DECISIONS_CSV, [_decision_csv_row(item) for item in decisions])
    _write_text(output_root / _REPORT_MARKDOWN, _report_markdown(report))
    for family, name in _MODEL_CARD_NAMES.items():
        _write_text(
            cards_root / name,
            _model_card_markdown(
                family=family,
                decisions=decisions,
                rul_summary=rul_summary,
                soh_summary=soh_summary,
                conformal_summary=conformal_summary,
                run_metrics=run_metrics,
                source_provenance=source_provenance,
                mlflow_delivered=mlflow_delivered,
                inference_latency_available=inference_latency_available,
            ),
        )

    relative_outputs = (
        _REPORT_MARKDOWN,
        _REPORT_JSON,
        _DECISIONS_CSV,
        _REFUSAL_POLICY,
        _SOURCE_EVIDENCE,
        *(
            f"model_cards/{name}"
            for name in sorted(_MODEL_CARD_NAMES.values())
        ),
    )
    artifact_manifest: dict[str, object] = {
        "schema_version": "advanced-promotion-artifact-manifest-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "activation_status": "NOT_ACTIVATED",
        "files": [
            {
                "relative_path": relative_path,
                "size_bytes": (output_root / Path(relative_path)).stat().st_size,
                "sha256": _sha256_file(output_root / Path(relative_path)),
            }
            for relative_path in relative_outputs
        ],
    }
    _write_json(output_root / _ARTIFACT_MANIFEST, artifact_manifest)
    return artifact_manifest


def _rul_decisions(
    *,
    cutoff: int,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    conformal: pd.DataFrame,
    representatives: dict[tuple[str, str, int], dict[str, object]],
    efficiency: dict[tuple[str, str, int], dict[str, float]],
) -> list[dict[str, object]]:
    cohort = summary.loc[summary["cutoff_cycle"] == cutoff]
    split = conformal.loc[
        (conformal["cutoff_cycle"] == cutoff)
        & (conformal["method"] == "split")
        & (conformal["target_coverage"].sub(_TARGET_COVERAGE).abs() < 1e-9)
    ]
    conformal_by_coordinate = {
        (str(row.family), str(row.candidate_id)): row
        for row in split.itertuples(index=False)
    }
    evidence = []
    summary_by_coordinate: dict[tuple[str, str], Any] = {}
    for row in cohort.itertuples(index=False):
        coordinate = (str(row.family), str(row.candidate_id))
        summary_by_coordinate[coordinate] = row
        interval = conformal_by_coordinate[coordinate]
        evidence.append(
            RulPromotionEvidence(
                family=coordinate[0],
                candidate_id=coordinate[1],
                cutoff_cycle=cutoff,
                mae_cycle=_as_float(row.mae_cycle),
                p90_absolute_error_cycle=_as_float(row.absolute_error_p90_cycle),
                split_picp=_as_float(interval.picp),
                split_mpiw_cycle=_as_float(interval.mpiw_cycle),
            )
        )
    recommendations = recommend_rul_routes(
        evidence,
        target_coverage=_TARGET_COVERAGE,
    )
    comparison_row = comparison.loc[comparison["cutoff_cycle"] == cutoff].iloc[0]
    ci_crosses_zero = (
        _as_float(comparison_row["delta_ci_lower"]) <= 0
        <= _as_float(comparison_row["delta_ci_upper"])
    )
    decisions: list[dict[str, object]] = []
    for recommendation in recommendations:
        coordinate = (recommendation.family, recommendation.candidate_id)
        row = summary_by_coordinate[coordinate]
        interval = conformal_by_coordinate[coordinate]
        representative = representatives[(coordinate[0], coordinate[1], cutoff)]
        warnings = list(recommendation.warnings)
        if ci_crosses_zero:
            warnings.append("RUL_MAE_PAIRWISE_CI_CROSSES_ZERO")
        decisions.append(
            {
                **recommendation.model_dump(mode="json"),
                "warnings": list(dict.fromkeys(warnings)),
                "representative_seed": representative["seed"],
                "representative_best_epoch": representative["best_epoch"],
                "representative_seed_rule": "MINIMUM_BEST_VALIDATION_METRIC",
                "evidence": {
                    "mae_cycle": _as_float(row.mae_cycle),
                    "rmse_cycle": _as_float(row.rmse_cycle),
                    "mape_percent": _as_float(row.mape_percent),
                    "r2": _as_float(row.r2),
                    "accuracy_at_15_percent": _as_float(row.accuracy_at_15_percent),
                    "p90_absolute_error_cycle": _as_float(
                        row.absolute_error_p90_cycle
                    ),
                    "p95_absolute_error_cycle": _as_float(
                        row.absolute_error_p95_cycle
                    ),
                    "max_absolute_error_cycle": _as_float(
                        row.absolute_error_p100_cycle
                    ),
                    "split_90_picp": _as_float(interval.picp),
                    "split_90_mpiw_cycle": _as_float(interval.mpiw_cycle),
                    "pairwise_delta_ci_lower": _as_float(
                        comparison_row["delta_ci_lower"]
                    ),
                    "pairwise_delta_ci_upper": _as_float(
                        comparison_row["delta_ci_upper"]
                    ),
                    **efficiency[(coordinate[0], coordinate[1], cutoff)],
                },
            }
        )
    return decisions


def _soh_decisions(
    *,
    cutoff: int,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    representatives: dict[tuple[str, str, int], dict[str, object]],
    efficiency: dict[tuple[str, str, int], dict[str, float]],
) -> list[dict[str, object]]:
    cohort = summary.loc[summary["cutoff_cycle"] == cutoff]
    summary_by_coordinate: dict[tuple[str, str], Any] = {}
    evidence = []
    for row in cohort.itertuples(index=False):
        coordinate = (str(row.family), str(row.candidate_id))
        summary_by_coordinate[coordinate] = row
        resources = efficiency[(coordinate[0], coordinate[1], cutoff)]
        evidence.append(
            SohPromotionEvidence(
                family=coordinate[0],
                candidate_id=coordinate[1],
                cutoff_cycle=cutoff,
                mae_soh=_as_float(row.mae_soh),
                rmse_soh=_as_float(row.rmse_soh),
                p90_cell_mae_soh=_as_float(row.cell_mae_p90_soh),
                monotonic_violation_rate_percent=_as_float(
                    row.monotonic_violation_rate_percent
                ),
                training_time_seconds=resources["training_time_seconds_mean"],
                peak_gpu_memory_mib=resources["peak_gpu_memory_mib_mean"],
            )
        )
    recommendations = recommend_soh_routes(evidence)
    comparison_row = comparison.loc[comparison["cutoff_cycle"] == cutoff].iloc[0]
    decisions: list[dict[str, object]] = []
    for recommendation in recommendations:
        coordinate = (recommendation.family, recommendation.candidate_id)
        row = summary_by_coordinate[coordinate]
        representative = representatives[(coordinate[0], coordinate[1], cutoff)]
        decisions.append(
            {
                **recommendation.model_dump(mode="json"),
                "warnings": list(recommendation.warnings),
                "representative_seed": representative["seed"],
                "representative_best_epoch": representative["best_epoch"],
                "representative_seed_rule": "MINIMUM_BEST_VALIDATION_METRIC",
                "evidence": {
                    "mae_soh": _as_float(row.mae_soh),
                    "rmse_soh": _as_float(row.rmse_soh),
                    "accuracy_at_1_soh_percentage_points": _as_float(
                        row.accuracy_at_1_soh_percentage_points
                    ),
                    "accuracy_at_2_soh_percentage_points": _as_float(
                        row.accuracy_at_2_soh_percentage_points
                    ),
                    "accuracy_at_5_soh_percentage_points": _as_float(
                        row.accuracy_at_5_soh_percentage_points
                    ),
                    "p90_cell_mae_soh": _as_float(row.cell_mae_p90_soh),
                    "p95_cell_mae_soh": _as_float(row.cell_mae_p95_soh),
                    "max_cell_mae_soh": _as_float(row.cell_mae_p100_soh),
                    "monotonic_violation_rate_percent": _as_float(
                        row.monotonic_violation_rate_percent
                    ),
                    "pairwise_mean_cell_mae_delta_soh": _as_float(
                        comparison_row["mean_cell_mae_delta_soh"]
                    ),
                    "pairwise_delta_ci_lower": _as_float(
                        comparison_row["delta_ci_lower"]
                    ),
                    "pairwise_delta_ci_upper": _as_float(
                        comparison_row["delta_ci_upper"]
                    ),
                    **efficiency[(coordinate[0], coordinate[1], cutoff)],
                },
            }
        )
    return decisions


def _representative_runs(
    run_metrics: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, object]]:
    result: dict[tuple[str, str, int], dict[str, object]] = {}
    for coordinate, group in run_metrics.groupby(
        ["family", "candidate_id", "cutoff_cycle"],
        sort=True,
        dropna=False,
    ):
        ordered = group.sort_values(
            ["best_validation_metric", "seed"],
            ascending=True,
        )
        selected = ordered.iloc[0]
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise ValueError("representative run coordinate is invalid")
        family, candidate_id, cutoff = coordinate
        result[(str(family), str(candidate_id), _as_int(cutoff))] = {
            "seed": _as_int(selected["seed"]),
            "best_epoch": _as_int(selected["best_epoch"]),
            "best_validation_metric": _as_float(selected["best_validation_metric"]),
        }
    return result


def _efficiency_rows(
    run_metrics: pd.DataFrame,
) -> dict[tuple[str, str, int], dict[str, float]]:
    grouped = run_metrics.groupby(
        ["family", "candidate_id", "cutoff_cycle"],
        sort=True,
        dropna=False,
    ).agg(
        training_time_seconds_mean=("training_time_seconds", "mean"),
        peak_gpu_memory_mib_mean=("peak_gpu_memory_mib", "mean"),
    )
    result: dict[tuple[str, str, int], dict[str, float]] = {}
    for coordinate, row in grouped.iterrows():
        if not isinstance(coordinate, tuple) or len(coordinate) != 3:
            raise ValueError("efficiency coordinate is invalid")
        family, candidate_id, cutoff = coordinate
        result[(str(family), str(candidate_id), _as_int(cutoff))] = {
            "training_time_seconds_mean": _as_float(
                row.training_time_seconds_mean
            ),
            "peak_gpu_memory_mib_mean": _as_float(row.peak_gpu_memory_mib_mean),
        }
    return result


def _validate_inputs(
    *,
    rul_summary: pd.DataFrame,
    rul_comparison: pd.DataFrame,
    soh_summary: pd.DataFrame,
    soh_comparison: pd.DataFrame,
    conformal_summary: pd.DataFrame,
    run_metrics: pd.DataFrame,
    expected_cutoffs: tuple[int, ...],
    expected_seeds: tuple[int, ...],
) -> None:
    required_by_frame = {
        "RUL summary": (
            rul_summary,
            {
                "family",
                "candidate_id",
                "cutoff_cycle",
                "seed_count",
                "mae_cycle",
                "rmse_cycle",
                "mape_percent",
                "r2",
                "accuracy_at_15_percent",
                "absolute_error_p90_cycle",
                "absolute_error_p95_cycle",
                "absolute_error_p100_cycle",
            },
        ),
        "RUL comparison": (
            rul_comparison,
            {"cutoff_cycle", "delta_ci_lower", "delta_ci_upper"},
        ),
        "SOH summary": (
            soh_summary,
            {
                "family",
                "candidate_id",
                "cutoff_cycle",
                "seed_count",
                "mae_soh",
                "rmse_soh",
                "cell_mae_p90_soh",
                "cell_mae_p95_soh",
                "cell_mae_p100_soh",
                "monotonic_violation_rate_percent",
            },
        ),
        "SOH comparison": (
            soh_comparison,
            {
                "cutoff_cycle",
                "mean_cell_mae_delta_soh",
                "delta_ci_lower",
                "delta_ci_upper",
            },
        ),
        "Conformal summary": (
            conformal_summary,
            {
                "method",
                "family",
                "candidate_id",
                "cutoff_cycle",
                "target_coverage",
                "picp",
                "mpiw_cycle",
                "calibration_cell_count",
            },
        ),
        "run metrics": (
            run_metrics,
            {
                "family",
                "candidate_id",
                "cutoff_cycle",
                "seed",
                "best_epoch",
                "best_validation_metric",
                "training_time_seconds",
                "peak_gpu_memory_mib",
            },
        ),
    }
    for name, (frame, required) in required_by_frame.items():
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
    expected_cutoff_set = set(expected_cutoffs)
    for name, frame in (
        ("RUL summary", rul_summary),
        ("RUL comparison", rul_comparison),
        ("SOH summary", soh_summary),
        ("SOH comparison", soh_comparison),
        ("Conformal summary", conformal_summary),
        ("run metrics", run_metrics),
    ):
        if {
            _as_int(value) for value in frame["cutoff_cycle"].unique()
        } != expected_cutoff_set:
            raise ValueError(f"{name} does not contain the expected cutoff matrix")
    if {_as_int(value) for value in run_metrics["seed"].unique()} != set(
        expected_seeds
    ):
        raise ValueError("run metrics do not contain the expected seed matrix")
    coordinate_counts = run_metrics.groupby(
        ["family", "candidate_id", "cutoff_cycle"]
    )["seed"].nunique()
    if (coordinate_counts != len(expected_seeds)).any():
        raise ValueError("run metrics are missing family/cutoff/seed coordinates")
    if run_metrics["best_validation_metric"].isnull().any():
        raise ValueError("representative seed selection requires validation metrics")
    if not all(
        math.isfinite(_as_float(value))
        for value in run_metrics["best_validation_metric"]
    ):
        raise ValueError("representative seed validation metrics must be finite")
    if set(rul_summary["family"]) != {
        "cyclepatch_direct",
        "cyclepatch_batlinet",
    }:
        raise ValueError("RUL promotion expects Direct and BatLiNet candidates")
    if set(soh_summary["family"]) != {"current_hybrid", "hybridpatch_v2"}:
        raise ValueError("SOH promotion expects Current Hybrid and HybridPatch-v2")
    split_90 = conformal_summary.loc[
        (conformal_summary["method"] == "split")
        & (conformal_summary["target_coverage"].sub(_TARGET_COVERAGE).abs() < 1e-9)
    ]
    expected_coordinates = set(
        rul_summary[["family", "candidate_id", "cutoff_cycle"]]
        .itertuples(index=False, name=None)
    )
    actual_coordinates = set(
        split_90[["family", "candidate_id", "cutoff_cycle"]].itertuples(
            index=False,
            name=None,
        )
    )
    if actual_coordinates != expected_coordinates:
        raise ValueError("Split 90% Conformal coordinates do not match RUL candidates")


def _validate_evidence_report(payload: dict[str, object], *, run_count: int) -> None:
    if payload.get("schema_version") != "advanced-final-evidence-report-v1":
        raise ValueError("advanced final evidence report schema is unsupported")
    if _as_int(payload.get("run_count", -1)) != run_count:
        raise ValueError("advanced final evidence report run_count does not match CSV")
    counts = payload.get("evidence_counts")
    if not isinstance(counts, dict):
        raise ValueError("advanced final evidence counts are missing")
    for name in (
        "training_log_jsonl",
        "metrics_epoch_csv",
        "metrics_validation_csv",
        "metrics_test_json",
    ):
        if _as_int(counts.get(name, -1)) != run_count:
            raise ValueError(f"advanced final evidence is incomplete: {name}")


def _load_closure_inputs(
    root: Path,
    *,
    manifest_name: str,
    schema_version: str,
    names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, dict[str, object]]]:
    manifest_path = root / manifest_name
    payload = _load_json_object(manifest_path)
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"unsupported evidence manifest schema: {manifest_path}")
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("evidence manifest outputs are missing")
    evidence: dict[str, dict[str, object]] = {}
    for name in names:
        path = (root / name).resolve(strict=True)
        registered = outputs.get(name)
        if not isinstance(registered, dict):
            raise ValueError(f"evidence manifest does not register {name}")
        actual = _file_evidence(path)
        if registered.get("sha256") != actual["sha256"]:
            raise ValueError(f"evidence file SHA-256 mismatch: {name}")
        if _as_int(registered.get("size_bytes", -1)) != actual["size_bytes"]:
            raise ValueError(f"evidence file size mismatch: {name}")
        evidence[f"{root.name}/{name}"] = actual
    return payload, evidence


def _shared_source_provenance(*sources: object) -> dict[str, object]:
    if not sources or any(not isinstance(source, dict) for source in sources):
        raise ValueError("source provenance is missing from evidence manifests")
    mappings = [source for source in sources if isinstance(source, dict)]
    fields = (
        "source_commit",
        "data_version",
        "split_version",
        "input_bundle_sha256",
        "local_reconstructed_input_bundle_sha256",
        "input_bundle_hashes_match",
    )
    result: dict[str, object] = {}
    for field in fields:
        values = {json.dumps(source.get(field), sort_keys=True) for source in mappings}
        if len(values) != 1:
            raise ValueError(f"promotion evidence provenance differs for {field}")
        result[field] = mappings[0].get(field)
    return result


def _refusal_policy(expected_cutoffs: tuple[int, ...]) -> dict[str, object]:
    conditions = (
        _condition(
            "MODEL_ARTIFACT_SHA256_MISMATCH",
            "PRE_INFERENCE",
            "artifact_sha256_verified",
            "EQ",
            False,
            "REJECT",
        ),
        _condition(
            "MODEL_CONTEXT_MISMATCH",
            "PRE_INFERENCE",
            "registered_context_matches",
            "EQ",
            False,
            "REJECT",
        ),
        _condition(
            "DANGEROUS_ARTIFACT_FORMAT",
            "PRE_INFERENCE",
            "dangerous_artifact_detected",
            "EQ",
            True,
            "REJECT",
        ),
        _condition(
            "UNSUPPORTED_CUTOFF",
            "PRE_INFERENCE",
            "cutoff_cycle",
            "NOT_IN",
            list(expected_cutoffs),
            "REJECT",
        ),
        _condition(
            "TARGET_DOMAIN_UNCALIBRATED",
            "PRE_DECISION",
            "target_domain_calibrated",
            "EQ",
            False,
            "RECHECK",
        ),
        _condition(
            "DATA_QUALITY_BLOCKED",
            "PRE_INFERENCE",
            "data_quality_blocked",
            "EQ",
            True,
            "REJECT",
        ),
        _condition(
            "SOH_HORIZON_EXCEEDS_CYCLE_500",
            "PRE_INFERENCE",
            "requested_soh_cycle",
            "GT",
            500,
            "REJECT",
        ),
        _condition(
            "SMALL_CALIBRATION_COHORT",
            "PRE_DECISION",
            "calibration_cell_count_below_20",
            "EQ",
            True,
            "RECHECK",
        ),
        _condition(
            "INTERVAL_CROSSES_REQUIREMENT",
            "PRE_DECISION",
            "interval_crosses_requirement",
            "EQ",
            True,
            "RECHECK",
        ),
        _condition(
            "FORMAL_TOOL_RESULT_CONTEXT_MISSING",
            "PRE_REPORT",
            "formal_tool_result_context_present",
            "EQ",
            False,
            "REJECT",
        ),
        _condition(
            "INPUT_BUNDLE_BYTE_IDENTITY_UNRESOLVED",
            "PRE_ACTIVATION",
            "input_bundle_hashes_match",
            "EQ",
            False,
            "RECHECK",
        ),
        _condition(
            "PROMOTION_TEST_EVIDENCE_REUSED_FOR_RETUNING",
            "PRE_ACTIVATION",
            "promotion_test_evidence_reused_for_retuning",
            "EQ",
            True,
            "REJECT",
        ),
    )
    return {
        "schema_version": "advanced-model-refusal-policy-v1",
        "policy_version": "matr-advanced-promotion-refusal-v1",
        "conditions": list(conditions),
        "warnings": [
            "SOH_DUAL_ROUTE_CONFLICT_IS_NOT_A_DATA_REJECTION",
            "TRAINING_RESOURCE_EVIDENCE_IS_NOT_INFERENCE_LATENCY",
        ],
    }


def _condition(
    reason_code: str,
    stage: str,
    field: str,
    operator: str,
    expected: object,
    decision: str,
) -> dict[str, object]:
    return {
        "reason_code": reason_code,
        "stage": stage,
        "predicate": {"field": field, "operator": operator, "expected": expected},
        "decision": decision,
        "fallback_role": None,
    }


def _decision_csv_row(decision: dict[str, object]) -> dict[str, object]:
    evidence = decision["evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("promotion decision evidence must be a mapping")
    reason_codes = decision["reason_codes"]
    warnings = decision["warnings"]
    if not isinstance(reason_codes, (list, tuple)) or not isinstance(
        warnings,
        (list, tuple),
    ):
        raise ValueError("promotion reason codes and warnings must be sequences")
    return {
        "task": decision["task"],
        "family": decision["family"],
        "candidate_id": decision["candidate_id"],
        "cutoff_cycle": decision["cutoff_cycle"],
        "role": decision["role"],
        "disposition": decision["disposition"],
        "representative_seed": decision["representative_seed"],
        "representative_best_epoch": decision["representative_best_epoch"],
        "representative_seed_rule": decision["representative_seed_rule"],
        "reason_codes": ";".join(str(value) for value in reason_codes),
        "warnings": ";".join(str(value) for value in warnings),
        **evidence,
    }


def _report_markdown(report: dict[str, object]) -> str:
    decisions = report["decisions"]
    if not isinstance(decisions, list):
        raise ValueError("report decisions must be a list")
    lines = [
        "# Advanced 模型正式晋级报告",
        "",
        "## 状态",
        "",
        "- 激活状态：`NOT_ACTIVATED`",
        "- 所有建议：`CONDITIONAL`，需人工审批和正式套件注册。",
        "- 不声明无条件冠军，不声明跨域覆盖，不声明工业质保。",
        "",
        "## 路由建议",
        "",
        "| 任务 | cutoff | 模型 | 角色 | representative seed |",
        "|---|---:|---|---|---:|",
    ]
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("report decision must be a mapping")
        lines.append(
            f"| {item['task']} | {item['cutoff_cycle']} | {item['family']} | "
            f"{item['role']} | {item['representative_seed']} |"
        )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- representative seed 仅按验证指标选择，未使用测试指标。",
            "- 冻结 test split 已被一次性用于晋级判断，不得继续用于调参或重选模型。",
            "- SOH 采用平均精度与尾部/效率双路由；运行时不得使用真实误差选择路由。",
            "- Split Conformal 为小 calibration 队列基线；Normalized 仅保留候选状态。",
            "- 训练耗时和显存不是推理延迟。",
            "- MLflow 目录未随离线结果包交付。",
            "- A100 input bundle 与本地重建 bundle 字节哈希不同，该差异继续保留。",
            "",
        ]
    )
    return "\n".join(lines)


def _model_card_markdown(
    *,
    family: str,
    decisions: list[dict[str, object]],
    rul_summary: pd.DataFrame,
    soh_summary: pd.DataFrame,
    conformal_summary: pd.DataFrame,
    run_metrics: pd.DataFrame,
    source_provenance: dict[str, object],
    mlflow_delivered: bool,
    inference_latency_available: bool,
) -> str:
    family_decisions = [item for item in decisions if item["family"] == family]
    if not family_decisions:
        raise ValueError(f"model family has no promotion decision: {family}")
    task = str(family_decisions[0]["task"])
    summary = rul_summary if task == "RUL" else soh_summary
    metrics = summary.loc[summary["family"] == family].sort_values("cutoff_cycle")
    resources = run_metrics.loc[run_metrics["family"] == family]
    lines = [
        f"# {family} 模型卡",
        "",
        "## 身份与状态",
        "",
        f"- 任务：`{task}`",
        "- 晋级状态：`CONDITIONAL`",
        "- 激活状态：`NOT_ACTIVATED`",
        f"- source commit：`{source_provenance['source_commit']}`",
        f"- data version：`{source_provenance['data_version']}`",
        f"- split version：`{source_provenance['split_version']}`",
        "",
        "## 预期用途",
        "",
        "仅用于冻结 MATR 三批 cell-level 划分上的科研、竞赛展示与离线辅助分析。",
        "",
        "## 正式指标",
        "",
    ]
    if task == "RUL":
        lines.extend(
            [
                "| cutoff | MAE | RMSE | MAPE (%) | R² | 15%-Acc (%) | P90 error |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics.itertuples(index=False):
            lines.append(
                f"| {_as_int(row.cutoff_cycle)} | {_as_float(row.mae_cycle):.6f} | "
                f"{_as_float(row.rmse_cycle):.6f} | "
                f"{_as_float(row.mape_percent):.6f} | "
                f"{_as_float(row.r2):.6f} | "
                f"{_as_float(row.accuracy_at_15_percent):.6f} | "
                f"{_as_float(row.absolute_error_p90_cycle):.6f} |"
            )
        lines.extend(["", "## Conformal", ""])
        split = conformal_summary.loc[
            (conformal_summary["family"] == family)
            & (conformal_summary["method"] == "split")
            & (conformal_summary["target_coverage"].sub(_TARGET_COVERAGE).abs() < 1e-9)
        ].sort_values("cutoff_cycle")
        lines.extend(
            [
                "| cutoff | target coverage | PICP | MPIW (cycles) | calibration cells |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in split.itertuples(index=False):
            lines.append(
                f"| {_as_int(row.cutoff_cycle)} | {_TARGET_COVERAGE:.2f} | "
                f"{_as_float(row.picp):.6f} | {_as_float(row.mpiw_cycle):.6f} | "
                f"{_as_int(row.calibration_cell_count)} |"
            )
    else:
        lines.extend(
            [
                "| cutoff | MAE (SOH) | RMSE (SOH) | P90 cell MAE | monotonic violation (%) |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in metrics.itertuples(index=False):
            lines.append(
                f"| {_as_int(row.cutoff_cycle)} | {_as_float(row.mae_soh):.6f} | "
                f"{_as_float(row.rmse_soh):.6f} | "
                f"{_as_float(row.cell_mae_p90_soh):.6f} | "
                f"{_as_float(row.monotonic_violation_rate_percent):.6f} |"
            )
    lines.extend(
        [
            "",
            "## 训练与资源证据",
            "",
            f"- 五种子/登记种子运行数：{len(resources)}",
            f"- 平均训练耗时：{_as_float(resources['training_time_seconds'].mean()):.6f} s",
            f"- 平均峰值 GPU 显存：{_as_float(resources['peak_gpu_memory_mib'].mean()):.6f} MiB",
            f"- 推理延迟证据：{'已提供' if inference_latency_available else '未提供'}",
            f"- MLflow 离线目录：{'已交付' if mlflow_delivered else '未交付'}",
            "",
            "## 晋级角色",
            "",
        ]
    )
    for item in family_decisions:
        lines.append(
            f"- cutoff {item['cutoff_cycle']}：`{item['role']}`，"
            f"representative seed `{item['representative_seed']}`。"
        )
    lines.extend(
        [
            "",
            "## 禁止用途与拒绝条件",
            "",
            "- 不用于安全认证、工业质保、无人值守 BMS/EMS 控制或经营收益承诺。",
            "- 域外、版本不匹配、制品哈希失败、未知 cutoff 或缺失 ToolResult 上下文时拒绝或复检。",
            "- SOH 不得外推到真实 cycle 500 之后；RUL 目标不得改写为统一 EOL80。",
            "- 详细机器规则见 `../refusal_policy.json`。",
            "",
            "## 已知限制",
            "",
            "- 仅有 MATR 冻结划分证据，无 Naumann 或工业域覆盖保证。",
            "- calibration 队列只有 12 个电芯。",
            "- A100 input bundle 与本地重建 bundle 字节哈希不同。",
            "- 聚合晋级推荐不等于具体模型制品已经激活。",
            "- 冻结 test split 已用于本次一次性晋级，后续调参必须使用新协议和新留出证据。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_int_values(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique values")
    return tuple(sorted(values))


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"expected valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _file_evidence(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
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
        raise ValueError(f"refusing to publish empty promotion CSV: {path.name}")
    data = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_atomic(path, data)


def _write_text(path: Path, content: str) -> None:
    _write_atomic(path, (content.rstrip() + "\n").encode("utf-8"))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
