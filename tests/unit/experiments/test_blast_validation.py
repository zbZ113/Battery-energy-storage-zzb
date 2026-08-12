from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from quanxin_life.core import DatasetBuildStatus, canonical_json_bytes
from quanxin_life.experiments.blast_validation import (
    evaluate_naumann_observations,
    run_naumann_blast_validation,
)


def test_naumann_validation_emits_pointwise_errors_and_support_groups() -> None:
    observations = {
        "calendar": [
            {
                "condition_id": "T25_SOC50",
                "source_file": "calendar.xlsx",
                "storage_time_h": 0.0,
                "temperature_c": 25.0,
                "mean_soc": 0.5,
                "capacity_ah": 3.0,
            },
            {
                "condition_id": "T25_SOC50",
                "source_file": "calendar.xlsx",
                "storage_time_h": 24.0,
                "temperature_c": 25.0,
                "mean_soc": 0.5,
                "capacity_ah": 2.97,
            },
        ],
        "cycle": [
            {
                "condition_id": "T40_SOC50_DOD80_C1_C1_CC",
                "source_file": "cycle.mat",
                "observation_axis": "equivalent_full_cycles",
                "observation_value": 0.0,
                "metric_value": 1.0,
                "temperature_c": 40.0,
                "mean_soc": 0.5,
                "dod": 0.8,
                "charge_c_rate": 1.0,
                "discharge_c_rate": 1.0,
            },
            {
                "condition_id": "T40_SOC50_DOD80_C1_C1_CC",
                "source_file": "cycle.mat",
                "observation_axis": "equivalent_full_cycles",
                "observation_value": 1.0,
                "metric_value": 0.99,
                "temperature_c": 40.0,
                "mean_soc": 0.5,
                "dod": 0.8,
                "charge_c_rate": 1.0,
                "discharge_c_rate": 1.0,
            },
            {
                "condition_id": "T40_SOC50_DOD20_C1_C1_CC",
                "source_file": "cycle.mat",
                "observation_axis": "equivalent_full_cycles",
                "observation_value": 0.0,
                "metric_value": 1.0,
                "temperature_c": 40.0,
                "mean_soc": 0.5,
                "dod": 0.2,
                "charge_c_rate": 1.0,
                "discharge_c_rate": 1.0,
            },
            {
                "condition_id": "T25_SOC50_DOD80_C0_5_C0_5_CC",
                "source_file": "cycle.mat",
                "observation_axis": "equivalent_full_cycles",
                "observation_value": 0.0,
                "metric_value": 1.0,
                "temperature_c": 25.0,
                "mean_soc": 0.5,
                "dod": 0.8,
                "charge_c_rate": 0.5,
                "discharge_c_rate": 0.5,
            },
        ],
    }

    result = evaluate_naumann_observations(
        observations,
        bundle_sha256="a" * 64,
    )

    assert len(result.predictions) == 6
    assert {point.validation_domain for point in result.predictions} == {
        "calendar",
        "cycle",
    }
    assert all(math.isfinite(point.predicted_soh) for point in result.predictions)
    assert all(math.isfinite(point.absolute_error) for point in result.predictions)
    assert result.predictions[0].observed_soh == 1.0
    assert result.predictions[1].observed_soh == pytest.approx(0.99, abs=1e-12)

    cycle_support = {
        point.condition_id: point.support_status
        for point in result.predictions
        if point.validation_domain == "cycle"
    }
    assert cycle_support["T40_SOC50_DOD80_C1_C1_CC"] == (
        "SUPPORTED_BY_ROUTE_MANIFEST"
    )
    assert cycle_support["T40_SOC50_DOD20_C1_C1_CC"] == (
        "OUTSIDE_ROUTE_MANIFEST"
    )
    assert any(
        metric.group_kind == "overall" and metric.group_value == "calendar"
        for metric in result.metrics
    )
    assert any(
        metric.group_kind == "temperature_c" and metric.group_value == "40"
        for metric in result.metrics
    )
    leave_one_axes = {
        metric.group_kind
        for metric in result.metrics
        if metric.evaluation_mode
        == "FIXED_UPSTREAM_PARAMETER_LEAVE_ONE_CONDITION_DIAGNOSTIC"
    }
    assert leave_one_axes == {"temperature_c", "dod", "c_rate_pair"}
    assert result.methodology["parameter_fitting_performed"] is False
    assert result.methodology["independent_holdout"] is False
    assert result.methodology["leave_one_condition_parameter_refit"] is False


def test_naumann_validation_run_is_immutable_and_idempotent(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    observations = {
        "schema_version": "blast-validation-observations-v1",
        "processor_version": "test-processor-v1",
        "calendar": [
            {
                "condition_id": "T25_SOC50",
                "source_file": "calendar.xlsx",
                "storage_time_h": 0.0,
                "temperature_c": 25.0,
                "mean_soc": 0.5,
                "capacity_ah": 3.0,
            },
            {
                "condition_id": "T25_SOC50",
                "source_file": "calendar.xlsx",
                "storage_time_h": 24.0,
                "temperature_c": 25.0,
                "mean_soc": 0.5,
                "capacity_ah": 2.97,
            },
        ],
        "cycle": [],
    }
    observation_bytes = canonical_json_bytes(observations)
    (bundle_dir / "observations.json").write_bytes(observation_bytes)
    input_manifest = {
        "schema_version": "blast-validation-bundle-v1",
        "processor_version": "test-processor-v1",
        "observation_counts": {"calendar": 2, "cycle": 0},
        "observations_sha256": hashlib.sha256(observation_bytes).hexdigest(),
    }
    input_manifest_bytes = canonical_json_bytes(input_manifest)
    input_bundle_sha = hashlib.sha256(input_manifest_bytes).hexdigest()
    (bundle_dir / "manifest.json").write_bytes(input_manifest_bytes)
    (bundle_dir / "COMMITTED").write_text(input_bundle_sha + "\n", encoding="ascii")

    output_dir = tmp_path / "result"
    first = run_naumann_blast_validation(
        bundle_dir,
        output_dir=output_dir,
        code_revision="test-revision",
    )
    second = run_naumann_blast_validation(
        bundle_dir,
        output_dir=output_dir,
        code_revision="test-revision",
    )

    assert first.status is DatasetBuildStatus.BUILT
    assert second.status is DatasetBuildStatus.SKIPPED_VALID
    assert first.result_sha256 == second.result_sha256
    assert first.prediction_count == 2
    assert first.metric_count > 0
    assert (output_dir / "COMMITTED").read_text(encoding="ascii").strip() == (
        first.result_sha256
    )
    for name in (
        "config.json",
        "summary.json",
        "predictions.csv",
        "metrics.csv",
        "manifest.json",
    ):
        assert (output_dir / name).is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["methodology"]["parameter_fitting_performed"] is False
    assert summary["methodology"]["independent_holdout"] is False
