from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.experiments.naumann_bridge import (
    MetricTargetTransform,
    NaumannGpBridgeMapping,
    ReviewedExperimentResources,
    TargetTransformMode,
)
from quanxin_life.experiments.naumann_pipeline import (
    NaumannAcquisitionConfig,
    NaumannOperatingBounds,
    NaumannPipelineRequest,
    NaumannPipelineStatus,
    run_naumann_gp_pipeline,
)

_SOURCE_SHA = "a" * 64


def _observation(
    condition_id: str,
    *,
    temperature_c: float,
    mean_soc: float,
    dod: float,
    observed_capacity: float,
) -> CycleMatrixObservation:
    return CycleMatrixObservation(
        condition_id=condition_id,
        observation_axis="equivalent_full_cycles",
        observation_value=100.0,
        temperature_c=temperature_c,
        mean_soc=mean_soc,
        dod=dod,
        charge_c_rate=0.5,
        discharge_c_rate=0.5,
        metric_name="capacity_ah",
        metric_value=observed_capacity,
        source_file="NAUMANN_CYCLE/reviewed-capacity.mat",
        source_sha256=_SOURCE_SHA,
        layout_version="reviewed-cycle-layout-v1",
    )


def _observations() -> tuple[CycleMatrixObservation, ...]:
    return (
        _observation(
            "condition-a",
            temperature_c=20.0,
            mean_soc=0.4,
            dod=0.4,
            observed_capacity=2.95,
        ),
        _observation(
            "condition-b",
            temperature_c=25.0,
            mean_soc=0.5,
            dod=0.5,
            observed_capacity=2.90,
        ),
        _observation(
            "condition-c",
            temperature_c=35.0,
            mean_soc=0.5,
            dod=0.6,
            observed_capacity=2.80,
        ),
        _observation(
            "condition-d",
            temperature_c=45.0,
            mean_soc=0.6,
            dod=0.4,
            observed_capacity=2.70,
        ),
    )


def _mapping() -> NaumannGpBridgeMapping:
    return NaumannGpBridgeMapping(
        mapping_version="reviewed-naumann-map-v1",
        resources=ReviewedExperimentResources(
            duration_hours=24.0,
            equipment_cost=5.0,
            review_statement="Reviewed historical replay resource allocation.",
            evidence_reference="reviewed-resource-note-v1",
        ),
        target_transform=MetricTargetTransform(
            source_metric_name="capacity_ah",
            target_name="capacity_ah",
            mode=TargetTransformMode.DIRECT_METRIC_VALUE,
        ),
    )


def _request(mapping: NaumannGpBridgeMapping | None = None) -> NaumannPipelineRequest:
    return NaumannPipelineRequest(
        pipeline_config_version="naumann-pipeline-config-v1",
        observations=_observations(),
        mapping=_mapping() if mapping is None else mapping,
        operating_bounds=NaumannOperatingBounds(
            temperature_c=(0.0, 60.0),
            mean_soc=(0.1, 0.9),
            dod=(0.1, 0.9),
            charge_c_rate=(0.1, 2.0),
            discharge_c_rate=(0.1, 2.0),
        ),
        acquisition_config=NaumannAcquisitionConfig(
            time_normalizer_hours=24.0,
            equipment_cost_normalizer=10.0,
            duplicate_penalty_weight=0.1,
            similarity_length_scale=0.5,
        ),
        initial_observation_ids=(),
        initial_condition_ids=("condition-a", "condition-b"),
        query_budget=1,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_exports_deterministic_traceable_dataset_and_gp_replay(tmp_path: Path) -> None:
    first = run_naumann_gp_pipeline(_request(), output_dir=tmp_path / "first")
    second = run_naumann_gp_pipeline(_request(), output_dir=tmp_path / "second")

    assert first.status is NaumannPipelineStatus.COMPLETED
    assert first.run_id == second.run_id
    assert first.config_hash == second.config_hash
    assert first.source_sha256s == (_SOURCE_SHA,)
    assert first.artifacts is not None
    assert second.artifacts is not None
    for name in ("dataset_json", "dataset_csv", "replay_json"):
        first_artifact = getattr(first.artifacts, name)
        second_artifact = getattr(second.artifacts, name)
        assert first_artifact.sha256 == second_artifact.sha256
        assert _sha256(tmp_path / "first" / first_artifact.relative_path) == first_artifact.sha256

    dataset = json.loads((tmp_path / "first" / "experiment_dataset.json").read_text())
    replay = json.loads((tmp_path / "first" / "gp_replay.json").read_text())
    assert dataset["config_hash"] == first.config_hash
    assert dataset["source_sha256s"] == [_SOURCE_SHA]
    assert len(dataset["observations"]) == 4
    assert all(row["source_sha256"] == _SOURCE_SHA for row in dataset["observations"])
    assert all(
        row["source_axis_name"] == "equivalent_full_cycles"
        and row["source_axis_value"] == 100.0
        and row["source_metric_name"] == "capacity_ah"
        for row in dataset["observations"]
    )
    assert replay["dataset_sha256"] == first.artifacts.dataset_json.sha256
    assert replay["winner"] is None
    assert {item["strategy"] for item in replay["trajectories"]} == {
        "random",
        "uniform_grid",
        "max_variance",
        "cost_aware_eivr",
    }

    with (tmp_path / "first" / "experiment_dataset.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(row["config_hash"] == first.config_hash for row in rows)
    assert all(row["source_sha256"] == _SOURCE_SHA for row in rows)


def test_pipeline_explicitly_degrades_when_reviewed_cost_metadata_is_missing(
    tmp_path: Path,
) -> None:
    request = _request().model_copy(update={"mapping": None})

    result = run_naumann_gp_pipeline(request, output_dir=tmp_path)

    assert result.status is NaumannPipelineStatus.DEGRADED_MISSING_REVIEWED_RESOURCES
    assert result.artifacts is None
    assert result.warnings == ["REVIEWED_EXPERIMENT_RESOURCES_REQUIRED_FOR_REPLAY"]
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "degraded_missing_reviewed_resources"
    assert "equipment_cost" not in manifest
    assert not (tmp_path / "experiment_dataset.json").exists()
    assert not (tmp_path / "gp_replay.json").exists()


def test_pipeline_rejects_duplicate_operating_conditions_instead_of_aggregating(
    tmp_path: Path,
) -> None:
    observations = (*_observations(), _observations()[0].model_copy(update={"condition_id": "x"}))
    request = _request().model_copy(update={"observations": observations})

    with pytest.raises(ValueError, match="duplicate replay condition vector"):
        run_naumann_gp_pipeline(request, output_dir=tmp_path)


def test_pipeline_requires_explicit_initial_cohort(tmp_path: Path) -> None:
    request = _request().model_copy(
        update={"initial_condition_ids": (), "initial_observation_ids": ()}
    )

    with pytest.raises(ValueError, match="initial observation cohort"):
        run_naumann_gp_pipeline(request, output_dir=tmp_path)


def test_degraded_run_refuses_to_leave_stale_replay_artifacts(tmp_path: Path) -> None:
    (tmp_path / "gp_replay.json").write_text('{"stale": true}', encoding="utf-8")
    request = _request().model_copy(update={"mapping": None})

    with pytest.raises(ValueError, match="already contains Naumann pipeline artifacts"):
        run_naumann_gp_pipeline(request, output_dir=tmp_path)
