from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from quanxin_life.data.adapters.naumann_cycle_mat import (
    CycleConditionColumn,
    NaumannCycleMatrixLayout,
    ReviewedAxisSelection,
)
from quanxin_life.data.manifest import AuditedDatasetFile, DatasetFileAuditManifest
from quanxin_life.experiments.naumann_bridge import (
    MetricTargetTransform,
    NaumannGpBridgeMapping,
    TargetTransformMode,
)
from quanxin_life.experiments.naumann_pipeline import (
    NaumannAcquisitionConfig,
    NaumannOperatingBounds,
    NaumannPipelineStatus,
)
from quanxin_life.experiments.naumann_reviewed_run import (
    ReviewedNaumannReplayConfig,
    run_reviewed_naumann_cycle_replay,
)


def test_reviewed_raw_matrix_runs_three_strategy_replay_without_fake_costs(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "data" / "raw" / "NAUMANN_CYCLE" / "v1" / "reviewed.mat"
    source_path.parent.mkdir(parents=True)
    x_axis = np.asarray([[0.0] * 4, [50.0] * 4, [100.0] * 4])
    y_axis = np.asarray(
        [
            [1.0, 1.0, 1.0, 1.0],
            [0.98, 0.97, 0.96, 0.95],
            [0.95, 0.93, 0.90, 0.87],
        ]
    )
    savemat(
        source_path,
        {
            "X_Axis_Data_Mat": x_axis,
            "Y_Axis_Data_Mat": y_axis,
            "Legend_Vec": np.asarray(["A", "B", "C", "D"], dtype=object),
        },
        do_compression=False,
    )
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()

    catalog_path = tmp_path / "configs" / "data_sources.json"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_text(
        json.dumps(
            [
                {
                    "dataset_id": "NAUMANN_CYCLE",
                    "version": "fixture-v1",
                    "source_uri": "https://example.invalid/naumann-cycle",
                    "paper_uri": "https://doi.org/10.1016/j.jpowsour.2019.227666",
                    "license_status": "CC BY 4.0",
                    "ingestion_mode": "matlab",
                    "expected_suffixes": [".mat"],
                    "downloaded_at": "2026-08-07T00:00:00Z",
                    "artifact_paths": ["data/raw/NAUMANN_CYCLE/v1/reviewed.mat"],
                    "artifact_sha256": [source_sha],
                }
            ]
        ),
        encoding="utf-8",
    )
    audit_path = tmp_path / "configs" / "data_manifests" / "audit.json"
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        DatasetFileAuditManifest(
            manifest_version="fixture-audit-v1",
            source_catalog="configs/data_sources.json",
            file_count=1,
            total_size_bytes=source_path.stat().st_size,
            files=(
                AuditedDatasetFile(
                    dataset_id="NAUMANN_CYCLE",
                        relative_path="data/raw/NAUMANN_CYCLE/v1/reviewed.mat",
                    size_bytes=source_path.stat().st_size,
                    sha256=source_sha,
                ),
            ),
        ).model_dump_json(),
        encoding="utf-8",
    )
    condition_columns = tuple(
        CycleConditionColumn(
            column_index=index,
            expected_legend=legend,
            condition_id=legend,
            temperature_c=temperature,
            mean_soc=mean_soc,
            dod=dod,
            charge_c_rate=0.5,
            discharge_c_rate=0.5,
        )
        for index, (legend, temperature, mean_soc, dod) in enumerate(
            (
                ("A", 20.0, 0.4, 0.4),
                ("B", 25.0, 0.5, 0.5),
                ("C", 35.0, 0.5, 0.6),
                ("D", 45.0, 0.6, 0.4),
            )
        )
    )
    layout_path = tmp_path / "configs" / "data_layouts" / "layout.json"
    layout_path.parent.mkdir(parents=True)
    layout_path.write_text(
        NaumannCycleMatrixLayout(
            layout_version="fixture-layout-v1",
            x_axis_variable="X_Axis_Data_Mat",
            y_axis_variable="Y_Axis_Data_Mat",
            legend_variable="Legend_Vec",
            observation_axis="equivalent_full_cycles",
            metric_name="relative_capacity_ratio",
            condition_columns=condition_columns,
        ).model_dump_json(),
        encoding="utf-8",
    )
    config = ReviewedNaumannReplayConfig(
        config_version="fixture-reviewed-run-v1",
        source_catalog_path="configs/data_sources.json",
        audit_manifest_path="configs/data_manifests/audit.json",
        source_file_path="data/raw/NAUMANN_CYCLE/v1/reviewed.mat",
        layout_path="configs/data_layouts/layout.json",
        selection=ReviewedAxisSelection(
            selection_version="fixture-near-100-v1",
            target_axis_value=90.0,
            max_absolute_deviation=15.0,
            condition_ids=("A", "B", "C", "D"),
        ),
        mapping=NaumannGpBridgeMapping(
            mapping_version="fixture-direct-relative-capacity-v1",
            resources=None,
            target_transform=MetricTargetTransform(
                source_metric_name="relative_capacity_ratio",
                target_name="relative_capacity_ratio",
                mode=TargetTransformMode.DIRECT_METRIC_VALUE,
            ),
        ),
        operating_bounds=NaumannOperatingBounds(
            temperature_c=(0.0, 60.0),
            mean_soc=(0.1, 0.9),
            dod=(0.1, 0.9),
            charge_c_rate=(0.1, 2.0),
            discharge_c_rate=(0.1, 2.0),
        ),
        acquisition_config=NaumannAcquisitionConfig(
            time_normalizer_hours=None,
            equipment_cost_normalizer=None,
            duplicate_penalty_weight=0.0,
            similarity_length_scale=0.5,
        ),
        initial_condition_ids=("A", "B"),
        query_budget=1,
    )

    result = run_reviewed_naumann_cycle_replay(
        tmp_path,
        config,
        output_dir=tmp_path / "artifacts",
    )

    assert result.status is NaumannPipelineStatus.COMPLETED_WITHOUT_COSTS
    replay = json.loads((tmp_path / "artifacts" / "gp_replay.json").read_text())
    assert {item["strategy"] for item in replay["trajectories"]} == {
        "random",
        "uniform_grid",
        "max_variance",
    }
    assert replay["unavailable_strategies"] == ["cost_aware_eivr"]
    dataset = json.loads((tmp_path / "artifacts" / "experiment_dataset.json").read_text())
    assert dataset["configuration"]["selection_audit"]["selection_version"] == (
        "fixture-near-100-v1"
    )
    assert all(item["duration_hours"] is None for item in dataset["observations"])
    assert all(item["equipment_cost"] is None for item in dataset["observations"])
