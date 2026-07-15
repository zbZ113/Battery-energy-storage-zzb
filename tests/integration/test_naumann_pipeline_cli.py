from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _request_payload() -> dict[str, object]:
    conditions = (
        ("condition-a", 20.0, 0.4, 0.4, 2.95),
        ("condition-b", 25.0, 0.5, 0.5, 2.90),
        ("condition-c", 35.0, 0.5, 0.6, 2.80),
        ("condition-d", 45.0, 0.6, 0.4, 2.70),
    )
    observations = [
        {
            "dataset_id": "NAUMANN_CYCLE",
            "condition_id": condition_id,
            "observation_axis": "equivalent_full_cycles",
            "observation_value": 100.0,
            "temperature_c": temperature_c,
            "mean_soc": mean_soc,
            "dod": dod,
            "charge_c_rate": 0.5,
            "discharge_c_rate": 0.5,
            "metric_name": "capacity_ah",
            "metric_value": metric_value,
            "source_file": "NAUMANN_CYCLE/reviewed-capacity.mat",
            "source_sha256": "a" * 64,
            "layout_version": "reviewed-cycle-layout-v1",
        }
        for condition_id, temperature_c, mean_soc, dod, metric_value in conditions
    ]
    return {
        "pipeline_config_version": "naumann-pipeline-config-v1",
        "observations": observations,
        "mapping": {
            "mapping_version": "reviewed-naumann-map-v1",
            "resources": {
                "duration_hours": 24.0,
                "equipment_cost": 5.0,
                "review_statement": "Reviewed historical replay resource allocation.",
                "evidence_reference": "reviewed-resource-note-v1",
            },
            "target_transform": {
                "source_metric_name": "capacity_ah",
                "target_name": "capacity_ah",
                "mode": "direct_metric_value",
            },
            "calendar_conditions": [],
        },
        "operating_bounds": {
            "temperature_c": [0.0, 60.0],
            "mean_soc": [0.1, 0.9],
            "dod": [0.1, 0.9],
            "charge_c_rate": [0.1, 2.0],
            "discharge_c_rate": [0.1, 2.0],
        },
        "acquisition_config": {
            "time_normalizer_hours": 24.0,
            "equipment_cost_normalizer": 10.0,
            "duplicate_penalty_weight": 0.1,
            "similarity_length_scale": 0.5,
        },
        "initial_observation_ids": [],
        "initial_condition_ids": ["condition-a", "condition-b"],
        "query_budget": 1,
    }


def test_naumann_pipeline_cli_writes_replay_artifacts(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_payload()), encoding="utf-8")
    output_dir = tmp_path / "output"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd() / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/naumann_gp_pipeline.py",
            "--request",
            str(request_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["status"] == "completed"
    assert (output_dir / "experiment_dataset.json").is_file()
    assert (output_dir / "experiment_dataset.csv").is_file()
    assert (output_dir / "gp_replay.json").is_file()
    assert (output_dir / "run_manifest.json").is_file()


def test_naumann_pipeline_cli_refuses_pickle_request(tmp_path: Path) -> None:
    request_path = tmp_path / "request.pkl"
    request_path.write_bytes(b"not-a-json-request")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path.cwd() / "src")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/naumann_gp_pipeline.py",
            "--request",
            str(request_path),
            "--output-dir",
            str(tmp_path / "output"),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )

    assert completed.returncode != 0
    assert ".json" in completed.stderr
