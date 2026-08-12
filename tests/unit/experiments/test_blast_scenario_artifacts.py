from __future__ import annotations

import csv
from pathlib import Path

from quanxin_life.experiments.blast_scenario_artifacts import (
    build_blast_scenario_artifacts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_scenario_artifacts_are_generated_from_numerical_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "scenario-artifacts"

    result = build_blast_scenario_artifacts(
        repository_root=REPOSITORY_ROOT,
        config_path=REPOSITORY_ROOT
        / "configs"
        / "scenarios"
        / "blast_reference_scenarios_v1.json",
        validation_result_dir=REPOSITORY_ROOT
        / "reports"
        / "experiments"
        / "blast_naumann_v1"
        / "validation-v1",
        large_format_validation_result_dir=REPOSITORY_ROOT
        / "reports"
        / "experiments"
        / "blast_280ah_v1"
        / "validation-v1",
        output_dir=output_dir,
        code_revision="test-revision",
    )

    assert result.scenario_count >= 8
    assert result.trajectory_point_count > 2000
    assert result.figure_count >= 10
    with (output_dir / "scenario_trajectories.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    baseline = [row for row in rows if row["scenario_id"] == "baseline"]
    assert baseline[0]["natural_year"] == "0.0"
    assert baseline[-1]["natural_year"] == "25.0"
    assert baseline[-1]["soh"]
    assert baseline[-1]["scheduled_efc"]
    assert (output_dir / "scenario_summary.csv").is_file()
    assert (output_dir / "figure_data" / "temperature_scenarios.csv").is_file()
    assert (output_dir / "figures" / "01_temperature_soh_year.png").is_file()
    assert (output_dir / "figures" / "09_naumann_predicted_vs_observed.png").is_file()
    assert (
        output_dir / "figure_data" / "280ah_observed_vs_reference.csv"
    ).is_file()
    assert (
        output_dir / "figures" / "10_280ah_observed_vs_reference.png"
    ).is_file()
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "COMMITTED").is_file()
