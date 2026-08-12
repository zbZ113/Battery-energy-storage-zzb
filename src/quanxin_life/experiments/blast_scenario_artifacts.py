"""Generate versioned BLAST scenario tables and technical figures from results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field

from quanxin_life.core import DatasetBuildStatus, canonical_json_bytes
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.scenarios import (
    BlastScenarioRunner,
    OperationScenario,
    ScenarioSegment,
    load_packaged_blast_route_catalog,
)

SCENARIO_ARTIFACT_IMPLEMENTATION_VERSION = "blast-scenario-artifacts-v1.1.0"
_DAYS_PER_YEAR = 365.25


class _ScenarioSpec(ContractModel):
    scenario_id: str = Field(min_length=1)
    scenario_version: str = Field(min_length=1)
    segments: tuple[ScenarioSegment, ...] = Field(min_length=1)


class _ScenarioSuiteConfig(ContractModel):
    schema_version: Literal["blast-reference-scenario-suite-v1"]
    suite_version: str = Field(min_length=1)
    route_id: str = Field(min_length=1)
    horizon_years: int = Field(ge=1, le=25)
    eol_threshold: float = Field(gt=0, lt=1, allow_inf_nan=False)
    scenarios: tuple[_ScenarioSpec, ...] = Field(min_length=1)
    figure_sets: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...] = Field(min_length=1)


class BlastScenarioArtifactResult(ContractModel):
    status: DatasetBuildStatus
    output_dir: str = Field(min_length=1)
    result_sha256: Sha256
    scenario_count: int = Field(gt=0)
    trajectory_point_count: int = Field(gt=0)
    figure_count: int = Field(gt=0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _verified_validation_predictions(
    result_dir: Path,
    *,
    label: str,
    expected_schema_version: str,
) -> tuple[list[dict[str, str]], str]:
    root = Path(result_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    committed_path = root / "COMMITTED"
    predictions_path = root / "predictions.csv"
    if not all(path.is_file() for path in (manifest_path, committed_path, predictions_path)):
        raise ValueError(f"{label} validation result is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != result_sha:
        raise ValueError(f"{label} validation result COMMITTED hash is invalid")
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != expected_schema_version:
        raise ValueError(f"{label} validation result schema is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError(f"{label} validation result artifacts are invalid")
    expected = next(
        (
            item
            for item in artifacts
            if isinstance(item, dict) and item.get("relative_path") == "predictions.csv"
        ),
        None,
    )
    if (
        expected is None
        or expected.get("size_bytes") != predictions_path.stat().st_size
        or expected.get("sha256") != _sha256(predictions_path)
    ):
        raise ValueError(f"{label} validation predictions hash is invalid")
    with predictions_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{label} validation predictions are empty")
    return rows, result_sha


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path.name}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(
                            value,
                            ensure_ascii=True,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _scenario_from_spec(
    config: _ScenarioSuiteConfig,
    spec: _ScenarioSpec,
) -> OperationScenario:
    return OperationScenario(
        scenario_id=spec.scenario_id,
        scenario_version=spec.scenario_version,
        horizon_years=config.horizon_years,
        eol_threshold=config.eol_threshold,
        segments=spec.segments,
    )


def _run_suite(
    config: _ScenarioSuiteConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    catalog = load_packaged_blast_route_catalog()
    route = next((item for item in catalog.routes if item.route_id == config.route_id), None)
    if route is None:
        raise ValueError("scenario suite references an unknown BLAST route")
    runner = BlastScenarioRunner()
    trajectory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for spec in config.scenarios:
        scenario = _scenario_from_spec(config, spec)
        projection = runner.run(route=route, scenario=scenario)
        for year, scheduled_efc, model_efc, soh in zip(
            projection.natural_years,
            projection.equivalent_full_cycles,
            projection.model_effective_full_cycles,
            projection.soh,
            strict=True,
        ):
            trajectory_rows.append(
                {
                    "scenario_id": projection.scenario_id,
                    "scenario_version": projection.scenario_version,
                    "route_id": projection.route_id,
                    "route_version": projection.route_version,
                    "evidence_level": projection.evidence_level.value,
                    "support_status": projection.support.status.value,
                    "natural_year": year,
                    "scheduled_efc": scheduled_efc,
                    "model_effective_efc": model_efc,
                    "soh": soh,
                }
            )
        summary_rows.append(
            {
                "scenario_id": projection.scenario_id,
                "scenario_version": projection.scenario_version,
                "route_id": projection.route_id,
                "support_status": projection.support.status.value,
                "near_boundary_fields": projection.support.near_boundary_fields,
                "eol_status": projection.eol.status,
                "eol_natural_year": projection.eol.natural_year,
                "eol_scheduled_efc": projection.eol.equivalent_full_cycles,
                "soh_year_15": projection.milestone_soh.get("15"),
                "soh_year_20": projection.milestone_soh.get("20"),
                "soh_year_25": projection.milestone_soh.get("25"),
                "warnings": projection.warnings,
            }
        )
    return trajectory_rows, summary_rows


def _select_trajectories(
    rows: list[dict[str, object]],
    scenario_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    selected = [row for row in rows if row["scenario_id"] in scenario_ids]
    if not selected:
        raise ValueError("figure set did not select any scenario trajectories")
    return selected


def _small_cell_boundary_rows(
    validation_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    calendar_rows = [
        row for row in validation_rows if row["validation_domain"] == "calendar"
    ]
    if not calendar_rows:
        raise ValueError("calendar observations are required for the observed-time boundary")
    observed_boundary_year = max(float(row["axis_value"]) for row in calendar_rows) / (
        24.0 * _DAYS_PER_YEAR
    )
    catalog = load_packaged_blast_route_catalog()
    route = next(item for item in catalog.routes if item.cell_format == "cylindrical")
    scenario = OperationScenario(
        scenario_id="naumann_3ah_boundary_reference",
        scenario_version="naumann-3ah-boundary-reference-v1",
        horizon_years=25,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="all-years",
                start_year=0,
                end_year=25,
                temperature_c=25.0,
                charge_c_rate=0.5,
                discharge_c_rate=0.5,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=300.0,
                rest_duration_hours=1.0,
            ),
        ),
    )
    projection = BlastScenarioRunner().run(route=route, scenario=scenario)
    return [
        {
            "scenario_id": projection.scenario_id,
            "route_id": projection.route_id,
            "natural_year": year,
            "scheduled_efc": efc,
            "soh": soh,
            "observed_time_boundary_year": observed_boundary_year,
            "region": (
                "OBSERVED_SOURCE_TIME_SPAN"
                if year <= observed_boundary_year
                else "LONG_HORIZON_SCENARIO_EXTRAPOLATION"
            ),
        }
        for year, efc, soh in zip(
            projection.natural_years,
            projection.equivalent_full_cycles,
            projection.soh,
            strict=True,
        )
    ]


def _figure_validation_rows(
    validation_rows: list[dict[str, str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    support_rows: list[dict[str, object]] = [
        {
            "validation_domain": row["validation_domain"],
            "condition_id": row["condition_id"],
            "support_status": row["support_status"],
            "observed_soh": float(row["observed_soh"]),
            "predicted_soh": float(row["predicted_soh"]),
            "residual": float(row["residual"]),
            "temperature_c": float(row["temperature_c"]),
            "dod": None if not row["dod"] else float(row["dod"]),
        }
        for row in validation_rows
    ]
    parity_rows: list[dict[str, object]] = [dict(row) for row in support_rows]
    return support_rows, parity_rows


def _large_format_figure_rows(
    validation_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    required = {
        "vendor",
        "cell_id",
        "cycle_number",
        "scheduled_efc",
        "elapsed_days",
        "observed_soh",
        "predicted_soh",
        "support_status",
        "temperature_c",
        "charge_c_rate",
        "route_id",
    }
    rows: list[dict[str, object]] = []
    for source in validation_rows:
        if not required <= source.keys():
            raise ValueError("280Ah validation predictions are missing required fields")
        rows.append(
            {
                "vendor": source["vendor"],
                "cell_id": source["cell_id"],
                "cycle_number": int(source["cycle_number"]),
                "scheduled_efc": float(source["scheduled_efc"]),
                "elapsed_year": float(source["elapsed_days"]) / _DAYS_PER_YEAR,
                "observed_soh": float(source["observed_soh"]),
                "predicted_soh": float(source["predicted_soh"]),
                "support_status": source["support_status"],
                "temperature_c": float(source["temperature_c"]),
                "effective_charge_c_rate": float(source["charge_c_rate"]),
                "route_id": source["route_id"],
                "comparison_scope": "OBSERVED_CYCLE_RANGE_ONLY",
            }
        )
    if not rows:
        raise ValueError("280Ah validation predictions are empty")
    return rows


def _plot_lines(
    plt: Any,
    rows: list[dict[str, object]],
    *,
    path: Path,
    title: str,
    ylabel: str = "SOH",
) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    scenario_ids = list(dict.fromkeys(str(row["scenario_id"]) for row in rows))
    for scenario_id in scenario_ids:
        selected = [row for row in rows if row["scenario_id"] == scenario_id]
        axis.plot(
            [_as_float(row["natural_year"], field="natural_year") for row in selected],
            [_as_float(row["soh"], field="soh") for row in selected],
            linewidth=2.0,
            label=scenario_id,
        )
    axis.set(xlabel="Natural year", ylabel=ylabel, title=title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _render_figures(
    *,
    trajectory_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
    figure_sets: dict[str, tuple[str, ...]],
    boundary_rows: list[dict[str, object]],
    support_rows: list[dict[str, object]],
    parity_rows: list[dict[str, object]],
    large_format_rows: list[dict[str, object]],
    figures_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figures: list[Path] = []
    line_specs = (
        ("temperature", "01_temperature_soh_year.png", "Temperature scenarios"),
        ("dod", "02_dod_soh_year.png", "DoD scenarios"),
        ("rate", "03_rate_soh_year.png", "Charge/discharge-rate scenarios"),
        ("comparison", "04_reference_scenario_comparison.png", "Reference scenarios"),
    )
    for key, name, title in line_specs:
        path = figures_dir / name
        _plot_lines(
            plt,
            _select_trajectories(trajectory_rows, figure_sets[key]),
            path=path,
            title=title,
        )
        figures.append(path)

    baseline = [row for row in trajectory_rows if row["scenario_id"] == "baseline"]
    path = figures_dir / "05_natural_year_and_efc_axes.png"
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    years = [_as_float(row["natural_year"], field="natural_year") for row in baseline]
    soh = [_as_float(row["soh"], field="soh") for row in baseline]
    axis.plot(years, soh, color="#176B87", linewidth=2.2)
    axis.set(xlabel="Natural year", ylabel="SOH", title="Natural time and scheduled EFC")
    axis.grid(alpha=0.25)
    top = axis.twiny()
    top.set_xlim(axis.get_xlim())
    ticks = [0, 5, 10, 15, 20, 25]
    top.set_xticks(ticks)
    top.set_xticklabels(
        [
            format(
                _as_float(
                    baseline[year * 12]["scheduled_efc"], field="scheduled_efc"
                ),
                ".0f",
            )
            for year in ticks
        ]
    )
    top.set_xlabel("Scheduled equivalent full cycles")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)

    comparison = _select_trajectories(trajectory_rows, figure_sets["comparison"])
    path = figures_dir / "06_eol_crossings.png"
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for scenario_id in figure_sets["comparison"]:
        selected = [row for row in comparison if row["scenario_id"] == scenario_id]
        line = axis.plot(
            [_as_float(row["natural_year"], field="natural_year") for row in selected],
            [_as_float(row["soh"], field="soh") for row in selected],
            linewidth=1.8,
            label=scenario_id,
        )[0]
        summary = next(row for row in summary_rows if row["scenario_id"] == scenario_id)
        if summary["eol_status"] == "REACHED":
            axis.scatter(
                [_as_float(summary["eol_natural_year"], field="eol_natural_year")],
                [0.8],
                color=line.get_color(),
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
    axis.axhline(0.8, color="#A61B1B", linestyle="--", linewidth=1.2, label="EOL threshold")
    axis.set(xlabel="Natural year", ylabel="SOH", title="First EOL crossings")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)

    path = figures_dir / "07_observed_and_long_horizon_boundary.png"
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    boundary = _as_float(
        boundary_rows[0]["observed_time_boundary_year"],
        field="observed_time_boundary_year",
    )
    axis.axvspan(0, boundary, color="#D9EAD3", alpha=0.7, label="Naumann observed time span")
    axis.axvspan(boundary, 25, color="#FCE5CD", alpha=0.55, label="Long-horizon scenario")
    axis.plot(
        [_as_float(row["natural_year"], field="natural_year") for row in boundary_rows],
        [_as_float(row["soh"], field="soh") for row in boundary_rows],
        color="#3D5A80",
        linewidth=2.0,
        label="Sony-Murata 3Ah reference scenario",
    )
    axis.axvline(boundary, color="#7F6000", linestyle="--", linewidth=1.2)
    axis.set(xlabel="Natural year", ylabel="SOH", title="Observed-time boundary and extrapolation")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)

    path = figures_dir / "08_support_range_and_ood.png"
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    styles = {
        "OBSERVED_SOURCE_RANGE": ("#2E7D32", "Calendar source range"),
        "SUPPORTED_BY_ROUTE_MANIFEST": ("#1565C0", "Cycle route-supported"),
        "OUTSIDE_ROUTE_MANIFEST": ("#C62828", "Cycle outside route"),
    }
    for status, (color, label) in styles.items():
        selected = [row for row in support_rows if row["support_status"] == status]
        if selected:
            axis.scatter(
                [_as_float(row["observed_soh"], field="observed_soh") for row in selected],
                [_as_float(row["residual"], field="residual") for row in selected],
                s=12,
                alpha=0.55,
                color=color,
                label=label,
            )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(
        xlabel="Observed SOH",
        ylabel="Prediction residual",
        title="Support range and OOD residuals",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)

    path = figures_dir / "09_naumann_predicted_vs_observed.png"
    figure, axis = plt.subplots(figsize=(6.2, 6.0), constrained_layout=True)
    for domain, color in (("calendar", "#00897B"), ("cycle", "#5E35B1")):
        selected = [row for row in parity_rows if row["validation_domain"] == domain]
        axis.scatter(
            [_as_float(row["observed_soh"], field="observed_soh") for row in selected],
            [_as_float(row["predicted_soh"], field="predicted_soh") for row in selected],
            s=13,
            alpha=0.55,
            color=color,
            label=domain,
        )
    values = [
        _as_float(row[key], field=key)
        for row in parity_rows
        for key in ("observed_soh", "predicted_soh")
    ]
    lower, upper = min(values), max(values)
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    axis.set(
        xlabel="Observed SOH",
        ylabel="BLAST predicted SOH",
        title="Naumann pointwise source replay",
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)

    path = figures_dir / "10_280ah_observed_vs_reference.png"
    figure, axis = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    cell_ids = list(dict.fromkeys(str(row["cell_id"]) for row in large_format_rows))
    for cell_id in cell_ids:
        selected = [row for row in large_format_rows if row["cell_id"] == cell_id]
        line = axis.plot(
            [_as_float(row["scheduled_efc"], field="scheduled_efc") for row in selected],
            [_as_float(row["observed_soh"], field="observed_soh") for row in selected],
            linewidth=1.5,
            alpha=0.8,
            label=f"{cell_id} observed",
        )[0]
        axis.plot(
            [_as_float(row["scheduled_efc"], field="scheduled_efc") for row in selected],
            [_as_float(row["predicted_soh"], field="predicted_soh") for row in selected],
            color=line.get_color(),
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
    outside = [
        row
        for row in large_format_rows
        if row["support_status"] == "OUTSIDE_ROUTE_MANIFEST"
    ]
    if outside:
        axis.scatter(
            [_as_float(row["scheduled_efc"], field="scheduled_efc") for row in outside],
            [_as_float(row["observed_soh"], field="observed_soh") for row in outside],
            color="#C62828",
            marker="x",
            s=28,
            linewidth=1.0,
            label="Observed point outside route temperature range",
            zorder=4,
        )
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.2,
        )
    )
    labels.append("BLAST 250 Ah reference (dashed)")
    axis.set(
        xlabel="Observed cycle offset / scheduled EFC",
        ylabel="Relative capacity / BLAST q",
        title="280 Ah observed-range external reference check",
    )
    axis.grid(alpha=0.2)
    axis.legend(handles, labels, fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figures.append(path)
    return figures


def _artifact(path: Path, root: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _existing_result(
    output_dir: Path,
    *,
    config_sha256: str,
    validation_result_sha256: str,
    large_format_validation_result_sha256: str,
    code_revision: str,
) -> BlastScenarioArtifactResult | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    committed_path = output_dir / "COMMITTED"
    if not manifest_path.is_file() or not committed_path.is_file():
        raise ValueError("existing scenario artifact result is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != result_sha:
        raise ValueError("existing scenario artifact COMMITTED hash is invalid")
    manifest = _load_json_object(manifest_path)
    if (
        manifest.get("implementation_version")
        != SCENARIO_ARTIFACT_IMPLEMENTATION_VERSION
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("validation_result_sha256") != validation_result_sha256
        or manifest.get("large_format_validation_result_sha256")
        != large_format_validation_result_sha256
        or manifest.get("code_revision") != code_revision
    ):
        raise ValueError("existing scenario artifacts do not match this run")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("existing scenario artifact descriptors are invalid")
    for item in artifacts:
        if not isinstance(item, dict) or not isinstance(item.get("relative_path"), str):
            raise ValueError("existing scenario artifact descriptor is invalid")
        path = output_dir / item["relative_path"]
        if (
            not path.is_file()
            or item.get("size_bytes") != path.stat().st_size
            or item.get("sha256") != _sha256(path)
        ):
            raise ValueError("existing scenario artifact hash is invalid")
    return BlastScenarioArtifactResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_dir=str(output_dir),
        result_sha256=result_sha,
        scenario_count=int(manifest["scenario_count"]),
        trajectory_point_count=int(manifest["trajectory_point_count"]),
        figure_count=int(manifest["figure_count"]),
    )


def build_blast_scenario_artifacts(
    *,
    repository_root: Path,
    config_path: Path,
    validation_result_dir: Path,
    large_format_validation_result_dir: Path,
    output_dir: Path,
    code_revision: str,
) -> BlastScenarioArtifactResult:
    """Run the configured candidate scenarios and publish tables plus figures."""

    root = Path(repository_root).resolve(strict=True)
    config_file = Path(config_path).resolve(strict=True)
    if root not in config_file.parents:
        raise ValueError("scenario config must remain inside the repository")
    config = _ScenarioSuiteConfig.model_validate(_load_json_object(config_file))
    validation_rows, validation_result_sha = _verified_validation_predictions(
        validation_result_dir,
        label="Naumann",
        expected_schema_version="blast-naumann-validation-result-v1",
    )
    large_format_validation_rows, large_format_validation_result_sha = (
        _verified_validation_predictions(
            large_format_validation_result_dir,
            label="280Ah",
            expected_schema_version="blast-280ah-validation-result-v1",
        )
    )
    normalized_revision = code_revision.strip()
    if not normalized_revision:
        raise ValueError("code_revision must not be blank")
    output = Path(output_dir).resolve()
    config_sha = _sha256(config_file)
    existing = _existing_result(
        output,
        config_sha256=config_sha,
        validation_result_sha256=validation_result_sha,
        large_format_validation_result_sha256=(
            large_format_validation_result_sha
        ),
        code_revision=normalized_revision,
    )
    if existing is not None:
        return existing

    trajectory_rows, summary_rows = _run_suite(config)
    boundary_rows = _small_cell_boundary_rows(validation_rows)
    support_rows, parity_rows = _figure_validation_rows(validation_rows)
    large_format_rows = _large_format_figure_rows(large_format_validation_rows)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise ValueError("scenario artifact temporary directory already exists")
    figures_dir = temporary / "figures"
    figure_data_dir = temporary / "figure_data"
    figures_dir.mkdir(parents=True)
    figure_data_dir.mkdir()

    trajectory_path = temporary / "scenario_trajectories.csv"
    summary_path = temporary / "scenario_summary.csv"
    _write_csv(trajectory_path, trajectory_rows)
    _write_csv(summary_path, summary_rows)
    figure_data = {
        "temperature_scenarios.csv": _select_trajectories(
            trajectory_rows, config.figure_sets["temperature"]
        ),
        "dod_scenarios.csv": _select_trajectories(
            trajectory_rows, config.figure_sets["dod"]
        ),
        "rate_scenarios.csv": _select_trajectories(
            trajectory_rows, config.figure_sets["rate"]
        ),
        "reference_comparison.csv": _select_trajectories(
            trajectory_rows, config.figure_sets["comparison"]
        ),
        "natural_year_efc.csv": [
            row for row in trajectory_rows if row["scenario_id"] == "baseline"
        ],
        "eol_crossings.csv": summary_rows,
        "observed_scenario_boundary.csv": boundary_rows,
        "support_ood.csv": support_rows,
        "naumann_parity.csv": parity_rows,
        "280ah_observed_vs_reference.csv": large_format_rows,
    }
    figure_data_paths: list[Path] = []
    for name, rows in figure_data.items():
        path = figure_data_dir / name
        _write_csv(path, rows)
        figure_data_paths.append(path)
    figure_paths = _render_figures(
        trajectory_rows=trajectory_rows,
        summary_rows=summary_rows,
        figure_sets=config.figure_sets,
        boundary_rows=boundary_rows,
        support_rows=support_rows,
        parity_rows=parity_rows,
        large_format_rows=large_format_rows,
        figures_dir=figures_dir,
    )
    metadata = {
        "schema_version": "blast-scenario-artifact-metadata-v1",
        "implementation_version": SCENARIO_ARTIFACT_IMPLEMENTATION_VERSION,
        "suite_version": config.suite_version,
        "route_id": config.route_id,
        "activation_status": "NOT_ACTIVATED",
        "scenario_count": len(config.scenarios),
        "trajectory_point_count": len(trajectory_rows),
        "figure_count": len(figure_paths),
        "uncertainty_kind": "DETERMINISTIC_SCENARIO",
        "prediction_interval_included": False,
        "warnings": [
            *config.warnings,
            "280AH_EXTERNAL_OBSERVED_RANGE_ONLY",
            "280AH_REFERENCE_MODEL_NOT_PRODUCT_SPECIFIC",
        ],
    }
    metadata_path = temporary / "metadata.json"
    metadata_path.write_bytes(canonical_json_bytes(metadata))
    artifact_paths = [
        trajectory_path,
        summary_path,
        metadata_path,
        *figure_data_paths,
        *figure_paths,
    ]
    manifest = {
        "schema_version": "blast-scenario-artifact-result-v1",
        "implementation_version": SCENARIO_ARTIFACT_IMPLEMENTATION_VERSION,
        "code_revision": normalized_revision,
        "config_path": config_file.relative_to(root).as_posix(),
        "config_sha256": config_sha,
        "validation_result_sha256": validation_result_sha,
        "large_format_validation_result_sha256": (
            large_format_validation_result_sha
        ),
        "scenario_count": len(config.scenarios),
        "trajectory_point_count": len(trajectory_rows),
        "figure_count": len(figure_paths),
        "artifacts": [_artifact(path, temporary) for path in artifact_paths],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (temporary / "manifest.json").write_bytes(manifest_bytes)
    (temporary / "COMMITTED").write_text(result_sha + "\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return BlastScenarioArtifactResult(
        status=DatasetBuildStatus.BUILT,
        output_dir=str(output),
        result_sha256=result_sha,
        scenario_count=len(config.scenarios),
        trajectory_point_count=len(trajectory_rows),
        figure_count=len(figure_paths),
    )


__all__ = [
    "SCENARIO_ARTIFACT_IMPLEMENTATION_VERSION",
    "BlastScenarioArtifactResult",
    "build_blast_scenario_artifacts",
]
