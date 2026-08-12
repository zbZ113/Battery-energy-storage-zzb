"""Observed-range reference check for the reviewed 280 Ah LFP dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from quanxin_life._vendor.blast_lite import Lfp_Gr_250AhPrismatic
from quanxin_life.core import DatasetBuildStatus, EvidenceLevel, canonical_json_bytes
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.lfp_280ah_dod import DodCapacityCellSeries
from quanxin_life.scenarios import load_packaged_blast_route_catalog

BLAST_280AH_VALIDATION_VERSION: Final = "blast-280ah-reference-check-v1.0.0"
_MODEL_CLASS: Final = "Lfp_Gr_250AhPrismatic"
_SECONDS_PER_HOUR: Final = 3600.0
_SECONDS_PER_DAY: Final = 86_400.0
_TIME_TOLERANCE_S: Final = 1e-6
FloatArray = NDArray[np.float64]


class _BlastBattery(Protocol):
    outputs: dict[str, FloatArray]

    def update_battery_state(
        self,
        time_s: FloatArray,
        soc: FloatArray,
        temperature_c: FloatArray,
    ) -> None: ...


class Blast280AhValidationPoint(ContractModel):
    dataset_id: Literal["LFP_280AH_DOD"] = "LFP_280AH_DOD"
    vendor: Literal["CATL", "EVE"]
    cell_id: str = Field(min_length=1)
    cycle_number: int = Field(ge=1)
    scheduled_efc: float = Field(ge=0, allow_inf_nan=False)
    elapsed_days: float = Field(ge=0, allow_inf_nan=False)
    observed_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    observed_soh: float = Field(gt=0, allow_inf_nan=False)
    predicted_soh: float = Field(ge=0, allow_inf_nan=False)
    residual: float = Field(allow_inf_nan=False)
    absolute_error: float = Field(ge=0, allow_inf_nan=False)
    temperature_c: float = Field(allow_inf_nan=False)
    dod: float = Field(default=1.0, ge=1.0, le=1.0, allow_inf_nan=False)
    charge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    discharge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    support_status: Literal[
        "SUPPORTED_BY_ROUTE_MANIFEST",
        "OUTSIDE_ROUTE_MANIFEST",
    ]
    out_of_range_fields: tuple[str, ...] = ()
    observed_evidence_level: Literal[EvidenceLevel.DATA_DIRECT] = EvidenceLevel.DATA_DIRECT
    prediction_evidence_level: Literal[EvidenceLevel.PHYSICS_REFERENCE] = (
        EvidenceLevel.PHYSICS_REFERENCE
    )
    model_class: Literal["Lfp_Gr_250AhPrismatic"] = _MODEL_CLASS
    route_id: str = Field(min_length=1)
    bundle_sha256: Sha256


class Blast280AhValidationMetric(ContractModel):
    group_kind: Literal["overall", "vendor", "cell_id", "support_status"]
    group_value: str = Field(min_length=1)
    observation_count: int = Field(gt=0)
    mae_soh: float = Field(ge=0, allow_inf_nan=False)
    rmse_soh: float = Field(ge=0, allow_inf_nan=False)
    bias_soh: float = Field(allow_inf_nan=False)


class Blast280AhValidationEvaluation(ContractModel):
    implementation_version: Literal["blast-280ah-reference-check-v1.0.0"] = (
        BLAST_280AH_VALIDATION_VERSION
    )
    predictions: tuple[Blast280AhValidationPoint, ...]
    metrics: tuple[Blast280AhValidationMetric, ...]
    methodology: dict[str, object]
    warnings: tuple[str, ...]


class Blast280AhValidationRunResult(ContractModel):
    status: DatasetBuildStatus
    output_dir: str = Field(min_length=1)
    result_sha256: Sha256
    prediction_count: int = Field(gt=0)
    metric_count: int = Field(gt=0)


def _last_q(battery: _BlastBattery) -> float:
    values = battery.outputs.get("q")
    if values is None or len(values) == 0:
        raise ValueError("BLAST 280Ah reference model did not emit q")
    value = float(values[-1])
    if not math.isfinite(value) or value < 0:
        raise ValueError("BLAST 280Ah reference model emitted an invalid q")
    return value


def _route_support(
    *,
    temperature_c: float,
    charge_c_rate: float,
    discharge_c_rate: float,
    route: Any,
) -> tuple[str, tuple[str, ...]]:
    limits = route.experimental_range
    outside: list[str] = []
    lower_temperature, upper_temperature = limits.cycling_temperature_c
    if not lower_temperature <= temperature_c <= upper_temperature:
        outside.append("temperature_c")
    if not limits.dod[0] <= 1.0 <= limits.dod[1]:
        outside.append("dod")
    if charge_c_rate > limits.max_rate_charge:
        outside.append("charge_c_rate")
    if discharge_c_rate > limits.max_rate_discharge:
        outside.append("discharge_c_rate")
    return (
        "OUTSIDE_ROUTE_MANIFEST" if outside else "SUPPORTED_BY_ROUTE_MANIFEST",
        tuple(outside),
    )


def _cycle_interval_profile(
    *,
    start_time_s: float,
    elapsed_time_s: float,
    cycle_count: int,
    charge_c_rate: float,
    discharge_c_rate: float,
    temperature_c: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    if cycle_count < 1:
        raise ValueError("280Ah reference replay requires a positive cycle increment")
    if elapsed_time_s <= 0:
        raise ValueError("280Ah reference replay requires positive source elapsed time")
    charge_s = _SECONDS_PER_HOUR / charge_c_rate
    discharge_s = _SECONDS_PER_HOUR / discharge_c_rate
    required_s = cycle_count * (charge_s + discharge_s)
    if elapsed_time_s + _TIME_TOLERANCE_S < required_s:
        scale = elapsed_time_s / required_s
        charge_s *= scale
        discharge_s *= scale
    effective_charge_c_rate = _SECONDS_PER_HOUR / charge_s
    effective_discharge_c_rate = _SECONDS_PER_HOUR / discharge_s

    times = [start_time_s]
    soc = [0.0]
    cursor = start_time_s
    for _ in range(cycle_count):
        cursor += charge_s
        times.append(cursor)
        soc.append(1.0)
        cursor += discharge_s
        times.append(cursor)
        soc.append(0.0)
    interval_end = start_time_s + elapsed_time_s
    if interval_end - cursor > _TIME_TOLERANCE_S:
        times.append(interval_end)
        soc.append(0.0)
    elif interval_end < cursor:
        times[-1] = interval_end
    time_array = np.asarray(times, dtype=np.float64)
    soc_array = np.asarray(soc, dtype=np.float64)
    temperature = np.full(time_array.shape, temperature_c, dtype=np.float64)
    return (
        time_array,
        soc_array,
        temperature,
        effective_charge_c_rate,
        effective_discharge_c_rate,
    )


def _metric(
    points: Sequence[Blast280AhValidationPoint],
    *,
    group_kind: str,
    group_value: str,
) -> Blast280AhValidationMetric:
    residuals = np.asarray([point.residual for point in points], dtype=np.float64)
    return Blast280AhValidationMetric(
        group_kind=cast(Any, group_kind),
        group_value=group_value,
        observation_count=len(points),
        mae_soh=float(np.mean(np.abs(residuals))),
        rmse_soh=float(np.sqrt(np.mean(np.square(residuals)))),
        bias_soh=float(np.mean(residuals)),
    )


def _group_metrics(
    points: Sequence[Blast280AhValidationPoint],
) -> tuple[Blast280AhValidationMetric, ...]:
    if not points:
        return ()
    metrics = [_metric(points, group_kind="overall", group_value="all")]
    groupers: tuple[
        tuple[str, Callable[[Blast280AhValidationPoint], str]], ...
    ] = (
        ("vendor", lambda point: point.vendor),
        ("cell_id", lambda point: point.cell_id),
        ("support_status", lambda point: point.support_status),
    )
    for kind, getter in groupers:
        grouped: dict[str, list[Blast280AhValidationPoint]] = defaultdict(list)
        for point in points:
            grouped[getter(point)].append(point)
        for value, selected in sorted(grouped.items()):
            metrics.append(_metric(selected, group_kind=kind, group_value=value))
    return tuple(metrics)


def evaluate_lfp_280ah_capacity_series(
    series: Sequence[DodCapacityCellSeries],
    *,
    bundle_sha256: str,
) -> Blast280AhValidationEvaluation:
    """Compare the fixed 250 Ah reference model only over observed 280 Ah cycles."""

    catalog = load_packaged_blast_route_catalog()
    route = next(item for item in catalog.routes if item.model_class == _MODEL_CLASS)
    points: list[Blast280AhValidationPoint] = []
    seen_cells: set[str] = set()
    ordered_series = sorted(series, key=lambda item: item.cell_id)
    for cell in ordered_series:
        if cell.cell_id in seen_cells:
            raise ValueError("280Ah validation cell IDs must be unique")
        seen_cells.add(cell.cell_id)
        if cell.dod_fraction != 1.0 or not math.isclose(cell.c_rate, 0.5):
            raise ValueError("280Ah reference check only supports reviewed 100% DoD at 0.5C")
        battery = cast(
            _BlastBattery,
            Lfp_Gr_250AhPrismatic(),
        )
        first_cycle = cell.observations[0].cycle_number
        previous = cell.observations[0]
        current_time_s = 0.0
        effective_charge_c_rate = cell.c_rate
        effective_discharge_c_rate = cell.c_rate
        for index, observation in enumerate(cell.observations):
            if index:
                cycle_count = observation.cycle_number - previous.cycle_number
                elapsed_time_s = (
                    observation.elapsed_days - previous.elapsed_days
                ) * _SECONDS_PER_DAY
                (
                    time_s,
                    soc,
                    temperature,
                    effective_charge_c_rate,
                    effective_discharge_c_rate,
                ) = _cycle_interval_profile(
                    start_time_s=current_time_s,
                    elapsed_time_s=elapsed_time_s,
                    cycle_count=cycle_count,
                    charge_c_rate=cell.c_rate,
                    discharge_c_rate=cell.c_rate,
                    temperature_c=observation.mean_temperature_c,
                )
                battery.update_battery_state(time_s, soc, temperature)
                current_time_s += elapsed_time_s
            predicted = _last_q(battery)
            observed = observation.relative_capacity_ratio
            residual = predicted - observed
            support_status, outside = _route_support(
                temperature_c=observation.mean_temperature_c,
                charge_c_rate=effective_charge_c_rate,
                discharge_c_rate=effective_discharge_c_rate,
                route=route,
            )
            points.append(
                Blast280AhValidationPoint(
                    vendor=cell.vendor,
                    cell_id=cell.cell_id,
                    cycle_number=observation.cycle_number,
                    scheduled_efc=float(observation.cycle_number - first_cycle),
                    elapsed_days=observation.elapsed_days,
                    observed_capacity_ah=observation.discharge_capacity_ah,
                    observed_soh=observed,
                    predicted_soh=predicted,
                    residual=residual,
                    absolute_error=abs(residual),
                    temperature_c=observation.mean_temperature_c,
                    charge_c_rate=effective_charge_c_rate,
                    discharge_c_rate=effective_discharge_c_rate,
                    support_status=cast(Any, support_status),
                    out_of_range_fields=outside,
                    route_id=route.route_id,
                    bundle_sha256=bundle_sha256,
                )
            )
            previous = observation
    if not points:
        raise ValueError("280Ah reference check requires at least one cell series")
    return Blast280AhValidationEvaluation(
        predictions=tuple(points),
        metrics=_group_metrics(points),
        methodology={
            "evaluation_kind": "EXTERNAL_SCALE_REFERENCE_CHECK",
            "comparison_scope": "OBSERVED_CYCLE_RANGE_ONLY",
            "parameter_fitting_performed": False,
            "independent_holdout": True,
            "observed_soh": "capacity_ah / first_valid_cycle_capacity_ah",
            "initial_alignment": "FIRST_RETAINED_OBSERVATION_ANCHORED_TO_ONE",
            "time_alignment": (
                "Complete 100% DoD 0.5C charge/discharge blocks use source elapsed "
                "time; residual interval time is modeled as rest at zero SOC."
            ),
            "model_capacity_reference_ah": route.nominal_capacity_reference_ah,
            "dataset_nominal_capacity_ah": ordered_series[0].nominal_capacity_ah,
            "source_protocol_c_rate": 0.5,
            "effective_c_rate": (
                "Nominal 0.5C with residual rest when source intervals are at least four "
                "hours per cycle; otherwise charge and discharge duration are scaled to "
                "the direct source elapsed time and the resulting effective rates are "
                "reported and checked against the route manifest."
            ),
        },
        warnings=(
            "REFERENCE_250AH_MODEL_NOT_280AH_PRODUCT_MODEL",
            "NO_CAPACITY_SCALING_OR_PARAMETER_FITTING",
            "FIRST_VALID_CYCLE_CAPACITY_NORMALIZATION",
            "EXTERNAL_OBSERVED_RANGE_CHECK_ONLY",
            "NOT_15_TO_25_YEAR_VALIDATION",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_bundle(bundle_dir: Path) -> tuple[tuple[DodCapacityCellSeries, ...], str]:
    root = Path(bundle_dir).resolve(strict=True)
    manifest_path = root / "manifest.json"
    series_path = root / "series.json"
    committed_path = root / "COMMITTED"
    if not all(path.is_file() for path in (manifest_path, series_path, committed_path)):
        raise ValueError("280Ah validation bundle is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    bundle_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != bundle_sha:
        raise ValueError("280Ah validation bundle COMMITTED hash is invalid")
    try:
        manifest = json.loads(manifest_bytes)
        payload = json.loads(series_path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("280Ah validation bundle JSON is invalid") from exc
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("280Ah validation bundle artifacts are invalid")
    for descriptor in artifacts:
        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("relative_path"), str
        ):
            raise ValueError("280Ah validation bundle artifact descriptor is invalid")
        path = root / descriptor["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("size_bytes")
            or _sha256(path) != descriptor.get("sha256")
        ):
            raise ValueError("280Ah validation bundle artifact hash is invalid")
    raw_series = payload.get("series") if isinstance(payload, dict) else None
    if not isinstance(raw_series, list):
        raise ValueError("280Ah validation series payload is invalid")
    parsed = tuple(DodCapacityCellSeries.model_validate(item) for item in raw_series)
    if manifest.get("cell_count") != len(parsed):
        raise ValueError("280Ah validation cell count does not match the manifest")
    return parsed, bundle_sha


def _write_models_csv(path: Path, rows: Sequence[ContractModel]) -> None:
    if not rows:
        raise ValueError("280Ah validation CSV requires at least one row")
    dumped = [row.model_dump(mode="json") for row in rows]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dumped[0]), lineterminator="\n")
        writer.writeheader()
        for row in dumped:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
                        if isinstance(value, (list, tuple, dict))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _artifact(path: Path) -> dict[str, object]:
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _existing_run(
    output: Path,
    *,
    bundle_sha256: str,
    code_revision: str,
) -> Blast280AhValidationRunResult | None:
    if not output.exists():
        return None
    manifest_path = output / "manifest.json"
    committed_path = output / "COMMITTED"
    if not manifest_path.is_file() or not committed_path.is_file():
        raise ValueError("existing 280Ah validation result is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != result_sha:
        raise ValueError("existing 280Ah validation COMMITTED hash is invalid")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing 280Ah validation manifest is invalid") from exc
    if (
        manifest.get("implementation_version") != BLAST_280AH_VALIDATION_VERSION
        or manifest.get("input_bundle_sha256") != bundle_sha256
        or manifest.get("code_revision") != code_revision
    ):
        raise ValueError("existing 280Ah validation result does not match this run")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("existing 280Ah validation artifacts are invalid")
    for descriptor in artifacts:
        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("relative_path"), str
        ):
            raise ValueError("existing 280Ah validation artifact descriptor is invalid")
        path = output / descriptor["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != descriptor.get("size_bytes")
            or _sha256(path) != descriptor.get("sha256")
        ):
            raise ValueError("existing 280Ah validation artifact hash is invalid")
    return Blast280AhValidationRunResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_dir=str(output),
        result_sha256=result_sha,
        prediction_count=int(manifest["prediction_count"]),
        metric_count=int(manifest["metric_count"]),
    )


def run_lfp_280ah_reference_validation(
    bundle_dir: Path,
    *,
    output_dir: Path,
    code_revision: str,
) -> Blast280AhValidationRunResult:
    """Run and atomically publish the fixed-parameter observed-range check."""

    revision = code_revision.strip()
    if not revision:
        raise ValueError("code_revision must not be blank")
    series, bundle_sha = _load_verified_bundle(bundle_dir)
    output = Path(output_dir).resolve()
    existing = _existing_run(
        output,
        bundle_sha256=bundle_sha,
        code_revision=revision,
    )
    if existing is not None:
        return existing
    evaluation = evaluate_lfp_280ah_capacity_series(series, bundle_sha256=bundle_sha)
    catalog = load_packaged_blast_route_catalog()
    route = next(item for item in catalog.routes if item.model_class == _MODEL_CLASS)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise ValueError("280Ah validation temporary result directory already exists")
    temporary.mkdir(parents=True)
    try:
        config = {
            "schema_version": "blast-280ah-validation-config-v1",
            "implementation_version": BLAST_280AH_VALIDATION_VERSION,
            "code_revision": revision,
            "input_bundle_sha256": bundle_sha,
            "route_id": route.route_id,
            "route_version": route.route_version,
            "model_class": route.model_class,
            "upstream_commit": catalog.upstream_commit,
            "activation_status": route.activation_status,
            "parameter_fitting_performed": False,
        }
        summary = {
            "schema_version": "blast-280ah-validation-summary-v1",
            "prediction_count": len(evaluation.predictions),
            "metric_count": len(evaluation.metrics),
            "methodology": evaluation.methodology,
            "warnings": list(evaluation.warnings),
            "metrics": [item.model_dump(mode="json") for item in evaluation.metrics],
        }
        (temporary / "config.json").write_bytes(canonical_json_bytes(config))
        (temporary / "summary.json").write_bytes(canonical_json_bytes(summary))
        _write_models_csv(temporary / "predictions.csv", evaluation.predictions)
        _write_models_csv(temporary / "metrics.csv", evaluation.metrics)
        names = ("config.json", "summary.json", "predictions.csv", "metrics.csv")
        manifest = {
            "schema_version": "blast-280ah-validation-result-v1",
            "implementation_version": BLAST_280AH_VALIDATION_VERSION,
            "code_revision": revision,
            "input_bundle_sha256": bundle_sha,
            "prediction_count": len(evaluation.predictions),
            "metric_count": len(evaluation.metrics),
            "artifacts": [_artifact(temporary / name) for name in names],
        }
        manifest_bytes = canonical_json_bytes(manifest)
        result_sha = hashlib.sha256(manifest_bytes).hexdigest()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "COMMITTED").write_text(result_sha + "\n", encoding="ascii")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return Blast280AhValidationRunResult(
        status=DatasetBuildStatus.BUILT,
        output_dir=str(output),
        result_sha256=result_sha,
        prediction_count=len(evaluation.predictions),
        metric_count=len(evaluation.metrics),
    )


__all__ = [
    "BLAST_280AH_VALIDATION_VERSION",
    "Blast280AhValidationEvaluation",
    "Blast280AhValidationMetric",
    "Blast280AhValidationPoint",
    "Blast280AhValidationRunResult",
    "evaluate_lfp_280ah_capacity_series",
    "run_lfp_280ah_reference_validation",
]
