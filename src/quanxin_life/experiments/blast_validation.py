"""Pointwise Naumann replay for the pinned BLAST-Lite LFP reference model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from quanxin_life._vendor.blast_lite import Lfp_Gr_SonyMurata3Ah_Battery
from quanxin_life.core import DatasetBuildStatus, EvidenceLevel, canonical_json_bytes
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.scenarios import load_packaged_blast_route_catalog

VALIDATION_IMPLEMENTATION_VERSION: Final = "blast-naumann-source-replay-v1.1.0"
_SONY_MODEL_CLASS: Final = "Lfp_Gr_SonyMurata3Ah_Battery"
_SECONDS_PER_HOUR = 3600.0
_MAX_SCHEDULED_EFC_PER_CYCLE_BLOCK = 20.0
FloatArray = NDArray[np.float64]


class _BlastBattery(Protocol):
    outputs: dict[str, FloatArray]
    stressors: dict[str, FloatArray]

    def update_battery_state(
        self,
        time_s: FloatArray,
        soc: FloatArray,
        temperature_c: FloatArray,
    ) -> None: ...


class BlastValidationPoint(ContractModel):
    validation_domain: Literal["calendar", "cycle"]
    series_id: str
    condition_id: str
    source_file: str
    observation_axis: Literal["storage_time_h", "equivalent_full_cycles"]
    axis_value: float = Field(ge=0, allow_inf_nan=False)
    observed_value_raw: float = Field(allow_inf_nan=False)
    observed_soh: float = Field(ge=0, allow_inf_nan=False)
    predicted_soh: float = Field(ge=0, allow_inf_nan=False)
    residual: float = Field(allow_inf_nan=False)
    absolute_error: float = Field(ge=0, allow_inf_nan=False)
    temperature_c: float = Field(allow_inf_nan=False)
    mean_soc: float = Field(ge=0, le=1, allow_inf_nan=False)
    dod: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    charge_c_rate: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    discharge_c_rate: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    support_status: Literal[
        "OBSERVED_SOURCE_RANGE",
        "SUPPORTED_BY_ROUTE_MANIFEST",
        "OUTSIDE_ROUTE_MANIFEST",
    ]
    out_of_range_fields: tuple[str, ...] = ()
    observed_evidence_level: Literal[EvidenceLevel.DATA_DIRECT] = (
        EvidenceLevel.DATA_DIRECT
    )
    prediction_evidence_level: Literal[EvidenceLevel.PHYSICS_REFERENCE] = (
        EvidenceLevel.PHYSICS_REFERENCE
    )
    model_class: Literal["Lfp_Gr_SonyMurata3Ah_Battery"] = _SONY_MODEL_CLASS
    route_id: str
    bundle_sha256: Sha256


class BlastValidationMetric(ContractModel):
    group_kind: Literal[
        "overall",
        "support_status",
        "condition_id",
        "temperature_c",
        "dod",
        "c_rate_pair",
        "mean_soc",
    ]
    group_value: str
    validation_domain: Literal["calendar", "cycle", "combined"]
    evaluation_mode: Literal[
        "UPSTREAM_SOURCE_REPLAY",
        "FIXED_UPSTREAM_PARAMETER_GROUP_DIAGNOSTIC",
        "FIXED_UPSTREAM_PARAMETER_LEAVE_ONE_CONDITION_DIAGNOSTIC",
    ]
    observation_count: int = Field(gt=0)
    mae_soh: float = Field(ge=0, allow_inf_nan=False)
    rmse_soh: float = Field(ge=0, allow_inf_nan=False)
    bias_soh: float = Field(allow_inf_nan=False)


class BlastValidationEvaluation(ContractModel):
    implementation_version: Literal["blast-naumann-source-replay-v1.1.0"] = (
        VALIDATION_IMPLEMENTATION_VERSION
    )
    predictions: tuple[BlastValidationPoint, ...]
    metrics: tuple[BlastValidationMetric, ...]
    methodology: dict[str, object]
    warnings: tuple[str, ...]


class BlastValidationRunResult(ContractModel):
    status: DatasetBuildStatus
    output_dir: str = Field(min_length=1)
    result_sha256: Sha256
    prediction_count: int = Field(gt=0)
    metric_count: int = Field(gt=0)


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value.strip()


def _last_q(battery: _BlastBattery) -> float:
    values = battery.outputs.get("q")
    if values is None or len(values) == 0:
        raise ValueError("BLAST validation model did not emit q")
    value = float(values[-1])
    if not math.isfinite(value) or value < 0:
        raise ValueError("BLAST validation model emitted non-finite or negative q")
    return value


def _series_id(row: Mapping[str, object]) -> str:
    return (
        f"{_text(row.get('source_file'), field='source_file')}::"
        f"{_text(row.get('condition_id'), field='condition_id')}"
    )


def _new_battery() -> _BlastBattery:
    return cast(_BlastBattery, Lfp_Gr_SonyMurata3Ah_Battery())


def _calendar_points(
    rows: Sequence[Mapping[str, object]],
    *,
    bundle_sha256: str,
    route_id: str,
) -> list[BlastValidationPoint]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_series_id(row)].append(row)

    points: list[BlastValidationPoint] = []
    for series_id in sorted(grouped):
        series = sorted(
            grouped[series_id],
            key=lambda item: _finite(item.get("storage_time_h"), field="storage_time_h"),
        )
        first = series[0]
        baseline_capacity = _finite(first.get("capacity_ah"), field="capacity_ah")
        if baseline_capacity <= 0:
            raise ValueError("calendar baseline capacity must be positive")
        temperature_c = _finite(first.get("temperature_c"), field="temperature_c")
        mean_soc = _finite(first.get("mean_soc"), field="mean_soc")
        if not 0 <= mean_soc <= 1:
            raise ValueError("calendar mean_soc must be between zero and one")
        battery = _new_battery()
        previous_time_h = 0.0
        for row in series:
            time_h = _finite(row.get("storage_time_h"), field="storage_time_h")
            if time_h < previous_time_h:
                raise ValueError("calendar storage time must be nondecreasing")
            if time_h > previous_time_h:
                time_s = np.asarray(
                    [previous_time_h * _SECONDS_PER_HOUR, time_h * _SECONDS_PER_HOUR],
                    dtype=np.float64,
                )
                soc = np.asarray([mean_soc, mean_soc], dtype=np.float64)
                temperature = np.asarray(
                    [temperature_c, temperature_c], dtype=np.float64
                )
                battery.update_battery_state(time_s, soc, temperature)
            predicted = _last_q(battery)
            raw = _finite(row.get("capacity_ah"), field="capacity_ah")
            observed = raw / baseline_capacity
            residual = predicted - observed
            points.append(
                BlastValidationPoint(
                    validation_domain="calendar",
                    series_id=series_id,
                    condition_id=_text(row.get("condition_id"), field="condition_id"),
                    source_file=_text(row.get("source_file"), field="source_file"),
                    observation_axis="storage_time_h",
                    axis_value=time_h,
                    observed_value_raw=raw,
                    observed_soh=observed,
                    predicted_soh=predicted,
                    residual=residual,
                    absolute_error=abs(residual),
                    temperature_c=temperature_c,
                    mean_soc=mean_soc,
                    support_status="OBSERVED_SOURCE_RANGE",
                    route_id=route_id,
                    bundle_sha256=bundle_sha256,
                )
            )
            previous_time_h = time_h
    return points


def _full_cycle_block_profile(
    *,
    start_time_s: float,
    cycle_count: int,
    mean_soc: float,
    dod: float,
    charge_c_rate: float,
    discharge_c_rate: float,
    temperature_c: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if cycle_count < 1:
        raise ValueError("cycle block requires at least one complete cycle")
    lower = mean_soc - dod / 2
    upper = mean_soc + dod / 2
    if lower < 0 or upper > 1:
        raise ValueError("cycle mean SOC and DoD exceed physical SOC bounds")

    times = [start_time_s]
    soc = [lower]
    elapsed_s = start_time_s
    for _ in range(cycle_count):
        elapsed_s += dod / charge_c_rate * _SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(upper)
        elapsed_s += dod / discharge_c_rate * _SECONDS_PER_HOUR
        times.append(elapsed_s)
        soc.append(lower)
    time_array = np.asarray(times, dtype=np.float64)
    soc_array = np.asarray(soc, dtype=np.float64)
    temperature = np.full(time_array.shape, temperature_c, dtype=np.float64)
    return time_array, soc_array, temperature


def _cycle_support(
    row: Mapping[str, object],
    *,
    route: Any,
) -> tuple[str, tuple[str, ...]]:
    outside: list[str] = []
    temperature_c = _finite(row.get("temperature_c"), field="temperature_c")
    mean_soc = _finite(row.get("mean_soc"), field="mean_soc")
    dod = _finite(row.get("dod"), field="dod")
    charge = _finite(row.get("charge_c_rate"), field="charge_c_rate")
    discharge = _finite(row.get("discharge_c_rate"), field="discharge_c_rate")
    ranges = route.experimental_range
    if not ranges.cycling_temperature_c[0] <= temperature_c <= ranges.cycling_temperature_c[1]:
        outside.append("temperature_c")
    if not ranges.dod[0] <= dod <= ranges.dod[1]:
        outside.append("dod")
    if not ranges.soc[0] <= mean_soc <= ranges.soc[1]:
        outside.append("mean_soc")
    if charge > ranges.max_rate_charge:
        outside.append("charge_c_rate")
    if discharge > ranges.max_rate_discharge:
        outside.append("discharge_c_rate")
    return (
        "OUTSIDE_ROUTE_MANIFEST" if outside else "SUPPORTED_BY_ROUTE_MANIFEST",
        tuple(outside),
    )


def _cycle_points(
    rows: Sequence[Mapping[str, object]],
    *,
    bundle_sha256: str,
    route: Any,
) -> list[BlastValidationPoint]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_series_id(row)].append(row)

    points: list[BlastValidationPoint] = []
    for series_id in sorted(grouped):
        series = sorted(
            grouped[series_id],
            key=lambda item: _finite(
                item.get("observation_value"), field="observation_value"
            ),
        )
        first = series[0]
        temperature_c = _finite(first.get("temperature_c"), field="temperature_c")
        mean_soc = _finite(first.get("mean_soc"), field="mean_soc")
        dod = _finite(first.get("dod"), field="dod")
        charge = _finite(first.get("charge_c_rate"), field="charge_c_rate")
        discharge = _finite(first.get("discharge_c_rate"), field="discharge_c_rate")
        if dod <= 0 or charge <= 0 or discharge <= 0:
            raise ValueError("cycle DoD and C-rates must be positive")
        battery = _new_battery()
        current_time_s = 0.0
        model_efc_axis = [0.0]
        predicted_soh_axis = [_last_q(battery)]
        maximum_target_efc = max(
            _finite(row.get("observation_value"), field="observation_value")
            for row in series
        )
        cycles_per_block = max(
            1,
            math.floor(_MAX_SCHEDULED_EFC_PER_CYCLE_BLOCK / dod),
        )
        while model_efc_axis[-1] < maximum_target_efc:
            profile = _full_cycle_block_profile(
                start_time_s=current_time_s,
                cycle_count=cycles_per_block,
                mean_soc=mean_soc,
                dod=dod,
                charge_c_rate=charge,
                discharge_c_rate=discharge,
                temperature_c=temperature_c,
            )
            battery.update_battery_state(*profile)
            current_time_s = float(profile[0][-1])
            predicted = _last_q(battery)
            model_efc_values = battery.stressors.get("efc")
            if model_efc_values is None or len(model_efc_values) == 0:
                raise ValueError("BLAST validation model did not emit cumulative EFC")
            model_efc = float(model_efc_values[-1])
            if not math.isfinite(model_efc) or model_efc <= model_efc_axis[-1]:
                raise ValueError("BLAST validation model emitted invalid cumulative EFC")
            model_efc_axis.append(model_efc)
            predicted_soh_axis.append(predicted)

        previous_target_efc = 0.0
        for row in series:
            target_efc = _finite(row.get("observation_value"), field="observation_value")
            if target_efc < previous_target_efc:
                raise ValueError("cycle FEC must be nondecreasing")
            predicted = float(
                np.interp(target_efc, model_efc_axis, predicted_soh_axis)
            )
            if not math.isfinite(predicted) or predicted < 0:
                raise ValueError("BLAST validation interpolation emitted invalid q")
            observed = _finite(row.get("metric_value"), field="metric_value")
            residual = predicted - observed
            support_status, out_of_range = _cycle_support(row, route=route)
            points.append(
                BlastValidationPoint(
                    validation_domain="cycle",
                    series_id=series_id,
                    condition_id=_text(row.get("condition_id"), field="condition_id"),
                    source_file=_text(row.get("source_file"), field="source_file"),
                    observation_axis="equivalent_full_cycles",
                    axis_value=target_efc,
                    observed_value_raw=observed,
                    observed_soh=observed,
                    predicted_soh=predicted,
                    residual=residual,
                    absolute_error=abs(residual),
                    temperature_c=temperature_c,
                    mean_soc=mean_soc,
                    dod=dod,
                    charge_c_rate=charge,
                    discharge_c_rate=discharge,
                    support_status=cast(Any, support_status),
                    out_of_range_fields=out_of_range,
                    route_id=route.route_id,
                    bundle_sha256=bundle_sha256,
                )
            )
            previous_target_efc = target_efc
    return points


def _format_group_value(value: object) -> str:
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def _metric(
    points: Sequence[BlastValidationPoint],
    *,
    group_kind: str,
    group_value: str,
    domain: str,
    evaluation_mode: str,
) -> BlastValidationMetric:
    residuals = np.asarray([point.residual for point in points], dtype=np.float64)
    return BlastValidationMetric(
        group_kind=cast(Any, group_kind),
        group_value=group_value,
        validation_domain=cast(Any, domain),
        evaluation_mode=cast(Any, evaluation_mode),
        observation_count=len(points),
        mae_soh=float(np.mean(np.abs(residuals))),
        rmse_soh=float(np.sqrt(np.mean(np.square(residuals)))),
        bias_soh=float(np.mean(residuals)),
    )


def _group_metrics(points: Sequence[BlastValidationPoint]) -> list[BlastValidationMetric]:
    metrics: list[BlastValidationMetric] = []
    for domain in ("calendar", "cycle"):
        selected = [point for point in points if point.validation_domain == domain]
        if selected:
            metrics.append(
                _metric(
                    selected,
                    group_kind="overall",
                    group_value=domain,
                    domain=domain,
                    evaluation_mode="UPSTREAM_SOURCE_REPLAY",
                )
            )
    if points:
        metrics.append(
            _metric(
                points,
                group_kind="overall",
                group_value="combined",
                domain="combined",
                evaluation_mode="UPSTREAM_SOURCE_REPLAY",
            )
        )

    groupers: tuple[tuple[str, Any], ...] = (
        ("support_status", lambda point: point.support_status),
        ("condition_id", lambda point: point.condition_id),
        ("temperature_c", lambda point: point.temperature_c),
        ("mean_soc", lambda point: point.mean_soc),
        ("dod", lambda point: point.dod),
        (
            "c_rate_pair",
            lambda point: (
                None
                if point.charge_c_rate is None or point.discharge_c_rate is None
                else f"{format(point.charge_c_rate, 'g')}/{format(point.discharge_c_rate, 'g')}"
            ),
        ),
    )
    for kind, getter in groupers:
        grouped: dict[tuple[str, object], list[BlastValidationPoint]] = defaultdict(list)
        for point in points:
            value = getter(point)
            if value is not None:
                grouped[(point.validation_domain, value)].append(point)
        for (domain, value), selected in sorted(
            grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))
        ):
            metrics.append(
                _metric(
                    selected,
                    group_kind=kind,
                    group_value=_format_group_value(value),
                    domain=domain,
                    evaluation_mode="FIXED_UPSTREAM_PARAMETER_GROUP_DIAGNOSTIC",
                )
            )

    leave_one_groupers = tuple(
        (kind, getter)
        for kind, getter in groupers
        if kind in {"temperature_c", "dod", "c_rate_pair"}
    )
    for kind, getter in leave_one_groupers:
        grouped_by_domain: dict[
            str, dict[object, list[BlastValidationPoint]]
        ] = defaultdict(lambda: defaultdict(list))
        for point in points:
            value = getter(point)
            if value is not None:
                grouped_by_domain[point.validation_domain][value].append(point)
        for domain, held_groups in sorted(grouped_by_domain.items()):
            if len(held_groups) < 2:
                continue
            for held_value, held_points in sorted(
                held_groups.items(), key=lambda item: str(item[0])
            ):
                metrics.append(
                    _metric(
                        held_points,
                        group_kind=kind,
                        group_value=_format_group_value(held_value),
                        domain=domain,
                        evaluation_mode=(
                            "FIXED_UPSTREAM_PARAMETER_LEAVE_ONE_CONDITION_DIAGNOSTIC"
                        ),
                    )
                )
    return metrics


def evaluate_naumann_observations(
    observations: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    bundle_sha256: str,
) -> BlastValidationEvaluation:
    """Replay fixed upstream parameters against reviewed Naumann observations."""

    catalog = load_packaged_blast_route_catalog()
    route = next(item for item in catalog.routes if item.model_class == _SONY_MODEL_CLASS)
    calendar_rows = observations.get("calendar", ())
    cycle_rows = observations.get("cycle", ())
    points = [
        *_calendar_points(
            calendar_rows,
            bundle_sha256=bundle_sha256,
            route_id=route.route_id,
        ),
        *_cycle_points(
            cycle_rows,
            bundle_sha256=bundle_sha256,
            route=route,
        ),
    ]
    return BlastValidationEvaluation(
        predictions=tuple(points),
        metrics=tuple(_group_metrics(points)),
        methodology={
            "evaluation_kind": "UPSTREAM_SOURCE_REPLAY",
            "parameter_fitting_performed": False,
            "independent_holdout": False,
            "calendar_observed_soh": "capacity_ah / condition_initial_capacity_ah",
            "cycle_observed_soh": "published_relative_capacity_ratio",
            "cycle_alignment": (
                "Complete-cycle BLAST blocks followed by interpolation on the model's "
                "cumulative effective-FEC axis."
            ),
            "maximum_scheduled_efc_per_cycle_block": (
                _MAX_SCHEDULED_EFC_PER_CYCLE_BLOCK
            ),
            "group_diagnostics": (
                "Fixed upstream parameters evaluated by temperature, DoD, C-rate, "
                "mean SOC, condition, and route support status without refitting."
            ),
            "leave_one_condition_axes": (
                "temperature_c",
                "dod",
                "c_rate_pair",
            ),
            "leave_one_condition_protocol": (
                "For each axis and validation domain with at least two observed values, "
                "report the held-value errors from the already fixed upstream model."
            ),
            "leave_one_condition_parameter_refit": False,
            "leave_one_condition_independent_training_holdout": False,
        },
        warnings=(
            "UPSTREAM_IDENTIFICATION_DATA_REPLAY_NOT_INDEPENDENT_HOLDOUT",
            "FIXED_UPSTREAM_PARAMETERS_NOT_PROJECT_TRAINING",
            "LEAVE_ONE_CONDITION_DIAGNOSTIC_USES_NO_PARAMETER_REFIT",
            "OUT_OF_ROUTE_POINTS_REPORTED_SEPARATELY",
            "RESULTS_DO_NOT_VALIDATE_PRODUCT_SPECIFIC_15_TO_25_YEAR_LIFE",
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_verified_bundle(bundle_dir: Path) -> tuple[dict[str, Any], str]:
    root = Path(bundle_dir).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("BLAST validation bundle must be a directory")
    manifest_path = root / "manifest.json"
    observations_path = root / "observations.json"
    committed_path = root / "COMMITTED"
    if not all(path.is_file() for path in (manifest_path, observations_path, committed_path)):
        raise ValueError("BLAST validation bundle is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    bundle_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != bundle_sha:
        raise ValueError("BLAST validation bundle COMMITTED hash is invalid")
    try:
        manifest = json.loads(manifest_bytes)
        observations = json.loads(observations_path.read_bytes())
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("BLAST validation bundle JSON is invalid") from exc
    if not isinstance(manifest, dict) or not isinstance(observations, dict):
        raise ValueError("BLAST validation bundle JSON roots must be objects")
    if manifest.get("observations_sha256") != _sha256(observations_path):
        raise ValueError("BLAST validation observations hash is invalid")
    counts = manifest.get("observation_counts")
    if not isinstance(counts, dict):
        raise ValueError("BLAST validation bundle counts are missing")
    calendar = observations.get("calendar")
    cycle = observations.get("cycle")
    if not isinstance(calendar, list) or not isinstance(cycle, list):
        raise ValueError("BLAST validation observations must contain calendar and cycle lists")
    if counts.get("calendar") != len(calendar) or counts.get("cycle") != len(cycle):
        raise ValueError("BLAST validation observation counts do not match the manifest")
    return observations, bundle_sha


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return value


def _write_models_csv(path: Path, rows: Sequence[ContractModel]) -> None:
    if not rows:
        raise ValueError("validation CSV requires at least one row")
    dumped = [row.model_dump(mode="json") for row in rows]
    fieldnames = list(dumped[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in dumped:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _artifact_descriptor(path: Path) -> dict[str, object]:
    return {
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _existing_run(
    output_dir: Path,
    *,
    input_bundle_sha256: str,
    code_revision: str,
) -> BlastValidationRunResult | None:
    if not output_dir.exists():
        return None
    manifest_path = output_dir / "manifest.json"
    committed_path = output_dir / "COMMITTED"
    if not manifest_path.is_file() or not committed_path.is_file():
        raise ValueError("existing BLAST validation result is incomplete")
    manifest_bytes = manifest_path.read_bytes()
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if committed_path.read_text(encoding="ascii").strip() != result_sha:
        raise ValueError("existing BLAST validation result COMMITTED hash is invalid")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("existing BLAST validation result manifest is invalid") from exc
    if (
        manifest.get("implementation_version") != VALIDATION_IMPLEMENTATION_VERSION
        or manifest.get("input_bundle_sha256") != input_bundle_sha256
        or manifest.get("code_revision") != code_revision
    ):
        raise ValueError("existing BLAST validation result does not match this run")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("existing BLAST validation result artifacts are invalid")
    for descriptor in artifacts:
        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("relative_path"), str
        ):
            raise ValueError("existing BLAST validation artifact descriptor is invalid")
        path = output_dir / descriptor["relative_path"]
        if (
            not path.is_file()
            or descriptor.get("size_bytes") != path.stat().st_size
            or descriptor.get("sha256") != _sha256(path)
        ):
            raise ValueError("existing BLAST validation artifact hash is invalid")
    return BlastValidationRunResult(
        status=DatasetBuildStatus.SKIPPED_VALID,
        output_dir=str(output_dir),
        result_sha256=result_sha,
        prediction_count=int(manifest["prediction_count"]),
        metric_count=int(manifest["metric_count"]),
    )


def run_naumann_blast_validation(
    bundle_dir: Path,
    *,
    output_dir: Path,
    code_revision: str,
) -> BlastValidationRunResult:
    """Run and atomically publish a fixed-parameter Naumann source replay."""

    normalized_revision = code_revision.strip()
    if not normalized_revision:
        raise ValueError("code_revision must not be blank")
    observations, bundle_sha = _load_verified_bundle(bundle_dir)
    output = Path(output_dir).resolve()
    existing = _existing_run(
        output,
        input_bundle_sha256=bundle_sha,
        code_revision=normalized_revision,
    )
    if existing is not None:
        return existing

    evaluation = evaluate_naumann_observations(
        observations,
        bundle_sha256=bundle_sha,
    )
    if not evaluation.predictions or not evaluation.metrics:
        raise ValueError("BLAST validation produced no reportable results")
    catalog = load_packaged_blast_route_catalog()
    route = next(item for item in catalog.routes if item.model_class == _SONY_MODEL_CLASS)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        raise ValueError("BLAST validation temporary result directory already exists")
    temporary.mkdir(parents=True)

    config = {
        "schema_version": "blast-naumann-validation-config-v1",
        "implementation_version": VALIDATION_IMPLEMENTATION_VERSION,
        "code_revision": normalized_revision,
        "input_bundle_sha256": bundle_sha,
        "route_id": route.route_id,
        "route_version": route.route_version,
        "model_class": route.model_class,
        "upstream_repository": catalog.upstream_repository,
        "upstream_commit": catalog.upstream_commit,
        "parameter_fitting_performed": False,
        "activation_status": route.activation_status,
    }
    summary = {
        "schema_version": "blast-naumann-validation-summary-v1",
        "implementation_version": VALIDATION_IMPLEMENTATION_VERSION,
        "prediction_count": len(evaluation.predictions),
        "metric_count": len(evaluation.metrics),
        "methodology": evaluation.methodology,
        "warnings": list(evaluation.warnings),
        "overall_metrics": [
            metric.model_dump(mode="json")
            for metric in evaluation.metrics
            if metric.group_kind == "overall"
        ],
    }
    (temporary / "config.json").write_bytes(canonical_json_bytes(config))
    (temporary / "summary.json").write_bytes(canonical_json_bytes(summary))
    _write_models_csv(temporary / "predictions.csv", evaluation.predictions)
    _write_models_csv(temporary / "metrics.csv", evaluation.metrics)
    artifact_names = ("config.json", "summary.json", "predictions.csv", "metrics.csv")
    manifest = {
        "schema_version": "blast-naumann-validation-result-v1",
        "implementation_version": VALIDATION_IMPLEMENTATION_VERSION,
        "code_revision": normalized_revision,
        "input_bundle_sha256": bundle_sha,
        "prediction_count": len(evaluation.predictions),
        "metric_count": len(evaluation.metrics),
        "artifacts": [
            _artifact_descriptor(temporary / name) for name in artifact_names
        ],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    result_sha = hashlib.sha256(manifest_bytes).hexdigest()
    (temporary / "manifest.json").write_bytes(manifest_bytes)
    (temporary / "COMMITTED").write_text(result_sha + "\n", encoding="ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(output)
    return BlastValidationRunResult(
        status=DatasetBuildStatus.BUILT,
        output_dir=str(output),
        result_sha256=result_sha,
        prediction_count=len(evaluation.predictions),
        metric_count=len(evaluation.metrics),
    )


__all__ = [
    "VALIDATION_IMPLEMENTATION_VERSION",
    "BlastValidationEvaluation",
    "BlastValidationMetric",
    "BlastValidationPoint",
    "BlastValidationRunResult",
    "evaluate_naumann_observations",
    "run_naumann_blast_validation",
]
