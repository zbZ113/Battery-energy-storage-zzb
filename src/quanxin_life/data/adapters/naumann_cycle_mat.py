"""Explicit MATLAB-matrix adapter for Naumann cycle-ageing condition data.

The source files contain reviewed condition-level matrices rather than raw
per-cell cycle traces.  This module reads only the variable names declared by
the caller's immutable layout after provenance verification, and emits
condition observations for GP and active-experiment workflows.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.manifest import RawFileManifest, Sha256, verify_raw_file
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalogEntry

_ADAPTER_VERSION = "naumann-cycle-mat-condition-v1.0.0"
_DATA_DEPENDENCY_ERROR = "Naumann cycle loading requires the 'data' optional dependencies"


class CycleConditionColumn(ContractModel):
    """A manually reviewed matrix column and its physical operating condition."""

    column_index: int = Field(ge=0)
    expected_legend: str = Field(min_length=1)
    condition_id: str = Field(min_length=1)
    temperature_c: float = Field(ge=-100, le=200, allow_inf_nan=False)
    mean_soc: float = Field(ge=0, le=1, allow_inf_nan=False)
    dod: float = Field(gt=0, le=1, allow_inf_nan=False)
    charge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    discharge_c_rate: float = Field(gt=0, allow_inf_nan=False)


class NaumannCycleMatrixLayout(ContractModel):
    """Immutable declaration of exactly which MATLAB matrices may be read."""

    layout_version: str = Field(min_length=1)
    x_axis_variable: str = Field(min_length=1)
    y_axis_variable: str = Field(min_length=1)
    legend_variable: str = Field(min_length=1)
    observation_axis: Literal["equivalent_full_cycles", "time_h"]
    metric_name: Literal["capacity_ah", "relative_capacity_ratio", "resistance_ohm"]
    condition_columns: tuple[CycleConditionColumn, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def variables_and_conditions_are_unambiguous(self) -> NaumannCycleMatrixLayout:
        variables = (self.x_axis_variable, self.y_axis_variable, self.legend_variable)
        if len(variables) != len(set(variables)):
            raise ValueError("cycle MATLAB variable names must be distinct")
        column_indices = [condition.column_index for condition in self.condition_columns]
        condition_ids = [condition.condition_id for condition in self.condition_columns]
        if len(column_indices) != len(set(column_indices)):
            raise ValueError("cycle condition column indices must be unique")
        if len(condition_ids) != len(set(condition_ids)):
            raise ValueError("cycle condition identifiers must be unique")
        return self


class CycleMatrixObservation(ContractModel):
    """One direct metric observation from a reviewed cycle-ageing matrix."""

    dataset_id: Literal["NAUMANN_CYCLE"] = "NAUMANN_CYCLE"
    condition_id: str = Field(min_length=1)
    observation_axis: Literal["equivalent_full_cycles", "time_h"]
    observation_value: float = Field(ge=0, allow_inf_nan=False)
    temperature_c: float = Field(ge=-100, le=200, allow_inf_nan=False)
    mean_soc: float = Field(ge=0, le=1, allow_inf_nan=False)
    dod: float = Field(gt=0, le=1, allow_inf_nan=False)
    charge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    discharge_c_rate: float = Field(gt=0, allow_inf_nan=False)
    metric_name: Literal["capacity_ah", "relative_capacity_ratio", "resistance_ohm"]
    metric_value: float = Field(ge=0, allow_inf_nan=False)
    source_file: str = Field(min_length=1)
    source_sha256: Sha256
    layout_version: str = Field(min_length=1)
    adapter_version: str = _ADAPTER_VERSION


class ReviewedAxisSelection(ContractModel):
    """Explicit fixed-axis cohort selection; it never interpolates or alters values."""

    selection_version: str = Field(min_length=1)
    target_axis_value: float = Field(gt=0, allow_inf_nan=False)
    max_absolute_deviation: float = Field(ge=0, allow_inf_nan=False)
    condition_ids: tuple[str, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def condition_ids_are_unique(self) -> ReviewedAxisSelection:
        if len(self.condition_ids) != len(set(self.condition_ids)):
            raise ValueError("reviewed axis selection condition_ids must be unique")
        return self


class ReviewedAxisSelectionResult(ContractModel):
    """Selected direct source rows plus transparent distance from the reviewed axis."""

    selection_version: str = Field(min_length=1)
    target_axis_value: float = Field(gt=0, allow_inf_nan=False)
    observations: tuple[CycleMatrixObservation, ...] = Field(min_length=2)
    absolute_deviations: tuple[float, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def observations_match_deviations(self) -> ReviewedAxisSelectionResult:
        if len(self.observations) != len(self.absolute_deviations):
            raise ValueError("selected observations and deviations must have equal length")
        return self


def load_naumann_cycle_layout(path: Path) -> NaumannCycleMatrixLayout:
    """Load a version-controlled, explicitly reviewed MATLAB layout JSON."""

    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError("cycle layout must use a .json file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"cycle layout does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("cycle layout must contain valid JSON") from exc
    return NaumannCycleMatrixLayout.model_validate(payload)


def select_reviewed_axis_observations(
    observations: tuple[CycleMatrixObservation, ...],
    *,
    selection: ReviewedAxisSelection,
) -> ReviewedAxisSelectionResult:
    """Select nearest measured rows under a versioned target and tolerance.

    The returned metric values and axes are direct source measurements.  No
    interpolation, extrapolation, averaging or missing-value fill is applied.
    """

    if not observations:
        raise ValueError("reviewed axis selection requires source observations")
    selection = ReviewedAxisSelection.model_validate(selection.model_dump(mode="json"))
    validated = tuple(
        CycleMatrixObservation.model_validate(item.model_dump(mode="json"))
        for item in observations
    )
    source_contexts = {
        (
            item.dataset_id,
            item.source_sha256,
            item.source_file,
            item.layout_version,
            item.adapter_version,
            item.observation_axis,
            item.metric_name,
        )
        for item in validated
    }
    if len(source_contexts) != 1:
        raise ValueError("reviewed axis selection cannot mix source or metric contexts")

    by_condition: dict[str, list[CycleMatrixObservation]] = {}
    for item in validated:
        by_condition.setdefault(item.condition_id, []).append(item)

    selected: list[CycleMatrixObservation] = []
    deviations: list[float] = []
    for condition_id in selection.condition_ids:
        candidates = by_condition.get(condition_id)
        if not candidates:
            raise ValueError(f"reviewed condition_id is missing: {condition_id}")
        ranked = sorted(
            (
                (abs(item.observation_value - selection.target_axis_value), item)
                for item in candidates
            ),
            key=lambda pair: (pair[0], pair[1].observation_value),
        )
        if len(ranked) > 1 and math.isclose(
            ranked[0][0], ranked[1][0], rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"reviewed axis selection is ambiguous for {condition_id}")
        deviation, chosen = ranked[0]
        if deviation > selection.max_absolute_deviation:
            raise ValueError(
                f"condition {condition_id} is outside reviewed axis tolerance"
            )
        selected.append(chosen)
        deviations.append(float(deviation))

    return ReviewedAxisSelectionResult(
        selection_version=selection.selection_version,
        target_axis_value=selection.target_axis_value,
        observations=tuple(selected),
        absolute_deviations=tuple(deviations),
    )


def _validated_layout(layout: NaumannCycleMatrixLayout) -> NaumannCycleMatrixLayout:
    """Revalidate a layout so model-constructed bypass instances cannot be used."""

    return NaumannCycleMatrixLayout.model_validate(layout.model_dump(mode="json"))


def _validate_source(
    path: Path,
    manifest: RawFileManifest,
    source: SourceCatalogEntry,
) -> None:
    if manifest.dataset_id != "NAUMANN_CYCLE" or source.dataset_id != manifest.dataset_id:
        raise ValueError("cycle manifest and source dataset_id must be NAUMANN_CYCLE")
    if source.ingestion_mode is not IngestionMode.MATLAB:
        raise ValueError("Naumann cycle source must use MATLAB ingestion")
    approved_suffixes = {suffix.lower() for suffix in source.expected_suffixes}
    if path.suffix.lower() != ".mat" or ".mat" not in approved_suffixes:
        raise ValueError("Naumann cycle source must use an approved .mat suffix")
    if manifest.source_uri != source.source_uri:
        raise ValueError("Naumann cycle manifest source URI does not match the source catalog")
    if manifest.license_name != source.license_status:
        raise ValueError("Naumann cycle manifest license does not match the source catalog")


def _require_matrix(value: object, *, variable_name: str) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - only without optional data extras
        raise RuntimeError(_DATA_DEPENDENCY_ERROR) from exc

    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"MATLAB variable {variable_name!r} must be a non-empty matrix")
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(f"MATLAB variable {variable_name!r} must be numeric")
    numeric = matrix.astype(float, copy=False)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"MATLAB variable {variable_name!r} contains non-finite values")
    return numeric


def _legend_text(value: object) -> str:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - only without optional data extras
        raise RuntimeError(_DATA_DEPENDENCY_ERROR) from exc

    array = np.asarray(value)
    if array.size == 0:
        raise ValueError("MATLAB legend value must not be empty")
    if array.dtype.kind in {"U", "S"}:
        flattened = array.reshape(-1)
        if flattened.size == 1:
            text = str(flattened[0])
        else:
            text = "".join(str(item) for item in flattened)
    elif array.dtype == object and array.size == 1:
        return _legend_text(array.reshape(-1)[0])
    else:
        raise ValueError("MATLAB legend value must be text")
    text = text.strip()
    if not text:
        raise ValueError("MATLAB legend value must not be blank")
    return text


def _require_legends(value: object, *, expected_count: int) -> tuple[str, ...]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - only without optional data extras
        raise RuntimeError(_DATA_DEPENDENCY_ERROR) from exc

    array = np.asarray(value)
    if array.size != expected_count:
        raise ValueError("MATLAB legend count must match matrix condition columns")
    return tuple(_legend_text(item) for item in array.reshape(-1))


def load_naumann_cycle_matrix(
    path: Path,
    manifest: RawFileManifest,
    source: SourceCatalogEntry,
    *,
    layout: NaumannCycleMatrixLayout,
) -> tuple[CycleMatrixObservation, ...]:
    """Load one explicitly declared condition matrix after source verification."""

    path = Path(path)
    _validate_source(path, manifest, source)
    layout = _validated_layout(layout)
    source_sha256 = verify_raw_file(path, manifest)

    try:
        import numpy as np
        from scipy.io import loadmat  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - only without optional data extras
        raise RuntimeError(_DATA_DEPENDENCY_ERROR) from exc

    raw = loadmat(
        path,
        appendmat=False,
        variable_names=[
            layout.x_axis_variable,
            layout.y_axis_variable,
            layout.legend_variable,
        ],
        chars_as_strings=True,
        squeeze_me=False,
    )
    required_variables = (
        layout.x_axis_variable,
        layout.y_axis_variable,
        layout.legend_variable,
    )
    missing = [name for name in required_variables if name not in raw]
    if missing:
        raise ValueError("MATLAB source is missing declared variables: " + ", ".join(missing))

    x_axis = _require_matrix(raw[layout.x_axis_variable], variable_name=layout.x_axis_variable)
    y_axis = _require_matrix(raw[layout.y_axis_variable], variable_name=layout.y_axis_variable)
    if x_axis.shape != y_axis.shape:
        raise ValueError("MATLAB x and y matrices must have identical shapes")
    if x_axis.shape[1] != len(layout.condition_columns):
        raise ValueError("reviewed condition columns must cover every MATLAB matrix column")
    legends = _require_legends(raw[layout.legend_variable], expected_count=x_axis.shape[1])

    observations: list[CycleMatrixObservation] = []
    for condition in layout.condition_columns:
        if condition.column_index >= x_axis.shape[1]:
            raise ValueError("cycle condition column index is outside the MATLAB matrix")
        if legends[condition.column_index] != condition.expected_legend:
            raise ValueError(
                "MATLAB expected_legend mismatch for " f"condition {condition.condition_id!r}"
            )
        x_values = x_axis[:, condition.column_index]
        y_values = y_axis[:, condition.column_index]
        if np.any(np.diff(x_values) <= 0):
            message = "cycle observation axis must strictly increase without reordering rows"
            raise ValueError(message)
        if np.any(y_values < 0):
            raise ValueError("cycle metric values must be non-negative")
        observations.extend(
            CycleMatrixObservation(
                condition_id=condition.condition_id,
                observation_axis=layout.observation_axis,
                observation_value=float(observation_value),
                temperature_c=condition.temperature_c,
                mean_soc=condition.mean_soc,
                dod=condition.dod,
                charge_c_rate=condition.charge_c_rate,
                discharge_c_rate=condition.discharge_c_rate,
                metric_name=layout.metric_name,
                metric_value=float(metric_value),
                source_file=manifest.relative_path,
                source_sha256=source_sha256,
                layout_version=layout.layout_version,
            )
            for observation_value, metric_value in zip(x_values, y_values, strict=True)
        )
    if not observations:
        raise ValueError("cycle MATLAB matrix contains no reviewed observations")
    return tuple(observations)
