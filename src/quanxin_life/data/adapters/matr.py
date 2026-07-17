import math
from collections.abc import Collection, Iterator
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any, Literal, TypeAlias

from quanxin_life.core import CellMetadata, sha256_canonical
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.schemas import CycleRecord

MatrCell: TypeAlias = tuple[CellMetadata, tuple[CycleRecord, ...]]

_ALLOWED_SUFFIXES = frozenset({".mat", ".h5", ".hdf5"})
_REQUIRED_SAMPLE_FIELDS = ("t", "V", "I", "Qc", "Qd")
_ADAPTER_VERSION = "matr-hdf5-v1.2.0"
_REFERENCE_CAPACITY_CYCLES = (1, 2, 3, 4, 5)


def _flatten_numeric(value: Any) -> list[float]:
    return [float(item) for item in value.reshape(-1)]


def _resolve_numeric_dataset(handle: Any, dataset: Any) -> list[float]:
    values = dataset[()]
    if values.dtype.kind != "O":
        return _flatten_numeric(values)

    resolved: list[float] = []
    for reference in values.reshape(-1):
        resolved.extend(_flatten_numeric(handle[reference][()]))
    return resolved


def _cycle_values(handle: Any, cycles: Any, field: str, cycle_index: int) -> list[float]:
    references = cycles[field]
    reference = references[cycle_index, 0]
    return _resolve_numeric_dataset(handle, handle[reference])


def _optional_summary_values(handle: Any, summary: Any, field: str) -> list[float] | None:
    if field not in summary:
        return None
    return _resolve_numeric_dataset(handle, summary[field])


def _resolve_text_reference(handle: Any, reference: Any) -> str:
    values = handle[reference][()].reshape(-1)
    if values.dtype.kind == "S":
        text = b"".join(bytes(item) for item in values).decode("utf-8")
    elif values.dtype.kind == "U":
        text = "".join(str(item) for item in values)
    else:
        text = "".join(chr(int(item)) for item in values if int(item) != 0)
    text = text.strip()
    if not text:
        raise ValueError("MATR charge policy is empty")
    return text


def _resolve_batch_date(handle: Any) -> date:
    if "batch_date" not in handle:
        raise ValueError("MATR file is missing batch_date metadata")
    values = handle["batch_date"][()].reshape(-1)
    text = "".join(chr(int(item)) for item in values if int(item) != 0).strip()
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError("MATR batch_date must use ISO YYYY-MM-DD format") from exc


def _select_cell_indices(
    *,
    batch_index: int,
    cell_count: int,
    selected_raw_cell_ids: Collection[str] | None,
) -> tuple[tuple[int, ...], list[str] | None]:
    """Resolve an explicit raw-cell selection without silently dropping requests."""
    if selected_raw_cell_ids is None:
        return tuple(range(cell_count)), None
    if isinstance(selected_raw_cell_ids, str):
        raise ValueError("selected raw MATR cell IDs must be a collection, not a string")

    selected = tuple(selected_raw_cell_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("selected raw MATR cell IDs must not contain duplicates")
    if any(not isinstance(raw_cell_id, str) or not raw_cell_id for raw_cell_id in selected):
        raise ValueError("selected raw MATR cell IDs must be non-empty strings")

    available_indices = {
        f"b{batch_index}c{cell_index}": cell_index for cell_index in range(cell_count)
    }
    unknown = sorted(set(selected).difference(available_indices))
    if unknown:
        raise ValueError("unknown raw MATR cell IDs: " + ", ".join(unknown))
    return tuple(available_indices[raw_cell_id] for raw_cell_id in selected), list(selected)


def iter_matr_batch(
    path: Path,
    manifest: RawFileManifest,
    *,
    batch_index: int,
    time_unit: Literal["seconds", "minutes"],
    skip_cycle_zero: bool = False,
    selected_raw_cell_ids: Collection[str] | None = None,
    max_cycle_index: int | None = None,
    expected_batch_date: date | None = None,
) -> Iterator[MatrCell]:
    """Yield one cell at a time from a verified MATR MATLAB v7.3/HDF5 batch.

    The raw ``t`` unit is intentionally explicit because published MATR readers
    disagree about whether it is seconds or minutes.  Cycle zero is retained as
    raw evidence but is not a post-formation diagnostic cycle.  The stable SOH
    reference is the median summary discharge capacity from cycles 1 through 5
    when all five cycles are inside the requested observation horizon.
    """
    path = Path(path)
    if manifest.dataset_id != "MATR":
        raise ValueError("MATR manifest dataset_id must be MATR")
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported MATR file suffix: {path.suffix}")
    source_sha256 = verify_raw_file(path, manifest)
    if batch_index < 1:
        raise ValueError("batch_index must be at least 1")
    if time_unit not in {"seconds", "minutes"}:
        raise ValueError("time_unit must be 'seconds' or 'minutes'")
    if max_cycle_index is not None and (
        not isinstance(max_cycle_index, int)
        or isinstance(max_cycle_index, bool)
        or max_cycle_index < 0
    ):
        raise ValueError("max_cycle_index must be a non-negative integer or None")
    time_scale = 1.0 if time_unit == "seconds" else 60.0

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised only without the data extra
        raise RuntimeError("MATR loading requires the 'data' optional dependencies") from exc

    with h5py.File(path, "r") as handle:
        resolved_batch_date: date | None = None
        if expected_batch_date is not None:
            resolved_batch_date = _resolve_batch_date(handle)
            if resolved_batch_date != expected_batch_date:
                raise ValueError(
                    "MATR batch_date does not match the reviewed conversion configuration"
                )
        if "batch" not in handle:
            raise ValueError("MATR file is missing the 'batch' group")
        batch = handle["batch"]
        required_batch_fields = ("summary", "cycles", "policy_readable", "cycle_life")
        missing_batch_fields = [field for field in required_batch_fields if field not in batch]
        if missing_batch_fields:
            raise ValueError(
                "MATR batch is missing required references: " + ", ".join(missing_batch_fields)
            )
        reference_counts = {field: batch[field].shape[0] for field in required_batch_fields}
        if len(set(reference_counts.values())) != 1:
            raise ValueError("MATR batch has inconsistent cell reference counts")

        selected_indices, selected_ids_for_metadata = _select_cell_indices(
            batch_index=batch_index,
            cell_count=batch["cycles"].shape[0],
            selected_raw_cell_ids=selected_raw_cell_ids,
        )
        for cell_index in selected_indices:
            raw_cell_id = f"b{batch_index}c{cell_index}"
            cell_id = f"MATR_{raw_cell_id}"
            summary = handle[batch["summary"][cell_index, 0]]
            cycles = handle[batch["cycles"][cell_index, 0]]
            policy = _resolve_text_reference(
                handle, batch["policy_readable"][cell_index, 0]
            )
            life_values = _resolve_numeric_dataset(
                handle, handle[batch["cycle_life"][cell_index, 0]]
            )
            if len(life_values) != 1:
                raise ValueError(f"{cell_id} has invalid official cycle-life label")
            raw_official_life = life_values[0]
            official_life_right_censored = math.isnan(raw_official_life)
            official_life: int | None = None
            if not official_life_right_censored:
                if not math.isfinite(raw_official_life):
                    raise ValueError(f"{cell_id} has invalid official cycle-life label")
                official_life = int(raw_official_life)
                if official_life < 0 or not math.isclose(raw_official_life, official_life):
                    raise ValueError(f"{cell_id} has invalid official cycle-life label")
            missing = [field for field in _REQUIRED_SAMPLE_FIELDS if field not in cycles]
            if missing:
                missing_fields = ", ".join(missing)
                raise ValueError(f"{cell_id} is missing required cycle arrays: {missing_fields}")

            cycle_count = cycles["I"].shape[0]
            consumed_fields = (*_REQUIRED_SAMPLE_FIELDS, "T")
            if any(
                cycles[field].shape[0] != cycle_count
                for field in consumed_fields
                if field in cycles
            ):
                raise ValueError(f"{cell_id} has inconsistent cycle counts")
            resistance = _optional_summary_values(handle, summary, "IR")
            if resistance is not None and len(resistance) != cycle_count:
                raise ValueError(f"{cell_id} has inconsistent internal-resistance cycle count")
            summary_discharge = _optional_summary_values(handle, summary, "QDischarge")
            if summary_discharge is None or len(summary_discharge) != cycle_count:
                raise ValueError(f"{cell_id} has inconsistent summary discharge-capacity count")

            records: list[CycleRecord] = []
            bounded_cycle_count = (
                cycle_count
                if max_cycle_index is None
                else min(cycle_count, max_cycle_index + 1)
            )
            for cycle_index in range(bounded_cycle_count):
                if skip_cycle_zero and cycle_index == 0:
                    continue
                arrays = {
                    field: _cycle_values(handle, cycles, field, cycle_index)
                    for field in _REQUIRED_SAMPLE_FIELDS
                }
                temperature = (
                    _cycle_values(handle, cycles, "T", cycle_index) if "T" in cycles else None
                )
                lengths = {len(values) for values in arrays.values()}
                if temperature is not None:
                    lengths.add(len(temperature))
                if len(lengths) != 1:
                    raise ValueError(
                        f"{cell_id} cycle {cycle_index} has inconsistent sample array lengths"
                    )

                for sample_index in range(len(arrays["t"])):
                    records.append(
                        CycleRecord(
                            dataset_id=manifest.dataset_id,
                            cell_id=cell_id,
                            cycle_index=cycle_index,
                            sample_index=sample_index,
                            time_s=arrays["t"][sample_index] * time_scale,
                            voltage_v=arrays["V"][sample_index],
                            current_a=arrays["I"][sample_index],
                            temperature_c=(
                                temperature[sample_index] if temperature is not None else None
                            ),
                            charge_capacity_ah=arrays["Qc"][sample_index],
                            discharge_capacity_ah=arrays["Qd"][sample_index],
                            internal_resistance_ohm=(
                                resistance[cycle_index] if resistance is not None else None
                            ),
                            diagnostic=cycle_index > 0,
                        )
                    )

            reference_capacity_ah: float | None = None
            if bounded_cycle_count > _REFERENCE_CAPACITY_CYCLES[-1]:
                reference_values = [
                    summary_discharge[cycle_index]
                    for cycle_index in _REFERENCE_CAPACITY_CYCLES
                ]
                if any(not math.isfinite(value) or value <= 0 for value in reference_values):
                    raise ValueError(f"{cell_id} has invalid reference-capacity window")
                reference_capacity_ah = float(median(reference_values))

            metadata = CellMetadata(
                dataset_id=manifest.dataset_id,
                cell_id=cell_id,
                raw_cell_id=raw_cell_id,
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.1,
                reference_capacity_ah=reference_capacity_ah,
                protocol_id=f"MATR_policy_{sha256_canonical(policy)[:16]}",
                protocol_description=policy,
                official_life_label=official_life,
                official_life_label_name="MATR_cycle_life",
                source_uri=manifest.source_uri,
                source_sha256=source_sha256,
                schema_version="1.0",
                adapter_version=_ADAPTER_VERSION,
                ingestion_parameters={
                    "batch_index": batch_index,
                    **(
                        {"batch_date": resolved_batch_date.isoformat()}
                        if resolved_batch_date is not None
                        else {}
                    ),
                    "max_cycle_index": max_cycle_index,
                    "observed_cycle_count": cycle_count,
                    "official_life_right_censored": official_life_right_censored,
                    "reference_capacity_cycles": list(_REFERENCE_CAPACITY_CYCLES),
                    "selected_raw_cell_ids": selected_ids_for_metadata,
                    "skip_cycle_zero": skip_cycle_zero,
                    "time_unit": time_unit,
                },
            )
            yield metadata, tuple(records)


def load_matr_batch(
    path: Path,
    manifest: RawFileManifest,
    *,
    batch_index: int,
    time_unit: Literal["seconds", "minutes"],
    skip_cycle_zero: bool = False,
    selected_raw_cell_ids: Collection[str] | None = None,
    max_cycle_index: int | None = None,
    expected_batch_date: date | None = None,
) -> tuple[MatrCell, ...]:
    """Materialize selected MATR cells; prefer :func:`iter_matr_batch` for full batches."""

    return tuple(
        iter_matr_batch(
            path,
            manifest,
            batch_index=batch_index,
            time_unit=time_unit,
            skip_cycle_zero=skip_cycle_zero,
            selected_raw_cell_ids=selected_raw_cell_ids,
            max_cycle_index=max_cycle_index,
            expected_batch_date=expected_batch_date,
        )
    )
