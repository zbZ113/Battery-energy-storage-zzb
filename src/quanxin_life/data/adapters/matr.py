from pathlib import Path
from typing import Any, Literal, TypeAlias

from quanxin_life.core import CellMetadata
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.schemas import CycleRecord

MatrCell: TypeAlias = tuple[CellMetadata, tuple[CycleRecord, ...]]

_ALLOWED_SUFFIXES = frozenset({".mat", ".h5", ".hdf5"})
_REQUIRED_SAMPLE_FIELDS = ("t", "V", "I", "Qc", "Qd")


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


def load_matr_batch(
    path: Path,
    manifest: RawFileManifest,
    *,
    batch_index: int,
    time_unit: Literal["seconds", "minutes"],
    skip_cycle_zero: bool = False,
) -> tuple[MatrCell, ...]:
    """Load one provenance-verified MATR MATLAB v7.3/HDF5 batch.

    The raw ``t`` unit is intentionally explicit because published MATR readers
    disagree about whether it is seconds or minutes.
    """
    path = Path(path)
    if path.suffix.lower() not in _ALLOWED_SUFFIXES:
        raise ValueError(f"unsupported MATR file suffix: {path.suffix}")
    source_sha256 = verify_raw_file(path, manifest)
    if batch_index < 1:
        raise ValueError("batch_index must be at least 1")
    if time_unit not in {"seconds", "minutes"}:
        raise ValueError("time_unit must be 'seconds' or 'minutes'")
    time_scale = 1.0 if time_unit == "seconds" else 60.0

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised only without the data extra
        raise RuntimeError("MATR loading requires the 'data' optional dependencies") from exc

    cells: list[MatrCell] = []
    with h5py.File(path, "r") as handle:
        if "batch" not in handle:
            raise ValueError("MATR file is missing the 'batch' group")
        batch = handle["batch"]
        if "summary" not in batch or "cycles" not in batch:
            raise ValueError("MATR batch is missing summary or cycles references")
        if batch["summary"].shape[0] != batch["cycles"].shape[0]:
            raise ValueError("MATR batch has inconsistent cell reference counts")

        for cell_index in range(batch["cycles"].shape[0]):
            cell_id = f"MATR_b{batch_index}c{cell_index}"
            summary = handle[batch["summary"][cell_index, 0]]
            cycles = handle[batch["cycles"][cell_index, 0]]
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

            records: list[CycleRecord] = []
            for cycle_index in range(cycle_count):
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
                        )
                    )

            metadata = CellMetadata(
                dataset_id=manifest.dataset_id,
                cell_id=cell_id,
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.1,
                reference_capacity_ah=None,
                protocol_id=f"MATR_batch_{batch_index}",
                source_uri=manifest.source_uri,
                source_sha256=source_sha256,
                schema_version="1.0",
            )
            cells.append((metadata, tuple(records)))
    return tuple(cells)
