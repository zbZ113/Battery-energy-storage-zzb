import math
from pathlib import Path
from typing import Any, Literal, TypeAlias

from quanxin_life.core import CellMetadata, sha256_canonical
from quanxin_life.data.manifest import RawFileManifest, verify_raw_file
from quanxin_life.data.schemas import CycleRecord

MatrCell: TypeAlias = tuple[CellMetadata, tuple[CycleRecord, ...]]

_ALLOWED_SUFFIXES = frozenset({".mat", ".h5", ".hdf5"})
_REQUIRED_SAMPLE_FIELDS = ("t", "V", "I", "Qc", "Qd")
_ADAPTER_VERSION = "matr-hdf5-v1.0.0"


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
    if manifest.dataset_id != "MATR":
        raise ValueError("MATR manifest dataset_id must be MATR")
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
        required_batch_fields = ("summary", "cycles", "policy_readable", "cycle_life")
        missing_batch_fields = [field for field in required_batch_fields if field not in batch]
        if missing_batch_fields:
            raise ValueError(
                "MATR batch is missing required references: " + ", ".join(missing_batch_fields)
            )
        if batch["summary"].shape[0] != batch["cycles"].shape[0]:
            raise ValueError("MATR batch has inconsistent cell reference counts")

        for cell_index in range(batch["cycles"].shape[0]):
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
            if len(life_values) != 1 or not math.isfinite(life_values[0]):
                raise ValueError(f"{cell_id} has invalid official cycle-life label")
            official_life = int(life_values[0])
            if official_life < 0 or not math.isclose(life_values[0], official_life):
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
                raw_cell_id=raw_cell_id,
                chemistry="LFP/graphite",
                nominal_capacity_ah=1.1,
                reference_capacity_ah=None,
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
                    "skip_cycle_zero": skip_cycle_zero,
                    "time_unit": time_unit,
                },
            )
            cells.append((metadata, tuple(records)))
    return tuple(cells)
