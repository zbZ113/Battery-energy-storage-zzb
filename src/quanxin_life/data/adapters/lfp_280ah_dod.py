"""Reviewed archive adapters for the 40/280 Ah multi-DoD dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from datetime import datetime
from io import TextIOWrapper
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.zip_safety import audit_zip_archive

_MEMBER = re.compile(
    r"^(?P<vendor>CATL|EVE)/(?P=vendor)-0\.5C-(?P<dod>20|60|100)% "
    r"Depth of Discharge/(?P<cell>\d+)\.zip$"
)
_INNER_CSV = re.compile(r"^[^/\\]+-0\.5C-100%DOD-(?P<part>\d+)\.csv$")
_SECONDS_PER_DAY = 86_400.0


class DodArchiveLayout(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    layout_version: str = Field(min_length=1)
    vendor: Literal["CATL", "EVE"]
    archive_sha256: Sha256
    expected_members: tuple[str, ...] = Field(min_length=1)
    task_role: Literal["calibration", "external_test"] = "external_test"


class DodCellSource(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Literal["LFP_280AH_DOD"] = "LFP_280AH_DOD"
    vendor: Literal["CATL", "EVE"]
    cell_id: str = Field(min_length=1)
    dod_fraction: float = Field(gt=0, le=1)
    c_rate: float = 0.5
    source_member: str = Field(min_length=1)
    archive_sha256: Sha256
    task_role: Literal["calibration", "external_test"]


class DodTelemetryColumns(ContractModel):
    """Reviewed source-column names used by the streaming telemetry reducer."""

    cycle_number: str = Field(min_length=1)
    discharge_capacity_ah: str = Field(min_length=1)
    absolute_time: str = Field(min_length=1)
    temperature_c: str = Field(min_length=1)


class DodCapacityArchiveSelection(ContractModel):
    """One verified outer archive and the nested cell archives approved for replay."""

    outer_relative_path: str = Field(min_length=1)
    archive: DodArchiveLayout
    selected_members: tuple[str, ...] = Field(min_length=1)

    @field_validator("outer_relative_path")
    @classmethod
    def outer_path_is_confined(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value.startswith("data/raw/")
            or "\\" in value
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("DoD outer archive path must remain under data/raw")
        return value

    @model_validator(mode="after")
    def selection_is_reviewed_and_unique(self) -> Self:
        selected = set(self.selected_members)
        expected = set(self.archive.expected_members)
        if len(selected) != len(self.selected_members):
            raise ValueError("selected DoD archive members must be unique")
        if not selected <= expected:
            raise ValueError("selected DoD archive member is not in the reviewed layout")
        for member in self.selected_members:
            match = _MEMBER.fullmatch(member)
            if match is None or match.group("vendor") != self.archive.vendor:
                raise ValueError("selected DoD archive member has an invalid identity")
            if int(match.group("dod")) != 100:
                raise ValueError("capacity replay is restricted to reviewed 100% DoD cells")
        return self


class DodCapacityDatasetLayout(ContractModel):
    """Frozen parser, selection, normalization, and resource limits."""

    schema_version: Literal["lfp-280ah-dod-capacity-layout-v1"]
    layout_version: str = Field(min_length=1)
    nominal_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    selected_dod_fraction: float = Field(ge=1.0, le=1.0, allow_inf_nan=False)
    c_rate: float = Field(gt=0, allow_inf_nan=False)
    csv_encoding: Literal["gb18030"]
    normalization: Literal["FIRST_VALID_CYCLE_CAPACITY"]
    temperature_optional_csv_members: tuple[str, ...] = ()
    columns: DodTelemetryColumns
    archives: tuple[DodCapacityArchiveSelection, ...] = Field(min_length=1)
    max_nested_archive_bytes: int = Field(
        default=768 * 1024 * 1024,
        gt=0,
    )
    max_inner_csv_members: int = Field(default=64, ge=1, le=256)
    max_inner_csv_bytes: int = Field(default=8 * 1024 * 1024 * 1024, gt=0)
    max_inner_compression_ratio: float = Field(
        default=250.0,
        gt=0,
        allow_inf_nan=False,
    )

    @model_validator(mode="after")
    def archives_are_unique(self) -> Self:
        vendors = [item.archive.vendor for item in self.archives]
        paths = [item.outer_relative_path for item in self.archives]
        if len(vendors) != len(set(vendors)) or len(paths) != len(set(paths)):
            raise ValueError("DoD capacity archive selections must be unique")
        optional = self.temperature_optional_csv_members
        if len(optional) != len(set(optional)):
            raise ValueError("temperature-optional CSV members must be unique")
        if any(_INNER_CSV.fullmatch(member) is None for member in optional):
            raise ValueError("temperature-optional CSV member has an invalid identity")
        return self


class DodCapacityObservation(ContractModel):
    """One cycle-level direct observation reduced from reviewed telemetry rows."""

    cycle_number: int = Field(ge=1)
    elapsed_days: float = Field(ge=0, allow_inf_nan=False)
    discharge_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    nominal_soh: float = Field(gt=0, allow_inf_nan=False)
    relative_capacity_ratio: float = Field(gt=0, allow_inf_nan=False)
    mean_temperature_c: float = Field(allow_inf_nan=False)
    sample_count: int = Field(gt=0)


class DodCapacityCellSeries(ContractModel):
    """A provenance-bound 100% DoD capacity trajectory for one 280 Ah cell."""

    dataset_id: Literal["LFP_280AH_DOD"] = "LFP_280AH_DOD"
    vendor: Literal["CATL", "EVE"]
    cell_id: str = Field(min_length=1)
    nominal_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    dod_fraction: float = Field(ge=1.0, le=1.0, allow_inf_nan=False)
    c_rate: float = Field(gt=0, allow_inf_nan=False)
    reference_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    source_outer_file: str = Field(min_length=1)
    source_member: str = Field(min_length=1)
    outer_archive_sha256: Sha256
    nested_archive_sha256: Sha256
    csv_members: tuple[str, ...] = Field(min_length=1)
    temperature_missing_csv_members: tuple[str, ...] = ()
    observations: tuple[DodCapacityObservation, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def observations_are_ordered_and_normalized(self) -> Self:
        cycles = [item.cycle_number for item in self.observations]
        elapsed = [item.elapsed_days for item in self.observations]
        if cycles != sorted(cycles) or len(cycles) != len(set(cycles)):
            raise ValueError("DoD capacity cycle numbers must be unique and ordered")
        if any(later < earlier for earlier, later in pairwise(elapsed)):
            raise ValueError("DoD capacity elapsed time must be nondecreasing")
        first = self.observations[0]
        if not math.isclose(first.elapsed_days, 0.0, abs_tol=1e-12):
            raise ValueError("DoD capacity series must start at elapsed day zero")
        if not math.isclose(first.relative_capacity_ratio, 1.0, abs_tol=1e-12):
            raise ValueError("DoD capacity series must use its first valid cycle as reference")
        if not math.isclose(
            first.discharge_capacity_ah,
            self.reference_capacity_ah,
            abs_tol=1e-9,
        ):
            raise ValueError("DoD capacity reference does not match the first observation")
        return self


class _CycleAccumulator:
    __slots__ = (
        "capacity_ah",
        "end_time",
        "sample_count",
        "temperature_count",
        "temperature_sum",
    )

    def __init__(self) -> None:
        self.capacity_ah = 0.0
        self.end_time: datetime | None = None
        self.sample_count = 0
        self.temperature_count = 0
        self.temperature_sum = 0.0

    def add(
        self,
        *,
        capacity_ah: float | None,
        timestamp: datetime,
        temperature_c: float | None,
    ) -> None:
        self.sample_count += 1
        if capacity_ah is not None:
            self.capacity_ah = max(self.capacity_ah, capacity_ah)
        if self.end_time is None or timestamp > self.end_time:
            self.end_time = timestamp
        if temperature_c is not None:
            self.temperature_sum += temperature_c
            self.temperature_count += 1


def load_dod_sources(path: Path, *, layout: DodArchiveLayout) -> tuple[DodCellSource, ...]:
    validated = DodArchiveLayout.model_validate(layout.model_dump(mode="json"))
    infos = audit_zip_archive(
        path,
        archive_sha256=validated.archive_sha256,
        expected_members=validated.expected_members,
        allowed_suffixes=frozenset({".zip"}),
    )
    records: list[DodCellSource] = []
    for info in infos:
        match = _MEMBER.fullmatch(info.filename)
        if match is None or match.group("vendor") != validated.vendor:
            raise ValueError("DoD ZIP member does not match the reviewed layout")
        records.append(
            DodCellSource(
                vendor=validated.vendor,
                cell_id=f"{validated.vendor}-{match.group('cell')}",
                dod_fraction=int(match.group("dod")) / 100.0,
                source_member=info.filename,
                archive_sha256=validated.archive_sha256,
                task_role=validated.task_role,
            )
        )
    return tuple(records)


def load_dod_capacity_dataset_layout(path: Path) -> DodCapacityDatasetLayout:
    """Load one strict UTF-8 capacity-processing layout."""

    layout_path = Path(path)
    if layout_path.suffix.lower() != ".json":
        raise ValueError("DoD capacity layout must use a .json file")
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("DoD capacity layout does not exist") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("DoD capacity layout must contain valid UTF-8 JSON") from exc
    return DodCapacityDatasetLayout.model_validate(payload)


def _copy_nested_archive(
    source: object,
    target: Path,
    *,
    expected_size: int,
) -> str:
    digest = hashlib.sha256()
    written = 0
    with target.open("xb") as output:
        while True:
            chunk = source.read(1024 * 1024)  # type: ignore[attr-defined]
            if not isinstance(chunk, bytes):
                raise ValueError("nested DoD archive stream must be binary")
            if not chunk:
                break
            written += len(chunk)
            if written > expected_size:
                raise ValueError("nested DoD archive exceeded its reviewed size")
            digest.update(chunk)
            output.write(chunk)
    if written != expected_size:
        raise ValueError("nested DoD archive size did not match the outer inventory")
    return digest.hexdigest()


def _safe_inner_member(name: str) -> tuple[int, PurePosixPath]:
    path = PurePosixPath(name)
    if (
        "\\" in name
        or path.is_absolute()
        or len(path.parts) != 1
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError("unsafe nested ZIP member")
    match = _INNER_CSV.fullmatch(name)
    if match is None:
        raise ValueError("nested DoD ZIP contains an unreviewed CSV name")
    return int(match.group("part")), path


def _optional_float(value: str, *, field_name: str) -> float | None:
    normalized = value.strip()
    if not normalized:
        return None
    try:
        result = float(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid numeric {field_name} in DoD telemetry") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field_name} in DoD telemetry")
    return result


def _cycle_number(value: str) -> int:
    try:
        numeric = float(value.strip())
    except ValueError as exc:
        raise ValueError("invalid cycle number in DoD telemetry") from exc
    if not math.isfinite(numeric) or numeric < 1 or not numeric.is_integer():
        raise ValueError("invalid cycle number in DoD telemetry")
    return int(numeric)


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise ValueError("invalid absolute time in DoD telemetry") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _column_indices(
    header: list[str],
    columns: DodTelemetryColumns,
    *,
    allow_missing_temperature: bool,
) -> dict[str, int | None]:
    normalized = [item.lstrip("\ufeff").strip() for item in header]
    required = columns.model_dump(mode="python")
    indices: dict[str, int | None] = {}
    for field_name, source_name in required.items():
        matches = [index for index, name in enumerate(normalized) if name == source_name]
        if field_name == "temperature_c" and not matches and allow_missing_temperature:
            indices[field_name] = None
            continue
        if len(matches) != 1:
            raise ValueError(f"DoD telemetry column is missing or duplicated: {source_name}")
        indices[field_name] = matches[0]
    return indices


def _parse_csv_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    layout: DodCapacityDatasetLayout,
    cycles: dict[int, _CycleAccumulator],
) -> bool:
    with archive.open(info, "r") as raw, TextIOWrapper(
        raw,
        encoding=layout.csv_encoding,
        newline="",
    ) as text:
        reader = csv.reader(text)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("DoD telemetry CSV is empty") from exc
        indices = _column_indices(
            header,
            layout.columns,
            allow_missing_temperature=(
                info.filename in layout.temperature_optional_csv_members
            ),
        )
        temperature_index = indices["temperature_c"]
        if temperature_index is None:
            return True
        required_indices = [index for index in indices.values() if index is not None]
        maximum_index = max(required_indices)
        cycle_index = indices["cycle_number"]
        capacity_index = indices["discharge_capacity_ah"]
        absolute_time_index = indices["absolute_time"]
        assert cycle_index is not None
        assert capacity_index is not None
        assert absolute_time_index is not None
        for row in reader:
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) <= maximum_index:
                raise ValueError("DoD telemetry row is shorter than its reviewed header")
            cycle = _cycle_number(row[cycle_index])
            capacity = _optional_float(
                row[capacity_index],
                field_name="discharge capacity",
            )
            if capacity is not None and capacity < 0:
                raise ValueError("negative discharge capacity in DoD telemetry")
            temperature = _optional_float(
                row[temperature_index],
                field_name="temperature",
            )
            cycles.setdefault(cycle, _CycleAccumulator()).add(
                capacity_ah=capacity,
                timestamp=_timestamp(row[absolute_time_index]),
                temperature_c=temperature,
            )
    return False


def _parse_nested_archive(
    path: Path,
    *,
    layout: DodCapacityDatasetLayout,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[int, _CycleAccumulator]]:
    cycles: dict[int, _CycleAccumulator] = {}
    with zipfile.ZipFile(path, "r") as archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("nested DoD ZIP contains duplicate members")
        if not infos or len(infos) > layout.max_inner_csv_members:
            raise ValueError("nested DoD ZIP CSV member count exceeds policy")
        ordered: list[tuple[int, zipfile.ZipInfo]] = []
        total = 0
        for info in infos:
            part, _ = _safe_inner_member(info.filename)
            total += info.file_size
            if total > layout.max_inner_csv_bytes:
                raise ValueError("nested DoD ZIP uncompressed bytes exceed policy")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > layout.max_inner_compression_ratio:
                raise ValueError("nested DoD ZIP compression ratio exceeds policy")
            ordered.append((part, info))
        ordered.sort(key=lambda item: (item[0], item[1].filename))
        temperature_missing: list[str] = []
        for _, info in ordered:
            if _parse_csv_member(archive, info, layout=layout, cycles=cycles):
                temperature_missing.append(info.filename)
    return (
        tuple(info.filename for _, info in ordered),
        tuple(temperature_missing),
        cycles,
    )


def _build_series(
    *,
    source: DodCellSource,
    selection: DodCapacityArchiveSelection,
    layout: DodCapacityDatasetLayout,
    nested_sha256: str,
    csv_members: tuple[str, ...],
    temperature_missing_csv_members: tuple[str, ...],
    cycles: dict[int, _CycleAccumulator],
) -> DodCapacityCellSeries:
    valid = [
        (cycle, accumulator)
        for cycle, accumulator in sorted(cycles.items())
        if accumulator.capacity_ah > 0
        and accumulator.end_time is not None
        and accumulator.temperature_count > 0
    ]
    if len(valid) < 2:
        raise ValueError("DoD capacity cell requires at least two valid cycles")
    reference_capacity = valid[0][1].capacity_ah
    first_end = valid[0][1].end_time
    assert first_end is not None
    observations: list[DodCapacityObservation] = []
    for cycle, accumulator in valid:
        assert accumulator.end_time is not None
        elapsed_days = (accumulator.end_time - first_end).total_seconds() / _SECONDS_PER_DAY
        if elapsed_days < 0:
            raise ValueError("DoD capacity cycle timestamps are not ordered")
        observations.append(
            DodCapacityObservation(
                cycle_number=cycle,
                elapsed_days=elapsed_days,
                discharge_capacity_ah=accumulator.capacity_ah,
                nominal_soh=accumulator.capacity_ah / layout.nominal_capacity_ah,
                relative_capacity_ratio=accumulator.capacity_ah / reference_capacity,
                mean_temperature_c=(
                    accumulator.temperature_sum / accumulator.temperature_count
                ),
                sample_count=accumulator.sample_count,
            )
        )
    return DodCapacityCellSeries(
        vendor=source.vendor,
        cell_id=source.cell_id,
        nominal_capacity_ah=layout.nominal_capacity_ah,
        dod_fraction=1.0,
        c_rate=layout.c_rate,
        reference_capacity_ah=reference_capacity,
        source_outer_file=selection.outer_relative_path,
        source_member=source.source_member,
        outer_archive_sha256=source.archive_sha256,
        nested_archive_sha256=nested_sha256,
        csv_members=csv_members,
        temperature_missing_csv_members=temperature_missing_csv_members,
        observations=tuple(observations),
    )


def load_dod_capacity_series(
    path: Path,
    *,
    layout: DodCapacityDatasetLayout,
    vendor: Literal["CATL", "EVE"],
    temporary_directory: Path,
) -> tuple[DodCapacityCellSeries, ...]:
    """Stream selected 100% DoD nested archives into cycle-level capacity series."""

    validated = DodCapacityDatasetLayout.model_validate(layout.model_dump(mode="json"))
    selection = next(
        (item for item in validated.archives if item.archive.vendor == vendor),
        None,
    )
    if selection is None:
        raise ValueError("DoD capacity layout does not contain the requested vendor")
    sources = load_dod_sources(path, layout=selection.archive)
    by_member = {item.source_member: item for item in sources}
    temp = Path(temporary_directory)
    if temp.exists():
        raise ValueError("DoD capacity temporary directory already exists")
    temp.mkdir(parents=True)
    results: list[DodCapacityCellSeries] = []
    try:
        with zipfile.ZipFile(Path(path).resolve(strict=True), "r") as outer:
            info_by_name = {item.filename: item for item in outer.infolist()}
            for member in selection.selected_members:
                source = by_member.get(member)
                info = info_by_name.get(member)
                if source is None or info is None:
                    raise ValueError("selected DoD capacity member was not audited")
                if source.dod_fraction != validated.selected_dod_fraction:
                    raise ValueError("capacity replay is restricted to reviewed 100% DoD cells")
                if info.file_size > validated.max_nested_archive_bytes:
                    raise ValueError("nested DoD archive size exceeds policy")
                nested_path = temp / f"{source.cell_id}.zip"
                with outer.open(info, "r") as stream:
                    nested_sha = _copy_nested_archive(
                        stream,
                        nested_path,
                        expected_size=info.file_size,
                    )
                csv_members, temperature_missing, cycles = _parse_nested_archive(
                    nested_path,
                    layout=validated,
                )
                results.append(
                    _build_series(
                        source=source,
                        selection=selection,
                        layout=validated,
                        nested_sha256=nested_sha,
                        csv_members=csv_members,
                        temperature_missing_csv_members=temperature_missing,
                        cycles=cycles,
                    )
                )
                nested_path.unlink()
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return tuple(results)


__all__ = [
    "DodArchiveLayout",
    "DodCapacityArchiveSelection",
    "DodCapacityCellSeries",
    "DodCapacityDatasetLayout",
    "DodCapacityObservation",
    "DodCellSource",
    "DodTelemetryColumns",
    "load_dod_capacity_dataset_layout",
    "load_dod_capacity_series",
    "load_dod_sources",
]
