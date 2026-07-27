"""Trusted, task-specific readers for frozen MATR calibration supervision."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core import AdvancedModelTask, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.matr_multibatch import (
    MatrThreeBatchManifest,
    MatrTrajectoryEligibilityAudit,
    combine_matr_batch_splits,
)
from quanxin_life.data.matr_pipeline import (
    MatrBatchConversionReport,
    MatrSupervisionArtifact,
)
from quanxin_life.data.schemas import SplitManifest

_THREE_BATCH_MANIFEST = PurePosixPath(
    "reports/data_quality/matr_three_batch_manifest_v1.json"
)
_APPROVED_CUTOFFS = {20, 50, 100, 150}
_SOH_HORIZON_CYCLE = 500
_SUPERVISION_COLUMNS = {
    "dataset_id",
    "cell_id",
    "cycle_index",
    "discharge_capacity_ah",
    "reference_capacity_ah",
    "soh",
}


class AdvancedCalibrationSourceRegistration(ContractModel):
    """Server-owned registration of one immutable MATR evidence root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registration_id: str = Field(min_length=1, max_length=200)
    evidence_root: Path
    three_batch_manifest_sha256: Sha256


class AdvancedCalibrationSourceIdentity(ContractModel):
    """Path-free identity for all source evidence used by one cohort."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    registration_id: str = Field(min_length=1)
    dataset_id: Literal["MATR"] = "MATR"
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    three_batch_manifest_sha256: Sha256
    combined_split_sha256: Sha256
    conversion_report_sha256s: tuple[Sha256, ...] = Field(min_length=3, max_length=3)
    component_split_sha256s: tuple[Sha256, ...] = Field(min_length=3, max_length=3)
    eligibility_report_sha256s: tuple[Sha256, ...] = Field(
        min_length=3,
        max_length=3,
    )
    supervision_report_sha256s: tuple[Sha256, ...] = Field(
        min_length=3,
        max_length=3,
    )
    supervision_parquet_sha256s: tuple[Sha256, ...] = Field(
        min_length=3,
        max_length=3,
    )
    source_identity_sha256: Sha256


class AdvancedRULObservedCell(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    observed_cycle: int = Field(gt=0)


class AdvancedSOHObservedCell(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    observed_soh: tuple[float, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def observations_are_finite(self) -> AdvancedSOHObservedCell:
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.5
            for value in self.observed_soh
        ):
            raise ValueError("observed SOH values must be finite and physically valid")
        return self


class AdvancedRULCalibrationEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_identity: AdvancedCalibrationSourceIdentity
    cells: tuple[AdvancedRULObservedCell, ...] = Field(min_length=1)


class AdvancedSOHCalibrationEvidence(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_identity: AdvancedCalibrationSourceIdentity
    prediction_cycles: tuple[int, ...] = Field(min_length=1)
    cells: tuple[AdvancedSOHObservedCell, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def finite_axis_matches_observations(self) -> AdvancedSOHCalibrationEvidence:
        if (
            self.prediction_cycles != tuple(
                range(self.prediction_cycles[0], _SOH_HORIZON_CYCLE + 1)
            )
            or self.prediction_cycles[-1] != _SOH_HORIZON_CYCLE
        ):
            raise ValueError("SOH prediction cycles must be continuous through cycle 500")
        if any(
            len(cell.observed_soh) != len(self.prediction_cycles)
            for cell in self.cells
        ):
            raise ValueError("SOH observations must match the finite prediction axis")
        return self


AdvancedCalibrationEvidence: TypeAlias = (
    AdvancedRULCalibrationEvidence | AdvancedSOHCalibrationEvidence
)


class AdvancedCalibrationEvidenceResolver(Protocol):
    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedCalibrationEvidence: ...


class RegisteredAdvancedCalibrationEvidenceResolver:
    """Resolve only pre-registered roots and verify every consumed source byte."""

    def __init__(
        self,
        registrations: Sequence[AdvancedCalibrationSourceRegistration],
    ) -> None:
        normalized = tuple(
            AdvancedCalibrationSourceRegistration.model_validate(
                registration.model_dump(mode="python")
            )
            for registration in registrations
        )
        if not normalized:
            raise ValueError("at least one Advanced calibration source is required")
        if len({item.registration_id for item in normalized}) != len(normalized):
            raise ValueError("Advanced calibration source registration IDs must be unique")
        self._registrations = {
            item.registration_id: _verified_registration(item)
            for item in normalized
        }

    def resolve(
        self,
        registration_id: str,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
    ) -> AdvancedCalibrationEvidence:
        try:
            registration = self._registrations[registration_id]
        except KeyError as exc:
            raise ValueError("Advanced calibration source is not registered") from exc
        normalized_task = AdvancedModelTask(task)
        if cutoff_cycle not in _APPROVED_CUTOFFS:
            raise ValueError("Advanced calibration requires an approved cutoff cycle")

        root = _require_root(registration.evidence_root)
        manifest_path = _registered_path(root, _THREE_BATCH_MANIFEST)
        manifest_payload = _read_verified_json(
            manifest_path,
            registration.three_batch_manifest_sha256,
        )
        manifest = MatrThreeBatchManifest.model_validate(manifest_payload)
        combined_path = _registered_path(root, manifest.combined_split_manifest)
        combined = SplitManifest.model_validate(
            _read_verified_json(combined_path, manifest.combined_split_sha256)
        )
        if (
            combined.dataset_id != "MATR"
            or len(set(combined.all_cells)) != len(combined.all_cells)
        ):
            raise ValueError("combined split is not a closed MATR cell partition")

        conversions: list[MatrBatchConversionReport] = []
        component_splits: list[SplitManifest] = []
        eligibility: list[MatrTrajectoryEligibilityAudit] = []
        supervision: list[tuple[MatrSupervisionArtifact, Path]] = []
        for component in manifest.batches:
            conversion = MatrBatchConversionReport.model_validate(
                _read_verified_json(
                    _registered_path(root, component.conversion_report),
                    component.conversion_report_sha256,
                )
            )
            split = SplitManifest.model_validate(
                _read_verified_json(
                    _registered_path(root, component.split_manifest),
                    component.split_manifest_sha256,
                )
            )
            _validate_component_conversion(component, conversion, split)
            conversions.append(conversion)
            component_splits.append(split)

            eligible = MatrTrajectoryEligibilityAudit.model_validate(
                _read_verified_json(
                    _registered_path(root, component.eligibility_report),
                    component.eligibility_report_sha256,
                )
            )
            supervision_artifact = MatrSupervisionArtifact.model_validate(
                _read_verified_json(
                    _registered_path(root, component.supervision_report),
                    component.supervision_report_sha256,
                )
            )
            supervision_root = _registered_path(root, component.supervision_root)
            parquet_path = _child_path(
                supervision_root,
                supervision_artifact.parquet_relative_path,
            )
            if _sha256_file(parquet_path) != supervision_artifact.parquet_sha256:
                raise ValueError("MATR supervision Parquet SHA-256 mismatch")
            _validate_component_supervision(
                component=component,
                conversion=conversion,
                eligibility=eligible,
                supervision=supervision_artifact,
            )
            eligibility.append(eligible)
            supervision.append((supervision_artifact, parquet_path))

        try:
            reconstructed = combine_matr_batch_splits(tuple(component_splits))
        except ValueError as exc:
            raise ValueError(
                "component split calibration inventory is invalid"
            ) from exc
        if reconstructed != combined:
            raise ValueError(
                "component splits do not reconstruct the combined split calibration inventory"
            )
        expected_cells = tuple(combined.calibration)
        if not expected_cells or len(set(expected_cells)) != len(expected_cells):
            raise ValueError("combined split calibration cells must be nonempty and unique")

        source_identity = _source_identity(
            registration=registration,
            manifest=manifest,
            supervision=supervision,
        )
        if normalized_task is AdvancedModelTask.RUL:
            return AdvancedRULCalibrationEvidence(
                source_identity=source_identity,
                cells=_rul_cells(
                    expected_cells=expected_cells,
                    conversions=conversions,
                    cutoff_cycle=cutoff_cycle,
                ),
            )
        return AdvancedSOHCalibrationEvidence(
            source_identity=source_identity,
            prediction_cycles=tuple(range(cutoff_cycle + 1, _SOH_HORIZON_CYCLE + 1)),
            cells=_soh_cells(
                expected_cells=expected_cells,
                component_splits=component_splits,
                eligibility=eligibility,
                supervision=supervision,
                cutoff_cycle=cutoff_cycle,
            ),
        )


def _verified_registration(
    registration: AdvancedCalibrationSourceRegistration,
) -> AdvancedCalibrationSourceRegistration:
    root = Path(registration.evidence_root)
    if root.is_symlink():
        raise ValueError("Advanced calibration evidence root must not be a symbolic link")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("Advanced calibration evidence root must be a directory")
    return registration.model_copy(update={"evidence_root": resolved})


def _require_root(root: Path) -> Path:
    if root.is_symlink():
        raise ValueError("Advanced calibration evidence root must not be a symbolic link")
    resolved = root.resolve(strict=True)
    if resolved != root or not resolved.is_dir():
        raise ValueError("Advanced calibration evidence root identity changed")
    return resolved


def _registered_path(root: Path, relative: str | PurePosixPath) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or "\\" in str(relative)
    ):
        raise ValueError("registered calibration path escapes the evidence root")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise ValueError("registered calibration path must not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("registered calibration path escapes the evidence root")
    return resolved


def _child_path(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("registered calibration directory must not be a symbolic link")
    resolved_root = root.resolve(strict=True)
    path = _registered_path(resolved_root, relative)
    if not path.is_relative_to(resolved_root):
        raise ValueError("registered calibration child path escapes its source root")
    return path


def _read_verified_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{path.name} SHA-256 mismatch")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_component_conversion(
    component: Any,
    conversion: MatrBatchConversionReport,
    split: SplitManifest,
) -> None:
    evidence_ids = tuple(cell.cell_id for cell in conversion.cells)
    if (
        conversion.dataset_id != "MATR"
        or split.dataset_id != "MATR"
        or conversion.batch_index != component.batch_index
        or conversion.batch_date != component.batch_date
        or conversion.raw_sha256 != component.raw_sha256
        or conversion.cell_count != component.cell_count
        or len(evidence_ids) != conversion.cell_count
        or len(set(evidence_ids)) != len(evidence_ids)
        or set(split.all_cells) != set(evidence_ids)
    ):
        raise ValueError("MATR component conversion and split inventory mismatch")
    scalar_count = sum(
        not cell.official_life_right_censored
        and cell.official_life_label is not None
        for cell in conversion.cells
    )
    if scalar_count != component.scalar_label_count:
        raise ValueError(
            "MATR component non-censored official cycle life inventory mismatch"
        )


def _validate_component_supervision(
    *,
    component: Any,
    conversion: MatrBatchConversionReport,
    eligibility: MatrTrajectoryEligibilityAudit,
    supervision: MatrSupervisionArtifact,
) -> None:
    conversion_by_id = {cell.cell_id: cell for cell in conversion.cells}
    supervision_ids = tuple(cell.cell_id for cell in supervision.cells)
    eligible_ids = tuple(eligibility.eligible_cell_ids)
    if (
        eligibility.batch_index != component.batch_index
        or eligibility.horizon_cycle != _SOH_HORIZON_CYCLE
        or supervision.dataset_id != "MATR"
        or supervision.horizon_cycle != _SOH_HORIZON_CYCLE
        or supervision.raw_sha256 != conversion.raw_sha256
        or supervision.source_report_sha256
        != sha256_canonical(conversion.model_dump(mode="json"))
        or len(set(supervision_ids)) != len(supervision_ids)
        or set(supervision_ids) != set(eligible_ids)
        or len(eligible_ids) != component.hybrid_eligible_count
        or len(eligibility.excluded) != component.hybrid_excluded_count
    ):
        raise ValueError("MATR supervision evidence inventory mismatch")
    for cell in supervision.cells:
        conversion_cell = conversion_by_id.get(cell.cell_id)
        if conversion_cell is None or (
            cell.raw_cell_id != conversion_cell.raw_cell_id
            or cell.official_life_label != conversion_cell.official_life_label
            or cell.official_life_right_censored
            != conversion_cell.official_life_right_censored
            or not math.isclose(
                cell.reference_capacity_ah,
                conversion_cell.reference_capacity_ah,
            )
        ):
            raise ValueError("MATR supervision cell differs from conversion evidence")


def _rul_cells(
    *,
    expected_cells: tuple[str, ...],
    conversions: Sequence[MatrBatchConversionReport],
    cutoff_cycle: int,
) -> tuple[AdvancedRULObservedCell, ...]:
    by_id = {
        cell.cell_id: cell
        for conversion in conversions
        for cell in conversion.cells
    }
    if set(expected_cells) - set(by_id):
        raise ValueError("RUL calibration cells are missing from conversion evidence")
    output: list[AdvancedRULObservedCell] = []
    for cell_id in expected_cells:
        cell = by_id[cell_id]
        if cell.official_life_right_censored or cell.official_life_label is None:
            raise ValueError(
                "RUL calibration requires non-censored official cycle life evidence"
            )
        if cell.official_life_label <= cutoff_cycle:
            raise ValueError("RUL official cycle life must be greater than cutoff")
        output.append(
            AdvancedRULObservedCell(
                cell_id=cell_id,
                observed_cycle=cell.official_life_label,
            )
        )
    return tuple(output)


def _soh_cells(
    *,
    expected_cells: tuple[str, ...],
    component_splits: Sequence[SplitManifest],
    eligibility: Sequence[MatrTrajectoryEligibilityAudit],
    supervision: Sequence[tuple[MatrSupervisionArtifact, Path]],
    cutoff_cycle: int,
) -> tuple[AdvancedSOHObservedCell, ...]:
    observed: dict[str, tuple[float, ...]] = {}
    expected_axis = tuple(range(cutoff_cycle + 1, _SOH_HORIZON_CYCLE + 1))
    eligible_ids = {
        cell_id
        for audit in eligibility
        for cell_id in audit.eligible_cell_ids
    }
    task_expected_cells = tuple(
        cell_id for cell_id in expected_cells if cell_id in eligible_ids
    )
    if not task_expected_cells:
        raise ValueError("SOH calibration has no verified eligible cells")
    for split, eligible, (artifact, parquet_path) in zip(
        component_splits,
        eligibility,
        supervision,
        strict=True,
    ):
        calibration = tuple(
            cell_id
            for cell_id in split.calibration
            if cell_id in set(eligible.eligible_cell_ids)
        )
        rows = _read_supervision_rows(parquet_path, artifact)
        for cell_id in calibration:
            cell_rows = rows.get(cell_id)
            if cell_rows is None:
                raise ValueError("SOH calibration cell is missing from supervision Parquet")
            cycles = tuple(item[0] for item in cell_rows)
            if len(set(cycles)) != len(cycles):
                raise ValueError("SOH supervision cycles must be unique per cell")
            if cycles != tuple(range(1, _SOH_HORIZON_CYCLE + 1)):
                raise ValueError(
                    "SOH supervision cycles must be continuous from 1 through 500"
                )
            selected = {
                cycle: soh
                for cycle, soh in cell_rows
                if cutoff_cycle < cycle <= _SOH_HORIZON_CYCLE
            }
            if tuple(selected) != expected_axis:
                raise ValueError("SOH observations do not match the finite prediction axis")
            observed[cell_id] = tuple(selected[cycle] for cycle in expected_axis)
    if (
        tuple(cell_id for cell_id in task_expected_cells if cell_id in observed)
        != task_expected_cells
    ):
        raise ValueError(
            "SOH calibration cells do not match the eligible combined split projection"
        )
    if set(observed) != set(task_expected_cells):
        raise ValueError("SOH calibration supervision contains an unexpected cell")
    return tuple(
        AdvancedSOHObservedCell(
            cell_id=cell_id,
            observed_soh=observed[cell_id],
        )
        for cell_id in task_expected_cells
    )


def _read_supervision_rows(
    path: Path,
    artifact: MatrSupervisionArtifact,
) -> Mapping[str, list[tuple[int, float]]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Advanced calibration SOH evidence requires the data dependencies"
        ) from exc
    table = pq.read_table(path)
    if (
        set(table.column_names) != _SUPERVISION_COLUMNS
        or table.num_rows != artifact.row_count
    ):
        raise ValueError("MATR supervision Parquet schema or row count mismatch")
    rows: dict[str, list[tuple[int, float]]] = {}
    for item in table.to_pylist():
        dataset_id = item["dataset_id"]
        cell_id = item["cell_id"]
        cycle = item["cycle_index"]
        capacity = item["discharge_capacity_ah"]
        reference = item["reference_capacity_ah"]
        soh = item["soh"]
        if (
            dataset_id != "MATR"
            or not isinstance(cell_id, str)
            or not cell_id
            or not isinstance(cycle, int)
            or isinstance(cycle, bool)
            or not all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in (capacity, reference, soh)
            )
        ):
            raise ValueError("MATR supervision values must be finite and typed")
        capacity_value = float(capacity)
        reference_value = float(reference)
        soh_value = float(soh)
        if (
            cycle < 1
            or cycle > _SOH_HORIZON_CYCLE
            or capacity_value <= 0.0
            or reference_value <= 0.0
            or soh_value <= 0.0
            or not math.isclose(
                soh_value,
                capacity_value / reference_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "MATR supervision requires finite, continuous, "
                "capacity-derived SOH evidence through cycle 500"
            )
        rows.setdefault(cell_id, []).append((cycle, soh_value))
    expected_ids = {cell.cell_id for cell in artifact.cells}
    if set(rows) != expected_ids:
        raise ValueError("MATR supervision Parquet cell inventory mismatch")
    return rows


def _source_identity(
    *,
    registration: AdvancedCalibrationSourceRegistration,
    manifest: MatrThreeBatchManifest,
    supervision: Sequence[tuple[MatrSupervisionArtifact, Path]],
) -> AdvancedCalibrationSourceIdentity:
    payload: dict[str, object] = {
        "registration_id": registration.registration_id,
        "dataset_id": "MATR",
        "data_version": manifest.data_version,
        "split_version": manifest.split_version,
        "three_batch_manifest_sha256": registration.three_batch_manifest_sha256,
        "combined_split_sha256": manifest.combined_split_sha256,
        "conversion_report_sha256s": tuple(
            component.conversion_report_sha256 for component in manifest.batches
        ),
        "component_split_sha256s": tuple(
            component.split_manifest_sha256 for component in manifest.batches
        ),
        "eligibility_report_sha256s": tuple(
            component.eligibility_report_sha256 for component in manifest.batches
        ),
        "supervision_report_sha256s": tuple(
            component.supervision_report_sha256 for component in manifest.batches
        ),
        "supervision_parquet_sha256s": tuple(
            item[0].parquet_sha256 for item in supervision
        ),
    }
    payload["source_identity_sha256"] = sha256_canonical(payload)
    return AdvancedCalibrationSourceIdentity.model_validate(payload)
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AdvancedCalibrationEvidence",
    "AdvancedCalibrationEvidenceResolver",
    "AdvancedCalibrationSourceIdentity",
    "AdvancedCalibrationSourceRegistration",
    "AdvancedRULCalibrationEvidence",
    "AdvancedRULObservedCell",
    "AdvancedSOHCalibrationEvidence",
    "AdvancedSOHObservedCell",
    "RegisteredAdvancedCalibrationEvidenceResolver",
]
