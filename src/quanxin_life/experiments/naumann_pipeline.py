"""Deterministic, provenance-preserving Naumann GP replay pipeline.

This pipeline consumes only condition observations already produced by the
reviewed Naumann adapters.  It never reads pickle-like files, infers physical
conditions from names, invents resource costs, or emits business decisions.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator

from quanxin_life.core.hashing import canonical_json_bytes, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.adapters.naumann_calendar import CalendarCapacityObservation
from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.experiments.gp_active import (
    GP_MODEL_VERSION,
    AcquisitionConfig,
    OperatingBounds,
    evaluate_finite_pool_replay,
)
from quanxin_life.experiments.naumann_bridge import (
    BridgedExperimentObservation,
    NaumannConditionObservation,
    NaumannGpBridgeMapping,
    bridge_naumann_observations,
)

NAUMANN_PIPELINE_VERSION = "naumann-gp-replay-pipeline-v1"
_DATASET_JSON = "experiment_dataset.json"
_DATASET_CSV = "experiment_dataset.csv"
_REPLAY_JSON = "gp_replay.json"
_RUN_MANIFEST_JSON = "run_manifest.json"
_MAX_REQUEST_BYTES = 64 * 1024 * 1024


class NaumannPipelineStatus(StrEnum):
    COMPLETED = "completed"
    DEGRADED_MISSING_REVIEWED_RESOURCES = "degraded_missing_reviewed_resources"


class NaumannOperatingBounds(ContractModel):
    temperature_c: tuple[float, float]
    mean_soc: tuple[float, float]
    dod: tuple[float, float]
    charge_c_rate: tuple[float, float]
    discharge_c_rate: tuple[float, float]

    def to_domain(self) -> OperatingBounds:
        return OperatingBounds(**self.model_dump())


class NaumannAcquisitionConfig(ContractModel):
    time_normalizer_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost_normalizer: float = Field(gt=0, allow_inf_nan=False)
    duplicate_penalty_weight: float = Field(ge=0, allow_inf_nan=False)
    similarity_length_scale: float = Field(gt=0, allow_inf_nan=False)

    def to_domain(self) -> AcquisitionConfig:
        return AcquisitionConfig(**self.model_dump())


class NaumannPipelineRequest(ContractModel):
    """Reviewed inputs and explicit replay configuration for one run."""

    pipeline_config_version: str = Field(min_length=1)
    observations: tuple[CycleMatrixObservation | CalendarCapacityObservation, ...] = Field(
        min_length=1
    )
    mapping: NaumannGpBridgeMapping | None = None
    operating_bounds: NaumannOperatingBounds
    acquisition_config: NaumannAcquisitionConfig
    initial_observation_ids: tuple[str, ...] = ()
    initial_condition_ids: tuple[str, ...] = ()
    query_budget: int = Field(gt=0)

    @field_validator("pipeline_config_version")
    @classmethod
    def config_version_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("pipeline_config_version must not be blank")
        return normalized

    @model_validator(mode="after")
    def initial_cohort_is_explicit_and_unambiguous(self) -> NaumannPipelineRequest:
        has_observation_ids = bool(self.initial_observation_ids)
        has_condition_ids = bool(self.initial_condition_ids)
        if has_observation_ids == has_condition_ids:
            raise ValueError(
                "exactly one explicit initial observation cohort must be supplied"
            )
        selected = self.initial_observation_ids or self.initial_condition_ids
        if len(selected) < 2:
            raise ValueError("initial observation cohort must contain at least two IDs")
        if len(selected) != len(set(selected)):
            raise ValueError("initial observation cohort IDs must be unique")
        return self


class NaumannArtifact(ContractModel):
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0)


class NaumannPipelineArtifacts(ContractModel):
    dataset_json: NaumannArtifact
    dataset_csv: NaumannArtifact
    replay_json: NaumannArtifact


class NaumannPipelineResult(ContractModel):
    status: NaumannPipelineStatus
    run_id: Sha256
    config_hash: Sha256
    source_sha256s: tuple[Sha256, ...]
    manifest_sha256: Sha256
    artifacts: NaumannPipelineArtifacts | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def artifacts_match_status(self) -> NaumannPipelineResult:
        if self.status is NaumannPipelineStatus.COMPLETED and self.artifacts is None:
            raise ValueError("completed Naumann pipeline requires artifacts")
        if (
            self.status is NaumannPipelineStatus.DEGRADED_MISSING_REVIEWED_RESOURCES
            and self.artifacts is not None
        ):
            raise ValueError("degraded Naumann pipeline must not claim replay artifacts")
        return self


def load_naumann_pipeline_request(path: Path) -> NaumannPipelineRequest:
    """Load a strict JSON request; executable serialization is never accepted."""

    request_path = Path(path)
    if request_path.suffix.lower() != ".json":
        raise ValueError("Naumann pipeline request must use a .json file")
    if not request_path.is_file():
        raise ValueError("Naumann pipeline request does not exist")
    if request_path.stat().st_size > _MAX_REQUEST_BYTES:
        raise ValueError("Naumann pipeline request exceeds the size limit")
    try:
        payload = json.loads(
            request_path.read_bytes(),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Naumann pipeline request must contain valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Naumann pipeline request must contain a JSON object")
    return NaumannPipelineRequest.model_validate(payload)


class _DatasetObservation(ContractModel):
    observation_id: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_condition_id: str = Field(min_length=1)
    source_observation_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    source_sha256: Sha256
    source_axis_name: str = Field(min_length=1)
    source_axis_value: float = Field(ge=0, allow_inf_nan=False)
    source_metric_name: str = Field(min_length=1)
    source_metric_value: float = Field(ge=0, allow_inf_nan=False)
    layout_version: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    bridge_version: str = Field(min_length=1)
    mapping_version: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    observed_target: float = Field(allow_inf_nan=False)
    temperature_c: float = Field(allow_inf_nan=False)
    mean_soc: float = Field(allow_inf_nan=False)
    dod: float = Field(allow_inf_nan=False)
    charge_c_rate: float = Field(allow_inf_nan=False)
    discharge_c_rate: float = Field(allow_inf_nan=False)
    duration_hours: float = Field(gt=0, allow_inf_nan=False)
    equipment_cost: float = Field(gt=0, allow_inf_nan=False)
    resource_review_statement: str = Field(min_length=1)
    resource_evidence_reference: str = Field(min_length=1)


def run_naumann_gp_pipeline(
    request: NaumannPipelineRequest,
    *,
    output_dir: Path,
) -> NaumannPipelineResult:
    """Build traceable GP inputs, run finite-pool replay and export artifacts."""

    validated = NaumannPipelineRequest.model_validate(request.model_dump(mode="json"))
    source_sha256s = tuple(sorted({item.source_sha256 for item in validated.observations}))
    config_payload = _config_payload(validated)
    config_hash = sha256_canonical(config_payload)
    run_id = sha256_canonical(
        {
            "pipeline_version": NAUMANN_PIPELINE_VERSION,
            "config_hash": config_hash,
            "source_observation_hashes": [
                sha256_canonical(item.model_dump(mode="json"))
                for item in sorted(
                    validated.observations,
                    key=lambda item: (
                        item.dataset_id,
                        item.condition_id,
                        _observation_axis_value(item),
                    ),
                )
            ],
        }
    )
    output_path = Path(output_dir)
    _prepare_output_dir(output_path)

    if validated.mapping is None:
        warning = "REVIEWED_EXPERIMENT_RESOURCES_REQUIRED_FOR_REPLAY"
        manifest = {
            "pipeline_version": NAUMANN_PIPELINE_VERSION,
            "status": NaumannPipelineStatus.DEGRADED_MISSING_REVIEWED_RESOURCES.value,
            "run_id": run_id,
            "config_hash": config_hash,
            "configuration": config_payload,
            "source_sha256s": list(source_sha256s),
            "source_observation_count": len(validated.observations),
            "artifacts": [],
            "warnings": [warning],
        }
        manifest_bytes = canonical_json_bytes(manifest)
        _write_bytes(output_path / _RUN_MANIFEST_JSON, manifest_bytes)
        return NaumannPipelineResult(
            status=NaumannPipelineStatus.DEGRADED_MISSING_REVIEWED_RESOURCES,
            run_id=run_id,
            config_hash=config_hash,
            source_sha256s=source_sha256s,
            manifest_sha256=_sha256_bytes(manifest_bytes),
            warnings=[warning],
        )

    bridged = bridge_naumann_observations(validated.observations, mapping=validated.mapping)
    source_by_observation_id = {
        bridged_item.experiment_observation.observation_id: source
        for source, bridged_item in zip(validated.observations, bridged, strict=True)
    }
    ordered = tuple(
        sorted(bridged, key=lambda item: item.experiment_observation.observation_id)
    )
    initial_ids = _resolve_initial_ids(validated, ordered)
    replay = evaluate_finite_pool_replay(
        [item.experiment_observation for item in ordered],
        operating_bounds=validated.operating_bounds.to_domain(),
        acquisition_config=validated.acquisition_config.to_domain(),
        initial_observation_ids=initial_ids,
        query_budget=validated.query_budget,
    )
    dataset_rows = tuple(
        _dataset_observation(
            item,
            source=source_by_observation_id[item.experiment_observation.observation_id],
        )
        for item in ordered
    )
    dataset_payload = {
        "pipeline_version": NAUMANN_PIPELINE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "configuration": config_payload,
        "source_sha256s": list(source_sha256s),
        "observations": [row.model_dump(mode="json") for row in dataset_rows],
    }
    dataset_json_bytes = canonical_json_bytes(dataset_payload)
    dataset_json_artifact = _write_artifact(output_path, _DATASET_JSON, dataset_json_bytes)
    dataset_csv_artifact = _write_artifact(
        output_path,
        _DATASET_CSV,
        _dataset_csv_bytes(dataset_rows, config_hash=config_hash),
    )
    replay_payload = {
        "pipeline_version": NAUMANN_PIPELINE_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset_sha256": dataset_json_artifact.sha256,
        "source_sha256s": list(source_sha256s),
        **asdict(replay),
    }
    replay_artifact = _write_artifact(
        output_path,
        _REPLAY_JSON,
        canonical_json_bytes(replay_payload),
    )
    artifacts = NaumannPipelineArtifacts(
        dataset_json=dataset_json_artifact,
        dataset_csv=dataset_csv_artifact,
        replay_json=replay_artifact,
    )
    manifest = {
        "pipeline_version": NAUMANN_PIPELINE_VERSION,
        "status": NaumannPipelineStatus.COMPLETED.value,
        "run_id": run_id,
        "config_hash": config_hash,
        "source_sha256s": list(source_sha256s),
        "source_observation_count": len(validated.observations),
        "gp_model_version": GP_MODEL_VERSION,
        "artifacts": artifacts.model_dump(mode="json"),
        "warnings": [],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    _write_bytes(output_path / _RUN_MANIFEST_JSON, manifest_bytes)
    return NaumannPipelineResult(
        status=NaumannPipelineStatus.COMPLETED,
        run_id=run_id,
        config_hash=config_hash,
        source_sha256s=source_sha256s,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        artifacts=artifacts,
    )


def _config_payload(request: NaumannPipelineRequest) -> dict[str, Any]:
    return {
        "pipeline_version": NAUMANN_PIPELINE_VERSION,
        "pipeline_config_version": request.pipeline_config_version,
        "mapping": (
            request.mapping.model_dump(mode="json") if request.mapping is not None else None
        ),
        "operating_bounds": request.operating_bounds.model_dump(mode="json"),
        "acquisition_config": request.acquisition_config.model_dump(mode="json"),
        "initial_observation_ids": list(request.initial_observation_ids),
        "initial_condition_ids": list(request.initial_condition_ids),
        "query_budget": request.query_budget,
    }


def _observation_axis_value(observation: NaumannConditionObservation) -> float:
    if isinstance(observation, CycleMatrixObservation):
        return observation.observation_value
    return observation.storage_time_h


def _resolve_initial_ids(
    request: NaumannPipelineRequest,
    observations: tuple[BridgedExperimentObservation, ...],
) -> tuple[str, ...]:
    if request.initial_observation_ids:
        return request.initial_observation_ids
    by_condition: dict[str, list[str]] = {}
    for item in observations:
        by_condition.setdefault(item.provenance.source_condition_id, []).append(
            item.experiment_observation.observation_id
        )
    resolved: list[str] = []
    for condition_id in request.initial_condition_ids:
        matches = by_condition.get(condition_id, [])
        if len(matches) != 1:
            raise ValueError(
                "each initial_condition_id must resolve to exactly one source observation"
            )
        resolved.append(matches[0])
    return tuple(resolved)


def _dataset_observation(
    item: BridgedExperimentObservation,
    *,
    source: NaumannConditionObservation,
) -> _DatasetObservation:
    observation = item.experiment_observation
    provenance = item.provenance
    condition = observation.condition
    return _DatasetObservation(
        observation_id=observation.observation_id,
        source_dataset_id=provenance.source_dataset_id,
        source_condition_id=provenance.source_condition_id,
        source_observation_id=provenance.source_observation_id,
        source_file=provenance.source_file,
        source_sha256=provenance.source_sha256,
        source_axis_name=(
            source.observation_axis
            if isinstance(source, CycleMatrixObservation)
            else "storage_time_h"
        ),
        source_axis_value=_observation_axis_value(source),
        source_metric_name=(
            source.metric_name if isinstance(source, CycleMatrixObservation) else "capacity_ah"
        ),
        source_metric_value=(
            source.metric_value
            if isinstance(source, CycleMatrixObservation)
            else source.capacity_ah
        ),
        layout_version=provenance.layout_version,
        adapter_version=provenance.adapter_version,
        bridge_version=provenance.bridge_version,
        mapping_version=provenance.mapping_version,
        target_name=observation.target_name,
        observed_target=observation.observed_target,
        temperature_c=condition.temperature_c,
        mean_soc=condition.mean_soc,
        dod=condition.dod,
        charge_c_rate=condition.charge_c_rate,
        discharge_c_rate=condition.discharge_c_rate,
        duration_hours=observation.duration_hours,
        equipment_cost=observation.equipment_cost,
        resource_review_statement=provenance.resource_review_statement,
        resource_evidence_reference=provenance.resource_evidence_reference,
    )


def _dataset_csv_bytes(
    rows: tuple[_DatasetObservation, ...], *, config_hash: str
) -> bytes:
    fieldnames = ("config_hash", *tuple(_DatasetObservation.model_fields))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({"config_hash": config_hash, **row.model_dump(mode="json")})
    return stream.getvalue().encode("utf-8")


def _write_artifact(output_dir: Path, relative_path: str, payload: bytes) -> NaumannArtifact:
    _write_bytes(output_dir / relative_path, payload)
    return NaumannArtifact(
        relative_path=relative_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
    )


def _prepare_output_dir(path: Path) -> None:
    known_artifacts = (_DATASET_JSON, _DATASET_CSV, _REPLAY_JSON, _RUN_MANIFEST_JSON)
    if path.exists() and any((path / name).exists() for name in known_artifacts):
        raise ValueError("output directory already contains Naumann pipeline artifacts")
    path.mkdir(parents=True, exist_ok=True)


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result
