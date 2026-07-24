"""Export SHA-bound deployment bundles from approved Advanced Final routes."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid5

import torch
from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.application.deep_model_artifacts import (
    DeepArtifactFile,
    DeepArtifactFileRole,
    DeepArtifactKind,
    DeepModelArtifactManifest,
    export_current_hybrid_artifact,
    export_cyclepatch_batlinet_artifact,
    export_cyclepatch_direct_artifact,
    export_hybridpatch_v2_artifact,
    load_current_hybrid_artifact,
    load_cyclepatch_batlinet_artifact,
    load_cyclepatch_direct_artifact,
    load_hybridpatch_v2_artifact,
    verify_deep_model_artifact,
)
from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.early_cycle_sequence import VARIABLE_NAMES
from quanxin_life.models.cyclepatch import EarlyCycleBatch
from quanxin_life.training.advanced_config import (
    AdvancedMatrThreeBatchRunConfig,
    AdvancedRunKey,
    build_advanced_run_matrix,
)
from quanxin_life.training.advanced_data import (
    AdvancedFinalMatrData,
    load_advanced_matr_final_data,
)
from quanxin_life.training.advanced_orchestrator import (
    ADVANCED_FEATURE_VERSION,
    _build_training_task,
    _candidate_for_key,
    _current_hybrid_batch,
    _normalization_hash,
)
from quanxin_life.training.advanced_outputs import (
    load_advanced_training_output_index,
    verify_advanced_training_output_index,
)
from quanxin_life.training.checkpoint import (
    AdvancedTrainingCheckpointManifest,
    load_advanced_inference_checkpoint,
    model_architecture_sha256,
)

_INDEX_NAME = "deployment_bundle_index.json"
_EXPECTED_ROUTE_COUNT = 15
_EXPECTED_ROUTE_MATRIX = frozenset(
    {
        ("RUL", 20, "DEFAULT"),
        ("RUL", 50, "POINT_ACCURACY"),
        ("RUL", 50, "COVERAGE"),
        ("RUL", 100, "POINT_ACCURACY"),
        ("RUL", 100, "COVERAGE"),
        ("RUL", 150, "POINT_ACCURACY"),
        ("RUL", 150, "COVERAGE"),
    }
    | {
        ("SOH", cutoff, role)
        for cutoff in (20, 50, 100, 150)
        for role in ("MEAN_ACCURACY", "TAIL_EFFICIENCY")
    }
)
_APPROVED_ROLES = frozenset(
    {
        "DEFAULT",
        "POINT_ACCURACY",
        "COVERAGE",
        "MEAN_ACCURACY",
        "TAIL_EFFICIENCY",
    }
)


@dataclass(frozen=True, slots=True)
class _RebuildData:
    config: AdvancedMatrThreeBatchRunConfig
    manifest: MatrThreeBatchManifest
    split: SplitManifest
    data: AdvancedFinalMatrData
    source_commit: str


class AdvancedDeploymentSourceRoute(ContractModel):
    """One promotion route bound to one concrete best checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: Literal["RUL", "SOH"]
    role: Literal[
        "DEFAULT",
        "POINT_ACCURACY",
        "COVERAGE",
        "MEAN_ACCURACY",
        "TAIL_EFFICIENCY",
    ]
    disposition: Literal["CONDITIONAL"] = "CONDITIONAL"
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    family: Literal[
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "current_hybrid",
        "hybridpatch_v2",
    ]
    candidate_id: str = Field(min_length=1)
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoff_cycle: Literal[20, 50, 100, 150]
    seed: Literal[38, 39, 40, 41, 42]
    best_epoch: int = Field(gt=0)
    representative_seed_rule: Literal["MINIMUM_BEST_VALIDATION_METRIC"]
    run_id: str = Field(min_length=1)
    checkpoint_directory: str = Field(min_length=1)
    checkpoint_manifest_sha256: Sha256
    checkpoint_manifest_file_sha256: Sha256
    checkpoint_model_sha256: Sha256
    checkpoint_context_sha256: Sha256
    selection_manifest_sha256: Sha256
    candidate_config_sha256: Sha256
    normalization_sha256: Sha256
    reference_library_sha256: Sha256 | None = None
    deep_artifact_id: str = Field(min_length=1)
    deep_artifact_kind: DeepArtifactKind
    deep_artifact_manifest_sha256: Sha256

    @field_validator("checkpoint_directory")
    @classmethod
    def checkpoint_path_is_safe(cls, value: str) -> str:
        path = Path(value.replace("\\", "/"))
        if (
            path.is_absolute()
            or path.drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint_directory must be result-root relative")
        return path.as_posix()

    @model_validator(mode="after")
    def task_matches_family_and_role(self) -> AdvancedDeploymentSourceRoute:
        rul = self.family in {"cyclepatch_direct", "cyclepatch_batlinet"}
        if (self.task == "RUL") != rul:
            raise ValueError("deployment route task and family do not match")
        if self.task == "RUL" and self.role not in {
            "DEFAULT",
            "POINT_ACCURACY",
            "COVERAGE",
        }:
            raise ValueError("RUL deployment route has an invalid role")
        if self.task == "SOH" and self.role not in {
            "MEAN_ACCURACY",
            "TAIL_EFFICIENCY",
        }:
            raise ValueError("SOH deployment route has an invalid role")
        is_batlinet = self.family == "cyclepatch_batlinet"
        if is_batlinet != (self.reference_library_sha256 is not None):
            raise ValueError("reference_library_sha256 is required only for BatLiNet")
        return self


class AdvancedDeploymentArtifact(ContractModel):
    """One de-duplicated deep artifact and its source checkpoint identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str = Field(min_length=1)
    artifact_kind: DeepArtifactKind
    artifact_manifest_sha256: Sha256
    source_checkpoint_manifest_sha256: Sha256
    source_checkpoint_model_sha256: Sha256
    dataset_id: Literal["MATR"] = "MATR"
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    cutoff_cycle: Literal[20, 50, 100, 150]
    candidate_config_sha256: Sha256
    normalization_sha256: Sha256
    target_scaler_context_sha256: Sha256 | None = None
    reference_library_sha256: Sha256 | None = None
    files: tuple[DeepArtifactFile, ...] = Field(min_length=3, max_length=4)
    round_trip_verified: Literal[True] = True

    @model_validator(mode="after")
    def optional_context_matches_kind(self) -> AdvancedDeploymentArtifact:
        weights = [
            item for item in self.files if item.role is DeepArtifactFileRole.WEIGHTS
        ]
        if (
            len(weights) != 1
            or weights[0].sha256 != self.source_checkpoint_model_sha256
        ):
            raise ValueError("artifact weights do not match the source checkpoint")
        is_rul = self.artifact_kind in {
            DeepArtifactKind.CYCLEPATCH_DIRECT,
            DeepArtifactKind.CYCLEPATCH_BATLINET,
        }
        if is_rul != (self.target_scaler_context_sha256 is not None):
            raise ValueError("target scaler context is required only for RUL artifacts")
        is_batlinet = self.artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET
        if is_batlinet != (self.reference_library_sha256 is not None):
            raise ValueError("reference library context is required only for BatLiNet")
        return self


class AdvancedDeploymentBundleIndex(ContractModel):
    """Closed-world deployment output that remains inactive until approval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["advanced-deployment-bundle-index-v1"] = (
        "advanced-deployment-bundle-index-v1"
    )
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    created_at: datetime
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    final_output_sha256: Sha256
    final_config_sha256: Sha256
    data_version: str = Field(min_length=1)
    split_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    training_input_bundle_sha256: Sha256
    local_reconstructed_input_bundle_sha256: Sha256
    input_bundle_hashes_match: bool
    promotion_manifest_sha256: Sha256
    promotion_decisions_sha256: Sha256
    promotion_source_evidence_sha256: Sha256
    selection_manifest_sha256: Sha256
    routes: tuple[AdvancedDeploymentSourceRoute, ...] = Field(min_length=1)
    artifacts: tuple[AdvancedDeploymentArtifact, ...] = Field(min_length=1)
    manifest_sha256: Sha256

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def index_is_closed_and_hash_bound(self) -> AdvancedDeploymentBundleIndex:
        route_keys = {(row.task, row.cutoff_cycle, row.role) for row in self.routes}
        if len(route_keys) != len(self.routes):
            raise ValueError("deployment index contains a duplicate route coordinate")
        artifact_ids = {item.artifact_id for item in self.artifacts}
        if len(artifact_ids) != len(self.artifacts):
            raise ValueError("deployment index contains duplicate artifacts")
        if {row.deep_artifact_id for row in self.routes} != artifact_ids:
            raise ValueError("deployment routes and artifacts do not form a closed set")
        artifact_by_id = {item.artifact_id: item for item in self.artifacts}
        expected_kinds = {
            "cyclepatch_direct": DeepArtifactKind.CYCLEPATCH_DIRECT,
            "cyclepatch_batlinet": DeepArtifactKind.CYCLEPATCH_BATLINET,
            "current_hybrid": DeepArtifactKind.CURRENT_HYBRID,
            "hybridpatch_v2": DeepArtifactKind.HYBRIDPATCH_V2,
        }
        for route in self.routes:
            artifact = artifact_by_id[route.deep_artifact_id]
            if (
                route.deep_artifact_kind is not expected_kinds[route.family]
                or artifact.artifact_kind is not route.deep_artifact_kind
                or artifact.artifact_manifest_sha256
                != route.deep_artifact_manifest_sha256
                or artifact.source_checkpoint_manifest_sha256
                != route.checkpoint_manifest_sha256
                or artifact.source_checkpoint_model_sha256
                != route.checkpoint_model_sha256
                or artifact.cutoff_cycle != route.cutoff_cycle
                or artifact.data_version != route.data_version
                or artifact.split_version != route.split_version
                or artifact.feature_version != route.feature_version
                or artifact.candidate_config_sha256
                != route.candidate_config_sha256
                or artifact.normalization_sha256 != route.normalization_sha256
                or artifact.reference_library_sha256
                != route.reference_library_sha256
            ):
                raise ValueError("deployment route and artifact contexts do not match")
            if route.selection_manifest_sha256 != self.selection_manifest_sha256:
                raise ValueError("deployment routes do not share one selection manifest")
        if any(
            item.data_version != self.data_version
            or item.split_version != self.split_version
            or item.feature_version != self.feature_version
            for item in self.artifacts
        ):
            raise ValueError("deployment artifacts do not share the indexed versions")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if sha256_canonical(payload) != self.manifest_sha256:
            raise ValueError("deployment bundle manifest_sha256 does not match")
        return self


def export_advanced_deployment_bundles(
    project_root: Path,
    result_root: Path,
    promotion_root: Path,
    output_root: Path,
    *,
    expected_promotion_manifest_sha256: str,
    resume_artifact_root: Path | None = None,
) -> AdvancedDeploymentBundleIndex:
    """Export all representative routes without activating any model."""

    project = _regular_directory(project_root, "project root")
    results = _regular_directory(result_root, "result root")
    promotion = _regular_directory(promotion_root, "promotion root")
    output = output_root.resolve(strict=False)
    if output.exists() or output.is_symlink():
        raise ValueError("deployment output root already exists")

    final_root = _regular_directory(
        results / "runs" / "a100" / "matr-three-batch" / "advanced" / "final",
        "Advanced Final root",
    )
    final_index = load_advanced_training_output_index(final_root / "output_index.json")
    if final_index.mode != "final" or final_index.operation_count != 80:
        raise ValueError("deployment export requires the complete Advanced Final index")
    verify_advanced_training_output_index(final_root, final_index)

    promotion_manifest_path = promotion / "artifact_manifest.json"
    expected_promotion_sha256 = _sha256(
        expected_promotion_manifest_sha256,
        "expected promotion manifest SHA-256",
    )
    if _sha256_file(promotion_manifest_path) != expected_promotion_sha256:
        raise ValueError("promotion manifest differs from the expected promotion SHA-256")
    promotion_manifest = _verify_promotion_manifest(
        promotion,
        promotion_manifest_path,
    )
    registered = {
        str(item["relative_path"]): item for item in promotion_manifest["files"]
    }
    decisions_item = registered.get("promotion_decisions.csv")
    source_item = registered.get("source_evidence.json")
    if decisions_item is None or source_item is None:
        raise ValueError("promotion manifest is missing route or source evidence")
    decisions_path = promotion / "promotion_decisions.csv"
    source_path = promotion / "source_evidence.json"
    source_evidence = _strict_json(source_path)
    provenance = source_evidence.get("source_provenance")
    if not isinstance(provenance, dict) or source_evidence.get("run_count") != 80:
        raise ValueError("promotion source evidence is incomplete")
    _validate_source_provenance(provenance, final_index.source_commit)

    route_rows = _load_route_rows(decisions_path)
    route_coordinates = {
        (row["task"], int(row["cutoff_cycle"]), row["role"])
        for row in route_rows
    }
    if len(route_coordinates) != len(route_rows):
        raise ValueError("promotion decisions contain a duplicate route coordinate")
    if route_coordinates != _EXPECTED_ROUTE_MATRIX:
        raise ValueError("promotion decisions do not contain the exact route matrix")
    _verify_representative_validation_minima(results, source_evidence, route_rows)
    source_routes = tuple(
        _bind_route_to_checkpoint(
            row,
            result_root=results,
            final_root=final_root,
            final_source_commit=final_index.source_commit,
            final_config_sha256=final_index.config_sha256,
            provenance=provenance,
        )
        for row in route_rows
    )

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("temporary deployment output already exists")
    temporary.mkdir(parents=True)
    try:
        artifact_root = temporary / "artifacts"
        artifact_root.mkdir()
        resume_root = (
            None
            if resume_artifact_root is None
            else _regular_directory(resume_artifact_root, "resume artifact root")
        )
        by_checkpoint: dict[str, DeepModelArtifactManifest] = {}
        artifacts: list[AdvancedDeploymentArtifact] = []
        for route in source_routes:
            checkpoint_hash = str(route["checkpoint_manifest_sha256"])
            if checkpoint_hash in by_checkpoint:
                continue
            manifest = _resume_artifact(
                resume_root,
                artifact_root=artifact_root,
                checkpoint_manifest_sha256=checkpoint_hash,
                checkpoint_model_sha256=str(route["checkpoint_model_sha256"]),
                family=str(route["family"]),
            )
            if manifest is None:
                manifest = _rebuild_and_export_artifact(
                    project_root=project,
                    result_root=results,
                    route=route,
                    artifact_root=artifact_root,
                )
            by_checkpoint[checkpoint_hash] = manifest
            feature_context = _artifact_feature_context(artifact_root, manifest)
            artifacts.append(
                AdvancedDeploymentArtifact(
                    artifact_id=manifest.artifact_id,
                    artifact_kind=manifest.artifact_kind,
                    artifact_manifest_sha256=manifest.manifest_sha256,
                    source_checkpoint_manifest_sha256=checkpoint_hash,
                    source_checkpoint_model_sha256=str(
                        route["checkpoint_model_sha256"]
                    ),
                    dataset_id=feature_context["dataset_id"],
                    data_version=feature_context["data_version"],
                    split_version=feature_context["split_version"],
                    feature_version=feature_context["feature_version"],
                    cutoff_cycle=feature_context["cutoff_cycle"],
                    candidate_config_sha256=feature_context[
                        "candidate_config_sha256"
                    ],
                    normalization_sha256=feature_context["normalization_sha256"],
                    target_scaler_context_sha256=feature_context.get(
                        "target_scaler_context_sha256"
                    ),
                    reference_library_sha256=feature_context.get(
                        "reference_library_sha256"
                    ),
                    files=manifest.files,
                    round_trip_verified=True,
                )
            )

        routes = tuple(
            AdvancedDeploymentSourceRoute.model_validate(
                {
                    **route,
                    "deep_artifact_id": by_checkpoint[
                        str(route["checkpoint_manifest_sha256"])
                    ].artifact_id,
                    "deep_artifact_kind": by_checkpoint[
                        str(route["checkpoint_manifest_sha256"])
                    ].artifact_kind,
                    "deep_artifact_manifest_sha256": by_checkpoint[
                        str(route["checkpoint_manifest_sha256"])
                    ].manifest_sha256,
                }
            )
            for route in source_routes
        )
        created_at = datetime.now(UTC)
        payload: dict[str, Any] = {
            "schema_version": "advanced-deployment-bundle-index-v1",
            "activation_status": "NOT_ACTIVATED",
            "created_at": created_at.isoformat().replace("+00:00", "Z"),
            "source_commit": final_index.source_commit,
            "final_output_sha256": final_index.output_sha256,
            "final_config_sha256": final_index.config_sha256,
            "data_version": provenance["data_version"],
            "split_version": provenance["split_version"],
            "feature_version": source_routes[0]["feature_version"],
            "training_input_bundle_sha256": provenance["input_bundle_sha256"],
            "local_reconstructed_input_bundle_sha256": provenance[
                "local_reconstructed_input_bundle_sha256"
            ],
            "input_bundle_hashes_match": provenance["input_bundle_hashes_match"],
            "promotion_manifest_sha256": _sha256_file(promotion_manifest_path),
            "promotion_decisions_sha256": decisions_item["sha256"],
            "promotion_source_evidence_sha256": source_item["sha256"],
            "selection_manifest_sha256": source_routes[0][
                "selection_manifest_sha256"
            ],
            "routes": [item.model_dump(mode="json") for item in routes],
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
        }
        index = AdvancedDeploymentBundleIndex.model_validate(
            {**payload, "manifest_sha256": sha256_canonical(payload)}
        )
        _write_json(temporary / _INDEX_NAME, index.model_dump(mode="json"))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
        return index
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _resume_artifact(
    resume_root: Path | None,
    *,
    artifact_root: Path,
    checkpoint_manifest_sha256: str,
    checkpoint_model_sha256: str,
    family: str,
) -> DeepModelArtifactManifest | None:
    if resume_root is None:
        return None
    artifact_id = str(uuid5(NAMESPACE_URL, checkpoint_manifest_sha256))
    source = resume_root / artifact_id
    if not source.exists():
        return None
    source = _regular_directory(source, "resumed artifact directory")
    manifest_path = _regular_file(source / "manifest.json", "resumed artifact manifest")
    manifest = DeepModelArtifactManifest.model_validate_json(manifest_path.read_bytes())
    expected_kind = {
        "cyclepatch_direct": DeepArtifactKind.CYCLEPATCH_DIRECT,
        "cyclepatch_batlinet": DeepArtifactKind.CYCLEPATCH_BATLINET,
        "current_hybrid": DeepArtifactKind.CURRENT_HYBRID,
        "hybridpatch_v2": DeepArtifactKind.HYBRIDPATCH_V2,
    }.get(family)
    if manifest.artifact_id != artifact_id or manifest.artifact_kind is not expected_kind:
        raise ValueError("resumed artifact identity differs from its checkpoint route")
    verify_deep_model_artifact(resume_root, manifest)
    weight = next(
        item for item in manifest.files if item.role is DeepArtifactFileRole.WEIGHTS
    )
    if weight.sha256 != checkpoint_model_sha256:
        raise ValueError("resumed artifact checkpoint weight binding does not match")
    destination = artifact_root / artifact_id
    shutil.copytree(source, destination, symlinks=False)
    verify_deep_model_artifact(artifact_root, manifest)
    return manifest


def _artifact_feature_context(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> dict[str, Any]:
    feature_file = next(
        item
        for item in manifest.files
        if item.role is DeepArtifactFileRole.FEATURE_CONFIG
    )
    feature = _strict_json(artifact_root / feature_file.relative_path)
    required = {
        "dataset_id",
        "data_version",
        "split_version",
        "feature_version",
        "cutoff_cycle",
        "candidate_config_sha256",
        "normalization_sha256",
    }
    if not required <= set(feature):
        raise ValueError("deep artifact feature context is incomplete")
    output = {name: feature[name] for name in required}
    target_scaler = feature.get("target_scaler")
    if target_scaler is not None:
        if not isinstance(target_scaler, dict) or not isinstance(
            target_scaler.get("context_sha256"), str
        ):
            raise ValueError("deep artifact target scaler context is invalid")
        output["target_scaler_context_sha256"] = target_scaler["context_sha256"]
    reference_sha256 = feature.get("reference_library_sha256")
    if reference_sha256 is not None:
        output["reference_library_sha256"] = reference_sha256
    return output


def _rebuild_and_export_artifact(
    *,
    project_root: Path,
    result_root: Path,
    route: dict[str, object],
    artifact_root: Path,
) -> DeepModelArtifactManifest:
    cutoff = _object_integer(route["cutoff_cycle"], "cutoff_cycle")
    rebuilt = _load_rebuild_data(project_root, result_root, cutoff)
    family = str(route["family"])
    candidate_id = str(route["candidate_id"])
    seed = _object_integer(route["seed"], "seed")
    key = _resolve_run_key(
        rebuilt.config,
        result_root=result_root,
        family=family,
        candidate_id=candidate_id,
        cutoff_cycle=cutoff,
        seed=seed,
    )
    candidate = _candidate_for_key(rebuilt.config, key)
    task = _build_training_task(
        run_key=key,
        candidate=candidate,
        data=rebuilt.data,
        split=rebuilt.split,
    )
    checkpoint_root = _inside_result(
        result_root,
        str(route["checkpoint_directory"]),
    )
    manifest = AdvancedTrainingCheckpointManifest.model_validate_json(
        (checkpoint_root / "manifest.json").read_bytes()
    )
    _validate_rebuilt_context(
        manifest,
        rebuilt=rebuilt,
        key=key,
        candidate_sha256=candidate.config_sha256,
        task=task,
    )
    task.model.eval()
    load_advanced_inference_checkpoint(
        checkpoint_root,
        manifest,
        expected_context=manifest.context,
        model=task.model,
    )

    artifact_id = str(uuid5(NAMESPACE_URL, manifest.manifest_sha256))
    early = _early_batch_for_family(rebuilt.data, family)
    if family == "cyclepatch_direct":
        deep_manifest = export_cyclepatch_direct_artifact(
            task.model,
            artifact_root=artifact_root,
            artifact_id=artifact_id,
            created_at=manifest.created_at,
            dataset_id=manifest.context.dataset_id,
            data_version=manifest.context.data_version,
            split_version=manifest.context.split_version,
            feature_version=manifest.context.feature_version,
            cutoff_cycle=cutoff,
            condition_names=early.condition_names,
            normalization_sha256=manifest.context.normalization_sha256,
            candidate_config_sha256=manifest.context.candidate_config_sha256,
            target_scaler=task.target_scaler,
        )
    elif family == "cyclepatch_batlinet":
        if manifest.context.reference_library_sha256 is None:
            raise ValueError("BatLiNet checkpoint is missing reference library hash")
        if (
            task.reference_library.library_sha256
            != manifest.context.reference_library_sha256
        ):
            raise ValueError("BatLiNet task reference library differs from checkpoint")
        deep_manifest = export_cyclepatch_batlinet_artifact(
            task.model,
            artifact_root=artifact_root,
            artifact_id=artifact_id,
            created_at=manifest.created_at,
            dataset_id=manifest.context.dataset_id,
            data_version=manifest.context.data_version,
            split_version=manifest.context.split_version,
            feature_version=manifest.context.feature_version,
            cutoff_cycle=cutoff,
            condition_names=early.condition_names,
            normalization_sha256=manifest.context.normalization_sha256,
            candidate_config_sha256=manifest.context.candidate_config_sha256,
            reference_library=task.reference_library,
            target_scaler=task.target_scaler,
        )
    elif family == "hybridpatch_v2":
        deep_manifest = export_hybridpatch_v2_artifact(
            task.model,
            artifact_root=artifact_root,
            artifact_id=artifact_id,
            created_at=manifest.created_at,
            dataset_id=manifest.context.dataset_id,
            data_version=manifest.context.data_version,
            split_version=manifest.context.split_version,
            feature_version=manifest.context.feature_version,
            cutoff_cycle=cutoff,
            condition_names=early.condition_names,
            normalization_sha256=manifest.context.normalization_sha256,
            candidate_config_sha256=manifest.context.candidate_config_sha256,
        )
    elif family == "current_hybrid":
        deep_manifest = export_current_hybrid_artifact(
            task.model,
            artifact_root=artifact_root,
            artifact_id=artifact_id,
            created_at=manifest.created_at,
            dataset_id=manifest.context.dataset_id,
            data_version=manifest.context.data_version,
            split_version=manifest.context.split_version,
            feature_version=manifest.context.feature_version,
            cutoff_cycle=cutoff,
            condition_names=early.condition_names,
            normalization_sha256=manifest.context.normalization_sha256,
            candidate_config_sha256=manifest.context.candidate_config_sha256,
            prediction_cycles=task.train_batch.prediction_cycles,
            variable_names=VARIABLE_NAMES,
            aggregation_version="masked-variable-mean-v1",
        )
    else:  # pragma: no cover - route contract closes this branch
        raise ValueError(f"unsupported Advanced model family: {family}")
    _verify_round_trip(
        family=family,
        task=task,
        data=rebuilt.data,
        artifact_root=artifact_root,
        manifest=deep_manifest,
        normalization_sha256=manifest.context.normalization_sha256,
        candidate_config_sha256=manifest.context.candidate_config_sha256,
    )
    return deep_manifest


@lru_cache(maxsize=4)
def _load_rebuild_data(
    project_root: Path,
    result_root: Path,
    cutoff_cycle: int,
) -> _RebuildData:
    config_path = (
        result_root
        / "runs"
        / "a100"
        / "matr-three-batch"
        / "advanced"
        / "selection"
        / "final_config_resolved.json"
    )
    config = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        _regular_file(config_path, "resolved Advanced Final config").read_bytes()
    )
    if config.mode != "final":
        raise ValueError("resolved Advanced config is not final")
    manifest_path = _inside_project(project_root, config.paths.three_batch_manifest)
    split_path = _inside_project(project_root, config.paths.split_manifest)
    manifest = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
    split = SplitManifest.model_validate_json(split_path.read_bytes())
    revision = _strict_json(result_root / "source_revision.json")
    if revision.get("git_dirty") is not False:
        raise ValueError("Advanced Final source revision must be clean")
    source_commit = str(revision.get("git_commit", ""))
    if len(source_commit) not in {40, 64}:
        raise ValueError("Advanced Final source commit is invalid")
    data = load_advanced_matr_final_data(
        project_root=project_root,
        manifest=manifest,
        combined_split=split,
        cutoff_cycle=cutoff_cycle,
        feature_version=ADVANCED_FEATURE_VERSION,
    )
    return _RebuildData(
        config=config,
        manifest=manifest,
        split=split,
        data=data,
        source_commit=source_commit,
    )


def _resolve_run_key(
    config: AdvancedMatrThreeBatchRunConfig,
    *,
    result_root: Path,
    family: str,
    candidate_id: str,
    cutoff_cycle: int,
    seed: int,
) -> AdvancedRunKey:
    matches = tuple(
        key
        for key in build_advanced_run_matrix(config, repository_root=result_root)
        if (
            key.family,
            key.candidate_id,
            key.cutoff_cycle,
            key.seed,
        )
        == (family, candidate_id, cutoff_cycle, seed)
    )
    if len(matches) != 1:
        raise ValueError("representative route does not resolve to one final run key")
    return matches[0]


def _validate_rebuilt_context(
    manifest: AdvancedTrainingCheckpointManifest,
    *,
    rebuilt: _RebuildData,
    key: AdvancedRunKey,
    candidate_sha256: str,
    task: Any,
) -> None:
    context = manifest.context
    expected = {
        "run_id": f"matr-{key.family}-{key.candidate_id}-c{key.cutoff_cycle}-s{key.seed}",
        "dataset_id": "MATR",
        "model_name": key.family,
        "cutoff_cycle": key.cutoff_cycle,
        "seed": key.seed,
        "config_sha256": rebuilt.config.config_sha256,
        "data_version": rebuilt.manifest.data_version,
        "split_version": rebuilt.manifest.split_version,
        "feature_version": ADVANCED_FEATURE_VERSION,
        "source_commit": rebuilt.source_commit,
        "run_mode": "final",
        "stage": "final",
        "candidate_config_sha256": candidate_sha256,
        "model_architecture_sha256": model_architecture_sha256(
            task.model,
            candidate_sha256,
        ),
        "normalization_sha256": _normalization_hash(rebuilt.data, key.family),
        "selection_manifest_sha256": rebuilt.config.expected_selection_sha256,
    }
    for name, value in expected.items():
        if getattr(context, name) != value:
            raise ValueError(f"rebuilt checkpoint context field mismatch: {name}")
    library = getattr(task, "reference_library", None)
    reference_sha256 = (
        None if library is None else getattr(library, "library_sha256", None)
    )
    if context.reference_library_sha256 != reference_sha256:
        raise ValueError("rebuilt checkpoint reference library hash mismatch")


def _early_batch_for_family(
    data: AdvancedFinalMatrData,
    family: str,
) -> EarlyCycleBatch:
    if family in {"cyclepatch_direct", "cyclepatch_batlinet"}:
        return data.scalar_train.early_batch
    return data.hybrid_train.inputs.early_batch


def _verify_round_trip(
    *,
    family: str,
    task: Any,
    data: AdvancedFinalMatrData,
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
    normalization_sha256: str,
    candidate_config_sha256: str,
) -> None:
    with torch.no_grad():
        if family == "cyclepatch_direct":
            batch = data.scalar_validation.early_batch
            expected = task.target_scaler.inverse_transform(task.model(batch))
            loaded_direct = load_cyclepatch_direct_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=normalization_sha256,
                expected_candidate_config_sha256=candidate_config_sha256,
                expected_target_scaler_context_sha256=task.target_scaler.context_sha256,
            )
            actual = loaded_direct.predict_raw(batch)
        elif family == "cyclepatch_batlinet":
            target = data.scalar_validation.early_batch
            reference = _reference_batch(
                data.scalar_train.early_batch,
                task.reference_library.cell_ids,
            )
            reference_labels = torch.tensor(
                task.reference_library.standardized_labels,
                dtype=target.values.dtype,
            )
            expected = task.target_scaler.inverse_transform(
                task.model.fuse_standardized(
                    task.model.encode(target),
                    task.model.encode(reference),
                    reference_labels,
                )
            )
            loaded_batlinet = load_cyclepatch_batlinet_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=normalization_sha256,
                expected_candidate_config_sha256=candidate_config_sha256,
                expected_reference_library_sha256=(
                    task.reference_library.library_sha256
                ),
                expected_target_scaler_context_sha256=task.target_scaler.context_sha256,
            )
            actual = loaded_batlinet.predict_raw(target, reference)
        elif family == "hybridpatch_v2":
            inputs = data.hybrid_validation.inputs
            expected = task.model(inputs).predicted_soh
            loaded_hybridpatch = load_hybridpatch_v2_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=normalization_sha256,
                expected_candidate_config_sha256=candidate_config_sha256,
            )
            actual = loaded_hybridpatch(inputs).predicted_soh
        else:
            inputs = data.hybrid_validation.inputs
            prediction_cycles = task.train_batch.prediction_cycles
            expected = task.predict(
                _current_hybrid_batch(
                    data.hybrid_validation,
                    prediction_cycles=prediction_cycles,
                ),
                device=torch.device("cpu"),
            )
            loaded_current = load_current_hybrid_artifact(
                artifact_root,
                manifest,
                expected_prediction_cycles=prediction_cycles,
                expected_variable_names=VARIABLE_NAMES,
                expected_aggregation_version="masked-variable-mean-v1",
                expected_normalization_sha256=normalization_sha256,
                expected_candidate_config_sha256=candidate_config_sha256,
            )
            actual = loaded_current(inputs.early_batch, inputs.initial_soh)
    if not torch.equal(expected.detach().cpu(), actual.detach().cpu()):
        difference = float(
            (expected.detach().cpu() - actual.detach().cpu()).abs().max().item()
        )
        raise ValueError(
            f"deep artifact round-trip prediction mismatch for {family}: {difference}"
        )
def _reference_batch(
    batch: EarlyCycleBatch,
    cell_ids: tuple[str, ...],
) -> EarlyCycleBatch:
    index_by_cell = {cell_id: index for index, cell_id in enumerate(batch.cell_ids)}
    if any(cell_id not in index_by_cell for cell_id in cell_ids):
        raise ValueError("reference library cells are missing from the training batch")
    indices = torch.tensor(
        [index_by_cell[cell_id] for cell_id in cell_ids],
        dtype=torch.int64,
    )
    return EarlyCycleBatch(
        dataset_id=batch.dataset_id,
        data_version=batch.data_version,
        feature_version=batch.feature_version,
        normalization_statistics_sha256=batch.normalization_statistics_sha256,
        cell_ids=cell_ids,
        condition_names=batch.condition_names,
        values=batch.values.index_select(0, indices),
        cycle_indices=batch.cycle_indices.index_select(0, indices),
        cycle_mask=batch.cycle_mask.index_select(0, indices),
        sample_mask=batch.sample_mask.index_select(0, indices),
        condition_values=batch.condition_values.index_select(0, indices),
        condition_mask=batch.condition_mask.index_select(0, indices),
    )


def _inside_project(root: Path, relative: str) -> Path:
    candidate = root / Path(*relative.replace("\\", "/").split("/"))
    if candidate.is_symlink():
        raise ValueError("Advanced input path must remain inside the project root")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Advanced input path must remain inside the project root")
    return path


def _inside_result(root: Path, relative: str) -> Path:
    candidate = root / Path(*relative.replace("\\", "/").split("/"))
    if candidate.is_symlink():
        raise ValueError("checkpoint path must remain inside the result root")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError("checkpoint path must remain inside the result root")
    return path


def _load_route_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "task",
            "family",
            "candidate_id",
            "cutoff_cycle",
            "role",
            "disposition",
            "representative_seed",
            "representative_best_epoch",
            "representative_seed_rule",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("promotion decisions CSV is missing required columns")
        rows = tuple(dict(row) for row in reader)
    if len(rows) != _EXPECTED_ROUTE_COUNT:
        raise ValueError("promotion decisions must contain exactly 15 routes")
    for row in rows:
        if row["role"] not in _APPROVED_ROLES:
            raise ValueError("promotion decisions contain an unapproved route role")
        if row["disposition"] != "CONDITIONAL":
            raise ValueError("only CONDITIONAL promotion routes may be bundled")
        if row["representative_seed_rule"] != "MINIMUM_BEST_VALIDATION_METRIC":
            raise ValueError("representative checkpoint was not selected by validation")
    return rows


def _verify_representative_validation_minima(
    result_root: Path,
    source_evidence: dict[str, Any],
    routes: tuple[dict[str, str], ...],
) -> None:
    inputs = source_evidence.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("promotion source evidence is missing run metrics")
    registered = inputs.get("advanced_final_run_metrics.csv")
    if not isinstance(registered, dict):
        raise ValueError("promotion source evidence is missing run metrics")
    metrics_path = (
        result_root / "analysis" / "advanced_final_run_metrics.csv"
    ).resolve(strict=True)
    recorded_path = Path(str(registered.get("path", ""))).resolve(strict=False)
    if recorded_path != metrics_path:
        raise ValueError("promotion run metrics path differs from the result root")
    if metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("promotion run metrics must be a regular file")
    if metrics_path.stat().st_size != registered.get("size_bytes"):
        raise ValueError("promotion run metrics size differs from source evidence")
    if _sha256_file(metrics_path) != registered.get("sha256"):
        raise ValueError("promotion run metrics SHA-256 differs from source evidence")
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "family",
            "candidate_id",
            "cutoff_cycle",
            "seed",
            "best_epoch",
            "best_validation_metric",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("Advanced Final run metrics are missing required columns")
        metric_rows = tuple(dict(row) for row in reader)
    if len(metric_rows) != 80:
        raise ValueError("Advanced Final run metrics must contain 80 runs")
    by_coordinate: dict[tuple[str, str, int], list[tuple[float, int, int]]] = {}
    for row in metric_rows:
        coordinate = (
            row["family"],
            row["candidate_id"],
            _integer(row["cutoff_cycle"], "run metric cutoff_cycle"),
        )
        seed = _integer(row["seed"], "run metric seed")
        best_epoch = _integer(row["best_epoch"], "run metric best_epoch")
        try:
            validation_metric = float(row["best_validation_metric"])
        except (TypeError, ValueError) as exc:
            raise ValueError("best_validation_metric must be finite") from exc
        if not math.isfinite(validation_metric):
            raise ValueError("best_validation_metric must be finite")
        by_coordinate.setdefault(coordinate, []).append(
            (validation_metric, seed, best_epoch)
        )
    for row in routes:
        coordinate = (
            row["family"],
            row["candidate_id"],
            _integer(row["cutoff_cycle"], "cutoff_cycle"),
        )
        candidates = by_coordinate.get(coordinate, [])
        if len(candidates) != 5 or len({item[1] for item in candidates}) != 5:
            raise ValueError("representative seed requires five validation runs")
        selected = min(candidates, key=lambda item: (item[0], item[1]))
        actual = (
            _integer(row["representative_seed"], "representative_seed"),
            _integer(
                row["representative_best_epoch"],
                "representative_best_epoch",
            ),
        )
        if actual != (selected[1], selected[2]):
            raise ValueError(
                "representative seed or best_epoch is not the validation minimum"
            )


def _bind_route_to_checkpoint(
    row: dict[str, str],
    *,
    result_root: Path,
    final_root: Path,
    final_source_commit: str,
    final_config_sha256: str,
    provenance: dict[str, object],
) -> dict[str, object]:
    family = row["family"]
    candidate_id = row["candidate_id"]
    cutoff = _integer(row["cutoff_cycle"], "cutoff_cycle")
    seed = _integer(row["representative_seed"], "representative_seed")
    best_epoch = _integer(row["representative_best_epoch"], "representative_best_epoch")
    run_id = f"matr-{family}-{candidate_id}-c{cutoff}-s{seed}"
    run_root = _regular_directory(
        final_root
        / f"cutoff-{cutoff}"
        / family
        / candidate_id
        / f"seed-{seed}",
        "representative run root",
    )
    pointer_path = _regular_file(run_root / "checkpoints" / "best.json", "best pointer")
    pointer = _strict_json(pointer_path).get("checkpoint")
    expected_pointer = f"epoch-{best_epoch:06d}"
    if pointer != expected_pointer:
        raise ValueError("representative best_epoch does not match best checkpoint")
    checkpoint_root = _regular_directory(
        run_root / "checkpoints" / expected_pointer,
        "representative checkpoint",
    )
    manifest_path = _regular_file(
        checkpoint_root / "manifest.json",
        "checkpoint manifest",
    )
    manifest = AdvancedTrainingCheckpointManifest.model_validate_json(
        manifest_path.read_bytes()
    )
    unsigned = manifest.model_dump(mode="json", exclude={"manifest_sha256"})
    if sha256_canonical(unsigned) != manifest.manifest_sha256:
        raise ValueError("checkpoint manifest SHA-256 does not match")
    _verify_checkpoint_files(checkpoint_root, manifest)
    context = manifest.context
    coordinates = (
        context.model_name,
        context.cutoff_cycle,
        context.seed,
        context.run_id,
    )
    expected_coordinates = (family, cutoff, seed, run_id)
    if coordinates != expected_coordinates:
        raise ValueError("representative checkpoint coordinate mismatch")
    if manifest.progress.best_epoch != best_epoch:
        raise ValueError("representative best_epoch differs from checkpoint progress")
    if (
        context.run_mode != "final"
        or context.stage != "final"
        or context.source_commit != final_source_commit
        or context.config_sha256 != final_config_sha256
    ):
        raise ValueError("representative checkpoint is not from this Advanced Final")
    for field in (
        "source_commit",
        "data_version",
        "split_version",
        "input_bundle_sha256",
    ):
        if getattr(context, field) != provenance.get(field):
            raise ValueError(f"checkpoint {field} differs from promotion evidence")
    if context.selection_manifest_sha256 is None:
        raise ValueError("final checkpoint is missing selection manifest provenance")
    context_payload = context.model_dump(mode="json")
    return {
        "task": row["task"],
        "role": row["role"],
        "disposition": row["disposition"],
        "activation_status": "NOT_ACTIVATED",
        "family": family,
        "candidate_id": candidate_id,
        "data_version": context.data_version,
        "split_version": context.split_version,
        "cutoff_cycle": cutoff,
        "seed": seed,
        "best_epoch": best_epoch,
        "representative_seed_rule": row["representative_seed_rule"],
        "run_id": run_id,
        "checkpoint_directory": checkpoint_root.relative_to(result_root).as_posix(),
        "checkpoint_manifest_sha256": manifest.manifest_sha256,
        "checkpoint_manifest_file_sha256": _sha256_file(manifest_path),
        "checkpoint_model_sha256": _sha256_file(
            checkpoint_root / "model.safetensors"
        ),
        "checkpoint_context_sha256": sha256_canonical(context_payload),
        "selection_manifest_sha256": context.selection_manifest_sha256,
        "candidate_config_sha256": context.candidate_config_sha256,
        "normalization_sha256": context.normalization_sha256,
        "reference_library_sha256": context.reference_library_sha256,
        "feature_version": context.feature_version,
    }


def _verify_checkpoint_files(
    checkpoint_root: Path,
    manifest: AdvancedTrainingCheckpointManifest,
) -> None:
    expected = {item.relative_path for item in manifest.files} | {"manifest.json"}
    actual = {path.name for path in checkpoint_root.iterdir()}
    if actual != expected:
        raise ValueError("checkpoint directory contains unexpected files")
    for item in manifest.files:
        path = _regular_file(
            checkpoint_root / item.relative_path,
            "checkpoint file",
        )
        if path.stat().st_size != item.size_bytes:
            raise ValueError("checkpoint file size differs from manifest")
        if _sha256_file(path) != item.sha256:
            raise ValueError("checkpoint file SHA-256 differs from manifest")


def _verify_promotion_manifest(root: Path, path: Path) -> dict[str, Any]:
    manifest = _strict_json(_regular_file(path, "promotion manifest"))
    if (
        manifest.get("schema_version")
        != "advanced-promotion-artifact-manifest-v1"
        or manifest.get("activation_status") != "NOT_ACTIVATED"
    ):
        raise ValueError("promotion manifest is not an inactive approved report")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("promotion manifest file inventory is invalid")
    paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("promotion manifest file entry is invalid")
        relative = str(item.get("relative_path", "")).replace("\\", "/")
        candidate = Path(relative)
        if (
            candidate.is_absolute()
            or candidate.drive
            or any(part in {"", ".", ".."} for part in candidate.parts)
            or relative in paths
        ):
            raise ValueError("promotion manifest contains an unsafe or duplicate path")
        paths.add(relative)
        registered = _regular_file(root / candidate, "promotion evidence file")
        if registered.stat().st_size != item.get("size_bytes"):
            raise ValueError("promotion file size differs from its manifest")
        if _sha256_file(registered) != item.get("sha256"):
            raise ValueError("promotion file SHA-256 differs from its manifest")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != paths | {"artifact_manifest.json"}:
        raise ValueError("promotion manifest does not describe the complete file set")
    return manifest


def _validate_source_provenance(
    provenance: dict[str, object],
    source_commit: str,
) -> None:
    if provenance.get("source_commit") != source_commit:
        raise ValueError("promotion source commit differs from Advanced Final")
    for field in (
        "data_version",
        "split_version",
        "input_bundle_sha256",
        "local_reconstructed_input_bundle_sha256",
    ):
        value = provenance.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"promotion source provenance {field} is invalid")
        if field.endswith("sha256") and len(value) != 64:
            raise ValueError(f"promotion source provenance {field} is invalid")
    if not isinstance(provenance.get("input_bundle_hashes_match"), bool):
        raise ValueError("promotion input bundle comparison is missing")


def _regular_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular directory")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a regular directory")
    return resolved


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlinked file")
    return path


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path.name}")
    return payload


def _integer(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"promotion {label} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"promotion {label} must be positive")
    return parsed


def _object_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"deployment route {label} must be an integer")
    return _integer(str(value), label)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: str, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return normalized


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
