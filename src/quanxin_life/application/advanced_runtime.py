"""Resolve active Advanced model runtimes without exposing managed file paths."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Protocol, TypeAlias, cast, runtime_checkable

import torch

from quanxin_life.application.advanced_deployment_bundles import (
    AdvancedDeploymentArtifact,
)
from quanxin_life.application.advanced_deployment_registry import (
    AdvancedDeploymentBundleRegistry,
    RegisteredAdvancedDeploymentBundle,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    CurrentHybridFeatureConfig,
    DeepArtifactKind,
    DeepModelArtifactManifest,
    LoadedCurrentHybrid,
    LoadedCyclePatchBatLiNet,
    LoadedCyclePatchDirect,
    LoadedHybridPatchV2,
    load_current_hybrid_artifact,
    load_cyclepatch_batlinet_artifact,
    load_cyclepatch_direct_artifact,
    load_hybridpatch_v2_artifact,
    verify_self_contained_advanced_artifact,
)
from quanxin_life.application.invocation_context import (
    VerifiedProjectInvocationContext,
)
from quanxin_life.application.model_route_activation import (
    VerifiedActiveModelRoute,
)
from quanxin_life.application.projects import ProjectVisibilitySubject
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask, UserRole
from quanxin_life.features.early_cycle_sequence import (
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)
from quanxin_life.models.cyclepatch import stack_early_cycle_sequences
from quanxin_life.models.hybridpatch_v2 import HybridPatchV2Inputs


class AdvancedRuntimeResolutionError(RuntimeError):
    """Raised when a requested Advanced runtime cannot be verified."""


class AdvancedRuntimeStateError(AdvancedRuntimeResolutionError):
    """Raised when route or artifact state changes during one invocation."""


class ProjectContextVerifier(Protocol):
    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext: ...


@dataclass(frozen=True, slots=True)
class _ProjectVisibilityIdentity:
    """Session-independent subject for read-only project route visibility."""

    user_id: str
    role: UserRole


class ActiveModelRouteResolver(Protocol):
    def resolve_verified_active_model_route(
        self,
        principal: ProjectVisibilitySubject,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedActiveModelRoute: ...


@runtime_checkable
class RULInferenceAdapter(Protocol):
    """Typed label-free boundary for official MATR cycle-life inference."""

    @property
    def normalization_statistics_sha256(self) -> str: ...

    def predict_cycle(self, raw_sequence: EarlyCycleSequence) -> float: ...


@runtime_checkable
class SOHInferenceAdapter(Protocol):
    """Typed label-free boundary for finite-horizon SOH inference."""

    @property
    def normalization_statistics_sha256(self) -> str: ...

    def predict_trajectory(
        self,
        raw_sequence: EarlyCycleSequence,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]: ...


@dataclass(frozen=True, slots=True)
class _VerifiedRULInference:
    artifact_kind: DeepArtifactKind
    cutoff_cycle: int
    _model: LoadedCyclePatchDirect | LoadedCyclePatchBatLiNet
    _normalizer: EarlyCycleNormalizer

    @property
    def normalization_statistics_sha256(self) -> str:
        return self._normalizer.statistics_sha256

    def predict_cycle(self, raw_sequence: EarlyCycleSequence) -> float:
        batch = stack_early_cycle_sequences(
            (self._normalizer.transform(raw_sequence),)
        )
        with torch.inference_mode():
            if isinstance(
                self._model,
                (LoadedCyclePatchDirect, LoadedCyclePatchBatLiNet),
            ):
                prediction = self._model.predict_raw(batch)
            else:  # pragma: no cover - closed by construction
                raise AdvancedRuntimeResolutionError(
                    "RUL inference adapter contains an unsupported model"
                )
        if prediction.shape != (1,) or not prediction.is_floating_point():
            raise AdvancedRuntimeResolutionError(
                "RUL runtime returned an invalid scalar output"
            )
        value = float(prediction.detach().cpu().item())
        if not math.isfinite(value) or value <= self.cutoff_cycle:
            raise AdvancedRuntimeResolutionError(
                "RUL runtime returned an invalid official cycle-life value"
            )
        return value


@dataclass(frozen=True, slots=True)
class _VerifiedSOHInference:
    artifact_kind: DeepArtifactKind
    cutoff_cycle: int
    prediction_cycles: tuple[int, ...]
    _model: LoadedHybridPatchV2 | LoadedCurrentHybrid
    _normalizer: EarlyCycleNormalizer

    def __post_init__(self) -> None:
        _validate_prediction_cycles(
            self.prediction_cycles,
            cutoff_cycle=self.cutoff_cycle,
        )

    @property
    def normalization_statistics_sha256(self) -> str:
        return self._normalizer.statistics_sha256

    def predict_trajectory(
        self,
        raw_sequence: EarlyCycleSequence,
        *,
        initial_soh: float,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        if (
            not isinstance(initial_soh, (int, float))
            or isinstance(initial_soh, bool)
            or not math.isfinite(float(initial_soh))
            or not 0.0 < float(initial_soh) <= 1.5
        ):
            raise ValueError("initial_soh must be finite and in (0, 1.5]")
        batch = stack_early_cycle_sequences(
            (self._normalizer.transform(raw_sequence),)
        )
        initial = torch.tensor(
            (float(initial_soh),),
            dtype=batch.values.dtype,
            device=batch.values.device,
        )
        cycle_axis = torch.tensor(
            self.prediction_cycles,
            dtype=torch.int64,
            device=batch.values.device,
        )
        with torch.inference_mode():
            if isinstance(self._model, LoadedHybridPatchV2):
                output = self._model(
                    HybridPatchV2Inputs(
                        early_batch=batch,
                        initial_soh=initial,
                        prediction_cycles=cycle_axis,
                    )
                )
                prediction = output.predicted_soh
            elif isinstance(self._model, LoadedCurrentHybrid):
                prediction = self._model(batch, initial)
            else:  # pragma: no cover - closed by construction
                raise AdvancedRuntimeResolutionError(
                    "SOH inference adapter contains an unsupported model"
                )
        expected_shape = (1, len(self.prediction_cycles))
        if (
            prediction.shape != expected_shape
            or not prediction.is_floating_point()
            or not bool(torch.isfinite(prediction).all().item())
        ):
            raise AdvancedRuntimeResolutionError(
                "SOH runtime returned an invalid trajectory output"
            )
        values = tuple(
            float(value)
            for value in prediction.detach().cpu().squeeze(0).tolist()
        )
        if any(value < 0.0 or value > 1.5 for value in values):
            raise AdvancedRuntimeResolutionError(
                "SOH runtime returned values outside the approved range"
            )
        if any(
            current > previous
            for previous, current in pairwise(values)
        ):
            raise AdvancedRuntimeResolutionError(
                "SOH runtime returned a non-monotonic trajectory"
            )
        return self.prediction_cycles, values


@dataclass(frozen=True, slots=True)
class VerifiedAdvancedRuntimeArtifact:
    """One loaded model whose managed bytes were freshly verified."""

    artifact_id: str
    artifact_manifest_sha256: str
    model_version: str
    artifact_kind: DeepArtifactKind
    output_target: AdvancedOutputTarget
    dataset_id: str
    data_version: str
    feature_version: str
    split_version: str
    normalization_sha256: str
    inference: RULInferenceAdapter | SOHInferenceAdapter
    model: torch.nn.Module


class AdvancedRuntimeProvider(Protocol):
    def resolve(
        self,
        route: VerifiedActiveModelRoute,
    ) -> VerifiedAdvancedRuntimeArtifact: ...


class AdvancedRuntimeModelLoader(Protocol):
    def load(
        self,
        *,
        artifact_root: Path,
        manifest: DeepModelArtifactManifest,
        deployment_artifact: AdvancedDeploymentArtifact,
    ) -> torch.nn.Module: ...


class ManagedAdvancedRuntimeProvider:
    """Freshly verify managed v2 bytes while caching only loaded models."""

    def __init__(
        self,
        registry: AdvancedDeploymentBundleRegistry,
        *,
        model_loader: AdvancedRuntimeModelLoader | None = None,
    ) -> None:
        self._registry = registry
        self._model_loader = model_loader or _SafetensorsAdvancedModelLoader()
        self._models: dict[str, torch.nn.Module] = {}

    def resolve(
        self,
        route: VerifiedActiveModelRoute,
    ) -> VerifiedAdvancedRuntimeArtifact:
        provenance = route.artifact.metadata.advanced_provenance
        if provenance is None:
            raise AdvancedRuntimeResolutionError(
                "active route is not backed by an Advanced deployment bundle"
            )
        registered = self._registry.resolve(
            provenance.deployment_bundle_manifest_sha256
        )
        deployment = _verified_deployment_artifact(registered, route)
        artifact_root = _artifact_root(registered.bundle_root)
        manifest_path = artifact_root / deployment.artifact_id / "manifest.json"
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or not manifest_path.resolve(strict=True).is_relative_to(artifact_root)
        ):
            raise AdvancedRuntimeResolutionError(
                "Advanced artifact manifest is unavailable"
            )
        try:
            manifest = DeepModelArtifactManifest.model_validate_json(
                manifest_path.read_bytes()
            )
        except (OSError, ValueError) as exc:
            raise AdvancedRuntimeResolutionError(
                "Advanced artifact manifest is invalid"
            ) from exc
        if (
            manifest.schema_version != "deep-model-artifact-v2"
            or manifest.artifact_id != deployment.artifact_id
            or manifest.artifact_kind is not deployment.artifact_kind
            or manifest.output_target is not deployment.output_target
            or manifest.manifest_sha256
            != deployment.artifact_manifest_sha256
        ):
            raise AdvancedRuntimeResolutionError(
                "Advanced artifact is not a matching self-contained v2 runtime"
            )
        inference_context = verify_self_contained_advanced_artifact(
            artifact_root,
            manifest,
        )
        if inference_context.output_target is not deployment.output_target:
            raise AdvancedRuntimeResolutionError(
                "Advanced inference context target does not match deployment"
            )
        cache_key = manifest.manifest_sha256
        model = self._models.get(cache_key)
        if model is None:
            model = self._model_loader.load(
                artifact_root=artifact_root,
                manifest=manifest,
                deployment_artifact=deployment,
            )
            if not isinstance(model, torch.nn.Module):
                raise AdvancedRuntimeResolutionError(
                    "Advanced runtime loader returned an invalid model"
                )
            model.eval()
            model.requires_grad_(False)
            self._models[cache_key] = model
        isolated_model = _isolated_inference_model(model)
        inference = _build_inference_adapter(
            isolated_model,
            deployment=deployment,
        )
        return VerifiedAdvancedRuntimeArtifact(
            artifact_id=manifest.artifact_id,
            artifact_manifest_sha256=manifest.manifest_sha256,
            model_version=route.artifact.model_version,
            artifact_kind=manifest.artifact_kind,
            output_target=inference_context.output_target,
            dataset_id=deployment.dataset_id,
            data_version=deployment.data_version,
            feature_version=deployment.feature_version,
            split_version=deployment.split_version,
            normalization_sha256=deployment.normalization_sha256,
            inference=inference,
            model=isolated_model,
        )


class _SafetensorsAdvancedModelLoader:
    def load(
        self,
        *,
        artifact_root: Path,
        manifest: DeepModelArtifactManifest,
        deployment_artifact: AdvancedDeploymentArtifact,
    ) -> torch.nn.Module:
        expected = deployment_artifact
        if manifest.artifact_kind is DeepArtifactKind.CYCLEPATCH_DIRECT:
            if expected.target_scaler_context_sha256 is None:
                raise AdvancedRuntimeResolutionError(
                    "RUL artifact target scaler context is missing"
                )
            return load_cyclepatch_direct_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=expected.normalization_sha256,
                expected_candidate_config_sha256=(
                    expected.candidate_config_sha256
                ),
                expected_target_scaler_context_sha256=(
                    expected.target_scaler_context_sha256
                ),
            )
        if manifest.artifact_kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
            if (
                expected.target_scaler_context_sha256 is None
                or expected.reference_library_sha256 is None
            ):
                raise AdvancedRuntimeResolutionError(
                    "BatLiNet scaler or reference context is missing"
                )
            return load_cyclepatch_batlinet_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=expected.normalization_sha256,
                expected_candidate_config_sha256=(
                    expected.candidate_config_sha256
                ),
                expected_reference_library_sha256=(
                    expected.reference_library_sha256
                ),
                expected_target_scaler_context_sha256=(
                    expected.target_scaler_context_sha256
                ),
            )
        if manifest.artifact_kind is DeepArtifactKind.HYBRIDPATCH_V2:
            return load_hybridpatch_v2_artifact(
                artifact_root,
                manifest,
                expected_normalization_sha256=expected.normalization_sha256,
                expected_candidate_config_sha256=(
                    expected.candidate_config_sha256
                ),
            )
        if manifest.artifact_kind is DeepArtifactKind.CURRENT_HYBRID:
            feature = _current_hybrid_feature(
                artifact_root,
                manifest,
            )
            return load_current_hybrid_artifact(
                artifact_root,
                manifest,
                expected_prediction_cycles=feature.prediction_cycles,
                expected_variable_names=feature.variable_names,
                expected_aggregation_version=feature.aggregation_version,
                expected_normalization_sha256=expected.normalization_sha256,
                expected_candidate_config_sha256=(
                    expected.candidate_config_sha256
                ),
            )
        raise AdvancedRuntimeResolutionError(
            "artifact is not an approved Advanced runtime model"
        )


@dataclass(frozen=True, slots=True)
class VerifiedRULRuntime:
    project_id: str
    task: AdvancedModelTask
    cutoff_cycle: int
    role: AdvancedModelRouteRole
    output_target: AdvancedOutputTarget
    artifact_kind: DeepArtifactKind
    dataset_id: str
    data_version: str
    feature_version: str
    split_version: str
    normalization_sha256: str
    artifact_id: str
    artifact_manifest_sha256: str
    model_version: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str
    inference: RULInferenceAdapter


@dataclass(frozen=True, slots=True)
class VerifiedSOHRuntime:
    project_id: str
    task: AdvancedModelTask
    cutoff_cycle: int
    role: AdvancedModelRouteRole
    output_target: AdvancedOutputTarget
    artifact_kind: DeepArtifactKind
    dataset_id: str
    data_version: str
    feature_version: str
    split_version: str
    normalization_sha256: str
    artifact_id: str
    artifact_manifest_sha256: str
    model_version: str
    decision_event_id: str
    ledger_sequence_number: int
    ledger_head_sha256: str
    inference: SOHInferenceAdapter


VerifiedAdvancedRuntime: TypeAlias = VerifiedRULRuntime | VerifiedSOHRuntime


class ActiveAdvancedRuntimeResolver:
    """Resolve one exact active route and recheck it after model preparation."""

    def __init__(
        self,
        *,
        context_service: ProjectContextVerifier,
        route_service: ActiveModelRouteResolver,
        runtime_provider: AdvancedRuntimeProvider,
    ) -> None:
        self._context_service = context_service
        self._route_service = route_service
        self._runtime_provider = runtime_provider

    def resolve(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedAdvancedRuntime:
        normalized_task = AdvancedModelTask(task)
        normalized_role = AdvancedModelRouteRole(role)
        if cutoff_cycle not in {20, 50, 100, 150}:
            raise AdvancedRuntimeResolutionError(
                "Advanced runtime requires an approved cutoff cycle"
            )
        verified_context = self._context_service.revalidate(context)
        principal = _visibility_subject_from_context(verified_context)
        first_route = self._resolve_route(
            principal,
            verified_context,
            task=normalized_task,
            cutoff_cycle=cutoff_cycle,
            role=normalized_role,
        )
        first_artifact = self._runtime_provider.resolve(first_route)
        _validate_artifact(first_route, first_artifact)

        second_route = self._resolve_route(
            principal,
            verified_context,
            task=normalized_task,
            cutoff_cycle=cutoff_cycle,
            role=normalized_role,
        )
        if _route_identity(first_route) != _route_identity(second_route):
            raise AdvancedRuntimeStateError(
                "active Advanced model route changed during resolution"
            )
        second_artifact = self._runtime_provider.resolve(second_route)
        _validate_artifact(second_route, second_artifact)
        if _artifact_identity(first_artifact) != _artifact_identity(second_artifact):
            raise AdvancedRuntimeStateError(
                "Advanced runtime artifact changed during resolution"
            )
        self._context_service.revalidate(verified_context)
        if normalized_task is AdvancedModelTask.RUL:
            rul_inference = cast(
                RULInferenceAdapter,
                second_artifact.inference,
            )
            return VerifiedRULRuntime(
                project_id=verified_context.project_id,
                task=normalized_task,
                cutoff_cycle=cutoff_cycle,
                role=normalized_role,
                output_target=second_artifact.output_target,
                artifact_kind=second_artifact.artifact_kind,
                dataset_id=second_artifact.dataset_id,
                data_version=second_artifact.data_version,
                feature_version=second_artifact.feature_version,
                split_version=second_artifact.split_version,
                normalization_sha256=(
                    second_artifact.normalization_sha256
                ),
                artifact_id=second_artifact.artifact_id,
                artifact_manifest_sha256=(
                    second_artifact.artifact_manifest_sha256
                ),
                model_version=second_artifact.model_version,
                decision_event_id=second_route.decision_event_id,
                ledger_sequence_number=second_route.ledger_sequence_number,
                ledger_head_sha256=second_route.ledger_head_sha256,
                inference=rul_inference,
            )
        soh_inference = cast(SOHInferenceAdapter, second_artifact.inference)
        return VerifiedSOHRuntime(
            project_id=verified_context.project_id,
            task=normalized_task,
            cutoff_cycle=cutoff_cycle,
            role=normalized_role,
            output_target=second_artifact.output_target,
            artifact_kind=second_artifact.artifact_kind,
            dataset_id=second_artifact.dataset_id,
            data_version=second_artifact.data_version,
            feature_version=second_artifact.feature_version,
            split_version=second_artifact.split_version,
            normalization_sha256=second_artifact.normalization_sha256,
            artifact_id=second_artifact.artifact_id,
            artifact_manifest_sha256=second_artifact.artifact_manifest_sha256,
            model_version=second_artifact.model_version,
            decision_event_id=second_route.decision_event_id,
            ledger_sequence_number=second_route.ledger_sequence_number,
            ledger_head_sha256=second_route.ledger_head_sha256,
            inference=soh_inference,
        )

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
        runtime: VerifiedAdvancedRuntime,
    ) -> VerifiedAdvancedRuntime:
        current = self.resolve(
            context,
            task=runtime.task,
            cutoff_cycle=runtime.cutoff_cycle,
            role=runtime.role,
        )
        if _runtime_identity(current) != _runtime_identity(runtime):
            raise AdvancedRuntimeStateError(
                "Advanced runtime route changed after inference"
            )
        return runtime

    def _resolve_route(
        self,
        principal: ProjectVisibilitySubject,
        context: VerifiedProjectInvocationContext,
        *,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> VerifiedActiveModelRoute:
        route = self._route_service.resolve_verified_active_model_route(
            principal,
            project_id=context.project_id,
            task=task,
            cutoff_cycle=cutoff_cycle,
            role=role,
        )
        if (
            route.project_id != context.project_id
            or route.task is not task
            or route.cutoff_cycle != cutoff_cycle
            or route.role is not role
        ):
            raise AdvancedRuntimeStateError(
                "resolved Advanced model route does not match the request"
            )
        return route


def _visibility_subject_from_context(
    context: VerifiedProjectInvocationContext,
) -> ProjectVisibilitySubject:
    return _ProjectVisibilityIdentity(
        user_id=context.actor_user_id,
        role=context.actor_role,
    )


def _verified_deployment_artifact(
    registered: RegisteredAdvancedDeploymentBundle,
    route: VerifiedActiveModelRoute,
) -> AdvancedDeploymentArtifact:
    if (
        registered.index.schema_version
        != "advanced-deployment-bundle-index-v2"
    ):
        raise AdvancedRuntimeResolutionError(
            "self-contained Advanced runtime requires deployment bundle v2"
        )
    provenance = route.artifact.metadata.advanced_provenance
    if (
        provenance is None
        or registered.index.manifest_sha256
        != provenance.deployment_bundle_manifest_sha256
    ):
        raise AdvancedRuntimeStateError(
            "managed deployment bundle does not match the active route"
        )
    deployment = next(
        (
            item
            for item in registered.index.artifacts
            if item.artifact_id == route.artifact.artifact_id
        ),
        None,
    )
    matching_routes = tuple(
        item
        for item in registered.index.routes
        if (
            item.task,
            item.cutoff_cycle,
            item.role,
            item.deep_artifact_id,
        )
        == (
            route.task.value,
            route.cutoff_cycle,
            route.role.value,
            route.artifact.artifact_id,
        )
    )
    if deployment is None or len(matching_routes) != 1:
        raise AdvancedRuntimeStateError(
            "managed deployment does not contain the exact active route"
        )
    bundle_route = matching_routes[0]
    route_provenance = route.route_provenance
    if (
        bundle_route.task != route_provenance.task.value
        or bundle_route.role != route_provenance.role.value
        or bundle_route.family != route_provenance.family
        or bundle_route.candidate_id != route_provenance.candidate_id
        or bundle_route.cutoff_cycle != route_provenance.cutoff_cycle
        or bundle_route.seed != route_provenance.seed
        or bundle_route.best_epoch != route_provenance.best_epoch
        or bundle_route.run_id != route_provenance.run_id
        or bundle_route.checkpoint_manifest_sha256
        != route_provenance.checkpoint_manifest_sha256
        or bundle_route.checkpoint_model_sha256
        != route_provenance.checkpoint_model_sha256
        or bundle_route.checkpoint_context_sha256
        != route_provenance.checkpoint_context_sha256
    ):
        raise AdvancedRuntimeStateError(
            "managed deployment route provenance differs from the active route"
        )
    metadata = route.artifact.metadata
    if (
        metadata.schema_version != "deep-model-artifact-v2"
        or metadata.artifact_kind != deployment.artifact_kind.value
        or metadata.dataset_id != deployment.dataset_id
        or metadata.data_version != deployment.data_version
        or metadata.split_version != deployment.split_version
        or metadata.feature_version != deployment.feature_version
        or metadata.cutoff_cycle != deployment.cutoff_cycle
        or route.artifact.model_version != bundle_route.run_id
        or deployment.output_target is None
        or bundle_route.output_target is not deployment.output_target
        or deployment.artifact_kind is not bundle_route.deep_artifact_kind
        or deployment.artifact_manifest_sha256
        != bundle_route.deep_artifact_manifest_sha256
        or deployment.source_checkpoint_manifest_sha256
        != bundle_route.checkpoint_manifest_sha256
        or deployment.source_checkpoint_model_sha256
        != bundle_route.checkpoint_model_sha256
        or deployment.artifact_manifest_sha256
        != route.artifact.manifest_sha256
        or deployment.artifact_manifest_sha256
        != route.artifact.artifact_sha256
        or provenance.candidate_config_sha256
        != deployment.candidate_config_sha256
        or provenance.normalization_sha256 != deployment.normalization_sha256
        or provenance.target_scaler_context_sha256
        != deployment.target_scaler_context_sha256
        or provenance.reference_library_sha256
        != deployment.reference_library_sha256
    ):
        raise AdvancedRuntimeStateError(
            "managed Advanced artifact differs from the active route"
        )
    return deployment


def _artifact_root(bundle_root: Path) -> Path:
    resolved_bundle = bundle_root.resolve(strict=True)
    artifact_root = (resolved_bundle / "artifacts").resolve(strict=True)
    if (
        artifact_root.is_symlink()
        or not artifact_root.is_dir()
        or not artifact_root.is_relative_to(resolved_bundle)
    ):
        raise AdvancedRuntimeResolutionError(
            "managed Advanced artifact root is invalid"
        )
    return artifact_root


def _current_hybrid_feature(
    artifact_root: Path,
    manifest: DeepModelArtifactManifest,
) -> CurrentHybridFeatureConfig:
    feature_file = next(
        item
        for item in manifest.files
        if item.role.value == "feature-config"
    )
    path = (artifact_root / feature_file.relative_path).resolve(strict=True)
    if path.is_symlink() or not path.is_relative_to(artifact_root):
        raise AdvancedRuntimeResolutionError(
            "Current Hybrid feature context is invalid"
        )
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        return CurrentHybridFeatureConfig.model_validate(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AdvancedRuntimeResolutionError(
            "Current Hybrid feature context is invalid"
        ) from exc


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _isolated_inference_model(template: torch.nn.Module) -> torch.nn.Module:
    try:
        model = copy.deepcopy(template)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise AdvancedRuntimeResolutionError(
            "Advanced runtime model could not be isolated"
        ) from exc
    model.eval()
    model.requires_grad_(False)
    return model


def _build_inference_adapter(
    model: torch.nn.Module,
    *,
    deployment: AdvancedDeploymentArtifact,
) -> RULInferenceAdapter | SOHInferenceAdapter:
    if isinstance(model, LoadedCyclePatchDirect):
        if (
            deployment.artifact_kind is not DeepArtifactKind.CYCLEPATCH_DIRECT
            or model.normalizer is None
        ):
            raise AdvancedRuntimeResolutionError(
                "CyclePatch Direct runtime context is incomplete"
            )
        return _VerifiedRULInference(
            artifact_kind=deployment.artifact_kind,
            cutoff_cycle=deployment.cutoff_cycle,
            _model=model,
            _normalizer=model.normalizer,
        )
    if isinstance(model, LoadedCyclePatchBatLiNet):
        if (
            deployment.artifact_kind
            is not DeepArtifactKind.CYCLEPATCH_BATLINET
            or model.normalizer is None
            or model.reference_batch is None
        ):
            raise AdvancedRuntimeResolutionError(
                "CyclePatch-BatLiNet runtime context is incomplete"
            )
        return _VerifiedRULInference(
            artifact_kind=deployment.artifact_kind,
            cutoff_cycle=deployment.cutoff_cycle,
            _model=model,
            _normalizer=model.normalizer,
        )
    if isinstance(model, LoadedHybridPatchV2):
        if (
            deployment.artifact_kind is not DeepArtifactKind.HYBRIDPATCH_V2
            or model.normalizer is None
        ):
            raise AdvancedRuntimeResolutionError(
                "HybridPatch-v2 runtime context is incomplete"
            )
        return _VerifiedSOHInference(
            artifact_kind=deployment.artifact_kind,
            cutoff_cycle=deployment.cutoff_cycle,
            prediction_cycles=tuple(
                range(deployment.cutoff_cycle + 1, 501)
            ),
            _model=model,
            _normalizer=model.normalizer,
        )
    if isinstance(model, LoadedCurrentHybrid):
        if (
            deployment.artifact_kind is not DeepArtifactKind.CURRENT_HYBRID
            or model.normalizer is None
        ):
            raise AdvancedRuntimeResolutionError(
                "Current Hybrid runtime context is incomplete"
            )
        return _VerifiedSOHInference(
            artifact_kind=deployment.artifact_kind,
            cutoff_cycle=deployment.cutoff_cycle,
            prediction_cycles=model.feature.prediction_cycles,
            _model=model,
            _normalizer=model.normalizer,
        )
    raise AdvancedRuntimeResolutionError(
        "Advanced runtime loader returned an unsupported model wrapper"
    )


def _validate_prediction_cycles(
    prediction_cycles: tuple[int, ...],
    *,
    cutoff_cycle: int,
) -> None:
    if (
        not prediction_cycles
        or prediction_cycles[0] <= cutoff_cycle
        or prediction_cycles[-1] != 500
        or any(cycle > 500 for cycle in prediction_cycles)
        or any(
            current <= previous
            for previous, current in pairwise(prediction_cycles)
        )
    ):
        raise AdvancedRuntimeResolutionError(
            "SOH runtime has an invalid finite prediction axis"
        )


def _validate_artifact(
    route: VerifiedActiveModelRoute,
    artifact: VerifiedAdvancedRuntimeArtifact,
) -> None:
    family = route.route_provenance.family
    provenance = route.artifact.metadata.advanced_provenance
    if provenance is None:
        raise AdvancedRuntimeStateError(
            "active route does not contain Advanced artifact provenance"
        )
    expected_kind = {
        "cyclepatch_direct": DeepArtifactKind.CYCLEPATCH_DIRECT,
        "cyclepatch_batlinet": DeepArtifactKind.CYCLEPATCH_BATLINET,
        "current_hybrid": DeepArtifactKind.CURRENT_HYBRID,
        "hybridpatch_v2": DeepArtifactKind.HYBRIDPATCH_V2,
    }.get(family)
    if expected_kind is None:
        raise AdvancedRuntimeResolutionError(
            "active route declares an unsupported Advanced model family"
        )
    expected_target = (
        AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE
        if route.task is AdvancedModelTask.RUL
        else AdvancedOutputTarget.SOH_TRAJECTORY
    )
    if (
        artifact.artifact_id != route.artifact.artifact_id
        or artifact.artifact_manifest_sha256
        != route.artifact.manifest_sha256
        or artifact.model_version != route.artifact.model_version
        or artifact.artifact_kind is not expected_kind
        or artifact.output_target is not expected_target
        or artifact.dataset_id != route.artifact.metadata.dataset_id
        or artifact.data_version != route.artifact.metadata.data_version
        or artifact.feature_version != route.artifact.metadata.feature_version
        or artifact.split_version != route.artifact.metadata.split_version
        or artifact.normalization_sha256
        != provenance.normalization_sha256
        or (
            route.task is AdvancedModelTask.RUL
            and not isinstance(artifact.inference, RULInferenceAdapter)
        )
        or (
            route.task is AdvancedModelTask.SOH
            and not isinstance(artifact.inference, SOHInferenceAdapter)
        )
        or artifact.inference.normalization_statistics_sha256
        != artifact.normalization_sha256
    ):
        raise AdvancedRuntimeStateError(
            "loaded Advanced artifact does not match the active route"
        )


def _route_identity(route: VerifiedActiveModelRoute) -> tuple[object, ...]:
    return (
        route.project_id,
        route.task,
        route.cutoff_cycle,
        route.role,
        route.artifact.artifact_id,
        route.artifact.artifact_sha256,
        route.artifact.manifest_sha256,
        route.route_provenance_sha256,
        route.decision_event_id,
        route.decision_type,
        route.rollback_target_event_id,
        route.ledger_sequence_number,
        route.ledger_head_sha256,
    )


def _artifact_identity(
    artifact: VerifiedAdvancedRuntimeArtifact,
) -> tuple[object, ...]:
    return (
        artifact.artifact_id,
        artifact.artifact_manifest_sha256,
        artifact.model_version,
        artifact.artifact_kind,
        artifact.output_target,
        artifact.dataset_id,
        artifact.data_version,
        artifact.feature_version,
        artifact.split_version,
        artifact.normalization_sha256,
    )


def _runtime_identity(runtime: VerifiedAdvancedRuntime) -> tuple[object, ...]:
    return (
        runtime.project_id,
        runtime.task,
        runtime.cutoff_cycle,
        runtime.role,
        runtime.output_target,
        runtime.artifact_kind,
        runtime.dataset_id,
        runtime.data_version,
        runtime.feature_version,
        runtime.split_version,
        runtime.normalization_sha256,
        runtime.artifact_id,
        runtime.artifact_manifest_sha256,
        runtime.model_version,
        runtime.decision_event_id,
        runtime.ledger_sequence_number,
        runtime.ledger_head_sha256,
    )


__all__ = [
    "ActiveAdvancedRuntimeResolver",
    "AdvancedRuntimeResolutionError",
    "AdvancedRuntimeStateError",
    "ManagedAdvancedRuntimeProvider",
    "RULInferenceAdapter",
    "SOHInferenceAdapter",
    "VerifiedAdvancedRuntime",
    "VerifiedAdvancedRuntimeArtifact",
    "VerifiedRULRuntime",
    "VerifiedSOHRuntime",
]
