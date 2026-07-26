from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch

from quanxin_life.application.advanced_deployment_bundles import (
    AdvancedDeploymentArtifact,
)
from quanxin_life.application.advanced_runtime import (
    ActiveAdvancedRuntimeResolver,
    AdvancedRuntimeResolutionError,
    AdvancedRuntimeStateError,
    ManagedAdvancedRuntimeProvider,
    VerifiedAdvancedRuntimeArtifact,
    VerifiedRULRuntime,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactKind,
    DeepModelArtifactManifest,
    export_current_hybrid_artifact,
    export_cyclepatch_batlinet_artifact,
    export_cyclepatch_direct_artifact,
    export_hybridpatch_v2_artifact,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationSource,
    VerifiedProjectInvocationContext,
)
from quanxin_life.core import (
    AdvancedModelRouteRole,
    AdvancedModelTask,
    ModelRouteDecisionType,
    UserRole,
)
from quanxin_life.models.batlinet import BatLiNetConfig, CyclePatchBatLiNet
from quanxin_life.models.cyclepatch import CyclePatchLifeRegressor
from quanxin_life.models.hybrid_degradation import _HybridNetwork
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Predictor,
)
from tests.unit.application.test_advanced_deployment_artifact_v2 import (
    _cyclepatch_config,
    _normalizer_and_batches,
    _reference_context,
)


def _context() -> VerifiedProjectInvocationContext:
    return VerifiedProjectInvocationContext(
        project_id="project-1",
        actor_user_id="user-1",
        actor_session_id="session-1",
        actor_role=UserRole.ADMIN,
        invocation_source=ProjectInvocationSource.HTTP,
        _authorization_tag="a" * 64,
    )


def _route(
    *,
    artifact_id: str | None = None,
    sequence: int = 1,
    task: AdvancedModelTask = AdvancedModelTask.RUL,
    role: AdvancedModelRouteRole = AdvancedModelRouteRole.DEFAULT,
    family: str = "cyclepatch_direct",
) -> object:
    resolved_artifact_id = artifact_id or str(uuid4())
    candidate_id = family.replace("_", "-")
    run_id = f"{candidate_id}-cutoff-20-seed-38"
    return SimpleNamespace(
        project_id="project-1",
        task=task,
        cutoff_cycle=20,
        role=role,
        artifact=SimpleNamespace(
            artifact_id=resolved_artifact_id,
            artifact_sha256="b" * 64,
            manifest_sha256="b" * 64,
            model_version=run_id,
        ),
        route_provenance=SimpleNamespace(
            task=task,
            role=role,
            family=family,
            candidate_id=candidate_id,
            cutoff_cycle=20,
            seed=38,
            best_epoch=7,
            run_id=run_id,
            checkpoint_manifest_sha256="e" * 64,
            checkpoint_model_sha256="f" * 64,
            checkpoint_context_sha256="1" * 64,
        ),
        route_provenance_sha256="c" * 64,
        decision_event_id=str(uuid4()),
        decision_type=ModelRouteDecisionType.ACTIVATE,
        rollback_target_event_id=None,
        ledger_sequence_number=sequence,
        ledger_head_sha256="d" * 64,
    )


def _artifact(route: object, model: torch.nn.Module) -> VerifiedAdvancedRuntimeArtifact:
    return VerifiedAdvancedRuntimeArtifact(
        artifact_id=route.artifact.artifact_id,
        artifact_manifest_sha256=route.artifact.manifest_sha256,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model=model,
    )


class _ContextService:
    def __init__(self) -> None:
        self.calls = 0

    def revalidate(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> VerifiedProjectInvocationContext:
        self.calls += 1
        return context


class _RouteService:
    def __init__(self, routes: list[object]) -> None:
        self.routes = routes
        self.calls: list[tuple[object, int, object]] = []

    def resolve_verified_active_model_route(
        self,
        _principal: object,
        *,
        project_id: str,
        task: AdvancedModelTask,
        cutoff_cycle: int,
        role: AdvancedModelRouteRole,
    ) -> object:
        assert project_id == "project-1"
        self.calls.append((task, cutoff_cycle, role))
        if len(self.routes) > 1:
            return self.routes.pop(0)
        return self.routes[0]


class _RuntimeProvider:
    def __init__(self, artifacts: list[VerifiedAdvancedRuntimeArtifact]) -> None:
        self.artifacts = artifacts
        self.calls = 0

    def resolve(self, _route: object) -> VerifiedAdvancedRuntimeArtifact:
        self.calls += 1
        if len(self.artifacts) > 1:
            return self.artifacts.pop(0)
        return self.artifacts[0]


def test_resolver_revalidates_exact_route_and_artifact_before_returning_runtime() -> None:
    context = _context()
    route = _route()
    model = torch.nn.Identity()
    context_service = _ContextService()
    route_service = _RouteService([route])
    provider = _RuntimeProvider([_artifact(route, model)])
    resolver = ActiveAdvancedRuntimeResolver(
        context_service=context_service,
        route_service=route_service,
        runtime_provider=provider,
    )

    runtime = resolver.resolve(
        context,
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    )

    assert isinstance(runtime, VerifiedRULRuntime)
    assert runtime.model is model
    assert runtime.output_target is AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert runtime.artifact_id == route.artifact.artifact_id
    assert route_service.calls == [
        (AdvancedModelTask.RUL, 20, AdvancedModelRouteRole.DEFAULT),
        (AdvancedModelTask.RUL, 20, AdvancedModelRouteRole.DEFAULT),
    ]
    assert provider.calls == 2
    assert context_service.calls == 2


def test_resolver_rejects_route_change_during_model_resolution() -> None:
    first = _route(sequence=1)
    second = _route(sequence=2)
    resolver = ActiveAdvancedRuntimeResolver(
        context_service=_ContextService(),
        route_service=_RouteService([first, second]),
        runtime_provider=_RuntimeProvider(
            [_artifact(first, torch.nn.Identity())]
        ),
    )

    with pytest.raises(AdvancedRuntimeStateError, match=r"route.*changed|changed.*route"):
        resolver.resolve(
            _context(),
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
        )


def test_runtime_revalidation_detects_a_later_route_switch() -> None:
    first = _route(sequence=1)
    switched = _route(sequence=2)
    model = torch.nn.Identity()
    route_service = _RouteService([first])
    provider = _RuntimeProvider([_artifact(first, model)])
    resolver = ActiveAdvancedRuntimeResolver(
        context_service=_ContextService(),
        route_service=route_service,
        runtime_provider=provider,
    )
    runtime = resolver.resolve(
        _context(),
        task=AdvancedModelTask.RUL,
        cutoff_cycle=20,
        role=AdvancedModelRouteRole.DEFAULT,
    )
    route_service.routes = [switched]
    provider.artifacts = [_artifact(switched, torch.nn.Identity())]

    with pytest.raises(AdvancedRuntimeStateError, match=r"route.*changed|runtime.*changed"):
        resolver.revalidate(_context(), runtime)


def test_runtime_contract_never_exposes_filesystem_paths() -> None:
    field_types = {field.name: field.type for field in fields(VerifiedRULRuntime)}

    assert Path not in field_types.values()
    assert "path" not in repr(
        VerifiedRULRuntime(
            project_id="project-1",
            task=AdvancedModelTask.RUL,
            cutoff_cycle=20,
            role=AdvancedModelRouteRole.DEFAULT,
            output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
            artifact_id=str(uuid4()),
            artifact_manifest_sha256="a" * 64,
            decision_event_id=str(uuid4()),
            ledger_sequence_number=1,
            ledger_head_sha256="b" * 64,
            model=torch.nn.Identity(),
        )
    ).casefold()


class _ManagedRegistry:
    def __init__(self, registered: object) -> None:
        self.registered = registered
        self.calls = 0

    def resolve(self, _registry_id: str) -> object:
        self.calls += 1
        return self.registered


class _ModelLoader:
    def __init__(self) -> None:
        self.calls = 0
        self.model = torch.nn.Linear(1, 1)

    def load(
        self,
        *,
        artifact_root: Path,
        manifest: object,
        deployment_artifact: object,
    ) -> torch.nn.Module:
        self.calls += 1
        assert artifact_root.is_dir()
        assert manifest.artifact_id == deployment_artifact.artifact_id
        return self.model


def _deployment_artifact(
    manifest: DeepModelArtifactManifest,
    *,
    normalization_sha256: str,
    target_scaler_context_sha256: str | None = None,
    reference_library_sha256: str | None = None,
) -> AdvancedDeploymentArtifact:
    return AdvancedDeploymentArtifact(
        artifact_id=manifest.artifact_id,
        artifact_kind=manifest.artifact_kind,
        output_target=manifest.output_target,
        artifact_manifest_sha256=manifest.manifest_sha256,
        source_checkpoint_manifest_sha256="b" * 64,
        source_checkpoint_model_sha256=next(
            item.sha256
            for item in manifest.files
            if item.role.value == "weights"
        ),
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        candidate_config_sha256="a" * 64,
        normalization_sha256=normalization_sha256,
        target_scaler_context_sha256=target_scaler_context_sha256,
        reference_library_sha256=reference_library_sha256,
        files=manifest.files,
        round_trip_verified=True,
    )


def _managed_fixture(
    tmp_path: Path,
    *,
    manifest: DeepModelArtifactManifest,
    deployment: AdvancedDeploymentArtifact,
    task: AdvancedModelTask,
    role: AdvancedModelRouteRole,
    family: str,
) -> object:
    bundle_sha256 = "f" * 64
    active_route = _route(
        artifact_id=manifest.artifact_id,
        task=task,
        role=role,
        family=family,
    )
    active_route.route_provenance.checkpoint_manifest_sha256 = (
        deployment.source_checkpoint_manifest_sha256
    )
    active_route.route_provenance.checkpoint_model_sha256 = (
        deployment.source_checkpoint_model_sha256
    )
    route_row = SimpleNamespace(
        task=task.value,
        role=role.value,
        cutoff_cycle=20,
        family=family,
        candidate_id=active_route.route_provenance.candidate_id,
        seed=active_route.route_provenance.seed,
        best_epoch=active_route.route_provenance.best_epoch,
        run_id=active_route.route_provenance.run_id,
        checkpoint_manifest_sha256=(
            active_route.route_provenance.checkpoint_manifest_sha256
        ),
        checkpoint_model_sha256=(
            active_route.route_provenance.checkpoint_model_sha256
        ),
        checkpoint_context_sha256=(
            active_route.route_provenance.checkpoint_context_sha256
        ),
        output_target=manifest.output_target,
        deep_artifact_id=manifest.artifact_id,
        deep_artifact_kind=manifest.artifact_kind,
        deep_artifact_manifest_sha256=manifest.manifest_sha256,
    )
    registered = SimpleNamespace(
        bundle_root=tmp_path,
        index=SimpleNamespace(
            schema_version="advanced-deployment-bundle-index-v2",
            manifest_sha256=bundle_sha256,
            artifacts=(deployment,),
            routes=(route_row,),
        ),
    )
    active_route.artifact.artifact_sha256 = manifest.manifest_sha256
    active_route.artifact.manifest_sha256 = manifest.manifest_sha256
    active_route.artifact.metadata = SimpleNamespace(
        artifact_kind=manifest.artifact_kind.value,
        dataset_id="MATR",
        data_version=deployment.data_version,
        split_version=deployment.split_version,
        feature_version=deployment.feature_version,
        cutoff_cycle=deployment.cutoff_cycle,
        schema_version="deep-model-artifact-v2",
        advanced_provenance=SimpleNamespace(
            deployment_bundle_manifest_sha256=bundle_sha256,
            candidate_config_sha256=deployment.candidate_config_sha256,
            normalization_sha256=deployment.normalization_sha256,
            target_scaler_context_sha256=(
                deployment.target_scaler_context_sha256
            ),
            reference_library_sha256=deployment.reference_library_sha256,
        ),
    )
    return SimpleNamespace(active_route=active_route, registered=registered)


def _managed_direct_fixture(tmp_path: Path) -> tuple[object, Path]:
    normalizer, _reference_batch, _target_batch = _normalizer_and_batches()
    scaler, _reference_library = _reference_context()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    network = CyclePatchLifeRegressor(
        _cyclepatch_config(),
        condition_count=1,
    ).eval()
    manifest = export_cyclepatch_direct_artifact(
        network,
        artifact_root=artifact_root,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256="a" * 64,
        target_scaler=scaler,
        normalizer=normalizer,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    deployment = _deployment_artifact(
        manifest,
        normalization_sha256=normalizer.statistics_sha256,
        target_scaler_context_sha256=scaler.context_sha256,
    )
    fixture = _managed_fixture(
        tmp_path,
        manifest=manifest,
        deployment=deployment,
        task=AdvancedModelTask.RUL,
        role=AdvancedModelRouteRole.DEFAULT,
        family="cyclepatch_direct",
    )
    return fixture, artifact_root


def _managed_all_family_fixtures(
    tmp_path: Path,
) -> tuple[tuple[object, DeepArtifactKind, AdvancedOutputTarget], ...]:
    normalizer, reference_batch, _target_batch = _normalizer_and_batches()
    scaler, reference_library = _reference_context()
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    direct = export_cyclepatch_direct_artifact(
        CyclePatchLifeRegressor(
            _cyclepatch_config(),
            condition_count=1,
        ).eval(),
        artifact_root=artifact_root,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256="a" * 64,
        target_scaler=scaler,
        normalizer=normalizer,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    batlinet = export_cyclepatch_batlinet_artifact(
        CyclePatchBatLiNet(
            BatLiNetConfig(
                encoder=_cyclepatch_config(),
                lambda_pair=0.5,
                lambda_rank=0.1,
                fusion_alpha=0.5,
                reference_count=16,
            ),
            condition_count=1,
        ).eval(),
        artifact_root=artifact_root,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256="a" * 64,
        reference_library=reference_library,
        reference_batch=reference_batch,
        target_scaler=scaler,
        normalizer=normalizer,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    hybridpatch = export_hybridpatch_v2_artifact(
        HybridPatchV2Predictor(
            HybridPatchV2Config(
                cyclepatch=_cyclepatch_config(),
                query_token_count=8,
                query_layers=1,
                decoder_hidden_dim=64,
                lambda_history=0.1,
                lambda_smooth=0.01,
                lambda_order=0.05,
                lambda_residual=0.01,
            ),
            condition_count=1,
        ).eval(),
        artifact_root=artifact_root,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256="a" * 64,
        normalizer=normalizer,
        output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
    )
    prediction_cycles = (21, 100, 500)
    current_hybrid = export_current_hybrid_artifact(
        _HybridNetwork(
            input_dim=3,
            horizon=len(prediction_cycles),
            hidden_dim=32,
        ).eval(),
        artifact_root=artifact_root,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 26, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="cyclepatch-multichannel-v1",
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        prediction_cycles=prediction_cycles,
        variable_names=("voltage_v", "current_a", "capacity_ah"),
        aggregation_version="masked-variable-mean-v1",
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256="a" * 64,
        normalizer=normalizer,
        output_target=AdvancedOutputTarget.SOH_TRAJECTORY,
    )
    specifications = (
        (
            direct,
            _deployment_artifact(
                direct,
                normalization_sha256=normalizer.statistics_sha256,
                target_scaler_context_sha256=scaler.context_sha256,
            ),
            AdvancedModelTask.RUL,
            AdvancedModelRouteRole.DEFAULT,
            "cyclepatch_direct",
        ),
        (
            batlinet,
            _deployment_artifact(
                batlinet,
                normalization_sha256=normalizer.statistics_sha256,
                target_scaler_context_sha256=scaler.context_sha256,
                reference_library_sha256=reference_library.library_sha256,
            ),
            AdvancedModelTask.RUL,
            AdvancedModelRouteRole.COVERAGE,
            "cyclepatch_batlinet",
        ),
        (
            hybridpatch,
            _deployment_artifact(
                hybridpatch,
                normalization_sha256=normalizer.statistics_sha256,
            ),
            AdvancedModelTask.SOH,
            AdvancedModelRouteRole.MEAN_ACCURACY,
            "hybridpatch_v2",
        ),
        (
            current_hybrid,
            _deployment_artifact(
                current_hybrid,
                normalization_sha256=normalizer.statistics_sha256,
            ),
            AdvancedModelTask.SOH,
            AdvancedModelRouteRole.TAIL_EFFICIENCY,
            "current_hybrid",
        ),
    )
    return tuple(
        (
            _managed_fixture(
                tmp_path,
                manifest=manifest,
                deployment=deployment,
                task=task,
                role=role,
                family=family,
            ),
            manifest.artifact_kind,
            manifest.output_target,
        )
        for manifest, deployment, task, role, family in specifications
    )


def test_managed_provider_reverifies_bytes_and_isolates_cached_model_instances(
    tmp_path: Path,
) -> None:
    fixture, _artifact_root = _managed_direct_fixture(tmp_path)
    registry = _ManagedRegistry(fixture.registered)
    loader = _ModelLoader()
    provider = ManagedAdvancedRuntimeProvider(registry, model_loader=loader)

    first = provider.resolve(fixture.active_route)
    expected_weight = loader.model.weight.detach().clone()
    first.model.train()
    with torch.no_grad():
        first.model.weight.add_(1.0)
    second = provider.resolve(fixture.active_route)

    assert first.model is not second.model
    assert first.model is not loader.model
    assert second.model is not loader.model
    assert second.model.training is False
    assert torch.equal(second.model.weight, expected_weight)
    assert all(not parameter.requires_grad for parameter in second.model.parameters())
    assert registry.calls == 2
    assert loader.calls == 1


def test_managed_provider_loads_each_self_contained_v2_family(
    tmp_path: Path,
) -> None:
    fixtures = _managed_all_family_fixtures(tmp_path)

    for fixture, expected_kind, expected_target in fixtures:
        runtime = ManagedAdvancedRuntimeProvider(
            _ManagedRegistry(fixture.registered)
        ).resolve(fixture.active_route)

        assert runtime.artifact_kind is expected_kind
        assert runtime.output_target is expected_target
        assert isinstance(runtime.model, torch.nn.Module)
        assert runtime.model.training is False
        assert runtime.model.normalizer is not None
    batlinet_runtime = ManagedAdvancedRuntimeProvider(
        _ManagedRegistry(fixtures[1][0].registered)
    ).resolve(fixtures[1][0].active_route)
    assert batlinet_runtime.model.reference_batch is not None


def test_managed_provider_cache_cannot_hide_artifact_tampering(
    tmp_path: Path,
) -> None:
    fixture, artifact_root = _managed_direct_fixture(tmp_path)
    registry = _ManagedRegistry(fixture.registered)
    loader = _ModelLoader()
    provider = ManagedAdvancedRuntimeProvider(registry, model_loader=loader)
    provider.resolve(fixture.active_route)
    weights = next(artifact_root.rglob("model.safetensors"))
    payload = bytearray(weights.read_bytes())
    payload[-1] ^= 1
    weights.write_bytes(payload)

    with pytest.raises(ValueError, match=r"SHA-256|artifact|manifest"):
        provider.resolve(fixture.active_route)

    assert registry.calls == 2
    assert loader.calls == 1


def test_managed_provider_rejects_legacy_v1_runtime(
    tmp_path: Path,
) -> None:
    fixture, _artifact_root = _managed_direct_fixture(tmp_path)
    fixture.registered.index.schema_version = "advanced-deployment-bundle-index-v1"
    registry = _ManagedRegistry(fixture.registered)
    provider = ManagedAdvancedRuntimeProvider(registry, model_loader=_ModelLoader())

    with pytest.raises(
        AdvancedRuntimeResolutionError,
        match=r"v2|self-contained",
    ):
        provider.resolve(fixture.active_route)


def test_managed_provider_rejects_route_provenance_drift(
    tmp_path: Path,
) -> None:
    fixture, _artifact_root = _managed_direct_fixture(tmp_path)
    fixture.active_route.route_provenance.run_id = "tampered-run"
    provider = ManagedAdvancedRuntimeProvider(
        _ManagedRegistry(fixture.registered),
        model_loader=_ModelLoader(),
    )

    with pytest.raises(
        AdvancedRuntimeStateError,
        match=r"route|provenance|deployment",
    ):
        provider.resolve(fixture.active_route)


def test_managed_provider_rejects_artifact_provenance_drift(
    tmp_path: Path,
) -> None:
    fixture, _artifact_root = _managed_direct_fixture(tmp_path)
    fixture.active_route.artifact.metadata.advanced_provenance.normalization_sha256 = (
        "9" * 64
    )
    provider = ManagedAdvancedRuntimeProvider(
        _ManagedRegistry(fixture.registered),
        model_loader=_ModelLoader(),
    )

    with pytest.raises(
        AdvancedRuntimeStateError,
        match=r"artifact|provenance|deployment",
    ):
        provider.resolve(fixture.active_route)
