from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

import quanxin_life.application.advanced_deployment_registry as deployment_registry
from quanxin_life.application.advanced_deployment_bundles import (
    AdvancedDeploymentArtifact,
    AdvancedDeploymentBundleIndex,
    AdvancedDeploymentSourceRoute,
)
from quanxin_life.application.advanced_deployment_registry import (
    AdvancedDeepModelArtifactCatalogSource,
    AdvancedDeploymentBundleRegistry,
    RegisteredAdvancedDeploymentBundle,
)
from quanxin_life.application.deep_model_artifacts import (
    DeepArtifactFile,
    DeepArtifactFileRole,
    DeepArtifactKind,
    DeepModelArtifactManifest,
)
from quanxin_life.core import sha256_canonical
from quanxin_life.features.early_cycle_sequence import VARIABLE_NAMES

NOW = datetime(2026, 7, 24, 16, 0, tzinfo=UTC)
SOURCE_COMMIT = "2" * 40
DATA_VERSION = "matr-three-batch-test-v1"
SPLIT_VERSION = "matr-three-batch-cell-split-v1"
FEATURE_VERSION = "cyclepatch-multichannel-v1"

ROUTES = (
    ("RUL", 20, "DEFAULT", "cyclepatch_direct"),
    ("RUL", 50, "POINT_ACCURACY", "cyclepatch_direct"),
    ("RUL", 50, "COVERAGE", "cyclepatch_batlinet"),
    ("RUL", 100, "POINT_ACCURACY", "cyclepatch_batlinet"),
    ("RUL", 100, "COVERAGE", "cyclepatch_direct"),
    ("RUL", 150, "POINT_ACCURACY", "cyclepatch_direct"),
    ("RUL", 150, "COVERAGE", "cyclepatch_batlinet"),
    *(("SOH", cutoff, "MEAN_ACCURACY", "hybridpatch_v2") for cutoff in (20, 50, 100, 150)),
    *(("SOH", cutoff, "TAIL_EFFICIENCY", "current_hybrid") for cutoff in (20, 50, 100, 150)),
)

KINDS = {
    "cyclepatch_direct": DeepArtifactKind.CYCLEPATCH_DIRECT,
    "cyclepatch_batlinet": DeepArtifactKind.CYCLEPATCH_BATLINET,
    "current_hybrid": DeepArtifactKind.CURRENT_HYBRID,
    "hybridpatch_v2": DeepArtifactKind.HYBRIDPATCH_V2,
}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _resign_index(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = sha256_canonical(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    _write_json(path, payload)
    return str(payload["manifest_sha256"])


def _create_junction(link: Path, target: Path) -> bool:
    created = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    return created.returncode == 0


def _build_artifact(
    artifact_root: Path,
    *,
    coordinate: tuple[str, int, str, str],
    ordinal: int,
) -> tuple[DeepModelArtifactManifest, AdvancedDeploymentArtifact]:
    task, cutoff, role, family = coordinate
    artifact_id = str(uuid5(NAMESPACE_URL, f"{task}:{cutoff}:{role}:{family}"))
    directory = artifact_root / artifact_id
    directory.mkdir()
    payloads: list[tuple[DeepArtifactFileRole, str, bytes]] = [
        (DeepArtifactFileRole.WEIGHTS, "model.safetensors", f"weights-{ordinal}".encode()),
        (DeepArtifactFileRole.ARCHITECTURE, "architecture.json", b"{}\n"),
        (
            DeepArtifactFileRole.FEATURE_CONFIG,
            "feature_config.json",
            (
                json.dumps(
                    {
                        "candidate_config_sha256": _digest(f"candidate-{ordinal}"),
                        "condition_names": [
                            "mean_temperature_c",
                            "mean_charge_current_a",
                            "mean_discharge_current_a",
                        ],
                        "cutoff_cycle": cutoff,
                        "data_version": DATA_VERSION,
                        "dataset_id": "MATR",
                        "feature_version": FEATURE_VERSION,
                        "normalization_sha256": _digest(f"normalization-{ordinal}"),
                        "schema_version": "advanced-feature-context-v1",
                        "split_version": SPLIT_VERSION,
                        "variable_names": list(VARIABLE_NAMES),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode(),
        ),
    ]
    if family == "cyclepatch_batlinet":
        payloads.append(
            (
                DeepArtifactFileRole.REFERENCE_LIBRARY,
                "reference_library.json",
                b"{}\n",
            )
        )
    files: list[DeepArtifactFile] = []
    for file_role, name, payload in payloads:
        path = directory / name
        path.write_bytes(payload)
        files.append(
            DeepArtifactFile(
                role=file_role,
                relative_path=f"{artifact_id}/{name}",
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    manifest_payload = {
        "schema_version": "deep-model-artifact-v1",
        "artifact_id": artifact_id,
        "artifact_kind": KINDS[family].value,
        "files": [item.model_dump(mode="json") for item in files],
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
    }
    manifest = DeepModelArtifactManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    _write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
    candidate_sha = _digest(f"candidate-{ordinal}")
    normalization_sha = _digest(f"normalization-{ordinal}")
    checkpoint_model_sha = files[0].sha256
    deployment = AdvancedDeploymentArtifact(
        artifact_id=artifact_id,
        artifact_kind=KINDS[family],
        artifact_manifest_sha256=manifest.manifest_sha256,
        source_checkpoint_manifest_sha256=_digest(f"checkpoint-{ordinal}"),
        source_checkpoint_model_sha256=checkpoint_model_sha,
        data_version=DATA_VERSION,
        split_version=SPLIT_VERSION,
        feature_version=FEATURE_VERSION,
        cutoff_cycle=cutoff,
        candidate_config_sha256=candidate_sha,
        normalization_sha256=normalization_sha,
        target_scaler_context_sha256=(
            _digest(f"scaler-{ordinal}") if task == "RUL" else None
        ),
        reference_library_sha256=(
            _digest(f"reference-{ordinal}")
            if family == "cyclepatch_batlinet"
            else None
        ),
        files=tuple(files),
        round_trip_verified=True,
    )
    return manifest, deployment


def _build_bundle(root: Path) -> AdvancedDeploymentBundleIndex:
    root.mkdir()
    artifact_root = root / "artifacts"
    artifact_root.mkdir()
    routes: list[AdvancedDeploymentSourceRoute] = []
    artifacts: list[AdvancedDeploymentArtifact] = []
    for ordinal, coordinate in enumerate(ROUTES, start=1):
        task, cutoff, role, family = coordinate
        manifest, artifact = _build_artifact(
            artifact_root,
            coordinate=coordinate,
            ordinal=ordinal,
        )
        artifacts.append(artifact)
        routes.append(
            AdvancedDeploymentSourceRoute(
                task=task,
                role=role,
                family=family,
                candidate_id=f"candidate-{ordinal}",
                data_version=DATA_VERSION,
                split_version=SPLIT_VERSION,
                feature_version=FEATURE_VERSION,
                cutoff_cycle=cutoff,
                seed=38 + (ordinal % 5),
                best_epoch=ordinal,
                representative_seed_rule="MINIMUM_BEST_VALIDATION_METRIC",
                run_id=f"advanced-run-{ordinal}",
                checkpoint_directory=f"final/run-{ordinal}/checkpoints/best",
                checkpoint_manifest_sha256=artifact.source_checkpoint_manifest_sha256,
                checkpoint_manifest_file_sha256=_digest(
                    f"checkpoint-file-{ordinal}"
                ),
                checkpoint_model_sha256=artifact.source_checkpoint_model_sha256,
                checkpoint_context_sha256=_digest(f"context-{ordinal}"),
                selection_manifest_sha256=_digest("selection"),
                candidate_config_sha256=artifact.candidate_config_sha256,
                normalization_sha256=artifact.normalization_sha256,
                reference_library_sha256=artifact.reference_library_sha256,
                deep_artifact_id=artifact.artifact_id,
                deep_artifact_kind=manifest.artifact_kind,
                deep_artifact_manifest_sha256=manifest.manifest_sha256,
            )
        )
    payload = {
        "schema_version": "advanced-deployment-bundle-index-v1",
        "activation_status": "NOT_ACTIVATED",
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "source_commit": SOURCE_COMMIT,
        "final_output_sha256": _digest("final-output"),
        "final_config_sha256": _digest("final-config"),
        "data_version": DATA_VERSION,
        "split_version": SPLIT_VERSION,
        "feature_version": FEATURE_VERSION,
        "training_input_bundle_sha256": _digest("training-input"),
        "local_reconstructed_input_bundle_sha256": _digest("local-input"),
        "input_bundle_hashes_match": False,
        "promotion_manifest_sha256": _digest("promotion-manifest"),
        "promotion_decisions_sha256": _digest("promotion-decisions"),
        "promotion_source_evidence_sha256": _digest("promotion-source"),
        "selection_manifest_sha256": _digest("selection"),
        "routes": [item.model_dump(mode="json") for item in routes],
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
    }
    index = AdvancedDeploymentBundleIndex.model_validate(
        {**payload, "manifest_sha256": sha256_canonical(payload)}
    )
    _write_json(root / "deployment_bundle_index.json", index.model_dump(mode="json"))
    return index


def test_registers_inactive_bundle_atomically_and_resolves_catalog_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")

    first = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )
    second = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW.replace(hour=17),
    )

    assert second == first
    assert first.activation_status == "NOT_ACTIVATED"
    assert first.route_count == 15
    assert first.artifact_count == 15
    assert first.input_bundle_hashes_match is False
    assert first.registered_relative_root == (
        f"bundles/{index.manifest_sha256[:16]}"
    )
    resolved = registry.resolve(first.registry_id)
    assert resolved.record == first
    assert resolved.index == index
    assert resolved.bundle_root.is_dir()

    source = AdvancedDeepModelArtifactCatalogSource(registry, first.registry_id)
    assert source.list_artifact_ids() == tuple(
        sorted(item.artifact_id for item in index.artifacts)
    )
    route = index.routes[0]
    candidate = source.resolve(route.deep_artifact_id)
    assert candidate.artifact_id == route.deep_artifact_id
    assert candidate.artifact_format == "safetensors-bundle"
    assert candidate.object_uri == (
        f"verified-model-artifact://{route.deep_artifact_id}/bundle"
    )
    assert candidate.artifact_sha256 == route.deep_artifact_manifest_sha256
    assert candidate.model_version == route.run_id
    assert candidate.manifest_sha256 == route.deep_artifact_manifest_sha256
    assert candidate.metadata.artifact_kind == route.deep_artifact_kind.value
    assert candidate.metadata.feature_names == VARIABLE_NAMES
    provenance = candidate.metadata.advanced_provenance
    assert provenance is not None
    assert provenance.lifecycle_status == "REGISTERED_CANDIDATE"
    assert provenance.activation_status == "NOT_ACTIVATED"
    assert provenance.deployment_bundle_manifest_sha256 == index.manifest_sha256
    assert provenance.final_output_sha256 == index.final_output_sha256
    assert provenance.training_input_bundle_sha256 != (
        provenance.local_reconstructed_input_bundle_sha256
    )
    assert provenance.input_bundle_hashes_match is False
    assert len(provenance.routes) == 1
    assert provenance.routes[0].task == route.task
    assert provenance.routes[0].role == route.role
    assert provenance.routes[0].disposition == "CONDITIONAL"
    assert provenance.routes[0].checkpoint_model_sha256 == (
        route.checkpoint_model_sha256
    )

    resolve_calls = 0
    original_resolve = registry.resolve

    def counted_resolve(registry_id: str) -> RegisteredAdvancedDeploymentBundle:
        nonlocal resolve_calls
        resolve_calls += 1
        return original_resolve(registry_id)

    monkeypatch.setattr(registry, "resolve", counted_resolve)
    candidates = source.resolve_all()
    assert len(candidates) == 15
    assert tuple(item.artifact_id for item in candidates) == tuple(
        sorted(item.artifact_id for item in index.artifacts)
    )
    assert resolve_calls == 1


def test_registration_requires_external_manifest_trust_anchor(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    index_path = source_root / "deployment_bundle_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["created_at"] = NOW.replace(hour=18).isoformat().replace("+00:00", "Z")
    _write_json(index_path, payload)
    changed_sha = _resign_index(index_path)
    assert changed_sha != index.manifest_sha256

    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"expected.*manifest|trust anchor"):
        registry.register(
            source_root,
            expected_manifest_sha256=index.manifest_sha256,
            registered_at=NOW,
        )
    assert not (tmp_path / "registry" / "bundles" / changed_sha).exists()


def test_registration_rejects_naive_timestamp(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")

    with pytest.raises(ValueError, match=r"registered_at.*timezone"):
        registry.register(
            source_root,
            expected_manifest_sha256=index.manifest_sha256,
            registered_at=datetime(2026, 7, 24, 16, 0),
        )


def test_registration_rejects_false_input_bundle_comparison(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _build_bundle(source_root)
    index_path = source_root / "deployment_bundle_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["training_input_bundle_sha256"] != (
        payload["local_reconstructed_input_bundle_sha256"]
    )
    payload["input_bundle_hashes_match"] = True
    _write_json(index_path, payload)
    changed_sha = _resign_index(index_path)

    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"input bundle comparison|digests"):
        registry.register(
            source_root,
            expected_manifest_sha256=changed_sha,
            registered_at=NOW,
        )


def test_registration_rejects_incomplete_or_activated_route_matrix(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _build_bundle(source_root)
    index_path = source_root / "deployment_bundle_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    removed = payload["routes"].pop()
    payload["artifacts"] = [
        item
        for item in payload["artifacts"]
        if item["artifact_id"] != removed["deep_artifact_id"]
    ]
    artifact_directory = source_root / "artifacts" / removed["deep_artifact_id"]
    for path in artifact_directory.iterdir():
        path.unlink()
    artifact_directory.rmdir()
    _write_json(index_path, payload)
    incomplete_sha = _resign_index(index_path)

    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"exact.*route|route matrix"):
        registry.register(
            source_root,
            expected_manifest_sha256=incomplete_sha,
            registered_at=NOW,
        )

    source_root = tmp_path / "activated-source"
    _build_bundle(source_root)
    index_path = source_root / "deployment_bundle_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["activation_status"] = "ACTIVE"
    payload["routes"][0]["activation_status"] = "ACTIVE"
    _write_json(index_path, payload)
    active_sha = _resign_index(index_path)
    with pytest.raises(ValueError, match=r"NOT_ACTIVATED|activation"):
        registry.register(
            source_root,
            expected_manifest_sha256=active_sha,
            registered_at=NOW,
        )


def test_resolve_reverifies_registered_bytes(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    record = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )
    registered = registry.resolve(record.registry_id)
    weight = next(
        registered.bundle_root.glob("artifacts/*/model.safetensors")
    )
    weight.write_bytes(b"tampered")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        registry.resolve(record.registry_id)
    source = AdvancedDeepModelArtifactCatalogSource(registry, record.registry_id)
    with pytest.raises(ValueError, match=r"SHA-256|size"):
        source.resolve(index.artifacts[0].artifact_id)


def test_registration_record_tampering_is_detected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry_root = tmp_path / "registry"
    registry = AdvancedDeploymentBundleRegistry(registry_root)
    record = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )
    record_path = registry_root / "records" / f"{record.registry_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["registered_at"] = "2035-01-01T00:00:00Z"
    _write_json(record_path, payload)

    with pytest.raises(ValueError, match=r"record SHA-256|registration.*contents"):
        registry.resolve(record.registry_id)


def test_registration_record_cannot_be_resigned_to_another_internal_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry_root = tmp_path / "registry"
    registry = AdvancedDeploymentBundleRegistry(registry_root)
    record = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )
    record_path = registry_root / "records" / f"{record.registry_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["registered_relative_root"] = "bundles/alternate"
    payload["record_sha256"] = sha256_canonical(
        {key: value for key, value in payload.items() if key != "record_sha256"}
    )
    _write_json(record_path, payload)

    with pytest.raises(ValueError, match=r"not canonical"):
        registry.resolve(record.registry_id)


def test_registration_recovers_a_verified_destination_after_process_crash(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry_root = tmp_path / "registry"
    registry = AdvancedDeploymentBundleRegistry(registry_root)
    destination = registry_root / "bundles" / index.manifest_sha256[:16]
    shutil.copytree(source_root, destination)

    record = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )

    assert record.registry_id == index.manifest_sha256
    assert registry.resolve(record.registry_id).index == index


@pytest.mark.parametrize("managed_name", ["bundles", "records"])
@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_registry_rejects_junction_outside_managed_root(
    tmp_path: Path,
    managed_name: str,
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    other_name = "records" if managed_name == "bundles" else "bundles"
    (registry_root / other_name).mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if not _create_junction(registry_root / managed_name, outside):
        pytest.skip("Windows junction creation is unavailable")

    with pytest.raises(ValueError, match=r"reparse|junction|registry directories"):
        AdvancedDeploymentBundleRegistry(registry_root)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_registration_rejects_junction_swap_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry_root = tmp_path / "registry"
    registry = AdvancedDeploymentBundleRegistry(registry_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_copytree = shutil.copytree
    swapped = False

    def swap_then_copy(*args: object, **kwargs: object) -> Path:
        nonlocal swapped
        if not swapped:
            swapped = True
            bundles = registry_root / "bundles"
            bundles.rmdir()
            if not _create_junction(bundles, outside):
                pytest.skip("Windows junction creation is unavailable")
        return original_copytree(*args, **kwargs)

    monkeypatch.setattr(deployment_registry.shutil, "copytree", swap_then_copy)

    with pytest.raises(ValueError, match=r"reparse|escaped"):
        registry.register(
            source_root,
            expected_manifest_sha256=index.manifest_sha256,
            registered_at=NOW,
        )
    assert tuple(outside.iterdir()) == ()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_resolve_rejects_nested_artifact_junction(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    registry_root = tmp_path / "registry"
    registry = AdvancedDeploymentBundleRegistry(registry_root)
    record = registry.register(
        source_root,
        expected_manifest_sha256=index.manifest_sha256,
        registered_at=NOW,
    )
    managed_root = registry_root / record.registered_relative_root
    artifact_root = managed_root / "artifacts"
    outside = tmp_path / "outside-artifacts"
    artifact_root.rename(outside)
    if not _create_junction(artifact_root, outside):
        pytest.skip("Windows junction creation is unavailable")

    with pytest.raises(ValueError, match=r"reparse|escaped|inside|regular"):
        registry.resolve(record.registry_id)


def test_registration_rejects_unindexed_files_without_deserializing_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    (source_root / "artifacts" / "untrusted.pth").write_bytes(b"forbidden")

    def fail_if_deserialized(*_: object, **__: object) -> None:
        raise AssertionError("registration must not deserialize model weights")

    monkeypatch.setattr("safetensors.torch.load_file", fail_if_deserialized)
    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"unexpected|closed|unindexed"):
        registry.register(
            source_root,
            expected_manifest_sha256=index.manifest_sha256,
            registered_at=NOW,
        )


def test_registration_rejects_symbolic_link_artifact_file(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    index = _build_bundle(source_root)
    weight = next(source_root.glob("artifacts/*/model.safetensors"))
    target = tmp_path / "outside.safetensors"
    target.write_bytes(weight.read_bytes())
    weight.unlink()
    try:
        weight.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable in this environment")

    registry = AdvancedDeploymentBundleRegistry(tmp_path / "registry")
    with pytest.raises(ValueError, match=r"symbolic|symlink|regular"):
        registry.register(
            source_root,
            expected_manifest_sha256=index.manifest_sha256,
            registered_at=NOW,
        )
