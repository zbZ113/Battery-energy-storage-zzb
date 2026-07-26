from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from quanxin_life.application import advanced_deployment_bundles as bundles
from quanxin_life.application.advanced_deployment_registry import (
    AdvancedDeepModelArtifactCatalogSource,
)
from quanxin_life.application.deep_model_artifacts import (
    AdvancedOutputTarget,
    DeepArtifactFile,
    DeepArtifactFileRole,
    DeepArtifactKind,
)
from quanxin_life.core.hashing import sha256_canonical

_NOW = datetime(2026, 7, 26, tzinfo=UTC)
_SHA = "a" * 64


@pytest.mark.parametrize(
    ("family", "expected_target", "normalizer_name", "expects_reference"),
    (
        (
            "cyclepatch_direct",
            AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
            "scalar_normalizer",
            False,
        ),
        (
            "cyclepatch_batlinet",
            AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
            "scalar_normalizer",
            True,
        ),
        (
            "hybridpatch_v2",
            AdvancedOutputTarget.SOH_TRAJECTORY,
            "hybrid_normalizer",
            False,
        ),
        (
            "current_hybrid",
            AdvancedOutputTarget.SOH_TRAJECTORY,
            "hybrid_normalizer",
            False,
        ),
    ),
)
def test_rebuild_exports_self_contained_v2_inference_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    expected_target: AdvancedOutputTarget,
    normalizer_name: str,
    expects_reference: bool,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    (checkpoint_root / "manifest.json").write_text("{}", encoding="utf-8")
    scalar_normalizer = object()
    hybrid_normalizer = object()
    reference_batch = object()
    early_batch = SimpleNamespace(condition_names=("temperature_c",))
    data = SimpleNamespace(
        scalar_train=SimpleNamespace(early_batch=early_batch),
        hybrid_train=SimpleNamespace(
            inputs=SimpleNamespace(early_batch=early_batch)
        ),
        scalar_normalizer=scalar_normalizer,
        hybrid_normalizer=hybrid_normalizer,
    )
    rebuilt = SimpleNamespace(
        config=object(),
        data=data,
        split=object(),
    )
    key = SimpleNamespace(family=family)
    candidate = SimpleNamespace(config_sha256=_SHA)
    model = SimpleNamespace(eval=lambda: None)
    task = SimpleNamespace(
        model=model,
        target_scaler=object(),
        reference_library=SimpleNamespace(
            library_sha256="b" * 64,
            cell_ids=("reference-cell",),
        ),
        train_batch=SimpleNamespace(prediction_cycles=(21, 100, 500)),
    )
    checkpoint_manifest = SimpleNamespace(
        manifest_sha256="c" * 64,
        created_at=_NOW,
        context=SimpleNamespace(
            dataset_id="MATR",
            data_version="matr-v1",
            split_version="split-v1",
            feature_version="feature-v1",
            normalization_sha256=_SHA,
            candidate_config_sha256=_SHA,
            reference_library_sha256=(
                "b" * 64 if family == "cyclepatch_batlinet" else None
            ),
        ),
    )
    monkeypatch.setattr(
        bundles,
        "_load_rebuild_data",
        lambda *_args, **_kwargs: rebuilt,
    )
    monkeypatch.setattr(
        bundles,
        "_resolve_run_key",
        lambda *_args, **_kwargs: key,
    )
    monkeypatch.setattr(
        bundles,
        "_candidate_for_key",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(
        bundles,
        "_build_training_task",
        lambda *_args, **_kwargs: task,
    )
    monkeypatch.setattr(
        bundles,
        "_validate_rebuilt_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bundles,
        "load_advanced_inference_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bundles,
        "_verify_round_trip",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        bundles,
        "_reference_batch",
        lambda *_args, **_kwargs: reference_batch,
    )

    class _ManifestParser:
        @staticmethod
        def model_validate_json(_payload: bytes) -> object:
            return checkpoint_manifest

    monkeypatch.setattr(bundles, "AdvancedTrainingCheckpointManifest", _ManifestParser)
    captured: dict[str, object] = {}
    exported = object()

    def capture_export(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return exported

    exporter_name = {
        "cyclepatch_direct": "export_cyclepatch_direct_artifact",
        "cyclepatch_batlinet": "export_cyclepatch_batlinet_artifact",
        "hybridpatch_v2": "export_hybridpatch_v2_artifact",
        "current_hybrid": "export_current_hybrid_artifact",
    }[family]
    monkeypatch.setattr(bundles, exporter_name, capture_export)

    result = bundles._rebuild_and_export_artifact(
        project_root=tmp_path,
        result_root=tmp_path,
        route={
            "cutoff_cycle": 20,
            "family": family,
            "candidate_id": "candidate",
            "seed": 38,
            "checkpoint_directory": "checkpoint",
        },
        artifact_root=tmp_path / "artifacts",
    )

    assert result is exported
    assert captured["normalizer"] is getattr(data, normalizer_name)
    assert captured["output_target"] is expected_target
    if expects_reference:
        assert captured["reference_batch"] is reference_batch
    else:
        assert "reference_batch" not in captured


def test_v2_deployment_contract_binds_output_target_and_keeps_v1_parseable(
    tmp_path: Path,
) -> None:
    artifact_id = "00000000-0000-0000-0000-000000000001"
    files = tuple(
        DeepArtifactFile(
            role=role,
            relative_path=f"{artifact_id}/{name}",
            size_bytes=1,
            sha256=digest,
        )
        for role, name, digest in (
            (DeepArtifactFileRole.WEIGHTS, "model.safetensors", "1" * 64),
            (DeepArtifactFileRole.ARCHITECTURE, "architecture.json", "2" * 64),
            (DeepArtifactFileRole.FEATURE_CONFIG, "feature_config.json", "3" * 64),
            (
                DeepArtifactFileRole.INFERENCE_CONTEXT,
                "inference_context.json",
                "4" * 64,
            ),
        )
    )
    artifact = bundles.AdvancedDeploymentArtifact(
        artifact_id=artifact_id,
        artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
        artifact_manifest_sha256="5" * 64,
        source_checkpoint_manifest_sha256="6" * 64,
        source_checkpoint_model_sha256="1" * 64,
        data_version="matr-v1",
        split_version="split-v1",
        feature_version="feature-v1",
        cutoff_cycle=20,
        candidate_config_sha256="7" * 64,
        normalization_sha256="8" * 64,
        target_scaler_context_sha256="9" * 64,
        files=files,
        round_trip_verified=True,
    )
    route = bundles.AdvancedDeploymentSourceRoute(
        task="RUL",
        role="DEFAULT",
        family="cyclepatch_direct",
        output_target=AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE,
        candidate_id="candidate",
        data_version="matr-v1",
        split_version="split-v1",
        feature_version="feature-v1",
        cutoff_cycle=20,
        seed=38,
        best_epoch=1,
        representative_seed_rule="MINIMUM_BEST_VALIDATION_METRIC",
        run_id="run",
        checkpoint_directory="checkpoint",
        checkpoint_manifest_sha256="6" * 64,
        checkpoint_manifest_file_sha256="a" * 64,
        checkpoint_model_sha256="1" * 64,
        checkpoint_context_sha256="b" * 64,
        selection_manifest_sha256="c" * 64,
        candidate_config_sha256="7" * 64,
        normalization_sha256="8" * 64,
        deep_artifact_id=artifact_id,
        deep_artifact_kind=DeepArtifactKind.CYCLEPATCH_DIRECT,
        deep_artifact_manifest_sha256="5" * 64,
    )
    payload = {
        "schema_version": "advanced-deployment-bundle-index-v2",
        "activation_status": "NOT_ACTIVATED",
        "created_at": _NOW.isoformat().replace("+00:00", "Z"),
        "source_commit": "d" * 40,
        "final_output_sha256": "e" * 64,
        "final_config_sha256": "f" * 64,
        "data_version": "matr-v1",
        "split_version": "split-v1",
        "feature_version": "feature-v1",
        "training_input_bundle_sha256": "0" * 64,
        "local_reconstructed_input_bundle_sha256": "0" * 64,
        "input_bundle_hashes_match": True,
        "promotion_manifest_sha256": "a" * 64,
        "promotion_decisions_sha256": "b" * 64,
        "promotion_source_evidence_sha256": "c" * 64,
        "selection_manifest_sha256": "c" * 64,
        "routes": [route.model_dump(mode="json")],
        "artifacts": [artifact.model_dump(mode="json")],
    }
    index = bundles.AdvancedDeploymentBundleIndex.model_validate(
        {**payload, "manifest_sha256": sha256_canonical(payload)}
    )

    assert index.schema_version == "advanced-deployment-bundle-index-v2"
    assert index.routes[0].output_target is AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert (
        index.artifacts[0].output_target
        is AdvancedOutputTarget.MATR_OFFICIAL_CYCLE_LIFE
    )
    feature_path = (
        tmp_path
        / "artifacts"
        / artifact_id
        / "feature_config.json"
    )
    feature_path.parent.mkdir(parents=True)
    feature_path.write_text(
        (
            '{"candidate_config_sha256":"'
            + "7" * 64
            + '","cutoff_cycle":20,"data_version":"matr-v1",'
            '"dataset_id":"MATR","feature_version":"feature-v1",'
            '"normalization_sha256":"'
            + "8" * 64
            + '","split_version":"split-v1"}'
        ),
        encoding="utf-8",
    )
    registration = AdvancedDeepModelArtifactCatalogSource._resolve_registered(
        SimpleNamespace(index=index, bundle_root=tmp_path),
        artifact_id,
    )
    assert registration.metadata.schema_version == "deep-model-artifact-v2"

    legacy_payload = {
        **payload,
        "schema_version": "advanced-deployment-bundle-index-v1",
        "routes": [
            route.model_dump(mode="json", exclude={"output_target"})
        ],
        "artifacts": [
            artifact.model_dump(mode="json", exclude={"output_target"})
        ],
    }
    legacy = bundles.AdvancedDeploymentBundleIndex.model_validate(
        {
            **legacy_payload,
            "manifest_sha256": sha256_canonical(legacy_payload),
        }
    )
    assert legacy.schema_version == "advanced-deployment-bundle-index-v1"
    assert legacy.routes[0].output_target is None
    assert legacy.artifacts[0].output_target is None
