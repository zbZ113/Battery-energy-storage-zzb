from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from quanxin_life.application.deep_model_artifacts import (
    CycleLifeReferenceLibraryConfig,
    CycleLifeTargetScalerConfig,
    DeepArtifactFileRole,
    DeepModelArtifactManifest,
    export_cpmlp_artifact,
    export_cyclepatch_batlinet_artifact,
    export_cyclepatch_direct_artifact,
    export_hybrid_artifact,
    export_hybridpatch_v2_artifact,
    load_cpmlp_artifact,
    load_cyclepatch_batlinet_artifact,
    load_cyclepatch_direct_artifact,
    load_hybrid_artifact,
    load_hybridpatch_v2_artifact,
)
from quanxin_life.core import LifePrediction, PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.curve_tensor import CurveTensor
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
    CyclePatchBatLiNet,
)
from quanxin_life.models.cpmlp import CPMLPLifePredictor
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchLifeRegressor,
    EarlyCycleBatch,
)
from quanxin_life.models.hybrid_degradation import (
    HybridDegradationPredictor,
    TrajectoryTrainingSample,
)
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Inputs,
    HybridPatchV2Predictor,
)

_CANDIDATE_SHA256 = "a" * 64
_NORMALIZATION_SHA256 = "b" * 64
_REFERENCE_SHA256 = "c" * 64


def _advanced_batch(cell_ids: tuple[str, ...]) -> EarlyCycleBatch:
    batch_size = len(cell_ids)
    cycle_count = 21
    values = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * cycle_count * 2 * 150 * 3,
        dtype=torch.float32,
    ).reshape(batch_size, cycle_count, 2, 150, 3)
    sample_mask = torch.ones((batch_size, cycle_count, 2, 150), dtype=torch.bool)
    return EarlyCycleBatch(
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        feature_version="multichannel-v1",
        normalization_statistics_sha256=_NORMALIZATION_SHA256,
        cell_ids=cell_ids,
        condition_names=("temperature_c",),
        values=values,
        cycle_indices=torch.arange(cycle_count).expand(batch_size, -1),
        cycle_mask=torch.ones((batch_size, cycle_count), dtype=torch.bool),
        sample_mask=sample_mask,
        condition_values=torch.full((batch_size, 1), 0.25, dtype=torch.float32),
        condition_mask=torch.ones((batch_size, 1), dtype=torch.bool),
    )


def _cyclepatch_config() -> CyclePatchConfig:
    return CyclePatchConfig(d_model=128, layers=2, heads=4, dropout=0.05)


def _target_scaler() -> CycleLifeTargetScaler:
    training_cells = ("train-a", "train-b", "train-c")
    split = SplitManifest(
        dataset_id="MATR",
        train=training_cells,
        validation=("validation",),
        calibration=("calibration",),
        test=("test",),
    )
    return CycleLifeTargetScaler.fit(
        {"train-a": 500.0, "train-b": 900.0, "train-c": 1300.0},
        training_cell_ids=training_cells,
        split_manifest=split,
        cutoff_cycle=20,
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )


def _reference_context() -> tuple[CycleLifeTargetScaler, CycleLifeReferenceLibrary]:
    training_cells = tuple(f"ref-{index}" for index in range(16))
    labels = {
        cell_id: 500.0 + index * 50.0
        for index, cell_id in enumerate(training_cells)
    }
    split = SplitManifest(
        dataset_id="MATR",
        train=training_cells,
        validation=("validation",),
        calibration=("calibration",),
        test=("test",),
    )
    scaler = CycleLifeTargetScaler.fit(
        labels,
        training_cell_ids=training_cells,
        split_manifest=split,
        cutoff_cycle=20,
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
    )
    library = CycleLifeReferenceLibrary.build(
        labels,
        training_cell_ids=training_cells,
        split_manifest=split,
        scaler=scaler,
        reference_count=16,
        seed=38,
    )
    return scaler, library


def _rewrite_registered_json(
    root: Path,
    manifest: DeepModelArtifactManifest,
    role: DeepArtifactFileRole,
    update: dict[str, object],
) -> DeepModelArtifactManifest:
    registered = next(item for item in manifest.files if item.role is role)
    path = root / registered.relative_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(update)
    path.write_text(json.dumps(payload), encoding="utf-8")
    files = tuple(
        item.model_copy(
            update={
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        if item.role is role
        else item
        for item in manifest.files
    )
    manifest_payload = {
        **manifest.model_dump(mode="json", exclude={"manifest_sha256", "files"}),
        "files": [item.model_dump(mode="json") for item in files],
    }
    modified = DeepModelArtifactManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    (root / manifest.artifact_id / "manifest.json").write_text(
        json.dumps(modified.model_dump(mode="json")),
        encoding="utf-8",
    )
    return modified


def _cpmlp_curve(cell_id: str, scale: float) -> CurveTensor:
    values = tuple(
        (0.6 * scale, 0.3 * scale, 0.0) if cycle == 1 else (None, None, None)
        for cycle in range(3)
    )
    return CurveTensor(
        dataset_id="SAFE-HUST",
        cell_id=cell_id,
        cutoff_cycle=2,
        feature_version="curve-v1",
        cycle_indices=(0, 1, 2),
        voltage_grid_v=(3.0, 3.1, 3.2),
        values=values,
        observed_mask=(False, True, False),
    )


def _fitted_cpmlp() -> CPMLPLifePredictor:
    predictor = CPMLPLifePredictor(
        model_version="cpmlp-v1",
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
        cutoff_cycle=2,
        curve_hidden_dim=4,
        aggregation_hidden_dim=4,
        epochs=1,
    )
    label = LifePrediction(
        dataset_id="SAFE-HUST",
        cell_id="train-cell",
        cutoff_cycle=2,
        predicted_eol_cycle=100,
        observed_eol_cycle=100,
        right_censored=False,
        feature_version="curve-v1",
        split_version="split-v1",
        model_version="label-v1",
        data_version="safe-hust-v1",
    )
    split = SplitManifest(
        dataset_id="SAFE-HUST",
        train=("train-cell",),
        validation=("validation-cell",),
        calibration=("calibration-cell",),
        test=("test-cell",),
    )
    return predictor.fit(
        [label],
        training_curves={"train-cell": _cpmlp_curve("train-cell", 0.9)},
        split_manifest=split,
    )


def _fitted_hybrid() -> HybridDegradationPredictor:
    predictor = HybridDegradationPredictor(
        model_version="hybrid-v1",
        feature_version="early-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
        cutoff_cycle=2,
        prediction_cycles=(3, 4, 5),
        condition_feature_names=("temperature_c",),
        hidden_dim=4,
        epochs=1,
    )
    split = SplitManifest(
        dataset_id="SAFE-HUST",
        train=("train-cell",),
        validation=("validation-cell",),
        calibration=("calibration-cell",),
        test=("test-cell",),
    )
    sample = TrajectoryTrainingSample(
        dataset_id="SAFE-HUST",
        cell_id="train-cell",
        cutoff_cycle=2,
        observed_cycles=(0, 1, 2),
        observed_soh=(1.0, 0.99, 0.98),
        target_cycles=(3, 4, 5),
        target_soh=(0.97, 0.96, 0.95),
        condition_features={"temperature_c": 25.0},
        feature_version="early-v1",
        data_version="safe-hust-v1",
    )
    return predictor.fit([sample], split_manifest=split)


def test_cpmlp_safetensors_round_trip_preserves_prediction(tmp_path: Path) -> None:
    predictor = _fitted_cpmlp()
    curve = _cpmlp_curve("test-cell", 1.0)
    expected = predictor.predict(
        curve=curve,
        cutoff_cycle=2,
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
    )

    manifest = export_cpmlp_artifact(
        predictor,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    loaded = load_cpmlp_artifact(tmp_path, manifest)
    actual = loaded.predict(
        curve=curve,
        cutoff_cycle=2,
        feature_version="curve-v1",
        split_version="split-v1",
        data_version="safe-hust-v1",
    )

    assert actual.predicted_eol_cycle == pytest.approx(expected.predicted_eol_cycle, abs=1e-7)


def test_hybrid_safetensors_round_trip_preserves_trajectory(tmp_path: Path) -> None:
    predictor = _fitted_hybrid()
    prediction_inputs = {
        "dataset_id": "SAFE-HUST",
        "cell_id": "test-cell",
        "observed_cycles": (0, 1, 2),
        "observed_soh": (1.0, 0.99, 0.98),
        "condition_features": {"temperature_c": 25.0},
        "cutoff_cycle": 2,
        "feature_version": "early-v1",
        "split_version": "split-v1",
        "data_version": "safe-hust-v1",
    }
    expected = predictor.predict(**prediction_inputs)

    manifest = export_hybrid_artifact(
        predictor,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    loaded = load_hybrid_artifact(tmp_path, manifest)
    actual = loaded.predict(**prediction_inputs)

    assert actual.predicted_soh == pytest.approx(expected.predicted_soh, abs=1e-7)


def test_deep_artifact_loader_rejects_changed_weights(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    weights = tmp_path / manifest.artifact_id / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match=r"SHA-256|size"):
        load_cpmlp_artifact(tmp_path, manifest)


def test_deep_artifact_loader_rejects_unknown_architecture(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    architecture = tmp_path / manifest.artifact_id / "architecture.json"
    payload = json.loads(architecture.read_text(encoding="utf-8"))
    payload["architecture"] = "unknown-network"
    architecture.write_text(json.dumps(payload), encoding="utf-8")

    files = tuple(
        item.model_copy(
            update={
                "size_bytes": architecture.stat().st_size,
                "sha256": hashlib.sha256(architecture.read_bytes()).hexdigest(),
            }
        )
        if item.role is DeepArtifactFileRole.ARCHITECTURE
        else item
        for item in manifest.files
    )
    manifest_payload = {
        **manifest.model_dump(mode="json", exclude={"manifest_sha256", "files"}),
        "files": [item.model_dump(mode="json") for item in files],
    }
    modified_manifest = DeepModelArtifactManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    (tmp_path / manifest.artifact_id / "manifest.json").write_text(
        json.dumps(modified_manifest.model_dump(mode="json")),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="architecture"):
        load_cpmlp_artifact(tmp_path, modified_manifest)


def test_deep_artifact_loader_rejects_unlisted_files(tmp_path: Path) -> None:
    manifest = export_cpmlp_artifact(
        _fitted_cpmlp(),
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    (tmp_path / manifest.artifact_id / "unlisted.pkl").write_bytes(b"forbidden")

    with pytest.raises(ValueError, match="unexpected file"):
        load_cpmlp_artifact(tmp_path, manifest)


def test_legacy_export_rejects_network_left_in_training_mode(tmp_path: Path) -> None:
    predictor = _fitted_cpmlp()
    assert predictor._network is not None
    predictor._network.train()

    with pytest.raises(ValueError, match="eval"):
        export_cpmlp_artifact(
            predictor,
            artifact_root=tmp_path,
            artifact_id=str(uuid4()),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )

    hybrid = _fitted_hybrid()
    assert hybrid._network is not None
    hybrid._network.train()
    with pytest.raises(ValueError, match="eval"):
        export_hybrid_artifact(
            hybrid,
            artifact_root=tmp_path,
            artifact_id=str(uuid4()),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
        )


def test_cyclepatch_direct_artifact_round_trip_preserves_prediction(
    tmp_path: Path,
) -> None:
    batch = _advanced_batch(("cell-1", "cell-2"))
    network = CyclePatchLifeRegressor(
        _cyclepatch_config(), condition_count=len(batch.condition_names)
    ).eval()
    target_scaler = _target_scaler()
    with torch.no_grad():
        expected = target_scaler.inverse_transform(network(batch))

    manifest = export_cyclepatch_direct_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version=batch.data_version,
        split_version="matr-cell-split-v1",
        feature_version=batch.feature_version,
        cutoff_cycle=20,
        condition_names=batch.condition_names,
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        target_scaler=target_scaler,
    )
    loaded = load_cyclepatch_direct_artifact(
        tmp_path,
        manifest,
        expected_normalization_sha256=_NORMALIZATION_SHA256,
        expected_candidate_config_sha256=_CANDIDATE_SHA256,
        expected_target_scaler_context_sha256=target_scaler.context_sha256,
    )

    with torch.no_grad():
        actual = loaded(batch)
    assert torch.equal(actual, expected)


def test_cyclepatch_batlinet_artifact_binds_reference_library(
    tmp_path: Path,
) -> None:
    target_batch = _advanced_batch(("target",))
    reference_batch = _advanced_batch(tuple(f"ref-{index}" for index in range(16)))
    network = CyclePatchBatLiNet(
        BatLiNetConfig(
            encoder=_cyclepatch_config(),
            lambda_pair=0.5,
            lambda_rank=0.1,
            fusion_alpha=0.5,
            reference_count=16,
        ),
        condition_count=1,
    ).eval()
    target_scaler, reference_library = _reference_context()
    reference_labels = torch.tensor(
        reference_library.standardized_labels,
        dtype=torch.float32,
    )
    with torch.no_grad():
        expected = target_scaler.inverse_transform(
            network.fuse_standardized(
                network.encode(target_batch),
                network.encode(reference_batch),
                reference_labels,
            )
        )

    manifest = export_cyclepatch_batlinet_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version=target_batch.data_version,
        split_version="matr-cell-split-v1",
        feature_version=target_batch.feature_version,
        cutoff_cycle=20,
        condition_names=target_batch.condition_names,
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        reference_library=reference_library,
        target_scaler=target_scaler,
    )
    with pytest.raises(ValueError, match="reference_library_sha256"):
        load_cyclepatch_batlinet_artifact(
            tmp_path,
            manifest,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_reference_library_sha256="d" * 64,
            expected_target_scaler_context_sha256=target_scaler.context_sha256,
        )
    loaded = load_cyclepatch_batlinet_artifact(
        tmp_path,
        manifest,
        expected_normalization_sha256=_NORMALIZATION_SHA256,
        expected_candidate_config_sha256=_CANDIDATE_SHA256,
        expected_reference_library_sha256=reference_library.library_sha256,
        expected_target_scaler_context_sha256=target_scaler.context_sha256,
    )
    with torch.no_grad():
        actual = loaded.predict_raw(target_batch, reference_batch)
    assert torch.equal(actual, expected)

    wrong_reference_batch = _advanced_batch(
        tuple(reversed(reference_library.cell_ids))
    )
    with pytest.raises(ValueError, match="cell_ids"):
        loaded.predict_raw(target_batch, wrong_reference_batch)
    with pytest.raises(ValueError, match="disabled"):
        loaded.fuse_standardized(
            loaded.encode(target_batch),
            loaded.encode(reference_batch),
            reference_labels,
        )


def test_batlinet_loader_rejects_tampered_self_contained_reference_library(
    tmp_path: Path,
) -> None:
    target_scaler, reference_library = _reference_context()
    network = CyclePatchBatLiNet(
        BatLiNetConfig(
            encoder=_cyclepatch_config(),
            lambda_pair=0.5,
            lambda_rank=0.1,
            fusion_alpha=0.5,
            reference_count=16,
        ),
        condition_count=1,
    ).eval()
    manifest = export_cyclepatch_batlinet_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        reference_library=reference_library,
        target_scaler=target_scaler,
    )
    changed = _rewrite_registered_json(
        tmp_path,
        manifest,
        DeepArtifactFileRole.REFERENCE_LIBRARY,
        {
            "standardized_labels": (
                reference_library.standardized_labels[0] + 1.0,
                *reference_library.standardized_labels[1:],
            )
        },
    )
    with pytest.raises(ValueError, match="reference library_sha256"):
        load_cyclepatch_batlinet_artifact(
            tmp_path,
            changed,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_reference_library_sha256=reference_library.library_sha256,
            expected_target_scaler_context_sha256=target_scaler.context_sha256,
        )


def test_batlinet_loader_rejects_self_consistent_counterfeit_reference_context(
    tmp_path: Path,
) -> None:
    target_scaler, reference_library = _reference_context()
    network = CyclePatchBatLiNet(
        BatLiNetConfig(
            encoder=_cyclepatch_config(),
            lambda_pair=0.5,
            lambda_rank=0.1,
            fusion_alpha=0.5,
            reference_count=16,
        ),
        condition_count=1,
    ).eval()
    manifest = export_cyclepatch_batlinet_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        reference_library=reference_library,
        target_scaler=target_scaler,
    )
    fake = CycleLifeReferenceLibraryConfig.from_runtime(reference_library).model_dump(
        mode="json"
    )
    fake["scaler_context_sha256"] = "d" * 64
    fake["library_sha256"] = sha256_canonical(
        {
            "schema_version": "cycle-life-reference-library-v1",
            **{
                key: value
                for key, value in fake.items()
                if key not in {"schema_version", "library_sha256"}
            },
        }
    )
    changed = _rewrite_registered_json(
        tmp_path,
        manifest,
        DeepArtifactFileRole.REFERENCE_LIBRARY,
        fake,
    )
    changed = _rewrite_registered_json(
        tmp_path,
        changed,
        DeepArtifactFileRole.FEATURE_CONFIG,
        {"reference_library_sha256": fake["library_sha256"]},
    )

    with pytest.raises(ValueError, match="scaler context"):
        load_cyclepatch_batlinet_artifact(
            tmp_path,
            changed,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_reference_library_sha256=str(fake["library_sha256"]),
            expected_target_scaler_context_sha256=target_scaler.context_sha256,
        )


def test_hybridpatch_v2_artifact_round_trip_preserves_monotone_trajectory(
    tmp_path: Path,
) -> None:
    batch = _advanced_batch(("cell-1", "cell-2"))
    config = HybridPatchV2Config(
        cyclepatch=_cyclepatch_config(),
        query_token_count=8,
        query_layers=1,
        decoder_hidden_dim=32,
        huber_delta=1.0,
        lambda_history=0.1,
        lambda_smooth=0.01,
        lambda_order=0.05,
        lambda_residual=0.01,
    )
    network = HybridPatchV2Predictor(config, condition_count=1).eval()
    inputs = HybridPatchV2Inputs(
        early_batch=batch,
        initial_soh=torch.tensor((0.99, 0.98), dtype=torch.float32),
        prediction_cycles=torch.tensor((50, 100, 300, 500), dtype=torch.int64),
    )
    with torch.no_grad():
        expected = network(inputs).predicted_soh

    manifest = export_hybridpatch_v2_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version=batch.data_version,
        split_version="matr-cell-split-v1",
        feature_version=batch.feature_version,
        cutoff_cycle=20,
        condition_names=batch.condition_names,
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
    )
    loaded = load_hybridpatch_v2_artifact(
        tmp_path,
        manifest,
        expected_normalization_sha256=_NORMALIZATION_SHA256,
        expected_candidate_config_sha256=_CANDIDATE_SHA256,
    )

    with torch.no_grad():
        actual = loaded(inputs).predicted_soh
    assert torch.equal(actual, expected)
    assert torch.all(actual[:, 1:] <= actual[:, :-1])


def test_advanced_export_rejects_network_left_in_training_mode(tmp_path: Path) -> None:
    network = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1)

    with pytest.raises(ValueError, match="eval"):
        export_cyclepatch_direct_artifact(
            network,
            artifact_root=tmp_path,
            artifact_id=str(uuid4()),
            created_at=datetime(2026, 7, 19, tzinfo=UTC),
            dataset_id="MATR",
            data_version="matr-three-batch-v1",
            split_version="matr-cell-split-v1",
            feature_version="multichannel-v1",
            cutoff_cycle=20,
            condition_names=("temperature_c",),
            normalization_sha256=_NORMALIZATION_SHA256,
            candidate_config_sha256=_CANDIDATE_SHA256,
            target_scaler=_target_scaler(),
        )


def test_cyclepatch_loader_rejects_unknown_registered_architecture(
    tmp_path: Path,
) -> None:
    network = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1).eval()
    manifest = export_cyclepatch_direct_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        target_scaler=_target_scaler(),
    )
    changed = _rewrite_registered_json(
        tmp_path,
        manifest,
        DeepArtifactFileRole.ARCHITECTURE,
        {"architecture": "unapproved-dynamic-class"},
    )

    with pytest.raises(ValueError, match="architecture"):
        load_cyclepatch_direct_artifact(
            tmp_path,
            changed,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_target_scaler_context_sha256=_target_scaler().context_sha256,
        )


def test_cyclepatch_loader_rejects_wrong_candidate_or_normalization_hash(
    tmp_path: Path,
) -> None:
    network = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1).eval()
    manifest = export_cyclepatch_direct_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        target_scaler=_target_scaler(),
    )

    with pytest.raises(ValueError, match="normalization_sha256"):
        load_cyclepatch_direct_artifact(
            tmp_path,
            manifest,
            expected_normalization_sha256="d" * 64,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_target_scaler_context_sha256=_target_scaler().context_sha256,
        )
    with pytest.raises(ValueError, match="candidate_config_sha256"):
        load_cyclepatch_direct_artifact(
            tmp_path,
            manifest,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256="e" * 64,
            expected_target_scaler_context_sha256=_target_scaler().context_sha256,
        )


def test_cyclepatch_loader_rejects_tampered_target_scaler_context(
    tmp_path: Path,
) -> None:
    target_scaler = _target_scaler()
    network = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1).eval()
    manifest = export_cyclepatch_direct_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        target_scaler=target_scaler,
    )
    changed = _rewrite_registered_json(
        tmp_path,
        manifest,
        DeepArtifactFileRole.FEATURE_CONFIG,
        {
            "target_scaler": {
                **CycleLifeTargetScalerConfig.from_runtime(target_scaler).model_dump(
                    mode="json"
                ),
                "mean": target_scaler.mean + 1.0,
            }
        },
    )

    with pytest.raises(ValueError, match="target scaler context_sha256"):
        load_cyclepatch_direct_artifact(
            tmp_path,
            changed,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_target_scaler_context_sha256=target_scaler.context_sha256,
        )
def test_hybridpatch_loader_rejects_registered_boundary_after_cycle_500(
    tmp_path: Path,
) -> None:
    network = HybridPatchV2Predictor(
        HybridPatchV2Config(
            cyclepatch=_cyclepatch_config(),
            query_token_count=0,
            query_layers=1,
            decoder_hidden_dim=32,
            lambda_history=0.0,
            lambda_smooth=0.0,
            lambda_order=0.0,
            lambda_residual=0.001,
        ),
        condition_count=1,
    ).eval()
    manifest = export_hybridpatch_v2_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
        dataset_id="MATR",
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        feature_version="multichannel-v1",
        cutoff_cycle=20,
        condition_names=("temperature_c",),
        normalization_sha256=_NORMALIZATION_SHA256,
        candidate_config_sha256=_CANDIDATE_SHA256,
    )
    changed = _rewrite_registered_json(
        tmp_path,
        manifest,
        DeepArtifactFileRole.FEATURE_CONFIG,
        {"max_prediction_cycle": 501},
    )

    with pytest.raises(ValueError, match="max_prediction_cycle"):
        load_hybridpatch_v2_artifact(
            tmp_path,
            changed,
            expected_normalization_sha256=_NORMALIZATION_SHA256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
        )
