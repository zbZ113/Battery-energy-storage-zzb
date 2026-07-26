from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from safetensors.torch import load_file

from quanxin_life.application import deep_model_artifacts as artifacts
from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.data.schemas import SplitManifest
from quanxin_life.features.early_cycle_sequence import (
    EarlyCycleNormalizer,
    EarlyCycleSequence,
)
from quanxin_life.models.batlinet import (
    BatLiNetConfig,
    CycleLifeReferenceLibrary,
    CycleLifeTargetScaler,
    CyclePatchBatLiNet,
)
from quanxin_life.models.cyclepatch import (
    CyclePatchConfig,
    CyclePatchLifeRegressor,
    EarlyCycleBatch,
    stack_early_cycle_sequences,
)
from quanxin_life.models.hybrid_degradation import _HybridNetwork
from quanxin_life.models.hybridpatch_v2 import (
    HybridPatchV2Config,
    HybridPatchV2Predictor,
)

_CANDIDATE_SHA256 = "a" * 64
_CREATED_AT = datetime(2026, 7, 26, tzinfo=UTC)
_DATA_VERSION = "matr-three-batch-v1"
_FEATURE_VERSION = "cyclepatch-multichannel-v1"
_SPLIT_VERSION = "matr-cell-split-v1"


def _raw_sequence(cell_id: str, offset: float) -> EarlyCycleSequence:
    cycle_count = 21
    values = torch.full(
        (cycle_count, 2, 150, 3),
        0.1 + offset,
        dtype=torch.float32,
    )
    sample_mask = torch.ones((cycle_count, 2, 150), dtype=torch.bool)
    return EarlyCycleSequence(
        dataset_id="MATR",
        cell_id=cell_id,
        cutoff_cycle=20,
        data_version=_DATA_VERSION,
        feature_version=_FEATURE_VERSION,
        cycle_indices=tuple(range(cycle_count)),
        values=values,
        cycle_mask=torch.ones(cycle_count, dtype=torch.bool),
        sample_mask=sample_mask,
        condition_names=("mean_temperature_c",),
        condition_values=torch.tensor([25.0 + offset], dtype=torch.float32),
        condition_mask=torch.ones(1, dtype=torch.bool),
    )


def _normalizer_and_batches() -> tuple[
    EarlyCycleNormalizer,
    EarlyCycleBatch,
    EarlyCycleBatch,
]:
    reference_sequences = tuple(
        _raw_sequence(f"ref-{index}", float(index) / 100.0) for index in range(16)
    )
    normalizer = EarlyCycleNormalizer.fit(
        reference_sequences,
        training_cell_ids=frozenset(sequence.cell_id for sequence in reference_sequences),
    )
    reference_batch = stack_early_cycle_sequences(
        tuple(normalizer.transform(sequence) for sequence in reference_sequences)
    )
    target_batch = stack_early_cycle_sequences(
        (normalizer.transform(_raw_sequence("target", 0.25)),)
    )
    return normalizer, reference_batch, target_batch


def _cyclepatch_config() -> CyclePatchConfig:
    return CyclePatchConfig(d_model=128, layers=2, heads=4, dropout=0.05)


def _reference_context() -> tuple[CycleLifeTargetScaler, CycleLifeReferenceLibrary]:
    training_cells = tuple(f"ref-{index}" for index in range(16))
    labels = {cell_id: 500.0 + index * 50.0 for index, cell_id in enumerate(training_cells)}
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


def _read_registered_json(
    root: Path,
    manifest: artifacts.DeepModelArtifactManifest,
    role: object,
) -> dict[str, object]:
    registered = next(item for item in manifest.files if item.role is role)
    payload = json.loads((root / registered.relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _assert_complete_normalizer(
    root: Path,
    manifest: artifacts.DeepModelArtifactManifest,
    normalizer: EarlyCycleNormalizer,
) -> None:
    inference_role = artifacts.DeepArtifactFileRole["INFERENCE_CONTEXT"]
    context = _read_registered_json(root, manifest, inference_role)
    assert context["schema_version"] == "advanced-inference-context-v1"
    assert context["inference_context_sha256"] == sha256_canonical(
        {key: value for key, value in context.items() if key != "inference_context_sha256"}
    )
    encoded = context["normalizer"]
    assert isinstance(encoded, dict)
    assert encoded["schema_version"] == "early-cycle-normalizer-artifact-v1"
    expected = json.loads(json.dumps(asdict(normalizer)))
    for field_name, expected_value in expected.items():
        assert encoded[field_name] == expected_value
    restored = artifacts.EarlyCycleNormalizerArtifactConfig.model_validate(
        encoded
    ).to_runtime()
    assert asdict(restored) == asdict(normalizer)
    feature_role = artifacts.DeepArtifactFileRole.FEATURE_CONFIG
    feature = _read_registered_json(root, manifest, feature_role)
    assert feature["inference_context_sha256"] == context["inference_context_sha256"]


def test_every_advanced_v2_artifact_contains_the_complete_fitted_normalizer(
    tmp_path: Path,
) -> None:
    normalizer, reference_batch, _target_batch = _normalizer_and_batches()
    scaler, reference_library = _reference_context()
    direct = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1).eval()
    batlinet = CyclePatchBatLiNet(
        BatLiNetConfig(
            encoder=_cyclepatch_config(),
            lambda_pair=0.5,
            lambda_rank=0.1,
            fusion_alpha=0.5,
            reference_count=16,
        ),
        condition_count=1,
    ).eval()
    hybridpatch = HybridPatchV2Predictor(
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
    ).eval()
    prediction_cycles = (21, 100, 500)
    current_hybrid = _HybridNetwork(
        input_dim=3,
        horizon=len(prediction_cycles),
        hidden_dim=32,
    ).eval()

    manifests = (
        (
            artifacts.export_cyclepatch_direct_artifact(
                direct,
                artifact_root=tmp_path,
                artifact_id=str(uuid4()),
                created_at=_CREATED_AT,
                dataset_id="MATR",
                data_version=_DATA_VERSION,
                split_version=_SPLIT_VERSION,
                feature_version=_FEATURE_VERSION,
                cutoff_cycle=20,
                condition_names=("mean_temperature_c",),
                normalization_sha256=normalizer.statistics_sha256,
                candidate_config_sha256=_CANDIDATE_SHA256,
                target_scaler=scaler,
                normalizer=normalizer,
                output_target="matr_official_cycle_life",
            ),
            "matr_official_cycle_life",
        ),
        (
            artifacts.export_cyclepatch_batlinet_artifact(
                batlinet,
                artifact_root=tmp_path,
                artifact_id=str(uuid4()),
                created_at=_CREATED_AT,
                dataset_id="MATR",
                data_version=_DATA_VERSION,
                split_version=_SPLIT_VERSION,
                feature_version=_FEATURE_VERSION,
                cutoff_cycle=20,
                condition_names=("mean_temperature_c",),
                normalization_sha256=normalizer.statistics_sha256,
                candidate_config_sha256=_CANDIDATE_SHA256,
                reference_library=reference_library,
                reference_batch=reference_batch,
                target_scaler=scaler,
                normalizer=normalizer,
                output_target="matr_official_cycle_life",
            ),
            "matr_official_cycle_life",
        ),
        (
            artifacts.export_hybridpatch_v2_artifact(
                hybridpatch,
                artifact_root=tmp_path,
                artifact_id=str(uuid4()),
                created_at=_CREATED_AT,
                dataset_id="MATR",
                data_version=_DATA_VERSION,
                split_version=_SPLIT_VERSION,
                feature_version=_FEATURE_VERSION,
                cutoff_cycle=20,
                condition_names=("mean_temperature_c",),
                normalization_sha256=normalizer.statistics_sha256,
                candidate_config_sha256=_CANDIDATE_SHA256,
                normalizer=normalizer,
                output_target="soh_trajectory",
            ),
            "soh_trajectory",
        ),
        (
            artifacts.export_current_hybrid_artifact(
                current_hybrid,
                artifact_root=tmp_path,
                artifact_id=str(uuid4()),
                created_at=_CREATED_AT,
                dataset_id="MATR",
                data_version=_DATA_VERSION,
                split_version=_SPLIT_VERSION,
                feature_version=_FEATURE_VERSION,
                cutoff_cycle=20,
                condition_names=("mean_temperature_c",),
                prediction_cycles=prediction_cycles,
                variable_names=("voltage_v", "current_a", "capacity_ah"),
                aggregation_version="masked-variable-mean-v1",
                normalization_sha256=normalizer.statistics_sha256,
                candidate_config_sha256=_CANDIDATE_SHA256,
                normalizer=normalizer,
                output_target="soh_trajectory",
            ),
            "soh_trajectory",
        ),
    )

    assert all(
        manifest.schema_version == "deep-model-artifact-v2"
        for manifest, _expected_target in manifests
    )
    for manifest, expected_target in manifests:
        assert manifest.output_target == expected_target
        _assert_complete_normalizer(tmp_path, manifest, normalizer)


def test_advanced_v2_exporters_reject_family_output_target_mismatches(
    tmp_path: Path,
) -> None:
    normalizer, _reference_batch, _target_batch = _normalizer_and_batches()
    scaler, _reference_library = _reference_context()
    direct = CyclePatchLifeRegressor(_cyclepatch_config(), condition_count=1).eval()
    hybridpatch = HybridPatchV2Predictor(
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
    ).eval()

    with pytest.raises(ValueError, match=r"output_target|family|artifact kind"):
        artifacts.export_cyclepatch_direct_artifact(
            direct,
            artifact_root=tmp_path,
            artifact_id=str(uuid4()),
            created_at=_CREATED_AT,
            dataset_id="MATR",
            data_version=_DATA_VERSION,
            split_version=_SPLIT_VERSION,
            feature_version=_FEATURE_VERSION,
            cutoff_cycle=20,
            condition_names=("mean_temperature_c",),
            normalization_sha256=normalizer.statistics_sha256,
            candidate_config_sha256=_CANDIDATE_SHA256,
            target_scaler=scaler,
            normalizer=normalizer,
            output_target="soh_trajectory",
        )

    with pytest.raises(ValueError, match=r"output_target|family|artifact kind"):
        artifacts.export_hybridpatch_v2_artifact(
            hybridpatch,
            artifact_root=tmp_path,
            artifact_id=str(uuid4()),
            created_at=_CREATED_AT,
            dataset_id="MATR",
            data_version=_DATA_VERSION,
            split_version=_SPLIT_VERSION,
            feature_version=_FEATURE_VERSION,
            cutoff_cycle=20,
            condition_names=("mean_temperature_c",),
            normalization_sha256=normalizer.statistics_sha256,
            candidate_config_sha256=_CANDIDATE_SHA256,
            normalizer=normalizer,
            output_target="matr_official_cycle_life",
        )


def test_batlinet_v2_is_self_contained_and_hash_binds_its_reference_batch(
    tmp_path: Path,
) -> None:
    normalizer, reference_batch, _target_batch = _normalizer_and_batches()
    scaler, reference_library = _reference_context()
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
    manifest = artifacts.export_cyclepatch_batlinet_artifact(
        network,
        artifact_root=tmp_path,
        artifact_id=str(uuid4()),
        created_at=_CREATED_AT,
        dataset_id="MATR",
        data_version=_DATA_VERSION,
        split_version=_SPLIT_VERSION,
        feature_version=_FEATURE_VERSION,
        cutoff_cycle=20,
        condition_names=("mean_temperature_c",),
        normalization_sha256=normalizer.statistics_sha256,
        candidate_config_sha256=_CANDIDATE_SHA256,
        reference_library=reference_library,
        reference_batch=reference_batch,
        target_scaler=scaler,
        normalizer=normalizer,
        output_target="matr_official_cycle_life",
    )
    assert manifest.output_target == "matr_official_cycle_life"
    reference_role = artifacts.DeepArtifactFileRole["REFERENCE_BATCH"]
    registered = next(item for item in manifest.files if item.role is reference_role)
    reference_path = tmp_path / registered.relative_path
    assert reference_path.name == "reference_batch.safetensors"
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == registered.sha256
    tensors = load_file(str(reference_path), device="cpu")
    assert set(tensors) == {
        "condition_mask",
        "condition_values",
        "cycle_indices",
        "cycle_mask",
        "sample_mask",
        "values",
    }
    restored_reference = EarlyCycleBatch(
        dataset_id=reference_batch.dataset_id,
        data_version=reference_batch.data_version,
        feature_version=reference_batch.feature_version,
        normalization_statistics_sha256=(reference_batch.normalization_statistics_sha256),
        cell_ids=reference_batch.cell_ids,
        condition_names=reference_batch.condition_names,
        values=tensors["values"],
        cycle_indices=tensors["cycle_indices"],
        cycle_mask=tensors["cycle_mask"],
        sample_mask=tensors["sample_mask"],
        condition_values=tensors["condition_values"],
        condition_mask=tensors["condition_mask"],
    )
    for field_name in (
        "values",
        "cycle_indices",
        "cycle_mask",
        "sample_mask",
        "condition_values",
        "condition_mask",
    ):
        assert torch.equal(
            getattr(restored_reference, field_name),
            getattr(reference_batch, field_name),
        )

    context = _read_registered_json(
        tmp_path,
        manifest,
        artifacts.DeepArtifactFileRole["INFERENCE_CONTEXT"],
    )
    metadata = context["reference_batch"]
    assert isinstance(metadata, dict)
    assert metadata == {
        "schema_version": "batlinet-reference-batch-v1",
        "dataset_id": reference_batch.dataset_id,
        "data_version": reference_batch.data_version,
        "feature_version": reference_batch.feature_version,
        "normalization_statistics_sha256": (reference_batch.normalization_statistics_sha256),
        "cell_ids": list(reference_library.cell_ids),
        "condition_names": list(reference_batch.condition_names),
        "reference_library_sha256": reference_library.library_sha256,
        "tensor_file_sha256": registered.sha256,
    }

    loaded = artifacts.load_cyclepatch_batlinet_artifact(
        tmp_path,
        manifest,
        expected_normalization_sha256=normalizer.statistics_sha256,
        expected_candidate_config_sha256=_CANDIDATE_SHA256,
        expected_reference_library_sha256=reference_library.library_sha256,
        expected_target_scaler_context_sha256=scaler.context_sha256,
    )
    assert loaded.reference_batch.cell_ids == reference_library.cell_ids
    assert loaded.reference_batch.normalization_statistics_sha256 == (normalizer.statistics_sha256)

    del tensors
    del restored_reference
    del loaded
    payload = bytearray(reference_path.read_bytes())
    payload[-1] ^= 1
    reference_path.write_bytes(payload)
    with pytest.raises(ValueError, match=r"reference batch|SHA-256"):
        artifacts.load_cyclepatch_batlinet_artifact(
            tmp_path,
            manifest,
            expected_normalization_sha256=normalizer.statistics_sha256,
            expected_candidate_config_sha256=_CANDIDATE_SHA256,
            expected_reference_library_sha256=reference_library.library_sha256,
            expected_target_scaler_context_sha256=scaler.context_sha256,
        )


def test_legacy_v1_manifest_remains_parseable_but_is_not_runtime_self_contained(
    tmp_path: Path,
) -> None:
    artifact_id = str(uuid4())
    files = (
        artifacts.DeepArtifactFile(
            role=artifacts.DeepArtifactFileRole.WEIGHTS,
            relative_path=f"{artifact_id}/model.safetensors",
            size_bytes=1,
            sha256="1" * 64,
        ),
        artifacts.DeepArtifactFile(
            role=artifacts.DeepArtifactFileRole.ARCHITECTURE,
            relative_path=f"{artifact_id}/architecture.json",
            size_bytes=1,
            sha256="2" * 64,
        ),
        artifacts.DeepArtifactFile(
            role=artifacts.DeepArtifactFileRole.FEATURE_CONFIG,
            relative_path=f"{artifact_id}/feature_config.json",
            size_bytes=1,
            sha256="3" * 64,
        ),
    )
    unsigned = {
        "schema_version": "deep-model-artifact-v1",
        "artifact_id": artifact_id,
        "artifact_kind": "cyclepatch-direct-official-cycle-life",
        "files": [item.model_dump(mode="json") for item in files],
        "created_at": _CREATED_AT.isoformat().replace("+00:00", "Z"),
    }
    legacy = artifacts.DeepModelArtifactManifest.model_validate(
        {**unsigned, "manifest_sha256": sha256_canonical(unsigned)}
    )
    assert legacy.schema_version == "deep-model-artifact-v1"
    assert getattr(legacy, "output_target", None) is None

    with pytest.raises(ValueError, match=r"self-contained|v2|inference context"):
        artifacts.verify_self_contained_advanced_artifact(tmp_path, legacy)
