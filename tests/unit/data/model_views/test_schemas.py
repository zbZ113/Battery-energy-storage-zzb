from __future__ import annotations

from pathlib import Path

import pytest

from quanxin_life.core import sha256_canonical
from quanxin_life.data.model_views.builder import build_model_view, verify_model_view
from quanxin_life.data.model_views.schemas import (
    ModelViewArtifact,
    ModelViewConfig,
    ModelViewManifest,
    ModelViewRow,
)


def test_manifest_binds_all_frozen_context_hashes() -> None:
    manifest = ModelViewManifest(
        view_id="early_life_sequence",
        view_version="v1",
        task_type="cycle_life",
        target_semantics="matr_official_cycle_life",
        mask_semantics=("missing_is_false",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256=sha256_canonical({"cell-1": "train"}),
        builder_version="builder-v1",
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
        normalization_sha256="e" * 64,
        training_entity_ids_sha256="f" * 64,
        row_count=4,
    )

    assert manifest.training_entity_ids_sha256 == "f" * 64


def _config() -> ModelViewConfig:
    return ModelViewConfig(
        view_id="early_life_sequence",
        view_version="v1",
        task_type="cycle_life",
        target_semantics="matr_official_cycle_life",
        entity_key="cell_id",
        feature_names=("x",),
        mask_semantics=("missing_is_false",),
        source_dataset_ids=("MATR",),
        cutoff_cycle=50,
    )


def _rows() -> tuple[ModelViewRow, ...]:
    return (
        ModelViewRow(
            entity_id="cell-1",
            split="train",
            features={"x": 1.0},
            feature_mask={"x": True},
            target=100.0,
        ),
    )


def test_second_identical_view_build_is_skipped(tmp_path: Path) -> None:
    output = tmp_path / "view"

    first = build_model_view(
        rows=_rows(),
        config=_config(),
        output_root=output,
        canonical_sha256="a" * 64,
        split_sha256=sha256_canonical({"cell-1": "train"}),
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    second = build_model_view(
        rows=_rows(),
        config=_config(),
        output_root=output,
        canonical_sha256="a" * 64,
        split_sha256=sha256_canonical({"cell-1": "train"}),
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
    )

    assert first.status == "BUILT"
    assert second.status == "SKIPPED_VALID"
    assert first.output_sha256 == second.output_sha256


def test_changed_view_context_requires_new_version(tmp_path: Path) -> None:
    output = tmp_path / "view"
    build_model_view(
        rows=_rows(),
        config=_config(),
        output_root=output,
        canonical_sha256="a" * 64,
        split_sha256=sha256_canonical({"cell-1": "train"}),
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
    )

    with pytest.raises(ValueError, match="new model view version"):
        build_model_view(
            rows=_rows(),
            config=_config(),
            output_root=output,
            canonical_sha256="e" * 64,
            split_sha256=sha256_canonical({"cell-1": "train"}),
            builder_code_sha256="c" * 64,
            config_sha256="d" * 64,
        )


def test_tampered_view_payload_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "view"
    build_model_view(
        rows=_rows(),
        config=_config(),
        output_root=output,
        canonical_sha256="a" * 64,
        split_sha256=sha256_canonical({"cell-1": "train"}),
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
    )
    (output / "view.json").write_text('{"rows":[]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="commit marker"):
        verify_model_view(output)


def test_split_sha_must_match_entity_assignments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="split SHA"):
        build_model_view(
            rows=_rows(),
            config=_config(),
            output_root=tmp_path / "view",
            canonical_sha256="a" * 64,
            split_sha256="b" * 64,
            builder_code_sha256="c" * 64,
            config_sha256="d" * 64,
        )


def test_row_features_must_match_registered_view_config(tmp_path: Path) -> None:
    config = _config().model_copy(update={"feature_names": ("x", "y")})

    with pytest.raises(ValueError, match="feature names"):
        build_model_view(
            rows=_rows(),
            config=config,
            output_root=tmp_path / "view",
            canonical_sha256="a" * 64,
            split_sha256=sha256_canonical({"cell-1": "train"}),
            builder_code_sha256="c" * 64,
            config_sha256="d" * 64,
        )


def test_missing_point_target_requires_explicit_evidence() -> None:
    with pytest.raises(ValueError, match="target evidence"):
        ModelViewRow(
            entity_id="condition-1",
            split="train",
            features={"x": 1.0},
            feature_mask={"x": True},
            target=None,
            evidence="OBSERVED",
        )


def test_tensor_manifest_binds_a_closed_artifact_inventory() -> None:
    manifest = ModelViewManifest(
        schema_version="model-view-manifest-v2",
        view_id="early_life_sequence",
        view_version="early-life-sequence-v1",
        task_type="cycle_life",
        target_semantics="matr_official_cycle_life",
        entity_key="cell_id",
        source_dataset_ids=("MATR",),
        mask_semantics=("explicit_tensor_masks",),
        cutoff_cycle=50,
        canonical_sha256="a" * 64,
        split_sha256="b" * 64,
        builder_version="builder-v2",
        builder_code_sha256="c" * 64,
        config_sha256="d" * 64,
        normalization_sha256="e" * 64,
        training_entity_ids_sha256="f" * 64,
        row_count=4,
        artifacts=(
            ModelViewArtifact(
                relative_path="metadata.json",
                size_bytes=10,
                sha256="1" * 64,
            ),
            ModelViewArtifact(
                relative_path="tensors.safetensors",
                size_bytes=20,
                sha256="2" * 64,
            ),
        ),
    )

    assert {artifact.relative_path for artifact in manifest.artifacts} == {
        "metadata.json",
        "tensors.safetensors",
    }
