from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_source(root: Path, *, model_name: str = "model.safetensors") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / model_name).write_bytes(b"safe-model")
    for name, payload in {
        "model_config.json": {"family": "demo"},
        "input_schema.json": {"fields": ["time"]},
        "preprocessing_manifest.json": {"sha256": "a" * 64},
        "normalization.json": {"version": "train-only-v1"},
        "calibration.json": {"interval_method": "NOT_AVAILABLE"},
        "supported_domain.json": {"dataset": "MATR"},
        "metrics_validation.json": {"status": "verified"},
        "metrics_test.json": {"status": "verified"},
        "inference_benchmark.json": {"status": "NOT_MEASURED"},
    }.items():
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "per_cell_summary.parquet").write_bytes(b"PAR1fixturePAR1")
    (root / "model_card.md").write_text("# Demo\n", encoding="utf-8")


def test_export_uses_best_only_and_is_verified_not_activated(tmp_path: Path) -> None:
    from quanxin_life.training.promotion_bundle import export_promoted_model_bundle

    source = tmp_path / "best"
    _write_source(source)
    (source / "optimizer.safetensors").write_bytes(b"resume-only")
    destination = tmp_path / "bundle"

    manifest = export_promoted_model_bundle(
        source,
        destination,
        task="RUL",
        family="demo",
        version="v1",
        source_commit="a" * 40,
        data_version="data-v1",
        split_version="cell-split-v1",
    )

    assert manifest.activation_status == "VERIFIED_NOT_ACTIVATED"
    assert not (destination / "optimizer.safetensors").exists()
    assert (destination / "artifact_manifest.json").is_file()


def test_export_rejects_last_checkpoint_and_unsafe_formats(tmp_path: Path) -> None:
    from quanxin_life.training.promotion_bundle import export_promoted_model_bundle

    source = tmp_path / "last"
    _write_source(source)
    (source / "model.pt").write_bytes(b"unsafe")

    with pytest.raises(ValueError, match=r"best|forbidden"):
        export_promoted_model_bundle(
            source,
            tmp_path / "bundle",
            task="RUL",
            family="demo",
            version="v1",
            source_commit="a" * 40,
            data_version="data-v1",
            split_version="cell-split-v1",
        )


def test_export_rejects_missing_model_sha_or_required_evidence(tmp_path: Path) -> None:
    from quanxin_life.training.promotion_bundle import export_promoted_model_bundle

    source = tmp_path / "best"
    _write_source(source)
    (source / "metrics_test.json").unlink()

    with pytest.raises(ValueError, match="required"):
        export_promoted_model_bundle(
            source,
            tmp_path / "bundle",
            task="RUL",
            family="demo",
            version="v1",
            source_commit="a" * 40,
            data_version="data-v1",
            split_version="cell-split-v1",
        )
