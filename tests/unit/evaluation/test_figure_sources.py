import json

import pytest

from quanxin_life.evaluation.figure_sources import (
    FigureKind,
    materialize_figure_source,
)


def test_figure_source_writes_same_stem_csv_and_hashed_manifest(tmp_path) -> None:
    png = tmp_path / "loss_validation.png"
    png.write_bytes(b"png evidence")
    artifact = materialize_figure_source(
        png,
        kind=FigureKind.LOSS_VALIDATION_CURVE,
        rows=({"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.4}),
    )

    assert artifact.source_path == png.with_suffix(".csv")
    assert artifact.manifest_path == tmp_path / "loss_validation.manifest.json"
    payload = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert payload["source_sha256"] == artifact.source_sha256
    assert payload["manifest_sha256"] == artifact.manifest_sha256


def test_figure_source_rejects_missing_png_or_inconsistent_rows(tmp_path) -> None:
    with pytest.raises(ValueError, match="PNG"):
        materialize_figure_source(
            tmp_path / "missing.png",
            kind=FigureKind.BEST_EPOCH,
            rows=({"epoch": 1},),
        )
    png = tmp_path / "best_epoch.png"
    png.write_bytes(b"png")
    with pytest.raises(ValueError, match="columns"):
        materialize_figure_source(
            png,
            kind=FigureKind.BEST_EPOCH,
            rows=({"epoch": 1}, {"value": 2}),
        )

