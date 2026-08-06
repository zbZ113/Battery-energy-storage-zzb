from pathlib import Path

import quanxin_life.data.model_views.matr as matr_views
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest
from quanxin_life.data.model_views.builder import FrozenModelViewDataset
from quanxin_life.data.model_views.matr import load_verified_matr_view_data
from quanxin_life.data.schemas import SplitManifest


def test_frozen_dataset_reads_only_view_artifacts(tmp_path: Path, monkeypatch) -> None:
    view = tmp_path / "model_views" / "view.json"
    view.parent.mkdir(parents=True)
    view.write_text('{"rows": []}', encoding="utf-8")
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if "data\\raw" in str(path) or "data/raw" in str(path):
            raise AssertionError("training view must not open raw data")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    assert len(FrozenModelViewDataset(view)) == 0


def test_matr_view_loader_delegates_without_opening_raw(monkeypatch) -> None:
    root = Path(__file__).parents[2]
    manifest = MatrThreeBatchManifest.model_validate_json(
        (root / "reports/data_quality/matr_three_batch_manifest_v1.json").read_bytes()
    )
    split = SplitManifest.model_validate_json(
        (root / "configs/data_splits/matr_three_batch_cell_split_v1.json").read_bytes()
    )
    original_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        normalized = str(path).replace("\\", "/")
        if "/data/raw/" in normalized:
            raise AssertionError("MATR model view construction must not open raw data")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    sentinel = object()

    def processed_loader(**kwargs):
        assert kwargs["project_root"] == root
        assert kwargs["manifest"] == manifest
        assert kwargs["combined_split"] == split
        return sentinel

    monkeypatch.setattr(matr_views, "load_advanced_matr_final_data", processed_loader)

    data = load_verified_matr_view_data(
        project_root=root,
        manifest=manifest,
        combined_split=split,
        cutoff_cycle=50,
        feature_version="multichannel-cycle-v1",
    )

    assert data is sentinel
