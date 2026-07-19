from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _module():
    path = Path(__file__).parents[3] / "scripts" / "prepare_advanced_matr_data.py"
    spec = importlib.util.spec_from_file_location("prepare_advanced_matr_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> SimpleNamespace:
    return SimpleNamespace(
        data_version="matr-v1",
        split_version="split-v1",
        combined_split_manifest="configs/data_splits/combined.json",
        combined_split_sha256="a" * 64,
        total_cell_count=140,
        scalar_label_count=138,
        hybrid_eligible_count=120,
    )


def _data(final: bool) -> SimpleNamespace:
    batches = {
        "scalar_train": SimpleNamespace(cell_ids=("s1", "s2")),
        "scalar_validation": SimpleNamespace(cell_ids=("s3",)),
        "hybrid_train": SimpleNamespace(cell_ids=("h1", "h2")),
        "hybrid_validation": SimpleNamespace(cell_ids=("h3",)),
        "scalar_normalizer": SimpleNamespace(
            statistics_sha256="1" * 64,
            training_cell_ids_sha256="2" * 64,
        ),
        "hybrid_normalizer": SimpleNamespace(
            statistics_sha256="3" * 64,
            training_cell_ids_sha256="4" * 64,
        ),
        "masked_cycle_audit": SimpleNamespace(
            masked_cycle_count=1,
            audit_sha256="5" * 64,
            entries=(SimpleNamespace(cell_id="h1", cycle_index=39, reason="bad_soh"),),
        ),
    }
    if final:
        batches.update(
            {
                "scalar_calibration": SimpleNamespace(cell_ids=("s4",)),
                "scalar_test": SimpleNamespace(cell_ids=("s5",)),
                "hybrid_calibration": SimpleNamespace(cell_ids=("h4",)),
                "hybrid_test": SimpleNamespace(cell_ids=("h5",)),
            }
        )
    return SimpleNamespace(**batches)


def test_plan_only_never_calls_data_loader(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_registry", lambda _root: (_manifest(), object()))
    called = False

    def fail_loader(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("plan-only must not open cell Parquet")

    monkeypatch.setattr(module, "load_advanced_matr_selection_data", fail_loader)
    report = module.prepare(project_root=tmp_path, mode="select", plan_only=True)

    assert report["status"] == "PLAN_READY"
    assert report["cutoffs"] == [20, 50, 100, 150]
    assert called is False
    assert "cutoffs_detail" in report and report["cutoffs_detail"] == []


def test_smoke_preflight_loads_only_cutoff_50_and_writes_atomic_report(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_registry", lambda _root: (_manifest(), object()))
    calls: list[int] = []
    monkeypatch.setattr(
        module,
        "load_advanced_matr_selection_data",
        lambda **kwargs: (calls.append(kwargs["cutoff_cycle"]) or _data(False)),
    )

    report = module.main(["smoke", "--project-root", str(tmp_path)])

    assert report == 0
    assert calls == [50]
    output = tmp_path / "reports" / "data_quality" / "advanced_matr_smoke_preflight.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["cutoffs_detail"][0]["counts"] == {
        "scalar": {"train": 2, "validation": 1},
        "hybrid": {"train": 2, "validation": 1},
    }
    assert payload["cutoffs_detail"][0]["masked_audit_sha256"] == "5" * 64
    assert len(payload["preflight_sha256"]) == 64


def test_final_preflight_includes_held_out_partitions(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_registry", lambda _root: (_manifest(), object()))
    calls: list[int] = []
    monkeypatch.setattr(
        module,
        "load_advanced_matr_final_data",
        lambda **kwargs: (calls.append(kwargs["cutoff_cycle"]) or _data(True)),
    )

    payload = module.prepare(project_root=tmp_path, mode="final")

    assert calls == [20, 50, 100, 150]
    assert payload["cutoffs_detail"][0]["counts"]["scalar"] == {
        "train": 2,
        "validation": 1,
        "calibration": 1,
        "test": 1,
    }
    assert payload["cutoffs_detail"][0]["counts"]["hybrid"]["test"] == 1
