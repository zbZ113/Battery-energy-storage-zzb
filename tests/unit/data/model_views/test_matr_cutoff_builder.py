from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _load_script():
    path = ROOT / "scripts" / "data" / "build_matr_cutoff_model_views.py"
    spec = importlib.util.spec_from_file_location("build_matr_cutoff_model_views", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cutoff model View builder is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cutoff_view_configs_cover_every_advanced_training_cutoff() -> None:
    module = _load_script()

    configs = module.load_cutoff_configs(ROOT)

    assert {
        (config.view_id, config.cutoff_cycle)
        for config in configs
    } == {
        ("early_life_sequence_c20", 20),
        ("soh_trajectory_c20", 20),
        ("early_life_sequence_c100", 100),
        ("soh_trajectory_c100", 100),
        ("early_life_sequence_c150", 150),
        ("soh_trajectory_c150", 150),
    }
