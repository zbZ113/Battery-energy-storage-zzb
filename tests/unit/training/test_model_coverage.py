from __future__ import annotations

from pathlib import Path

from quanxin_life.core import TrainingMode
from quanxin_life.training.coverage import load_model_coverage


def test_required_model_families_are_exactly_covered() -> None:
    coverage = load_model_coverage(Path("configs/training/model_coverage_v1.json"))

    assert set(coverage.required_model_families) == {
        "cpmlp",
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "pbt",
        "diting_cptransformer",
        "current_hybrid",
        "hybridpatch_v2",
        "batterymformer",
        "magnet",
        "battgp",
        "blast_lite",
        "smart_feature",
    }
    assert set(coverage.required_dataset_ids) == {
        "MATR",
        "HUST",
        "NAUMANN_CYCLE",
        "NAUMANN_CALENDAR",
        "LFP_280AH_DOD",
        "LFP_280AH_TEMPEST",
        "LFP_180AH_FORKLIFT",
        "LFP_FIELD_160AH",
    }
    assert coverage.required_modes == (
        TrainingMode.SMOKE,
        TrainingMode.SELECT,
        TrainingMode.FINAL,
    )
    assert coverage.final_seeds == (38, 39, 40, 41, 42)
    assert coverage.early_life_cutoffs == (20, 50, 100, 150)
