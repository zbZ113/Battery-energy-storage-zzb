from __future__ import annotations

import hashlib
import json
from pathlib import Path

from quanxin_life.core import DatasetBuildStatus
from quanxin_life.experiments.blast_validation_data import (
    BLAST_VALIDATION_PROCESSOR_VERSION,
    prepare_blast_validation_bundle,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_prepare_blast_validation_bundle_is_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    raw_cycle = (
        REPOSITORY_ROOT
        / "data"
        / "raw"
        / "NAUMANN_CYCLE"
        / "v1"
        / "xDOD_1C1C_40°C_Capacity_CC_CV_FEC.mat"
    )
    before = _sha256(raw_cycle)
    output_dir = tmp_path / "blast-validation-v1"

    first = prepare_blast_validation_bundle(REPOSITORY_ROOT, output_dir=output_dir)
    second = prepare_blast_validation_bundle(REPOSITORY_ROOT, output_dir=output_dir)

    assert first.status is DatasetBuildStatus.BUILT
    assert second.status is DatasetBuildStatus.SKIPPED_VALID
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.calendar_observation_count == 595
    assert first.cycle_observation_count == 665
    assert _sha256(raw_cycle) == before
    assert (output_dir / "COMMITTED").read_text(encoding="ascii").strip() == (
        first.bundle_sha256
    )

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    observations = json.loads(
        (output_dir / "observations.json").read_text(encoding="utf-8")
    )
    assert manifest["processor_version"] == BLAST_VALIDATION_PROCESSOR_VERSION
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["observation_counts"] == {"calendar": 595, "cycle": 665}
    assert manifest["excluded_conditions"] == [
        {
            "condition_id": "T25_SOC50_DOD100_C1_C1_CC",
            "reason": "SOURCE_COLUMN_CONTAINS_NO_FINITE_OBSERVATIONS",
            "source_file": "data/raw/NAUMANN_CYCLE/v1/xDOD_1C1C_x°C_Capacity_CC_CV_FEC.mat",
        }
    ]
    assert len(manifest["input_files"]) == 5
    assert len(observations["calendar"]) == 595
    assert len(observations["cycle"]) == 665
    assert manifest["observations_sha256"] == _sha256(output_dir / "observations.json")
