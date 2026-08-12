from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from quanxin_life.core import DatasetBuildStatus
from quanxin_life.data.adapters.lfp_280ah_dod import (
    load_dod_capacity_dataset_layout,
)
from quanxin_life.data.manifest import load_dataset_file_audit_manifest
from quanxin_life.experiments.blast_280ah_data import (
    BLAST_280AH_PROCESSOR_VERSION,
    prepare_blast_280ah_validation_bundle,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

HEADER = (
    "数据序号,循环号,工步号,工步类型,时间,总时间,电流(A),电压(V),容量(Ah),"
    "充电容量(Ah),放电容量(Ah),能量(Wh),充电能量(Wh),放电能量(Wh),"
    "绝对时间,功率(W),temp1_1,\n"
)


def _write_fixture_project(root: Path) -> Path:
    member = "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip"
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "cell-0.5C-100%DOD-1.csv",
            (
                HEADER
                + "1,1,1,放电,00:00:00,00:00:00,-140,3.2,0,0,282,0,0,0,"
                "2024-01-01 00:00:00,0,25,\n"
                + "2,2,1,放电,00:00:00,05:00:00,-140,3.2,0,0,280,0,0,0,"
                "2024-01-01 05:00:00,0,25,\n"
            ).encode("gb18030"),
        )
    outer = root / "data/raw/LFP_280AH_DOD/v3/CATL.zip"
    outer.parent.mkdir(parents=True)
    with zipfile.ZipFile(outer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, nested.getvalue())
    outer_sha = hashlib.sha256(outer.read_bytes()).hexdigest()

    readme = root / "data/raw/LFP_280AH_DOD/v3/Readme.txt"
    readme.write_text("Reviewed 280Ah fixture.\n", encoding="utf-8")
    readme_sha = hashlib.sha256(readme.read_bytes()).hexdigest()
    catalog = [
        {
            "dataset_id": "LFP_280AH_DOD",
            "version": "Zenodo-14576042-v3",
            "source_uri": "https://zenodo.org/records/14576042",
            "paper_uri": "https://doi.org/10.5281/zenodo.14576042",
            "license_status": "CC BY 4.0",
            "ingestion_mode": "archive_tabular",
            "expected_suffixes": [".zip", ".txt"],
            "downloaded_at": "2026-08-04T03:20:08Z",
            "artifact_paths": [
                "data/raw/LFP_280AH_DOD/v3/CATL.zip",
                "data/raw/LFP_280AH_DOD/v3/Readme.txt",
            ],
            "artifact_sha256": [outer_sha, readme_sha],
        }
    ]
    configs = root / "configs"
    (configs / "data_manifests").mkdir(parents=True)
    (configs / "data_layouts").mkdir(parents=True)
    (configs / "data_sources.json").write_text(json.dumps(catalog), encoding="utf-8")
    audit = {
        "manifest_version": "lfp-280ah-dod-public-files-v1",
        "source_catalog": "configs/data_sources.json",
        "file_count": 2,
        "total_size_bytes": outer.stat().st_size + readme.stat().st_size,
        "files": [
            {
                "dataset_id": "LFP_280AH_DOD",
                "relative_path": "data/raw/LFP_280AH_DOD/v3/CATL.zip",
                "size_bytes": outer.stat().st_size,
                "sha256": outer_sha,
            },
            {
                "dataset_id": "LFP_280AH_DOD",
                "relative_path": "data/raw/LFP_280AH_DOD/v3/Readme.txt",
                "size_bytes": readme.stat().st_size,
                "sha256": readme_sha,
            },
        ],
    }
    (configs / "data_manifests/lfp_280ah_dod_v1.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )
    layout = {
        "schema_version": "lfp-280ah-dod-capacity-layout-v1",
        "layout_version": "fixture-v1",
        "nominal_capacity_ah": 280.0,
        "selected_dod_fraction": 1.0,
        "c_rate": 0.5,
        "csv_encoding": "gb18030",
        "normalization": "FIRST_VALID_CYCLE_CAPACITY",
        "columns": {
            "cycle_number": "循环号",
            "discharge_capacity_ah": "放电容量(Ah)",
            "absolute_time": "绝对时间",
            "temperature_c": "temp1_1",
        },
        "archives": [
            {
                "outer_relative_path": "data/raw/LFP_280AH_DOD/v3/CATL.zip",
                "archive": {
                    "layout_version": "fixture-v1",
                    "vendor": "CATL",
                    "archive_sha256": outer_sha,
                    "expected_members": [member],
                    "task_role": "external_test",
                },
                "selected_members": [member],
            }
        ],
    }
    (configs / "data_layouts/lfp_280ah_dod_capacity_v1.json").write_text(
        json.dumps(layout, ensure_ascii=False), encoding="utf-8"
    )
    return outer


def test_prepare_280ah_bundle_is_source_verified_and_idempotent(tmp_path: Path) -> None:
    outer = _write_fixture_project(tmp_path)
    before = hashlib.sha256(outer.read_bytes()).hexdigest()
    output = tmp_path / "data/processed/LFP_280AH_DOD/test/capacity-v1"

    first = prepare_blast_280ah_validation_bundle(tmp_path, output_dir=output)
    second = prepare_blast_280ah_validation_bundle(tmp_path, output_dir=output)

    assert first.status is DatasetBuildStatus.BUILT
    assert second.status is DatasetBuildStatus.SKIPPED_VALID
    assert first.bundle_sha256 == second.bundle_sha256
    assert first.cell_count == 1
    assert first.observation_count == 2
    assert hashlib.sha256(outer.read_bytes()).hexdigest() == before
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["processor_version"] == BLAST_280AH_PROCESSOR_VERSION
    assert manifest["license"] == "CC BY 4.0"
    assert manifest["selection"]["dod_fraction"] == 1.0
    assert manifest["selection"]["cell_ids"] == ["CATL-2763"]
    assert (output / "capacity_trajectories.csv").is_file()
    assert (output / "series.json").is_file()
    assert (output / "COMMITTED").is_file()


def test_checked_in_280ah_governance_selects_all_reviewed_100_percent_cells() -> None:
    audit = load_dataset_file_audit_manifest(
        REPOSITORY_ROOT / "configs/data_manifests/lfp_280ah_dod_v1.json"
    )
    layout = load_dod_capacity_dataset_layout(
        REPOSITORY_ROOT / "configs/data_layouts/lfp_280ah_dod_capacity_v1.json"
    )

    assert audit.file_count == 3
    assert audit.total_size_bytes == 5_174_692_220
    assert {item.relative_path for item in audit.files} == {
        "data/raw/LFP_280AH_DOD/v3/CATL.zip",
        "data/raw/LFP_280AH_DOD/v3/EVE.zip",
        "data/raw/LFP_280AH_DOD/v3/Readme.txt",
    }
    assert layout.layout_version == "lfp-280ah-dod-capacity-zenodo-14576042-v1"
    assert {
        member
        for selection in layout.archives
        for member in selection.selected_members
    } == {
        "CATL/CATL-0.5C-100% Depth of Discharge/2763.zip",
        "CATL/CATL-0.5C-100% Depth of Discharge/4296.zip",
        "CATL/CATL-0.5C-100% Depth of Discharge/4870.zip",
        "EVE/EVE-0.5C-100% Depth of Discharge/1092.zip",
        "EVE/EVE-0.5C-100% Depth of Discharge/1859.zip",
        "EVE/EVE-0.5C-100% Depth of Discharge/5726.zip",
    }


def test_280ah_cli_builds_bundle_and_observed_range_validation(tmp_path: Path) -> None:
    _write_fixture_project(tmp_path)
    bundle = tmp_path / "data/processed/LFP_280AH_DOD/test/capacity-v1"
    result = tmp_path / "reports/experiments/blast_280ah_v1/test-v1"

    prepared = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/data/prepare_blast_280ah_validation.py"),
            "--repository-root",
            str(tmp_path),
            "--output-dir",
            str(bundle),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert prepared.returncode == 0, prepared.stderr
    assert json.loads(prepared.stdout)["status"] == "BUILT"

    validated = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts/run_blast_280ah_validation.py"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--bundle-dir",
            str(bundle),
            "--output-dir",
            str(result),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert validated.returncode == 0, validated.stderr
    payload = json.loads(validated.stdout)
    assert payload["status"] == "BUILT"
    assert payload["prediction_count"] == 2
    assert (result / "predictions.csv").is_file()
    assert (result / "metrics.csv").is_file()
    assert (result / "COMMITTED").is_file()
