"""Prepare leakage-governed advanced MATR sequence caches and preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.core.hashing import sha256_canonical  # noqa: E402
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest  # noqa: E402
from quanxin_life.data.schemas import SplitManifest  # noqa: E402
from quanxin_life.training.advanced_data import (  # noqa: E402
    AdvancedFinalMatrData,
    AdvancedSelectionMatrData,
    load_advanced_matr_final_data,
    load_advanced_matr_selection_data,
)

FEATURE_VERSION = "cyclepatch-multichannel-v1"
MANIFEST_RELATIVE = Path("reports/data_quality/matr_three_batch_manifest_v1.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _registered_path(root: Path, relative: str, expected_sha256: str) -> Path:
    candidate = (root / Path(*relative.replace("\\", "/").split("/"))).resolve(strict=True)
    if not candidate.is_relative_to(root.resolve(strict=True)) or candidate.is_symlink():
        raise ValueError("registered MATR manifest escaped the project root")
    if _sha256(candidate) != expected_sha256:
        raise ValueError(f"registered file hash mismatch: {relative}")
    return candidate


def _load_registry(root: Path) -> tuple[MatrThreeBatchManifest, SplitManifest]:
    manifest_path = (root / MANIFEST_RELATIVE).resolve(strict=True)
    manifest = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
    combined_path = _registered_path(
        root, manifest.combined_split_manifest, manifest.combined_split_sha256
    )
    split = SplitManifest.model_validate_json(combined_path.read_bytes())
    return manifest, split


def _cutoffs(mode: str) -> tuple[int, ...]:
    return {"smoke": (50,), "select": (20, 50, 100, 150), "final": (20, 50, 100, 150)}[mode]


def _mode_loader(mode: str) -> Callable[..., AdvancedSelectionMatrData | AdvancedFinalMatrData]:
    return (
        load_advanced_matr_final_data
        if mode == "final"
        else load_advanced_matr_selection_data
    )


def _batch_counts(data: AdvancedSelectionMatrData | AdvancedFinalMatrData) -> dict[str, Any]:
    partitions: tuple[str, ...] = ("train", "validation")
    if hasattr(data, "scalar_calibration") and hasattr(data, "scalar_test"):
        partitions += ("calibration", "test")
    scalar: dict[str, int] = {}
    hybrid: dict[str, int] = {}
    for partition in partitions:
        scalar_batch = getattr(data, f"scalar_{partition}")
        hybrid_batch = getattr(data, f"hybrid_{partition}")
        scalar[partition] = len(scalar_batch.cell_ids)
        hybrid[partition] = len(hybrid_batch.cell_ids)
    return {"scalar": scalar, "hybrid": hybrid}


def _normalization_hashes(
    data: AdvancedSelectionMatrData | AdvancedFinalMatrData,
) -> dict[str, str]:
    return {
        "scalar": data.scalar_normalizer.statistics_sha256,
        "hybrid": data.hybrid_normalizer.statistics_sha256,
        "scalar_training_cells": data.scalar_normalizer.training_cell_ids_sha256,
        "hybrid_training_cells": data.hybrid_normalizer.training_cell_ids_sha256,
    }


def prepare(
    *,
    project_root: Path = REPO_ROOT,
    mode: str,
    feature_version: str = FEATURE_VERSION,
    plan_only: bool = False,
) -> dict[str, Any]:
    if mode not in {"smoke", "select", "final"}:
        raise ValueError("mode must be smoke, select or final")
    root = project_root.resolve(strict=True)
    manifest, split = _load_registry(root)
    cutoffs = _cutoffs(mode)
    result: dict[str, Any] = {
        "schema_version": "advanced-matr-preflight-v1",
        "status": "PLAN_READY" if plan_only else "READY",
        "mode": mode,
        "feature_version": feature_version,
        "data_version": manifest.data_version,
        "split_version": manifest.split_version,
        "cutoffs": list(cutoffs),
        "total_cell_count": manifest.total_cell_count,
        "scalar_label_count": manifest.scalar_label_count,
        "hybrid_eligible_count": manifest.hybrid_eligible_count,
        "combined_split_sha256": manifest.combined_split_sha256,
        "cutoffs_detail": [],
    }
    if plan_only:
        return result

    loader = _mode_loader(mode)
    started = time.perf_counter()
    details: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        cutoff_started = time.perf_counter()
        data = loader(
            project_root=root,
            manifest=manifest,
            combined_split=split,
            cutoff_cycle=cutoff,
            feature_version=feature_version,
        )
        audit = data.masked_cycle_audit
        details.append(
            {
                "cutoff_cycle": cutoff,
                "counts": _batch_counts(data),
                "normalization_hashes": _normalization_hashes(data),
                "masked_cycle_count": audit.masked_cycle_count,
                "masked_audit_sha256": audit.audit_sha256,
                "masked_audit_entries": [
                    {
                        "cell_id": entry.cell_id,
                        "cycle_index": entry.cycle_index,
                        "reason": entry.reason,
                    }
                    for entry in audit.entries
                ],
                "elapsed_seconds": round(time.perf_counter() - cutoff_started, 3),
            }
        )
    result["cutoffs_detail"] = details
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["preflight_sha256"] = sha256_canonical(result)
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "select", "final"))
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--feature-version", default=FEATURE_VERSION)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    report = prepare(
        project_root=args.project_root,
        mode=args.mode,
        feature_version=args.feature_version,
        plan_only=args.plan_only,
    )
    if not args.plan_only:
        output = (
            args.project_root
            / "reports"
            / "data_quality"
            / f"advanced_matr_{args.mode}_preflight.json"
        )
        _write_json_atomic(output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
