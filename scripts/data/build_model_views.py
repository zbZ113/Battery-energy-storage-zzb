"""Plan or build frozen model views from reviewed processed row inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Literal, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quanxin_life.core import TrainingReadableSplit, sha256_canonical  # noqa: E402
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest  # noqa: E402
from quanxin_life.data.model_views.builder import (  # noqa: E402
    build_model_view,
    verify_model_view,
)
from quanxin_life.data.model_views.matr import (  # noqa: E402
    build_tensor_model_view,
    load_verified_matr_view_data,
    matr_tensor_payload,
    verify_existing_tensor_model_view,
)
from quanxin_life.data.model_views.registry import ModelViewRegistry  # noqa: E402
from quanxin_life.data.model_views.schemas import ModelViewRow  # noqa: E402
from quanxin_life.data.processing import DatasetProcessor  # noqa: E402
from quanxin_life.data.schemas import SplitManifest  # noqa: E402

_MATR_VIEW_IDS = frozenset({"early_life_sequence", "soh_trajectory"})
_MATR_FEATURE_VERSION = "matr-fixed-grid-and-capacity-trend-v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build"))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--all-ready", action="store_true")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    registry = ModelViewRegistry.load(root / "configs" / "model_views")
    builder_sha = _builder_code_sha256(root)
    matr_context: tuple[MatrThreeBatchManifest, SplitManifest, str] | None = None
    matr_data = None
    for config in registry.configs:
        processed_ready = _processed_source_ready(root, config.view_id)
        if not processed_ready:
            print(
                json.dumps(
                    {"status": "BLOCKED_DATA_VIEW_SOURCE", "view_id": config.view_id},
                    sort_keys=True,
                )
            )
            continue
        if args.mode == "plan":
            print(json.dumps({"status": "READY", "view_id": config.view_id}, sort_keys=True))
            continue
        config_path = next(
            path
            for path in (root / "configs" / "model_views").glob("*.json")
            if json.loads(path.read_text(encoding="utf-8"))["view_id"] == config.view_id
        )
        output = root / "data" / "model_views" / config.view_id / config.view_version
        config_sha256 = _sha256_file(config_path)
        if config.view_id not in _MATR_VIEW_IDS and output.exists():
            existing = verify_model_view(output)
            if (
                existing.view_id != config.view_id
                or existing.view_version != config.view_version
                or existing.task_type != config.task_type
                or existing.target_semantics != config.target_semantics
                or existing.mask_semantics != config.mask_semantics
                or existing.cutoff_cycle != config.cutoff_cycle
                or existing.config_sha256 != config_sha256
            ):
                raise ValueError(
                    "changed model view context requires a new model view version"
                )
            print(
                json.dumps(
                    {
                        "row_count": existing.row_count,
                        "status": "SKIPPED_VALID",
                        "view_id": config.view_id,
                    },
                    sort_keys=True,
                )
            )
            continue
        if config.view_id in _MATR_VIEW_IDS:
            if matr_context is None:
                matr_context = _verified_matr_context(root)
            manifest, split, canonical_sha256 = matr_context
            result = verify_existing_tensor_model_view(
                output,
                view_id=config.view_id,
                view_version=config.view_version,
                task_type=config.task_type,
                target_semantics=config.target_semantics,
                mask_semantics=config.mask_semantics,
                cutoff_cycle=config.cutoff_cycle or 0,
                canonical_sha256=canonical_sha256,
                split_sha256=manifest.combined_split_sha256,
                builder_code_sha256=builder_sha,
                config_sha256=config_sha256,
            )
            if result is None:
                if matr_data is None:
                    matr_data = load_verified_matr_view_data(
                        project_root=root,
                        manifest=manifest,
                        combined_split=split,
                        cutoff_cycle=config.cutoff_cycle or 0,
                        feature_version=_MATR_FEATURE_VERSION,
                    )
                metadata, normalization, tensors, training_sha = matr_tensor_payload(
                    matr_data,
                    view_id=cast(
                        Literal["early_life_sequence", "soh_trajectory"],
                        config.view_id,
                    ),
                )
                if metadata["target_semantics"] != config.target_semantics:
                    raise ValueError("MATR tensor target semantics differ from view config")
                result = build_tensor_model_view(
                    output_root=output,
                    view_id=config.view_id,
                    view_version=config.view_version,
                    task_type=config.task_type,
                    target_semantics=config.target_semantics,
                    mask_semantics=config.mask_semantics,
                    cutoff_cycle=config.cutoff_cycle or 0,
                    canonical_sha256=canonical_sha256,
                    split_sha256=manifest.combined_split_sha256,
                    builder_code_sha256=builder_sha,
                    config_sha256=config_sha256,
                    normalization_sha256=sha256_canonical(normalization),
                    training_entity_ids_sha256=training_sha,
                    metadata=metadata,
                    normalization=normalization,
                    tensors=tensors,
                )
        else:
            rows, canonical_sha256, split_sha256 = _processed_rows(
                root, config.view_id, config.feature_names
            )
            result = build_model_view(
                rows=rows,
                config=config,
                output_root=output,
                canonical_sha256=canonical_sha256,
                split_sha256=split_sha256,
                builder_code_sha256=builder_sha,
                config_sha256=config_sha256,
            )
        print(
            json.dumps(
                {
                    "row_count": result.manifest.row_count,
                    "status": result.status.value,
                    "view_id": config.view_id,
                },
                sort_keys=True,
            )
        )
    return 0


def _processed_source_ready(root: Path, view_id: str) -> bool:
    if view_id in _MATR_VIEW_IDS:
        try:
            _verified_matr_context(root)
        except (OSError, ValueError):
            return False
        return True
    if view_id != "degradation_condition":
        return False
    processor = DatasetProcessor(root)
    try:
        for dataset_id in ("NAUMANN_CYCLE", "NAUMANN_CALENDAR"):
            processor.verify(_canonical_bundle(root, dataset_id))
    except (OSError, ValueError):
        return False
    return True


def _verified_matr_context(
    root: Path,
) -> tuple[MatrThreeBatchManifest, SplitManifest, str]:
    manifest_path = root / "reports" / "data_quality" / "matr_three_batch_manifest_v1.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("verified MATR three-batch manifest is missing")
    manifest = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
    split_path = root / manifest.combined_split_manifest
    if (
        split_path.is_symlink()
        or not split_path.is_file()
        or _sha256_file(split_path) != manifest.combined_split_sha256
    ):
        raise ValueError("verified MATR combined split SHA-256 mismatch")
    split = SplitManifest.model_validate_json(split_path.read_bytes())
    if split.dataset_id != "MATR" or set(split.all_cells) != {
        cell_id
        for component in manifest.batches
        for partition in ("train", "validation", "calibration", "test")
        for cell_id in getattr(
            SplitManifest.model_validate_json(
                (root / component.split_manifest).read_bytes()
            ),
            partition,
        )
    }:
        raise ValueError("MATR combined split does not match component splits")
    for component in manifest.batches:
        for relative, expected_sha in (
            (component.conversion_report, component.conversion_report_sha256),
            (component.split_manifest, component.split_manifest_sha256),
            (component.eligibility_report, component.eligibility_report_sha256),
            (component.supervision_report, component.supervision_report_sha256),
        ):
            path = root / relative
            if path.is_symlink() or not path.is_file() or _sha256_file(path) != expected_sha:
                raise ValueError("registered MATR evidence SHA-256 mismatch")
        if not (root / component.processed_root).is_dir() or not (
            root / component.supervision_root
        ).is_dir():
            raise ValueError("registered MATR processed artifacts are missing")
    return manifest, split, _sha256_file(manifest_path)


def _processed_rows(
    root: Path,
    view_id: str,
    feature_names: tuple[str, ...],
) -> tuple[tuple[ModelViewRow, ...], str, str]:
    if view_id != "degradation_condition":
        raise ValueError(f"processed row adapter is unavailable for {view_id}")
    records: list[dict[str, object]] = []
    artifact_hashes: dict[str, str] = {}
    processor = DatasetProcessor(root)
    for dataset_id in ("NAUMANN_CYCLE", "NAUMANN_CALENDAR"):
        bundle = _canonical_bundle(root, dataset_id)
        manifest = processor.verify(bundle)
        if manifest.dataset_id != dataset_id:
            raise ValueError("canonical bundle dataset identity mismatch")
        payload = json.loads(
            (bundle / "observations" / "condition_observations.json").read_text(
                encoding="utf-8"
            )
        )
        records.extend(payload["records"])
        artifact_hashes[dataset_id] = _sha256_file(bundle / "artifact_manifest.json")
    grouped: dict[str, dict[str, float]] = {}
    for record in records:
        record_id = str(record["record_id"])
        group_id = record_id.rsplit(":", 1)[0]
        grouped.setdefault(group_id, {})[str(record["quantity_name"])] = float(
            cast(str | int | float, record["value"])
        )
    entity_ids = sorted({":".join(group_id.split(":")[:2]) for group_id in grouped})
    assignments = _assign_entity_splits(entity_ids)
    feature_aliases = {"temperature": "temperature_c"}
    rows: list[ModelViewRow] = []
    for group_id, values in sorted(grouped.items()):
        entity_id = ":".join(group_id.split(":")[:2])
        features = {
            name: values.get(
                next(
                    (raw for raw, alias in feature_aliases.items() if alias == name),
                    name,
                )
            )
            for name in feature_names
        }
        rows.append(
            ModelViewRow(
                entity_id=entity_id,
                split=assignments[entity_id],
                features=features,
                feature_mask={name: value is not None for name, value in features.items()},
                target=None,
                right_censored=False,
                evidence="NO_POINT_TARGET:INCOMPATIBLE_RESPONSE_UNITS",
            )
        )
    split_sha256 = sha256_canonical(
        {entity_id: split.value for entity_id, split in sorted(assignments.items())}
    )
    return tuple(rows), sha256_canonical(artifact_hashes), split_sha256


def _canonical_bundle(root: Path, dataset_id: str) -> Path:
    return root / "data" / "processed" / dataset_id / "Mendeley-v1" / "canonical-v1"


def _builder_code_sha256(root: Path) -> str:
    model_view_root = root / "src" / "quanxin_life" / "data" / "model_views"
    files = [*sorted(model_view_root.glob("*.py")), Path(__file__).resolve()]
    return sha256_canonical(
        {path.relative_to(root).as_posix(): _sha256_file(path) for path in files}
    )


def _assign_entity_splits(
    entity_ids: list[str],
) -> dict[str, TrainingReadableSplit]:
    by_dataset: dict[str, list[str]] = {}
    for entity_id in entity_ids:
        by_dataset.setdefault(entity_id.split(":", 1)[0], []).append(entity_id)
    assignments: dict[str, TrainingReadableSplit] = {}
    for identities in by_dataset.values():
        total = len(identities)
        train_end = max(1, round(total * 0.60))
        validation_end = max(train_end + 1, round(total * 0.75))
        calibration_end = max(validation_end + 1, round(total * 0.85))
        for index, entity_id in enumerate(sorted(identities)):
            if index < train_end:
                split = TrainingReadableSplit.TRAIN
            elif index < validation_end:
                split = TrainingReadableSplit.VALIDATION
            elif index < calibration_end:
                split = TrainingReadableSplit.CALIBRATION
            else:
                split = TrainingReadableSplit.TEST
            assignments[entity_id] = split
    return assignments


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
