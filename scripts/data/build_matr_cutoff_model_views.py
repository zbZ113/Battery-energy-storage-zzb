"""Build cutoff-specific frozen MATR tensor Views without changing cutoff-50 v2."""

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

from quanxin_life.core import sha256_canonical  # noqa: E402
from quanxin_life.data.matr_multibatch import MatrThreeBatchManifest  # noqa: E402
from quanxin_life.data.model_views.matr import (  # noqa: E402
    build_tensor_model_view,
    load_verified_matr_view_data,
    matr_tensor_payload,
    verify_existing_tensor_model_view,
)
from quanxin_life.data.model_views.schemas import ModelViewConfig  # noqa: E402
from quanxin_life.data.schemas import SplitManifest  # noqa: E402

_FEATURE_VERSION = "matr-fixed-grid-and-capacity-trend-v1"


def load_cutoff_configs(project_root: Path) -> tuple[ModelViewConfig, ...]:
    root = Path(project_root)
    directory = root / "configs" / "model_views" / "matr_cutoffs"
    configs = tuple(
        ModelViewConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    )
    if not configs:
        raise ValueError("MATR cutoff model View configs are missing")
    identities = {(config.view_id, config.cutoff_cycle) for config in configs}
    if len(identities) != len(configs):
        raise ValueError("MATR cutoff model View configs must be unique")
    return configs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build"))
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    configs = load_cutoff_configs(root)
    manifest, split, canonical_sha256 = _verified_matr_context(root)
    builder_sha = _builder_code_sha256(root)
    data_by_cutoff: dict[int, object] = {}
    for config in configs:
        cutoff = int(config.cutoff_cycle or 0)
        if args.mode == "plan":
            print(
                json.dumps(
                    {"cutoff_cycle": cutoff, "status": "READY", "view_id": config.view_id},
                    sort_keys=True,
                )
            )
            continue
        config_path = (
            root / "configs" / "model_views" / "matr_cutoffs" / f"{config.view_id}.json"
        )
        output = root / "data" / "model_views" / config.view_id / config.view_version
        result = verify_existing_tensor_model_view(
            output,
            view_id=config.view_id,
            view_version=config.view_version,
            task_type=config.task_type,
            target_semantics=config.target_semantics,
            mask_semantics=config.mask_semantics,
            cutoff_cycle=cutoff,
            canonical_sha256=canonical_sha256,
            split_sha256=manifest.combined_split_sha256,
            builder_code_sha256=builder_sha,
            config_sha256=_sha256_file(config_path),
        )
        if result is None:
            if cutoff not in data_by_cutoff:
                data_by_cutoff[cutoff] = load_verified_matr_view_data(
                    project_root=root,
                    manifest=manifest,
                    combined_split=split,
                    cutoff_cycle=cutoff,
                    feature_version=_FEATURE_VERSION,
                )
            base_view_id = _base_view_id(config.view_id)
            metadata, normalization, tensors, training_sha = matr_tensor_payload(
                data_by_cutoff[cutoff],  # type: ignore[arg-type]
                view_id=cast(
                    Literal["early_life_sequence", "soh_trajectory"],
                    base_view_id,
                ),
            )
            metadata["view_id"] = config.view_id
            metadata["view_family"] = base_view_id
            result = build_tensor_model_view(
                output_root=output,
                view_id=config.view_id,
                view_version=config.view_version,
                task_type=config.task_type,
                target_semantics=config.target_semantics,
                mask_semantics=config.mask_semantics,
                cutoff_cycle=cutoff,
                canonical_sha256=canonical_sha256,
                split_sha256=manifest.combined_split_sha256,
                builder_code_sha256=builder_sha,
                config_sha256=_sha256_file(config_path),
                normalization_sha256=sha256_canonical(normalization),
                training_entity_ids_sha256=training_sha,
                metadata=metadata,
                normalization=normalization,
                tensors=tensors,
            )
        print(
            json.dumps(
                {
                    "cutoff_cycle": cutoff,
                    "row_count": result.manifest.row_count,
                    "status": result.status.value,
                    "view_id": config.view_id,
                },
                sort_keys=True,
            )
        )
    return 0


def _base_view_id(view_id: str) -> str:
    if view_id.startswith("early_life_sequence_c"):
        return "early_life_sequence"
    if view_id.startswith("soh_trajectory_c"):
        return "soh_trajectory"
    raise ValueError("unsupported MATR cutoff model View family")


def _verified_matr_context(
    root: Path,
) -> tuple[MatrThreeBatchManifest, SplitManifest, str]:
    manifest_path = root / "reports" / "data_quality" / "matr_three_batch_manifest_v1.json"
    manifest = MatrThreeBatchManifest.model_validate_json(manifest_path.read_bytes())
    split_path = root / manifest.combined_split_manifest
    if _sha256_file(split_path) != manifest.combined_split_sha256:
        raise ValueError("verified MATR combined split SHA-256 mismatch")
    return manifest, SplitManifest.model_validate_json(split_path.read_bytes()), _sha256_file(
        manifest_path
    )


def _builder_code_sha256(root: Path) -> str:
    files = [
        *sorted((root / "src" / "quanxin_life" / "data" / "model_views").glob("*.py")),
        Path(__file__).resolve(),
    ]
    return sha256_canonical(
        {path.relative_to(root).as_posix(): _sha256_file(path) for path in files}
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
