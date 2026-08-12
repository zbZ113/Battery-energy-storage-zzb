"""Build versioned BLAST reference-scenario tables and technical figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from quanxin_life.experiments.blast_scenario_artifacts import (
    build_blast_scenario_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/scenarios/blast_reference_scenarios_v1.json"),
    )
    parser.add_argument(
        "--validation-result-dir",
        type=Path,
        default=Path("reports/experiments/blast_naumann_v1/validation-v2"),
    )
    parser.add_argument(
        "--large-format-validation-result-dir",
        type=Path,
        default=Path("reports/experiments/blast_280ah_v1/validation-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/experiments/blast_scenarios_v1/scenario-v3"),
    )
    return parser


def _inside(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def _source_revision(root: Path, config_path: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    head = completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN_HEAD"
    files = [
        root / "src" / "quanxin_life" / "experiments" / "blast_scenario_artifacts.py",
        root / "src" / "quanxin_life" / "scenarios" / "runner.py",
        root / "src" / "quanxin_life" / "scenarios" / "support.py",
        root / "src" / "quanxin_life" / "scenarios" / "routes.py",
        root
        / "src"
        / "quanxin_life"
        / "scenarios"
        / "manifests"
        / "blast_lite_routes_v1.json",
        config_path,
        *sorted((root / "src" / "quanxin_life" / "_vendor" / "blast_lite").glob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in files:
        resolved = path.resolve(strict=True)
        digest.update(resolved.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return f"{head}+blast-scenario-source-sha256:{digest.hexdigest()}"


def main() -> int:
    args = _parser().parse_args()
    root = args.repository_root.resolve(strict=True)
    config_path = _inside(root, args.config).resolve(strict=True)
    result = build_blast_scenario_artifacts(
        repository_root=root,
        config_path=config_path,
        validation_result_dir=_inside(root, args.validation_result_dir),
        large_format_validation_result_dir=_inside(
            root,
            args.large_format_validation_result_dir,
        ),
        output_dir=_inside(root, args.output_dir),
        code_revision=_source_revision(root, config_path),
    )
    print(
        json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
