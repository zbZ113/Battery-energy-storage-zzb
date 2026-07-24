from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

import quanxin_life.application.advanced_deployment_bundles as deployment_bundles
from quanxin_life.application.advanced_deployment_bundles import (
    AdvancedDeploymentBundleIndex,
    export_advanced_deployment_bundles,
)
from quanxin_life.application.deep_model_artifacts import (
    DeepArtifactFile,
    DeepArtifactFileRole,
    DeepArtifactKind,
    DeepModelArtifactManifest,
)
from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.training.advanced_outputs import (
    write_advanced_training_output_index,
)
from quanxin_life.training.checkpoint import (
    AdvancedCheckpointContext,
    AdvancedTrainingCheckpointManifest,
    CheckpointFile,
    TrainingProgress,
)

_SOURCE_COMMIT = "2" * 40
_CONFIG_SHA256 = "3" * 64
_TRAINING_INPUT_SHA256 = "4" * 64
_LOCAL_INPUT_SHA256 = "5" * 64
_CREATED_AT = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)
_FAMILIES = {
    "cyclepatch_direct": "cpd-test",
    "cyclepatch_batlinet": "cpb-test",
    "current_hybrid": "current-hybrid-reference",
    "hybridpatch_v2": "hpv2-test",
}


@dataclass(frozen=True)
class _EvidenceFixture:
    project_root: Path
    result_root: Path
    promotion_root: Path
    output_root: Path
    final_root: Path
    routes: tuple[dict[str, object], ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _route_value(route: object, name: str) -> Any:
    if isinstance(route, dict):
        return route[name]
    return getattr(route, name)


def _routes() -> tuple[dict[str, object], ...]:
    # POINT_ACCURACY and COVERAGE at cutoff 50 intentionally share one checkpoint.
    # This verifies checkpoint-level artifact de-duplication independently of routes.
    return (
        _route("RUL", "DEFAULT", "cyclepatch_direct", 20, 39, 11),
        _route("RUL", "POINT_ACCURACY", "cyclepatch_direct", 50, 39, 12),
        _route("RUL", "COVERAGE", "cyclepatch_direct", 50, 39, 12),
        _route("RUL", "POINT_ACCURACY", "cyclepatch_batlinet", 100, 39, 13),
        _route("RUL", "COVERAGE", "cyclepatch_direct", 100, 39, 14),
        _route("RUL", "POINT_ACCURACY", "cyclepatch_direct", 150, 38, 15),
        _route("RUL", "COVERAGE", "cyclepatch_batlinet", 150, 38, 16),
        _route("SOH", "MEAN_ACCURACY", "hybridpatch_v2", 20, 38, 21),
        _route("SOH", "TAIL_EFFICIENCY", "current_hybrid", 20, 42, 22),
        _route("SOH", "MEAN_ACCURACY", "hybridpatch_v2", 50, 38, 23),
        _route("SOH", "TAIL_EFFICIENCY", "current_hybrid", 50, 42, 24),
        _route("SOH", "MEAN_ACCURACY", "hybridpatch_v2", 100, 42, 25),
        _route("SOH", "TAIL_EFFICIENCY", "current_hybrid", 100, 42, 26),
        _route("SOH", "MEAN_ACCURACY", "hybridpatch_v2", 150, 42, 27),
        _route("SOH", "TAIL_EFFICIENCY", "current_hybrid", 150, 42, 28),
    )


def _route(
    task: str,
    role: str,
    family: str,
    cutoff_cycle: int,
    seed: int,
    best_epoch: int,
) -> dict[str, object]:
    return {
        "task": task,
        "family": family,
        "candidate_id": _FAMILIES[family],
        "cutoff_cycle": cutoff_cycle,
        "role": role,
        "disposition": "CONDITIONAL",
        "representative_seed": seed,
        "representative_best_epoch": best_epoch,
        "representative_seed_rule": "MINIMUM_BEST_VALIDATION_METRIC",
    }


@pytest.fixture
def evidence(tmp_path: Path) -> _EvidenceFixture:
    routes = _routes()
    project_root = tmp_path / "project"
    result_root = tmp_path / "result"
    promotion_root = tmp_path / "promotion"
    output_root = tmp_path / "deployment"
    project_root.mkdir()
    promotion_root.mkdir()
    final_root = (
        result_root
        / "runs"
        / "a100"
        / "matr-three-batch"
        / "advanced"
        / "final"
    )
    _write_final_result(final_root, routes)
    run_metrics = _write_run_metrics(result_root, routes)
    _write_promotion_evidence(promotion_root, routes, run_metrics)
    return _EvidenceFixture(
        project_root=project_root,
        result_root=result_root,
        promotion_root=promotion_root,
        output_root=output_root,
        final_root=final_root,
        routes=routes,
    )


def _write_final_result(
    final_root: Path,
    routes: tuple[dict[str, object], ...],
) -> None:
    final_root.mkdir(parents=True)
    for family, candidate_id in _FAMILIES.items():
        for cutoff_cycle in (20, 50, 100, 150):
            for seed in (38, 39, 40, 41, 42):
                run_root = (
                    final_root
                    / f"cutoff-{cutoff_cycle}"
                    / family
                    / candidate_id
                    / f"seed-{seed}"
                )
                run_root.mkdir(parents=True)
                (run_root / "run_status.json").write_text(
                    json.dumps({"status": "COMPLETED"}) + "\n",
                    encoding="utf-8",
                )
                (run_root / "training_log.jsonl").write_text(
                    '{"event":"completed"}\n', encoding="utf-8"
                )
                (run_root / "metrics_validation.csv").write_text(
                    "epoch,loss\n1,1.0\n", encoding="utf-8"
                )
                (run_root / "metrics_test.json").write_text(
                    '{"mae":1.0}\n', encoding="utf-8"
                )
                _write_representative_checkpoint(
                    run_root=run_root,
                    family=family,
                    candidate_id=candidate_id,
                    cutoff_cycle=cutoff_cycle,
                    seed=seed,
                    best_epoch=_run_epoch(family, cutoff_cycle, seed, routes),
                )
    (final_root / "aggregate_metrics.json").write_text(
        '{"operation_count":80}\n', encoding="utf-8"
    )
    write_advanced_training_output_index(
        final_root,
        mode="final",
        source_commit=_SOURCE_COMMIT,
        config_sha256=_CONFIG_SHA256,
        created_at=_CREATED_AT,
    )


def _run_epoch(
    family: str,
    cutoff_cycle: int,
    seed: int,
    routes: tuple[dict[str, object], ...],
) -> int:
    matches = {
        int(row["representative_best_epoch"])
        for row in routes
        if row["family"] == family
        and row["cutoff_cycle"] == cutoff_cycle
        and row["representative_seed"] == seed
    }
    if len(matches) > 1:
        raise AssertionError("fixture routes disagree on representative best epoch")
    return matches.pop() if matches else 100 + cutoff_cycle + seed


def _write_run_metrics(
    result_root: Path,
    routes: tuple[dict[str, object], ...],
) -> Path:
    selected_seeds: dict[tuple[str, int], int] = {}
    for row in routes:
        coordinate = (str(row["family"]), int(row["cutoff_cycle"]))
        seed = int(row["representative_seed"])
        previous = selected_seeds.setdefault(coordinate, seed)
        if previous != seed:
            raise AssertionError("fixture routes disagree on representative seed")
    rows = []
    for family, candidate_id in _FAMILIES.items():
        for cutoff_cycle in (20, 50, 100, 150):
            selected_seed = selected_seeds.get((family, cutoff_cycle), 38)
            for seed in (38, 39, 40, 41, 42):
                rows.append(
                    {
                        "family": family,
                        "candidate_id": candidate_id,
                        "cutoff_cycle": cutoff_cycle,
                        "seed": seed,
                        "status": "COMPLETED",
                        "best_epoch": _run_epoch(
                            family, cutoff_cycle, seed, routes
                        ),
                        "best_validation_metric": 0.1
                        if seed == selected_seed
                        else float(seed),
                    }
                )
    path = result_root / "analysis" / "advanced_final_run_metrics.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_representative_checkpoint(
    *,
    run_root: Path,
    family: str,
    candidate_id: str,
    cutoff_cycle: int,
    seed: int,
    best_epoch: int,
) -> None:
    checkpoint_name = f"epoch-{best_epoch:06d}"
    checkpoint_root = run_root / "checkpoints" / checkpoint_name
    checkpoint_root.mkdir(parents=True)
    context = AdvancedCheckpointContext(
        run_id=f"matr-{family}-{candidate_id}-c{cutoff_cycle}-s{seed}",
        dataset_id="MATR",
        target=PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE,
        model_name=family,
        cutoff_cycle=cutoff_cycle,
        seed=seed,
        config_sha256=_CONFIG_SHA256,
        input_bundle_sha256=_TRAINING_INPUT_SHA256,
        data_version="matr-test-v1",
        split_version="matr-cell-split-v1",
        feature_version="advanced-feature-v1",
        source_commit=_SOURCE_COMMIT,
        run_mode="final",
        stage="final",
        candidate_config_sha256="6" * 64,
        model_architecture_sha256="7" * 64,
        normalization_sha256="8" * 64,
        selection_manifest_sha256="9" * 64,
        reference_library_sha256="a" * 64
        if family == "cyclepatch_batlinet"
        else None,
    )
    progress = TrainingProgress(
        epoch=best_epoch,
        global_step=best_epoch,
        best_epoch=best_epoch,
        best_metric=1.0,
    )
    file_payloads = {
        "model.safetensors": b"safe-model-state",
        "optimizer.safetensors": b"safe-optimizer-state",
        "optimizer_state.json": b'{"param_groups":[]}',
        "scheduler_state.json": b'{"scheduler":null}',
        "rng_state.safetensors": b"safe-rng-state",
        "progress.json": (
            json.dumps(
                {
                    "context": context.model_dump(mode="json"),
                    "progress": progress.model_dump(mode="json"),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    files = []
    for name, payload in file_payloads.items():
        path = checkpoint_root / name
        path.write_bytes(payload)
        files.append(
            CheckpointFile(
                relative_path=name,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    manifest_payload = {
        "schema_version": "safe-training-checkpoint-v2",
        "context": context.model_dump(mode="json"),
        "progress": progress.model_dump(mode="json"),
        "files": [item.model_dump(mode="json") for item in files],
        "created_at": _CREATED_AT.isoformat().replace("+00:00", "Z"),
    }
    manifest = AdvancedTrainingCheckpointManifest.model_validate(
        {
            **manifest_payload,
            "manifest_sha256": sha256_canonical(manifest_payload),
        }
    )
    (checkpoint_root / "manifest.json").write_text(
        manifest.model_dump_json() + "\n", encoding="utf-8"
    )
    (run_root / "checkpoints" / "best.json").write_text(
        json.dumps({"checkpoint": checkpoint_name}) + "\n",
        encoding="utf-8",
    )


def _write_promotion_evidence(
    promotion_root: Path,
    routes: tuple[dict[str, object], ...],
    run_metrics: Path,
) -> None:
    decisions_path = promotion_root / "promotion_decisions.csv"
    _write_decisions(decisions_path, routes)
    source_path = promotion_root / "source_evidence.json"
    source_path.write_text(
        json.dumps(
            {
                "schema_version": "advanced-promotion-source-evidence-v1",
                "run_count": 80,
                "inputs": {
                    "advanced_final_run_metrics.csv": {
                        "path": str(run_metrics.resolve()),
                        "size_bytes": run_metrics.stat().st_size,
                        "sha256": _sha256(run_metrics),
                    }
                },
                "source_provenance": {
                    "source_commit": _SOURCE_COMMIT,
                    "data_version": "matr-test-v1",
                    "split_version": "matr-cell-split-v1",
                    "input_bundle_sha256": _TRAINING_INPUT_SHA256,
                    "local_reconstructed_input_bundle_sha256": _LOCAL_INPUT_SHA256,
                    "input_bundle_hashes_match": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "advanced-promotion-artifact-manifest-v1",
        "activation_status": "NOT_ACTIVATED",
        "created_at": _CREATED_AT.isoformat(),
        "files": [
            {
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in (decisions_path, source_path)
        ],
    }
    (promotion_root / "artifact_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_decisions(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _refresh_promotion_manifest(promotion_root: Path) -> None:
    path = promotion_root / "artifact_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload["files"]:
        registered = promotion_root / item["relative_path"]
        item["size_bytes"] = registered.stat().st_size
        item["sha256"] = _sha256(registered)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fake_exporter(call_keys: list[str]):
    def export_one(
        *,
        project_root: Path,
        result_root: Path,
        route: object,
        artifact_root: Path,
    ) -> DeepModelArtifactManifest:
        assert project_root.is_dir()
        assert result_root.is_dir()
        checkpoint_sha256 = str(
            _route_value(route, "checkpoint_manifest_sha256")
        )
        call_keys.append(checkpoint_sha256)
        artifact_id = str(uuid5(NAMESPACE_URL, checkpoint_sha256))
        root = artifact_root / artifact_id
        root.mkdir(parents=True)
        family = str(_route_value(route, "family"))
        kind = {
            "cyclepatch_direct": DeepArtifactKind.CYCLEPATCH_DIRECT,
            "cyclepatch_batlinet": DeepArtifactKind.CYCLEPATCH_BATLINET,
            "current_hybrid": DeepArtifactKind.CURRENT_HYBRID,
            "hybridpatch_v2": DeepArtifactKind.HYBRIDPATCH_V2,
        }[family]
        feature_context: dict[str, object] = {
            "dataset_id": "MATR",
            "data_version": "matr-test-v1",
            "split_version": "matr-cell-split-v1",
            "feature_version": "advanced-feature-v1",
            "cutoff_cycle": int(_route_value(route, "cutoff_cycle")),
            "candidate_config_sha256": "6" * 64,
            "normalization_sha256": "8" * 64,
        }
        if family in {"cyclepatch_direct", "cyclepatch_batlinet"}:
            feature_context["target_scaler"] = {"context_sha256": "b" * 64}
        if family == "cyclepatch_batlinet":
            feature_context["reference_library_sha256"] = "a" * 64
        checkpoint_model = (
            result_root
            / Path(str(_route_value(route, "checkpoint_directory")))
            / "model.safetensors"
        )
        role_payloads = {
            DeepArtifactFileRole.WEIGHTS: (
                "model.safetensors",
                checkpoint_model.read_bytes(),
            ),
            DeepArtifactFileRole.ARCHITECTURE: (
                "architecture.json",
                b'{"architecture":"verified"}',
            ),
            DeepArtifactFileRole.FEATURE_CONFIG: (
                "feature_config.json",
                json.dumps(feature_context, sort_keys=True).encode(),
            ),
        }
        files = []
        for role, (name, content) in role_payloads.items():
            target = root / name
            target.write_bytes(content)
            files.append(
                DeepArtifactFile(
                    role=role,
                    relative_path=f"{artifact_id}/{name}",
                    size_bytes=target.stat().st_size,
                    sha256=_sha256(target),
                )
            )
        if kind is DeepArtifactKind.CYCLEPATCH_BATLINET:
            target = root / "reference_library.json"
            target.write_text('{"reference_count":16}', encoding="utf-8")
            files.append(
                DeepArtifactFile(
                    role=DeepArtifactFileRole.REFERENCE_LIBRARY,
                    relative_path=f"{artifact_id}/{target.name}",
                    size_bytes=target.stat().st_size,
                    sha256=_sha256(target),
                )
            )
        manifest_payload = {
            "schema_version": "deep-model-artifact-v1",
            "artifact_id": artifact_id,
            "artifact_kind": kind,
            "files": [item.model_dump(mode="json") for item in files],
            "created_at": _CREATED_AT.isoformat().replace("+00:00", "Z"),
        }
        manifest = DeepModelArtifactManifest.model_validate(
            {
                **manifest_payload,
                "manifest_sha256": sha256_canonical(manifest_payload),
            }
        )
        (root / "manifest.json").write_text(
            manifest.model_dump_json() + "\n", encoding="utf-8"
        )
        round_trip = DeepModelArtifactManifest.model_validate_json(
            (root / "manifest.json").read_bytes()
        )
        assert round_trip == manifest
        return manifest

    return export_one


def test_exports_sha_bound_routes_and_deduplicated_artifacts(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )

    index = export_advanced_deployment_bundles(
        evidence.project_root,
        evidence.result_root,
        evidence.promotion_root,
        evidence.output_root,
        expected_promotion_manifest_sha256=_sha256(
            evidence.promotion_root / "artifact_manifest.json"
        ),
    )

    assert isinstance(index, AdvancedDeploymentBundleIndex)
    payload = index.model_dump(mode="json")
    assert payload["schema_version"] == "advanced-deployment-bundle-index-v1"
    assert payload["activation_status"] == "NOT_ACTIVATED"
    assert payload["source_commit"] == _SOURCE_COMMIT
    final_index = json.loads(
        (evidence.final_root / "output_index.json").read_text(encoding="utf-8")
    )
    assert payload["final_output_sha256"] == final_index["output_sha256"]
    assert payload["training_input_bundle_sha256"] == _TRAINING_INPUT_SHA256
    assert payload["local_reconstructed_input_bundle_sha256"] == _LOCAL_INPUT_SHA256
    assert payload["input_bundle_hashes_match"] is False
    assert len(payload["routes"]) == 15
    assert len(payload["artifacts"]) == 14
    assert len(calls) == len(set(calls)) == 14

    first = payload["routes"][0]
    assert first["task"] == "RUL"
    assert first["role"] == "DEFAULT"
    assert first["family"] == "cyclepatch_direct"
    assert first["candidate_id"] == "cpd-test"
    assert first["cutoff_cycle"] == 20
    assert first["seed"] == 39
    assert first["best_epoch"] == 11
    assert first["run_id"] == "matr-cyclepatch_direct-cpd-test-c20-s39"
    assert len(first["checkpoint_manifest_sha256"]) == 64
    assert len(first["checkpoint_model_sha256"]) == 64
    assert first["deep_artifact_id"]
    assert len(first["deep_artifact_manifest_sha256"]) == 64

    without_self_hash = {
        key: value for key, value in payload.items() if key != "manifest_sha256"
    }
    assert payload["manifest_sha256"] == sha256_canonical(without_self_hash)
    index_path = evidence.output_root / "deployment_bundle_index.json"
    assert AdvancedDeploymentBundleIndex.model_validate_json(
        index_path.read_bytes()
    ) == index


def test_resumes_verified_artifacts_without_rebuilding(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    first = export_advanced_deployment_bundles(
        evidence.project_root,
        evidence.result_root,
        evidence.promotion_root,
        evidence.output_root,
        expected_promotion_manifest_sha256=_sha256(
            evidence.promotion_root / "artifact_manifest.json"
        ),
    )
    assert len(first.routes) == 15
    assert len(first.artifacts) == 14
    assert len(calls) == 14

    def fail_if_rebuilt(**_: object) -> DeepModelArtifactManifest:
        pytest.fail("a verified resumed artifact must not be rebuilt")

    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        fail_if_rebuilt,
    )
    resumed_output = evidence.output_root.with_name("deployment-resumed")
    resumed = export_advanced_deployment_bundles(
        evidence.project_root,
        evidence.result_root,
        evidence.promotion_root,
        resumed_output,
        expected_promotion_manifest_sha256=_sha256(
            evidence.promotion_root / "artifact_manifest.json"
        ),
        resume_artifact_root=evidence.output_root / "artifacts",
    )

    assert len(resumed.routes) == 15
    assert len(resumed.artifacts) == 14
    assert {item.artifact_id for item in resumed.artifacts} == {
        item.artifact_id for item in first.artifacts
    }


def test_resume_rejects_rehashed_weights_that_no_longer_match_the_checkpoint(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    first = export_advanced_deployment_bundles(
        evidence.project_root,
        evidence.result_root,
        evidence.promotion_root,
        evidence.output_root,
        expected_promotion_manifest_sha256=_sha256(
            evidence.promotion_root / "artifact_manifest.json"
        ),
    )
    artifact = first.artifacts[0]
    artifact_root = evidence.output_root / "artifacts"
    manifest_path = artifact_root / artifact.artifact_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_item = next(
        item for item in manifest["files"] if item["role"] == "weights"
    )
    weight_path = artifact_root / weight_item["relative_path"]
    original = weight_path.read_bytes()
    replacement = b"X" * len(original)
    assert replacement != original
    weight_path.write_bytes(replacement)
    weight_item["size_bytes"] = weight_path.stat().st_size
    weight_item["sha256"] = _sha256(weight_path)
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_canonical(unsigned)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    def fail_if_rebuilt(**_: object) -> DeepModelArtifactManifest:
        pytest.fail("a mismatched resumed artifact must fail closed")

    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        fail_if_rebuilt,
    )
    with pytest.raises(ValueError, match=r"checkpoint.*weight|weight.*binding"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root.with_name("deployment-adversarial-resume"),
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
            resume_artifact_root=artifact_root,
        )


def test_rejects_promotion_file_that_differs_from_its_manifest(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    decisions = evidence.promotion_root / "promotion_decisions.csv"
    decisions.write_bytes(decisions.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=r"promotion.*SHA-256|promotion.*manifest"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []
    assert not (evidence.output_root / "deployment_bundle_index.json").exists()


def test_rejects_a_rehashed_promotion_that_differs_from_the_expected_manifest(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    original_manifest_sha256 = _sha256(
        evidence.promotion_root / "artifact_manifest.json"
    )
    decisions = evidence.promotion_root / "promotion_decisions.csv"
    decisions.write_bytes(decisions.read_bytes() + b"\n")
    _refresh_promotion_manifest(evidence.promotion_root)

    with pytest.raises(ValueError, match=r"expected promotion|promotion.*expected"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=original_manifest_sha256,
        )

    assert calls == []


def test_rejects_representative_seed_that_is_not_the_validation_minimum(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    changed = [dict(row) for row in evidence.routes]
    changed[0]["representative_seed"] = 40
    changed[0]["representative_best_epoch"] = _run_epoch(
        "cyclepatch_direct", 20, 40, evidence.routes
    )
    _write_decisions(
        evidence.promotion_root / "promotion_decisions.csv", tuple(changed)
    )
    _refresh_promotion_manifest(evidence.promotion_root)

    with pytest.raises(ValueError, match=r"validation.*minimum|representative seed"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []


def test_rejects_a_unique_but_incomplete_route_matrix(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    changed = [dict(row) for row in evidence.routes]
    changed[0]["role"] = "POINT_ACCURACY"
    _write_decisions(
        evidence.promotion_root / "promotion_decisions.csv", tuple(changed)
    )
    _refresh_promotion_manifest(evidence.promotion_root)

    with pytest.raises(ValueError, match=r"exact route matrix|route matrix"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []


def test_rejects_duplicate_route_coordinate(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    duplicated = (*evidence.routes[:-1], dict(evidence.routes[0]))
    _write_decisions(
        evidence.promotion_root / "promotion_decisions.csv", duplicated
    )
    _refresh_promotion_manifest(evidence.promotion_root)

    with pytest.raises(ValueError, match=r"duplicate.*route|route.*coordinate"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []


def test_rejects_representative_best_epoch_mismatch(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    changed = [dict(row) for row in evidence.routes]
    changed[0]["representative_best_epoch"] = 99
    _write_decisions(
        evidence.promotion_root / "promotion_decisions.csv", tuple(changed)
    )
    _refresh_promotion_manifest(evidence.promotion_root)

    with pytest.raises(ValueError, match=r"best.?epoch"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []


def test_rejects_representative_checkpoint_coordinate_mismatch(
    evidence: _EvidenceFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        deployment_bundles,
        "_rebuild_and_export_artifact",
        _fake_exporter(calls),
    )
    first = evidence.routes[0]
    run_root = (
        evidence.final_root
        / f"cutoff-{first['cutoff_cycle']}"
        / str(first["family"])
        / str(first["candidate_id"])
        / f"seed-{first['representative_seed']}"
    )
    pointer = json.loads(
        (run_root / "checkpoints" / "best.json").read_text(encoding="utf-8")
    )["checkpoint"]
    checkpoint_root = run_root / "checkpoints" / pointer
    manifest_path = checkpoint_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["context"]["seed"] = 41
    progress_path = checkpoint_root / "progress.json"
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_payload["context"]["seed"] = 41
    progress_path.write_text(
        json.dumps(progress_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress_item = next(
        item for item in manifest["files"] if item["relative_path"] == "progress.json"
    )
    progress_item["size_bytes"] = progress_path.stat().st_size
    progress_item["sha256"] = _sha256(progress_path)
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    manifest["manifest_sha256"] = sha256_canonical(unsigned)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_advanced_training_output_index(
        evidence.final_root,
        mode="final",
        source_commit=_SOURCE_COMMIT,
        config_sha256=_CONFIG_SHA256,
        created_at=_CREATED_AT,
    )

    with pytest.raises(ValueError, match=r"checkpoint.*coordinate|coordinate.*mismatch"):
        export_advanced_deployment_bundles(
            evidence.project_root,
            evidence.result_root,
            evidence.promotion_root,
            evidence.output_root,
            expected_promotion_manifest_sha256=_sha256(
                evidence.promotion_root / "artifact_manifest.json"
            ),
        )

    assert calls == []
