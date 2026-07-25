from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

import quanxin_life.application.advanced_final_suite_import as advanced_final_import
from quanxin_life.application.advanced_final_suite_import import (
    AdvancedFinalSuiteImporter,
)
from quanxin_life.core import PredictionTarget, sha256_canonical
from quanxin_life.training.advanced_outputs import (
    AdvancedTrainingOutputIndex,
    write_advanced_training_output_index,
)
from quanxin_life.training.checkpoint import (
    AdvancedCheckpointContext,
    AdvancedTrainingCheckpointManifest,
    CheckpointFile,
    TrainingProgress,
)
from quanxin_life.training.engine import TrainingRunResult, TrainingRunStatus

_SOURCE_COMMIT = "2" * 40
_CONFIG_SHA256 = "3" * 64
_TRAINING_INPUT_SHA256 = "4" * 64
_LOCAL_INPUT_SHA256 = "5" * 64
_SELECTION_SHA256 = "6" * 64
_CREATED_AT = datetime(2026, 7, 23, 1, 52, 11, tzinfo=UTC)
_FAMILIES = {
    "current_hybrid": "current-hybrid-reference",
    "cyclepatch_batlinet": "cpb-d128-p50-r10-n32-a50",
    "cyclepatch_direct": "cpd-d128-l2-h4-p05",
    "hybridpatch_v2": "hpv2-d256-q16-reg-full",
}


@dataclass(frozen=True)
class _Fixture:
    evidence_root: Path
    final_root: Path
    transfer_archive: Path
    index: AdvancedTrainingOutputIndex
    index_file_sha256: str
    transfer_sha256: str


def test_registers_exact_mixed_80_run_suite_without_loading_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)

    def forbidden_loader(*args: object, **kwargs: object) -> None:
        raise AssertionError("Advanced Final registration must not deserialize tensors")

    monkeypatch.setattr("quanxin_life.training.checkpoint.load_file", forbidden_loader)
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )

    record = importer.register(
        fixture.final_root,
        expected_output_sha256=fixture.index.output_sha256,
        expected_output_index_file_sha256=fixture.index_file_sha256,
        transfer_archive=fixture.transfer_archive,
        transfer_sha256=fixture.transfer_sha256,
        local_reconstructed_input_bundle_sha256=_LOCAL_INPUT_SHA256,
        registered_at=_CREATED_AT,
    )
    registered = importer.resolve(record.import_id)

    assert record.task_scope == "MIXED_RUL_SOH"
    assert record.task_count == 80
    assert record.rul_task_count == 40
    assert record.soh_task_count == 40
    assert record.completed_task_count == 21
    assert record.early_stopped_task_count == 59
    assert record.training_input_bundle_sha256 == _TRAINING_INPUT_SHA256
    assert record.local_reconstructed_input_bundle_sha256 == _LOCAL_INPUT_SHA256
    assert record.input_bundle_hashes_match is False
    assert record.selection_manifest_sha256 == _SELECTION_SHA256
    assert len(registered.tasks) == 80
    assert {
        (task.task, task.output_target)
        for task in registered.tasks
    } == {
        ("RUL", "matr_official_cycle_life"),
        ("SOH", "soh_trajectory"),
    }
    assert all(
        task.checkpoint_target == PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
        for task in registered.tasks
    )


def test_registration_is_idempotent_and_reverifies_external_bytes(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )
    kwargs = {
        "expected_output_sha256": fixture.index.output_sha256,
        "expected_output_index_file_sha256": fixture.index_file_sha256,
        "transfer_archive": fixture.transfer_archive,
        "transfer_sha256": fixture.transfer_sha256,
        "local_reconstructed_input_bundle_sha256": _LOCAL_INPUT_SHA256,
    }

    first = importer.register(
        fixture.final_root,
        registered_at=_CREATED_AT,
        **kwargs,
    )
    second = importer.register(
        fixture.final_root,
        registered_at=datetime(2026, 7, 24, tzinfo=UTC),
        **kwargs,
    )

    assert second == first
    metrics = next(fixture.final_root.rglob("metrics_test.json"))
    metrics.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from its index"):
        importer.resolve(first.import_id)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("expected_output_sha256", "a" * 64, "output SHA-256"),
        ("expected_output_index_file_sha256", "b" * 64, "index file SHA-256"),
        ("transfer_sha256", "c" * 64, "transfer archive SHA-256"),
    ),
)
def test_rejects_external_trust_anchor_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )
    kwargs = {
        "expected_output_sha256": fixture.index.output_sha256,
        "expected_output_index_file_sha256": fixture.index_file_sha256,
        "transfer_archive": fixture.transfer_archive,
        "transfer_sha256": fixture.transfer_sha256,
        "local_reconstructed_input_bundle_sha256": _LOCAL_INPUT_SHA256,
        "registered_at": _CREATED_AT,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        importer.register(fixture.final_root, **kwargs)


def test_rejects_internally_resigned_unexpected_candidate_matrix(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    source = (
        fixture.final_root
        / "cutoff-20"
        / "cyclepatch_direct"
        / _FAMILIES["cyclepatch_direct"]
    )
    source.rename(source.with_name("unexpected-candidate"))
    index = write_advanced_training_output_index(
        fixture.final_root,
        mode="final",
        source_commit=_SOURCE_COMMIT,
        config_sha256=_CONFIG_SHA256,
        created_at=_CREATED_AT,
    )
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )

    with pytest.raises(ValueError, match="candidate matrix"):
        importer.register(
            fixture.final_root,
            expected_output_sha256=index.output_sha256,
            expected_output_index_file_sha256=_sha256(
                fixture.final_root / "output_index.json"
            ),
            transfer_archive=fixture.transfer_archive,
            transfer_sha256=fixture.transfer_sha256,
            local_reconstructed_input_bundle_sha256=_LOCAL_INPUT_SHA256,
            registered_at=_CREATED_AT,
        )


def test_concurrent_first_registration_preserves_one_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )
    verified = advanced_final_import._verify_final(
        fixture.final_root,
        expected_output_sha256=fixture.index.output_sha256,
        expected_output_index_file_sha256=fixture.index_file_sha256,
    )
    monkeypatch.setattr(advanced_final_import, "_verify_final", lambda *args, **kwargs: verified)
    original_write = advanced_final_import._write_record_atomic
    barrier = Barrier(2)

    def synchronized_write(*args: object, **kwargs: object) -> bool:
        barrier.wait(timeout=10)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(advanced_final_import, "_write_record_atomic", synchronized_write)

    def register(day: int):
        return importer.register(
            fixture.final_root,
            expected_output_sha256=fixture.index.output_sha256,
            expected_output_index_file_sha256=fixture.index_file_sha256,
            transfer_archive=fixture.transfer_archive,
            transfer_sha256=fixture.transfer_sha256,
            local_reconstructed_input_bundle_sha256=_LOCAL_INPUT_SHA256,
            registered_at=datetime(2026, 7, day, tzinfo=UTC),
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = tuple(pool.map(register, (23, 24)))

    assert records[0] == records[1]
    assert len(tuple((tmp_path / "registry" / "records").glob("*.json"))) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_resolve_rejects_nested_evidence_junction(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    importer = AdvancedFinalSuiteImporter(
        tmp_path / "registry",
        evidence_root=fixture.evidence_root,
    )
    record = importer.register(
        fixture.final_root,
        expected_output_sha256=fixture.index.output_sha256,
        expected_output_index_file_sha256=fixture.index_file_sha256,
        transfer_archive=fixture.transfer_archive,
        transfer_sha256=fixture.transfer_sha256,
        local_reconstructed_input_bundle_sha256=_LOCAL_INPUT_SHA256,
        registered_at=_CREATED_AT,
    )
    nested = fixture.final_root / "cutoff-20"
    outside = tmp_path / "outside-cutoff-20"
    nested.rename(outside)
    if not _create_junction(nested, outside):
        pytest.skip("Windows junction creation is unavailable")

    with pytest.raises(ValueError, match=r"reparse|junction|regular"):
        importer.resolve(record.import_id)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_registration_rejects_records_junction_swap(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    registry_root = tmp_path / "registry"
    importer = AdvancedFinalSuiteImporter(
        registry_root,
        evidence_root=fixture.evidence_root,
    )
    records = registry_root / "records"
    records.rmdir()
    outside = tmp_path / "outside-records"
    outside.mkdir()
    if not _create_junction(records, outside):
        pytest.skip("Windows junction creation is unavailable")

    with pytest.raises(ValueError, match=r"reparse|registry"):
        importer.register(
            fixture.final_root,
            expected_output_sha256=fixture.index.output_sha256,
            expected_output_index_file_sha256=fixture.index_file_sha256,
            transfer_archive=fixture.transfer_archive,
            transfer_sha256=fixture.transfer_sha256,
            local_reconstructed_input_bundle_sha256=_LOCAL_INPUT_SHA256,
            registered_at=_CREATED_AT,
        )
    assert tuple(outside.iterdir()) == ()


def _write_fixture(tmp_path: Path) -> _Fixture:
    evidence_root = tmp_path / "evidence"
    final_root = (
        evidence_root
        / "runs"
        / "a100"
        / "matr-three-batch"
        / "advanced"
        / "final"
    )
    final_root.mkdir(parents=True)
    aggregate_runs: list[dict[str, object]] = []
    completed = 0
    for family, candidate_id in _FAMILIES.items():
        for cutoff_cycle in (20, 50, 100, 150):
            for seed in (38, 39, 40, 41, 42):
                status = (
                    TrainingRunStatus.COMPLETED
                    if family == "current_hybrid" or completed == 20
                    else TrainingRunStatus.EARLY_STOPPED
                )
                if status is TrainingRunStatus.COMPLETED:
                    completed += 1
                best_epoch = cutoff_cycle + seed
                run_root = (
                    final_root
                    / f"cutoff-{cutoff_cycle}"
                    / family
                    / candidate_id
                    / f"seed-{seed}"
                )
                _write_run(
                    run_root,
                    family=family,
                    candidate_id=candidate_id,
                    cutoff_cycle=cutoff_cycle,
                    seed=seed,
                    best_epoch=best_epoch,
                    status=status,
                )
                aggregate_runs.append(
                    {
                        "family": family,
                        "candidate_id": candidate_id,
                        "cutoff_cycle": cutoff_cycle,
                        "seed": seed,
                        "status": status.value,
                        "best_epoch": best_epoch,
                        "best_metric": 1.0,
                        "run_directory": (
                            "runs/a100/matr-three-batch/advanced/final/"
                            f"cutoff-{cutoff_cycle}/{family}/{candidate_id}/seed-{seed}"
                        ),
                    }
                )
    (final_root / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "mode": "final",
                "dataset_id": "MATR",
                "target": PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE.value,
                "runs": aggregate_runs,
                "run_count": 80,
                "input_bundle_sha256": _TRAINING_INPUT_SHA256,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    index = write_advanced_training_output_index(
        final_root,
        mode="final",
        source_commit=_SOURCE_COMMIT,
        config_sha256=_CONFIG_SHA256,
        created_at=_CREATED_AT,
    )
    transfer = evidence_root / "advanced-final.tgz"
    transfer.write_bytes(b"offline-transfer-archive")
    return _Fixture(
        evidence_root=evidence_root,
        final_root=final_root,
        transfer_archive=transfer,
        index=index,
        index_file_sha256=_sha256(final_root / "output_index.json"),
        transfer_sha256=_sha256(transfer),
    )


def _write_run(
    run_root: Path,
    *,
    family: str,
    candidate_id: str,
    cutoff_cycle: int,
    seed: int,
    best_epoch: int,
    status: TrainingRunStatus,
) -> None:
    run_root.mkdir(parents=True)
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
        candidate_config_sha256=sha256_canonical({"candidate_id": candidate_id}),
        model_architecture_sha256="7" * 64,
        normalization_sha256="8" * 64,
        selection_manifest_sha256=_SELECTION_SHA256,
        reference_library_sha256="9" * 64
        if family == "cyclepatch_batlinet"
        else None,
    )
    progress = TrainingProgress(
        epoch=best_epoch,
        global_step=best_epoch,
        best_epoch=best_epoch,
        best_metric=1.0,
    )
    checkpoint_name = f"epoch-{best_epoch:06d}"
    checkpoint_root = run_root / "checkpoints" / checkpoint_name
    checkpoint_root.mkdir(parents=True)
    file_payloads = {
        "model.safetensors": b"opaque-model-bytes",
        "optimizer.safetensors": b"opaque-optimizer-bytes",
        "optimizer_state.json": b'{"param_groups":[]}',
        "scheduler_state.json": b'{"state":null}',
        "rng_state.safetensors": b"opaque-rng-bytes",
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
    files: list[CheckpointFile] = []
    for name, payload in file_payloads.items():
        path = checkpoint_root / name
        path.write_bytes(payload)
        files.append(
            CheckpointFile(
                relative_path=name,
                size_bytes=len(payload),
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
        manifest.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (run_root / "checkpoints" / "best.json").write_text(
        json.dumps({"checkpoint": checkpoint_name}) + "\n",
        encoding="utf-8",
    )
    result = TrainingRunResult(
        run_id=context.run_id,
        context_sha256=sha256_canonical(context.model_dump(mode="json")),
        status=status,
        last_epoch=best_epoch,
        best_epoch=best_epoch,
        best_metric=1.0,
        training_time_seconds=1.0,
        peak_gpu_memory_bytes=1,
        finished_at=_CREATED_AT,
    )
    (run_root / "run_status.json").write_text(
        result.model_dump_json() + "\n",
        encoding="utf-8",
    )
    (run_root / "training_log.jsonl").write_text(
        '{"event":"completed"}\n',
        encoding="utf-8",
    )
    (run_root / "metrics_epoch.csv").write_text(
        "epoch,loss\n1,1.0\n",
        encoding="utf-8",
    )
    (run_root / "metrics_validation.csv").write_text(
        "epoch,loss\n1,1.0\n",
        encoding="utf-8",
    )
    (run_root / "metrics_test.json").write_text(
        '{"schema_version":"advanced-test-metrics-v1"}\n',
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_junction(link: Path, target: Path) -> bool:
    created = subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    return created.returncode == 0
