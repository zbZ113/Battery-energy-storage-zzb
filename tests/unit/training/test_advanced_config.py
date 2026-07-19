from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quanxin_life.core import PredictionTarget
from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.training.advanced_config import (
    AdvancedCandidate,
    AdvancedCandidateSearchConfig,
    AdvancedMatrThreeBatchRunConfig,
    AdvancedSelectionEvidence,
    AdvancedSelectionManifest,
    CyclePatchDirectCandidate,
    build_advanced_run_matrix,
)

CONFIG_ROOT = Path("configs/training/advanced")


def _candidate_payload(family: str, candidate_id: str) -> dict[str, object]:
    common: dict[str, object] = {
        "family": family,
        "candidate_id": candidate_id,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
    }
    if family == "cyclepatch_direct":
        return {
            **common,
            "d_model": 128,
            "layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "huber_delta": 1.0,
        }
    if family == "cyclepatch_batlinet":
        return {
            **common,
            "d_model": 128,
            "layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "lambda_pair": 0.5,
            "lambda_rank": 0.1,
            "reference_count": 32,
            "fusion_alpha": 0.5,
            "huber_delta": 1.0,
        }
    if family == "current_hybrid":
        return {
            "family": family,
            "candidate_id": candidate_id,
            "learning_rate": 0.0003,
            "hidden_dim": 64,
        }
    if family == "hybridpatch_v2":
        return {
            **common,
            "d_model": 128,
            "layers": 2,
            "heads": 4,
            "dropout": 0.1,
            "query_token_count": 8,
            "query_layers": 1,
            "decoder_hidden_dim": 64,
            "huber_delta": 1.0,
            "lambda_history": 0.1,
            "lambda_smooth": 0.01,
            "lambda_order": 0.05,
            "lambda_residual": 0.01,
        }
    raise AssertionError(family)


def _candidate(family: str, candidate_id: str) -> AdvancedCandidate:
    from pydantic import TypeAdapter

    return TypeAdapter(AdvancedCandidate).validate_python(
        _candidate_payload(family, candidate_id)
    )


def _manifest_payload() -> dict[str, object]:
    selected = tuple(
        _candidate(family, f"{family}-selected")
        for family in (
            "cyclepatch_direct",
            "cyclepatch_batlinet",
            "current_hybrid",
            "hybridpatch_v2",
        )
    )
    evidence = tuple(
        AdvancedSelectionEvidence(
            family=item.family,
            candidate_id=item.candidate_id,
            candidate_config_sha256=item.config_sha256,
            stage="selection_recheck",
            cutoff_cycle=cutoff,
            seed=seed,
            validation_mae=80.0,
        )
        for item in selected
        for cutoff in (20, 50, 100, 150)
        for seed in (38, 39, 40)
    )
    payload: dict[str, object] = {
        "schema_version": "advanced-selection-manifest-v1",
        "input_bundle_sha256": "1" * 64,
        "data_sha256": "2" * 64,
        "split_sha256": "3" * 64,
        "feature_sha256": "4" * 64,
        "normalization_sha256": "5" * 64,
        "source_commit": "6" * 40,
        "validation_cell_sha256": "7" * 64,
        "validation_results": [item.model_dump(mode="json") for item in evidence],
        "selected_candidates": [item.model_dump(mode="json") for item in selected],
    }
    return {**payload, "manifest_sha256": sha256_canonical(payload)}


def _outer_manifest_payload() -> dict[str, object]:
    selection = _manifest_payload()
    selected = selection["selected_candidates"]
    assert isinstance(selected, list)
    stage1 = []
    stage2 = []
    recheck = []
    survivors: dict[str, list[str]] = {}
    for candidate in selected:
        assert isinstance(candidate, dict)
        family = str(candidate["family"])
        candidate_id = str(candidate["candidate_id"])
        config_hash = sha256_canonical(candidate)
        survivors[family] = [candidate_id]
        common = {
            "family": family,
            "candidate_id": candidate_id,
            "candidate_config_sha256": config_hash,
            "cutoff_cycle": 100,
            "seed": 38,
            "status": "completed",
            "validation_mae": 80.0,
            "validation_rmse": 81.0,
            "monotonic_violation_rate": 0.0 if "hybrid" in family else None,
        }
        stage1.append({**common, "stage": "selection_stage1"})
        stage2.append({**common, "stage": "selection_stage2"})
        for cutoff in (20, 50, 100, 150):
            for seed in (38, 39, 40):
                recheck.append(
                    {
                        **common,
                        "stage": "selection_recheck",
                        "cutoff_cycle": cutoff,
                        "seed": seed,
                    }
                )
    baselines = [
        {
            "baseline": baseline,
            "cutoff_cycle": cutoff,
            "seed": seed,
            "validation_mae": 82.0,
            "validation_rmse": 83.0,
            "monotonic_violation_rate": 0.0 if baseline == "current_hybrid" else None,
        }
        for baseline in ("xgboost", "current_hybrid")
        for cutoff in (20, 50, 100, 150)
        for seed in (38, 39, 40)
    ]
    provisional = [
        {
            "family": "cyclepatch_batlinet",
            "candidate_id": "cyclepatch_batlinet-selected",
            "baseline": "xgboost",
            "eligible_for_final": True,
            "reason_codes": [],
        },
        {
            "family": "hybridpatch_v2",
            "candidate_id": "hybridpatch_v2-selected",
            "baseline": "current_hybrid",
            "eligible_for_final": True,
            "reason_codes": [],
        },
    ]
    payload: dict[str, object] = {
        "schema_version": "advanced-model-selection-v1",
        "selection": selection,
        "selection_trace": {
            "stage1_evidence": stage1,
            "stage1_survivors": survivors,
            "stage2_evidence": stage2,
            "stage2_finalists": survivors,
            "recheck_evidence": recheck,
        },
        "baseline_evidence": baselines,
        "provisional_eligibility": provisional,
    }
    return {**payload, "manifest_sha256": sha256_canonical(payload)}


def test_candidate_is_frozen_strict_hashed_and_path_safe() -> None:
    candidate = CyclePatchDirectCandidate.model_validate(
        _candidate_payload("cyclepatch_direct", "cp-direct-01")
    )
    assert len(candidate.config_sha256) == 64
    assert candidate.config_sha256 == CyclePatchDirectCandidate.model_validate(
        candidate.model_dump(mode="json")
    ).config_sha256
    with pytest.raises(ValidationError):
        CyclePatchDirectCandidate.model_validate(
            {**candidate.model_dump(mode="json"), "unknown": True}
        )
    with pytest.raises(ValidationError):
        candidate.candidate_id = "changed"  # type: ignore[misc]
    for unsafe_id in ("../candidate", "folder\\candidate", "/candidate"):
        with pytest.raises(ValidationError):
            CyclePatchDirectCandidate.model_validate(
                {**candidate.model_dump(mode="json"), "candidate_id": unsafe_id}
            )


def test_candidate_architecture_and_loss_boundaries_are_enforced() -> None:
    payload = _candidate_payload("cyclepatch_direct", "cp-direct-01")
    for update in (
        {"d_model": 96},
        {"heads": 3},
        {"dropout": 0.075},
        {"huber_delta": 0.0},
        {"learning_rate": 1.0},
    ):
        with pytest.raises(ValidationError):
            CyclePatchDirectCandidate.model_validate({**payload, **update})


def test_current_hybrid_hash_contains_only_real_consumable_parameters() -> None:
    candidate = _candidate("current_hybrid", "current-hybrid-reference")
    assert set(candidate.model_dump(mode="json")) == {
        "family",
        "candidate_id",
        "learning_rate",
        "hidden_dim",
    }
    with pytest.raises(ValidationError):
        type(candidate).model_validate(
            {**candidate.model_dump(mode="json"), "lambda_monotone": 0.1}
        )


def test_selection_policy_is_exactly_the_approved_two_stage_protocol() -> None:
    selection = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "selection.json").read_bytes()
    )
    policy = selection.selection_policy
    assert policy.stage1_epochs == 30
    assert policy.stage2_epochs == 90
    assert policy.keep_fraction == 0.5
    assert policy.per_family_finalists == 2
    assert (policy.initial_cutoff, policy.initial_seed) == (100, 38)
    assert policy.recheck_seeds == (38, 39, 40)
    assert policy.recheck_cutoffs == (20, 50, 100, 150)


@pytest.mark.parametrize(
    ("filename", "mode", "seeds", "cutoffs"),
    [
        ("smoke.json", "smoke", (38,), (50,)),
        ("selection.json", "select", (38, 39, 40), (20, 50, 100, 150)),
        ("final.json", "final", (38, 39, 40, 41, 42), (20, 50, 100, 150)),
    ],
)
def test_run_configs_are_real_and_exact(
    filename: str,
    mode: str,
    seeds: tuple[int, ...],
    cutoffs: tuple[int, ...],
) -> None:
    config = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / filename).read_bytes()
    )
    assert config.mode == mode
    assert config.dataset_id == "MATR"
    assert config.target is PredictionTarget.MATR_OFFICIAL_CYCLE_LIFE
    assert config.physical_gpu_index == 1
    assert config.precision == "fp32"
    assert config.seeds == seeds
    assert config.cutoffs == cutoffs
    if mode == "smoke":
        assert config.max_epochs == 10


def test_run_config_rejects_wrong_seed_cutoff_gpu_precision_and_cli_override() -> None:
    payload = json.loads((CONFIG_ROOT / "smoke.json").read_text(encoding="utf-8"))
    for update in (
        {"seeds": [20260712]},
        {"cutoffs": [20]},
        {"physical_gpu_index": 0},
        {"precision": "bf16"},
        {"hyperparameter_overrides": {"learning_rate": 1.0}},
    ):
        with pytest.raises(ValidationError):
            AdvancedMatrThreeBatchRunConfig.model_validate({**payload, **update})


def test_final_template_forbids_overrides_and_cannot_claim_placeholder_hash() -> None:
    payload = json.loads((CONFIG_ROOT / "final.json").read_text(encoding="utf-8"))
    config = AdvancedMatrThreeBatchRunConfig.model_validate(payload)
    assert config.selection_manifest_path is not None
    assert config.expected_selection_sha256 is None
    with pytest.raises(ValidationError):
        AdvancedMatrThreeBatchRunConfig.model_validate(
            {**payload, "cli_overrides": {"dropout": 0.2}}
        )


def test_select_path_contract_rejects_test_or_calibration_fields() -> None:
    payload = json.loads(
        (CONFIG_ROOT / "selection.json").read_text(encoding="utf-8")
    )
    for forbidden in ("test_manifest", "test_metrics", "calibration_manifest"):
        changed = json.loads(json.dumps(payload))
        changed["paths"][forbidden] = f"data/{forbidden}.json"
        with pytest.raises(ValidationError):
            AdvancedMatrThreeBatchRunConfig.model_validate(changed)


def test_selection_manifest_rejects_test_fields_and_hash_tampering() -> None:
    manifest = AdvancedSelectionManifest.model_validate(_manifest_payload())
    assert len(manifest.selected_candidates) == 4
    payload = manifest.model_dump(mode="json")
    payload["data_sha256"] = "8" * 64
    with pytest.raises(ValidationError, match="manifest_sha256"):
        AdvancedSelectionManifest.model_validate(payload)
    for forbidden in ("test_mae", "test_cell_ids", "calibration_metrics"):
        contaminated = _manifest_payload()
        contaminated[forbidden] = []
        with pytest.raises(ValidationError):
            AdvancedSelectionManifest.model_validate(contaminated)


def test_selection_manifest_requires_unique_complete_recheck_grid() -> None:
    payload = _manifest_payload()
    assert len(payload["validation_results"]) == 48  # type: ignore[arg-type]
    missing = json.loads(json.dumps(payload))
    missing["validation_results"].pop()
    manifest_payload = {
        key: value for key, value in missing.items() if key != "manifest_sha256"
    }
    missing["manifest_sha256"] = sha256_canonical(manifest_payload)
    with pytest.raises(ValidationError, match=r"12|recheck|complete"):
        AdvancedSelectionManifest.model_validate(missing)

    duplicate = json.loads(json.dumps(payload))
    duplicate["validation_results"].append(duplicate["validation_results"][0])
    manifest_payload = {
        key: value for key, value in duplicate.items() if key != "manifest_sha256"
    }
    duplicate["manifest_sha256"] = sha256_canonical(manifest_payload)
    with pytest.raises(ValidationError, match=r"unique|duplicate"):
        AdvancedSelectionManifest.model_validate(duplicate)

    contaminated = json.loads(json.dumps(payload))
    contaminated["validation_results"][0]["validation_rmse"] = 90.0
    with pytest.raises(ValidationError):
        AdvancedSelectionManifest.model_validate(contaminated)


def test_search_files_load_and_cover_all_searchable_families() -> None:
    filenames = (
        "cyclepatch_search.json",
        "batlinet_search.json",
        "hybridpatch_search.json",
    )
    searches = tuple(
        AdvancedCandidateSearchConfig.model_validate_json(
            (CONFIG_ROOT / filename).read_bytes()
        )
        for filename in filenames
    )
    assert tuple(search.family for search in searches) == (
        "cyclepatch_direct",
        "cyclepatch_batlinet",
        "hybridpatch_v2",
    )
    assert all(len(search.candidates) >= 4 for search in searches)
    assert all(
        candidate.family == search.family
        for search in searches
        for candidate in search.candidates
    )
    assert all(
        candidate.d_model in {128, 256}
        and candidate.layers in {2, 4}
        and candidate.heads in {4, 8}
        and candidate.dropout in {0.05, 0.1}
        for search in searches
        for candidate in search.candidates
    )
    batlinet = searches[1]
    assert {candidate.lambda_pair for candidate in batlinet.candidates} <= {
        0.25,
        0.5,
        1.0,
    }
    assert {candidate.reference_count for candidate in batlinet.candidates} <= {
        16,
        32,
        64,
    }
    hybridpatch = searches[2]
    assert {candidate.query_token_count for candidate in hybridpatch.candidates} <= {
        0,
        8,
        16,
    }
    assert all(
        candidate.lambda_history in {0.0, 0.1}
        and candidate.lambda_smooth in {0.0, 0.01}
        and candidate.lambda_order in {0.0, 0.05}
        and candidate.lambda_residual in {0.001, 0.01}
        for candidate in hybridpatch.candidates
    )


def test_matrix_builder_encodes_all_selection_stages() -> None:
    smoke = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "smoke.json").read_bytes()
    )
    selection = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "selection.json").read_bytes()
    )
    selection_keys = build_advanced_run_matrix(selection)
    assert selection_keys
    assert {key.stage for key in selection_keys} == {"selection_stage1"}
    assert {key.max_epochs for key in selection_keys} == {30}
    survivor_ids = tuple(key.candidate_id for key in selection_keys[:3])
    stage2 = build_advanced_run_matrix(
        selection,
        selection_stage="selection_stage2",
        candidate_ids=survivor_ids,
    )
    assert len(stage2) == 3
    assert {key.stage for key in stage2} == {"selection_stage2"}
    assert {key.max_epochs for key in stage2} == {90}
    finalist_ids = survivor_ids[:2]
    recheck = build_advanced_run_matrix(
        selection,
        selection_stage="selection_recheck",
        candidate_ids=finalist_ids,
    )
    assert len(recheck) == 24
    assert {key.stage for key in recheck} == {"selection_recheck"}
    assert {(key.cutoff_cycle, key.seed) for key in recheck} == {
        (cutoff, seed)
        for cutoff in (20, 50, 100, 150)
        for seed in (38, 39, 40)
    }
    assert len(build_advanced_run_matrix(smoke)) == 4


def test_final_matrix_fails_closed_until_actual_selection_is_bound(
    tmp_path: Path,
) -> None:
    final = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "final.json").read_bytes()
    )
    with pytest.raises(ValueError, match=r"selection|template|SHA"):
        build_advanced_run_matrix(final, repository_root=tmp_path)

    outer_payload = _outer_manifest_payload()
    manifest_bytes = (
        json.dumps(outer_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    manifest_path = tmp_path / "selection" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_bytes(manifest_bytes)
    resolved = final.model_copy(
        update={
            "selection_manifest_path": "selection/manifest.json",
            "expected_selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "candidates": tuple(
                AdvancedSelectionManifest.model_validate(
                    outer_payload["selection"]  # type: ignore[arg-type]
                ).selected_candidates
            ),
        }
    )

    final_keys = build_advanced_run_matrix(resolved, repository_root=tmp_path)
    assert len(final_keys) == 80
    assert len({key.family for key in final_keys}) == 4
    assert all(key.stage == "final" for key in final_keys)
    assert all(key.max_epochs == 500 for key in final_keys)

    manifest_path.write_bytes(manifest_bytes + b" ")
    with pytest.raises(ValueError, match=r"SHA-256|selection"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)


def test_final_matrix_rejects_inner_only_selection_manifest(tmp_path: Path) -> None:
    final = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "final.json").read_bytes()
    )
    inner = _manifest_payload()
    manifest_bytes = (
        json.dumps(inner, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    path = tmp_path / "selection" / "manifest.json"
    path.parent.mkdir()
    path.write_bytes(manifest_bytes)
    resolved = final.model_copy(
        update={
            "selection_manifest_path": "selection/manifest.json",
            "expected_selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "candidates": tuple(
                AdvancedSelectionManifest.model_validate(inner).selected_candidates
            ),
        }
    )
    with pytest.raises(ValueError, match=r"outer|complete|schema"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)


@pytest.mark.parametrize("field", ["selection_trace", "baseline_evidence"])
def test_final_matrix_rejects_outer_evidence_tampering_with_same_nested_selection(
    tmp_path: Path,
    field: str,
) -> None:
    final = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "final.json").read_bytes()
    )
    outer = _outer_manifest_payload()
    tampered = json.loads(json.dumps(outer))
    if field == "baseline_evidence":
        tampered[field][0]["validation_mae"] = 999.0
    else:
        tampered[field]["stage1_evidence"][0]["validation_mae"] = 999.0
    manifest_bytes = (
        json.dumps(tampered, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    path = tmp_path / "selection" / "manifest.json"
    path.parent.mkdir()
    path.write_bytes(manifest_bytes)
    resolved = final.model_copy(
        update={
            "selection_manifest_path": "selection/manifest.json",
            "expected_selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "candidates": tuple(
                AdvancedSelectionManifest.model_validate(outer["selection"]).selected_candidates  # type: ignore[arg-type]
            ),
        }
    )
    with pytest.raises(ValueError, match=r"outer|manifest_sha256|evidence"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)


def _write_resolved_outer_final(
    tmp_path: Path,
    outer_payload: dict[str, object],
) -> AdvancedMatrThreeBatchRunConfig:
    final = AdvancedMatrThreeBatchRunConfig.model_validate_json(
        (CONFIG_ROOT / "final.json").read_bytes()
    )
    manifest_bytes = (
        json.dumps(outer_payload, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode()
    path = tmp_path / "selection" / "manifest.json"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(manifest_bytes)
    selection = AdvancedSelectionManifest.model_validate(outer_payload["selection"])  # type: ignore[arg-type]
    return final.model_copy(
        update={
            "selection_manifest_path": "selection/manifest.json",
            "expected_selection_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "candidates": selection.selected_candidates,
        }
    )


def test_final_matrix_rejects_self_hashed_pseudo_trace(tmp_path: Path) -> None:
    outer = _outer_manifest_payload()
    tampered = json.loads(json.dumps(outer))
    tampered["selection_trace"]["recheck_evidence"][0]["validation_mae"] = 999.0  # type: ignore[index]
    payload = {
        key: value for key, value in tampered.items() if key != "manifest_sha256"
    }
    tampered["manifest_sha256"] = sha256_canonical(payload)
    resolved = _write_resolved_outer_final(tmp_path, tampered)

    with pytest.raises(ValueError, match=r"nested selection results|recheck"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)


def test_final_matrix_rejects_self_hashed_incomplete_baseline_grid(tmp_path: Path) -> None:
    outer = _outer_manifest_payload()
    tampered = json.loads(json.dumps(outer))
    tampered["baseline_evidence"].pop()  # type: ignore[union-attr]
    payload = {
        key: value for key, value in tampered.items() if key != "manifest_sha256"
    }
    tampered["manifest_sha256"] = sha256_canonical(payload)
    resolved = _write_resolved_outer_final(tmp_path, tampered)

    with pytest.raises(ValueError, match=r"24|baseline|grid"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)


@pytest.mark.parametrize("mutation", ["candidate", "baseline", "reason"])
def test_final_matrix_rejects_self_hashed_inconsistent_provisional_eligibility(
    tmp_path: Path,
    mutation: str,
) -> None:
    outer = _outer_manifest_payload()
    tampered = json.loads(json.dumps(outer))
    row = tampered["provisional_eligibility"][0]  # type: ignore[index]
    if mutation == "candidate":
        row["candidate_id"] = "not-the-selected-candidate"
    elif mutation == "baseline":
        row["baseline"] = "current_hybrid"
    else:
        row["eligible_for_final"] = False
        row["reason_codes"] = ["MANUALLY_CHANGED"]
    payload = {
        key: value for key, value in tampered.items() if key != "manifest_sha256"
    }
    tampered["manifest_sha256"] = sha256_canonical(payload)
    resolved = _write_resolved_outer_final(tmp_path, tampered)

    with pytest.raises(ValueError, match=r"provisional eligibility|selection evidence"):
        build_advanced_run_matrix(resolved, repository_root=tmp_path)
