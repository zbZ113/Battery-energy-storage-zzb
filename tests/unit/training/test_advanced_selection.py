from __future__ import annotations

from itertools import product

import pytest
from pydantic import ValidationError

from quanxin_life.core.hashing import sha256_canonical
from quanxin_life.training.advanced_config import (
    CurrentHybridCandidate,
    CyclePatchBatLiNetCandidate,
    CyclePatchDirectCandidate,
    HybridPatchV2Candidate,
)
from quanxin_life.training.advanced_selection import (
    AdvancedValidationEvidence,
    BaselineValidationEvidence,
    build_advanced_model_selection_manifest,
    select_stage1_candidates,
    select_stage2_finalists,
)


def _evidence(
    candidate_id: str,
    mae: float | None,
    *,
    family: str = "cyclepatch_direct",
    stage: str = "selection_stage1",
    status: str = "completed",
) -> AdvancedValidationEvidence:
    return AdvancedValidationEvidence.model_validate(
        {
            "family": family,
            "candidate_id": candidate_id,
            "candidate_config_sha256": sha256_canonical({"candidate": candidate_id}),
            "stage": stage,
            "cutoff_cycle": 100,
            "seed": 38,
            "status": status,
            "validation_mae": mae,
            "validation_rmse": None if mae is None else mae + 1.0,
            "monotonic_violation_rate": 0.0 if family == "hybridpatch_v2" else None,
        }
    )


def test_validation_evidence_is_validation_only_and_eliminates_failed_nonfinite() -> None:
    with pytest.raises(ValidationError):
        AdvancedValidationEvidence.model_validate(
            {**_evidence("c1", 1.0).model_dump(mode="python"), "test_mae": 0.1}
        )
    with pytest.raises(ValidationError):
        AdvancedValidationEvidence.model_validate(
            {
                **_evidence("c1", 1.0).model_dump(mode="python"),
                "validation_mae": float("nan"),
            }
        )

    stage1 = (
        *(_evidence(f"c{index}", mae) for index, mae in enumerate((1.0, 2.0, 3.0, 4.0), start=1)),
        _evidence("failed", None, status="failed"),
    )
    assert select_stage1_candidates(stage1) == {"cyclepatch_direct": ("c1", "c2")}


def test_stage2_keeps_two_per_family_by_validation_mae() -> None:
    evidence = tuple(
        _evidence(
            candidate_id,
            mae,
            family=family,
            stage="selection_stage2",
        )
        for family in ("cyclepatch_direct", "hybridpatch_v2")
        for candidate_id, mae in ((f"{family}-a", 3.0), (f"{family}-b", 1.0), (f"{family}-c", 2.0))
    )
    assert select_stage2_finalists(evidence) == {
        "cyclepatch_direct": ("cyclepatch_direct-b", "cyclepatch_direct-c"),
        "hybridpatch_v2": ("hybridpatch_v2-b", "hybridpatch_v2-c"),
    }


def _candidates():
    return (
        CyclePatchDirectCandidate(
            candidate_id="direct",
            learning_rate=3e-4,
            weight_decay=0.01,
            d_model=128,
            layers=2,
            heads=4,
            dropout=0.05,
            huber_delta=1.0,
        ),
        CyclePatchBatLiNetCandidate(
            candidate_id="batlinet",
            learning_rate=3e-4,
            weight_decay=0.01,
            d_model=128,
            layers=2,
            heads=4,
            dropout=0.05,
            lambda_pair=0.5,
            lambda_rank=0.0,
            reference_count=16,
            fusion_alpha=0.5,
            huber_delta=1.0,
        ),
        CurrentHybridCandidate(
            candidate_id="current",
            learning_rate=3e-4,
            hidden_dim=64,
        ),
        HybridPatchV2Candidate(
            candidate_id="hybridpatch",
            learning_rate=3e-4,
            weight_decay=0.01,
            d_model=128,
            layers=2,
            heads=4,
            dropout=0.05,
            query_token_count=8,
            query_layers=1,
            decoder_hidden_dim=64,
            huber_delta=1.0,
            lambda_history=0.1,
            lambda_smooth=0.01,
            lambda_order=0.05,
            lambda_residual=0.01,
        ),
    )


def _recheck(*, eligibility_good: bool = True):
    rows = []
    candidates = _candidates()
    for candidate in candidates:
        for cutoff, seed in product((20, 50, 100, 150), (38, 39, 40)):
            if candidate.family in {"cyclepatch_batlinet", "hybridpatch_v2"}:
                mae = 98.0 if eligibility_good else 101.0
            else:
                mae = 105.0
            rows.append(
                AdvancedValidationEvidence(
                    family=candidate.family,
                    candidate_id=candidate.candidate_id,
                    candidate_config_sha256=candidate.config_sha256,
                    stage="selection_recheck",
                    cutoff_cycle=cutoff,
                    seed=seed,
                    status="completed",
                    validation_mae=mae,
                    validation_rmse=mae + (0.0 if eligibility_good else 3.0),
                    monotonic_violation_rate=(
                        0.0 if candidate.family in {"current_hybrid", "hybridpatch_v2"} else None
                    ),
                )
            )
    baselines = tuple(
        BaselineValidationEvidence(
            baseline=baseline,
            cutoff_cycle=cutoff,
            seed=seed,
            validation_mae=100.0,
            validation_rmse=100.0,
            monotonic_violation_rate=0.0 if baseline == "current_hybrid" else None,
        )
        for baseline in ("xgboost", "current_hybrid")
        for cutoff, seed in product((20, 50, 100, 150), (38, 39, 40))
    )
    return candidates, tuple(rows), baselines


def _selection_stages(candidates):
    stage1 = tuple(
        AdvancedValidationEvidence(
            family=candidate.family,
            candidate_id=candidate.candidate_id,
            candidate_config_sha256=candidate.config_sha256,
            stage="selection_stage1",
            cutoff_cycle=100,
            seed=38,
            status="completed",
            validation_mae=100.0,
            validation_rmse=101.0,
            monotonic_violation_rate=(
                0.0 if candidate.family in {"current_hybrid", "hybridpatch_v2"} else None
            ),
        )
        for candidate in candidates
    )
    stage2 = tuple(
        row.model_copy(update={"stage": "selection_stage2", "validation_mae": 99.0})
        for row in stage1
    )
    return stage1, stage2


def test_recheck_builds_hash_bound_manifest_and_provisional_eligibility() -> None:
    candidates, rows, baselines = _recheck()
    stage1, stage2 = _selection_stages(candidates)
    manifest = build_advanced_model_selection_manifest(
        stage1_evidence=stage1,
        stage2_evidence=stage2,
        finalists=candidates,
        recheck_evidence=rows,
        baselines=baselines,
        input_bundle_sha256="a" * 64,
        data_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        normalization_sha256="e" * 64,
        source_commit="f" * 40,
        validation_cell_sha256="1" * 64,
    )

    assert len(manifest.selection.validation_results) == 48
    assert len(manifest.selection.selected_candidates) == 4
    assert len(manifest.selection_trace.stage1_evidence) == 4
    assert len(manifest.selection_trace.stage2_evidence) == 4
    assert len(manifest.selection_trace.recheck_evidence) == 48
    assert len(manifest.baseline_evidence) == 24
    assert manifest.selection.manifest_sha256 == sha256_canonical(
        manifest.selection.model_dump(mode="json", exclude={"manifest_sha256"})
    )
    eligibility = {
        item.family: item.eligible_for_final for item in manifest.provisional_eligibility
    }
    assert eligibility == {"cyclepatch_batlinet": True, "hybridpatch_v2": True}
    dumped = manifest.model_dump(mode="json")
    assert "promoted" not in str(dumped).lower()
    assert "flagship" not in str(dumped).lower()


def test_failed_provisional_gate_retains_selected_candidate_for_final_ablation() -> None:
    candidates, rows, baselines = _recheck(eligibility_good=False)
    stage1, stage2 = _selection_stages(candidates)
    manifest = build_advanced_model_selection_manifest(
        stage1_evidence=stage1,
        stage2_evidence=stage2,
        finalists=candidates,
        recheck_evidence=rows,
        baselines=baselines,
        input_bundle_sha256="a" * 64,
        data_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        normalization_sha256="e" * 64,
        source_commit="f" * 40,
        validation_cell_sha256="1" * 64,
    )
    assert len(manifest.selection.selected_candidates) == 4
    assert all(not item.eligible_for_final for item in manifest.provisional_eligibility)

    with pytest.raises(ValueError, match=r"12|complete|grid"):
        build_advanced_model_selection_manifest(
            stage1_evidence=stage1,
            stage2_evidence=stage2,
            finalists=candidates,
            recheck_evidence=rows[:-1],
            baselines=baselines,
            input_bundle_sha256="a" * 64,
            data_sha256="b" * 64,
            split_sha256="c" * 64,
            feature_sha256="d" * 64,
            normalization_sha256="e" * 64,
            source_commit="f" * 40,
            validation_cell_sha256="1" * 64,
        )


def test_selection_trace_rejects_stage2_or_finalists_not_derived_from_prior_stage() -> None:
    candidates, rows, baselines = _recheck()
    stage1, stage2 = _selection_stages(candidates)
    rogue_stage2 = stage2[0].model_copy(update={"candidate_id": "rogue"})
    with pytest.raises(ValueError, match=r"stage1|subset|survivor"):
        build_advanced_model_selection_manifest(
            stage1_evidence=stage1,
            stage2_evidence=(*stage2[1:], rogue_stage2),
            finalists=candidates,
            recheck_evidence=rows,
            baselines=baselines,
            input_bundle_sha256="a" * 64,
            data_sha256="b" * 64,
            split_sha256="c" * 64,
            feature_sha256="d" * 64,
            normalization_sha256="e" * 64,
            source_commit="f" * 40,
            validation_cell_sha256="1" * 64,
        )

    with pytest.raises(ValueError, match=r"stage2|top|finalist"):
        build_advanced_model_selection_manifest(
            stage1_evidence=stage1,
            stage2_evidence=stage2[:-1],
            finalists=candidates,
            recheck_evidence=rows,
            baselines=baselines,
            input_bundle_sha256="a" * 64,
            data_sha256="b" * 64,
            split_sha256="c" * 64,
            feature_sha256="d" * 64,
            normalization_sha256="e" * 64,
            source_commit="f" * 40,
            validation_cell_sha256="1" * 64,
        )


def test_selection_trace_rejects_stage2_that_drops_a_stage1_survivor() -> None:
    candidates, rows, baselines = _recheck()
    stage1, stage2 = _selection_stages(candidates)
    direct = candidates[0]
    alternative = CyclePatchDirectCandidate(
        candidate_id="direct-survivor",
        learning_rate=1e-4,
        weight_decay=0.01,
        d_model=128,
        layers=2,
        heads=4,
        dropout=0.05,
        huber_delta=1.0,
    )
    eliminated = alternative.model_copy(
        update={"candidate_id": "direct-eliminated", "learning_rate": 1e-3}
    )
    direct_stage1 = next(row for row in stage1 if row.candidate_id == direct.candidate_id)
    stage1 = (
        *stage1,
        direct_stage1.model_copy(
            update={
                "candidate_id": alternative.candidate_id,
                "candidate_config_sha256": alternative.config_sha256,
                "validation_mae": 101.0,
            }
        ),
        direct_stage1.model_copy(
            update={
                "candidate_id": eliminated.candidate_id,
                "candidate_config_sha256": eliminated.config_sha256,
                "validation_mae": 102.0,
            }
        ),
    )

    with pytest.raises(ValueError, match=r"stage1 survivors|exactly|stage2"):
        build_advanced_model_selection_manifest(
            stage1_evidence=stage1,
            stage2_evidence=stage2,
            finalists=candidates,
            recheck_evidence=rows,
            baselines=baselines,
            input_bundle_sha256="a" * 64,
            data_sha256="b" * 64,
            split_sha256="c" * 64,
            feature_sha256="d" * 64,
            normalization_sha256="e" * 64,
            source_commit="f" * 40,
            validation_cell_sha256="1" * 64,
        )


def test_manifest_hash_binds_all_baseline_values() -> None:
    candidates, rows, baselines = _recheck()
    stage1, stage2 = _selection_stages(candidates)
    manifest = build_advanced_model_selection_manifest(
        stage1_evidence=stage1,
        stage2_evidence=stage2,
        finalists=candidates,
        recheck_evidence=rows,
        baselines=baselines,
        input_bundle_sha256="a" * 64,
        data_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        normalization_sha256="e" * 64,
        source_commit="f" * 40,
        validation_cell_sha256="1" * 64,
    )
    tampered = manifest.model_copy(
        update={
            "baseline_evidence": tuple(
                row.model_copy(update={"validation_mae": row.validation_mae + 0.001})
                if row.baseline == "xgboost" and row.cutoff_cycle == 20 and row.seed == 38
                else row
                for row in manifest.baseline_evidence
            )
        }
    )
    with pytest.raises(ValueError, match="manifest_sha256"):
        type(manifest).model_validate(tampered.model_dump(mode="python"))


def test_selection_trace_retains_failed_nonselected_finalist_evidence() -> None:
    candidates, rows, baselines = _recheck()
    direct = candidates[0]
    direct_alternative = CyclePatchDirectCandidate(
        candidate_id="direct-alternative",
        learning_rate=1e-4,
        weight_decay=0.01,
        d_model=128,
        layers=2,
        heads=4,
        dropout=0.05,
        huber_delta=1.0,
    )
    direct_eliminated = CyclePatchDirectCandidate(
        candidate_id="direct-eliminated",
        learning_rate=1e-3,
        weight_decay=0.01,
        d_model=128,
        layers=2,
        heads=4,
        dropout=0.05,
        huber_delta=1.0,
    )
    stage1, stage2 = _selection_stages(candidates)
    direct_stage1 = next(row for row in stage1 if row.candidate_id == direct.candidate_id)
    stage1 = (
        *stage1,
        direct_stage1.model_copy(
            update={
                "candidate_id": direct_alternative.candidate_id,
                "candidate_config_sha256": direct_alternative.config_sha256,
                "validation_mae": 101.0,
            }
        ),
        direct_stage1.model_copy(
            update={
                "candidate_id": direct_eliminated.candidate_id,
                "candidate_config_sha256": direct_eliminated.config_sha256,
                "validation_mae": 102.0,
            }
        ),
    )
    direct_stage2 = next(row for row in stage2 if row.candidate_id == direct.candidate_id)
    stage2 = (
        *stage2,
        direct_stage2.model_copy(
            update={
                "candidate_id": direct_alternative.candidate_id,
                "candidate_config_sha256": direct_alternative.config_sha256,
                "validation_mae": 110.0,
            }
        ),
    )
    alternative_rows = tuple(
        AdvancedValidationEvidence(
            family=direct_alternative.family,
            candidate_id=direct_alternative.candidate_id,
            candidate_config_sha256=direct_alternative.config_sha256,
            stage="selection_recheck",
            cutoff_cycle=cutoff,
            seed=seed,
            status="failed" if (cutoff, seed) == (20, 38) else "completed",
            validation_mae=None if (cutoff, seed) == (20, 38) else 200.0,
            validation_rmse=None if (cutoff, seed) == (20, 38) else 201.0,
        )
        for cutoff, seed in product((20, 50, 100, 150), (38, 39, 40))
    )

    manifest = build_advanced_model_selection_manifest(
        stage1_evidence=stage1,
        stage2_evidence=stage2,
        finalists=(*candidates, direct_alternative),
        recheck_evidence=(*rows, *alternative_rows),
        baselines=baselines,
        input_bundle_sha256="a" * 64,
        data_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        normalization_sha256="e" * 64,
        source_commit="f" * 40,
        validation_cell_sha256="1" * 64,
    )

    assert len(manifest.selection_trace.stage1_evidence) == 6
    assert len(manifest.selection_trace.stage2_evidence) == 5
    assert len(manifest.selection_trace.recheck_evidence) == 60
    retained = [
        row
        for row in manifest.selection_trace.recheck_evidence
        if row.candidate_id == direct_alternative.candidate_id
    ]
    assert len(retained) == 12
    assert any(row.status == "failed" for row in retained)
    assert direct_alternative.candidate_id not in {
        candidate.candidate_id for candidate in manifest.selection.selected_candidates
    }
