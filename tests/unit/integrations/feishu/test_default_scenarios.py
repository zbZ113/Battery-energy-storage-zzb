from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.data.schemas import CycleRecord
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.default_scenarios import (
    ReviewedDefaultScenarioProfile,
    ReviewedDefaultScenarioRegistry,
    load_reviewed_default_scenario_registry,
)
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.tools.early_cycle_features import VerifiedEarlyCycleBatch

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def _batch() -> VerifiedEarlyCycleBatch:
    records = (
        CycleRecord(
            dataset_id="MATR",
            cell_id="MATR_b3c34",
            cycle_index=1,
            sample_index=0,
            time_s=0.0,
            voltage_v=3.6,
            current_a=1.0,
            temperature_c=25.0,
            charge_capacity_ah=1.1,
            discharge_capacity_ah=1.05,
            internal_resistance_ohm=0.02,
            diagnostic=True,
            valid=True,
        ),
        CycleRecord(
            dataset_id="MATR",
            cell_id="MATR_b3c34",
            cycle_index=50,
            sample_index=0,
            time_s=0.0,
            voltage_v=3.5,
            current_a=1.0,
            temperature_c=25.0,
            charge_capacity_ah=1.02,
            discharge_capacity_ah=1.0,
            internal_resistance_ohm=0.03,
            diagnostic=True,
            valid=True,
        ),
    )
    return VerifiedEarlyCycleBatch(
        record_batch_id="canonical-csv-" + "a" * 64,
        metadata=CellMetadata(
            dataset_id="MATR",
            cell_id="MATR_b3c34",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.1,
            source_uri="upload://competition-demo/MATR_b3c34-cutoff-50.csv",
            source_sha256="b" * 64,
            schema_version="cycle-record-v1",
        ),
        records=records,
        feature_config=EarlyCycleFeatureConfig(
            cutoff_cycle=50,
            feature_version="cyclepatch-multichannel-v1",
        ),
        data_version="matr-three-batch-v1",
        split_version="matr-cell-split-v1",
        source_manifest_hash="b" * 64,
        provenance=(
            ProvenanceRecord(
                source_id="MATR_b3c34-cutoff-50",
                source_kind=SourceKind.OBSERVED,
                uri="upload://competition-demo/MATR_b3c34-cutoff-50.csv",
                sha256="b" * 64,
                description="Reviewed MATR early-cycle upload.",
                created_at=NOW,
            ),
        ),
    )


def _profile_payload() -> dict[str, object]:
    def scenario(
        scenario_id: str,
        *,
        temperature_c: float,
        charge_c_rate: float = 0.5,
        discharge_c_rate: float = 0.5,
    ) -> dict[str, object]:
        return {
            "scenario_id": scenario_id,
            "scenario_version": f"{scenario_id}-v1",
            "horizon_years": 25,
            "eol_threshold": 0.8,
            "segments": [
                {
                    "segment_id": "years-1-25",
                    "start_year": 0,
                    "end_year": 25,
                    "temperature_c": temperature_c,
                    "charge_c_rate": charge_c_rate,
                    "discharge_c_rate": discharge_c_rate,
                    "soc_lower_bound": 0.1,
                    "soc_upper_bound": 0.9,
                    "dod": 0.8,
                    "equivalent_full_cycles_per_year": 300.0,
                    "rest_duration_hours": 1.0,
                }
            ],
        }

    return {
        "profile_id": "matr-b3c34-prismatic-reference",
        "profile_version": "matr-b3c34-prismatic-reference-v1",
        "review_status": "APPROVED",
        "task_type": "compare_operation_scenarios",
        "dataset_id": "MATR",
        "cell_id": "MATR_b3c34",
        "data_version": "matr-three-batch-v1",
        "split_version": "matr-cell-split-v1",
        "feature_version": "cyclepatch-multichannel-v1",
        "allowed_cutoff_cycles": [50],
        "allowed_source_sha256s": ["b" * 64],
        "chemistry": "LFP/graphite",
        "nominal_capacity_ah": 1.1,
        "source_cell_format": "cylindrical",
        "reference_cell_format": "prismatic",
        "route_id": "blast-lite-lfp-gr-250ah-prismatic-2019-v1",
        "trusted_reference_use": True,
        "reference_use_reason_code": "MATR_TO_LARGE_FORMAT_LFP_REFERENCE_ONLY",
        "initial_state_policy": "BOL_ONLY",
        "baseline": scenario("baseline", temperature_c=25.0),
        "comparisons": [
            scenario("temperature-35c", temperature_c=35.0),
            scenario(
                "higher-rate",
                temperature_c=25.0,
                charge_c_rate=0.65,
                discharge_c_rate=1.0,
            ),
        ],
    }


def _profile() -> ReviewedDefaultScenarioProfile:
    return ReviewedDefaultScenarioProfile.model_validate(_profile_payload())


def test_registry_resolves_exact_reviewed_profile_and_builds_context() -> None:
    profile = _profile()
    registry = ReviewedDefaultScenarioRegistry((profile,))

    resolved = registry.resolve(_batch())
    context = registry.create_context_template(
        batch=_batch(),
        profile=resolved,
        source_job_id="3a3c972b-a23e-42c3-af76-e39038806f13",
    )

    assert resolved.profile_sha256 == profile.profile_sha256
    assert context.task is FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS
    assert context.profile_sha256 == profile.profile_sha256
    assert context.verified_context.trusted_reference_use is True
    assert context.verified_context.cell.nominal_capacity_ah == 1.1
    assert context.analysis_input.route_id == profile.route_id
    assert context.analysis_input.baseline.scenario_id == "baseline"
    assert len(context.analysis_input.comparisons) == 2
    assert context.analysis_input.run_id == context.scenario_context_id
    UUID(context.scenario_context_id)


def test_registry_returns_none_without_an_exact_profile() -> None:
    profile = _profile().model_copy(update={"data_version": "other-data-v1"})

    assert ReviewedDefaultScenarioRegistry((profile,)).resolve(_batch()) is None


def test_registry_rejects_duplicate_matches_or_unapproved_reference_use() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="unique"):
        ReviewedDefaultScenarioRegistry((profile, profile))

    unsafe = _profile_payload()
    unsafe["trusted_reference_use"] = False
    with pytest.raises(ValueError, match="reference use"):
        ReviewedDefaultScenarioProfile.model_validate(unsafe)


def test_registry_rejects_partially_overlapping_match_sets() -> None:
    first = _profile()
    second = first.model_copy(
        update={
            "profile_id": "overlapping-profile",
            "profile_version": "overlapping-profile-v1",
            "allowed_cutoff_cycles": (50, 100),
            "allowed_source_sha256s": ("b" * 64, "c" * 64),
        }
    )

    with pytest.raises(ValueError, match="overlap"):
        ReviewedDefaultScenarioRegistry((first, second))


def test_registry_rejects_duplicate_profile_identity_across_batches() -> None:
    first = _profile()
    second = first.model_copy(
        update={
            "dataset_id": "OTHER",
            "cell_id": "other-cell",
        }
    )

    with pytest.raises(ValueError, match="profile identity"):
        ReviewedDefaultScenarioRegistry((first, second))


@pytest.mark.parametrize(
    ("profile_id", "profile_version"),
    (
        ("x" * 200, "profile-v1"),
        ("默认工况", "profile-v1"),
    ),
)
def test_profile_identity_must_form_a_safe_created_by_reference(
    profile_id: str,
    profile_version: str,
) -> None:
    payload = _profile_payload()
    payload["profile_id"] = profile_id
    payload["profile_version"] = profile_version

    with pytest.raises(ValueError, match="created-by reference"):
        ReviewedDefaultScenarioProfile.model_validate(payload)


def test_profile_loader_is_strict_and_retains_file_sha(tmp_path: Path) -> None:
    registry_path = tmp_path / "default-scenarios.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "feishu-default-scenario-registry-v1",
                "profiles": [_profile_payload()],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_reviewed_default_scenario_registry(registry_path)

    assert len(loaded.profiles) == 1
    assert len(loaded.file_sha256) == 64
    assert loaded.profiles[0].profile_sha256 == _profile().profile_sha256

    registry_path.write_text('{"profiles":[],"profiles":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_reviewed_default_scenario_registry(registry_path)


@pytest.mark.parametrize("approval", (1, "yes"))
def test_profile_loader_rejects_coerced_reference_approval(
    tmp_path: Path,
    approval: object,
) -> None:
    registry_path = tmp_path / "default-scenarios.json"
    profile = _profile_payload()
    profile["trusted_reference_use"] = approval
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "feishu-default-scenario-registry-v1",
                "profiles": [profile],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_reviewed_default_scenario_registry(registry_path)


def test_profile_loader_rejects_coerced_scenario_numbers(tmp_path: Path) -> None:
    registry_path = tmp_path / "default-scenarios.json"
    profile = _profile_payload()
    baseline = cast(dict[str, Any], profile["baseline"])
    segments = cast(list[dict[str, Any]], baseline["segments"])
    segments[0]["temperature_c"] = "25.0"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "feishu-default-scenario-registry-v1",
                "profiles": [profile],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_reviewed_default_scenario_registry(registry_path)

    registry_path.write_text(
        '{"schema_version":"feishu-default-scenario-registry-v1",'
        '"profiles":[]}',
        encoding="utf-16",
    )
    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        load_reviewed_default_scenario_registry(registry_path)
