from __future__ import annotations

from copy import deepcopy

from workbench.presenters import (
    UNAVAILABLE_LABEL,
    EvidenceSection,
    build_audit_section,
    build_decision_section,
    build_lifetime_section,
    build_quality_section,
)


def _field_values(section: EvidenceSection) -> dict[str, object]:
    return {field.key: field.value for field in section.fields}


def _availability(section: EvidenceSection) -> dict[str, bool]:
    return {field.key: field.available for field in section.fields}


def test_quality_section_reads_signed_values_including_zero_and_false() -> None:
    payload = {
        "result_id": "quality-result",
        "tool_name": "validate_battery_data",
        "values": {
            "dataset_id": "MATR",
            "blocked": False,
            "quality_score": 0,
            "issue_count": 0,
            "issues": [],
        },
        "warnings": [],
    }

    section = build_quality_section(payload)

    assert _field_values(section) == {
        "dataset_id": "MATR",
        "blocked": False,
        "quality_score": 0,
        "issue_count": 0,
        "issues": [],
    }
    assert all(_availability(section).values())


def test_lifetime_section_never_derives_missing_interval_or_rul() -> None:
    prediction = {
        "values": {
            "artifact": {
                "life_prediction": {
                    "predicted_eol_cycle": 812.5,
                    "cutoff_cycle": 100,
                }
            }
        }
    }
    interval = {
        "values": {
            "artifact": {
                "prediction_interval": {
                    "lower_eol_cycle": 744.25,
                }
            }
        }
    }

    section = build_lifetime_section(prediction, interval)
    values = _field_values(section)
    available = _availability(section)

    assert values["predicted_eol_cycle"] == 812.5
    assert values["lower_eol_cycle"] == 744.25
    assert values["upper_eol_cycle"] is None
    assert values["derived_rul_cycle"] is None
    assert available["upper_eol_cycle"] is False
    assert available["derived_rul_cycle"] is False
    assert section.field("upper_eol_cycle").display_value() == UNAVAILABLE_LABEL


def test_decision_section_reads_policy_values_without_reapplying_policy() -> None:
    payload = {
        "values": {
            "decision": "recheck",
            "required_eol_cycle": 900.0,
            "reason_codes": ["INTERVAL_CROSSES_THRESHOLD"],
            "target_domain_calibrated": False,
            "policy_id": "policy-A",
            "policy_version": "v1",
        }
    }

    section = build_decision_section(payload)

    assert _field_values(section) == {
        "decision": "recheck",
        "required_eol_cycle": 900.0,
        "reason_codes": ["INTERVAL_CROSSES_THRESHOLD"],
        "target_domain_calibrated": False,
        "policy_id": "policy-A",
        "policy_version": "v1",
    }


def test_audit_section_preserves_evidence_and_does_not_mutate_payload() -> None:
    payload = {
        "result_id": "result-A",
        "tool_name": "predict_cycle_life",
        "tool_version": "tool-v1",
        "model_version": "model-v1",
        "data_version": "data-v1",
        "feature_version": "feature-v1",
        "input_hash": "a" * 64,
        "created_at": "2026-07-15T00:00:00Z",
        "warnings": ["server-warning"],
        "provenance": [{"source_id": "source-A", "sha256": "b" * 64}],
    }
    original = deepcopy(payload)

    section = build_audit_section(payload)

    assert _field_values(section) == payload
    assert payload == original


def test_missing_backend_fields_are_explicitly_unavailable() -> None:
    section = build_quality_section({"values": {}})

    assert all(field.available is False for field in section.fields)
    assert all(field.display_value() == UNAVAILABLE_LABEL for field in section.fields)
