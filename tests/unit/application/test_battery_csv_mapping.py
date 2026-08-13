from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pytest

from quanxin_life.application.battery_csv_mapping import (
    BatteryCsvColumnMapping,
    BatteryCsvMappingError,
    BatteryCsvMappingProfile,
    ReviewedBatteryCsvNormalizer,
    load_battery_csv_mapping_profile,
    map_battery_csv,
)
from quanxin_life.application.ingestion import CANONICAL_CYCLE_CSV_FIELDS


def _profile(*, aliases: bool = False) -> BatteryCsvMappingProfile:
    units = {
        "dataset_id": "text",
        "cell_id": "text",
        "cycle_index": "count",
        "sample_index": "count",
        "time_s": "s",
        "voltage_v": "V",
        "current_a": "A",
        "temperature_c": "degC",
        "charge_capacity_ah": "Ah",
        "discharge_capacity_ah": "Ah",
        "internal_resistance_ohm": "ohm",
        "diagnostic": "boolean",
        "valid": "boolean",
    }
    alias_headers = {
        "dataset_id": "Dataset",
        "cell_id": "Cell",
        "cycle_index": "Cycle",
        "sample_index": "Sample",
        "time_s": "Time_ms",
        "voltage_v": "Voltage_mV",
        "current_a": "Current_mA",
        "temperature_c": "Temperature_C",
        "charge_capacity_ah": "Charge_mAh",
        "discharge_capacity_ah": "Discharge_mAh",
        "internal_resistance_ohm": "Resistance_mOhm",
        "diagnostic": "IsDiagnostic",
        "valid": "IsValid",
    }
    source_units = {
        **units,
        "time_s": "ms",
        "voltage_v": "mV",
        "current_a": "mA",
        "charge_capacity_ah": "mAh",
        "discharge_capacity_ah": "mAh",
        "internal_resistance_ohm": "mohm",
    }
    scaled = {
        "time_s",
        "voltage_v",
        "current_a",
        "charge_capacity_ah",
        "discharge_capacity_ah",
        "internal_resistance_ohm",
    }
    return BatteryCsvMappingProfile(
        profile_id="test-profile",
        version="test-profile-v1",
        review_status="APPROVED",
        mappings=tuple(
            BatteryCsvColumnMapping(
                source_header=alias_headers[field] if aliases else field,
                target_field=field,
                source_unit=source_units[field] if aliases else units[field],
                canonical_unit=units[field],
                scale="0.001" if aliases and field in scaled else "1",
                offset="0",
                sign=1,
            )
            for field in CANONICAL_CYCLE_CSV_FIELDS
        ),
    )


def _canonical_payload() -> bytes:
    return (
        ",".join(CANONICAL_CYCLE_CSV_FIELDS)
        + "\nMATR,MATR_b3c34,1,0,1,3.6,1.2,25,1.1,1.0,0.02,true,true\n"
    ).encode()


def _alias_payload() -> bytes:
    return (
        b"Dataset,Cell,Cycle,Sample,Time_ms,Voltage_mV,Current_mA,Temperature_C,"
        b"Charge_mAh,Discharge_mAh,Resistance_mOhm,IsDiagnostic,IsValid\n"
        b"MATR,MATR_b3c34,1,0,1000,3600,1200,25,1100,1000,20,true,true\n"
    )


def test_canonical_profile_is_deterministic_identity_mapping() -> None:
    payload = _canonical_payload()

    result = map_battery_csv(payload, profile=_profile())

    assert result.canonical_payload == payload
    assert result.raw_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.canonical_sha256 == result.raw_sha256
    assert result.profile_version == "test-profile-v1"
    assert len(result.profile_sha256) == 64
    assert tuple(item.target_field for item in result.column_evidence) == (
        CANONICAL_CYCLE_CSV_FIELDS
    )


def test_reviewed_aliases_and_explicit_units_are_converted_without_guessing() -> None:
    raw = _alias_payload()

    result = map_battery_csv(raw, profile=_profile(aliases=True))

    actual = next(csv.DictReader(io.StringIO(result.canonical_payload.decode())))
    expected = next(csv.DictReader(io.StringIO(_canonical_payload().decode())))
    assert actual.keys() == expected.keys()
    for field in ("dataset_id", "cell_id", "diagnostic", "valid"):
        assert actual[field] == expected[field]
    for field in set(CANONICAL_CYCLE_CSV_FIELDS) - {
        "dataset_id",
        "cell_id",
        "diagnostic",
        "valid",
    }:
        assert float(actual[field]) == float(expected[field])
    assert result.raw_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.canonical_sha256 == hashlib.sha256(result.canonical_payload).hexdigest()
    assert result.raw_sha256 != result.canonical_sha256
    time_mapping = next(
        item for item in result.column_evidence if item.target_field == "time_s"
    )
    assert time_mapping.source_header == "Time_ms"
    assert time_mapping.source_unit == "ms"
    assert time_mapping.canonical_unit == "s"
    assert time_mapping.scale == "0.001"


def test_multiple_source_columns_for_one_target_are_rejected_as_ambiguous() -> None:
    profile = _profile()
    with pytest.raises(ValueError, match="duplicate target field"):
        BatteryCsvMappingProfile(
            profile_id=profile.profile_id,
            version=profile.version,
            review_status="APPROVED",
            mappings=(
                *profile.mappings,
                BatteryCsvColumnMapping(
                source_header="Time_ms",
                target_field="time_s",
                    source_unit="ms",
                    canonical_unit="s",
                    scale="0.001",
                    offset="0",
                    sign=1,
                ),
            ),
        )


def test_missing_required_and_unknown_columns_fail_closed() -> None:
    missing = _canonical_payload().replace(b"current_a,", b"").replace(b"1.2,25", b"25")
    with pytest.raises(ValueError, match=r"missing source columns.*current_a"):
        map_battery_csv(missing, profile=_profile())

    unknown = _canonical_payload().replace(b"valid\n", b"valid,Unreviewed\n").replace(
        b"true,true\n", b"true,true,value\n"
    )
    with pytest.raises(ValueError, match=r"unknown source columns.*Unreviewed"):
        map_battery_csv(unknown, profile=_profile())


@pytest.mark.parametrize("unsafe", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_numeric_values_are_rejected(unsafe: str) -> None:
    payload = _canonical_payload().replace(b"3.6,1.2", f"{unsafe},1.2".encode())

    with pytest.raises(ValueError, match=r"voltage_v.*finite decimal"):
        map_battery_csv(payload, profile=_profile())


def test_extreme_finite_decimal_is_rejected_before_fixed_point_expansion() -> None:
    payload = _canonical_payload().replace(b"3.6,1.2", b"1e999999999,1.2")

    with pytest.raises(ValueError, match="safe range"):
        map_battery_csv(payload, profile=_profile())


def test_unit_conversion_never_silently_rounds_a_decimal() -> None:
    raw = _alias_payload().replace(
        b"1000,3600",
        b"123456789012345678901234567890,3600",
    )

    result = map_battery_csv(raw, profile=_profile(aliases=True))

    row = next(csv.DictReader(io.StringIO(result.canonical_payload.decode())))
    assert row["time_s"] == "123456789012345678901234567.89"


def test_reviewed_current_sign_inversion_is_explicit_and_audited() -> None:
    profile = _profile(aliases=True)
    payload = profile.model_dump(mode="json")
    for item in payload["mappings"]:
        if item["target_field"] == "current_a":
            item["sign"] = -1
    inverted = BatteryCsvMappingProfile.model_validate(payload)

    result = map_battery_csv(_alias_payload(), profile=inverted)

    row = next(csv.DictReader(io.StringIO(result.canonical_payload.decode())))
    assert row["current_a"] == "-1.2"
    evidence = next(
        item for item in result.column_evidence if item.target_field == "current_a"
    )
    assert evidence.sign == -1


def test_identifier_formula_content_is_not_propagated() -> None:
    payload = _canonical_payload().replace(b"MATR_b3c34", b"=cmd")

    with pytest.raises(ValueError, match="cell_id contains spreadsheet formula"):
        map_battery_csv(payload, profile=_profile())


def test_repository_profile_is_strict_versioned_and_replayable() -> None:
    profile = load_battery_csv_mapping_profile(
        Path("configs/data_layouts/feishu_battery_csv_v1.json")
    )

    assert profile.profile_id == "feishu-reviewed-cycle-csv"
    assert profile.version == "feishu-reviewed-cycle-csv-v1"
    assert profile.review_status == "APPROVED"
    assert len(profile.profile_sha256) == 64
    first = map_battery_csv(_alias_payload(), profile=profile)
    second = map_battery_csv(_alias_payload(), profile=profile)
    assert first == second


def test_normalizer_rejects_profile_without_explicit_approval() -> None:
    payload = _profile(aliases=True).model_dump(mode="json")
    payload["review_status"] = "CANDIDATE"
    candidate = BatteryCsvMappingProfile.model_validate(payload)

    with pytest.raises(ValueError, match="explicitly approved"):
        ReviewedBatteryCsvNormalizer((candidate,))


@pytest.mark.parametrize("missing_rule", ("scale", "offset", "sign"))
def test_mapping_profile_requires_explicit_conversion_and_sign_rules(
    missing_rule: str,
) -> None:
    payload = _profile(aliases=True).model_dump(mode="json")
    del payload["mappings"][6][missing_rule]

    with pytest.raises(ValueError, match=missing_rule):
        BatteryCsvMappingProfile.model_validate(payload)


def test_reviewed_normalizer_selects_exact_header_and_keeps_canonical_identity() -> None:
    normalizer = ReviewedBatteryCsvNormalizer((_profile(aliases=True),))

    mapped = normalizer.normalize(_alias_payload())
    identity = normalizer.normalize(_canonical_payload())

    assert mapped.profile_id == "test-profile"
    assert identity.profile_id == "canonical-cycle-record"
    assert identity.canonical_payload == _canonical_payload()
    assert identity.raw_sha256 == identity.canonical_sha256


def test_reviewed_normalizer_rejects_unreviewed_layout_without_fuzzy_matching() -> None:
    normalizer = ReviewedBatteryCsvNormalizer((_profile(aliases=True),))

    with pytest.raises(
        BatteryCsvMappingError,
        match="source CSV layout is not reviewed",
    ) as captured:
        normalizer.normalize(b"Cell Name,Cycle Number\nMATR_b3c34,1\n")

    assert captured.value.reason_code == "UNREVIEWED_LAYOUT"
    assert captured.value.details.model_dump(mode="json") == {
        "conflicts": [],
        "missing_fields": list(CANONICAL_CYCLE_CSV_FIELDS),
        "unknown_fields": ["Cell Name", "Cycle Number"],
        "unit_required": [],
        "value_errors": [],
    }


def test_reviewed_normalizer_accepts_reviewed_columns_in_a_different_order() -> None:
    profile = _profile(aliases=True)
    normalizer = ReviewedBatteryCsvNormalizer((profile,))
    header = [item.source_header for item in profile.mappings]
    row = _alias_payload().decode().splitlines()[1].split(",")
    reordered = (",".join(reversed(header)) + "\n" + ",".join(reversed(row)) + "\n").encode()

    result = normalizer.normalize(reordered)

    assert result.profile_id == profile.profile_id
    assert result.row_count == 1


def test_unreviewed_canonical_extension_reports_only_the_extra_column() -> None:
    normalizer = ReviewedBatteryCsvNormalizer((_profile(aliases=True),))
    payload = _canonical_payload().replace(b"valid\n", b"valid,extra\n").replace(
        b"true,true\n", b"true,true,value\n"
    )

    with pytest.raises(BatteryCsvMappingError) as captured:
        normalizer.normalize(payload)

    assert captured.value.details.model_dump(mode="json") == {
        "conflicts": [],
        "missing_fields": [],
        "unknown_fields": ["extra"],
        "unit_required": [],
        "value_errors": [],
    }


def test_ambiguous_profiles_report_explicit_current_sign_conflict() -> None:
    positive = _profile(aliases=True)
    negative_payload = positive.model_dump(mode="json")
    negative_payload["profile_id"] = "negative-current-profile"
    negative_payload["version"] = "negative-current-profile-v1"
    for item in negative_payload["mappings"]:
        if item["target_field"] == "current_a":
            item["sign"] = -1
    negative = BatteryCsvMappingProfile.model_validate(negative_payload)
    normalizer = ReviewedBatteryCsvNormalizer((positive, negative))

    with pytest.raises(BatteryCsvMappingError) as captured:
        normalizer.normalize(_alias_payload())

    assert captured.value.reason_code == "AMBIGUOUS_LAYOUT"
    assert captured.value.details.model_dump(mode="json") == {
        "conflicts": [
                {
                    "source_header": "Current_mA",
                    "target_field": "current_a",
                    "issue_code": "AMBIGUOUS_SIGN",
                }
            ],
        "missing_fields": [],
        "unknown_fields": [],
        "unit_required": [],
        "value_errors": [],
    }


def test_value_validation_rejection_identifies_field_without_persisting_value() -> None:
    normalizer = ReviewedBatteryCsvNormalizer((_profile(aliases=True),))
    payload = _alias_payload().replace(b"3600,1200", b"not-a-number,1200")

    with pytest.raises(BatteryCsvMappingError) as captured:
        normalizer.normalize(payload)

    details = captured.value.details.model_dump(mode="json")
    assert details == {
        "conflicts": [],
        "missing_fields": [],
        "unknown_fields": [],
        "unit_required": [],
        "value_errors": [
            {
                "source_header": "Voltage_mV",
                "target_field": "voltage_v",
                "reason_code": "INVALID_DECIMAL",
            }
        ],
    }
    assert "not-a-number" not in str(details)
