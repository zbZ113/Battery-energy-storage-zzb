"""Deterministic reviewed-profile mapping into the canonical cycle CSV contract.

This module only recognizes declared headers and applies declared Decimal unit
conversions. It never asks an LLM to inspect or rewrite observation values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from decimal import Decimal, DecimalException, Inexact, InvalidOperation, Rounded, localcontext
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from quanxin_life.application.ingestion import CANONICAL_CYCLE_CSV_FIELDS
from quanxin_life.core.schemas import ContractModel

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MetadataIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
MetadataDescription = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000),
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BATTERY_CSV_METADATA_ENVELOPE_V1 = "# quanxin_life:battery_csv_metadata:v1"
_BATTERY_CSV_METADATA_ENVELOPE_PREFIX = b"# quanxin_life:battery_csv_metadata:"
_SELF_DESCRIBED_SUPPORTED_CUTOFFS = (20, 50, 100, 150)
BatteryCsvMappingIssueCode = Literal[
    "AMBIGUOUS_CONVERSION",
    "AMBIGUOUS_DEFAULT",
    "AMBIGUOUS_OFFSET",
    "AMBIGUOUS_SCALE",
    "AMBIGUOUS_SIGN",
    "AMBIGUOUS_TARGET",
    "AMBIGUOUS_UNIT",
]
BatteryCsvValueReasonCode = Literal[
    "EMPTY_IDENTIFIER",
    "EXPONENT_OUT_OF_RANGE",
    "FORMULA_CONTENT",
    "INVALID_BOOLEAN",
    "INVALID_DECIMAL",
    "NONFINITE_DECIMAL",
    "NONINTEGER_VALUE",
    "OUTPUT_TOO_LONG",
    "PRECISION_LOSS",
]

_CANONICAL_UNITS = {
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
_CONVERSIONS = {
    ("text", "text"): (Decimal("1"), Decimal("0")),
    ("count", "count"): (Decimal("1"), Decimal("0")),
    ("s", "s"): (Decimal("1"), Decimal("0")),
    ("ms", "s"): (Decimal("0.001"), Decimal("0")),
    ("V", "V"): (Decimal("1"), Decimal("0")),
    ("mV", "V"): (Decimal("0.001"), Decimal("0")),
    ("A", "A"): (Decimal("1"), Decimal("0")),
    ("mA", "A"): (Decimal("0.001"), Decimal("0")),
    ("degC", "degC"): (Decimal("1"), Decimal("0")),
    ("K", "degC"): (Decimal("1"), Decimal("-273.15")),
    ("Ah", "Ah"): (Decimal("1"), Decimal("0")),
    ("mAh", "Ah"): (Decimal("0.001"), Decimal("0")),
    ("ohm", "ohm"): (Decimal("1"), Decimal("0")),
    ("mohm", "ohm"): (Decimal("0.001"), Decimal("0")),
    ("boolean", "boolean"): (Decimal("1"), Decimal("0")),
}
_INTEGER_FIELDS = {"cycle_index", "sample_index"}
_BOOLEAN_FIELDS = {"diagnostic", "valid"}
_IDENTIFIER_FIELDS = {"dataset_id", "cell_id"}
_OPTIONAL_MEASUREMENTS = {
    "temperature_c",
    "charge_capacity_ah",
    "discharge_capacity_ah",
    "internal_resistance_ohm",
}
_NUMERIC_FIELDS = set(CANONICAL_CYCLE_CSV_FIELDS) - (
    _IDENTIFIER_FIELDS | _BOOLEAN_FIELDS
)


def _decimal(value: str, *, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} must be a finite decimal")
    if parsed != 0 and (parsed.adjusted() > 100 or parsed.adjusted() < -100):
        raise ValueError(f"{label} exponent exceeds the safe range")
    return parsed


class BatteryCsvColumnMapping(ContractModel):
    """One reviewed source-to-canonical column rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_header: NonBlank
    target_field: NonBlank
    source_unit: NonBlank
    canonical_unit: NonBlank
    scale: str
    offset: str
    sign: Literal[-1, 1]

    @model_validator(mode="after")
    def validate_declared_conversion(self) -> BatteryCsvColumnMapping:
        expected_unit = _CANONICAL_UNITS.get(self.target_field)
        if expected_unit is None:
            raise ValueError(f"unknown canonical target field: {self.target_field}")
        if self.canonical_unit != expected_unit:
            raise ValueError("canonical unit does not match the target field")
        conversion = _CONVERSIONS.get((self.source_unit, self.canonical_unit))
        if conversion is None:
            raise ValueError("unknown or ambiguous unit conversion")
        scale = _decimal(self.scale, label="mapping scale")
        offset = _decimal(self.offset, label="mapping offset")
        if (scale, offset) != conversion:
            raise ValueError("declared affine values do not match the unit conversion")
        if self.sign == -1 and self.target_field != "current_a":
            raise ValueError("sign inversion is only permitted for current_a")
        return self


class BatteryCsvStructuralDefault(ContractModel):
    """Reviewed safe default for a structural or absent optional output field."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_field: NonBlank
    value: str

    @model_validator(mode="after")
    def validate_safe_default(self) -> BatteryCsvStructuralDefault:
        if self.target_field in _OPTIONAL_MEASUREMENTS and self.value == "":
            return self
        if self.target_field == "valid" and self.value == "true":
            return self
        raise ValueError("only empty optional measurements or valid=true may be defaulted")


class BatteryCsvMappingProfile(ContractModel):
    """Immutable, complete reviewed mapping profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: NonBlank
    version: NonBlank
    review_status: Literal["CANDIDATE", "APPROVED"]
    mappings: tuple[BatteryCsvColumnMapping, ...] = Field(min_length=1)
    defaults: tuple[BatteryCsvStructuralDefault, ...] = ()

    @model_validator(mode="after")
    def validate_complete_unambiguous_profile(self) -> BatteryCsvMappingProfile:
        source_headers = [item.source_header for item in self.mappings]
        if len(source_headers) != len(set(source_headers)):
            raise ValueError("profile contains a duplicate source header")
        targets = [item.target_field for item in self.mappings]
        targets.extend(item.target_field for item in self.defaults)
        if len(targets) != len(set(targets)):
            raise ValueError("profile contains a duplicate target field")
        missing = set(CANONICAL_CYCLE_CSV_FIELDS) - set(targets)
        extra = set(targets) - set(CANONICAL_CYCLE_CSV_FIELDS)
        if missing:
            raise ValueError(f"profile has missing canonical output fields: {sorted(missing)}")
        if extra:
            raise ValueError(f"profile has unknown canonical output fields: {sorted(extra)}")
        return self

    @property
    def profile_sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class BatteryCsvColumnEvidence(ContractModel):
    """Detached evidence for one canonical output column."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_field: NonBlank
    source_header: str | None
    source_unit: str | None
    canonical_unit: str | None
    scale: str | None
    offset: str | None
    sign: Literal[-1, 1] | None
    default_value: str | None


class BatteryCsvFileMetadata(ContractModel):
    """Strict cell-level metadata carried by a self-describing CSV envelope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["battery-csv-metadata-v1"]
    cell_id: MetadataIdentifier
    chemistry: MetadataIdentifier
    nominal_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    reference_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    cell_format: Literal["cylindrical", "prismatic"]
    eol_threshold: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    protocol_id: MetadataIdentifier | None = None
    protocol_description: MetadataDescription | None = None


class BatteryCsvMappingResult(ContractModel):
    """Canonical bytes and hashes produced by one deterministic mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    canonical_payload: bytes
    raw_sha256: Sha256
    canonical_sha256: Sha256
    profile_id: NonBlank
    profile_version: NonBlank
    profile_sha256: Sha256
    row_count: int = Field(gt=0)
    column_evidence: tuple[BatteryCsvColumnEvidence, ...]
    metadata: BatteryCsvFileMetadata | None = None


class BatteryCsvMappingConflictEvidence(ContractModel):
    """Sanitized description of one profile-selection conflict."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_header: NonBlank
    target_field: NonBlank
    issue_code: BatteryCsvMappingIssueCode


class BatteryCsvValueErrorEvidence(ContractModel):
    """Sanitized field reference for one rejected observation value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_header: NonBlank
    target_field: NonBlank
    reason_code: BatteryCsvValueReasonCode


class BatteryCsvMappingSuccessEvidence(ContractModel):
    """Metadata-only evidence for one accepted mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_sha256: Sha256
    canonical_sha256: Sha256
    profile_id: NonBlank
    profile_version: NonBlank
    profile_sha256: Sha256
    column_evidence: tuple[BatteryCsvColumnEvidence, ...]
    battery_csv_metadata: BatteryCsvFileMetadata | None = None


class BatteryCsvMappingRejectionEvidence(ContractModel):
    """Metadata-only evidence for one rejected mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_sha256: Sha256
    reason_code: Literal[
        "AMBIGUOUS_LAYOUT",
        "INVALID_HEADER",
        "INVALID_MAPPING_INPUT",
        "INVALID_METADATA",
        "UNREVIEWED_LAYOUT",
        "VALUE_VALIDATION_FAILED",
    ]
    missing_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    conflicts: tuple[BatteryCsvMappingConflictEvidence, ...] = ()
    unit_required: tuple[str, ...] = ()
    value_errors: tuple[BatteryCsvValueErrorEvidence, ...] = ()


class BatteryCsvMappingRejectionDetails(ContractModel):
    """Sanitized field-level details carried before the raw SHA is known."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    missing_fields: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()
    conflicts: tuple[BatteryCsvMappingConflictEvidence, ...] = ()
    unit_required: tuple[str, ...] = ()
    value_errors: tuple[BatteryCsvValueErrorEvidence, ...] = ()


class BatteryCsvValueError(ValueError):
    """Internal value rejection with sanitized field-level evidence."""

    def __init__(
        self,
        *,
        target_field: str,
        reason_code: BatteryCsvValueReasonCode,
        message: str,
    ) -> None:
        super().__init__(message)
        self.target_field = target_field
        self.reason_code = reason_code


class BatteryCsvMappingError(ValueError):
    """Machine-readable rejection that never carries observation values."""

    _MESSAGES: ClassVar[dict[str, str]] = {
        "AMBIGUOUS_LAYOUT": "source CSV layout matches multiple reviewed profiles",
        "INVALID_HEADER": "source CSV header is invalid",
        "INVALID_MAPPING_INPUT": "source CSV mapping input is invalid",
        "INVALID_METADATA": "source CSV metadata is invalid or inconsistent",
        "UNREVIEWED_LAYOUT": "source CSV layout is not reviewed",
        "VALUE_VALIDATION_FAILED": "source CSV value validation failed",
    }

    def __init__(
        self,
        reason_code: Literal[
            "AMBIGUOUS_LAYOUT",
            "INVALID_HEADER",
            "INVALID_MAPPING_INPUT",
            "INVALID_METADATA",
            "UNREVIEWED_LAYOUT",
            "VALUE_VALIDATION_FAILED",
        ],
        *,
        details: BatteryCsvMappingRejectionDetails | None = None,
    ) -> None:
        super().__init__(self._MESSAGES[reason_code])
        self.reason_code = reason_code
        self.details = details or _empty_rejection_details()


class ReviewedBatteryCsvNormalizer:
    """Select only an exact reviewed header layout; never perform fuzzy matching."""

    def __init__(self, profiles: tuple[BatteryCsvMappingProfile, ...]) -> None:
        checked = tuple(
            BatteryCsvMappingProfile.model_validate(item.model_dump(mode="json"))
            for item in profiles
        )
        if any(item.review_status != "APPROVED" for item in checked):
            raise ValueError("CSV mapping profiles must be explicitly approved")
        self._profiles = checked

    def normalize(self, payload: bytes) -> BatteryCsvMappingResult:
        try:
            metadata, csv_payload = _extract_battery_csv_metadata(payload)
        except (TypeError, ValueError) as exc:
            raise BatteryCsvMappingError("INVALID_METADATA") from exc
        try:
            header = _source_header(csv_payload)
        except (TypeError, ValueError) as exc:
            raise BatteryCsvMappingError("INVALID_HEADER") from exc
        if tuple(header) == CANONICAL_CYCLE_CSV_FIELDS:
            mapped = self._map(csv_payload, profile=_canonical_profile(), header=header)
            return _finalize_mapping_result(
                mapped,
                raw_payload=payload,
                canonical_payload=csv_payload,
                metadata=metadata,
            )
        if len(header) == len(CANONICAL_CYCLE_CSV_FIELDS) and set(header) == set(
            CANONICAL_CYCLE_CSV_FIELDS
        ):
            mapped = self._map(csv_payload, profile=_canonical_profile(), header=header)
            return _finalize_mapping_result(
                mapped,
                raw_payload=payload,
                metadata=metadata,
            )
        matches = tuple(
            profile
            for profile in self._profiles
            if len(profile.mappings) == len(header)
            and {item.source_header for item in profile.mappings} == set(header)
        )
        if not matches:
            details = _unreviewed_layout_details(header, self._profiles)
            raise BatteryCsvMappingError(
                "UNREVIEWED_LAYOUT",
                details=details,
            )
        if len(matches) != 1:
            raise BatteryCsvMappingError(
                "AMBIGUOUS_LAYOUT",
                details=_ambiguous_layout_details(header, matches),
            )
        mapped = self._map(csv_payload, profile=matches[0], header=header)
        return _finalize_mapping_result(
            mapped,
            raw_payload=payload,
            metadata=metadata,
        )

    @staticmethod
    def _map(
        payload: bytes,
        *,
        profile: BatteryCsvMappingProfile,
        header: tuple[str, ...],
    ) -> BatteryCsvMappingResult:
        try:
            return map_battery_csv(payload, profile=profile)
        except BatteryCsvValueError as exc:
            rule = next(
                item
                for item in profile.mappings
                if item.target_field == exc.target_field
            )
            raise BatteryCsvMappingError(
                "VALUE_VALIDATION_FAILED",
                details=BatteryCsvMappingRejectionDetails(
                    value_errors=(
                        BatteryCsvValueErrorEvidence(
                            source_header=rule.source_header,
                            target_field=rule.target_field,
                            reason_code=exc.reason_code,
                        ),
                    ),
                ),
            ) from exc
        except (TypeError, ValueError) as exc:
            raise BatteryCsvMappingError(
                "VALUE_VALIDATION_FAILED",
                details=BatteryCsvMappingRejectionDetails(),
            ) from exc


def _empty_rejection_details() -> BatteryCsvMappingRejectionDetails:
    return BatteryCsvMappingRejectionDetails()


def _unreviewed_layout_details(
    header: tuple[str, ...],
    profiles: tuple[BatteryCsvMappingProfile, ...],
) -> BatteryCsvMappingRejectionDetails:
    actual = set(header)
    candidates = (_canonical_profile(), *profiles)
    ranked = sorted(
        candidates,
        key=lambda profile: (
            len(
                {
                    mapping.source_header for mapping in profile.mappings
                }
                - actual
            )
            + len(
                actual
                - {mapping.source_header for mapping in profile.mappings}
            ),
            profile.profile_id != "canonical-cycle-record",
            profile.profile_id,
        ),
    )
    best = ranked[0] if ranked else _canonical_profile()
    expected_by_source = {
        mapping.source_header: mapping.target_field for mapping in best.mappings
    }
    matched = actual & set(expected_by_source)
    return BatteryCsvMappingRejectionDetails(
        missing_fields=tuple(
            target
            for target in CANONICAL_CYCLE_CSV_FIELDS
            if target not in {
                expected_by_source[source]
                for source in matched
            }
        ),
        unknown_fields=tuple(sorted(actual - set(expected_by_source))),
    )


def _ambiguous_layout_details(
    header: tuple[str, ...],
    profiles: tuple[BatteryCsvMappingProfile, ...],
) -> BatteryCsvMappingRejectionDetails:
    conflicts: list[BatteryCsvMappingConflictEvidence] = []
    for source_header in sorted(header):
        rules = tuple(
            mapping
            for profile in profiles
            for mapping in profile.mappings
            if mapping.source_header == source_header
        )
        target_fields = {rule.target_field for rule in rules}
        units = {(rule.source_unit, rule.canonical_unit) for rule in rules}
        conversions = {
            (
                rule.target_field,
                rule.source_unit,
                rule.canonical_unit,
                rule.scale,
                rule.offset,
                rule.sign,
            )
            for rule in rules
        }
        if len(target_fields) > 1:
            conflicts.extend(
                BatteryCsvMappingConflictEvidence(
                    source_header=source_header,
                    target_field=target_field,
                    issue_code="AMBIGUOUS_TARGET",
                )
                for target_field in sorted(target_fields)
            )
        elif len(units) > 1:
            target_field = next(iter(target_fields))
            conflicts.append(
                BatteryCsvMappingConflictEvidence(
                    source_header=source_header,
                    target_field=target_field,
                    issue_code="AMBIGUOUS_UNIT",
                )
            )
        elif len(conversions) > 1:
            target_field = next(iter(target_fields))
            scales = {rule.scale for rule in rules}
            offsets = {rule.offset for rule in rules}
            signs = {rule.sign for rule in rules}
            if len(scales) > 1:
                issue_code: BatteryCsvMappingIssueCode = "AMBIGUOUS_SCALE"
            elif len(offsets) > 1:
                issue_code = "AMBIGUOUS_OFFSET"
            elif len(signs) > 1:
                issue_code = "AMBIGUOUS_SIGN"
            else:
                issue_code = "AMBIGUOUS_CONVERSION"
            conflicts.append(
                BatteryCsvMappingConflictEvidence(
                    source_header=source_header,
                    target_field=target_field,
                    issue_code=issue_code,
                )
            )
    default_conflicts = _default_policy_conflicts(profiles)
    conflicts.extend(default_conflicts)
    return BatteryCsvMappingRejectionDetails(
        conflicts=tuple(conflicts),
        unit_required=tuple(
            sorted(
            {
                item.source_header
                for item in conflicts
                if item.issue_code == "AMBIGUOUS_UNIT"
            }
            )
        ),
    )


def _default_policy_conflicts(
    profiles: tuple[BatteryCsvMappingProfile, ...],
) -> tuple[BatteryCsvMappingConflictEvidence, ...]:
    defaults: dict[str, set[str]] = {}
    for profile in profiles:
        for item in profile.defaults:
            defaults.setdefault(item.target_field, set()).add(item.value)
    return tuple(
        BatteryCsvMappingConflictEvidence(
            source_header="<default>",
            target_field=target_field,
            issue_code="AMBIGUOUS_DEFAULT",
        )
        for target_field, values in sorted(defaults.items())
        if len(values) > 1
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"mapping profile JSON contains duplicate key: {key}")
        result[key] = value
    return result


def _strict_metadata_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"CSV metadata contains duplicate key: {key}")
        result[key] = value
    return result


def _extract_battery_csv_metadata(
    payload: bytes,
) -> tuple[BatteryCsvFileMetadata | None, bytes]:
    if not isinstance(payload, bytes) or not payload:
        return None, payload
    source = payload[3:] if payload.startswith(b"\xef\xbb\xbf") else payload
    first_line_end = source.find(b"\n")
    first_line = (
        source if first_line_end < 0 else source[:first_line_end]
    ).removesuffix(b"\r")
    if first_line != BATTERY_CSV_METADATA_ENVELOPE_V1.encode("ascii"):
        if first_line.startswith(_BATTERY_CSV_METADATA_ENVELOPE_PREFIX):
            raise ValueError("CSV metadata envelope version is not supported")
        return None, payload
    if first_line_end < 0:
        raise ValueError("CSV metadata envelope is incomplete")
    remaining = source[first_line_end + 1 :]
    metadata_line_end = remaining.find(b"\n")
    if metadata_line_end < 0:
        raise ValueError("CSV metadata envelope has no CSV body")
    metadata_bytes = remaining[:metadata_line_end].removesuffix(b"\r")
    csv_payload = remaining[metadata_line_end + 1 :]
    if not metadata_bytes or metadata_bytes != metadata_bytes.strip() or not csv_payload:
        raise ValueError("CSV metadata envelope is invalid")
    try:
        metadata_text = metadata_bytes.decode("utf-8")
        metadata_payload = json.loads(
            metadata_text,
            object_pairs_hook=_strict_metadata_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"CSV metadata contains invalid constant: {value}")
            ),
        )
        metadata = BatteryCsvFileMetadata.model_validate(metadata_payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("CSV metadata must be strict UTF-8 JSON") from exc
    return metadata, csv_payload


def _finalize_mapping_result(
    mapped: BatteryCsvMappingResult,
    *,
    raw_payload: bytes,
    metadata: BatteryCsvFileMetadata | None,
    canonical_payload: bytes | None = None,
) -> BatteryCsvMappingResult:
    if metadata is not None:
        rows = csv.DictReader(io.StringIO(mapped.canonical_payload.decode("utf-8")))
        cell_ids = {row["cell_id"] for row in rows}
        if cell_ids != {metadata.cell_id}:
            raise BatteryCsvMappingError("INVALID_METADATA")
    final_payload = canonical_payload or mapped.canonical_payload
    final_row_count = mapped.row_count
    if metadata is not None:
        final_payload, final_row_count = _truncate_self_described_batch(final_payload)
    return BatteryCsvMappingResult(
        canonical_payload=final_payload,
        raw_sha256=hashlib.sha256(raw_payload).hexdigest(),
        canonical_sha256=hashlib.sha256(final_payload).hexdigest(),
        profile_id=mapped.profile_id,
        profile_version=mapped.profile_version,
        profile_sha256=mapped.profile_sha256,
        row_count=final_row_count,
        column_evidence=mapped.column_evidence,
        metadata=metadata,
    )


def _truncate_self_described_batch(payload: bytes) -> tuple[bytes, int]:
    text = payload.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader)
    cycle_index = header.index("cycle_index")
    valid_index = header.index("valid")
    rows = list(reader)
    valid_cycles = {
        int(row[cycle_index])
        for row in rows
        if row[valid_index].strip().lower() in {"true", "1"}
    }
    eligible = tuple(
        cutoff
        for cutoff in _SELF_DESCRIBED_SUPPORTED_CUTOFFS
        if valid_cycles and cutoff <= max(valid_cycles)
    )
    if not eligible:
        return payload, len(rows)
    cutoff_cycle = eligible[-1]
    selected = [row for row in rows if int(row[cycle_index]) <= cutoff_cycle]
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(selected)
    return output.getvalue().encode("utf-8"), len(selected)


def _source_header(payload: bytes) -> tuple[str, ...]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("source CSV payload must be non-empty bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("source CSV payload must be UTF-8") from exc
    try:
        header = next(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (StopIteration, csv.Error) as exc:
        raise ValueError("source CSV has no header") from exc
    if not header or any(not value or value != value.strip() for value in header):
        raise ValueError("source CSV header is invalid")
    if len(header) != len(set(header)):
        raise ValueError("source CSV contains a duplicate source header")
    return tuple(header)


def _canonical_profile() -> BatteryCsvMappingProfile:
    return BatteryCsvMappingProfile(
        profile_id="canonical-cycle-record",
        version="canonical-cycle-record-v1",
        review_status="APPROVED",
        mappings=tuple(
            BatteryCsvColumnMapping(
                source_header=field,
                target_field=field,
                source_unit=_CANONICAL_UNITS[field],
                canonical_unit=_CANONICAL_UNITS[field],
                scale="1",
                offset="0",
                sign=1,
            )
            for field in CANONICAL_CYCLE_CSV_FIELDS
        ),
    )


def load_battery_csv_mapping_profile(path: str | Path) -> BatteryCsvMappingProfile:
    """Load a strict UTF-8 JSON mapping profile without accepting NaN constants."""

    profile_path = Path(path)
    try:
        text = profile_path.read_text(encoding="utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"mapping profile contains invalid constant: {value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("mapping profile must be strict UTF-8 JSON") from exc
    return BatteryCsvMappingProfile.model_validate(payload)


def _formula_identifier(value: str) -> bool:
    stripped = value.lstrip()
    if not stripped:
        return False
    if stripped[0] in {"=", "+", "@"}:
        return True
    if stripped[0] != "-":
        return False
    try:
        Decimal(stripped)
    except InvalidOperation:
        return True
    return False


def _normalize_decimal(value: Decimal, *, target_field: str) -> str:
    if value.adjusted() > 100 or value.adjusted() < -100:
        raise BatteryCsvValueError(
            target_field=target_field,
            reason_code="EXPONENT_OUT_OF_RANGE",
            message="converted decimal exponent exceeds the safe range",
        )
    rendered = format(value, "f")
    if len(rendered) > 1_000:
        raise BatteryCsvValueError(
            target_field=target_field,
            reason_code="OUTPUT_TOO_LONG",
            message="converted decimal text exceeds the safe length",
        )
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _exact_affine_conversion(
    parsed: Decimal,
    *,
    sign: int,
    scale: Decimal,
    offset: Decimal,
) -> Decimal:
    parsed_exponent = parsed.as_tuple().exponent
    scale_exponent = scale.as_tuple().exponent
    offset_exponent = offset.as_tuple().exponent
    if not all(
        isinstance(item, int)
        for item in (parsed_exponent, scale_exponent, offset_exponent)
    ):
        raise ValueError("unit conversion requires finite decimal exponents")
    assert isinstance(parsed_exponent, int)
    assert isinstance(scale_exponent, int)
    assert isinstance(offset_exponent, int)
    precision = (
        len(parsed.as_tuple().digits)
        + len(scale.as_tuple().digits)
        + len(offset.as_tuple().digits)
        + abs(parsed_exponent)
        + abs(scale_exponent)
        + abs(offset_exponent)
        + 16
    )
    try:
        with localcontext() as context:
            context.prec = max(precision, 50)
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            return Decimal(sign) * parsed * scale + offset
    except DecimalException as exc:
        raise ValueError("unit conversion would lose decimal precision") from exc


def _map_value(raw: str, rule: BatteryCsvColumnMapping, *, row_number: int) -> str:
    target = rule.target_field
    value = raw.strip()
    if target in _IDENTIFIER_FIELDS:
        if not value:
            raise BatteryCsvValueError(
                target_field=target,
                reason_code="EMPTY_IDENTIFIER",
                message=f"{target} is empty at row {row_number}",
            )
        if _formula_identifier(value):
            raise BatteryCsvValueError(
                target_field=target,
                reason_code="FORMULA_CONTENT",
                message=f"{target} contains spreadsheet formula content",
            )
        return value
    if target in _BOOLEAN_FIELDS:
        normalized = value.lower()
        if normalized in {"true", "1"}:
            return "true"
        if normalized in {"false", "0"}:
            return "false"
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="INVALID_BOOLEAN",
            message=f"{target} must be a boolean at row {row_number}",
        )
    if target not in _NUMERIC_FIELDS:
        raise ValueError(f"unsupported canonical target field: {target}")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="INVALID_DECIMAL",
            message=f"{target} at row {row_number} must be a finite decimal",
        ) from exc
    if not parsed.is_finite():
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="NONFINITE_DECIMAL",
            message=f"{target} at row {row_number} must be a finite decimal",
        )
    if parsed != 0 and (parsed.adjusted() > 100 or parsed.adjusted() < -100):
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="EXPONENT_OUT_OF_RANGE",
            message=f"{target} at row {row_number} exponent exceeds the safe range",
        )
    if target in _INTEGER_FIELDS and parsed != parsed.to_integral_value():
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="NONINTEGER_VALUE",
            message=f"{target} must be an integer at row {row_number}",
        )
    scale = _decimal(rule.scale, label="mapping scale")
    offset = _decimal(rule.offset, label="mapping offset")
    if rule.sign == 1 and scale == 1 and offset == 0:
        return value
    try:
        converted = _exact_affine_conversion(
            parsed,
            sign=rule.sign,
            scale=scale,
            offset=offset,
        )
    except ValueError as exc:
        raise BatteryCsvValueError(
            target_field=target,
            reason_code="PRECISION_LOSS",
            message=f"{target} unit conversion would lose decimal precision",
        ) from exc
    if not converted.is_finite():
        raise ValueError(f"{target} must be a finite decimal at row {row_number}")
    return _normalize_decimal(converted, target_field=target)


def map_battery_csv(
    payload: bytes,
    *,
    profile: BatteryCsvMappingProfile,
) -> BatteryCsvMappingResult:
    """Map untrusted UTF-8 CSV bytes using one complete reviewed profile."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError("source CSV payload must be non-empty bytes")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("source CSV payload must be UTF-8") from exc
    checked_profile = BatteryCsvMappingProfile.model_validate(
        profile.model_dump(mode="json")
    )
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise ValueError("source CSV has no header") from exc
    if len(header) != len(set(header)):
        raise ValueError("source CSV contains a duplicate source header")
    expected_headers = {item.source_header for item in checked_profile.mappings}
    actual_headers = set(header)
    unknown = actual_headers - expected_headers
    missing = expected_headers - actual_headers
    if unknown:
        raise ValueError(f"source CSV has unknown source columns: {sorted(unknown)}")
    if missing:
        raise ValueError(f"source CSV has missing source columns: {sorted(missing)}")

    mapping_by_target = {item.target_field: item for item in checked_profile.mappings}
    defaults_by_target = {item.target_field: item for item in checked_profile.defaults}
    source_index = {name: index for index, name in enumerate(header)}
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(CANONICAL_CYCLE_CSV_FIELDS)
    row_count = 0
    identities: set[tuple[str, str, str, str]] = set()
    try:
        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            if len(row) != len(header):
                raise ValueError(f"source CSV row {row_number} has an invalid column count")
            canonical: list[str] = []
            for target in CANONICAL_CYCLE_CSV_FIELDS:
                rule = mapping_by_target.get(target)
                if rule is not None:
                    canonical.append(
                        _map_value(
                            row[source_index[rule.source_header]],
                            rule,
                            row_number=row_number,
                        )
                    )
                else:
                    canonical.append(defaults_by_target[target].value)
            identity = (
                canonical[0],
                canonical[1],
                canonical[2],
                canonical[3],
            )
            if identity in identities:
                raise ValueError("canonical sample identity collision")
            identities.add(identity)
            writer.writerow(canonical)
    except csv.Error as exc:
        raise ValueError("source CSV syntax is invalid") from exc
    if row_count == 0:
        raise ValueError("source CSV has no data rows")

    canonical_payload = output.getvalue().encode("utf-8")
    evidence: list[BatteryCsvColumnEvidence] = []
    for target in CANONICAL_CYCLE_CSV_FIELDS:
        rule = mapping_by_target.get(target)
        if rule is None:
            default = defaults_by_target[target]
            evidence.append(
                BatteryCsvColumnEvidence(
                    target_field=target,
                    source_header=None,
                    source_unit=None,
                    canonical_unit=None,
                    scale=None,
                    offset=None,
                    sign=None,
                    default_value=default.value,
                )
            )
        else:
            evidence.append(
                BatteryCsvColumnEvidence(
                    target_field=target,
                    source_header=rule.source_header,
                    source_unit=rule.source_unit,
                    canonical_unit=rule.canonical_unit,
                    scale=rule.scale,
                    offset=rule.offset,
                    sign=rule.sign,
                    default_value=None,
                )
            )
    return BatteryCsvMappingResult(
        canonical_payload=canonical_payload,
        raw_sha256=hashlib.sha256(payload).hexdigest(),
        canonical_sha256=hashlib.sha256(canonical_payload).hexdigest(),
        profile_id=checked_profile.profile_id,
        profile_version=checked_profile.version,
        profile_sha256=checked_profile.profile_sha256,
        row_count=row_count,
        column_evidence=tuple(evidence),
    )


__all__ = [
    "BATTERY_CSV_METADATA_ENVELOPE_V1",
    "BatteryCsvColumnEvidence",
    "BatteryCsvColumnMapping",
    "BatteryCsvFileMetadata",
    "BatteryCsvMappingConflictEvidence",
    "BatteryCsvMappingError",
    "BatteryCsvMappingProfile",
    "BatteryCsvMappingRejectionDetails",
    "BatteryCsvMappingRejectionEvidence",
    "BatteryCsvMappingResult",
    "BatteryCsvMappingSuccessEvidence",
    "BatteryCsvStructuralDefault",
    "BatteryCsvValueErrorEvidence",
    "ReviewedBatteryCsvNormalizer",
    "load_battery_csv_mapping_profile",
    "map_battery_csv",
]
