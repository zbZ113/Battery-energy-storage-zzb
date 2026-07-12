from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
)

from quanxin_life.core.enums import SourceKind
from quanxin_life.core.hashing import sha256_canonical

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
JsonMapping = dict[str, Any]


def _uuid_string(value: str) -> str:
    try:
        UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("must be a UUID string") from exc
    return value


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value.astimezone(UTC)


def _json_mapping(value: JsonMapping | None) -> JsonMapping | None:
    if value is not None:
        try:
            sha256_canonical(value)
        except TypeError as exc:
            raise ValueError("mapping must contain only JSON-compatible values") from exc
    return value


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProvenanceRecord(ContractModel):
    source_id: str = Field(min_length=1)
    source_kind: SourceKind
    uri: str = Field(min_length=1)
    sha256: Sha256
    description: str = Field(min_length=1)
    created_at: datetime

    _created_at_utc = field_validator("created_at")(_utc_datetime)


class ToolResult(ContractModel):
    result_id: str
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    model_version: str | None = None
    data_version: str | None = None
    feature_version: str | None = None
    input_hash: Sha256
    values: JsonMapping
    uncertainty: JsonMapping | None = None
    warnings: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    _result_id_uuid = field_validator("result_id")(_uuid_string)
    _created_at_utc = field_validator("created_at")(_utc_datetime)
    _values_are_json = field_validator("values")(_json_mapping)
    _uncertainty_is_json = field_validator("uncertainty")(_json_mapping)


class CellMetadata(ContractModel):
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    raw_cell_id: str | None = None
    chemistry: str = Field(min_length=1)
    nominal_capacity_ah: float = Field(gt=0)
    reference_capacity_ah: float | None = Field(default=None, gt=0)
    eol_threshold: float = Field(default=0.8, gt=0, lt=1)
    protocol_id: str | None = None
    protocol_description: str | None = None
    official_life_label: int | None = Field(default=None, ge=0)
    official_life_label_name: str | None = None
    source_uri: str = Field(min_length=1)
    source_sha256: Sha256
    schema_version: str = Field(min_length=1)
    adapter_version: str | None = None
    ingestion_parameters: JsonMapping = Field(default_factory=dict)

    @field_validator("ingestion_parameters")
    @classmethod
    def ingestion_parameters_are_json(cls, value: JsonMapping) -> JsonMapping:
        _json_mapping(value)
        return value


class AnalysisState(ContractModel):
    request_id: str
    status: str = Field(min_length=1)
    ingestion_result_id: str | None = None
    soh_result_id: str | None = None
    rul_result_id: str | None = None
    diagnosis_result_id: str | None = None
    decision_result_id: str | None = None
    report_result_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    _request_id_uuid = field_validator("request_id")(_uuid_string)

    @field_validator(
        "ingestion_result_id",
        "soh_result_id",
        "rul_result_id",
        "diagnosis_result_id",
        "decision_result_id",
        "report_result_id",
    )
    @classmethod
    def result_ids_are_uuids(cls, value: str | None) -> str | None:
        return _uuid_string(value) if value is not None else None
