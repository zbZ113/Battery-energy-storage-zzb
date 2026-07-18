"""Credential-free industrial protocol sandboxes over existing trusted tools."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import Lock
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, field_validator

from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    Decision,
    ProvenanceRecord,
    SourceKind,
    sha256_canonical,
)
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.batch_decision import BATCH_DECISION_TOOL_VERSION
from quanxin_life.tools.observed_soh_ingestion import (
    VerifiedNewlyObservedMeasurement,
    VerifiedObservationBatch,
)
from quanxin_life.tools.registry import StandardToolName

MQTT_TELEMETRY_TOPIC = re.compile(
    r"^quanxin/v1/bms/(?P<batch_id>[A-Za-z0-9._-]{1,100})$"
)
MAX_MQTT_PAYLOAD_BYTES = 1_048_576
MODBUS_REGISTER_MAP_VERSION = "quanxin-modbus-bms-v1"


class IndustrialSandboxTransport(StrEnum):
    """Credential-free protocol paths; none implies a production connection."""

    REST_SANDBOX = "REST_SANDBOX"
    MQTT_SANDBOX = "MQTT_SANDBOX"
    MODBUS_SANDBOX = "MODBUS_SANDBOX"


class BmsTelemetryPayload(ContractModel):
    """Transport-neutral measurement input; callers cannot supply SOH."""

    message_id: str
    measurement_batch_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    dataset_id: str = Field(min_length=1, max_length=100)
    cell_id: str = Field(min_length=1, max_length=200)
    cycle: int = Field(ge=0, strict=True)
    discharge_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    reference_capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    measured_at: datetime
    reference_capacity_method: str = Field(min_length=1, max_length=200)
    data_version: str = Field(min_length=1, max_length=100)
    feature_version: str = Field(min_length=1, max_length=100)
    split_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)

    @field_validator("message_id")
    @classmethod
    def message_id_is_canonical_uuid(cls, value: str) -> str:
        try:
            parsed = UUID(value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("message_id must be a UUID") from exc
        if str(parsed) != value:
            raise ValueError("message_id must be a canonical lowercase UUID")
        return value

    @field_validator(
        "dataset_id",
        "cell_id",
        "reference_capacity_method",
        "data_version",
        "feature_version",
        "split_version",
        "schema_version",
    )
    @classmethod
    def text_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("industrial telemetry text fields must not be blank")
        return normalized

    @field_validator("measured_at")
    @classmethod
    def measured_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("measured_at must include a timezone")
        return value.astimezone(UTC)


class BmsSandboxReceipt(ContractModel):
    """Idempotent acknowledgement without echoing raw telemetry."""

    message_id: str
    measurement_batch_id: str = Field(min_length=1)
    transport: IndustrialSandboxTransport
    record_hash: Sha256
    replayed: bool


class ModbusBmsSnapshot(ContractModel):
    """Fixed v1 simulator metadata plus eight unsigned 16-bit registers.

    Register pairs encode cycle, discharge capacity in mAh, reference capacity
    in mAh, and UTC Unix seconds, in that order and most-significant word first.
    """

    message_id: str
    measurement_batch_id: str = Field(min_length=1, max_length=100)
    dataset_id: str = Field(min_length=1, max_length=100)
    cell_id: str = Field(min_length=1, max_length=200)
    registers: tuple[int, ...] = Field(min_length=8, max_length=8)
    reference_capacity_method: str = Field(min_length=1, max_length=200)
    data_version: str = Field(min_length=1, max_length=100)
    feature_version: str = Field(min_length=1, max_length=100)
    split_version: str = Field(min_length=1, max_length=100)
    register_map_version: str = Field(min_length=1, max_length=100)

    @field_validator("message_id")
    @classmethod
    def message_id_is_canonical_uuid(cls, value: str) -> str:
        return BmsTelemetryPayload.message_id_is_canonical_uuid(value)

    @field_validator("registers")
    @classmethod
    def registers_are_unsigned_16_bit(
        cls,
        value: tuple[int, ...],
    ) -> tuple[int, ...]:
        if any(
            not isinstance(item, int)
            or isinstance(item, bool)
            or item < 0
            or item > 65_535
            for item in value
        ):
            raise ValueError("Modbus registers must be unsigned 16-bit integers")
        return value

    @field_validator("register_map_version")
    @classmethod
    def register_map_is_supported(cls, value: str) -> str:
        if value != MODBUS_REGISTER_MAP_VERSION:
            raise ValueError("unsupported Modbus BMS register map version")
        return value

    def as_telemetry(self) -> BmsTelemetryPayload:
        return BmsTelemetryPayload(
            message_id=self.message_id,
            measurement_batch_id=self.measurement_batch_id,
            dataset_id=self.dataset_id,
            cell_id=self.cell_id,
            cycle=_decode_u32(self.registers, 0),
            discharge_capacity_ah=_decode_u32(self.registers, 2) / 1_000.0,
            reference_capacity_ah=_decode_u32(self.registers, 4) / 1_000.0,
            measured_at=datetime.fromtimestamp(
                _decode_u32(self.registers, 6),
                tz=UTC,
            ),
            reference_capacity_method=self.reference_capacity_method,
            data_version=self.data_version,
            feature_version=self.feature_version,
            split_version=self.split_version,
            schema_version="bms-telemetry-v1",
        )


class EmsDecisionSandboxMessage(ContractModel):
    """Traceable categorical decision message; thresholds remain in ToolResult."""

    source_result_id: str
    source_result_hash: Sha256
    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    decision: Decision
    policy_version: str = Field(min_length=1)
    reason_codes: tuple[str, ...]
    created_at: datetime
    sandbox_only: Literal[True] = True

    @field_validator("source_result_id")
    @classmethod
    def source_result_id_is_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("source_result_id must be a UUID") from exc
        return value

    @field_validator("reason_codes")
    @classmethod
    def reason_codes_are_unique(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("EMS reason codes must not be blank")
        if len(value) != len(set(value)):
            raise ValueError("EMS reason codes must be unique")
        return value

    @field_validator("created_at")
    @classmethod
    def created_at_is_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class _StoredMeasurement:
    payload: BmsTelemetryPayload
    transport: IndustrialSandboxTransport
    record_hash: str


class IndustrialBmsSandbox:
    """In-memory REST/MQTT/Modbus simulator implementing the trusted resolver."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._records: dict[str, _StoredMeasurement] = {}
        self._batches: dict[str, list[str]] = {}

    def ingest_rest(self, payload: BmsTelemetryPayload) -> BmsSandboxReceipt:
        return self._ingest(payload, IndustrialSandboxTransport.REST_SANDBOX)

    def ingest_mqtt(self, topic: str, payload: bytes) -> BmsSandboxReceipt:
        topic_match = MQTT_TELEMETRY_TOPIC.fullmatch(topic)
        if topic_match is None:
            raise ValueError("MQTT telemetry topic does not match the approved v1 layout")
        decoded = _decode_mqtt_payload(payload)
        telemetry = BmsTelemetryPayload.model_validate(decoded)
        if telemetry.measurement_batch_id != topic_match.group("batch_id"):
            raise ValueError("MQTT topic batch does not match payload batch")
        return self._ingest(telemetry, IndustrialSandboxTransport.MQTT_SANDBOX)

    def ingest_modbus(self, snapshot: ModbusBmsSnapshot) -> BmsSandboxReceipt:
        validated = ModbusBmsSnapshot.model_validate(snapshot.model_dump(mode="json"))
        return self._ingest(
            validated.as_telemetry(),
            IndustrialSandboxTransport.MODBUS_SANDBOX,
        )

    def _ingest(
        self,
        payload: BmsTelemetryPayload,
        transport: IndustrialSandboxTransport,
    ) -> BmsSandboxReceipt:
        validated = BmsTelemetryPayload.model_validate(payload.model_dump(mode="json"))
        record_hash = sha256_canonical(
            {
                "transport": transport.value,
                "payload": validated.model_dump(mode="json"),
            }
        )
        with self._lock:
            existing = self._records.get(validated.message_id)
            if existing is not None:
                if existing.record_hash != record_hash:
                    raise ValueError("industrial message idempotency conflict")
                return self._receipt(existing, replayed=True)
            batch_records = [
                self._records[message_id]
                for message_id in self._batches.get(
                    validated.measurement_batch_id,
                    (),
                )
            ]
            self._validate_batch_append(validated, batch_records)
            stored = _StoredMeasurement(validated, transport, record_hash)
            self._records[validated.message_id] = stored
            self._batches.setdefault(validated.measurement_batch_id, []).append(
                validated.message_id
            )
        return self._receipt(stored, replayed=False)

    @staticmethod
    def _validate_batch_append(
        payload: BmsTelemetryPayload,
        records: list[_StoredMeasurement],
    ) -> None:
        for record in records:
            existing = record.payload
            if existing.cycle == payload.cycle:
                raise ValueError("industrial batch contains a duplicate cycle")
            context = (
                existing.dataset_id,
                existing.cell_id,
                existing.reference_capacity_ah,
                existing.reference_capacity_method,
                existing.data_version,
                existing.feature_version,
                existing.split_version,
            )
            incoming_context = (
                payload.dataset_id,
                payload.cell_id,
                payload.reference_capacity_ah,
                payload.reference_capacity_method,
                payload.data_version,
                payload.feature_version,
                payload.split_version,
            )
            if context != incoming_context:
                raise ValueError("industrial batch context conflict")

    @staticmethod
    def _receipt(
        record: _StoredMeasurement,
        *,
        replayed: bool,
    ) -> BmsSandboxReceipt:
        return BmsSandboxReceipt(
            message_id=record.payload.message_id,
            measurement_batch_id=record.payload.measurement_batch_id,
            transport=record.transport,
            record_hash=record.record_hash,
            replayed=replayed,
        )

    def resolve_verified_observation_batch(
        self,
        measurement_batch_id: str,
    ) -> VerifiedObservationBatch:
        normalized = measurement_batch_id.strip()
        with self._lock:
            message_ids = tuple(self._batches.get(normalized, ()))
            records = tuple(self._records[item] for item in message_ids)
        if not records:
            raise ValueError("industrial sandbox batch was not found")
        ordered = tuple(sorted(records, key=lambda item: item.payload.cycle))
        first = ordered[0].payload
        batch_hash = sha256_canonical(
            {
                "measurement_batch_id": normalized,
                "record_hashes": [item.record_hash for item in ordered],
            }
        )
        return VerifiedObservationBatch(
            measurement_batch_id=normalized,
            measurements=tuple(
                VerifiedNewlyObservedMeasurement(
                    measurement_id=item.payload.message_id,
                    dataset_id=item.payload.dataset_id,
                    cell_id=item.payload.cell_id,
                    cycle=item.payload.cycle,
                    discharge_capacity_ah=item.payload.discharge_capacity_ah,
                    reference_capacity_ah=item.payload.reference_capacity_ah,
                    measured_at=item.payload.measured_at,
                    source_record_hash=item.record_hash,
                )
                for item in ordered
            ),
            reference_capacity_method=first.reference_capacity_method,
            data_version=first.data_version,
            feature_version=first.feature_version,
            split_version=first.split_version,
            provenance=(
                ProvenanceRecord(
                    source_id=f"industrial-sandbox:{normalized}",
                    source_kind=SourceKind.NEWLY_OBSERVED,
                    uri=f"sandbox://industrial-bms/{normalized}",
                    sha256=batch_hash,
                    description=(
                        "Credential-free industrial protocol sandbox; "
                        "not a production BMS connection"
                    ),
                    created_at=max(item.payload.measured_at for item in ordered),
                ),
            ),
        )


class EmsDecisionSandboxPublisher:
    """Build a sandbox EMS message only from one registered decision result."""

    def __init__(self, ledger: AuditLedger) -> None:
        self._ledger = ledger

    def build_message(self, result_id: str) -> EmsDecisionSandboxMessage:
        result = self._ledger.resolve_registered_result(result_id)
        if (
            result.tool_name != StandardToolName.MAKE_BATCH_DECISION.value
            or result.tool_version != BATCH_DECISION_TOOL_VERSION
        ):
            raise ValueError("EMS message source must be a supported batch decision")
        values = result.values
        dataset_id = _required_string(values, "dataset_id")
        cell_id = _required_string(values, "cell_id")
        policy_version = _required_string(values, "policy_version")
        try:
            decision = Decision(_required_string(values, "decision"))
        except ValueError as exc:
            raise ValueError("batch decision contains an unsupported decision") from exc
        reason_value = values.get("reason_codes")
        if not isinstance(reason_value, list) or any(
            not isinstance(item, str) for item in reason_value
        ):
            raise ValueError("batch decision reason_codes are invalid")
        return EmsDecisionSandboxMessage(
            source_result_id=result.result_id,
            source_result_hash=sha256_canonical(result.model_dump(mode="json")),
            dataset_id=dataset_id,
            cell_id=cell_id,
            decision=decision,
            policy_version=policy_version,
            reason_codes=tuple(reason_value),
            created_at=result.created_at,
            sandbox_only=True,
        )


def _decode_mqtt_payload(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("MQTT telemetry payload must be nonempty bytes")
    if len(payload) > MAX_MQTT_PAYLOAD_BYTES:
        raise ValueError("MQTT telemetry payload exceeds the sandbox limit")
    try:
        text = payload.decode("utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MQTT telemetry payload must be strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("MQTT telemetry payload must be one JSON object")
    return decoded


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("MQTT telemetry JSON contains a duplicate key")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"MQTT telemetry JSON contains a non-finite constant: {value}")


def _decode_u32(registers: tuple[int, ...], offset: int) -> int:
    return registers[offset] * 65_536 + registers[offset + 1]


def _required_string(values: dict[str, Any], field_name: str) -> str:
    value = values.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"batch decision {field_name} is invalid")
    return value


__all__ = [
    "MAX_MQTT_PAYLOAD_BYTES",
    "MODBUS_REGISTER_MAP_VERSION",
    "MQTT_TELEMETRY_TOPIC",
    "BmsSandboxReceipt",
    "BmsTelemetryPayload",
    "EmsDecisionSandboxMessage",
    "EmsDecisionSandboxPublisher",
    "IndustrialBmsSandbox",
    "IndustrialSandboxTransport",
    "ModbusBmsSnapshot",
]
