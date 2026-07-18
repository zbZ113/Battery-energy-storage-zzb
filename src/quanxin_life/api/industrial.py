"""Authenticated HTTP adapter for credential-free industrial sandboxes."""

from __future__ import annotations

from base64 import b64decode
from binascii import Error as Base64DecodeError
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel
from quanxin_life.integrations.industrial import (
    MAX_MQTT_PAYLOAD_BYTES,
    BmsSandboxReceipt,
    BmsTelemetryPayload,
    EmsDecisionSandboxMessage,
    EmsDecisionSandboxPublisher,
    IndustrialBmsSandbox,
    ModbusBmsSnapshot,
)

_PayloadT = TypeVar("_PayloadT")


class MqttSandboxRequest(ContractModel):
    """HTTP envelope for exercising the MQTT sandbox without a broker."""

    topic: str = Field(min_length=1, max_length=300)
    payload_base64: str = Field(min_length=1)

    def decoded_payload(self) -> bytes:
        try:
            payload = b64decode(self.payload_base64, validate=True)
        except (Base64DecodeError, ValueError) as exc:
            raise ValueError("payload_base64 must be valid base64") from exc
        if not payload:
            raise ValueError("decoded MQTT payload must not be empty")
        if len(payload) > MAX_MQTT_PAYLOAD_BYTES:
            raise ValueError("decoded MQTT payload exceeds the sandbox limit")
        return payload


@dataclass(frozen=True, slots=True)
class IndustrialSandboxHttpAdapter:
    router: APIRouter


def create_industrial_sandbox_http_adapter(
    bms_sandbox: IndustrialBmsSandbox,
    ems_publisher: EmsDecisionSandboxPublisher,
    *,
    auth_adapter: AuthHttpAdapter,
) -> IndustrialSandboxHttpAdapter:
    """Expose protocol sandboxes without implying a production connection."""

    router = APIRouter(
        prefix="/v1/integrations/industrial/sandbox",
        tags=["industrial-sandbox"],
    )
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )
    write_dependencies = [
        Depends(operator_dependency),
        Depends(auth_adapter.require_trusted_origin),
    ]

    def ingest(
        call: Callable[[_PayloadT], BmsSandboxReceipt],
        payload: _PayloadT,
    ) -> BmsSandboxReceipt:
        try:
            return call(payload)
        except ValueError as exc:
            detail = str(exc)
            status_code = (
                409
                if "conflict" in detail.casefold()
                or "duplicate cycle" in detail.casefold()
                else 422
            )
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @router.post(
        "/bms/rest",
        response_model=BmsSandboxReceipt,
        status_code=201,
        dependencies=write_dependencies,
    )
    def ingest_rest(payload: BmsTelemetryPayload) -> BmsSandboxReceipt:
        return ingest(bms_sandbox.ingest_rest, payload)

    @router.post(
        "/bms/mqtt",
        response_model=BmsSandboxReceipt,
        status_code=201,
        dependencies=write_dependencies,
    )
    def ingest_mqtt(payload: MqttSandboxRequest) -> BmsSandboxReceipt:
        try:
            raw_payload = payload.decoded_payload()
            return bms_sandbox.ingest_mqtt(payload.topic, raw_payload)
        except ValueError as exc:
            detail = str(exc)
            status_code = 409 if "conflict" in detail.casefold() else 422
            raise HTTPException(status_code=status_code, detail=detail) from exc

    @router.post(
        "/bms/modbus",
        response_model=BmsSandboxReceipt,
        status_code=201,
        dependencies=write_dependencies,
    )
    def ingest_modbus(payload: ModbusBmsSnapshot) -> BmsSandboxReceipt:
        return ingest(bms_sandbox.ingest_modbus, payload)

    @router.get(
        "/ems/decisions/{result_id}",
        response_model=EmsDecisionSandboxMessage,
        dependencies=[Depends(auth_adapter.require_ready_user)],
    )
    def get_ems_decision(result_id: str) -> EmsDecisionSandboxMessage:
        try:
            return ems_publisher.build_message(result_id)
        except ValueError as exc:
            detail = str(exc)
            status_code = 404 if "not registered" in detail else 422
            raise HTTPException(status_code=status_code, detail=detail) from exc

    return IndustrialSandboxHttpAdapter(router=router)


__all__ = [
    "IndustrialSandboxHttpAdapter",
    "MqttSandboxRequest",
    "create_industrial_sandbox_http_adapter",
]
