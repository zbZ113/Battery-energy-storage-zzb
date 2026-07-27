"""Authenticated HTTP adapter for Advanced calibration materializations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.advanced_calibration_jobs import (
    AdvancedCalibrationDispatchReceipt,
    AdvancedCalibrationMaterializationError,
    AdvancedCalibrationMaterializationRecord,
)
from quanxin_life.application.invocation_context import (
    ProjectInvocationAccessError,
    ProjectInvocationContextService,
    ProjectInvocationNotFoundError,
    VerifiedProjectInvocationContext,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask, UserRole
from quanxin_life.core.schemas import ContractModel

DEFAULT_ADVANCED_CALIBRATION_SOURCE_REGISTRATION_ID = (
    "matr-three-batch-final-v1"
)


class CreateAdvancedCalibrationMaterializationRequest(ContractModel):
    """Identity-only API payload; all evidence and runtime fields stay server-owned."""

    task: AdvancedModelTask
    cutoff_cycle: int = Field(gt=0)
    route_role: AdvancedModelRouteRole
    source_registration_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )


@dataclass(frozen=True, slots=True)
class CreateAdvancedCalibrationMaterializationResponse:
    materialization: AdvancedCalibrationMaterializationRecord
    dispatch: AdvancedCalibrationDispatchReceipt


class AdvancedCalibrationMaterializationApplication(Protocol):
    def create(
        self,
        context: VerifiedProjectInvocationContext,
        request: Any,
        *,
        idempotency_key: str,
    ) -> AdvancedCalibrationMaterializationRecord: ...

    def get(
        self,
        context: VerifiedProjectInvocationContext,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationMaterializationRecord: ...

    def list(
        self,
        context: VerifiedProjectInvocationContext,
    ) -> tuple[AdvancedCalibrationMaterializationRecord, ...]: ...


class AdvancedCalibrationQueue(Protocol):
    def enqueue(
        self,
        *,
        materialization_id: str,
    ) -> AdvancedCalibrationDispatchReceipt: ...


@dataclass(frozen=True, slots=True)
class AdvancedCalibrationHttpAdapter:
    router: APIRouter


def create_advanced_calibration_http_adapter(
    service: AdvancedCalibrationMaterializationApplication,
    *,
    queue: AdvancedCalibrationQueue,
    context_service: ProjectInvocationContextService,
    auth_adapter: AuthHttpAdapter,
    default_source_registration_id: str = (
        DEFAULT_ADVANCED_CALIBRATION_SOURCE_REGISTRATION_ID
    ),
) -> AdvancedCalibrationHttpAdapter:
    """Build project-scoped ADMIN mutation and operator read endpoints."""

    normalized_default_source = default_source_registration_id.strip()
    if not normalized_default_source:
        raise ValueError(
            "default Advanced calibration source registration must not be blank"
        )
    router = APIRouter(
        prefix="/v1/projects/{project_id}/advanced-calibration/materializations",
        tags=["advanced-calibration"],
    )
    admin = auth_adapter.require_roles({UserRole.ADMIN})
    operator = auth_adapter.require_roles({UserRole.ADMIN, UserRole.MEMBER})
    admin_principal = Depends(admin)
    operator_principal = Depends(operator)
    idempotency_header = Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=200,
    )

    @router.post(
        "",
        response_model=CreateAdvancedCalibrationMaterializationResponse,
        status_code=202,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_materialization(
        project_id: str,
        payload: CreateAdvancedCalibrationMaterializationRequest,
        principal: AuthPrincipal = admin_principal,
        idempotency_key: str = idempotency_header,
    ) -> Any:
        from quanxin_life.application.advanced_calibration_materialization import (
            AdvancedCalibrationMaterializationRequest,
        )

        try:
            context = context_service.resolve_http(principal, project_id)
            request = AdvancedCalibrationMaterializationRequest(
                project_id=project_id,
                task=payload.task,
                cutoff_cycle=payload.cutoff_cycle,
                route_role=payload.route_role,
                source_registration_id=(
                    payload.source_registration_id
                    or normalized_default_source
                ),
            )
            materialization = service.create(
                context,
                request,
                idempotency_key=idempotency_key,
            )
            dispatch = queue.enqueue(
                materialization_id=materialization.materialization_id
            )
            return CreateAdvancedCalibrationMaterializationResponse(
                materialization=materialization,
                dispatch=dispatch,
            )
        except ProjectInvocationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="advanced_calibration_not_found",
            ) from exc
        except ProjectInvocationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except AdvancedCalibrationMaterializationError as exc:
            raise HTTPException(
                status_code=409,
                detail="advanced_calibration_state_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="invalid_advanced_calibration",
            ) from exc

    @router.get(
        "",
        response_model=list[AdvancedCalibrationMaterializationRecord],
    )
    def list_materializations(
        project_id: str,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            context = context_service.resolve_http(principal, project_id)
            return list(service.list(context))
        except (
            ProjectInvocationNotFoundError,
            AdvancedCalibrationMaterializationError,
        ) as exc:
            raise HTTPException(
                status_code=404,
                detail="advanced_calibration_not_found",
            ) from exc
        except ProjectInvocationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc

    @router.get(
        "/{materialization_id}",
        response_model=AdvancedCalibrationMaterializationRecord,
    )
    def get_materialization(
        project_id: str,
        materialization_id: str,
        principal: AuthPrincipal = operator_principal,
    ) -> Any:
        try:
            context = context_service.resolve_http(principal, project_id)
            return service.get(
                context,
                materialization_id=materialization_id,
            )
        except (
            ProjectInvocationNotFoundError,
            AdvancedCalibrationMaterializationError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=404,
                detail="advanced_calibration_not_found",
            ) from exc
        except ProjectInvocationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc

    return AdvancedCalibrationHttpAdapter(router=router)


__all__ = [
    "DEFAULT_ADVANCED_CALIBRATION_SOURCE_REGISTRATION_ID",
    "AdvancedCalibrationHttpAdapter",
    "CreateAdvancedCalibrationMaterializationRequest",
    "CreateAdvancedCalibrationMaterializationResponse",
    "create_advanced_calibration_http_adapter",
]
