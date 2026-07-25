"""FastAPI adapter for append-only Advanced model route decisions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.model_route_activation import (
    ModelRouteActivationAccessError,
    ModelRouteActivationEventRecord,
    ModelRouteActivationNotFoundError,
    ModelRouteActivationService,
    ModelRouteActivationStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import AdvancedModelRouteRole, AdvancedModelTask, UserRole
from quanxin_life.core.schemas import ContractModel, Sha256


class ActivateModelRouteRequest(ContractModel):
    """Manual activation input; provenance remains server-derived."""

    project_id: str = Field(min_length=1, max_length=64)
    task: AdvancedModelTask
    cutoff_cycle: int = Field(gt=0)
    role: AdvancedModelRouteRole
    artifact_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    expected_previous_event_sha256: Sha256


class RollbackModelRouteRequest(ContractModel):
    """Manual rollback input; the target artifact is derived from history."""

    project_id: str = Field(min_length=1, max_length=64)
    task: AdvancedModelTask
    cutoff_cycle: int = Field(gt=0)
    role: AdvancedModelRouteRole
    target_activation_event_id: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    expected_previous_event_sha256: Sha256


@dataclass(frozen=True, slots=True)
class ModelRouteHttpAdapter:
    router: APIRouter


def create_model_route_http_adapter(
    service: ModelRouteActivationService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> ModelRouteHttpAdapter:
    """Build authenticated route-decision endpoints around the domain service."""

    router = APIRouter(tags=["model-routes"])
    ready_user = auth_adapter.require_ready_user
    admin = auth_adapter.require_roles({UserRole.ADMIN})
    idempotency_header = Header(
        alias="Idempotency-Key",
        min_length=8,
        max_length=200,
    )

    @router.post(
        "/v1/admin/model-routes/activations",
        response_model=ModelRouteActivationEventRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def activate_model_route(
        payload: ActivateModelRouteRequest,
        principal: Annotated[AuthPrincipal, Depends(admin)],
        idempotency_key: str = idempotency_header,
    ) -> Any:
        try:
            return service.activate(
                principal,
                project_id=payload.project_id,
                task=payload.task,
                cutoff_cycle=payload.cutoff_cycle,
                role=payload.role,
                artifact_id=payload.artifact_id,
                reason=payload.reason,
                idempotency_key=idempotency_key,
                expected_previous_event_sha256=(
                    payload.expected_previous_event_sha256
                ),
                occurred_at=datetime.now(UTC),
            )
        except ModelRouteActivationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except ModelRouteActivationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="model_route_not_found") from exc
        except ModelRouteActivationStateError as exc:
            raise HTTPException(
                status_code=409,
                detail="model_route_state_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_model_route") from exc

    @router.post(
        "/v1/admin/model-routes/rollbacks",
        response_model=ModelRouteActivationEventRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def rollback_model_route(
        payload: RollbackModelRouteRequest,
        principal: Annotated[AuthPrincipal, Depends(admin)],
        idempotency_key: str = idempotency_header,
    ) -> Any:
        try:
            return service.rollback(
                principal,
                project_id=payload.project_id,
                task=payload.task,
                cutoff_cycle=payload.cutoff_cycle,
                role=payload.role,
                target_activation_event_id=payload.target_activation_event_id,
                reason=payload.reason,
                idempotency_key=idempotency_key,
                expected_previous_event_sha256=(
                    payload.expected_previous_event_sha256
                ),
                occurred_at=datetime.now(UTC),
            )
        except ModelRouteActivationAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except ModelRouteActivationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="model_route_not_found") from exc
        except ModelRouteActivationStateError as exc:
            raise HTTPException(
                status_code=409,
                detail="model_route_state_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_model_route") from exc

    @router.get(
        "/v1/model-routes/activation-events",
        response_model=list[ModelRouteActivationEventRecord],
    )
    def list_model_route_events(
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
        project_id: Annotated[str, Query(min_length=1, max_length=64)],
        task: AdvancedModelTask,
        cutoff_cycle: Annotated[int, Query(gt=0)],
        role: AdvancedModelRouteRole,
    ) -> Any:
        try:
            return list(
                service.list_events(
                    principal,
                    project_id=project_id,
                    task=task,
                    cutoff_cycle=cutoff_cycle,
                    role=role,
                )
            )
        except ModelRouteActivationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="model_route_not_found") from exc
        except ModelRouteActivationStateError as exc:
            raise HTTPException(
                status_code=500,
                detail="invalid_model_route_ledger",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_model_route") from exc

    return ModelRouteHttpAdapter(router=router)


__all__ = ["ModelRouteHttpAdapter", "create_model_route_http_adapter"]
