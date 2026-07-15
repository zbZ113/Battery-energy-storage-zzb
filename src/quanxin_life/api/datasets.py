"""FastAPI adapter for project-scoped dataset registration and freezing."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.datasets import (
    DatasetAccessError,
    DatasetNotFoundError,
    DatasetRecord,
    DatasetService,
    DatasetStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel


class CreateDatasetRequest(ContractModel):
    project_id: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    data_version: str = Field(min_length=1, max_length=100)
    schema_version: str = Field(min_length=1, max_length=100)


@dataclass(frozen=True, slots=True)
class DatasetHttpAdapter:
    router: APIRouter


def create_dataset_http_adapter(
    service: DatasetService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> DatasetHttpAdapter:
    """Build dataset routes over authenticated project-scoped services."""

    router = APIRouter(prefix="/v1/datasets", tags=["datasets"])
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )

    @router.post(
        "",
        response_model=DatasetRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_dataset(
        payload: CreateDatasetRequest,
        principal: Annotated[AuthPrincipal, Depends(operator_dependency)],
    ) -> Any:
        try:
            return service.create_dataset(
                principal,
                project_id=payload.project_id,
                name=payload.name,
                data_version=payload.data_version,
                schema_version=payload.schema_version,
                now=datetime.now(UTC),
            )
        except DatasetAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_dataset") from exc

    @router.get("/{dataset_id}", response_model=DatasetRecord)
    def get_dataset(
        dataset_id: str,
        principal: Annotated[AuthPrincipal, Depends(auth_adapter.require_ready_user)],
    ) -> Any:
        try:
            return service.get_dataset(principal, dataset_id)
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="dataset_not_found") from exc
        except DatasetStateError as exc:
            raise HTTPException(status_code=500, detail="invalid_dataset_state") from exc

    @router.post(
        "/{dataset_id}/freeze",
        response_model=DatasetRecord,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def freeze_dataset(
        dataset_id: str,
        principal: Annotated[AuthPrincipal, Depends(operator_dependency)],
    ) -> Any:
        try:
            return service.freeze_dataset(
                principal,
                dataset_id,
                now=datetime.now(UTC),
            )
        except DatasetAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except DatasetNotFoundError as exc:
            raise HTTPException(status_code=404, detail="dataset_not_found") from exc
        except DatasetStateError as exc:
            raise HTTPException(status_code=409, detail="dataset_state_conflict") from exc

    return DatasetHttpAdapter(router=router)


__all__ = ["DatasetHttpAdapter", "create_dataset_http_adapter"]
