"""FastAPI adapter for the governed, project-scoped model artifact catalog."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.model_artifact_catalog import (
    ModelArtifactCatalogAccessError,
    ModelArtifactCatalogNotFoundError,
    ModelArtifactCatalogRecord,
    ModelArtifactCatalogService,
    ModelArtifactCatalogSourceError,
    ModelArtifactCatalogStateError,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel


class RegisterModelArtifactRequest(ContractModel):
    """Caller-selected scope and opaque identity; source context stays server-side."""

    project_id: str = Field(min_length=1, max_length=64)
    artifact_id: str = Field(min_length=1, max_length=64)


@dataclass(frozen=True, slots=True)
class ModelArtifactHttpAdapter:
    router: APIRouter


def create_model_artifact_http_adapter(
    service: ModelArtifactCatalogService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> ModelArtifactHttpAdapter:
    """Build authenticated routes without accepting paths, hashes, or status."""

    router = APIRouter(tags=["model-artifacts"])
    ready_user = auth_adapter.require_ready_user
    admin = auth_adapter.require_roles({UserRole.ADMIN})

    @router.post(
        "/v1/admin/model-artifacts",
        response_model=ModelArtifactCatalogRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def register_model_artifact(
        payload: RegisterModelArtifactRequest,
        principal: Annotated[AuthPrincipal, Depends(admin)],
    ) -> Any:
        try:
            return service.register_artifact(
                principal,
                project_id=payload.project_id,
                artifact_id=payload.artifact_id,
                registered_at=datetime.now(UTC),
            )
        except ModelArtifactCatalogAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except ModelArtifactCatalogNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except ModelArtifactCatalogSourceError as exc:
            raise HTTPException(
                status_code=409,
                detail="model_artifact_source_unavailable",
            ) from exc
        except ModelArtifactCatalogStateError as exc:
            raise HTTPException(
                status_code=409,
                detail="model_artifact_catalog_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="invalid_model_artifact",
            ) from exc

    @router.get(
        "/v1/model-artifacts",
        response_model=list[ModelArtifactCatalogRecord],
    )
    def list_model_artifacts(
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
        project_id: Annotated[str | None, Query(max_length=64)] = None,
        artifact_kind: Annotated[str | None, Query(max_length=100)] = None,
        artifact_format: Annotated[str | None, Query(max_length=50)] = None,
        dataset_id: Annotated[str | None, Query(max_length=100)] = None,
        cutoff_cycle: Annotated[int | None, Query(ge=0)] = None,
    ) -> Any:
        try:
            return list(
                service.list_artifacts(
                    principal,
                    project_id=project_id,
                    artifact_kind=artifact_kind,
                    artifact_format=artifact_format,
                    dataset_id=dataset_id,
                    cutoff_cycle=cutoff_cycle,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_filter") from exc

    @router.get(
        "/v1/model-artifacts/{artifact_id}",
        response_model=ModelArtifactCatalogRecord,
    )
    def get_model_artifact(
        artifact_id: str,
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
    ) -> Any:
        try:
            return service.get_artifact(principal, artifact_id)
        except ModelArtifactCatalogNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="model_artifact_not_found",
            ) from exc
        except ModelArtifactCatalogStateError as exc:
            raise HTTPException(
                status_code=500,
                detail="invalid_model_artifact_state",
            ) from exc

    return ModelArtifactHttpAdapter(router=router)


__all__ = [
    "ModelArtifactHttpAdapter",
    "create_model_artifact_http_adapter",
]
