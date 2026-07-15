"""FastAPI adapter for project creation and object-authorized reads."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.projects import (
    ProjectAccessError,
    ProjectNotFoundError,
    ProjectRecord,
    ProjectService,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel


class CreateProjectRequest(ContractModel):
    name: str = Field(min_length=1, max_length=200)


@dataclass(frozen=True, slots=True)
class ProjectHttpAdapter:
    router: APIRouter


def create_project_http_adapter(
    service: ProjectService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> ProjectHttpAdapter:
    """Build project routes over the existing authenticated principal dependencies."""

    router = APIRouter(prefix="/v1/projects", tags=["projects"])
    operator_dependency = auth_adapter.require_roles(
        {UserRole.ADMIN, UserRole.MEMBER}
    )

    @router.post(
        "",
        response_model=ProjectRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def create_project(
        payload: CreateProjectRequest,
        principal: Annotated[AuthPrincipal, Depends(operator_dependency)],
    ) -> Any:
        try:
            return service.create_project(
                principal,
                name=payload.name,
                now=datetime.now(UTC),
            )
        except ProjectAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_project") from exc

    @router.get("", response_model=list[ProjectRecord])
    def list_projects(
        principal: Annotated[AuthPrincipal, Depends(auth_adapter.require_ready_user)],
    ) -> Any:
        return list(service.list_projects(principal))

    @router.get("/{project_id}", response_model=ProjectRecord)
    def get_project(
        project_id: str,
        principal: Annotated[AuthPrincipal, Depends(auth_adapter.require_ready_user)],
    ) -> Any:
        try:
            return service.get_project(principal, project_id)
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc

    return ProjectHttpAdapter(router=router)


__all__ = ["ProjectHttpAdapter", "create_project_http_adapter"]
