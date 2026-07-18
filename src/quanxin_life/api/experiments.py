"""FastAPI adapter for project-scoped verified experiment metadata."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import Field

from quanxin_life.api.auth import AuthHttpAdapter
from quanxin_life.application.experiments import (
    ExperimentAccessError,
    ExperimentNotFoundError,
    ExperimentRegistryService,
    ExperimentRunRecord,
    ExperimentSourceError,
    ExperimentStateError,
    ExperimentSuiteRecord,
)
from quanxin_life.auth import AuthPrincipal
from quanxin_life.core import UserRole
from quanxin_life.core.schemas import ContractModel, Sha256


class RegisterExperimentRequest(ContractModel):
    project_id: str = Field(min_length=1, max_length=64)
    import_id: Sha256


@dataclass(frozen=True, slots=True)
class ExperimentHttpAdapter:
    router: APIRouter


def create_experiment_http_adapter(
    service: ExperimentRegistryService,
    *,
    auth_adapter: AuthHttpAdapter,
) -> ExperimentHttpAdapter:
    """Build authenticated routes without exposing local evidence paths or metrics."""

    router = APIRouter(tags=["experiments"])
    ready_user = auth_adapter.require_ready_user
    operator = auth_adapter.require_roles({UserRole.ADMIN, UserRole.MEMBER})

    @router.post(
        "/v1/experiments",
        response_model=ExperimentSuiteRecord,
        status_code=201,
        dependencies=[Depends(auth_adapter.require_trusted_origin)],
    )
    def register_experiment(
        payload: RegisterExperimentRequest,
        principal: Annotated[AuthPrincipal, Depends(operator)],
    ) -> Any:
        try:
            return service.register_suite(
                principal,
                project_id=payload.project_id,
                import_id=payload.import_id,
                registered_at=datetime.now(UTC),
            )
        except ExperimentAccessError as exc:
            raise HTTPException(status_code=403, detail="role_not_allowed") from exc
        except ExperimentNotFoundError as exc:
            raise HTTPException(status_code=404, detail="project_not_found") from exc
        except ExperimentSourceError as exc:
            raise HTTPException(
                status_code=409,
                detail="experiment_source_unavailable",
            ) from exc
        except ExperimentStateError as exc:
            raise HTTPException(
                status_code=409,
                detail="experiment_registry_conflict",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid_experiment") from exc

    @router.get("/v1/experiments", response_model=list[ExperimentSuiteRecord])
    def list_experiments(
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
        project_id: Annotated[str | None, Query(max_length=64)] = None,
        dataset_id: Annotated[str | None, Query(max_length=100)] = None,
        mode: Literal["smoke", "final"] | None = None,
    ) -> Any:
        try:
            return list(
                service.list_suites(
                    principal,
                    project_id=project_id,
                    dataset_id=dataset_id,
                    mode=mode,
                )
            )
        except ExperimentNotFoundError as exc:
            raise HTTPException(status_code=422, detail="invalid_filter") from exc

    @router.get("/v1/experiment-runs", response_model=list[ExperimentRunRecord])
    def list_experiment_runs(
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
        project_id: Annotated[str | None, Query(max_length=64)] = None,
        experiment_id: Annotated[str | None, Query(max_length=64)] = None,
        dataset_id: Annotated[str | None, Query(max_length=100)] = None,
        model_name: Annotated[str | None, Query(max_length=100)] = None,
        cutoff_cycle: Annotated[int | None, Query(gt=0)] = None,
        seed: Annotated[int | None, Query(gt=0)] = None,
    ) -> Any:
        try:
            return list(
                service.list_runs(
                    principal,
                    project_id=project_id,
                    experiment_id=experiment_id,
                    dataset_id=dataset_id,
                    model_name=model_name,
                    cutoff_cycle=cutoff_cycle,
                    seed=seed,
                )
            )
        except (ExperimentNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="invalid_filter") from exc

    @router.get(
        "/v1/experiments/{experiment_id}",
        response_model=ExperimentSuiteRecord,
    )
    def get_experiment(
        experiment_id: str,
        principal: Annotated[AuthPrincipal, Depends(ready_user)],
    ) -> Any:
        try:
            return service.get_suite(principal, experiment_id)
        except ExperimentNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="experiment_not_found",
            ) from exc
        except ExperimentStateError as exc:
            raise HTTPException(
                status_code=500,
                detail="invalid_experiment_state",
            ) from exc

    return ExperimentHttpAdapter(router=router)


__all__ = ["ExperimentHttpAdapter", "create_experiment_http_adapter"]
