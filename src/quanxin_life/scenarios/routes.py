"""Reviewed BLAST-Lite route metadata and reference-use authorization."""

from __future__ import annotations

import json
import math
from importlib.resources import files
from typing import Literal, Self

from pydantic import Field, model_validator

from quanxin_life.core import EvidenceLevel
from quanxin_life.core.schemas import ContractModel, Sha256

BLAST_LITE_UPSTREAM_COMMIT = "b093495b47dc40dd96dba865d91f553619501e94"
BLAST_ROUTE_MANIFEST_RESOURCE = "manifests/blast_lite_routes_v1.json"


class BlastRouteRejected(ValueError):
    """Raised when trusted cell metadata cannot use a requested reference route."""


class BlastExperimentalRange(ContractModel):
    cycling_temperature_c: tuple[float, float]
    dod: tuple[float, float]
    soc: tuple[float, float]
    max_rate_charge: float = Field(gt=0, allow_inf_nan=False)
    max_rate_discharge: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ranges_are_ordered(self) -> Self:
        for name in ("cycling_temperature_c", "dod", "soc"):
            lower, upper = getattr(self, name)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"{name} must contain two ordered finite bounds")
        return self


class BlastRouteManifest(ContractModel):
    route_id: str = Field(min_length=1)
    route_version: str = Field(min_length=1)
    task_types: tuple[
        Literal["compare_operation_scenarios", "project_storage_lifetime"], ...
    ] = Field(min_length=2)
    model_class: Literal[
        "Lfp_Gr_SonyMurata3Ah_Battery", "Lfp_Gr_250AhPrismatic"
    ]
    chemistry_aliases: tuple[str, ...] = Field(min_length=1)
    cell_format: Literal["cylindrical", "prismatic"]
    nominal_capacity_reference_ah: float = Field(gt=0, allow_inf_nan=False)
    capacity_policy: Literal["EXACT_REFERENCE_ONLY", "TRUSTED_REFERENCE_USE_ONLY"]
    experimental_range: BlastExperimentalRange
    upstream_model_source_sha256: Sha256
    data_sources: tuple[str, ...] = Field(min_length=1)
    lifecycle_status: Literal["REGISTERED_CANDIDATE"] = "REGISTERED_CANDIDATE"
    activation_status: Literal["NOT_ACTIVATED"] = "NOT_ACTIVATED"
    evidence_level: Literal[EvidenceLevel.PHYSICS_REFERENCE] = (
        EvidenceLevel.PHYSICS_REFERENCE
    )
    warnings: tuple[str, ...] = Field(min_length=1)


class BlastRouteCatalog(ContractModel):
    schema_version: Literal["blast-lite-route-catalog-v1"]
    upstream_repository: str = Field(min_length=1)
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    upstream_version: str = Field(min_length=1)
    license_spdx: Literal["BSD-3-Clause"]
    packaged_license_resource: str = Field(min_length=1)
    packaged_notice_resource: str = Field(min_length=1)
    routes: tuple[BlastRouteManifest, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def route_ids_are_unique_and_source_is_frozen(self) -> Self:
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("BLAST route IDs must be unique")
        if self.upstream_commit != BLAST_LITE_UPSTREAM_COMMIT:
            raise ValueError("BLAST route catalog does not match the reviewed upstream commit")
        return self

    def authorize_reference_use(
        self,
        *,
        route_id: str,
        chemistry: str,
        nominal_capacity_ah: float,
        cell_format: str,
        trusted_reference_use: bool,
    ) -> BlastRouteManifest:
        route = next((item for item in self.routes if item.route_id == route_id), None)
        if route is None:
            raise BlastRouteRejected("UNKNOWN_BLAST_ROUTE")
        normalized_chemistry = chemistry.strip().casefold()
        aliases = {alias.strip().casefold() for alias in route.chemistry_aliases}
        if normalized_chemistry not in aliases:
            raise BlastRouteRejected("CHEMISTRY_NOT_SUPPORTED")
        if cell_format.strip().casefold() != route.cell_format:
            raise BlastRouteRejected("CELL_REFERENCE_NOT_SUPPORTED")
        exact_reference = math.isclose(
            nominal_capacity_ah,
            route.nominal_capacity_reference_ah,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        if route.capacity_policy == "EXACT_REFERENCE_ONLY" and not exact_reference:
            raise BlastRouteRejected("CELL_REFERENCE_NOT_SUPPORTED")
        if (
            route.capacity_policy == "TRUSTED_REFERENCE_USE_ONLY"
            and not exact_reference
            and not trusted_reference_use
        ):
            raise BlastRouteRejected("REFERENCE_USE_NOT_APPROVED")
        return route


def load_packaged_blast_route_catalog() -> BlastRouteCatalog:
    resource = files("quanxin_life.scenarios").joinpath(BLAST_ROUTE_MANIFEST_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    return BlastRouteCatalog.model_validate(payload)


__all__ = [
    "BLAST_LITE_UPSTREAM_COMMIT",
    "BlastExperimentalRange",
    "BlastRouteCatalog",
    "BlastRouteManifest",
    "BlastRouteRejected",
    "load_packaged_blast_route_catalog",
]
