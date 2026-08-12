from __future__ import annotations

import pytest

from quanxin_life.core import EvidenceLevel
from quanxin_life.scenarios import (
    BLAST_LITE_UPSTREAM_COMMIT,
    BlastRouteRejected,
    load_packaged_blast_route_catalog,
)


def test_packaged_blast_routes_preserve_upstream_and_candidate_status() -> None:
    catalog = load_packaged_blast_route_catalog()

    assert catalog.upstream_repository == "https://github.com/NREL/BLAST-Lite.git"
    assert catalog.upstream_commit == BLAST_LITE_UPSTREAM_COMMIT
    assert catalog.upstream_version == "1.1.0"
    assert catalog.license_spdx == "BSD-3-Clause"
    assert len(catalog.routes) == 2
    assert all(route.lifecycle_status == "REGISTERED_CANDIDATE" for route in catalog.routes)
    assert all(route.activation_status == "NOT_ACTIVATED" for route in catalog.routes)
    assert all(route.evidence_level is EvidenceLevel.PHYSICS_REFERENCE for route in catalog.routes)


def test_route_catalog_rejects_non_lfp_chemistry() -> None:
    catalog = load_packaged_blast_route_catalog()

    with pytest.raises(BlastRouteRejected, match="CHEMISTRY_NOT_SUPPORTED"):
        catalog.authorize_reference_use(
            route_id="blast-lite-lfp-gr-sony-murata-3ah-2018-v1",
            chemistry="NMC/graphite",
            nominal_capacity_ah=3.0,
            cell_format="cylindrical",
            trusted_reference_use=False,
        )


@pytest.mark.parametrize(
    ("route_id", "capacity_ah", "cell_format"),
    [
        ("blast-lite-lfp-gr-sony-murata-3ah-2018-v1", 250.0, "prismatic"),
        ("blast-lite-lfp-gr-250ah-prismatic-2019-v1", 3.0, "cylindrical"),
    ],
)
def test_route_catalog_rejects_3ah_and_250ah_misrouting(
    route_id: str,
    capacity_ah: float,
    cell_format: str,
) -> None:
    catalog = load_packaged_blast_route_catalog()

    with pytest.raises(BlastRouteRejected, match="CELL_REFERENCE_NOT_SUPPORTED"):
        catalog.authorize_reference_use(
            route_id=route_id,
            chemistry="LFP/graphite",
            nominal_capacity_ah=capacity_ah,
            cell_format=cell_format,
            trusted_reference_use=False,
        )


def test_280ah_reference_requires_server_side_approval() -> None:
    catalog = load_packaged_blast_route_catalog()
    route_id = "blast-lite-lfp-gr-250ah-prismatic-2019-v1"

    with pytest.raises(BlastRouteRejected, match="REFERENCE_USE_NOT_APPROVED"):
        catalog.authorize_reference_use(
            route_id=route_id,
            chemistry="LFP/graphite",
            nominal_capacity_ah=280.0,
            cell_format="prismatic",
            trusted_reference_use=False,
        )

    authorized = catalog.authorize_reference_use(
        route_id=route_id,
        chemistry="LFP/graphite",
        nominal_capacity_ah=280.0,
        cell_format="prismatic",
        trusted_reference_use=True,
    )
    assert authorized.route_id == route_id
    assert "REFERENCE_MODEL_NOT_CELL_SPECIFIC" in authorized.warnings
