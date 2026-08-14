from __future__ import annotations

import warnings
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.scenario_plot import (
    SCENARIO_PLOT_FONT_SHA256,
    FeishuScenarioPlotter,
)
from quanxin_life.tools.blast_scenarios import (
    COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
)

NOW = datetime(2026, 8, 11, 13, 0, tzinfo=UTC)


def _projection(*, scenario_id: str, soh: list[float]) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "scenario_version": f"{scenario_id}-v1",
        "natural_years": [0.0, 0.5, 1.0],
        "equivalent_full_cycles": [0.0, 60.0, 120.0],
        "soh": soh,
        "eol_threshold": 0.8,
        "eol": {
            "status": "NOT_REACHED",
            "natural_year": None,
            "equivalent_full_cycles": None,
        },
        "support": {"status": "SUPPORTED"},
    }


def _result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="compare_operation_scenarios",
        tool_version=COMPARE_OPERATION_SCENARIOS_TOOL_VERSION,
        model_version="blast-lite-route-v1",
        data_version="scenario-data-v1",
        feature_version="blast-scenario-v1",
        input_hash="a" * 64,
        values={
            "artifact": {
                "status": "COMPLETED",
                "baseline": _projection(
                    scenario_id="baseline",
                    soh=[1.0, 0.96, 0.92],
                ),
                "comparisons": [
                    _projection(
                        scenario_id="warmer",
                        soh=[1.0, 0.94, 0.88],
                    )
                ],
            }
        },
        warnings=["CANDIDATE_ROUTE_RESEARCH_USE_ONLY"],
        provenance=[
            ProvenanceRecord(
                source_id="scenario-source",
                source_kind=SourceKind.OBSERVED,
                uri="test://scenario/source",
                sha256="b" * 64,
                description="Verified scenario plot fixture",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_scenario_plotter_renders_deterministic_png_from_tool_result_only() -> None:
    result = _result()
    plotter = FeishuScenarioPlotter()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first = plotter.render(result)
        second = plotter.render(result)

    assert first.source_result_id == result.result_id
    assert first.media_type == "image/png"
    assert first.filename == f"scenario-{result.result_id}.png"
    assert first.payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.sha256 == sha256(first.payload).hexdigest()
    assert second.sha256 == first.sha256
    assert plotter.renderer_version == "feishu-scenario-plot-v2"
    assert SCENARIO_PLOT_FONT_SHA256 == (
        "b981c60b7f35d4730109fd6d0baee61d7b5bf29ea488cd2ddec87554b50fe457"
    )
