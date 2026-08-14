from __future__ import annotations

from hashlib import sha256

import pytest

from quanxin_life.integrations.feishu.analysis_plots import (
    ANALYSIS_PLOT_DISPATCHER_VERSION,
    AnalysisPlotTemplate,
    FeishuAnalysisPlotError,
    FeishuAnalysisPlotter,
)
from tests.unit.integrations.feishu.test_audited_cards import (
    _cycle_life_result,
)
from tests.unit.integrations.feishu.test_scenario_plot import _result as _scenario_result
from tests.unit.integrations.feishu.test_soh_plot import _result as _soh_result


def test_selects_and_renders_only_the_controlled_template_for_each_result_type() -> None:
    plotter = FeishuAnalysisPlotter()

    assert plotter.dispatcher_version == ANALYSIS_PLOT_DISPATCHER_VERSION
    assert plotter.renderer_version == "feishu-soh-plot-v1"

    soh_plan = plotter.plan(_soh_result())
    scenario_plan = plotter.plan(_scenario_result())
    cycle_plan = plotter.plan(_cycle_life_result())

    assert soh_plan.template is AnalysisPlotTemplate.FINITE_SOH_CURVE
    assert soh_plan.image_required is True
    assert scenario_plan.template is AnalysisPlotTemplate.SCENARIO_COMPARISON
    assert scenario_plan.image_required is True
    assert cycle_plan.template is AnalysisPlotTemplate.CYCLE_LIFE_SUMMARY
    assert cycle_plan.image_required is False

    soh_plot = plotter.render(_soh_result())
    scenario_plot = plotter.render(_scenario_result())

    assert soh_plot.template is AnalysisPlotTemplate.FINITE_SOH_CURVE
    assert scenario_plot.template is AnalysisPlotTemplate.SCENARIO_COMPARISON
    assert soh_plot.payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert scenario_plot.payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert soh_plot.sha256 == sha256(soh_plot.payload).hexdigest()
    assert scenario_plot.sha256 == sha256(scenario_plot.payload).hexdigest()
    assert soh_plot.renderer_version
    assert scenario_plot.renderer_version


def test_rejects_incompatible_or_unneeded_template_requests_without_reading_numbers() -> None:
    plotter = FeishuAnalysisPlotter()
    cycle = _cycle_life_result()

    with pytest.raises(FeishuAnalysisPlotError, match="does not require"):
        plotter.render(cycle)
    with pytest.raises(FeishuAnalysisPlotError, match="not allowed"):
        plotter.render(
            _soh_result(),
            template=AnalysisPlotTemplate.SCENARIO_COMPARISON,
        )
    with pytest.raises(TypeError):
        plotter.render(  # type: ignore[call-arg]
            _soh_result(),
            template=AnalysisPlotTemplate.FINITE_SOH_CURVE,
            values=[0.99, 0.87],
        )

    damaged_cycle = cycle.model_copy(
        update={
            "values": {
                **cycle.values,
                "artifact": {
                    key: value
                    for key, value in cycle.values["artifact"].items()
                    if key != "cell_metadata"
                },
            }
        }
    )
    with pytest.raises(FeishuAnalysisPlotError, match="metadata"):
        plotter.plan(damaged_cycle)
