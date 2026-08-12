from __future__ import annotations

import pytest
from pydantic import ValidationError

from quanxin_life.scenarios import OperationScenario, ScenarioSegment


def _segment(
    *,
    segment_id: str = "segment-1",
    start_year: int = 0,
    end_year: int = 5,
    soc_lower_bound: float = 0.1,
    soc_upper_bound: float = 0.9,
    dod: float = 0.8,
) -> ScenarioSegment:
    return ScenarioSegment(
        segment_id=segment_id,
        start_year=start_year,
        end_year=end_year,
        temperature_c=25.0,
        charge_c_rate=0.5,
        discharge_c_rate=0.5,
        soc_lower_bound=soc_lower_bound,
        soc_upper_bound=soc_upper_bound,
        dod=dod,
        equivalent_full_cycles_per_year=300.0,
        rest_duration_hours=1.0,
    )


def test_scenario_segment_rejects_soc_window_dod_conflict() -> None:
    with pytest.raises(ValidationError, match="DoD must equal the declared SOC window"):
        _segment(dod=0.7)


@pytest.mark.parametrize(
    "segments",
    [
        (_segment(end_year=2), _segment(segment_id="segment-2", start_year=3)),
        (_segment(end_year=3), _segment(segment_id="segment-2", start_year=2)),
    ],
)
def test_operation_scenario_rejects_segment_gaps_or_overlaps(
    segments: tuple[ScenarioSegment, ...],
) -> None:
    with pytest.raises(ValidationError, match="contiguous"):
        OperationScenario(
            scenario_id="invalid-stages",
            scenario_version="scenario-v1",
            horizon_years=5,
            eol_threshold=0.8,
            segments=segments,
        )


def test_operation_scenario_rejects_horizon_over_25_years() -> None:
    with pytest.raises(ValidationError):
        OperationScenario(
            scenario_id="too-long",
            scenario_version="scenario-v1",
            horizon_years=26,
            eol_threshold=0.8,
            segments=(_segment(end_year=26),),
        )


def test_operation_scenario_requires_segments_to_cover_the_horizon() -> None:
    with pytest.raises(ValidationError, match="cover the full horizon"):
        OperationScenario(
            scenario_id="short-stage",
            scenario_version="scenario-v1",
            horizon_years=10,
            eol_threshold=0.8,
            segments=(_segment(end_year=5),),
        )
