from __future__ import annotations

import json
from importlib.resources import files

import numpy as np
import pytest

from quanxin_life._vendor.blast_lite import (
    Lfp_Gr_250AhPrismatic,
    Lfp_Gr_SonyMurata3Ah_Battery,
)


def _one_day_one_efc_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_s = np.arange(25, dtype=float) * 3600.0
    soc = np.concatenate(
        (
            np.linspace(0.0, 1.0, 13, dtype=float),
            np.linspace(11.0 / 12.0, 0.0, 12, dtype=float),
        )
    )
    temperature_c = np.full(time_s.shape, 25.0, dtype=float)
    return time_s, soc, temperature_c


def test_vendored_blast_preserves_license_notice_and_provenance() -> None:
    package = files("quanxin_life._vendor.blast_lite")

    assert "Redistribution and use in source and binary forms" in package.joinpath(
        "LICENSE"
    ).read_text(encoding="utf-8")
    assert "BLAST-Python" in package.joinpath("NOTICE").read_text(encoding="utf-8")
    provenance = json.loads(package.joinpath("UPSTREAM.json").read_text(encoding="utf-8"))
    assert provenance["repository"] == "https://github.com/NREL/BLAST-Lite.git"
    assert provenance["commit"] == "b093495b47dc40dd96dba865d91f553619501e94"
    assert provenance["license_spdx"] == "BSD-3-Clause"
    assert provenance["modifications"] == [
        "Replaced removed numpy.trapz calls with numpy.trapezoid for NumPy 2.x.",
        "Replaced upstream absolute imports with package-relative imports.",
        "Removed an unused matplotlib plotting import from the runtime core.",
        "Vendored only the two reviewed LFP models and their minimal runtime dependencies.",
    ]


@pytest.mark.parametrize(
    ("battery_type", "expected_q", "expected_r"),
    [
        (Lfp_Gr_250AhPrismatic, 0.9994229022207307, None),
        (
            Lfp_Gr_SonyMurata3Ah_Battery,
            0.9981474176497881,
            1.0000805355688578,
        ),
    ],
)
def test_vendored_blast_matches_pinned_upstream_one_day_golden_case(
    battery_type: type[Lfp_Gr_250AhPrismatic]
    | type[Lfp_Gr_SonyMurata3Ah_Battery],
    expected_q: float,
    expected_r: float | None,
) -> None:
    time_s, soc, temperature_c = _one_day_one_efc_profile()
    battery = battery_type()

    battery.simulate_battery_life(
        {
            "Time_s": time_s,
            "SOC": soc,
            "Temperature_C": temperature_c,
        }
    )

    assert battery.outputs["q"][-1] == pytest.approx(expected_q, abs=1e-12)
    assert battery.stressors["efc"][-1] == pytest.approx(1.0, abs=1e-12)
    assert battery.stressors["t_days"][-1] == pytest.approx(1.0, abs=1e-12)
    assert battery.stressors["Crate"][-1] == pytest.approx(
        0.07986111111111112,
        abs=1e-12,
    )
    if expected_r is not None:
        assert battery.outputs["r"][-1] == pytest.approx(expected_r, abs=1e-12)
