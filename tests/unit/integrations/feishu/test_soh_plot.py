from __future__ import annotations

import warnings
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult
from quanxin_life.integrations.feishu.soh_plot import (
    SOH_PLOT_FONT_SHA256,
    FeishuSohPlotError,
    FeishuSohPlotter,
)

NOW = datetime(2026, 8, 13, 8, 0, tzinfo=UTC)


def _result() -> ToolResult:
    return ToolResult(
        result_id=str(uuid4()),
        tool_name="predict_soh_trajectory",
        tool_version="advanced-soh-prediction-tool-v1",
        model_version="hybridpatch-v2-cutoff-50-seed-38",
        data_version="matr-three-batch-v1",
        feature_version="cyclepatch-multichannel-v1",
        input_hash="a" * 64,
        values={
            "artifact_type": "quanxin_life.advanced_soh_trajectory.v1",
            "artifact": {
                "cell_id": "MATR_b3c34",
                "cutoff_cycle": 50,
                "prediction_cycles": [51, 250, 500],
                "predicted_soh": [0.99, 0.94, 0.87],
                "horizon_end_cycle": 500,
            },
        },
        uncertainty={
            "finite_horizon_only": True,
            "conformal_interval_included": False,
        },
        warnings=[],
        provenance=[
            ProvenanceRecord(
                source_id="advanced-model",
                source_kind=SourceKind.PREDICTED,
                uri="artifact://advanced-model/test",
                sha256="b" * 64,
                description="Verified SOH plot fixture",
                created_at=NOW,
            )
        ],
        created_at=NOW,
    )


def test_soh_plotter_renders_deterministic_png_from_tool_result_only() -> None:
    result = _result()
    plotter = FeishuSohPlotter()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first = plotter.render(result)
        second = plotter.render(result)

    assert first.source_result_id == result.result_id
    assert first.filename == f"soh-trajectory-{result.result_id}.png"
    assert first.media_type == "image/png"
    assert first.payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert first.sha256 == sha256(first.payload).hexdigest()
    assert second.sha256 == first.sha256
    assert SOH_PLOT_FONT_SHA256 == (
        "cd42dca9abc49fc97b6e5426afd8bcf87b6b002ace89bb1d03e9b2f4ecfa32d5"
    )


def test_soh_plotter_rejects_non_soh_or_tampered_axes() -> None:
    result = _result()
    plotter = FeishuSohPlotter()

    with pytest.raises(FeishuSohPlotError, match="supported SOH"):
        plotter.render(result.model_copy(update={"tool_name": "predict_cycle_life"}))
    with pytest.raises(FeishuSohPlotError, match="align"):
        plotter.render(
            result.model_copy(
                update={
                    "values": {
                        **result.values,
                        "artifact": {
                            **result.values["artifact"],
                            "predicted_soh": [0.99, 0.94],
                        },
                    }
                }
            )
        )
