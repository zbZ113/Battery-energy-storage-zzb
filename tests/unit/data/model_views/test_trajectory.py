from quanxin_life.data.model_views.schemas import ModelViewRow


def test_trajectory_row_keeps_unavailable_target_explicit() -> None:
    row = ModelViewRow(
        entity_id="cell-1",
        split="calibration",
        features={"cycle": 100.0},
        feature_mask={"cycle": True},
        target=None,
        right_censored=True,
        evidence="OBSERVED_NO_TARGET",
    )

    assert row.target is None
    assert row.right_censored is True
