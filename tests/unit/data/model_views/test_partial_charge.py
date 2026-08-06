from quanxin_life.data.model_views.partial_charge import partial_charge_row


def test_partial_charge_missing_feature_has_explicit_mask() -> None:
    row = partial_charge_row("cell-1", split="validation", features={"slope": None})

    assert row.features["slope"] is None
    assert row.feature_mask["slope"] is False
