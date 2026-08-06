from quanxin_life.data.model_views.degradation_conditions import condition_view_row


def test_condition_view_uses_condition_identity() -> None:
    row = condition_view_row("T40_SOC50", split="train", temperature_c=40.0, mean_soc=0.5)

    assert row.entity_id == "T40_SOC50"
    assert row.target is None
