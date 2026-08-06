from quanxin_life.data.model_views.builder import fit_normalizer
from quanxin_life.data.model_views.schemas import ModelViewRow


def test_normalizer_uses_train_rows_only() -> None:
    rows = (
        ModelViewRow(
            entity_id="train-1",
            split="train",
            features={"x": 1.0},
            feature_mask={"x": True},
            evidence="NO_POINT_TARGET:NORMALIZER_TEST",
        ),
        ModelViewRow(
            entity_id="test-1",
            split="test",
            features={"x": 100.0},
            feature_mask={"x": True},
            evidence="NO_POINT_TARGET:NORMALIZER_TEST",
        ),
    )

    normalizer = fit_normalizer(rows, feature_names=("x",))

    assert normalizer.means["x"] == 1.0
