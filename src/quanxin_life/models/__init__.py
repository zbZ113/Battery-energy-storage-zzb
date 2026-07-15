"""Reproducible battery lifetime model implementations."""

from typing import TYPE_CHECKING

from quanxin_life.models.dummy import DummyLifePredictor
from quanxin_life.models.metrics import evaluate_eol80_predictions

if TYPE_CHECKING:
    from quanxin_life.models.cpmlp import CPMLPLifePredictor
    from quanxin_life.models.hybrid_degradation import HybridDegradationPredictor
    from quanxin_life.models.variance import VarianceLifePredictor
    from quanxin_life.models.xgboost import XGBoostLifePredictor

__all__ = [
    "CPMLPLifePredictor",
    "DummyLifePredictor",
    "HybridDegradationPredictor",
    "VarianceLifePredictor",
    "XGBoostLifePredictor",
    "evaluate_eol80_predictions",
]


def __getattr__(name: str) -> object:
    """Load optional ML implementations only when a caller explicitly requests them."""
    if name == "CPMLPLifePredictor":
        from quanxin_life.models.cpmlp import CPMLPLifePredictor

        return CPMLPLifePredictor
    if name == "XGBoostLifePredictor":
        from quanxin_life.models.xgboost import XGBoostLifePredictor

        return XGBoostLifePredictor
    if name == "HybridDegradationPredictor":
        from quanxin_life.models.hybrid_degradation import HybridDegradationPredictor

        return HybridDegradationPredictor
    if name == "VarianceLifePredictor":
        from quanxin_life.models.variance import VarianceLifePredictor

        return VarianceLifePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
