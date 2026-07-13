"""Leakage-safe cross-domain adaptation methods for battery feature spaces."""

from typing import TYPE_CHECKING

from quanxin_life.adaptation.coral import CORALFeatureAdapter

if TYPE_CHECKING:
    from quanxin_life.adaptation.dann import (
        CPMLPDANNAdapter,
        CurveFeatureContract,
        DANNConfig,
        DANNPrediction,
        SourceDomainBatch,
        TargetDomainBatch,
        TargetDomainCohort,
        gradient_reverse,
    )

__all__ = [
    "CORALFeatureAdapter",
    "CPMLPDANNAdapter",
    "CurveFeatureContract",
    "DANNConfig",
    "DANNPrediction",
    "SourceDomainBatch",
    "TargetDomainBatch",
    "TargetDomainCohort",
    "gradient_reverse",
]


def __getattr__(name: str) -> object:
    """Load optional PyTorch DANN support only when it is explicitly requested."""

    dann_exports = {
        "CPMLPDANNAdapter",
        "CurveFeatureContract",
        "DANNConfig",
        "DANNPrediction",
        "SourceDomainBatch",
        "TargetDomainBatch",
        "TargetDomainCohort",
        "gradient_reverse",
    }
    if name in dann_exports:
        from quanxin_life.adaptation import dann

        return getattr(dann, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
