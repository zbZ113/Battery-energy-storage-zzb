from __future__ import annotations


def test_adaptation_package_lazily_exposes_dann_research_contracts() -> None:
    from quanxin_life.adaptation import (
        CPMLPDANNAdapter,
        CurveFeatureContract,
        DANNConfig,
        TargetDomainCohort,
        gradient_reverse,
    )

    assert CPMLPDANNAdapter.__name__ == "CPMLPDANNAdapter"
    assert CurveFeatureContract.__name__ == "CurveFeatureContract"
    assert DANNConfig.__name__ == "DANNConfig"
    assert TargetDomainCohort.ADAPTATION.value == "adaptation"
    assert callable(gradient_reverse)
