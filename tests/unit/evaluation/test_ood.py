from quanxin_life.evaluation.ood import (
    OODObservation,
    evaluate_ood,
    fit_support_domain,
)


def _observation(identifier: str, value: float, *, label: bool | None = None) -> OODObservation:
    return OODObservation(
        observation_id=identifier,
        dataset_id="dataset",
        temperature_c=value,
        c_rate=value,
        dod=0.8,
        soc=0.5,
        capacity_ah=1.0,
        packaging="pouch",
        features={"f1": value},
        is_ood=label,
    )


def test_unlabelled_ood_reports_support_without_classification_metrics() -> None:
    train = (_observation("t1", 1.0), _observation("t2", 2.0))
    validation = (_observation("v1", 1.5),)
    domain = fit_support_domain(train, validation_observations=validation)
    report = evaluate_ood(domain, (_observation("x1", 10.0),))

    assert report.auroc is None
    assert report.auprc is None
    assert report.fpr95 is None
    assert report.label_status == "UNLABELLED_DISTANCE_ONLY"
    assert report.support_coverage == 0.0
    assert report.score_semantics == "DISTANCE_NOT_PROBABILITY"


def test_labelled_ood_metrics_require_both_classes() -> None:
    train = (_observation("t1", 1.0), _observation("t2", 2.0))
    domain = fit_support_domain(train, validation_observations=(_observation("v1", 1.5),))
    report = evaluate_ood(
        domain,
        (_observation("id", 1.5, label=False), _observation("ood", 10.0, label=True)),
    )
    assert report.auroc == 1.0
    assert report.auprc == 1.0
    assert report.fpr95 == 0.0


def test_unseen_dataset_is_outside_the_fitted_support_domain() -> None:
    domain = fit_support_domain(
        (_observation("t1", 1.0), _observation("t2", 2.0)),
        validation_observations=(_observation("v1", 1.5),),
    )
    report = evaluate_ood(
        domain,
        (
            OODObservation(
                observation_id="other",
                dataset_id="other-dataset",
                temperature_c=1.5,
                c_rate=1.5,
                dod=0.8,
                soc=0.5,
                capacity_ah=1.0,
                packaging="pouch",
                features={"f1": 1.5},
            ),
        ),
    )
    assert report.support_coverage == 0.0
