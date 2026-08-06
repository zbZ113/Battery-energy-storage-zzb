"""Train-fitted support domains with explicitly non-probabilistic OOD scores."""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import fmean
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256


class OODObservation(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    temperature_c: float = Field(allow_inf_nan=False)
    c_rate: float = Field(ge=0, allow_inf_nan=False)
    dod: float = Field(ge=0, le=1, allow_inf_nan=False)
    soc: float = Field(ge=0, le=1, allow_inf_nan=False)
    capacity_ah: float = Field(gt=0, allow_inf_nan=False)
    packaging: str = Field(min_length=1)
    features: dict[str, float] = Field(min_length=1)
    is_ood: bool | None = None

    @field_validator("features")
    @classmethod
    def features_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not key.strip() or not math.isfinite(item) for key, item in value.items()):
            raise ValueError("OOD features require non-empty names and finite values")
        return value


class NumericSupport(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float = Field(allow_inf_nan=False)
    maximum: float = Field(allow_inf_nan=False)
    center: float = Field(allow_inf_nan=False)
    scale: float = Field(gt=0, allow_inf_nan=False)


class SupportDomain(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "ood-support-domain-v1"
    dataset_ids: tuple[str, ...]
    numeric: dict[str, NumericSupport]
    packaging: tuple[str, ...]
    feature_names: tuple[str, ...]
    distance_threshold: float = Field(ge=0, allow_inf_nan=False)
    fit_observation_count: int = Field(gt=0)
    threshold_observation_count: int = Field(gt=0)
    support_sha256: Sha256

    @model_validator(mode="after")
    def hash_matches_support(self) -> SupportDomain:
        if sha256_canonical(
            self.model_dump(mode="json", exclude={"support_sha256"})
        ) != self.support_sha256:
            raise ValueError("support_sha256 does not match support-domain contents")
        return self


class OODScore(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    observation_id: str
    distance: float = Field(ge=0, allow_inf_nan=False)
    within_support: bool
    is_ood: bool | None


class OODEvaluation(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    support_sha256: Sha256
    observation_count: int = Field(gt=0)
    support_coverage: float = Field(ge=0, le=1, allow_inf_nan=False)
    mean_distance: float = Field(ge=0, allow_inf_nan=False)
    scores: tuple[OODScore, ...]
    label_status: Literal["LABELLED", "UNLABELLED_DISTANCE_ONLY"]
    score_semantics: Literal["DISTANCE_NOT_PROBABILITY"] = "DISTANCE_NOT_PROBABILITY"
    auroc: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    auprc: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    fpr95: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


_BASE_FIELDS = ("temperature_c", "c_rate", "dod", "soc", "capacity_ah")


def fit_support_domain(
    train_observations: Sequence[OODObservation],
    *,
    validation_observations: Sequence[OODObservation] = (),
    threshold_quantile: float = 0.95,
) -> SupportDomain:
    train = tuple(OODObservation.model_validate(item) for item in train_observations)
    validation = tuple(OODObservation.model_validate(item) for item in validation_observations)
    if not train:
        raise ValueError("train observations are required to fit an OOD support domain")
    if any(item.is_ood is not None for item in (*train, *validation)):
        raise ValueError("support fitting cannot consume OOD labels")
    if not 0 < threshold_quantile <= 1 or not math.isfinite(threshold_quantile):
        raise ValueError("threshold_quantile must be finite and in (0, 1]")
    feature_names = tuple(sorted(train[0].features))
    if any(tuple(sorted(item.features)) != feature_names for item in (*train, *validation)):
        raise ValueError("support observations must share the same feature schema")
    numeric: dict[str, NumericSupport] = {}
    for name in (*_BASE_FIELDS, *(f"feature:{feature}" for feature in feature_names)):
        values = [_numeric_value(item, name) for item in train]
        center = fmean(values)
        scale = max(max(values) - min(values), 1e-12)
        numeric[name] = NumericSupport(
            minimum=min(values), maximum=max(values), center=center, scale=scale
        )
    unsealed = {
        "schema_version": "ood-support-domain-v1",
        "dataset_ids": sorted({item.dataset_id for item in train}),
        "numeric": {key: value.model_dump(mode="json") for key, value in numeric.items()},
        "packaging": sorted({item.packaging for item in train}),
        "feature_names": list(feature_names),
        "distance_threshold": 0.0,
        "fit_observation_count": len(train),
        "threshold_observation_count": len(validation) or len(train),
    }
    provisional = SupportDomain.model_validate(
        {**unsealed, "support_sha256": sha256_canonical(unsealed)}
    )
    threshold_cohort = validation or train
    distances = sorted(_distance(provisional, item) for item in threshold_cohort)
    threshold = _nearest_rank(distances, threshold_quantile)
    payload = {**unsealed, "distance_threshold": threshold}
    return SupportDomain.model_validate(
        {**payload, "support_sha256": sha256_canonical(payload)}
    )


def evaluate_ood(
    support: SupportDomain,
    observations: Sequence[OODObservation],
) -> OODEvaluation:
    domain = SupportDomain.model_validate(support)
    cohort = tuple(OODObservation.model_validate(item) for item in observations)
    if not cohort:
        raise ValueError("at least one OOD evaluation observation is required")
    labels = [item.is_ood for item in cohort]
    if any(label is None for label in labels) and any(label is not None for label in labels):
        raise ValueError("OOD evaluation labels must be complete or entirely absent")
    scores = tuple(
        OODScore(
            observation_id=item.observation_id,
            distance=_distance(domain, item),
            within_support=_distance(domain, item) <= domain.distance_threshold,
            is_ood=item.is_ood,
        )
        for item in cohort
    )
    labelled = all(label is not None for label in labels)
    auroc: float | None = None
    auprc: float | None = None
    fpr95: float | None = None
    if labelled:
        definite_labels = [bool(label) for label in labels]
        if set(definite_labels) != {False, True}:
            raise ValueError("labelled OOD metrics require both ID and OOD observations")
        distances = [score.distance for score in scores]
        auroc = _auroc(definite_labels, distances)
        auprc = _average_precision(definite_labels, distances)
        fpr95 = _fpr_at_tpr(definite_labels, distances, target_tpr=0.95)
    return OODEvaluation(
        support_sha256=domain.support_sha256,
        observation_count=len(scores),
        support_coverage=sum(score.within_support for score in scores) / len(scores),
        mean_distance=fmean(score.distance for score in scores),
        scores=scores,
        label_status="LABELLED" if labelled else "UNLABELLED_DISTANCE_ONLY",
        auroc=auroc,
        auprc=auprc,
        fpr95=fpr95,
    )


def _numeric_value(observation: OODObservation, name: str) -> float:
    if name.startswith("feature:"):
        return observation.features[name.removeprefix("feature:")]
    return float(getattr(observation, name))


def _distance(support: SupportDomain, observation: OODObservation) -> float:
    if tuple(sorted(observation.features)) != support.feature_names:
        raise ValueError("OOD observation feature schema does not match support domain")
    normalized = [
        max(
            (definition.minimum - _numeric_value(observation, name)) / definition.scale,
            (_numeric_value(observation, name) - definition.maximum) / definition.scale,
            0.0,
        )
        for name, definition in support.numeric.items()
    ]
    dataset_penalty = 1.0 if observation.dataset_id not in support.dataset_ids else 0.0
    packaging_penalty = 1.0 if observation.packaging not in support.packaging else 0.0
    return math.sqrt(
        sum(value * value for value in normalized)
        + packaging_penalty
        + dataset_penalty
    )


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    return values[max(0, math.ceil(quantile * len(values)) - 1)]


def _auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    wins = sum((left > right) + 0.5 * (left == right) for left in positives for right in negatives)
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: Sequence[bool], scores: Sequence[float]) -> float:
    ranked = sorted(zip(scores, labels, strict=True), reverse=True)
    positive_count = sum(labels)
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positive_count


def _fpr_at_tpr(labels: Sequence[bool], scores: Sequence[float], *, target_tpr: float) -> float:
    positives = sorted(score for label, score in zip(labels, scores, strict=True) if label)
    threshold = positives[max(0, math.floor((1 - target_tpr) * len(positives)))]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    return sum(score >= threshold for score in negatives) / len(negatives)


__all__ = [
    "OODEvaluation",
    "OODObservation",
    "OODScore",
    "SupportDomain",
    "evaluate_ood",
    "fit_support_domain",
]
