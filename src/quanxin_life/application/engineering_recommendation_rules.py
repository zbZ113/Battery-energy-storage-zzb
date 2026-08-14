"""Strict operator-reviewed rule registries for engineering recommendations."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from hmac import compare_digest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from quanxin_life.core import ProvenanceRecord, SourceKind, sha256_canonical
from quanxin_life.core.schemas import ContractModel, Sha256
from quanxin_life.tools.engineering_recommendation import (
    EngineeringRecommendationRule,
    VerifiedEngineeringRecommendationRuleset,
)
from quanxin_life.tools.registry import StandardToolName

_MAX_RULESET_BYTES = 1_048_576
_REGISTRY_SCHEMA_VERSION = "engineering-recommendation-ruleset-registry-v1"
_ALLOWED_EVIDENCE_TOOLS = frozenset(
    {
        StandardToolName.VALIDATE_BATTERY_DATA,
        StandardToolName.PREDICT_CYCLE_LIFE,
        StandardToolName.PREDICT_SOH_TRAJECTORY,
        StandardToolName.COMPARE_OPERATION_SCENARIOS,
        StandardToolName.PROJECT_STORAGE_LIFETIME,
    }
)


class _ReviewedRulesetEntry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    ruleset_id: str = Field(min_length=1, max_length=200)
    ruleset_version: str = Field(min_length=1, max_length=200)
    review_status: Literal["APPROVED"]
    reviewed_at: datetime
    unresolved_reason_code: str = Field(min_length=1, max_length=200)
    rules: tuple[EngineeringRecommendationRule, ...] = Field(min_length=1)
    ruleset_manifest_sha256: Sha256

    @field_validator(
        "ruleset_id",
        "ruleset_version",
        "unresolved_reason_code",
    )
    @classmethod
    def text_is_safe(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized or any(ord(character) < 32 for character in normalized):
            raise ValueError("recommendation ruleset text must be safe and nonblank")
        return normalized

    @field_validator("reviewed_at")
    @classmethod
    def reviewed_at_is_utc_datetime(cls, value: object) -> datetime:
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("reviewed_at must be an ISO-8601 datetime") from exc
        elif isinstance(value, datetime):
            parsed = value
        else:
            raise ValueError("reviewed_at must be an ISO-8601 datetime")
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return parsed.astimezone(UTC)

    @model_validator(mode="after")
    def rules_are_unique_and_allowlisted(self) -> Self:
        rule_ids = tuple(rule.rule_id for rule in self.rules)
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("recommendation rule IDs must be unique")
        if any(rule.result_tool_name not in _ALLOWED_EVIDENCE_TOOLS for rule in self.rules):
            raise ValueError("recommendation rule references an unsupported evidence tool")
        return self

    def semantic_payload(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={"ruleset_manifest_sha256"},
        )


class _ReviewedRulesetRegistryFile(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["engineering-recommendation-ruleset-registry-v1"]
    default_ruleset_id: str = Field(min_length=1, max_length=200)
    rulesets: tuple[_ReviewedRulesetEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identities_are_unique_and_default_exists(self) -> Self:
        identities = tuple(item.ruleset_id for item in self.rulesets)
        if len(identities) != len(set(identities)):
            raise ValueError("recommendation ruleset IDs must be unique")
        if self.default_ruleset_id not in identities:
            raise ValueError("default recommendation ruleset is not registered")
        return self


class ReviewedEngineeringRecommendationRulesetRegistry:
    """Resolve exact reviewed rulesets whose bytes and semantics were verified."""

    def __init__(
        self,
        *,
        rulesets: tuple[VerifiedEngineeringRecommendationRuleset, ...],
        default_ruleset_id: str,
        file_sha256: str,
    ) -> None:
        validated = tuple(
            VerifiedEngineeringRecommendationRuleset.model_validate(
                item.model_dump(mode="json")
            )
            for item in rulesets
        )
        by_id = {item.ruleset_id: item for item in validated}
        if len(by_id) != len(validated):
            raise ValueError("recommendation ruleset IDs must be unique")
        if default_ruleset_id not in by_id:
            raise ValueError("default recommendation ruleset is not registered")
        self._rulesets = MappingProxyType(by_id)
        self._default_ruleset_id = default_ruleset_id
        self._file_sha256 = _sha256(file_sha256, field_name="file_sha256")

    @property
    def default_ruleset_id(self) -> str:
        return self._default_ruleset_id

    @property
    def file_sha256(self) -> str:
        return self._file_sha256

    def resolve_verified_engineering_recommendation_ruleset(
        self,
        ruleset_id: str,
    ) -> VerifiedEngineeringRecommendationRuleset:
        normalized = ruleset_id.strip() if isinstance(ruleset_id, str) else ""
        try:
            ruleset = self._rulesets[normalized]
        except KeyError as exc:
            raise ValueError("engineering recommendation ruleset is not registered") from exc
        return VerifiedEngineeringRecommendationRuleset.model_validate(
            ruleset.model_dump(mode="json")
        )


def load_engineering_recommendation_ruleset_registry(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> ReviewedEngineeringRecommendationRulesetRegistry:
    """Load one strict registry after verifying exact file and semantic digests."""

    expected_sha256 = _sha256(
        expected_file_sha256,
        field_name="expected_file_sha256",
    )
    ruleset_path = Path(path)
    if not ruleset_path.is_absolute():
        raise ValueError("recommendation ruleset registry path must be absolute")
    if ruleset_path.is_symlink() or not ruleset_path.is_file():
        raise ValueError("recommendation ruleset registry must be a regular file")
    if ruleset_path.stat().st_size > _MAX_RULESET_BYTES:
        raise ValueError("recommendation ruleset registry is too large")
    try:
        payload = ruleset_path.read_bytes()
    except OSError as exc:
        raise ValueError("recommendation ruleset registry cannot be read") from exc
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if not compare_digest(actual_sha256, expected_sha256):
        raise ValueError("recommendation ruleset file SHA-256 does not match")
    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        if not isinstance(parsed, dict):
            raise ValueError("recommendation ruleset registry must be an object")
        registry_file = _ReviewedRulesetRegistryFile.model_validate_json(
            payload,
            strict=True,
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(
            "recommendation ruleset registry must be strict UTF-8 JSON"
        ) from exc

    verified: list[VerifiedEngineeringRecommendationRuleset] = []
    for entry in registry_file.rulesets:
        semantic_sha256 = sha256_canonical(entry.semantic_payload())
        if not compare_digest(semantic_sha256, entry.ruleset_manifest_sha256):
            raise ValueError("recommendation ruleset manifest SHA-256 does not match")
        verified.append(
            VerifiedEngineeringRecommendationRuleset(
                ruleset_id=entry.ruleset_id,
                ruleset_version=entry.ruleset_version,
                ruleset_manifest_sha256=entry.ruleset_manifest_sha256,
                unresolved_reason_code=entry.unresolved_reason_code,
                rules=entry.rules,
                provenance=(
                    ProvenanceRecord(
                        source_id=entry.ruleset_id,
                        source_kind=SourceKind.OBSERVED,
                        uri=(
                            "configuration://engineering-recommendation/"
                            f"{entry.ruleset_id}/{entry.ruleset_version}"
                        ),
                        sha256=actual_sha256,
                        description=(
                            "Operator-reviewed engineering recommendation ruleset "
                            f"with status {entry.review_status}."
                        ),
                        created_at=entry.reviewed_at,
                    ),
                ),
            )
        )
    return ReviewedEngineeringRecommendationRulesetRegistry(
        rulesets=tuple(verified),
        default_ruleset_id=registry_file.default_ruleset_id,
        file_sha256=actual_sha256,
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _sha256(value: object, *, field_name: str) -> str:
    normalized = value.strip().lower() if isinstance(value, str) else ""
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return normalized


__all__ = [
    "ReviewedEngineeringRecommendationRulesetRegistry",
    "load_engineering_recommendation_ruleset_registry",
]
