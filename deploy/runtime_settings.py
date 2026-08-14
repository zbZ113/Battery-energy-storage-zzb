"""Fail-closed settings for the single-node competition deployment."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import SecretStr

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_SECRET_BYTES = 8_192
_FEISHU_AILY_ENABLED = "QUANXIN_FEISHU_AILY_ENABLED"
_FEISHU_AILY_CONFIGURATION_KEYS = (
    "QUANXIN_FEISHU_APP_ID",
    "QUANXIN_FEISHU_APP_SECRET_FILE",
    "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE",
    "QUANXIN_FEISHU_ENCRYPT_KEY_FILE",
    "QUANXIN_AILY_CONNECTOR_API_KEY_FILE",
    "QUANXIN_FEISHU_BITABLE_APP_TOKEN",
    "QUANXIN_FEISHU_BITABLE_TABLE_ID",
    "QUANXIN_EXTERNAL_HTTPS_BASE_URL",
    "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE",
    "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE",
    "QUANXIN_FEISHU_ENGINEERING_RULESETS_FILE",
    "QUANXIN_FEISHU_ENGINEERING_RULESETS_SHA256",
    "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION",
    "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS",
)


@dataclass(frozen=True, slots=True)
class FeishuAilyRuntimeSettings:
    """Complete opt-in production identity for the audited integration bundle."""

    app_id: str
    app_secret: SecretStr
    verification_token: SecretStr
    encrypt_key: SecretStr
    aily_connector_api_key: SecretStr
    bitable_app_token: str
    bitable_table_id: str
    external_https_base_url: str
    csv_registrations_file: Path
    default_scenario_profiles_file: Path
    allow_candidate_scenario_execution: bool
    allow_candidate_scenario_results: bool
    engineering_recommendation_rulesets_file: Path | None = None
    engineering_recommendation_rulesets_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class CompetitionRuntimeSettings:
    """Operator-owned runtime identity without implicit environment fallbacks."""

    database_url: SecretStr
    redis_url: SecretStr
    trusted_origin: str
    data_root: Path
    artifact_root: Path
    policy_root: Path
    deployment_registry_root: Path
    deployment_registry_id: str
    calibration_registrations_file: Path
    calibration_evidence_root: Path
    agent_policy_file: Path
    feishu_aily: FeishuAilyRuntimeSettings | None = None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str],
    ) -> CompetitionRuntimeSettings:
        database_url = _secret_url(
            environment,
            "QUANXIN_DATABASE_URL_FILE",
            schemes={"postgresql", "postgresql+psycopg"},
            label="database URL",
        )
        redis_url = _secret_url(
            environment,
            "QUANXIN_REDIS_URL_FILE",
            schemes={"redis", "rediss"},
            label="Redis URL",
        )
        trusted_origin = _trusted_origin(
            _required(environment, "QUANXIN_TRUSTED_ORIGIN")
        )
        registry_id = _required(
            environment,
            "QUANXIN_DEPLOYMENT_REGISTRY_ID",
        ).lower()
        if _SHA256.fullmatch(registry_id) is None:
            raise ValueError("deployment registry ID must be a SHA-256 digest")
        data_root = _directory(environment, "QUANXIN_DATA_ROOT")
        artifact_root = _directory(environment, "QUANXIN_ARTIFACT_ROOT")
        policy_root = _directory(environment, "QUANXIN_POLICY_ROOT")
        deployment_registry_root = _directory(
            environment,
            "QUANXIN_DEPLOYMENT_REGISTRY_ROOT",
        )
        calibration_registrations_file = _regular_file(
            environment,
            "QUANXIN_CALIBRATION_REGISTRATIONS_FILE",
        )
        calibration_evidence_root = _directory(
            environment,
            "QUANXIN_CALIBRATION_EVIDENCE_ROOT",
        )
        agent_policy_file = _regular_file(
            environment,
            "QUANXIN_AGENT_POLICY_FILE",
        )
        if not agent_policy_file.is_relative_to(policy_root):
            raise ValueError("Agent policy file must remain inside QUANXIN_POLICY_ROOT")
        feishu_aily = _feishu_aily_settings(environment)
        return cls(
            database_url=database_url,
            redis_url=redis_url,
            trusted_origin=trusted_origin,
            data_root=data_root,
            artifact_root=artifact_root,
            policy_root=policy_root,
            deployment_registry_root=deployment_registry_root,
            deployment_registry_id=registry_id,
            calibration_registrations_file=calibration_registrations_file,
            calibration_evidence_root=calibration_evidence_root,
            agent_policy_file=agent_policy_file,
            feishu_aily=feishu_aily,
        )


def _required(environment: Mapping[str, str], key: str) -> str:
    value = environment.get(key, "")
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized:
        raise ValueError(f"required runtime setting is missing: {key}")
    return normalized


def _operator_path(environment: Mapping[str, str], key: str) -> Path:
    path = Path(_required(environment, key))
    if not path.is_absolute():
        raise ValueError(f"operator path must be absolute: {key}")
    if path.is_symlink():
        raise ValueError(f"operator path must not be a symbolic link: {key}")
    return path


def _directory(environment: Mapping[str, str], key: str) -> Path:
    path = _operator_path(environment, key)
    if not path.is_dir():
        raise ValueError(f"operator path must be an existing directory: {key}")
    return path.resolve(strict=True)


def _regular_file(environment: Mapping[str, str], key: str) -> Path:
    path = _operator_path(environment, key)
    if not path.is_file():
        raise ValueError(f"operator path must be an existing regular file: {key}")
    return path.resolve(strict=True)


def _secret_url(
    environment: Mapping[str, str],
    key: str,
    *,
    schemes: set[str],
    label: str,
) -> SecretStr:
    path = _regular_file(environment, key)
    if path.stat().st_size > _MAX_SECRET_BYTES:
        raise ValueError(f"{label} secret file is too large")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} secret file cannot be read") from exc
    value = raw.strip()
    if not value or "\x00" in value or len(value.splitlines()) != 1:
        raise ValueError(f"{label} secret file must contain one nonblank line")
    parsed = urlsplit(value)
    if parsed.scheme not in schemes or not parsed.hostname:
        raise ValueError(f"{label} uses an unsupported or incomplete URL")
    return SecretStr(value)


def _secret_text(environment: Mapping[str, str], key: str, *, label: str) -> SecretStr:
    path = _regular_file(environment, key)
    if path.stat().st_size > _MAX_SECRET_BYTES:
        raise ValueError(f"{label} secret file is too large")
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"{label} secret file cannot be read") from exc
    value = raw.strip()
    if not value or "\x00" in value or len(value.splitlines()) != 1:
        raise ValueError(f"{label} secret file must contain one nonblank line")
    return SecretStr(value)


def _strict_boolean(environment: Mapping[str, str], key: str) -> bool:
    value = _required(environment, key).casefold()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"runtime setting must be a true/false boolean: {key}")


def _optional_boolean(
    environment: Mapping[str, str],
    key: str,
    *,
    default: bool,
) -> bool:
    raw = environment.get(key, "")
    normalized = raw.strip() if isinstance(raw, str) else ""
    if not normalized:
        return default
    return _strict_boolean(environment, key)


def _feishu_aily_settings(
    environment: Mapping[str, str],
) -> FeishuAilyRuntimeSettings | None:
    enabled = _optional_boolean(
        environment,
        _FEISHU_AILY_ENABLED,
        default=False,
    )
    configured = tuple(
        key
        for key in _FEISHU_AILY_CONFIGURATION_KEYS
        if isinstance(environment.get(key), str) and environment[key].strip()
    )
    if not enabled:
        if configured:
            raise ValueError(
                "Feishu/Aily integration is disabled but integration settings are configured"
            )
        return None
    recommendation_rulesets_file, recommendation_rulesets_sha256 = (
        _optional_recommendation_rulesets(environment)
    )
    return FeishuAilyRuntimeSettings(
        app_id=_required(environment, "QUANXIN_FEISHU_APP_ID"),
        app_secret=_secret_text(
            environment,
            "QUANXIN_FEISHU_APP_SECRET_FILE",
            label="Feishu app secret",
        ),
        verification_token=_secret_text(
            environment,
            "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE",
            label="Feishu verification token",
        ),
        encrypt_key=_secret_text(
            environment,
            "QUANXIN_FEISHU_ENCRYPT_KEY_FILE",
            label="Feishu encrypt key",
        ),
        aily_connector_api_key=_secret_text(
            environment,
            "QUANXIN_AILY_CONNECTOR_API_KEY_FILE",
            label="Aily connector API key",
        ),
        bitable_app_token=_required(
            environment,
            "QUANXIN_FEISHU_BITABLE_APP_TOKEN",
        ),
        bitable_table_id=_required(
            environment,
            "QUANXIN_FEISHU_BITABLE_TABLE_ID",
        ),
        external_https_base_url=_https_base_url(
            _required(environment, "QUANXIN_EXTERNAL_HTTPS_BASE_URL")
        ),
        csv_registrations_file=_regular_file(
            environment,
            "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE",
        ),
        default_scenario_profiles_file=_regular_file(
            environment,
            "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE",
        ),
        engineering_recommendation_rulesets_file=recommendation_rulesets_file,
        engineering_recommendation_rulesets_sha256=recommendation_rulesets_sha256,
        allow_candidate_scenario_execution=_strict_boolean(
            environment,
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION",
        ),
        allow_candidate_scenario_results=_strict_boolean(
            environment,
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS",
        ),
    )


def _optional_recommendation_rulesets(
    environment: Mapping[str, str],
) -> tuple[Path | None, str | None]:
    path_key = "QUANXIN_FEISHU_ENGINEERING_RULESETS_FILE"
    sha_key = "QUANXIN_FEISHU_ENGINEERING_RULESETS_SHA256"
    has_path = isinstance(environment.get(path_key), str) and bool(
        environment[path_key].strip()
    )
    has_sha = isinstance(environment.get(sha_key), str) and bool(
        environment[sha_key].strip()
    )
    if has_path != has_sha:
        raise ValueError(
            "engineering recommendation ruleset file and SHA-256 must be configured together"
        )
    if not has_path:
        return None, None
    digest = _required(environment, sha_key).lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError("engineering recommendation ruleset SHA-256 is invalid")
    return _regular_file(environment, path_key), digest


def _trusted_origin(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError("production trusted Origin must use HTTPS")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("production trusted origin must contain only scheme and authority")
    return value.removesuffix("/")


def _https_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https":
        raise ValueError("external integration base URL must use HTTPS")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "external integration base URL must contain only scheme and authority"
        )
    return value.removesuffix("/")


__all__ = ["CompetitionRuntimeSettings", "FeishuAilyRuntimeSettings"]
