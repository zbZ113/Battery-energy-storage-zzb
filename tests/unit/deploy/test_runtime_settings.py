from __future__ import annotations

from pathlib import Path

import pytest

from deploy.runtime_settings import CompetitionRuntimeSettings


def _environment(tmp_path: Path) -> dict[str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database_secret = tmp_path / "database_url"
    database_secret.write_text(
        "postgresql+psycopg://quanxin:database-secret@postgres:5432/quanxin\n",
        encoding="utf-8",
    )
    redis_secret = tmp_path / "redis_url"
    redis_secret.write_text(
        "redis://:redis-secret@redis:6379/0\n",
        encoding="utf-8",
    )
    roots = {}
    for name in (
        "data",
        "artifacts",
        "policies",
        "deployment-registry",
        "calibration-evidence",
    ):
        root = tmp_path / name
        root.mkdir()
        roots[name] = root
    registrations = tmp_path / "calibration-registrations.json"
    registrations.write_text("[]\n", encoding="utf-8")
    agent_policy = roots["policies"] / "advanced-agent.json"
    agent_policy.write_text(
        '{"schema_version":"advanced-agent-policy-v1","conformal_alpha":0.1}\n',
        encoding="utf-8",
    )
    return {
        "QUANXIN_DATABASE_URL_FILE": str(database_secret),
        "QUANXIN_REDIS_URL_FILE": str(redis_secret),
        "QUANXIN_TRUSTED_ORIGIN": "https://demo.example.test",
        "QUANXIN_DATA_ROOT": str(roots["data"]),
        "QUANXIN_ARTIFACT_ROOT": str(roots["artifacts"]),
        "QUANXIN_POLICY_ROOT": str(roots["policies"]),
        "QUANXIN_DEPLOYMENT_REGISTRY_ROOT": str(
            roots["deployment-registry"]
        ),
        "QUANXIN_DEPLOYMENT_REGISTRY_ID": "a" * 64,
        "QUANXIN_CALIBRATION_REGISTRATIONS_FILE": str(registrations),
        "QUANXIN_CALIBRATION_EVIDENCE_ROOT": str(
            roots["calibration-evidence"]
        ),
        "QUANXIN_AGENT_POLICY_FILE": str(agent_policy),
    }


def _complete_feishu_environment(tmp_path: Path) -> dict[str, str]:
    environment = _environment(tmp_path)
    for secret_key in (
        "QUANXIN_FEISHU_APP_SECRET_FILE",
        "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE",
        "QUANXIN_FEISHU_ENCRYPT_KEY_FILE",
        "QUANXIN_AILY_CONNECTOR_API_KEY_FILE",
    ):
        path = tmp_path / secret_key.casefold()
        path.write_text("reviewed-secret\n", encoding="utf-8")
        environment[secret_key] = str(path)
    csv_registrations = tmp_path / "feishu-csv-registrations.json"
    csv_registrations.write_text(
        '{"schema_version":"feishu-canonical-csv-registration-registry-v1",'
        '"registrations":[]}\n',
        encoding="utf-8",
    )
    default_scenarios = tmp_path / "feishu-default-scenario-profiles.json"
    default_scenarios.write_text(
        '{"schema_version":"feishu-default-scenario-registry-v1",'
        '"profiles":[]}\n',
        encoding="utf-8",
    )
    environment.update(
        {
            "QUANXIN_FEISHU_AILY_ENABLED": "true",
            "QUANXIN_FEISHU_APP_ID": "cli-reviewed-app",
            "QUANXIN_FEISHU_BITABLE_APP_TOKEN": "bascn-reviewed",
            "QUANXIN_FEISHU_BITABLE_TABLE_ID": "tbl-reviewed",
            "QUANXIN_EXTERNAL_HTTPS_BASE_URL": "https://integration.example.test",
            "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE": str(csv_registrations),
            "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE": str(
                default_scenarios
            ),
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION": "false",
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS": "false",
        }
    )
    return environment


def test_runtime_settings_load_only_explicit_files_and_paths(tmp_path: Path) -> None:
    settings = CompetitionRuntimeSettings.from_environment(
        _environment(tmp_path)
    )

    assert settings.trusted_origin == "https://demo.example.test"
    assert settings.database_url.get_secret_value().startswith(
        "postgresql+psycopg://"
    )
    assert settings.redis_url.get_secret_value().startswith("redis://")
    assert settings.deployment_registry_id == "a" * 64
    assert settings.data_root.is_absolute()
    assert settings.calibration_registrations_file.is_file()
    assert settings.calibration_evidence_root.is_dir()
    assert settings.agent_policy_file.is_file()
    assert settings.feishu_aily is None


def test_runtime_settings_repr_never_exposes_credentials(tmp_path: Path) -> None:
    settings = CompetitionRuntimeSettings.from_environment(
        _environment(tmp_path)
    )

    rendered = repr(settings) + str(settings)

    assert "database-secret" not in rendered
    assert "redis-secret" not in rendered
    assert "**********" in rendered


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("QUANXIN_TRUSTED_ORIGIN", "http://demo.example.test", "HTTPS"),
        ("QUANXIN_TRUSTED_ORIGIN", "https://demo.example.test/path", "origin"),
        ("QUANXIN_DEPLOYMENT_REGISTRY_ID", "not-a-sha", "SHA-256"),
    ),
)
def test_runtime_settings_reject_unsafe_public_identity(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    environment = _environment(tmp_path)
    environment[key] = value

    with pytest.raises(ValueError, match=message):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_relative_or_missing_operator_paths(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["QUANXIN_ARTIFACT_ROOT"] = "relative/artifacts"

    with pytest.raises(ValueError, match="absolute"):
        CompetitionRuntimeSettings.from_environment(environment)

    environment = _environment(tmp_path / "second")
    missing = tmp_path / "missing"
    environment["QUANXIN_POLICY_ROOT"] = str(missing)
    with pytest.raises(ValueError, match="directory"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_secret_symlinks(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    link = tmp_path / "database-link"
    try:
        link.symlink_to(Path(environment["QUANXIN_DATABASE_URL_FILE"]))
    except OSError:
        pytest.skip("symbolic links are unavailable on this Windows host")
    environment["QUANXIN_DATABASE_URL_FILE"] = str(link)

    with pytest.raises(ValueError, match="symbolic link"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_load_complete_feishu_aily_bundle_from_secret_files(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    secret_values = {
        "QUANXIN_FEISHU_APP_SECRET_FILE": "feishu-app-secret",
        "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE": "verification-token",
        "QUANXIN_FEISHU_ENCRYPT_KEY_FILE": "encrypt-key",
        "QUANXIN_AILY_CONNECTOR_API_KEY_FILE": "aily-connector-key",
    }
    for key, value in secret_values.items():
        path = tmp_path / key.casefold()
        path.write_text(value + "\n", encoding="utf-8")
        environment[key] = str(path)
    csv_registrations = tmp_path / "feishu-csv-registrations.json"
    csv_registrations.write_text(
        '{"schema_version":"feishu-canonical-csv-registration-registry-v1",'
        '"registrations":[]}\n',
        encoding="utf-8",
    )
    default_scenarios = tmp_path / "feishu-default-scenario-profiles.json"
    default_scenarios.write_text(
        '{"schema_version":"feishu-default-scenario-registry-v1",'
        '"profiles":[]}\n',
        encoding="utf-8",
    )
    recommendation_rules = tmp_path / "engineering-recommendation-rules.json"
    recommendation_rules.write_text("{}\n", encoding="utf-8")
    environment.update(
        {
            "QUANXIN_FEISHU_AILY_ENABLED": "true",
            "QUANXIN_FEISHU_APP_ID": "cli-reviewed-app",
            "QUANXIN_FEISHU_BITABLE_APP_TOKEN": "bascn-reviewed",
            "QUANXIN_FEISHU_BITABLE_TABLE_ID": "tbl-reviewed",
            "QUANXIN_EXTERNAL_HTTPS_BASE_URL": "https://integration.example.test",
            "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE": str(csv_registrations),
            "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE": str(
                default_scenarios
            ),
            "QUANXIN_FEISHU_ENGINEERING_RULESETS_FILE": str(recommendation_rules),
            "QUANXIN_FEISHU_ENGINEERING_RULESETS_SHA256": (
                "5f36b2ea290645ee34d943220a14b54ee5ea5be5b0c32ef47eb3291b6f214a1e"
            ),
            "QUANXIN_FEISHU_RECHECK_TABLE_ID": "tbl-reviewed-rechecks",
            "QUANXIN_FEISHU_RECHECK_PERMISSION_REFERENCE": (
                "permission-reviewed-v1"
            ),
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION": "true",
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS": "false",
        }
    )

    settings = CompetitionRuntimeSettings.from_environment(environment)

    assert settings.feishu_aily is not None
    integration = settings.feishu_aily
    assert integration.app_id == "cli-reviewed-app"
    assert integration.bitable_app_token == "bascn-reviewed"
    assert integration.bitable_table_id == "tbl-reviewed"
    assert integration.external_https_base_url == "https://integration.example.test"
    assert integration.csv_registrations_file == csv_registrations.resolve(strict=True)
    assert integration.default_scenario_profiles_file == default_scenarios.resolve(
        strict=True
    )
    assert integration.engineering_recommendation_rulesets_file == (
        recommendation_rules.resolve(strict=True)
    )
    assert integration.engineering_recommendation_rulesets_sha256 == (
        "5f36b2ea290645ee34d943220a14b54ee5ea5be5b0c32ef47eb3291b6f214a1e"
    )
    assert integration.recheck_table_id == "tbl-reviewed-rechecks"
    assert integration.recheck_permission_reference == "permission-reviewed-v1"
    assert integration.allow_candidate_scenario_execution is True
    assert integration.allow_candidate_scenario_results is False
    rendered = repr(settings) + str(settings)
    assert "feishu-app-secret" not in rendered
    assert "verification-token" not in rendered
    assert "encrypt-key" not in rendered
    assert "aily-connector-key" not in rendered


def test_runtime_settings_reject_partial_or_unsafe_feishu_aily_configuration(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    environment["QUANXIN_FEISHU_AILY_ENABLED"] = "true"

    with pytest.raises(ValueError, match="QUANXIN_FEISHU_APP_ID"):
        CompetitionRuntimeSettings.from_environment(environment)

    environment = _environment(tmp_path / "unsafe")
    environment["QUANXIN_FEISHU_AILY_ENABLED"] = "false"
    environment["QUANXIN_FEISHU_APP_ID"] = "partially-configured"
    with pytest.raises(ValueError, match="disabled"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_load_opt_in_aily_mcp_from_separate_operator_files(
    tmp_path: Path,
) -> None:
    environment = _complete_feishu_environment(tmp_path)
    endpoint_token = tmp_path / "aily_mcp_endpoint_token"
    endpoint_token.write_text(
        "mcp_endpoint_token_0123456789abcdef\n",
        encoding="utf-8",
    )
    identity_bindings = tmp_path / "aily-mcp-identities.json"
    identity_bindings.write_text(
        '{"schema_version":"quanxin-aily-mcp-identity-bindings-v1",'
        '"bindings":[{"aily_user_id":"aily-user-1",'
        '"local_user_id":"local-user-1"}]}\n',
        encoding="utf-8",
    )
    environment.update(
        {
            "QUANXIN_AILY_MCP_ENABLED": "true",
            "QUANXIN_AILY_MCP_ENDPOINT_TOKEN_FILE": str(endpoint_token),
            "QUANXIN_AILY_MCP_IDENTITY_BINDINGS_FILE": str(identity_bindings),
            "QUANXIN_AILY_MCP_ALLOWED_SOURCE_IPS": (
                "101.126.59.88,101.126.59.89,122.14.241.34"
            ),
        }
    )

    settings = CompetitionRuntimeSettings.from_environment(environment)

    assert settings.feishu_aily is not None
    assert settings.feishu_aily.aily_mcp is not None
    mcp = settings.feishu_aily.aily_mcp
    assert mcp.endpoint_token.get_secret_value().startswith("mcp_endpoint_token_")
    assert mcp.identity_bindings_file == identity_bindings.resolve(strict=True)
    assert mcp.allowed_source_ips == (
        "101.126.59.88",
        "101.126.59.89",
        "122.14.241.34",
    )
    rendered = repr(settings) + str(settings)
    assert "mcp_endpoint_token_0123456789abcdef" not in rendered


def test_runtime_settings_reject_partial_or_disabled_aily_mcp_configuration(
    tmp_path: Path,
) -> None:
    environment = _complete_feishu_environment(tmp_path)
    environment["QUANXIN_AILY_MCP_ENABLED"] = "true"

    with pytest.raises(ValueError, match="QUANXIN_AILY_MCP_ENDPOINT_TOKEN_FILE"):
        CompetitionRuntimeSettings.from_environment(environment)

    environment = _complete_feishu_environment(tmp_path / "disabled")
    endpoint_token = tmp_path / "disabled-token"
    endpoint_token.write_text("mcp_endpoint_token_0123456789abcdef\n", encoding="utf-8")
    environment["QUANXIN_AILY_MCP_ENABLED"] = "false"
    environment["QUANXIN_AILY_MCP_ENDPOINT_TOKEN_FILE"] = str(endpoint_token)

    with pytest.raises(ValueError, match="MCP is disabled"):
        CompetitionRuntimeSettings.from_environment(environment)

    environment = _environment(tmp_path / "feishu-disabled")
    environment["QUANXIN_AILY_MCP_ENDPOINT_TOKEN_FILE"] = str(endpoint_token)
    with pytest.raises(ValueError, match="Feishu/Aily integration is disabled"):
        CompetitionRuntimeSettings.from_environment(environment)


@pytest.mark.parametrize(
    "allowed_source_ips",
    (
        "101.126.59.88/32",
        "101.126.59.88,101.126.59.88",
        "not-an-ip",
    ),
)
def test_runtime_settings_reject_non_exact_aily_mcp_source_ips(
    tmp_path: Path,
    allowed_source_ips: str,
) -> None:
    environment = _complete_feishu_environment(tmp_path)
    endpoint_token = tmp_path / "aily_mcp_endpoint_token"
    endpoint_token.write_text(
        "mcp_endpoint_token_0123456789abcdef\n",
        encoding="utf-8",
    )
    identity_bindings = tmp_path / "aily-mcp-identities.json"
    identity_bindings.write_text(
        '{"schema_version":"quanxin-aily-mcp-identity-bindings-v1",'
        '"bindings":[{"aily_user_id":"aily-user-1",'
        '"local_user_id":"local-user-1"}]}\n',
        encoding="utf-8",
    )
    environment.update(
        {
            "QUANXIN_AILY_MCP_ENABLED": "true",
            "QUANXIN_AILY_MCP_ENDPOINT_TOKEN_FILE": str(endpoint_token),
            "QUANXIN_AILY_MCP_IDENTITY_BINDINGS_FILE": str(identity_bindings),
            "QUANXIN_AILY_MCP_ALLOWED_SOURCE_IPS": allowed_source_ips,
        }
    )

    with pytest.raises(ValueError, match="source IP"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_partial_recommendation_ruleset_configuration(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path)
    for secret_key in (
        "QUANXIN_FEISHU_APP_SECRET_FILE",
        "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE",
        "QUANXIN_FEISHU_ENCRYPT_KEY_FILE",
        "QUANXIN_AILY_CONNECTOR_API_KEY_FILE",
    ):
        path = tmp_path / secret_key.casefold()
        path.write_text("reviewed-secret\n", encoding="utf-8")
        environment[secret_key] = str(path)
    csv_registrations = tmp_path / "feishu-csv-registrations.json"
    csv_registrations.write_text(
        '{"schema_version":"feishu-canonical-csv-registration-registry-v1",'
        '"registrations":[]}\n',
        encoding="utf-8",
    )
    default_scenarios = tmp_path / "feishu-default-scenario-profiles.json"
    default_scenarios.write_text(
        '{"schema_version":"feishu-default-scenario-registry-v1",'
        '"profiles":[]}\n',
        encoding="utf-8",
    )
    environment.update(
        {
            "QUANXIN_FEISHU_AILY_ENABLED": "true",
            "QUANXIN_FEISHU_APP_ID": "cli-reviewed-app",
            "QUANXIN_FEISHU_BITABLE_APP_TOKEN": "bascn-reviewed",
            "QUANXIN_FEISHU_BITABLE_TABLE_ID": "tbl-reviewed",
            "QUANXIN_EXTERNAL_HTTPS_BASE_URL": "https://integration.example.test",
            "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE": str(csv_registrations),
            "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE": str(default_scenarios),
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION": "false",
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS": "false",
            "QUANXIN_FEISHU_ENGINEERING_RULESETS_SHA256": "2" * 64,
        }
    )

    with pytest.raises(ValueError, match="recommendation ruleset"):
        CompetitionRuntimeSettings.from_environment(environment)


def test_runtime_settings_reject_partial_recheck_action_configuration(
    tmp_path: Path,
) -> None:
    environment = _complete_feishu_environment(tmp_path)
    environment["QUANXIN_FEISHU_RECHECK_TABLE_ID"] = "tbl-reviewed-rechecks"

    with pytest.raises(ValueError, match="recheck table and permission"):
        CompetitionRuntimeSettings.from_environment(environment)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    (
        ("QUANXIN_FEISHU_AILY_ENABLED", "sometimes", "boolean"),
        (
            "QUANXIN_EXTERNAL_HTTPS_BASE_URL",
            "http://integration.example.test",
            "HTTPS",
        ),
        ("QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION", "1", "boolean"),
    ),
)
def test_runtime_settings_reject_invalid_feishu_aily_flags_and_url(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    environment = _environment(tmp_path)
    for secret_key in (
        "QUANXIN_FEISHU_APP_SECRET_FILE",
        "QUANXIN_FEISHU_VERIFICATION_TOKEN_FILE",
        "QUANXIN_FEISHU_ENCRYPT_KEY_FILE",
        "QUANXIN_AILY_CONNECTOR_API_KEY_FILE",
    ):
        path = tmp_path / secret_key.casefold()
        path.write_text("reviewed-secret\n", encoding="utf-8")
        environment[secret_key] = str(path)
    csv_registrations = tmp_path / "feishu-csv-registrations.json"
    csv_registrations.write_text(
        '{"schema_version":"feishu-canonical-csv-registration-registry-v1",'
        '"registrations":[]}\n',
        encoding="utf-8",
    )
    default_scenarios = tmp_path / "feishu-default-scenario-profiles.json"
    default_scenarios.write_text(
        '{"schema_version":"feishu-default-scenario-registry-v1",'
        '"profiles":[]}\n',
        encoding="utf-8",
    )
    environment.update(
        {
            "QUANXIN_FEISHU_AILY_ENABLED": "true",
            "QUANXIN_FEISHU_APP_ID": "cli-reviewed-app",
            "QUANXIN_FEISHU_BITABLE_APP_TOKEN": "bascn-reviewed",
            "QUANXIN_FEISHU_BITABLE_TABLE_ID": "tbl-reviewed",
            "QUANXIN_EXTERNAL_HTTPS_BASE_URL": "https://integration.example.test",
            "QUANXIN_FEISHU_CSV_REGISTRATIONS_FILE": str(csv_registrations),
            "QUANXIN_FEISHU_DEFAULT_SCENARIO_PROFILES_FILE": str(
                default_scenarios
            ),
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_EXECUTION": "false",
            "QUANXIN_ALLOW_CANDIDATE_SCENARIO_RESULTS": "false",
            key: value,
        }
    )

    with pytest.raises(ValueError, match=message):
        CompetitionRuntimeSettings.from_environment(environment)
