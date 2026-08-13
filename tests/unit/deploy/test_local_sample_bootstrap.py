from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/local/bootstrap_sample_api.py"


def _module():
    if not SCRIPT.is_file():
        pytest.fail("local sample API bootstrap is not implemented")
    spec = importlib.util.spec_from_file_location("bootstrap_sample_api", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_required_calibration_matrix_is_the_reviewed_twelve_routes() -> None:
    module = _module()

    assert module.required_calibration_routes() == (
        ("RUL", 20, "DEFAULT"),
        ("RUL", 50, "COVERAGE"),
        ("RUL", 100, "COVERAGE"),
        ("RUL", 150, "COVERAGE"),
        ("SOH", 20, "MEAN_ACCURACY"),
        ("SOH", 20, "TAIL_EFFICIENCY"),
        ("SOH", 50, "MEAN_ACCURACY"),
        ("SOH", 50, "TAIL_EFFICIENCY"),
        ("SOH", 100, "MEAN_ACCURACY"),
        ("SOH", 100, "TAIL_EFFICIENCY"),
        ("SOH", 150, "MEAN_ACCURACY"),
        ("SOH", 150, "TAIL_EFFICIENCY"),
    )


def test_bootstrap_source_uses_hidden_login_and_formal_product_apis() -> None:
    source = SCRIPT.read_text(encoding="utf-8") if SCRIPT.is_file() else ""

    assert "getpass.getpass" in source
    assert "--password" not in source
    assert "password=" not in source
    for endpoint in (
        "/v1/auth/login",
        "/v1/projects",
        "/v1/datasets",
        "/batches/canonical-csv",
        "/freeze",
        "/v1/admin/model-artifacts/advanced-candidates",
        "/v1/admin/model-routes/activations",
        "/advanced-calibration/materializations",
    ):
        assert endpoint in source
    assert "payload_base64" in source
    assert "MATR_b3c34" in source
    assert "(20, 50, 100, 150)" in source


def test_unique_record_selection_fails_closed_on_ambiguous_matches() -> None:
    module = _module()

    assert module.unique_match([], label="project") is None
    assert module.unique_match([{"id": "one"}], label="project") == {"id": "one"}
    with pytest.raises(RuntimeError, match="multiple project"):
        module.unique_match([{"id": "one"}, {"id": "two"}], label="project")


class _DatasetClient:
    def __init__(self, datasets: list[dict[str, object]]) -> None:
        self.datasets = datasets
        self.created_payloads: list[dict[str, object]] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> object:
        if method == "GET":
            return list(self.datasets)
        assert method == "POST"
        assert path == "/v1/datasets"
        assert payload is not None
        self.created_payloads.append(payload)
        return {"dataset_id": "dataset-new", "status": "DRAFT", **payload}


def test_bootstrap_ignores_legacy_dataset_schema_and_creates_cycle_records() -> None:
    module = _module()
    client = _DatasetClient(
        [
            {
                "dataset_id": "dataset-legacy",
                "name": module.DATASET_NAME,
                "data_version": "matr-v1",
                "schema_version": "canonical-v1",
                "status": "FROZEN",
            }
        ]
    )

    dataset = module._dataset(
        client,
        project_id="project-1",
        data_version="matr-v1",
    )

    assert dataset["dataset_id"] == "dataset-new"
    assert client.created_payloads == [
        {
            "project_id": "project-1",
            "name": module.DATASET_NAME,
            "data_version": "matr-v1",
            "schema_version": "cycle-record-v1",
        }
    ]


def test_bootstrap_reuses_only_the_matching_cycle_record_dataset() -> None:
    module = _module()
    expected = {
        "dataset_id": "dataset-current",
        "name": module.DATASET_NAME,
        "data_version": "matr-v1",
        "schema_version": "cycle-record-v1",
        "status": "FROZEN",
    }
    client = _DatasetClient(
        [
            {
                "dataset_id": "dataset-legacy",
                "name": module.DATASET_NAME,
                "data_version": "matr-v1",
                "schema_version": "canonical-v1",
                "status": "FROZEN",
            },
            expected,
        ]
    )

    dataset = module._dataset(
        client,
        project_id="project-1",
        data_version="matr-v1",
    )

    assert dataset == expected
    assert client.created_payloads == []


def test_calibration_source_identity_comes_from_reviewed_runtime_registry(
    tmp_path: Path,
) -> None:
    module = _module()
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "calibration-sources.json").write_text(
        json.dumps(
            {
                "schema_version": "advanced-calibration-source-registry-v1",
                "sources": [
                    {
                        "registration_id": "reviewed-source-v1",
                        "evidence_relative_root": "reviewed-source",
                        "three_batch_manifest_sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        module.calibration_source_registration_id(tmp_path)
        == "reviewed-source-v1"
    )
