from __future__ import annotations

import asyncio
import base64
import importlib.util
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import Field

from quanxin_life.core import ProvenanceRecord, SourceKind, ToolResult, sha256_canonical
from quanxin_life.core.schemas import ContractModel
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry


class _Input(ContractModel):
    batch_id: str = Field(min_length=1)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    def executor(value: _Input) -> ToolResult:
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA.value,
            tool_version="api-service-test-v1",
            model_version="api-service-model-v1",
            data_version="api-service-data-v1",
            feature_version="api-service-feature-v1",
            input_hash=sha256_canonical(value.model_dump(mode="json")),
            values={"validated_batch": value.batch_id},
            provenance=[
                ProvenanceRecord(
                    source_id="api-service-fixture",
                    source_kind=SourceKind.OBSERVED,
                    uri="test://api-service/fixture",
                    sha256=sha256_canonical({"fixture": "api-service"}),
                    description="Tool service test fixture",
                    created_at=datetime(2026, 7, 13, tzinfo=UTC),
                )
            ],
            created_at=datetime(2026, 7, 13, tzinfo=UTC),
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            tool_version="api-service-test-v1",
            input_model=_Input,
            executor=executor,
        )
    )
    return registry


def test_service_delegates_external_invocation_to_shared_registry() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService

    service = ToolInvocationService(registry=_registry())
    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": "batch-A"},
        )
    )

    assert result.tool_name == StandardToolName.VALIDATE_BATTERY_DATA.value
    assert result.values == {"validated_batch": "batch-A"}
    assert result.input_hash == sha256_canonical({"batch_id": "batch-A"})


def test_service_registers_each_successful_result_in_shared_audit_ledger() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService
    from quanxin_life.audit import AuditLedger

    ledger = AuditLedger()
    service = ToolInvocationService(registry=_registry(), audit_ledger=ledger)

    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": "batch-A"},
        )
    )

    assert ledger.resolve_registered_result(result.result_id) == result


def test_agent_service_path_registers_result_and_no_ledger_remains_compatible() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService
    from quanxin_life.audit import AuditLedger

    invocation = ToolInvocation(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        input_value={"batch_id": "batch-A"},
    )
    ledger = AuditLedger()
    audited_service = ToolInvocationService(registry=_registry(), audit_ledger=ledger)

    audited = audited_service.invoke_for_agent(
        invocation,
        allowed_tool_names=(StandardToolName.VALIDATE_BATTERY_DATA,),
    )
    unaudited = ToolInvocationService(registry=_registry()).invoke(invocation)

    assert ledger.resolve_registered_result(audited.result_id) == audited
    assert unaudited.tool_name == StandardToolName.VALIDATE_BATTERY_DATA.value


def test_available_service_uses_the_shared_available_tool_assembly() -> None:
    from quanxin_life.api.service import create_available_tool_invocation_service

    service = create_available_tool_invocation_service()

    assert [schema.tool_name.value for schema in service.registry.list_schemas()] == [
        "audit_dataset_split",
        "check_operating_condition",
        "validate_battery_data",
    ]


def test_service_requires_explicit_nonempty_allowlist_for_agent_invocation() -> None:
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService
    from quanxin_life.tools import ToolAuthorizationError

    service = ToolInvocationService(registry=_registry())
    invocation = ToolInvocation(
        tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
        input_value={"batch_id": "batch-A"},
    )

    with pytest.raises(ToolAuthorizationError, match="non-empty allowlist"):
        service.invoke_for_agent(invocation, allowed_tool_names=())


def test_invocation_rejects_non_json_input_value() -> None:
    from quanxin_life.api.service import ToolInvocation

    with pytest.raises(ValueError, match="JSON-compatible"):
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": object()},
        )


def test_fastapi_factory_fails_explicitly_when_optional_dependency_is_unavailable() -> None:
    if importlib.util.find_spec("fastapi") is not None:
        pytest.skip("FastAPI is installed; transport behavior belongs to the API dependency suite")

    from quanxin_life.api.app import FastApiDependencyUnavailable, create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService

    with pytest.raises(FastApiDependencyUnavailable, match=r"quanxin-life\[api\]"):
        create_fastapi_app(ToolInvocationService(registry=_registry()))


class _FakeFastApiApp:
    def __init__(self, **options: object) -> None:
        self.options = options
        self.routes: dict[tuple[str, str], Callable[..., object]] = {}

    def get(self, path: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
        return self._route("GET", path)

    def post(self, path: str) -> Callable[[Callable[..., object]], Callable[..., object]]:
        return self._route("POST", path)

    def _route(
        self, method: str, path: str
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        def decorator(function: Callable[..., object]) -> Callable[..., object]:
            self.routes[(method, path)] = function
            return function

        return decorator


class _FakeHttpException(Exception):
    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FakeFastApiModule:
    FastAPI = _FakeFastApiApp
    HTTPException = _FakeHttpException


def test_fastapi_factory_uses_the_project_title(monkeypatch: pytest.MonkeyPatch) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )

    app = create_fastapi_app(ToolInvocationService(registry=_registry()))

    assert app.options["title"] == "泉芯智寿 Tool API"


def test_fastapi_exposes_planned_domain_routes_through_shared_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import DOMAIN_ROUTE_TOOL_MAP, create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService

    expected_routes = {
        "/v1/analyses/quality": StandardToolName.VALIDATE_BATTERY_DATA,
        "/v1/predictions/lifetime": StandardToolName.PREDICT_CYCLE_LIFE,
        "/v1/predictions/trajectory": StandardToolName.PREDICT_SOH_TRAJECTORY,
        "/v1/predictions/update": StandardToolName.UPDATE_CELL_PARAMETERS,
        "/v1/physics/check": StandardToolName.CHECK_OPERATING_CONDITION,
        "/v1/experiments/recommend": StandardToolName.RECOMMEND_NEXT_EXPERIMENT,
        "/v1/decisions/batch": StandardToolName.MAKE_BATCH_DECISION,
    }
    assert expected_routes == DOMAIN_ROUTE_TOOL_MAP

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    app = create_fastapi_app(ToolInvocationService(registry=_registry()))

    for path in expected_routes:
        assert ("POST", path) in app.routes

    response = asyncio.run(
        app.routes[("POST", "/v1/analyses/quality")]({"batch_id": "batch-A"})
    )
    assert response["tool_name"] == StandardToolName.VALIDATE_BATTERY_DATA.value
    assert response["values"] == {"validated_batch": "batch-A"}


def test_fastapi_lifetime_workflow_endpoint_delegates_without_numeric_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.application.lifetime_workflow import (
        LifetimeDecisionWorkflowRequest,
        LifetimeDecisionWorkflowResult,
        LifetimeDecisionWorkflowStatus,
    )

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    captured: list[LifetimeDecisionWorkflowRequest] = []
    expected = LifetimeDecisionWorkflowResult(
        status=LifetimeDecisionWorkflowStatus.QUALITY_BLOCKED,
        quality_result_id=str(uuid4()),
        warnings=["blocked-by-fixture"],
    )

    def runner(
        _: ToolInvocationService, request: LifetimeDecisionWorkflowRequest
    ) -> LifetimeDecisionWorkflowResult:
        captured.append(request)
        return expected

    service = ToolInvocationService(registry=_registry())
    app = create_fastapi_app(service, lifetime_workflow_runner=runner)
    endpoint = app.routes[("POST", "/v1/workflows/lifetime-decision")]
    payload: dict[str, Any] = {
        "record_batch_id": "batch-1",
        "calibration_cohort_id": "calibration-1",
        "policy_id": "policy-1",
    }

    response = asyncio.run(endpoint(payload))

    assert response == expected.model_dump(mode="json")
    assert captured == [LifetimeDecisionWorkflowRequest.model_validate(payload)]


def test_fastapi_exposes_detached_audit_result_and_markdown_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocation, ToolInvocationService
    from quanxin_life.audit import AuditLedger

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    ledger = AuditLedger()
    service = ToolInvocationService(registry=_registry(), audit_ledger=ledger)
    result = service.invoke(
        ToolInvocation(
            tool_name=StandardToolName.VALIDATE_BATTERY_DATA,
            input_value={"batch_id": "batch-A"},
        )
    )
    report = result.model_copy(
        update={
            "result_id": str(uuid4()),
            "tool_name": StandardToolName.GENERATE_AUDITED_REPORT.value,
            "values": {"markdown": "# Audited report\n"},
        }
    )
    ledger.register_result(report)

    app = create_fastapi_app(service)

    resolved = asyncio.run(app.routes[("GET", "/v1/results/{result_id}")](result.result_id))
    markdown = asyncio.run(app.routes[("GET", "/v1/reports/{result_id}")](report.result_id))

    assert resolved == result.model_dump(mode="json")
    assert markdown == {"result_id": report.result_id, "markdown": "# Audited report\n"}


def test_fastapi_audit_read_endpoints_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.audit import AuditLedger

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    unknown_result_id = str(uuid4())
    without_ledger = create_fastapi_app(ToolInvocationService(registry=_registry()))
    with_ledger = create_fastapi_app(
        ToolInvocationService(registry=_registry(), audit_ledger=AuditLedger())
    )

    with pytest.raises(_FakeHttpException) as unavailable:
        asyncio.run(
            without_ledger.routes[("GET", "/v1/results/{result_id}")](unknown_result_id)
        )
    with pytest.raises(_FakeHttpException) as missing:
        asyncio.run(with_ledger.routes[("GET", "/v1/results/{result_id}")](unknown_result_id))
    with pytest.raises(_FakeHttpException) as not_report:
        asyncio.run(with_ledger.routes[("GET", "/v1/reports/{result_id}")](unknown_result_id))

    assert unavailable.value.status_code == 503
    assert missing.value.status_code == 404
    assert not_report.value.status_code == 404


def test_fastapi_canonical_csv_endpoint_delegates_to_server_registrar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService
    from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
    from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind, sha256_canonical
    from quanxin_life.features import EarlyCycleFeatureConfig

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    payload = b"canonical-csv-fixture"
    payload_sha = sha256_canonical("canonical-csv-fixture")
    registration = CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="UPLOAD",
            cell_id="cell-1",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.0,
            source_uri="upload://cell-1.csv",
            source_sha256=payload_sha,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="upload-v1",
        split_version="split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="upload-cell-1",
                source_kind=SourceKind.OBSERVED,
                uri="upload://cell-1.csv",
                sha256=payload_sha,
                description="API delegation fixture",
                created_at=datetime(2026, 7, 15, tzinfo=UTC),
            ),
        ),
    )
    captured: list[tuple[bytes, CanonicalCsvBatchRegistration]] = []

    def registrar(
        raw_payload: bytes, metadata: CanonicalCsvBatchRegistration
    ) -> str:
        captured.append((raw_payload, metadata))
        return "canonical-csv-batch-id"

    app = create_fastapi_app(
        ToolInvocationService(registry=_registry()),
        canonical_csv_registrar=registrar,
    )
    endpoint = app.routes[("POST", "/v1/batches/canonical-csv")]

    response = asyncio.run(
        endpoint(
            {
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "registration": registration.model_dump(mode="json"),
            }
        )
    )

    assert response == {"record_batch_id": "canonical-csv-batch-id"}
    assert captured == [(payload, registration)]


def test_fastapi_canonical_csv_endpoint_rejects_invalid_base64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quanxin_life.api.app import create_fastapi_app
    from quanxin_life.api.service import ToolInvocationService

    monkeypatch.setattr(
        "quanxin_life.api.app.importlib.import_module",
        lambda _: _FakeFastApiModule,
    )
    app = create_fastapi_app(
        ToolInvocationService(registry=_registry()),
        canonical_csv_registrar=lambda _payload, _registration: "unused",
    )

    with pytest.raises(_FakeHttpException) as invalid:
        asyncio.run(
            app.routes[("POST", "/v1/batches/canonical-csv")](
                {"payload_base64": "not base64!", "registration": {}}
            )
        )

    assert invalid.value.status_code == 422
