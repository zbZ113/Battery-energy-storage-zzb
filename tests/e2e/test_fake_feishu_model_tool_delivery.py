from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from quanxin_life.api.service import ToolInvocationService
from quanxin_life.application.ingestion import (
    CanonicalCsvBatchRegistration,
    InMemoryVerifiedEarlyCycleBatchStore,
)
from quanxin_life.audit import AuditLedger
from quanxin_life.core import (
    CellMetadata,
    EvidenceLevel,
    ProvenanceRecord,
    SourceKind,
    ToolResult,
    sha256_canonical,
)
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.attachments import FeishuAttachmentPolicy
from quanxin_life.integrations.feishu.bitable import FeishuBitableWriter
from quanxin_life.integrations.feishu.cards import (
    AuditedCardBuilder,
    AuditedResultAuthorization,
    FeishuCardStatus,
    build_status_card,
)
from quanxin_life.integrations.feishu.client import (
    FeishuClient,
    FeishuClientConfig,
    FeishuHttpRequest,
    FeishuHttpResponse,
)
from quanxin_life.integrations.feishu.report_delivery import FeishuReportDelivery
from quanxin_life.integrations.feishu.routing import (
    FeishuInboundEventKind,
    parse_feishu_event_reference,
)
from quanxin_life.integrations.feishu.sandbox import create_fake_feishu_sandbox_app
from quanxin_life.integrations.feishu.workflow import (
    FeishuAnalysisTask,
    FeishuAnalysisWorkflow,
    FeishuWorkflowRejected,
)
from quanxin_life.reporting import AuditedReportArtifactExporter
from quanxin_life.tools import StandardToolName, ToolDefinition, ToolRegistry
from quanxin_life.tools.audited_report import (
    AuditedReportClaimReference,
    GenerateAuditedReportToolInput,
    NumericEvidenceReference,
    ReportClaimKind,
    ReportKind,
    execute_generate_audited_report_tool,
)
from quanxin_life.tools.data_quality import (
    ValidateBatteryDataToolInput,
    register_validate_battery_data_tool,
)

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
CSV_PAYLOAD = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"sandbox-data,cell-sandbox,1,0,0.0,3.6,1.0,25.0,1.2,1.1,0.02,true,true\n"
    b"sandbox-data,cell-sandbox,1,1,1.0,3.7,1.0,25.0,1.2,1.1,0.02,true,true\n"
    b"sandbox-data,cell-sandbox,20,0,0.0,3.5,1.0,25.0,1.1,1.0,0.03,true,true\n"
    b"sandbox-data,cell-sandbox,20,1,1.0,3.6,1.0,25.0,1.1,1.0,0.03,true,true\n"
)


class _SandboxAsgiTransport:
    """Drive the real outbound client against the local Fake Feishu ASGI app."""

    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.requests: list[FeishuHttpRequest] = []

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        self.requests.append(request)
        path = urlsplit(request.url).path
        files = [
            (
                item.field_name,
                (item.filename, item.payload, item.content_type),
            )
            for item in request.files
        ]
        response = self._client.request(
            request.method,
            path,
            headers=request.headers,
            params=request.params,
            json=request.json_body,
            data=request.form or None,
            files=files or None,
        )
        json_body: dict[str, object] | None = None
        if "application/json" in response.headers.get("content-type", "").lower():
            candidate = response.json()
            if isinstance(candidate, dict):
                json_body = candidate
        return FeishuHttpResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            json_body=json_body,
            body=response.content,
        )


class _SandboxPredictionInput(ValidateBatteryDataToolInput):
    """Test-only typed input for a deterministic non-production tool route."""


class _SandboxRouteAuthorizer:
    def __init__(self, *, active: bool) -> None:
        self._active = active

    def authorize(
        self,
        *,
        tool_name: StandardToolName,
        validation_result: ToolResult,
    ) -> None:
        del tool_name, validation_result
        if not self._active:
            raise FeishuWorkflowRejected("MODEL_ROUTE_NOT_ACTIVATED")


class _SameEvidenceInputBinder:
    def bind_analysis_input(
        self,
        *,
        task: FeishuAnalysisTask,
        validation_result: ToolResult,
        validation_input: Mapping[str, object],
        requested_analysis_input: Mapping[str, object],
    ) -> Mapping[str, object]:
        del task
        if validation_result.data_version != validation_input.get("data_version"):
            raise FeishuWorkflowRejected("ANALYSIS_INPUT_NOT_BOUND_TO_VALIDATION")
        for field in ("data_version", "feature_version", "provenance"):
            if validation_input.get(field) != requested_analysis_input.get(field):
                raise FeishuWorkflowRejected("ANALYSIS_INPUT_NOT_BOUND_TO_VALIDATION")
        return dict(requested_analysis_input)


class _SandboxCardAuthorizer:
    def authorize(self, result: ToolResult) -> AuditedResultAuthorization:
        if "SANDBOX_TEST_ONLY_NOT_A_MODEL" not in result.warnings:
            raise ValueError("sandbox card received a non-sandbox result")
        return AuditedResultAuthorization(
            allowed=True,
            route_id="sandbox-test-only-route",
            activation_status="SANDBOX_TEST_ONLY",
            evidence_level=EvidenceLevel.MODEL_INFERENCE,
            supported_domain="local-fake-feishu-test",
        )


class _SandboxBindingVerifier:
    def __init__(self, *, run_id: str, result_ids: set[str]) -> None:
        self._run_id = run_id
        self._result_ids = result_ids

    def is_result_bound_to_run(self, *, run_id: str, result_id: str) -> bool:
        return run_id == self._run_id and result_id in self._result_ids


def test_fake_feishu_pipeline_traverses_transport_routing_attachment_and_workflow() -> None:
    outcome = _run_sandbox_pipeline(route_active=True)

    assert outcome["prediction_calls"] == 1
    assert outcome["sandbox_state"] == {
        "messages": 2,
        "resources": 1,
        "bitable_records": 1,
    }
    assert outcome["report_delivered"] is True
    assert outcome["used_sandbox_transport"] is True


def test_fake_feishu_pipeline_rejects_inactive_route_before_prediction_or_values() -> None:
    outcome = _run_sandbox_pipeline(route_active=False)

    assert outcome["prediction_calls"] == 0
    assert outcome["rejection_code"] == "MODEL_ROUTE_NOT_ACTIVATED"
    assert outcome["card_contains_business_values"] is False


def _run_sandbox_pipeline(*, route_active: bool) -> dict[str, object]:
    sandbox = TestClient(create_fake_feishu_sandbox_app())
    seeded = sandbox.put(
        "/sandbox/resources/om-source/file-source",
        content=CSV_PAYLOAD,
        headers={"content-type": "text/csv"},
    )
    assert seeded.status_code == 204
    event = parse_feishu_event_reference(
        {
            "header": {
                "event_id": "evt-sandbox-file",
                "event_type": "im.message.receive_v1",
            },
            "event": {
                "sender": {"sender_id": {"open_id": "ou-sandbox"}},
                "message": {
                    "message_id": "om-source",
                    "chat_id": "oc-sandbox",
                    "message_type": "file",
                    "content": json.dumps(
                        {
                            "file_key": "file-source",
                            "file_name": "sandbox-cell.csv",
                        }
                    ),
                },
            },
        }
    )
    assert event.kind is FeishuInboundEventKind.FILE
    assert event.message_id is not None
    assert event.file_key is not None
    assert event.file_name is not None
    assert event.chat_id is not None

    transport = _SandboxAsgiTransport(sandbox)
    feishu = FeishuClient(
        FeishuClientConfig(
            app_id="sandbox-app",
            app_secret=SecretStr("sandbox-secret"),
            base_url="http://testserver/open-apis",
            backoff_base_seconds=0,
        ),
        transport=transport,
    )
    downloaded = feishu.download_message_resource(
        message_id=event.message_id,
        file_key=event.file_key,
        resource_type="file",
    )
    attachment = FeishuAttachmentPolicy().verify(
        filename=event.file_name,
        content_type="text/csv",
        payload=downloaded,
    )
    registration = _registration(attachment.sha256)
    batch_store = InMemoryVerifiedEarlyCycleBatchStore()
    batch_id = batch_store.register_canonical_csv(
        attachment.payload,
        registration=registration,
    )
    batch = batch_store.resolve_verified_early_cycle_batch(batch_id)
    tool_input = ValidateBatteryDataToolInput(
        records=batch.records,
        data_version=batch.data_version,
        feature_version=batch.feature_config.feature_version,
        provenance=batch.provenance,
        validated_at=NOW,
    ).model_dump(mode="json")
    prediction_calls: list[str] = []
    service = _sandbox_tool_service(prediction_calls)
    workflow = FeishuAnalysisWorkflow(
        service,
        route_authorizer=_SandboxRouteAuthorizer(active=route_active),
        input_binder=_SameEvidenceInputBinder(),
    )
    run_id = str(uuid4())
    try:
        workflow_outcome = workflow.run(
            task=FeishuAnalysisTask.PREDICT_CYCLE_LIFE,
            validation_input=tool_input,
            analysis_input=tool_input,
        )
    except FeishuWorkflowRejected as exc:
        rejection_code = str(exc)
        card = build_status_card(
            status=FeishuCardStatus.REJECTED,
            run_id=run_id,
            reason_code=rejection_code,
        )
        feishu.update_card(message_id=event.message_id, card=card)
        rendered = json.dumps(card, ensure_ascii=False)
        return {
            "prediction_calls": len(prediction_calls),
            "rejection_code": rejection_code,
            "card_contains_business_values": "values." in rendered,
        }

    validation = workflow_outcome.validation_result
    prediction = workflow_outcome.analysis_result
    assert service.audit_ledger is not None
    ledger = service.audit_ledger
    card = AuditedCardBuilder(
        ledger,
        authorizer=_SandboxCardAuthorizer(),
        binding_verifier=_SandboxBindingVerifier(
            run_id=run_id,
            result_ids={validation.result_id, prediction.result_id},
        ),
    ).build_result_card(run_id=run_id, result_id=prediction.result_id)
    feishu.update_card(message_id=event.message_id, card=card)

    report = execute_generate_audited_report_tool(
        GenerateAuditedReportToolInput(
            report_kind=ReportKind.LIFETIME_DECISION,
            claims=(
                AuditedReportClaimReference(
                    claim_kind=ReportClaimKind.LIFETIME_PREDICTION,
                    numeric_evidence=(
                        NumericEvidenceReference(
                            result_id=prediction.result_id,
                            json_path=(
                                "values.artifact.cycle_life_prediction.predicted_cycle"
                            ),
                        ),
                    ),
                ),
            ),
            upstream_result_ids=(prediction.result_id,),
        ),
        audit_ledger=ledger,
        clock=lambda: NOW,
    )
    ledger.register_result(report)
    delivered = FeishuReportDelivery(
        AuditedReportArtifactExporter(ledger),
        feishu,
        result_resolver=ledger,
    ).deliver(result_id=report.result_id, chat_id=event.chat_id)
    FeishuBitableWriter(
        feishu,
        app_token="app-sandbox",
        table_id="tbl-sandbox",
    ).upsert(
        {
            "run_id": run_id,
            "task_type": FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
            "task_status": "COMPLETED",
            "data_batch_id": batch_id,
            "input_file_sha256": attachment.sha256,
            "cell_reference": batch.metadata.cell_id,
            "primary_result_id": prediction.result_id,
            "model_route": "sandbox-test-only-route",
            "model_version": prediction.model_version,
            "data_version": prediction.data_version,
            "feature_version": prediction.feature_version,
            "evidence_level": EvidenceLevel.MODEL_INFERENCE.value,
            "warnings": "SANDBOX_TEST_ONLY_NOT_A_MODEL",
            "report_link": f"https://sandbox.invalid/files/{delivered.file_key}",
            "created_at_utc": NOW,
            "updated_at_utc": NOW,
        }
    )
    snapshot = sandbox.get("/sandbox/state")
    assert snapshot.status_code == 200
    return {
        "prediction_calls": len(prediction_calls),
        "sandbox_state": snapshot.json(),
        "report_delivered": delivered.source_result_id == report.result_id,
        "used_sandbox_transport": any(
            request.url.endswith(
                "/im/v1/messages/om-source/resources/file-source"
            )
            for request in transport.requests
        ),
    }


def _registration(payload_sha256: str) -> CanonicalCsvBatchRegistration:
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="sandbox-data",
            cell_id="cell-sandbox",
            chemistry="LFP/graphite",
            nominal_capacity_ah=1.2,
            source_uri="feishu-sandbox://message/om-source/resource/file-source",
            source_sha256=payload_sha256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="sandbox-csv-v1",
        split_version="sandbox-cell-split-v1",
        provenance=(
            ProvenanceRecord(
                source_id="file-source",
                source_kind=SourceKind.OBSERVED,
                uri="feishu-sandbox://message/om-source/resource/file-source",
                sha256=payload_sha256,
                description="Verified local Fake Feishu canonical CSV",
                created_at=NOW,
            ),
        ),
    )


def _sandbox_tool_service(prediction_calls: list[str]) -> ToolInvocationService:
    registry = ToolRegistry()
    register_validate_battery_data_tool(registry)

    def predict(input_value: _SandboxPredictionInput) -> ToolResult:
        prediction_calls.append(input_value.data_version)
        last_observed_cycle = max(record.cycle_index for record in input_value.records)
        sandbox_projection = last_observed_cycle + len(input_value.records)
        return ToolResult(
            result_id=str(uuid4()),
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE.value,
            tool_version="advanced-rul-prediction-tool-v1",
            model_version="sandbox-deterministic-test-tool-v1",
            data_version=input_value.data_version,
            feature_version=input_value.feature_version,
            input_hash=sha256_canonical(input_value.model_dump(mode="json")),
            values={
                "artifact_type": "quanxin_life.advanced_rul_prediction.v1",
                "artifact": {
                    "cell_id": "cell-sandbox",
                    "cutoff_cycle": last_observed_cycle,
                    "cycle_life_prediction": {
                        "predicted_cycle": sandbox_projection
                    },
                    "derived_remaining_cycles": (
                        sandbox_projection - last_observed_cycle
                    ),
                }
            },
            uncertainty=None,
            warnings=["SANDBOX_TEST_ONLY_NOT_A_MODEL"],
            provenance=list(input_value.provenance),
            created_at=input_value.validated_at,
        )

    registry.register(
        ToolDefinition(
            tool_name=StandardToolName.PREDICT_CYCLE_LIFE,
            tool_version="advanced-rul-prediction-tool-v1",
            input_model=_SandboxPredictionInput,
            executor=predict,
        )
    )
    return ToolInvocationService(registry=registry, audit_ledger=AuditLedger())


def test_sandbox_csv_fixture_sha_is_stable() -> None:
    assert sha256(CSV_PAYLOAD).hexdigest() == _registration(
        sha256(CSV_PAYLOAD).hexdigest()
    ).metadata.source_sha256
