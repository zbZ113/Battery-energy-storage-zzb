from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import create_engine

from quanxin_life.application.feishu_aily_assembly import (
    FeishuAilyAssemblyConfig,
    create_feishu_aily_components,
)
from quanxin_life.application.ingestion import CanonicalCsvBatchRegistration
from quanxin_life.core import CellMetadata, ProvenanceRecord, SourceKind
from quanxin_life.features import EarlyCycleFeatureConfig
from quanxin_life.integrations.feishu.client import (
    FeishuHttpRequest,
    FeishuHttpResponse,
)
from quanxin_life.integrations.feishu.jobs import (
    FeishuAnalysisJobOrigin,
    FeishuAnalysisJobStatus,
)
from quanxin_life.integrations.feishu.sandbox import create_fake_feishu_sandbox_app
from quanxin_life.integrations.feishu.workflow import FeishuAnalysisTask
from quanxin_life.persistence import Base, create_session_factory
from quanxin_life.persistence.models import FeishuEventReceipt
from quanxin_life.scenarios import OperationScenario, ScenarioSegment

NOW = datetime(2026, 8, 12, 13, 0, tzinfo=UTC)
SOURCE_RUN_ID = "9fcc13e6-753e-4dc8-850c-f7a57cd6072a"
CSV_PAYLOAD = (
    b"dataset_id,cell_id,cycle_index,sample_index,time_s,voltage_v,current_a,"
    b"temperature_c,charge_capacity_ah,discharge_capacity_ah,"
    b"internal_resistance_ohm,diagnostic,valid\n"
    b"aily-scenario,cell-250ah,1,0,0.0,3.6,125.0,25.0,250.0,249.0,0.02,true,true\n"
    b"aily-scenario,cell-250ah,20,0,0.0,3.5,125.0,25.0,248.0,247.0,0.03,true,true\n"
)


class _AsyncResult:
    id = "task-fake-aily-scenario"


class _CeleryApp:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    def send_task(
        self,
        name: str,
        *,
        kwargs: dict[str, str],
        queue: str,
    ) -> _AsyncResult:
        self.sent.append({"name": name, "kwargs": kwargs, "queue": queue})
        return _AsyncResult()


class _AuthorizedDataIdentity:
    def __init__(self) -> None:
        self.data_batch_id: str | None = None
        self.calls: list[tuple[str, str]] = []

    def resolve_source_job(
        self,
        *,
        source_run_id: str,
        data_batch_id: str,
    ) -> None:
        self.calls.append((source_run_id, data_batch_id))
        if source_run_id != SOURCE_RUN_ID or data_batch_id != self.data_batch_id:
            raise ValueError("Aily data identity is not authorized")


class _SandboxAsgiTransport:
    def __init__(self, client: TestClient) -> None:
        self._client = client
        self.requests: list[FeishuHttpRequest] = []

    def request(self, request: FeishuHttpRequest) -> FeishuHttpResponse:
        self.requests.append(request)
        files = [
            (
                item.field_name,
                (item.filename, item.payload, item.content_type),
            )
            for item in request.files
        ]
        response = self._client.request(
            request.method,
            urlsplit(request.url).path,
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


def _scenario(
    *,
    scenario_id: str,
    temperature_c: float,
    discharge_c_rate: float,
) -> dict[str, object]:
    return OperationScenario(
        scenario_id=scenario_id,
        scenario_version=f"{scenario_id}-v1",
        horizon_years=25,
        eol_threshold=0.8,
        segments=(
            ScenarioSegment(
                segment_id="years-1-25",
                start_year=0,
                end_year=25,
                temperature_c=temperature_c,
                charge_c_rate=0.5,
                discharge_c_rate=discharge_c_rate,
                soc_lower_bound=0.1,
                soc_upper_bound=0.9,
                dod=0.8,
                equivalent_full_cycles_per_year=120.0,
                rest_duration_hours=1.0,
            ),
        ),
    ).model_dump(mode="json")


def _registration() -> CanonicalCsvBatchRegistration:
    payload_sha256 = sha256(CSV_PAYLOAD).hexdigest()
    return CanonicalCsvBatchRegistration(
        metadata=CellMetadata(
            dataset_id="aily-scenario",
            cell_id="cell-250ah",
            chemistry="LFP/graphite",
            nominal_capacity_ah=250.0,
            source_uri="fixture://fake-aily/canonical.csv",
            source_sha256=payload_sha256,
            schema_version="cycle-record-v1",
        ),
        feature_config=EarlyCycleFeatureConfig(cutoff_cycle=20),
        data_version="fake-aily-scenario-v1",
        split_version="fake-aily-scenario-cell-v1",
        provenance=(
            ProvenanceRecord(
                source_id="fake-aily-canonical-csv",
                source_kind=SourceKind.OBSERVED,
                uri="fixture://fake-aily/canonical.csv",
                sha256=payload_sha256,
                description="Verified canonical CSV for the Fake Aily scenario E2E.",
                created_at=NOW,
            ),
        ),
    )


def test_production_assembly_runs_fake_aily_scenario_without_chat_side_effects(
    tmp_path,
) -> None:
    sandbox = TestClient(create_fake_feishu_sandbox_app())
    transport = _SandboxAsgiTransport(sandbox)
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'runtime.sqlite3'}")
    Base.metadata.create_all(engine)
    session_factory = create_session_factory(engine)
    celery_app = _CeleryApp()
    identities = _AuthorizedDataIdentity()
    components = create_feishu_aily_components(
        session_factory=session_factory,
        celery_app=celery_app,
        config=FeishuAilyAssemblyConfig(
            app_id="fake-aily-app",
            app_secret=SecretStr("fake-aily-secret"),
            verification_token=SecretStr("fake-verification-token"),
            encrypt_key=SecretStr("0123456789abcdef"),
            aily_connector_api_key=SecretStr("connector-secret"),
            bitable_app_token="app-sandbox",
            bitable_table_id="tbl-sandbox",
            external_https_base_url="https://integration.example.test",
            data_root=tmp_path / "verified-batches",
            allow_candidate_scenario_execution=True,
            allow_candidate_scenario_results=True,
        ),
        feishu_transport=transport,
        aily_data_identity_resolver=identities,
        clock=lambda: NOW,
    )
    batch_id = components.batch_store.register_canonical_csv(
        CSV_PAYLOAD,
        registration=_registration(),
    )
    identities.data_batch_id = batch_id
    with session_factory.begin() as session:
        session.add(
            FeishuEventReceipt(
                id=str(uuid4()),
                event_id="evt-fake-aily-source",
                event_type="im.message.receive_v1",
                payload_sha256="a" * 64,
                status="PROCESSED",
                attempt_count=1,
                received_at=NOW,
                processed_at=NOW,
                job_id=SOURCE_RUN_ID,
                job_origin=FeishuAnalysisJobOrigin.FEISHU.value,
                job_status=FeishuAnalysisJobStatus.SUCCEEDED.value,
                job_stage="SUCCEEDED",
                task_type=FeishuAnalysisTask.PREDICT_CYCLE_LIFE.value,
                run_id=SOURCE_RUN_ID,
                chat_id="oc-fake-aily",
                sender_id="ou-fake-aily",
                receive_id_type="chat_id",
                event_time=NOW,
                record_batch_id=batch_id,
                cell_reference="cell-250ah",
                input_file_sha256=sha256(CSV_PAYLOAD).hexdigest(),
                validation_result_id=str(uuid4()),
                job_attempt_count=1,
                job_created_at=NOW,
                job_updated_at=NOW,
                job_completed_at=NOW,
            )
        )
    app = FastAPI()
    app.include_router(components.aily_http_adapter.router)
    client = TestClient(app)
    headers = {"Authorization": "Bearer connector-secret"}

    context_response = client.post(
        "/v1/aily/scenario-contexts",
        headers=headers,
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "source_run_id": SOURCE_RUN_ID,
            "data_batch_id": batch_id,
            "cell_format": "prismatic",
            "baseline": _scenario(
                scenario_id="baseline",
                temperature_c=25.0,
                discharge_c_rate=0.5,
            ),
            "comparisons": [
                _scenario(
                    scenario_id="warmer-higher-rate",
                    temperature_c=35.0,
                    discharge_c_rate=1.0,
                )
            ],
        },
    )
    assert context_response.status_code == 201
    scenario_context_id = context_response.json()["scenario_context_id"]

    task_response = client.post(
        "/v1/aily/analysis-tasks",
        headers=headers,
        json={
            "task_type": FeishuAnalysisTask.COMPARE_OPERATION_SCENARIOS.value,
            "source_run_id": SOURCE_RUN_ID,
            "scenario_context_id": scenario_context_id,
        },
    )
    assert task_response.status_code == 202
    run_id = task_response.json()["run_id"]
    assert celery_app.sent[0]["kwargs"] == {"job_id": run_id}

    status = components.worker.execute(job_id=run_id)

    assert status is FeishuAnalysisJobStatus.SUCCEEDED
    completed = client.get(f"/v1/aily/analysis-tasks/{run_id}", headers=headers)
    assert completed.status_code == 200
    assert completed.json()["status"] == "COMPLETED"
    job = components.job_store.get(run_id)
    assert job.job_origin is FeishuAnalysisJobOrigin.AILY
    assert job.source_job_id == SOURCE_RUN_ID
    assert job.record_batch_id == batch_id
    assert job.analysis_result_id is not None
    assert job.report_result_id is not None
    assert job.bitable_curve_file_token is not None
    assert job.bitable_curve_source_result_id == job.analysis_result_id
    assert job.bitable_curve_renderer_version == "feishu-scenario-plot-v2"
    assert job.bitable_curve_template == "SCENARIO_COMPARISON"
    result_response = client.get(
        f"/v1/aily/analysis-tasks/{run_id}/results/{job.analysis_result_id}",
        headers=headers,
    )
    assert result_response.status_code == 200
    artifact = result_response.json()["values"]["artifact"]
    assert artifact["status"] == "COMPLETED"
    assert set(artifact["baseline"]["milestone_soh"]) == {"15", "20", "25"}
    assert artifact["route_id"] == "blast-lite-lfp-gr-250ah-prismatic-2019-v1"
    report_response = client.get(
        f"/v1/aily/analysis-tasks/{run_id}/reports/{job.report_result_id}",
        headers=headers,
    )
    assert report_response.status_code == 200
    assert report_response.headers["x-tool-result-id"] == job.report_result_id
    assert b"PHYSICS_REFERENCE" in report_response.content

    assert sandbox.get("/sandbox/state").json() == {
        "messages": 0,
        "resources": 0,
        "bitable_records": 1,
    }
    fields = _created_bitable_fields(transport)
    assert fields["任务ID"] == run_id
    assert fields["结果ID"] == job.analysis_result_id
    assert fields["工况ID"] == "baseline"
    assert fields["工况版本"] == "baseline-v1"
    assert fields["证据类型"] == "物理参考推演"
    assert fields["分析摘要"] == "已完成参考工况退化对比"
    assert fields["适用边界"] == (
        "结果为物理参考工况。不是目标电芯个体寿命结论"
    )
    assert fields["详细报告"] == (
        f"https://integration.example.test/v1/aily/analysis-tasks/{run_id}"
        f"/reports/{job.report_result_id}"
    )
    assert fields["分析曲线"] == [
        {"file_token": job.bitable_curve_file_token}
    ]
    assert fields["曲线来源结果ID"] == job.analysis_result_id
    assert fields["曲线渲染器版本"] == job.bitable_curve_renderer_version
    assert fields["曲线SHA256"] == job.bitable_curve_sha256
    assert fields["曲线模板"] == "工况退化对比"
    assert all(
        not isinstance(value, list | dict)
        for key, value in fields.items()
        if key != "分析曲线"
    )
    assert "natural_years" not in repr(fields)
    assert "soh" not in repr(fields).casefold()
    assert all(
        urlsplit(request.url).path not in {
            "/open-apis/im/v1/messages",
            "/open-apis/im/v1/files",
            "/open-apis/im/v1/images",
        }
        for request in transport.requests
    )
    assert sum(
        urlsplit(request.url).path == "/open-apis/drive/v1/medias/upload_all"
        for request in transport.requests
    ) == 1
    assert identities.calls == [(SOURCE_RUN_ID, batch_id)] * 6


def _created_bitable_fields(
    transport: _SandboxAsgiTransport,
) -> dict[str, object]:
    request = next(
        item
        for item in transport.requests
        if item.method == "POST"
        and urlsplit(item.url).path
        == "/open-apis/bitable/v1/apps/app-sandbox/tables/tbl-sandbox/records"
    )
    assert isinstance(request.json_body, dict)
    fields = request.json_body.get("fields")
    assert isinstance(fields, dict)
    return fields
