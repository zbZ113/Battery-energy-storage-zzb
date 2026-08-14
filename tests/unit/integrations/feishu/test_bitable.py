from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from hashlib import sha256

import pytest

from quanxin_life.integrations.feishu.bitable import (
    CHINESE_ANALYSIS_BITABLE_PROFILE,
    BitableAttachment,
    BitableConflictError,
    BitableMediaUploader,
    BitableProtocolError,
    BitableValidationError,
    BitableWriteAction,
    FeishuBitableWriter,
)

_PLUS_EIGHT = timezone(timedelta(hours=8))


class _FakeBitableClient:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.created = 0
        self.updated = 0
        self.searches: list[str] = []
        self._lock = threading.Lock()

    def search_bitable_records(
        self,
        *,
        app_token: str,
        table_id: str,
        field_name: str,
        field_value: str,
    ) -> dict[str, object]:
        assert app_token == "app_table"
        assert table_id == "tbl_runs"
        assert field_name == "run_id"
        with self._lock:
            self.searches.append(field_value)
            record = self.records.get(field_value)
            return {"items": [] if record is None else [dict(record)]}

    def create_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        assert app_token == "app_table"
        assert table_id == "tbl_runs"
        run_id = fields["run_id"]
        assert isinstance(run_id, str)
        with self._lock:
            self.created += 1
            record_id = f"rec_{self.created}"
            self.records[run_id] = {
                "record_id": record_id,
                "fields": dict(fields),
            }
        return {"record": {"record_id": record_id}}

    def update_bitable_record(
        self,
        *,
        app_token: str,
        table_id: str,
        record_id: str,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        assert app_token == "app_table"
        assert table_id == "tbl_runs"
        run_id = fields["run_id"]
        assert isinstance(run_id, str)
        with self._lock:
            self.updated += 1
            self.records[run_id] = {
                "record_id": record_id,
                "fields": dict(fields),
            }
        return {"record": {"record_id": record_id}}


def _fields(**updates: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": "run-safe",
        "task_type": "predict_cycle_life",
        "task_status": "QUEUED",
        "data_batch_id": "batch-safe",
        "input_file_sha256": "a" * 64,
        "cell_reference": "batch:batch-safe",
        "primary_result_id": "result-safe",
        "model_route": "route-safe",
        "model_version": "model-safe",
        "data_version": "data-safe",
        "feature_version": "feature-safe",
        "evidence_level": "MODEL_INFERENCE",
        "warnings": "route requires manual approval",
        "report_link": "https://reports.example.invalid/report-safe",
        "created_at_utc": datetime(
            2026,
            8,
            7,
            16,
            0,
            tzinfo=_PLUS_EIGHT,
        ),
        "updated_at_utc": datetime(2026, 8, 7, 8, 30, tzinfo=UTC),
    }
    fields.update(updates)
    return fields


def _writer(client: object) -> FeishuBitableWriter:
    return FeishuBitableWriter(
        client,  # type: ignore[arg-type]
        app_token="app_table",
        table_id="tbl_runs",
    )


def test_upsert_creates_then_updates_one_run_record_with_utc_fields() -> None:
    client = _FakeBitableClient()
    writer = _writer(client)

    created = writer.upsert(_fields())
    updated = writer.upsert(_fields(task_status="COMPLETED"))

    assert created.action is BitableWriteAction.CREATED
    assert updated.action is BitableWriteAction.UPDATED
    assert created.record_id == updated.record_id == "rec_1"
    assert client.created == 1
    assert client.updated == 1
    stored = client.records["run-safe"]["fields"]
    assert isinstance(stored, dict)
    assert stored["task_status"] == "COMPLETED"
    assert stored["created_at_utc"] == "2026-08-07T08:00:00+00:00"
    assert stored["updated_at_utc"] == "2026-08-07T08:30:00+00:00"


def test_upsert_accepts_only_scalar_scenario_references() -> None:
    client = _FakeBitableClient()

    _writer(client).upsert(
        _fields(
            task_type="project_storage_lifetime",
            scenario_context_id="3a3c972b-a23e-42c3-af76-e39038806f13",
            scenario_id="baseline",
            scenario_version="baseline-v1",
        )
    )

    stored = client.records["run-safe"]["fields"]
    assert isinstance(stored, dict)
    assert stored["scenario_context_id"] == "3a3c972b-a23e-42c3-af76-e39038806f13"
    assert stored["scenario_id"] == "baseline"
    assert stored["scenario_version"] == "baseline-v1"
    assert all(not isinstance(value, list | dict) for value in stored.values())


def test_chinese_profile_maps_business_fields_and_audited_attachment() -> None:
    class ChineseClient(_FakeBitableClient):
        def __init__(self) -> None:
            super().__init__()
            self.search_field_names: list[str] = []
            self.remote_fields: dict[str, object] = {}

        def search_bitable_records(
            self,
            *,
            app_token: str,
            table_id: str,
            field_name: str,
            field_value: str,
        ) -> dict[str, object]:
            del app_token, table_id
            self.search_field_names.append(field_name)
            self.searches.append(field_value)
            return {"items": []}

        def create_bitable_record(
            self,
            *,
            app_token: str,
            table_id: str,
            fields: Mapping[str, object],
        ) -> dict[str, object]:
            del app_token, table_id
            self.remote_fields = dict(fields)
            return {"record": {"record_id": "rec_chinese"}}

    client = ChineseClient()
    writer = FeishuBitableWriter(
        client,
        app_token="app_table",
        table_id="tbl_runs",
        field_profile=CHINESE_ANALYSIS_BITABLE_PROFILE,
    )

    writer.upsert(
        _fields(
            task_status="SUCCEEDED",
            analysis_summary="已完成有限循环 SOH 轨迹分析",
            curve_attachment=BitableAttachment(file_token="file_curve_safe"),
            curve_source_result_id="result-safe",
            curve_renderer_version="feishu-soh-plot-v1",
            curve_sha256="b" * 64,
            curve_template="FINITE_SOH_CURVE",
        )
    )

    assert client.search_field_names == ["任务ID"]
    assert client.remote_fields["任务ID"] == "run-safe"
    assert client.remote_fields["分析类型"] == "个体早期循环寿命预测"
    assert client.remote_fields["任务状态"] == "已完成"
    assert client.remote_fields["证据类型"] == "模型推理"
    assert client.remote_fields["分析摘要"] == "已完成有限循环 SOH 轨迹分析"
    assert client.remote_fields["分析曲线"] == [
        {"file_token": "file_curve_safe"}
    ]
    assert client.remote_fields["曲线来源结果ID"] == "result-safe"
    assert client.remote_fields["曲线渲染器版本"] == "feishu-soh-plot-v1"
    assert client.remote_fields["曲线SHA256"] == "b" * 64
    assert client.remote_fields["曲线模板"] == "FINITE_SOH_CURVE"


def test_bitable_media_uploader_binds_payload_sha_and_table_token() -> None:
    class MediaClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def upload_bitable_media(self, **kwargs: object) -> dict[str, object]:
            self.calls.append(dict(kwargs))
            return {"file_token": "file_curve_safe"}

    client = MediaClient()
    uploader = BitableMediaUploader(client, app_token="app_table")
    payload = b"audited-curve"

    attachment = uploader.upload_image(
        filename="soh-curve.png",
        content_type="image/png",
        payload=payload,
        expected_sha256=sha256(payload).hexdigest(),
    )

    assert attachment == BitableAttachment(file_token="file_curve_safe")
    assert client.calls == [
        {
            "app_token": "app_table",
            "filename": "soh-curve.png",
            "content_type": "image/png",
            "payload": payload,
        }
    ]
    with pytest.raises(BitableValidationError, match="SHA-256"):
        uploader.upload_image(
            filename="soh-curve.png",
            content_type="image/png",
            payload=payload,
            expected_sha256="0" * 64,
        )
    assert len(client.calls) == 1


def test_upsert_rejects_explicit_null_curve_evidence_before_remote_calls() -> None:
    client = _FakeBitableClient()

    with pytest.raises(BitableValidationError, match="curve evidence is incomplete"):
        _writer(client).upsert(
            _fields(
                curve_attachment=None,
                curve_source_result_id=None,
                curve_renderer_version=None,
                curve_sha256=None,
                curve_template=None,
            )
        )

    assert client.searches == []
    assert client.created == 0
    assert client.updated == 0


def test_upsert_rejects_curve_source_not_bound_to_primary_result() -> None:
    client = _FakeBitableClient()

    with pytest.raises(BitableValidationError, match="primary result"):
        _writer(client).upsert(
            _fields(
                primary_result_id="result-a",
                curve_attachment=BitableAttachment(file_token="file-curve"),
                curve_source_result_id="result-b",
                curve_renderer_version="feishu-rul-summary-v1",
                curve_sha256="b" * 64,
                curve_template="CYCLE_LIFE_SUMMARY",
            )
        )

    assert client.searches == []
    assert client.created == 0
    assert client.updated == 0


def test_upsert_rejects_duplicate_remote_run_records_without_writing() -> None:
    class DuplicateClient(_FakeBitableClient):
        def search_bitable_records(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            return {
                "items": [
                    {"record_id": "rec_1", "fields": {"run_id": "run-safe"}},
                    {"record_id": "rec_2", "fields": {"run_id": "run-safe"}},
                ]
            }

    client = DuplicateClient()

    with pytest.raises(BitableConflictError, match="multiple records"):
        _writer(client).upsert(_fields())

    assert client.created == 0
    assert client.updated == 0


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"run_id": 'run"forged'}, "run_id"),
        ({"trajectory": [1, 2, 3]}, "allowlisted"),
        ({"warnings": ["warning"]}, "scalar"),
        ({"report_link": {"href": "https://example.invalid"}}, "scalar"),
        ({"updated_at_utc": datetime(2026, 8, 7, 8, 30)}, "timezone"),
    ],
)
def test_upsert_rejects_unsafe_fields_before_remote_calls(
    updates: dict[str, object],
    message: str,
) -> None:
    client = _FakeBitableClient()

    with pytest.raises(BitableValidationError, match=message):
        _writer(client).upsert(_fields(**updates))

    assert client.searches == []
    assert client.created == 0
    assert client.updated == 0


def test_upsert_rejects_malformed_search_or_record_responses() -> None:
    class MalformedSearchClient(_FakeBitableClient):
        def search_bitable_records(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            return {"items": "not-a-list"}

    class MalformedRecordClient(_FakeBitableClient):
        def create_bitable_record(
            self,
            *,
            app_token: str,
            table_id: str,
            fields: Mapping[str, object],
        ) -> dict[str, object]:
            del app_token, table_id, fields
            return {"record": {"record_id": "bad/record"}}

    with pytest.raises(BitableProtocolError, match="search response"):
        _writer(MalformedSearchClient()).upsert(_fields())
    with pytest.raises(BitableProtocolError, match="record response"):
        _writer(MalformedRecordClient()).upsert(_fields())


def test_upsert_rejects_a_paginated_search_that_cannot_prove_uniqueness() -> None:
    class PaginatedClient(_FakeBitableClient):
        def search_bitable_records(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            return {
                "items": [
                    {"record_id": "rec_1", "fields": {"run_id": "run-safe"}}
                ],
                "has_more": True,
            }

    with pytest.raises(BitableConflictError, match="additional records"):
        _writer(PaginatedClient()).upsert(_fields())


def test_local_lock_serializes_concurrent_upserts_for_one_run() -> None:
    client = _FakeBitableClient()
    writer = _writer(client)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(lambda _: writer.upsert(_fields()), range(8)))

    assert sum(item.action is BitableWriteAction.CREATED for item in results) == 1
    assert sum(item.action is BitableWriteAction.UPDATED for item in results) == 7
    assert {item.record_id for item in results} == {"rec_1"}
    assert client.created == 1
    assert client.updated == 7
