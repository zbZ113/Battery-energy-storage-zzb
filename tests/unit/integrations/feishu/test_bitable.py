from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

from quanxin_life.integrations.feishu.bitable import (
    BitableConflictError,
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
