from __future__ import annotations

from fastapi.testclient import TestClient

from quanxin_life.integrations.feishu.sandbox import create_fake_feishu_sandbox_app


def test_fake_feishu_sandbox_supports_resources_messages_and_bitable() -> None:
    client = TestClient(create_fake_feishu_sandbox_app())

    health = client.get("/health")
    seeded = client.put(
        "/sandbox/resources/om-source/file-source",
        content=b"cell_id,cycle_index\ncell-a,1\n",
        headers={"content-type": "text/csv"},
    )
    downloaded = client.get(
        "/open-apis/im/v1/messages/om-source/resources/file-source",
        params={"type": "file"},
        headers={"authorization": "Bearer sandbox-token"},
    )
    sent = client.post(
        "/open-apis/im/v1/messages",
        params={"receive_id_type": "chat_id"},
        json={
            "receive_id": "oc-sandbox",
            "msg_type": "text",
            "content": '{"text":"status-only"}',
        },
        headers={"authorization": "Bearer sandbox-token"},
    )
    created = client.post(
        "/open-apis/bitable/v1/apps/app-table/tables/tbl-runs/records",
        json={"fields": {"run_id": "run-safe", "task_status": "QUEUED"}},
        headers={"authorization": "Bearer sandbox-token"},
    )
    searched = client.post(
        "/open-apis/bitable/v1/apps/app-table/tables/tbl-runs/records/search",
        json={
            "filter": {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "run_id",
                        "operator": "is",
                        "value": ["run-safe"],
                    }
                ],
            }
        },
        headers={"authorization": "Bearer sandbox-token"},
    )
    snapshot = client.get("/sandbox/state")

    assert health.json() == {"status": "ok", "service": "fake-feishu-sandbox"}
    assert seeded.status_code == 204
    assert downloaded.content == b"cell_id,cycle_index\ncell-a,1\n"
    assert sent.json()["code"] == 0
    assert created.json()["data"]["record"]["record_id"].startswith("rec-")
    assert len(searched.json()["data"]["items"]) == 1
    assert snapshot.json() == {
        "messages": 1,
        "resources": 1,
        "bitable_records": 1,
    }


def test_fake_feishu_sandbox_rejects_missing_sandbox_bearer() -> None:
    client = TestClient(create_fake_feishu_sandbox_app())

    response = client.post(
        "/open-apis/im/v1/messages",
        json={"receive_id": "oc-sandbox", "msg_type": "text", "content": "{}"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "fake_feishu_authentication_failed"
