from __future__ import annotations

from fastapi.testclient import TestClient

from quanxin_life.integrations.feishu.sandbox import (
    create_fake_feishu_sandbox_app,
)


def test_sandbox_accepts_bitable_media_chinese_fields_and_attachment_search() -> None:
    client = TestClient(create_fake_feishu_sandbox_app())
    headers = {"Authorization": "Bearer sandbox-token"}

    uploaded = client.post(
        "/open-apis/drive/v1/medias/upload_all",
        headers=headers,
        data={
            "file_name": "寿命分析.png",
            "parent_type": "bitable_image",
            "parent_node": "app-sandbox",
            "size": "12",
        },
        files={"file": ("寿命分析.png", b"audited-png", "image/png")},
    )

    assert uploaded.status_code == 200
    file_token = uploaded.json()["data"]["file_token"]
    created = client.post(
        "/open-apis/bitable/v1/apps/app-sandbox/tables/tbl-sandbox/records",
        headers=headers,
        json={
            "fields": {
                "任务ID": "run-safe",
                "分析曲线": [{"file_token": file_token}],
                "分析摘要": "已完成个体早期循环寿命预测",
            }
        },
    )
    assert created.status_code == 200

    searched = client.post(
        "/open-apis/bitable/v1/apps/app-sandbox/tables/tbl-sandbox/records/search",
        headers=headers,
        json={
            "filter": {
                "conjunction": "and",
                "conditions": [
                    {
                        "field_name": "任务ID",
                        "operator": "is",
                        "value": ["run-safe"],
                    }
                ],
            }
        },
    )

    assert searched.status_code == 200
    items = searched.json()["data"]["items"]
    assert len(items) == 1
    assert items[0]["fields"]["分析曲线"] == [{"file_token": file_token}]
