from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_0019_adds_only_external_delivery_checkpoint_references() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0019_feishu_delivery_checkpoints.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "0019"' in migration
    assert 'down_revision: str | None = "0018"' in migration
    for column in (
        "scenario_image_key",
        "result_card_message_id",
        "report_message_id",
        "report_card_message_id",
    ):
        assert f'"{column}"' in migration
    assert "soh" not in migration.casefold()
    assert "trajectory" not in migration.casefold()
