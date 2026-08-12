from __future__ import annotations

from pathlib import Path

from quanxin_life.persistence.models import FeishuEventReceipt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_0018_adds_integrity_checked_scenario_contexts_and_job_reference() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0018_feishu_scenario_contexts.py"
    ).read_text(encoding="utf-8")

    assert 'revision: str = "0018"' in migration
    assert 'down_revision: str | None = "0017"' in migration
    assert '"feishu_scenario_contexts"' in migration
    for column in (
        "task_type",
        "data_batch_id",
        "route_id",
        "verified_context_json",
        "analysis_input_json",
        "input_sha256",
        "created_by_reference",
        "created_at",
        "scenario_context_id",
    ):
        assert f'"{column}"' in migration


def test_0018_does_not_store_scenario_outputs_or_trajectories() -> None:
    migration = (
        PROJECT_ROOT / "migrations/versions/0018_feishu_scenario_contexts.py"
    ).read_text(encoding="utf-8")

    assert "soh_trajectory" not in migration
    assert "natural_years" not in migration
    assert "equivalent_full_cycles" not in migration


def test_scenario_context_receipt_index_matches_orm_metadata() -> None:
    assert "ix_feishu_receipts_scenario_context_id" in {
        index.name for index in FeishuEventReceipt.__table__.indexes
    }
