from __future__ import annotations

import json

import pytest

from quanxin_life.integrations.feishu.cards import build_run_reference_card


def test_running_card_hides_machine_references_from_the_primary_view() -> None:
    card = build_run_reference_card(run_id="run-001", result_id="result-001")

    rendered = json.dumps(card, ensure_ascii=False)

    assert "分析进行中" in rendered
    assert "run-001" not in rendered
    assert "result-001" not in rendered
    assert "SOH" not in rendered
    assert "RUL" not in rendered
    assert "寿命" not in rendered
    assert "置信区间" not in rendered
    assert "业务指标" not in rendered


def test_running_card_is_stable_before_any_result_exists() -> None:
    card = build_run_reference_card(run_id="run-001")

    rendered = json.dumps(card, ensure_ascii=False)

    assert "分析进行中" in rendered
    assert "run-001" not in rendered
    assert "result_id" not in rendered


@pytest.mark.parametrize("run_id", ["", " ", "\n"])
def test_message_card_rejects_blank_run_identifiers(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_run_reference_card(run_id=run_id)


@pytest.mark.parametrize("run_id", ["run\n**forged**", "run/value", "运行-001"])
def test_message_card_rejects_identifiers_that_could_inject_card_content(run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_run_reference_card(run_id=run_id)
