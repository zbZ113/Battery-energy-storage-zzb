from __future__ import annotations

from pathlib import Path

import pytest

from quanxin_life.data.adapters.hust import load_hust_layout, verify_hust_output


def test_unapproved_layout_is_explicitly_blocked() -> None:
    layout = load_hust_layout(
        Path("configs/data_layouts/hust_mendeley_v2_layout_v1.json")
    )

    assert layout.review_status == "BLOCKED_REVIEW"
    assert layout.member_count == 77
    assert layout.fields == ()
    assert layout.unresolved_reasons
    with pytest.raises(ValueError, match="BLOCKED_REVIEW"):
        verify_hust_output(Path("does-not-exist"), layout=layout)
