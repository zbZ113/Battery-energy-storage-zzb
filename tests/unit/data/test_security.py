from pathlib import Path

import pytest

from quanxin_life.data.security import assert_safe_external_data_file


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".joblib", ".pth", ".pt"])
def test_unsafe_serialized_external_files_are_rejected(suffix: str) -> None:
    with pytest.raises(ValueError, match="unsafe serialized artifact"):
        assert_safe_external_data_file(Path(f"incoming{suffix}"))


def test_non_executable_data_format_is_allowed() -> None:
    assert_safe_external_data_file(Path("cycles.parquet"))
