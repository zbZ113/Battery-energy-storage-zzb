from quanxin_life.core import ProjectStatus


def test_project_status_values_are_stable() -> None:
    assert [item.value for item in ProjectStatus] == ["ACTIVE", "ARCHIVED"]
