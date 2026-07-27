from quanxin_life.core import (
    AdvancedCalibrationMaterializationStatus,
    ProjectStatus,
)


def test_project_status_values_are_stable() -> None:
    assert [item.value for item in ProjectStatus] == ["ACTIVE", "ARCHIVED"]


def test_advanced_calibration_materialization_status_values_are_stable() -> None:
    assert [item.value for item in AdvancedCalibrationMaterializationStatus] == [
        "PENDING",
        "RUNNING",
        "READY",
        "FAILED",
        "STALE",
    ]
