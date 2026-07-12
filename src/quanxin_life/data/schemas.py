from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitName(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


class SplitManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    seed: int = 20260712
    train: tuple[str, ...]
    validation: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]

    @model_validator(mode="after")
    def cells_are_unique_and_disjoint(self) -> "SplitManifest":
        groups = (self.train, self.validation, self.calibration, self.test)
        flattened = [cell for group in groups for cell in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("cell overlap exists across dataset splits")
        if any(not cell for cell in flattened):
            raise ValueError("cell identifiers must be non-empty")
        return self

    @property
    def all_cells(self) -> tuple[str, ...]:
        return self.train + self.validation + self.calibration + self.test


class DataQualitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class DataQualityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(min_length=1)
    severity: DataQualitySeverity
    message: str = Field(min_length=1)
    cell_id: str | None = None
    cycle_index: int | None = Field(default=None, ge=0)


class DataQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    issues: tuple[DataQualityIssue, ...] = ()

    @property
    def blocked(self) -> bool:
        return any(issue.severity == DataQualitySeverity.BLOCKING for issue in self.issues)

