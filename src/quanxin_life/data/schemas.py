from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CycleRecord(BaseModel):
    """Canonical sample-level record with explicit SI-derived engineering units."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dataset_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    cycle_index: int = Field(ge=0)
    sample_index: int = Field(ge=0)
    time_s: float = Field(ge=0)
    voltage_v: float = Field(ge=0, le=10)
    current_a: float
    temperature_c: float | None = Field(default=None, ge=-100, le=200)
    charge_capacity_ah: float | None = Field(default=None, ge=0)
    discharge_capacity_ah: float | None = Field(default=None, ge=0)
    internal_resistance_ohm: float | None = Field(default=None, ge=0)
    diagnostic: bool = False
    valid: bool = True


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

    @property
    def quality_score(self) -> float:
        """Conservative, deterministic triage score; not a model performance metric."""
        if self.blocked:
            return 0.0
        penalties = {
            DataQualitySeverity.ERROR: 0.25,
            DataQualitySeverity.WARNING: 0.10,
            DataQualitySeverity.INFO: 0.02,
        }
        return max(0.0, 1.0 - sum(penalties.get(issue.severity, 0.0) for issue in self.issues))

