from enum import StrEnum


class EvidenceLevel(StrEnum):
    DATA_DIRECT = "DATA_DIRECT"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    PHYSICS_REFERENCE = "PHYSICS_REFERENCE"
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"
    UNDETERMINED = "UNDETERMINED"


class Decision(StrEnum):
    ADMIT = "ADMIT"
    RECHECK = "RECHECK"
    DOWNGRADE = "DOWNGRADE"
    REJECT = "REJECT"


class SourceKind(StrEnum):
    OBSERVED = "OBSERVED"
    PREDICTED = "PREDICTED"
    NEWLY_OBSERVED = "NEWLY_OBSERVED"
    SIMULATED = "SIMULATED"


class PredictionTarget(StrEnum):
    """Prediction labels permitted by the battery lifetime modeling contract."""

    EOL80_CYCLE = "eol80_cycle"
