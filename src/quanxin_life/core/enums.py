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


class AgentRole(StrEnum):
    """Bounded professional roles used by planners and the safe executor."""

    DATA_QUALITY = "data_quality"
    LIFETIME = "lifetime"
    PHYSICS = "physics"
    EXPERIMENT = "experiment"
    SUPERVISOR = "supervisor"


class AgentFailurePolicy(StrEnum):
    """Allowed handling for one failed planned step."""

    STOP = "STOP"
    RETRY_ONCE = "RETRY_ONCE"
    REPLAN = "REPLAN"


class AgentRunStatus(StrEnum):
    """Persisted lifecycle states for one user-visible Agent run."""

    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    FALLBACK = "FALLBACK"


class AgentDispatchStatus(StrEnum):
    """Durable outbox state for an Agent run."""

    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"


class AgentEventType(StrEnum):
    """Public, non-sensitive Agent run timeline events."""

    RUN_CREATED = "RUN_CREATED"
    RUN_DISPATCHED = "RUN_DISPATCHED"
    DISPATCH_FAILED = "DISPATCH_FAILED"
    RUN_CANCELLED = "RUN_CANCELLED"
    APPROVAL_REQUESTED = "APPROVAL_REQUESTED"
    APPROVAL_APPROVED = "APPROVAL_APPROVED"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    RUN_COMPLETED = "RUN_COMPLETED"
    RUN_FAILED = "RUN_FAILED"


class AgentPlanningMode(StrEnum):
    """Auditable source of one Agent plan."""

    LLM = "LLM"
    FIXED_FALLBACK = "FIXED_FALLBACK"


class ApprovalKind(StrEnum):
    """Actions that may not be performed without a human decision."""

    EXTERNAL_WRITE = "EXTERNAL_WRITE"
    FORMAL_DECISION = "FORMAL_DECISION"
    EXPERIMENT_ACTION = "EXPERIMENT_ACTION"


class ApprovalStatus(StrEnum):
    """Persisted state of one approval request."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class KnowledgeReviewStatus(StrEnum):
    """Review state controlling whether a document may be cited."""

    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class UserRole(StrEnum):
    """Application roles used for server-side authorization."""

    ADMIN = "ADMIN"
    MEMBER = "MEMBER"
    JUDGE = "JUDGE"


class UserStatus(StrEnum):
    """Account states enforced by server-side authentication."""

    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"


class SessionStatus(StrEnum):
    """Opaque browser-session states stored only on the server."""

    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class ProjectStatus(StrEnum):
    """Lifecycle states for a persisted analysis project."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class DatasetStatus(StrEnum):
    """Lifecycle states for a versioned dataset registration."""

    DRAFT = "DRAFT"
    FROZEN = "FROZEN"
