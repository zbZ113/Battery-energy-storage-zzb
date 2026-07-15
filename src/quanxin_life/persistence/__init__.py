"""Optional persistence APIs with no import-time database connection."""

from quanxin_life.persistence.database import (
    DatabaseConfig,
    SessionFactory,
    create_engine_from_config,
    create_session_factory,
    session_scope,
)
from quanxin_life.persistence.models import Base

__all__ = [
    "Base",
    "DatabaseConfig",
    "SessionFactory",
    "create_engine_from_config",
    "create_session_factory",
    "session_scope",
]
