"""Explicit SQLAlchemy engine and transaction lifecycle helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Database settings supplied by an application composition root."""

    url: str
    echo: bool = False
    pool_pre_ping: bool = True

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("database URL must be provided explicitly")


SessionFactory = sessionmaker[Session]


def create_engine_from_config(config: DatabaseConfig) -> Engine:
    """Create an engine without reading environment variables or global state."""

    return create_engine(
        config.url,
        echo=config.echo,
        pool_pre_ping=config.pool_pre_ping,
    )


def create_session_factory(engine: Engine) -> SessionFactory:
    """Build sessions bound to an explicitly constructed engine."""

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
        close_resets_only=False,
    )


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    """Commit successful work and always roll back failures and close the session."""

    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "DatabaseConfig",
    "SessionFactory",
    "create_engine_from_config",
    "create_session_factory",
    "session_scope",
]
