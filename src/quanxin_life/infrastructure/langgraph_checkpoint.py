"""Strict PostgreSQL checkpointer construction for LangGraph control state."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection
from psycopg.rows import dict_row


def create_checkpoint_serializer() -> JsonPlusSerializer:
    """Create a serializer that never falls back to executable pickle payloads."""

    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_json_modules=(),
        allowed_msgpack_modules=(),
    )


def validate_postgres_checkpoint_dsn(value: str) -> str:
    """Reject silent in-memory or file-based storage in the formal runtime."""

    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized.startswith(("postgresql://", "postgres://")):
        raise ValueError("formal LangGraph checkpoints require a PostgreSQL DSN")
    return normalized


@contextmanager
def postgres_checkpoint_saver(dsn: str) -> Iterator[PostgresSaver]:
    """Open a strict saver; schema setup remains an explicit deployment action."""

    validated = validate_postgres_checkpoint_dsn(dsn)
    with Connection.connect(
        validated,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
    ) as connection:
        yield PostgresSaver(
            connection,
            serde=create_checkpoint_serializer(),
        )


__all__ = [
    "create_checkpoint_serializer",
    "postgres_checkpoint_saver",
    "validate_postgres_checkpoint_dsn",
]
