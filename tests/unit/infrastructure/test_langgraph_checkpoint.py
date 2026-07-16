import pytest


def test_checkpoint_serializer_never_enables_pickle_fallback() -> None:
    from quanxin_life.infrastructure.langgraph_checkpoint import (
        create_checkpoint_serializer,
    )

    serializer = create_checkpoint_serializer()

    assert serializer.pickle_fallback is False


def test_formal_checkpoint_configuration_rejects_non_postgres_storage() -> None:
    from quanxin_life.infrastructure.langgraph_checkpoint import (
        validate_postgres_checkpoint_dsn,
    )

    with pytest.raises(ValueError, match="PostgreSQL"):
        validate_postgres_checkpoint_dsn("sqlite:///unsafe-local-checkpoints.db")

    assert validate_postgres_checkpoint_dsn(
        "postgresql://checkpoint_user:secret@postgres:5432/quanxin"
    ).startswith("postgresql://")
