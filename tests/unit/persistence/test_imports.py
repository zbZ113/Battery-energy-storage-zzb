import importlib
import sys

import sqlalchemy
from pytest import MonkeyPatch


def test_importing_persistence_does_not_create_an_engine(monkeypatch: MonkeyPatch) -> None:
    calls: list[object] = []

    def unexpected_create_engine(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(sqlalchemy, "create_engine", unexpected_create_engine)
    for module_name in tuple(sys.modules):
        if module_name == "quanxin_life.persistence" or module_name.startswith(
            "quanxin_life.persistence."
        ):
            del sys.modules[module_name]

    importlib.import_module("quanxin_life.persistence")

    assert calls == []
