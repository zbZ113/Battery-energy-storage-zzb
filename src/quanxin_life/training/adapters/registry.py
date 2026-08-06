"""Closed registry for audited training adapters."""

from __future__ import annotations

import ast
import inspect
import textwrap

from quanxin_life.core import SelectionMetricDirection
from quanxin_life.training.adapters.base import TrainingAdapter

_UNSAFE_CALLS = frozenset(
    {
        "joblib.load",
        "pickle.load",
        "pickle.loads",
        "torch.load",
    }
)


class TrainingAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, TrainingAdapter] = {}

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def register(self, model_family: str, adapter: object) -> None:
        if not model_family:
            raise ValueError("model_family must not be empty")
        if model_family in self._adapters:
            raise ValueError(f"adapter family {model_family!r} is already registered")
        _assert_safe_adapter_source(adapter)
        if not isinstance(adapter, TrainingAdapter):
            raise ValueError("adapter does not implement the TrainingAdapter protocol")
        if not adapter.adapter_version:
            raise ValueError("adapter_version must not be empty")
        if not adapter.selection_metric_name:
            raise ValueError("selection_metric_name must not be empty")
        if not isinstance(adapter.selection_metric_direction, SelectionMetricDirection):
            raise ValueError("selection_metric_direction must use the public core enum")
        self._adapters[model_family] = adapter

    def get(self, model_family: str) -> TrainingAdapter:
        try:
            return self._adapters[model_family]
        except KeyError as exc:
            raise KeyError(f"no training adapter registered for {model_family!r}") from exc


def build_governed_training_adapter_registry() -> TrainingAdapterRegistry:
    """Build the closed registry for reviewed external training families."""

    from quanxin_life.training.adapters.batterymformer import BatteryMFormerAdapter
    from quanxin_life.training.adapters.battgp import BattGPAdapter
    from quanxin_life.training.adapters.blast import BLASTAdapter
    from quanxin_life.training.adapters.diting import DITINGAdapter
    from quanxin_life.training.adapters.magnet import MAGNetAdapter
    from quanxin_life.training.adapters.pbt import PBTAdapter
    from quanxin_life.training.adapters.smart_feature import SmartFeatureAdapter

    registry = TrainingAdapterRegistry()
    registry.register("battgp", BattGPAdapter())
    registry.register("batterymformer", BatteryMFormerAdapter())
    registry.register("blast_lite", BLASTAdapter())
    registry.register("diting_cptransformer", DITINGAdapter())
    registry.register("magnet", MAGNetAdapter())
    registry.register("pbt", PBTAdapter())
    registry.register("smart_feature", SmartFeatureAdapter())
    return registry


def _assert_safe_adapter_source(adapter: object) -> None:
    try:
        source = inspect.getsource(type(adapter))
    except (OSError, TypeError) as exc:
        raise ValueError("adapter source must be inspectable for safety review") from exc
    tree = ast.parse(textwrap.dedent(source))
    aliases = _import_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func, aliases)
        if name in _UNSAFE_CALLS:
            raise ValueError(f"adapter source calls forbidden deserializer {name}")


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in {"joblib", "pickle", "torch"}:
                    aliases[imported.asname or imported.name] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module in {
            "joblib",
            "pickle",
            "torch",
        }:
            for imported in node.names:
                if imported.name in {"load", "loads"}:
                    aliases[imported.asname or imported.name] = (
                        f"{node.module}.{imported.name}"
                    )
    return aliases


def _call_name(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
        return None
    root = aliases.get(node.value.id, node.value.id)
    return f"{root}.{node.attr}"


__all__ = ["TrainingAdapterRegistry", "build_governed_training_adapter_registry"]
