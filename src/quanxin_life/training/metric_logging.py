"""Strict metric registry and append-only JSONL persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from quanxin_life.evaluation.metric_contracts import MetricDefinition, MetricRecord


class MetricRegistry:
    def __init__(self, definitions: Iterable[MetricDefinition] = ()) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        existing = self._definitions.get(definition.name)
        if existing is not None and existing != definition:
            raise ValueError("metric name cannot be registered with conflicting definition")
        self._definitions[definition.name] = definition

    def definition(self, name: str) -> MetricDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ValueError(f"metric is not registered: {name}") from exc

    def validate(self, record: MetricRecord) -> MetricRecord:
        definition = self.definition(record.metric_name)
        if definition.unit != record.unit or definition.stage != record.split:
            raise ValueError("metric record does not match its definition")
        return record


def append_metric_record(path: Path, record: MetricRecord, registry: MetricRegistry) -> None:
    validated = registry.validate(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(validated.model_dump(mode="json"), sort_keys=True) + "\n")


__all__ = ["MetricRegistry", "append_metric_record"]
