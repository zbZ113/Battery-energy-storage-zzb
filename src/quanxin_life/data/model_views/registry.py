"""Versioned registry for the five reviewed model-view families."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ConfigDict, Field, model_validator

from quanxin_life.core.schemas import ContractModel
from quanxin_life.data.model_views.schemas import ModelViewConfig


class ModelViewRegistry(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configs: tuple[ModelViewConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def view_ids_are_unique(self) -> ModelViewRegistry:
        ids = [item.view_id for item in self.configs]
        if len(ids) != len(set(ids)):
            raise ValueError("model view ids must be unique")
        return self

    @classmethod
    def load(cls, directory: Path) -> ModelViewRegistry:
        root = Path(directory)
        configs = tuple(
            ModelViewConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(root.glob("*.json"))
        )
        return cls(configs=configs)

    def require(self, view_id: str) -> ModelViewConfig:
        for config in self.configs:
            if config.view_id == view_id:
                return config
        raise KeyError(f"model view is not registered: {view_id}")


__all__ = ["ModelViewRegistry"]
