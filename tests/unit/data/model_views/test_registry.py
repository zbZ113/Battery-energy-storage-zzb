import importlib.util
import json
from pathlib import Path

from quanxin_life.data.model_views.registry import ModelViewRegistry


def test_registry_loads_all_five_reviewed_views() -> None:
    registry = ModelViewRegistry.load(Path("configs/model_views"))

    assert {item.view_id for item in registry.configs} == {
        "early_life_sequence",
        "soh_trajectory",
        "field_monitoring",
        "degradation_condition",
        "partial_charge",
    }


def test_plan_marks_verified_matr_views_ready(capsys) -> None:
    script = Path("scripts/data/build_model_views.py")
    spec = importlib.util.spec_from_file_location("build_model_views", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main(["plan", "--all", "--project-root", str(Path.cwd())]) == 0

    results = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    statuses = {item["view_id"]: item["status"] for item in results}
    assert statuses["early_life_sequence"] == "READY"
    assert statuses["soh_trajectory"] == "READY"
