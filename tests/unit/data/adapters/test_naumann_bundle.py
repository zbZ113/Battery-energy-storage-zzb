from __future__ import annotations

from pathlib import Path

from quanxin_life.data.adapters.naumann_bundle import (
    build_naumann_condition_bundle,
    to_naumann_build_spec,
)
from quanxin_life.data.adapters.naumann_calendar import CalendarCapacityObservation
from quanxin_life.data.adapters.naumann_cycle_mat import CycleMatrixObservation
from quanxin_life.data.processing import DatasetProcessor, RawDatasetFile


def test_cycle_bundle_preserves_condition_level_semantics_and_units() -> None:
    observation = CycleMatrixObservation(
        condition_id="T40_SOC50_DOD80",
        observation_axis="equivalent_full_cycles",
        observation_value=100.0,
        temperature_c=40.0,
        mean_soc=0.5,
        dod=0.8,
        charge_c_rate=1.0,
        discharge_c_rate=1.0,
        metric_name="relative_capacity_ratio",
        metric_value=0.95,
        source_file="data/raw/NAUMANN_CYCLE/v1/xdod.mat",
        source_sha256="a" * 64,
        layout_version="cycle-layout-v1",
    )

    bundle = build_naumann_condition_bundle(
        dataset_version="naumann-cycle-v1",
        observations=(observation,),
    )

    assert bundle.identity_level == "condition"
    assert bundle.cell_level_split_supported is False
    assert {item.unit.value for item in bundle.values} >= {"FEC", "ratio", "degC"}
    assert {item.table_type.value for item in bundle.values} == {
        "condition_observations"
    }


def test_calendar_bundle_never_invents_cell_or_lifetime_labels() -> None:
    observation = CalendarCapacityObservation(
        condition_id="T40_SOC50",
        storage_time_h=100.0,
        temperature_c=40.0,
        mean_soc=0.5,
        capacity_ah=2.9,
        source_file="data/raw/NAUMANN_CALENDAR/v1/DischargeCapacity.xlsx",
        source_sha256="b" * 64,
        layout_version="calendar-layout-v1",
    )

    bundle = build_naumann_condition_bundle(
        dataset_version="naumann-calendar-v1",
        observations=(observation,),
    )
    payload = bundle.model_dump(mode="json")

    assert {item.unit.value for item in bundle.values} >= {"h", "Ah", "degC"}
    serialized = str(payload).lower()
    assert "cell_id" not in serialized
    assert "eol" not in serialized
    assert "rul" not in serialized


def test_condition_bundle_can_be_published_atomically(tmp_path: Path) -> None:
    import hashlib

    raw = tmp_path / "data" / "raw" / "NAUMANN_CYCLE" / "v1" / "xdod.mat"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"reviewed matrix bytes")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    observation = CycleMatrixObservation(
        condition_id="T40_SOC50_DOD80",
        observation_axis="equivalent_full_cycles",
        observation_value=100.0,
        temperature_c=40.0,
        mean_soc=0.5,
        dod=0.8,
        charge_c_rate=1.0,
        discharge_c_rate=1.0,
        metric_name="relative_capacity_ratio",
        metric_value=0.95,
        source_file="data/raw/NAUMANN_CYCLE/v1/xdod.mat",
        source_sha256=raw_sha,
        layout_version="cycle-layout-v1",
    )
    bundle = build_naumann_condition_bundle(
        dataset_version="naumann-cycle-v1", observations=(observation,)
    )
    spec = to_naumann_build_spec(
        bundle,
        raw_files=(
            RawDatasetFile(
                relative_path="data/raw/NAUMANN_CYCLE/v1/xdod.mat",
                sha256=raw_sha,
            ),
        ),
    )

    result = DatasetProcessor(tmp_path).build(spec)

    output = tmp_path / result.output_root
    assert (output / "observations" / "condition_observations.json").is_file()
