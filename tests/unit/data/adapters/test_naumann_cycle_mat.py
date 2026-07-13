import hashlib
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from quanxin_life.data.adapters.naumann_cycle_mat import (
    CycleConditionColumn,
    NaumannCycleMatrixLayout,
    load_naumann_cycle_matrix,
)
from quanxin_life.data.manifest import RawFileManifest
from quanxin_life.data.source_catalog import IngestionMode, SourceCatalogEntry


def _write_cycle_matrix(path: Path, *, nonmonotonic: bool = False) -> None:
    x_axis = np.asarray([[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]], dtype=float)
    if nonmonotonic:
        x_axis[2, 0] = 40.0
    savemat(
        path,
        {
            "X_Axis_Data_Mat": x_axis,
            "Y_Axis_Data_Mat": np.asarray(
                [[3.0, 3.0], [2.95, 2.90], [2.90, 2.80]], dtype=float
            ),
            "Legend_Vec": np.asarray(["T25_SOC50", "T40_SOC75"], dtype=object),
        },
        do_compression=False,
    )


def _source(**updates: object) -> SourceCatalogEntry:
    values: dict[str, object] = {
        "dataset_id": "NAUMANN_CYCLE",
        "version": "Mendeley-v1",
        "source_uri": "https://data.mendeley.com/datasets/6hgyr25h8d/1",
        "paper_uri": "https://doi.org/10.1016/j.jpowsour.2019.227666",
        "license_status": "CC BY 4.0",
        "ingestion_mode": IngestionMode.MATLAB,
        "expected_suffixes": (".mat",),
    }
    values.update(updates)
    return SourceCatalogEntry.model_validate(values)


def _manifest(path: Path, **updates: object) -> RawFileManifest:
    values: dict[str, object] = {
        "dataset_id": "NAUMANN_CYCLE",
        "relative_path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_uri": "https://data.mendeley.com/datasets/6hgyr25h8d/1",
        "license_name": "CC BY 4.0",
        "paper_doi": "10.1016/j.jpowsour.2019.227666",
    }
    values.update(updates)
    return RawFileManifest.model_validate(values)


def _layout() -> NaumannCycleMatrixLayout:
    return NaumannCycleMatrixLayout(
        layout_version="naumann-cycle-fixture-v1",
        x_axis_variable="X_Axis_Data_Mat",
        y_axis_variable="Y_Axis_Data_Mat",
        legend_variable="Legend_Vec",
        observation_axis="equivalent_full_cycles",
        metric_name="capacity_ah",
        condition_columns=(
            CycleConditionColumn(
                column_index=0,
                expected_legend="T25_SOC50",
                condition_id="T25_SOC50",
                temperature_c=25.0,
                mean_soc=0.50,
                dod=0.80,
                charge_c_rate=1.0,
                discharge_c_rate=1.0,
            ),
            CycleConditionColumn(
                column_index=1,
                expected_legend="T40_SOC75",
                condition_id="T40_SOC75",
                temperature_c=40.0,
                mean_soc=0.75,
                dod=0.80,
                charge_c_rate=1.0,
                discharge_c_rate=1.0,
            ),
        ),
    )


def test_loads_reviewed_matlab_cycle_matrix_as_condition_level_observations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capacity_by_condition.mat"
    _write_cycle_matrix(path)

    observations = load_naumann_cycle_matrix(path, _manifest(path), _source(), layout=_layout())

    assert len(observations) == 6
    first = observations[0]
    assert first.dataset_id == "NAUMANN_CYCLE"
    assert first.condition_id == "T25_SOC50"
    assert first.observation_axis == "equivalent_full_cycles"
    assert first.observation_value == 0.0
    assert first.metric_name == "capacity_ah"
    assert first.metric_value == 3.0
    assert first.source_sha256 == _manifest(path).sha256
    assert first.layout_version == "naumann-cycle-fixture-v1"


def test_rejects_legend_mismatch_without_guessing_condition_semantics(tmp_path: Path) -> None:
    path = tmp_path / "capacity_by_condition.mat"
    _write_cycle_matrix(path)
    layout = _layout().model_copy(
        update={
            "condition_columns": (
                _layout().condition_columns[0].model_copy(update={"expected_legend": "unreviewed"}),
                _layout().condition_columns[1],
            )
        }
    )

    with pytest.raises(ValueError, match="expected_legend"):
        load_naumann_cycle_matrix(path, _manifest(path), _source(), layout=layout)


def test_rejects_nonmonotonic_axis_without_reordering_matrix_rows(tmp_path: Path) -> None:
    path = tmp_path / "capacity_by_condition.mat"
    _write_cycle_matrix(path, nonmonotonic=True)

    with pytest.raises(ValueError, match="observation axis must strictly increase"):
        load_naumann_cycle_matrix(path, _manifest(path), _source(), layout=_layout())


def test_rejects_catalog_mismatch_before_loading_matlab_payload(tmp_path: Path) -> None:
    path = tmp_path / "capacity_by_condition.mat"
    _write_cycle_matrix(path)

    with pytest.raises(ValueError, match="source URI"):
        load_naumann_cycle_matrix(
            path,
            _manifest(path),
            _source(source_uri="https://example.invalid/cycle"),
            layout=_layout(),
        )
