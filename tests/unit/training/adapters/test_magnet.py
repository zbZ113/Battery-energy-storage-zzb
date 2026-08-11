from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from quanxin_life.core import TrainingReadableSplit, TrainingTaskType
from quanxin_life.data.model_views.schemas import ModelViewArtifact, ModelViewManifest
from quanxin_life.training.adapters.base import TrainingAdapter
from quanxin_life.training.adapters.magnet import (
    MAGNET_LICENSE_STATUS,
    MAGNET_UPSTREAM_COMMIT,
    MAGNetAdapter,
    MAGNetModel,
    MAGNetTensorDataset,
    compute_magnet_awmse,
    compute_magnet_losses,
    magnet_qd_ed_metrics,
)


def _tiny_manifest(root: Path) -> ModelViewManifest:
    rows = 8
    tensor_path = root / "magnet_tensors.safetensors"
    generator = torch.Generator().manual_seed(71)
    history = torch.rand((rows, 4, 2), generator=generator)
    decoder_known = history[:, -2:, :].clone()
    targets = torch.rand((rows, 3, 2), generator=generator)
    save_file(
        {
            "history": history,
            "history_cycle": torch.arange(4).reshape(1, 4, 1).repeat(rows, 1, 1).float(),
            "decoder_known": decoder_known,
            "targets": targets,
            "target_cycle": torch.arange(2, 7).reshape(1, 5, 1).repeat(rows, 1, 1).float(),
            "observation_mask": torch.ones((rows, 3, 2), dtype=torch.bool),
            "cycle_distance": torch.linspace(5, 12, rows).reshape(rows, 1),
            "condition_vectors": torch.tensor(
                [[25.0, 1.0, 1.0, 1.0, 0.5]] * 2
                + [[35.0, 1.0, 1.0, 0.8, 0.5]] * 2
                + [[45.0, 0.5, 1.0, 1.0, 0.8]] * 2
                + [[15.0, 1.0, 0.5, 0.6, 0.2]] * 2
            ),
        },
        str(tensor_path),
    )
    metadata_path = root / "magnet_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "entity_ids": [f"cell-{index}" for index in range(rows)],
                "condition_keys": ["train-a"] * 2
                + ["train-b"] * 2
                + ["validation-c"] * 2
                + ["test-d"] * 2,
                "splits": ["train"] * 4 + ["validation"] * 2 + ["test"] * 2,
                "seen": [True] * 6 + [False] * 2,
                "condition_feature_names": [
                    "temperature_c",
                    "charge_rate_c",
                    "discharge_rate_c",
                    "dod_fraction",
                    "soc_fraction",
                ],
                "protocols": ["CCCV"] * rows,
                "monotonic_applicable": [True] * rows,
            }
        ),
        encoding="utf-8",
    )
    artifacts = tuple(
        sorted(
            (
                ModelViewArtifact(
                    relative_path=path.name,
                    size_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in (tensor_path, metadata_path)
            ),
            key=lambda item: item.relative_path,
        )
    )
    return ModelViewManifest(
        schema_version="model-view-manifest-v2",
        view_id="magnet_multicondition_qd_ed",
        view_version="magnet-condition-v1",
        task_type=TrainingTaskType.CONDITION_DEGRADATION,
        target_semantics="Qd_Ed_trajectory",
        mask_semantics=("observation_mask",),
        canonical_sha256="0" * 64,
        split_sha256="1" * 64,
        builder_version="test",
        builder_code_sha256="2" * 64,
        config_sha256="3" * 64,
        normalization_sha256="4" * 64,
        training_entity_ids_sha256="5" * 64,
        row_count=rows,
        entity_key="cell_id",
        source_dataset_ids=("NAUMANN_CYCLE", "LFP_280AH_DOD", "LFP_280AH_TEMPEST"),
        artifacts=artifacts,
    )


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_family="magnet",
        seq_len=4,
        label_len=2,
        prediction_horizon=3,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_layers=1,
        d_ff=4,
        factor=1,
        factor2=1,
        dropout=0.0,
        micro_batch_size=4,
        gradient_accumulation_steps=1,
        seed=38,
    )


def test_magnet_binds_reviewed_upstream_and_data_gate() -> None:
    adapter = MAGNetAdapter()

    assert MAGNET_UPSTREAM_COMMIT == "aafb90c551d20748251a35fd51a34eae2539aaca"
    assert MAGNET_LICENSE_STATUS == "VERIFIED_LICENSE_PRESENT"
    assert isinstance(adapter, TrainingAdapter)
    with pytest.raises(ValueError, match="committed multi-condition"):
        adapter.readiness(has_multi_condition_view=False)
    assert adapter.readiness(has_multi_condition_view=True) is None


def test_magnet_pytorch_212_forward_is_deterministic() -> None:
    torch.manual_seed(17)
    model = MAGNetModel(
        seq_len=2,
        label_len=1,
        pred_len=1,
        d_model=4,
        n_heads=1,
        e_layers=1,
        d_layers=1,
        d_ff=4,
        factor=1,
        factor2=1,
    )
    for parameter in model.parameters():
        torch.nn.init.constant_(parameter, 0.05)
    history = torch.tensor([[[0.1, 0.2], [0.2, 0.3]]])
    history_cycle = torch.tensor([[[0.0], [1.0]]])
    decoder = torch.tensor([[[0.2, 0.3], [0.0, 0.0]]])
    target_cycle = torch.tensor([[[1.0], [2.0]]])

    torch.manual_seed(91)
    first = model(history, history_cycle, decoder, target_cycle)[0]
    torch.manual_seed(91)
    second = model(history, history_cycle, decoder, target_cycle)[0]

    assert first.shape == (1, 1, 2)
    assert torch.allclose(first, second)


def test_magnet_uses_reviewed_upstream_informer_forward() -> None:
    config = _tiny_config()
    torch.manual_seed(17)
    model = MAGNetModel.from_training_config(config)
    model.eval()
    history = torch.rand((2, 4, 2))
    history_cycle = torch.arange(4).reshape(1, 4, 1).repeat(2, 1, 1).float()
    decoder = torch.rand((2, 5, 2))
    target_cycle = torch.arange(2, 7).reshape(1, 5, 1).repeat(2, 1, 1).float()

    torch.manual_seed(123)
    adapted = model(history, history_cycle, decoder, target_cycle)
    torch.manual_seed(123)
    upstream = model.upstream_model(history, history_cycle, decoder, target_cycle)

    assert torch.allclose(adapted[0], upstream[0])
    assert torch.allclose(adapted[1], upstream[1])
    assert torch.allclose(adapted[2], upstream[2])


def test_magnet_tensor_view_and_meta_lifecycle_are_real(tmp_path: Path) -> None:
    manifest = _tiny_manifest(tmp_path)
    config = _tiny_config()
    adapter = MAGNetAdapter(view_root=tmp_path)
    dataset = adapter.load_view(manifest)
    assert isinstance(dataset, MAGNetTensorDataset)
    model = adapter.build_model(config)
    task = adapter.attach_training(config, device=torch.device("cpu"), seed=38)
    task.micro_batch_size = 3
    batches = task._condition_batches([0, 1, 2, 3], epoch=1)
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]
    assert all(len({dataset[index]["condition_key"] for index in batch}) >= 2 for batch in batches)
    task.micro_batch_size = 4

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    epoch = task.train_epoch(1)
    assert epoch.metrics["support_loss"] >= 0
    assert epoch.metrics["query_loss"] >= 0
    assert epoch.metrics["meta_outer_loss"] == pytest.approx(epoch.loss)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())

    validation = task.validate(1, split=TrainingReadableSplit.VALIDATION)
    assert validation.metrics["validation_observed_mae"] >= 0
    assert validation.metrics["qd_mae"] >= 0
    assert validation.metrics["ed_rmse"] >= 0
    assert validation.metrics["seen_condition_observed_mae"] >= 0
    test_metrics = task.validate(1, split=TrainingReadableSplit.TEST)
    assert test_metrics.metrics["unseen_condition_observed_mae"] >= 0
    predictions = task.predict(split=TrainingReadableSplit.TEST)
    assert len(predictions.entity_ids) == 12
    assert all("|Q" in entity_id or "|E" in entity_id for entity_id in predictions.entity_ids)
    records = task.predict_records(split=TrainingReadableSplit.TEST)
    assert {record["target_name"] for record in records} == {"Qd", "Ed"}
    assert all("condition_distance" in record and "ood_status" in record for record in records)
    reconciled, reconciled_records = task.evaluate_with_records(
        split=TrainingReadableSplit.TEST
    )
    observed_errors = torch.tensor(
        [float(record["abs_error"]) for record in reconciled_records]
    )
    assert reconciled.metrics["validation_observed_mae"] == pytest.approx(
        float(observed_errors.mean()), abs=1e-8
    )
    first_records = task.evaluate_with_records(split=TrainingReadableSplit.TEST)[1]
    second_records = task.evaluate_with_records(split=TrainingReadableSplit.TEST)[1]
    assert first_records == second_records

    saved = adapter.save(tmp_path / "checkpoint")
    reference = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    adapter.restore(saved)
    assert all(torch.equal(reference[name], value) for name, value in model.state_dict().items())
    exported = adapter.export_best(tmp_path / "export")
    assert exported.files[0].relative_path == "targets/magnet_model.json"
    assert (tmp_path / "export" / "targets" / "magnet_model.safetensors").is_file()


def test_magnet_physics_and_meta_losses_follow_reviewed_equations() -> None:
    predictions = torch.tensor([[[1.0, 3.0], [0.5, 2.5]]], requires_grad=True)
    targets = torch.tensor([[[0.9, 3.1], [0.4, 2.4]]])
    soc_markers = torch.tensor([[[0.0], [50.0]]])
    mask = torch.ones_like(predictions, dtype=torch.bool)

    losses = compute_magnet_losses(
        predictions=predictions,
        targets=targets,
        observation_mask=mask,
        soc_markers=soc_markers,
        cutoff_voltage=2.5,
        meta_train_loss=torch.tensor(0.2),
        meta_test_loss=torch.tensor(0.3),
        meta_beta=2.0,
        proportion_weight=0.5,
        voltage_weight=0.25,
    )
    losses["total_loss"].backward()

    assert set(losses) == {
        "raw_mse",
        "soc_proportion_loss",
        "cutoff_voltage_loss",
        "meta_train_loss",
        "meta_test_loss",
        "total_loss",
    }
    assert torch.allclose(
        losses["total_loss"],
        losses["meta_train_loss"]
        + 2.0 * losses["meta_test_loss"]
        + 0.5 * losses["soc_proportion_loss"]
        + 0.25 * losses["cutoff_voltage_loss"],
    )
    assert torch.allclose(losses["raw_mse"], torch.tensor(0.01))
    assert torch.allclose(losses["soc_proportion_loss"], torch.zeros(()))
    assert torch.allclose(losses["cutoff_voltage_loss"], torch.zeros(()))
    assert torch.allclose(losses["total_loss"], torch.tensor(0.8))

    metrics = magnet_qd_ed_metrics(predictions.detach(), targets, mask)
    assert set(metrics) == {"observed_mae", "qd_mae", "ed_mae"}
    assert torch.allclose(metrics["observed_mae"], torch.tensor(0.1))
    assert torch.allclose(metrics["qd_mae"], torch.tensor(0.1))
    assert torch.allclose(metrics["ed_mae"], torch.tensor(0.1))


def test_magnet_awmse_matches_upstream_condition_weighting_and_tddg() -> None:
    predictions = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[10.0, 12.0]]], requires_grad=True)
    targets = torch.zeros_like(predictions)
    losses = compute_magnet_awmse(
        predictions=predictions,
        targets=targets,
        observation_mask=torch.ones_like(predictions, dtype=torch.bool),
        condition_keys=("condition-a", "condition-a", "condition-b"),
        cycle_distance_predictions=torch.tensor([[1.0], [2.0], [3.0]]),
        cycle_distance_targets=torch.zeros((3, 1)),
        auxiliary_gamma=0.2,
    )
    condition_a = torch.tensor((1.0 + 4.0 + 9.0 + 16.0) / 4.0)
    condition_b = torch.tensor((100.0 + 144.0) / 2.0)
    expected_forecast = (condition_a + condition_b) / 2.0
    expected_cycle = torch.tensor((1.0 + 4.0 + 9.0) / 3.0)

    assert torch.allclose(losses["forecast_loss"], expected_forecast)
    assert torch.allclose(losses["cycle_distance_loss"], expected_cycle)
    assert torch.allclose(losses["total_loss"], expected_forecast + 0.2 * expected_cycle)


def test_magnet_missing_observations_fail_closed() -> None:
    predictions = torch.zeros((1, 2, 2))
    targets = torch.zeros_like(predictions)
    missing = torch.zeros_like(predictions, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one observed"):
        compute_magnet_losses(
            predictions=predictions,
            targets=targets,
            observation_mask=missing,
            soc_markers=torch.zeros((1, 2, 1)),
            cutoff_voltage=2.5,
            meta_train_loss=torch.zeros(()),
            meta_test_loss=torch.zeros(()),
            meta_beta=2.0,
            proportion_weight=0.5,
            voltage_weight=0.25,
        )
    with pytest.raises(ValueError, match="at least one observed"):
        magnet_qd_ed_metrics(predictions, targets, missing)


@pytest.mark.parametrize("stage", ["smoke", "selection", "final"])
def test_magnet_configs_freeze_compatibility_and_data_gate(stage: str) -> None:
    root = Path(__file__).parents[4]
    payload = json.loads(
        (root / "configs" / "training" / "magnet" / f"{stage}.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "magnet-training-v2"
    assert payload["enabled"] is True
    assert payload["blocked_reasons"] == []
    assert payload["upstream_commit"] == MAGNET_UPSTREAM_COMMIT
    assert payload["runtime_dependencies"] == ["numpy", "torch"]
    assert payload["compatibility_changes"] == "compatibility_changes.json"
    assert payload["upstream_provenance"] == "upstream_provenance.json"
    assert payload["model"]["family"] == "Informer"
    assert payload["model"]["output_channels"] == ["Qd", "Ed"]
    assert payload["optimization"]["micro_batch_size"] == 128


def test_magnet_training_matrix_matches_frozen_adapter_config() -> None:
    root = Path(__file__).parents[4]
    entries = json.loads(
        (root / "configs" / "training" / "task_matrix_v1.json").read_text(encoding="utf-8")
    )["entries"]
    entry = next(item for item in entries if item["model_family"] == "magnet")
    assert entry["loss_names"]
    assert entry["selection_metric_name"] == "validation_observed_mae"
    assert entry["source_commit"] == MAGNET_UPSTREAM_COMMIT


@pytest.mark.parametrize("field", ["meta_beta", "proportion_weight", "voltage_weight"])
def test_magnet_rejects_negative_loss_weights(field: str) -> None:
    kwargs = {
        "predictions": torch.ones((1, 2, 2)),
        "targets": torch.ones((1, 2, 2)),
        "observation_mask": torch.ones((1, 2, 2), dtype=torch.bool),
        "soc_markers": torch.ones((1, 2, 1)),
        "cutoff_voltage": 2.5,
        "meta_train_loss": torch.tensor(1.0),
        "meta_test_loss": torch.tensor(1.0),
        "meta_beta": 1.0,
        "proportion_weight": 1.0,
        "voltage_weight": 1.0,
    }
    kwargs[field] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        compute_magnet_losses(**kwargs)


@pytest.mark.parametrize("bad_loss", [torch.tensor(float("nan")), torch.ones(2)])
def test_magnet_rejects_nonfinite_or_nonscalar_meta_losses(bad_loss: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="meta losses"):
        compute_magnet_losses(
            predictions=torch.ones((1, 2, 2)),
            targets=torch.ones((1, 2, 2)),
            observation_mask=torch.ones((1, 2, 2), dtype=torch.bool),
            soc_markers=torch.ones((1, 2, 1)),
            cutoff_voltage=2.5,
            meta_train_loss=bad_loss,
            meta_test_loss=torch.tensor(1.0),
            meta_beta=1.0,
            proportion_weight=1.0,
            voltage_weight=1.0,
        )


def test_magnet_metrics_only_emit_observed_channels() -> None:
    mask = torch.ones((1, 2, 2), dtype=torch.bool)
    mask[..., 1] = False
    predictions = torch.tensor([[[1.1, 9.0], [0.8, 9.0]]])
    targets = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    metrics = magnet_qd_ed_metrics(predictions, targets, mask)

    assert set(metrics) == {"observed_mae", "qd_mae"}
    assert torch.allclose(metrics["observed_mae"], torch.tensor(0.15))
    assert torch.allclose(metrics["qd_mae"], torch.tensor(0.15))


def test_magnet_metrics_reject_rows_without_any_observed_target() -> None:
    missing = torch.zeros((1, 2, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one observed"):
        magnet_qd_ed_metrics(torch.ones((1, 2, 2)), torch.ones((1, 2, 2)), missing)
