"""CPU tests for the fully identifiable hierarchical-router repair."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.test_prior_anchored_router_smoke import (  # noqa: E402
    _smoke_config,
    _write_preview,
)


def _repair_config(tmp_path):
    prior_path = _write_preview(tmp_path)
    config = _smoke_config(tmp_path, prior_path)
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    router.update({
        "initial_destination_native_probability": 0.75,
        "destination_bootstrap_probability": 0.25,
        "destination_probability_floor": 0.02,
        "destination_probability_ceiling": 0.75,
        "spatial_destination_enabled": True,
        "spatial_destination_hidden_channels": 8,
        "spatial_destination_delta_max": 3.0,
    })
    frequency = config["modules"]["residual_frequency"]
    frequency.update({
        "low_frequency_background_enabled": True,
        "low_frequency_gate_max": 0.10,
        "low_frequency_projection_init_scale": 0.005,
        "dct_descriptor": {
            "enabled": True,
            "pooled_size": 8,
            "selected_frequencies": 12,
        },
    })
    config["losses"]["route_utility_supervision"] = {
        "enabled": True,
        "weight": 0.05,
        "lesion_dilate_radius": 2,
        "spatial_weight": 1.0,
        "global_weight": 0.25,
        "spatial_tv_weight": 0.001,
        "positive_weight": 2.0,
        "active_tau_max": 0.7,
        "epoch_warmup": [2, 2],
        "epoch_window": [2, 2],
    }
    return config


def _batch():
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 13:19, 13:19] = 1.0
    return {
        "ct": torch.randn(2, 1, 32, 32),
        "pet": torch.randn(2, 1, 32, 32).clamp(-1.0, 1.0),
        "mask": mask,
        "organ_mask": torch.zeros(2, 6, 32, 32),
        "organ_distance": torch.zeros(2, 6, 32, 32),
        "mu_map": torch.zeros(2, 1, 32, 32),
        "meta": [
            {"pet_suv_max": 20.0, "suv_ok": True}
            for _ in range(2)
        ],
    }


def test_full_repair_has_bootstrap_spatial_utility_and_low_frequency(tmp_path):
    from src.model.slmf_bbdm import SLMFBBDM

    config = _repair_config(tmp_path)
    model = SLMFBBDM.from_config(config)
    router = model.residual_preconditioner
    router.set_training_epoch(0)
    model.set_training_epoch(1)

    loss, logs = model(_batch())

    assert torch.isfinite(loss)
    assert logs["frequency/route_shallow_mass"].item() > 0.0
    assert logs["loss/route_utility_supervision/enabled"].item() == 1.0
    assert logs["loss/route_utility_supervision/route_spatial_std"].item() >= 0.0
    # Correct semantic labels: L3 is the deliberate zero injection.
    assert logs["frequency/injection_l3_rms"].item() == pytest.approx(0.0)
    assert logs["frequency/effective_low_frequency_l1_rms"].item() > 0.0

    spatial_l2 = model._last_frequency_diagnostics[
        "route_spatial_conditional_shallow_l2"
    ]
    spatial_l1 = model._last_frequency_diagnostics[
        "route_spatial_conditional_shallow_l1"
    ]
    assert spatial_l2.shape == (2, 3, 8, 8)
    assert spatial_l1.shape == (2, 3, 16, 16)
    torch.testing.assert_close(
        spatial_l2,
        torch.full_like(spatial_l2, 0.25),
    )
    spatial_delta_l2 = model._last_frequency_diagnostics[
        "route_spatial_delta_l2"
    ]
    torch.testing.assert_close(
        spatial_delta_l2.mean(dim=(-2, -1)),
        torch.zeros_like(spatial_delta_l2[..., 0, 0]),
        atol=1.0e-6,
        rtol=0.0,
    )

    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in router.spatial_destination_heads.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in router.low_frequency_projection.parameters()
    )


def test_frequency_branch_interventions_are_exact_and_state_free(
    tmp_path: Path,
) -> None:
    from src.model.slmf_bbdm import SLMFBBDM

    model = SLMFBBDM.from_config(_repair_config(tmp_path))
    router = model.residual_preconditioner
    router.set_training_epoch(5)
    router.eval()

    residual = torch.randn(2, 1, 32, 32)
    ct = torch.randn_like(residual)
    timestep = torch.tensor([20, 20])
    state_before = {
        key: value.detach().clone()
        for key, value in router.state_dict().items()
    }

    outputs = {}
    diagnostics = {}
    with torch.no_grad():
        for mode in (
            "full",
            "detail_off",
            "ll_off",
            "all_frequency_off",
        ):
            router.set_inference_frequency_intervention(mode)
            outputs[mode], diagnostics[mode] = router(
                residual,
                timestep,
                model.noise_schedule,
                ct,
            )

    for level in range(4):
        torch.testing.assert_close(
            outputs["all_frequency_off"][level],
            torch.zeros_like(outputs["all_frequency_off"][level]),
        )
        torch.testing.assert_close(
            outputs["detail_off"][level] + outputs["ll_off"][level],
            outputs["full"][level],
        )

    for key in (
        "effective/native_l2_rms",
        "effective/native_l1_rms",
        "effective/shallow_l2_to_l1_rms",
        "effective/shallow_l1_to_l0_rms",
    ):
        assert diagnostics["detail_off"][key].item() == pytest.approx(0.0)
    assert (
        diagnostics["ll_off"]["effective/low_frequency_l1_rms"].item()
        == pytest.approx(0.0)
    )
    assert (
        diagnostics["detail_off"]["effective/low_frequency_l1_rms"].item()
        > 0.0
    )
    assert router.inference_frequency_intervention() == {
        "mode": "all_frequency_off",
        "detail_enabled": False,
        "low_frequency_enabled": False,
    }
    with pytest.raises(ValueError, match="Unknown frequency intervention"):
        router.set_inference_frequency_intervention("unknown")

    state_after = router.state_dict()
    assert state_after.keys() == state_before.keys()
    for key, value in state_before.items():
        torch.testing.assert_close(state_after[key], value)


def test_phase_schedule_prevents_router_projection_coadaptation(tmp_path):
    from src.data.dataset import FakeDataset
    from src.model.slmf_bbdm import SLMFBBDM
    from src.model.trainer import Trainer
    from torch.utils.data import DataLoader

    config = _repair_config(tmp_path)
    config["training"]["num_epochs"] = 3
    config["training"]["spectral_training_phases"] = {
        "enabled": True,
        "phases": [
            {
                "name": "bootstrap",
                "start_epoch": 1,
                "end_epoch": 1,
                "train_groups": ["base", "projection"],
            },
            {
                "name": "identify",
                "start_epoch": 2,
                "end_epoch": 2,
                "train_groups": ["router", "descriptor"],
            },
            {
                "name": "stabilize",
                "start_epoch": 3,
                "end_epoch": 3,
                "train_groups": ["base", "projection"],
            },
        ],
    }
    model = SLMFBBDM.from_config(config)
    loader = DataLoader(FakeDataset(2, image_size=32), batch_size=2)
    trainer = Trainer(model, config, loader, loader, device="cpu")

    projection = next(
        model.residual_preconditioner.l2_to_l1_projection.parameters()
    )
    destination = next(
        model.residual_preconditioner.prior_destination_heads.parameters()
    )
    spatial = next(
        model.residual_preconditioner.spatial_destination_heads.parameters()
    )
    descriptor = next(
        model.residual_preconditioner.dct_descriptor.parameters()
    )

    trainer.epoch_count = 0
    trainer._apply_spectral_training_phase()
    assert projection.requires_grad
    assert not destination.requires_grad
    assert not spatial.requires_grad
    assert not descriptor.requires_grad

    trainer.epoch_count = 1
    trainer._apply_spectral_training_phase()
    assert not projection.requires_grad
    assert destination.requires_grad
    assert spatial.requires_grad
    assert descriptor.requires_grad

    trainer.epoch_count = 2
    trainer._apply_spectral_training_phase()
    assert projection.requires_grad
    assert not destination.requires_grad
    assert not spatial.requires_grad
    assert not descriptor.requires_grad

    model.set_training_epoch(0)
    assert model._loss_epoch_scale("route_utility_supervision") == 0.0
    model.set_training_epoch(1)
    assert model._loss_epoch_scale("route_utility_supervision") == 1.0
    model.set_training_epoch(2)
    assert model._loss_epoch_scale("route_utility_supervision") == 0.0


def test_conditional_mean_can_be_recovered_from_full_model_checkpoint(
    tmp_path: Path,
):
    from src.model.slmf_bbdm import SLMFBBDM

    source_config = _repair_config(tmp_path)
    source = SLMFBBDM.from_config(source_config)
    expected = {
        key: value.detach().clone()
        for key, value in source.mean_predictor.state_dict().items()
    }
    checkpoint_path = tmp_path / "full_model.pt"
    torch.save(
        {
            "model": {
                f"mean_predictor.{key}": value
                for key, value in expected.items()
            },
        },
        checkpoint_path,
    )

    restored_config = _repair_config(tmp_path)
    restored_config["modules"]["conditional_mean"].update({
        "checkpoint": str(checkpoint_path),
        "checkpoint_format": "full_model",
        "freeze": True,
    })
    restored = SLMFBBDM.from_config(restored_config)

    assert not any(
        parameter.requires_grad
        for parameter in restored.mean_predictor.parameters()
    )
    for key, value in restored.mean_predictor.state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_route_utility_is_safe_under_bfloat16_autocast(tmp_path: Path):
    from src.model.slmf_bbdm import SLMFBBDM

    config = _repair_config(tmp_path)
    model = SLMFBBDM.from_config(config)
    model.set_training_epoch(1)
    model.residual_preconditioner.set_training_epoch(0)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, logs = model(_batch())

    assert torch.isfinite(loss)
    assert logs["loss/route_utility_supervision/epoch_scale"].item() == 1.0
