"""Focused contracts for the conditional-mean residual-frequency BBDM."""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _residual_config(*, frequency=True, gabor=True, mean_weight=1.0):
    return {
        "experiment": {"name": "residual_test", "seed": 42},
        "data": {"image_size": 32, "mode": "png"},
        "runtime": {"eval_sampling_steps": 2},
        "model": {
            "objective": "pred_x0",
            "sample_scheduler": "ddim",
            "enable_heteroscedastic": False,
            "self_conditioning": {"enabled": False},
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 1.0,
                "gradient_weight": 0.0,
                "min_snr_enabled": False,
            },
        },
        "modules": {
            "gabor": {
                "enabled": gabor,
                "scales": 2,
                "orientations": 4,
                "kernel_size": 9,
                "inject_adapter": False,
                "use_for_noise": False,
                "use_for_hotspot": False,
                "use_for_loss": False,
            },
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "bbdm_bridge": {
                "enabled": True,
                "name": "bbdm_bridge",
                "num_train_timesteps": 100,
            },
            "conditional_mean": {
                "enabled": True,
                "levels": 2,
                "base_channels": 8,
                "loss_weight": mean_weight,
                "detach_bridge": True,
            },
            "residual_bridge": {"enabled": True},
            "residual_frequency": {
                "enabled": frequency,
                "use_gabor_gate": frequency and gabor,
                "gabor_orientations": 4,
                "band_scales": [1.0, 0.5, 0.25],
                "gate_strength": 0.1,
            },
        },
        "losses": {},
    }


def _model_batch(batch_size=1, size=32):
    return {
        "ct": torch.randn(batch_size, 1, size, size),
        "pet": torch.randn(batch_size, 1, size, size),
        "mask": torch.zeros(batch_size, 1, size, size),
        "organ_mask": torch.zeros(batch_size, 6, size, size),
        "organ_distance": torch.zeros(batch_size, 6, size, size),
        "mu_map": torch.zeros(batch_size, 1, size, size),
    }


class TestHaarTransform:
    def test_two_level_round_trip_is_exact(self):
        from src.model.frequency.haar import haar_dwt2, haar_idwt2

        torch.manual_seed(0)
        x = torch.randn(2, 3, 32, 40)
        ll1, detail1 = haar_dwt2(x)
        ll2, detail2 = haar_dwt2(ll1)
        restored_ll1 = haar_idwt2(ll2, detail2)
        restored = haar_idwt2(restored_ll1, detail1)

        assert restored.shape == x.shape
        assert torch.allclose(restored, x, atol=1e-6, rtol=1e-6)

    def test_lowpass_reconstruction_has_zero_detail_coefficients(self):
        from src.model.frequency.haar import haar_dwt2, haar_idwt2, reconstruct_lowpass

        x = torch.randn(1, 1, 32, 32)
        ll1, _ = haar_dwt2(x)
        ll2, _ = haar_dwt2(ll1)
        lowpass = reconstruct_lowpass(ll2, levels=2)
        rebuilt_ll1, rebuilt_detail1 = haar_dwt2(lowpass)
        rebuilt_ll2, rebuilt_detail2 = haar_dwt2(rebuilt_ll1)

        assert torch.allclose(rebuilt_ll2, ll2, atol=1e-6, rtol=1e-6)
        for detail in (*rebuilt_detail1, *rebuilt_detail2):
            assert torch.count_nonzero(detail).item() == 0

    def test_odd_spatial_shape_is_rejected(self):
        from src.model.frequency.haar import haar_dwt2

        with pytest.raises(ValueError, match="even"):
            haar_dwt2(torch.randn(1, 1, 31, 32))


class TestLowFrequencyPETPredictor:
    def test_predictor_can_only_emit_level_two_lowpass(self):
        from src.model.frequency.haar import haar_dwt2, reconstruct_lowpass
        from src.model.mean_predictor import LowFrequencyPETPredictor

        predictor = LowFrequencyPETPredictor(in_channels=1, base_channels=8, levels=2)
        ct = torch.randn(2, 1, 32, 32)
        output = predictor(ct)

        assert output["ll2"].shape == (2, 1, 8, 8)
        assert output["mean_pet"].shape == ct.shape
        assert torch.allclose(
            output["mean_pet"], reconstruct_lowpass(output["ll2"], levels=2)
        )

        ll1, detail1 = haar_dwt2(output["mean_pet"])
        _, detail2 = haar_dwt2(ll1)
        assert sum(t.abs().max().item() for t in (*detail1, *detail2)) < 1e-6

    def test_lowpass_supervision_has_finite_gradients(self):
        from src.model.frequency.haar import haar_dwt2
        from src.model.mean_predictor import LowFrequencyPETPredictor

        predictor = LowFrequencyPETPredictor(in_channels=1, base_channels=8, levels=2)
        ct = torch.randn(2, 1, 32, 32)
        pet = torch.randn_like(ct)
        target_ll1, _ = haar_dwt2(pet)
        target_ll2, _ = haar_dwt2(target_ll1)
        output = predictor(ct)
        loss = torch.sqrt((output["ll2"] - target_ll2).square() + 1e-6).mean()
        loss.backward()

        grads = [p.grad for p in predictor.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


class TestComplexGaborDescriptor:
    def test_cartesian_bank_and_descriptor_shapes(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(scales=2, orientations=4, kernel_size=9)
        batch = {"ct": torch.randn(2, 1, 24, 24)}
        bundle = prior(batch, torch.zeros(2, dtype=torch.long))

        assert prior.filters == 8
        assert bundle.maps["gabor_feat"].shape == (2, 8, 24, 24)
        assert bundle.maps["gabor_orientation"].shape == (2, 4, 24, 24)
        assert bundle.maps["gabor_energy"].shape == (2, 1, 24, 24)
        assert bundle.maps["gabor_anisotropy"].shape == (2, 1, 24, 24)
        assert torch.isfinite(bundle.maps["gabor_feat"]).all()
        assert bundle.maps["gabor_anisotropy"].min() >= 0
        assert bundle.maps["gabor_anisotropy"].max() <= 1 + 1e-6

    def test_quadrature_amplitude_is_invariant_to_phase_rotation(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(scales=1, orientations=2, kernel_size=9)
        image = torch.randn(1, 1, 24, 24)
        real, imag = prior._build_quadrature_kernels(image.dtype, image.device)
        response_real = torch.nn.functional.conv2d(image, real, padding=4)
        response_imag = torch.nn.functional.conv2d(image, imag, padding=4)
        amplitude = torch.sqrt(response_real.square() + response_imag.square() + 1e-8)

        angle = torch.tensor(0.73)
        rotated_real = response_real * angle.cos() - response_imag * angle.sin()
        rotated_imag = response_real * angle.sin() + response_imag * angle.cos()
        rotated_amplitude = torch.sqrt(rotated_real.square() + rotated_imag.square() + 1e-8)
        assert torch.allclose(amplitude, rotated_amplitude, atol=1e-6, rtol=1e-5)

    def test_parameters_remain_bounded_under_extreme_raw_values(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(scales=2, orientations=4, kernel_size=9)
        with torch.no_grad():
            prior.log_frequency.fill_(100.0)
            prior.theta_raw.fill_(-100.0)
            prior.log_sigma.fill_(100.0)
            prior.gamma_raw.fill_(-100.0)
        params = prior._bounded_parameters()

        assert params["frequency"].min() >= 0.03
        assert params["frequency"].max() <= 0.35
        assert params["sigma"].min() >= 1.0
        assert params["sigma"].max() <= prior.kernel_size / 2 + 1e-6
        assert params["gamma"].min() >= 0.25
        assert params["gamma"].max() <= 2.0
        theta_delta = torch.atan2(
            torch.sin(params["theta"] - prior.base_theta),
            torch.cos(params["theta"] - prior.base_theta),
        ).abs()
        assert theta_delta.max() <= math.pi / 16 + 1e-6

    def test_descriptor_has_finite_parameter_gradients(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(scales=2, orientations=4, kernel_size=9)
        maps = prior.describe(torch.randn(2, 1, 24, 24))
        loss = maps["gabor_energy"].mean() + maps["gabor_anisotropy"].mean()
        loss.backward()

        grads = [p.grad for p in prior.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    def test_legacy_gabor_checkpoint_keys_still_load_strictly(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(filters=8, kernel_size=9)
        state = {
            "log_frequency": torch.linspace(-2.8, -1.2, 8),
            "theta_raw": torch.linspace(0, math.pi, 8),
            "log_sigma": torch.full((8,), math.log(1.8)),
            "gamma_raw": torch.zeros(8),
            "phase": torch.zeros(8),
        }
        prior.load_state_dict(state, strict=True)
        assert set(prior.state_dict()) == set(state)


class TestResidualFrequencyPreconditioner:
    def _inputs(self):
        from src.model.noise.base import BBDMBridgeSchedule

        residual = torch.randn(2, 1, 32, 32)
        timesteps = torch.tensor([100, 700], dtype=torch.long)
        schedule = BBDMBridgeSchedule(num_train_timesteps=1000)
        orientation = torch.rand(2, 8, 32, 32)
        return residual, timesteps, schedule, orientation

    def test_zero_initialized_injections_match_decoder_shapes(self):
        from src.model.frequency.residual_preconditioner import ResidualFrequencyPreconditioner

        residual, timesteps, schedule, orientation = self._inputs()
        module = ResidualFrequencyPreconditioner(
            output_channels=(256, 256, 128, 64),
            gabor_orientations=8,
            use_gabor_gate=True,
        )
        injections, diagnostics = module(residual, timesteps, schedule, orientation)

        assert [tuple(x.shape) for x in injections] == [
            (2, 256, 4, 4),
            (2, 256, 8, 8),
            (2, 128, 16, 16),
            (2, 64, 32, 32),
        ]
        assert all(torch.count_nonzero(x).item() == 0 for x in injections)
        assert diagnostics["band_gates"].shape == (2, 3)
        assert torch.all((diagnostics["band_gates"] > 0) & (diagnostics["band_gates"] < 1))

    def test_gates_are_independent_not_softmax_normalized(self):
        from src.model.frequency.residual_preconditioner import ResidualFrequencyPreconditioner

        residual, timesteps, schedule, orientation = self._inputs()
        module = ResidualFrequencyPreconditioner(gabor_orientations=8)
        _, diagnostics = module(residual, timesteps, schedule, orientation)
        sums = diagnostics["band_gates"].sum(dim=1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
        assert torch.isfinite(diagnostics["band_log_snr"]).all()

    def test_gabor_factor_starts_at_identity_and_stays_bounded(self):
        from src.model.frequency.residual_preconditioner import ResidualFrequencyPreconditioner

        _, _, _, orientation = self._inputs()
        module = ResidualFrequencyPreconditioner(
            gabor_orientations=8, use_gabor_gate=True, gate_strength=0.1
        )
        factor = module.gabor_factor(orientation, size=(16, 16))
        assert torch.count_nonzero(factor - 1).item() == 0

        with torch.no_grad():
            module.gabor_gate.weight.fill_(10.0)
            module.gabor_gate.bias.fill_(10.0)
        factor = module.gabor_factor(orientation, size=(16, 16))
        assert factor.min() >= 0.9 - 1e-6
        assert factor.max() <= 1.1 + 1e-6

    def test_zero_projection_receives_gradient_and_can_leave_noop(self):
        from src.model.frequency.residual_preconditioner import ResidualFrequencyPreconditioner

        residual, timesteps, schedule, orientation = self._inputs()
        module = ResidualFrequencyPreconditioner(gabor_orientations=8)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        injections, _ = module(residual, timesteps, schedule, orientation)
        loss = sum((x - 1).square().mean() for x in injections)
        loss.backward()
        final_weights = [head.final.weight for head in module.projection_heads]
        assert all(w.grad is not None and torch.isfinite(w.grad).all() for w in final_weights)
        optimizer.step()
        updated, _ = module(residual, timesteps, schedule, orientation)
        assert any(torch.count_nonzero(x).item() > 0 for x in updated)

    def test_gabor_only_variant_gates_high_band_without_wavelet_injection(self):
        from src.model.frequency.residual_preconditioner import ResidualFrequencyPreconditioner

        residual, timesteps, schedule, orientation = self._inputs()
        module = ResidualFrequencyPreconditioner(
            inject_wavelet=False,
            use_gabor_gate=True,
            gabor_orientations=8,
            gate_strength=0.1,
        )
        initial = module.modulate_residual(residual, orientation)
        injections, _ = module(residual, timesteps, schedule, orientation)
        assert torch.allclose(initial, residual, atol=1e-6, rtol=1e-6)
        assert injections == []

        with torch.no_grad():
            module.gabor_gate.weight.fill_(0.5)
            module.gabor_gate.bias.fill_(0.1)
        modulated = module.modulate_residual(residual, orientation)
        assert not torch.allclose(modulated, residual)
        assert modulated.shape == residual.shape
        assert torch.isfinite(modulated).all()


class TestResidualBBDMIntegration:
    def test_boundary_reliable_mode_constructs_without_noisy_state_modulation(self):
        from src.model.frequency.boundary_reliable import (
            BoundaryReliableFrequencyInjector,
        )
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=False)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": False,
        })

        model = SLMFBBDM.from_config(cfg)

        assert model.residual_frequency_mode == "boundary_reliable"
        assert isinstance(
            model.residual_preconditioner, BoundaryReliableFrequencyInjector
        )
        assert not hasattr(model.residual_preconditioner, "modulate_residual")

    def test_boundary_reliable_frequency_is_added_without_gating_adapter_output(
        self, monkeypatch
    ):
        from src.model.interfaces import ConditionBundle
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=False)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": False,
        })
        cfg["modules"]["zero_adapter"]["enabled"] = True
        cfg["modules"]["organ_prior"] = {"enabled": True, "organ_channels": 6}
        model = SLMFBBDM.from_config(cfg)
        shapes = [
            (1, 256, 4, 4),
            (1, 256, 8, 8),
            (1, 128, 16, 16),
            (1, 64, 32, 32),
        ]
        adapter = [torch.full(shape, 2.0) for shape in shapes]
        frequency = [torch.full(shape, 3.0) for shape in shapes]
        monkeypatch.setattr(
            model, "_build_adapter_injections", lambda *args, **kwargs: adapter
        )
        monkeypatch.setattr(
            model, "_build_frequency_injections", lambda *args, **kwargs: frequency
        )

        combined = model._build_skip_injections(
            ConditionBundle.empty(),
            torch.tensor([10]),
            noisy_residual=torch.randn(1, 1, 32, 32),
        )

        assert all(torch.all(tensor == 5.0) for tensor in combined)

    def test_boundary_reliable_keeps_real_organ_adapter_and_roi_suv_interfaces(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=True)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": True,
        })
        cfg["modules"]["zero_adapter"] = {"enabled": True}
        cfg["modules"]["organ_prior"] = {
            "enabled": True,
            "organ_channels": 6,
        }
        cfg["losses"]["roi_suv"] = {"enabled": True, "weight": 0.1}
        model = SLMFBBDM.from_config(cfg)
        batch = _model_batch()
        batch["organ_mask"][:, 1, 8:20, 8:20] = 1.0

        loss, logs = model(batch, timesteps=torch.tensor([25]))

        assert torch.isfinite(loss)
        assert "loss/roi_suv/loss" in logs
        assert logs["module/organ_prior"].item() == 1.0

    def test_boundary_reliable_forward_routes_only_inference_available_ct_and_gabor(
        self, monkeypatch
    ):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=True)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": True,
        })
        model = SLMFBBDM.from_config(cfg)
        injector = model.residual_preconditioner
        original_forward = injector.forward
        captured = {}

        def _capture(*args, **kwargs):
            captured["ct"] = args[3]
            captured["gabor_orientation"] = kwargs.get("gabor_orientation")
            captured["keywords"] = set(kwargs)
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(injector, "forward", _capture)
        loss, _ = model(_model_batch(), timesteps=torch.tensor([25]))

        assert torch.isfinite(loss)
        assert captured["ct"].shape == (1, 1, 32, 32)
        assert captured["gabor_orientation"].shape == (1, 4, 32, 32)
        assert captured["keywords"] == {"gabor_orientation"}

    def test_boundary_reliable_forward_logs_native_subband_gates_and_tv(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=True)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": True,
        })
        model = SLMFBBDM.from_config(cfg)

        _, logs = model(_model_batch(), timesteps=torch.tensor([25]))

        for level in (2, 1):
            for band in ("lh", "hl", "hh"):
                assert f"frequency/gate_l{level}_{band}" in logs
        assert "frequency/gate_tv" in logs
        assert torch.isfinite(logs["frequency/gate_tv"])

    def test_safe_gabor_route_passes_orientation_and_anisotropy(self, monkeypatch):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=True)
        cfg["modules"]["residual_frequency"].update({
            "mode": "boundary_reliable",
            "band_scales": [0.5, 0.25],
            "use_directional_reliability": False,
            "use_gabor_agreement": True,
            "gabor_agreement_alpha": 0.10,
        })
        model = SLMFBBDM.from_config(cfg)
        injector = model.residual_preconditioner
        original_forward = injector.forward
        captured = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(injector, "forward", _capture)
        loss, _ = model(_model_batch(), timesteps=torch.tensor([25]))

        assert torch.isfinite(loss)
        assert captured["gabor_orientation"].shape[1] == 4
        assert captured["gabor_anisotropy"].shape == (1, 1, 32, 32)

    def test_legacy_state_modulation_can_be_disabled_without_disabling_skips(
        self, monkeypatch
    ):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=True, gabor=True)
        cfg["modules"]["residual_frequency"].update({
            "mode": "legacy",
            "inject_wavelet": False,
            "state_modulation": False,
        })
        model = SLMFBBDM.from_config(cfg)
        calls = []

        def _unexpected(*args, **kwargs):
            calls.append(True)
            return args[0]

        monkeypatch.setattr(
            model.residual_preconditioner, "modulate_residual", _unexpected
        )
        loss, _ = model(_model_batch(), timesteps=torch.tensor([25]))

        assert torch.isfinite(loss)
        assert calls == []

    def test_wavelet_unet_is_selected_only_when_explicitly_enabled(self):
        from src.model.bbdm_unet import BBDMUNet
        from src.model.slmf_bbdm import SLMFBBDM
        from src.model.wavelet_unet import WaveletBBDMUNet

        legacy = SLMFBBDM.from_config(
            _residual_config(frequency=False, gabor=False)
        )
        wavelet_cfg = _residual_config(frequency=False, gabor=False)
        wavelet_cfg["modules"]["wavelet_unet"] = {
            "enabled": True,
            "mix_kernel_size": 3,
        }
        wavelet = SLMFBBDM.from_config(wavelet_cfg)

        assert isinstance(legacy.unet, BBDMUNet)
        assert isinstance(wavelet.unet, WaveletBBDMUNet)

    def test_wavelet_unet_keeps_optional_organ_and_suv_interfaces(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=False, gabor=False)
        cfg["modules"]["wavelet_unet"] = {"enabled": True}
        cfg["modules"]["organ_prior"] = {
            "enabled": True,
            "organ_channels": 6,
        }
        cfg["modules"]["zero_adapter"] = {"enabled": True}
        cfg["losses"]["roi_suv"] = {"enabled": True, "weight": 0.1}
        model = SLMFBBDM.from_config(cfg)
        batch = _model_batch()
        batch["organ_mask"][:, 1, 8:16, 8:16] = 1.0

        loss, logs = model(batch, timesteps=torch.tensor([10]))

        assert torch.isfinite(loss)
        assert "loss/roi_suv/loss" in logs
        assert logs["module/wavelet_unet"].item() == 1.0

    def test_initialization_seed_makes_unet_identical_across_variants(self):
        from src.model.slmf_bbdm import SLMFBBDM

        residual_only = _residual_config(frequency=False, gabor=False)
        residual_only["model"]["initialization_seed"] = 4242
        frequency_gabor = _residual_config(frequency=True, gabor=True)
        frequency_gabor["model"]["initialization_seed"] = 4242

        torch.manual_seed(123)
        residual_model = SLMFBBDM.from_config(residual_only)
        torch.manual_seed(123)
        frequency_model = SLMFBBDM.from_config(frequency_gabor)

        residual_state = residual_model.unet.state_dict()
        frequency_state = frequency_model.unet.state_dict()
        assert residual_state.keys() == frequency_state.keys()
        for name in residual_state:
            assert torch.equal(residual_state[name], frequency_state[name]), name

    def test_frozen_mean_requires_checkpoint(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=False, gabor=False)
        cfg["modules"]["conditional_mean"]["freeze"] = True
        with pytest.raises(ValueError, match="checkpoint"):
            SLMFBBDM.from_config(cfg)

    def test_pretrained_mean_loads_strictly_and_stays_frozen_in_train_mode(self, tmp_path):
        from src.model.mean_predictor import LowFrequencyPETPredictor
        from src.model.slmf_bbdm import SLMFBBDM

        source = LowFrequencyPETPredictor(base_channels=8, levels=2)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(0.125)
        checkpoint_path = tmp_path / "mean_best.pt"
        torch.save(
            {
                "format_version": 1,
                "model": source.state_dict(),
                "mean_config": {"base_channels": 8, "levels": 2},
                "epoch": 3,
                "val_loss": 0.1,
            },
            checkpoint_path,
        )

        cfg = _residual_config(frequency=False, gabor=False, mean_weight=0.0)
        cfg["modules"]["conditional_mean"].update({
            "checkpoint": str(checkpoint_path),
            "freeze": True,
        })
        model = SLMFBBDM.from_config(cfg)
        assert model.mean_frozen is True
        assert all(not parameter.requires_grad for parameter in model.mean_predictor.parameters())
        assert all(
            torch.allclose(parameter, torch.full_like(parameter, 0.125))
            for parameter in model.mean_predictor.parameters()
        )
        model.train()
        assert model.training is True
        assert model.mean_predictor.training is False

    def test_mean_checkpoint_rejects_unsupported_format(self, tmp_path):
        from src.model.mean_predictor import LowFrequencyPETPredictor
        from src.model.slmf_bbdm import SLMFBBDM

        predictor = LowFrequencyPETPredictor(base_channels=8, levels=2)
        checkpoint_path = tmp_path / "bad_mean.pt"
        torch.save(
            {"format_version": 99, "model": predictor.state_dict()},
            checkpoint_path,
        )
        cfg = _residual_config(frequency=False, gabor=False)
        cfg["modules"]["conditional_mean"].update({
            "checkpoint": str(checkpoint_path),
            "freeze": True,
        })
        with pytest.raises(ValueError, match="format_version"):
            SLMFBBDM.from_config(cfg)

    def test_invalid_module_combinations_fail_at_construction(self):
        from src.model.slmf_bbdm import SLMFBBDM

        no_mean = _residual_config()
        no_mean["modules"]["conditional_mean"]["enabled"] = False
        with pytest.raises(ValueError, match="conditional_mean"):
            SLMFBBDM.from_config(no_mean)

        no_bridge = _residual_config()
        no_bridge["modules"]["residual_bridge"]["enabled"] = False
        with pytest.raises(ValueError, match="residual_bridge"):
            SLMFBBDM.from_config(no_bridge)

        ddpm = _residual_config(frequency=False, gabor=False)
        ddpm["modules"].pop("bbdm_bridge")
        ddpm["modules"]["noise"] = {"enabled": True, "name": "ddpm"}
        with pytest.raises(ValueError, match="bbdm_bridge"):
            SLMFBBDM.from_config(ddpm)

        no_gabor = _residual_config(frequency=True, gabor=False)
        no_gabor["modules"]["residual_frequency"]["use_gabor_gate"] = True
        with pytest.raises(ValueError, match="Gabor"):
            SLMFBBDM.from_config(no_gabor)

    def test_residual_bridge_forward_and_sampling_reconstruct_pet(self):
        from src.model.slmf_bbdm import SLMFBBDM

        model = SLMFBBDM.from_config(_residual_config())
        batch = _model_batch()
        loss, logs = model(batch, timesteps=torch.tensor([50]))
        assert torch.isfinite(loss)
        assert logs["module/conditional_mean"].item() == 1
        assert logs["module/residual_bridge"].item() == 1
        assert logs["module/residual_frequency"].item() == 1
        assert "loss/mean_lowpass" in logs

        result = model.sample(batch, num_steps=2)
        assert result["synthetic_pet"].shape == batch["pet"].shape
        assert torch.allclose(
            result["synthetic_pet"], result["mean_pet"] + result["pred_residual"]
        )
        assert torch.isfinite(result["synthetic_pet"]).all()

    def test_detached_mean_receives_only_its_explicit_lowpass_loss(self):
        from src.model.slmf_bbdm import SLMFBBDM

        model = SLMFBBDM.from_config(
            _residual_config(frequency=False, gabor=False, mean_weight=0.0)
        )
        loss, _ = model(_model_batch(), timesteps=torch.tensor([50]))
        loss.backward()
        grads = [p.grad for p in model.mean_predictor.parameters()]
        assert all(g is None or torch.count_nonzero(g).item() == 0 for g in grads)

    def test_explicitly_disabled_path_constructs_no_new_modules(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config(frequency=False, gabor=False)
        cfg["modules"]["conditional_mean"]["enabled"] = False
        cfg["modules"]["residual_bridge"]["enabled"] = False
        model = SLMFBBDM.from_config(cfg)
        assert model.mean_predictor is None
        assert model.residual_preconditioner is None

    def test_frequency_losses_require_their_real_routes(self):
        from src.model.slmf_bbdm import SLMFBBDM

        bad_gabor = _residual_config()
        bad_gabor["losses"]["gabor_consistency"] = {"enabled": True}
        with pytest.raises(ValueError, match="use_for_loss"):
            SLMFBBDM.from_config(bad_gabor)

        bad_residual = _residual_config(frequency=False, gabor=False)
        bad_residual["modules"]["conditional_mean"]["enabled"] = False
        bad_residual["modules"]["residual_bridge"]["enabled"] = False
        bad_residual["losses"]["residual_wavelet"] = {"enabled": True}
        with pytest.raises(ValueError, match="residual_wavelet"):
            SLMFBBDM.from_config(bad_residual)

    def test_frequency_losses_are_registered_and_run_in_model_forward(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _residual_config()
        cfg["modules"]["gabor"]["use_for_loss"] = True
        cfg["losses"] = {
            "residual_wavelet": {"enabled": True, "weight": 0.05},
            "gabor_consistency": {
                "enabled": True,
                "weight": 0.02,
                "orientation_weight": 0.1,
            },
        }
        model = SLMFBBDM.from_config(cfg)
        loss, logs = model(_model_batch(), timesteps=torch.tensor([25]))
        assert torch.isfinite(loss)
        assert "loss/residual_wavelet/loss" in logs
        assert "loss/gabor_consistency/amplitude" in logs
        assert "loss/gabor_consistency/orientation_js" in logs


def _loss_context(pred_residual, target_residual, mask, condition=None):
    from src.model.interfaces import ConditionBundle, LossContext

    pred_pet = pred_residual
    target_pet = target_residual
    batch_size = pred_pet.shape[0]
    return LossContext(
        model_pred=pred_residual,
        loss_target=target_residual,
        target_pet=target_pet,
        pred_x0=pred_pet,
        timesteps=torch.zeros(batch_size, dtype=torch.long),
        tau=torch.zeros(batch_size),
        batch={"mask": mask},
        condition=condition or ConditionBundle.empty(),
        pred_residual=pred_residual,
        target_residual=target_residual,
        mean_pet=torch.zeros_like(pred_pet),
    )


class TestResidualFrequencyLosses:
    def test_residual_wavelet_loss_is_lesion_weighted_and_mask_optional(self):
        from src.model.loss_terms.residual_frequency import ResidualWaveletLoss

        target = torch.zeros(1, 1, 32, 32)
        pred = target.clone()
        pred[:, :, 14:18, 14:18] = 1.0
        empty = torch.zeros_like(target)
        lesion = empty.clone()
        lesion[:, :, 14:18, 14:18] = 1.0
        term = ResidualWaveletLoss(
            lesion_weight=4.0,
            band_weights=(0.5, 1.0, 1.5),
            active_tau_max=1.0,
            weight=1.0,
        )

        global_loss, global_logs = term(_loss_context(pred, target, empty))
        lesion_loss, lesion_logs = term(_loss_context(pred, target, lesion))
        assert torch.isfinite(global_loss) and global_loss > 0
        assert lesion_loss > global_loss
        assert set(("residual_wavelet/low", "residual_wavelet/mid", "residual_wavelet/high")) <= set(lesion_logs)
        assert torch.isfinite(global_logs["residual_wavelet/loss"])

    def test_residual_wavelet_loss_is_noop_without_residual_context(self):
        from src.model.loss_terms.residual_frequency import ResidualWaveletLoss

        ctx = _loss_context(
            torch.zeros(1, 1, 32, 32),
            torch.zeros(1, 1, 32, 32),
            torch.zeros(1, 1, 32, 32),
        )
        ctx.pred_residual = None
        ctx.target_residual = None
        loss, logs = ResidualWaveletLoss()(ctx)
        assert loss.item() == 0
        assert logs["residual_wavelet/available"].item() == 0

    def test_gabor_consistency_matches_target_direction_not_isotropy(self):
        from src.model.interfaces import ConditionBundle
        from src.model.loss_terms.residual_frequency import GaborConsistencyLoss
        from src.model.priors.gabor import GaborPrior

        descriptor = GaborPrior(scales=1, orientations=4, kernel_size=9)
        target = torch.zeros(1, 1, 32, 32)
        target[:, :, :, ::4] = 1.0
        different = torch.zeros_like(target)
        different[:, :, ::4, :] = 1.0
        target_maps = descriptor.describe(target, detach_parameters=True)
        same_maps = descriptor.describe(target.clone(), detach_parameters=True)
        different_maps = descriptor.describe(different, detach_parameters=True)

        def bundle(pred_maps):
            return ConditionBundle(maps={
                "gabor_pred_feat": pred_maps["gabor_feat"],
                "gabor_target_feat": target_maps["gabor_feat"],
                "gabor_pred_orientation": pred_maps["gabor_orientation"],
                "gabor_target_orientation": target_maps["gabor_orientation"],
            })

        mask = torch.zeros_like(target)
        term = GaborConsistencyLoss(orientation_weight=1.0, weight=1.0)
        same, _ = term(_loss_context(target, target, mask, bundle(same_maps)))
        mismatch, logs = term(_loss_context(different, target, mask, bundle(different_maps)))
        assert torch.isfinite(same) and torch.isfinite(mismatch)
        assert mismatch > same
        assert logs["gabor_consistency/orientation_js"] > 0

    def test_gabor_consistency_handles_nonempty_lesion_mask(self):
        from src.model.interfaces import ConditionBundle
        from src.model.loss_terms.residual_frequency import GaborConsistencyLoss

        maps = {
            "gabor_pred_feat": torch.rand(1, 8, 16, 16, requires_grad=True),
            "gabor_target_feat": torch.rand(1, 8, 16, 16),
            "gabor_pred_orientation": torch.rand(1, 4, 16, 16, requires_grad=True),
            "gabor_target_orientation": torch.rand(1, 4, 16, 16),
        }
        lesion = torch.zeros(1, 1, 16, 16)
        lesion[:, :, 6:10, 6:10] = 1
        ctx = _loss_context(
            torch.zeros(1, 1, 16, 16),
            torch.zeros(1, 1, 16, 16),
            lesion,
            ConditionBundle(maps=maps),
        )
        loss, _ = GaborConsistencyLoss(lesion_weight=3.0)(ctx)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.isfinite(maps["gabor_pred_feat"].grad).all()
        assert torch.isfinite(maps["gabor_pred_orientation"].grad).all()
