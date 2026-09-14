"""Tests for the PNG-only baseline fixes.

Covers:
  - Gabor routing (inject_adapter / use_for_noise / use_for_hotspot / use_for_loss)
    are real, independently-switchable routes — not just YAML fields.
  - sample() returns the terminal pred_x0 (not the previous x_t).
  - HotspotPriorLoss target_mode=mask_only excludes off-mask hotspots.
  - PNG baseline has no trainable OrganPrior parameters.
  - PNG baseline config validation rejects SUV/organ/metadata activation.
  - PNG baseline forward + sample run cleanly with all routes closed.

Run:  python -m pytest tests/test_png_baseline.py -q
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.slmf_bbdm import SLMFBBDM
from src.model.config_utils import (
    validate_png_baseline_config,
    PNGBaselineConfigError,
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _base_cfg(**over_modules):
    """Minimal config for routing tests; gabor routes fully configurable."""
    mods = {
        "gabor": {"enabled": False},
        "organ_prior": {"enabled": False},
        "hotspot_prior": {"enabled": False},
        "semantic_prior": {"enabled": False},
        "zero_adapter": {"enabled": False},
        "condition_dropout": {"enabled": False},
        "scale_adaptive_noise": {"enabled": False, "name": "bbdm_bridge"},
    }
    mods.update(over_modules)
    return {
        "experiment": {"name": "png_test", "seed": 42},
        "data": {"image_size": 32, "mode": "png"},
        "runtime": {"eval_sampling_steps": 3},
        "model": {
            "objective": "pred_x0", "sample_scheduler": "ddim",
            "enable_heteroscedastic": False,
            "self_conditioning": {"enabled": False},
            "base_loss": {"mse_weight": 1.0, "l1_weight": 1.0, "gradient_weight": 0.1,
                          "min_snr_enabled": True, "min_snr_gamma": 5.0},
        },
        "modules": mods,
        "losses": {},
    }


def _fake_batch(B=2, H=32):
    return {
        "ct": torch.randn(B, 1, H, H),
        "pet": torch.randn(B, 1, H, H),
        "mask": torch.zeros(B, 1, H, H),
        "organ_mask": torch.zeros(B, 6, H, H),
        "organ_distance": torch.zeros(B, 6, H, H),
        "mu_map": torch.zeros(B, 1, H, H),
    }


# ---------------------------------------------------------------------------
# 1. Gabor routing is parsed and applied
# ---------------------------------------------------------------------------

class TestGaborRouting:

    def test_routes_parsed_from_config(self):
        cfg = _base_cfg(gabor={
            "enabled": True, "inject_adapter": True, "use_for_noise": True,
            "use_for_hotspot": True, "use_for_loss": True,
        })
        m = SLMFBBDM.from_config(cfg)
        assert m.gabor_routes == {
            "enabled": True, "inject_adapter": True, "use_for_noise": True,
            "use_for_hotspot": True, "use_for_loss": True,
        }

    def test_routes_default_to_false(self):
        cfg = _base_cfg(gabor={"enabled": True})  # no routes specified
        m = SLMFBBDM.from_config(cfg)
        for k in ("inject_adapter", "use_for_noise", "use_for_hotspot", "use_for_loss"):
            assert m.gabor_routes[k] is False

    def test_inject_adapter_false_passes_none_to_adapter(self):
        """When inject_adapter=False, _build_skip_injections must pass gabor_feat=None."""
        from unittest.mock import patch

        cfg = _base_cfg(
            gabor={"enabled": True, "inject_adapter": False},
            zero_adapter={"enabled": True},
        )
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        t = torch.zeros(2, dtype=torch.long)
        cond = m.build_condition_bundle(batch, t)

        captured = {}
        original = m.adapter.get_zero_conv_outputs

        def spy(*args, **kwargs):
            captured["gabor_feat"] = kwargs.get("gabor_feat", args[2] if len(args) > 2 else None)
            return original(*args, **kwargs)

        with patch.object(m.adapter, "get_zero_conv_outputs", side_effect=spy):
            m._build_skip_injections(cond, timesteps=t)
        assert captured["gabor_feat"] is None, \
            "inject_adapter=False must prevent gabor_feat from reaching the adapter"

    def test_inject_adapter_true_passes_gabor(self):
        from unittest.mock import patch

        cfg = _base_cfg(
            gabor={"enabled": True, "inject_adapter": True},
            zero_adapter={"enabled": True},
        )
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        t = torch.zeros(2, dtype=torch.long)
        cond = m.build_condition_bundle(batch, t)

        captured = {}
        original = m.adapter.get_zero_conv_outputs

        def spy(*args, **kwargs):
            captured["gabor_feat"] = kwargs.get("gabor_feat", args[2] if len(args) > 2 else None)
            return original(*args, **kwargs)

        with patch.object(m.adapter, "get_zero_conv_outputs", side_effect=spy):
            m._build_skip_injections(cond, timesteps=t)
        assert captured["gabor_feat"] is not None, \
            "inject_adapter=True must pass gabor_feat to the adapter"

    def test_use_for_noise_false_ignores_gabor_energy(self):
        """ScaleAdaptiveNoise with use_gabor_energy=False must be invariant to gabor_energy."""
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise
        from src.model.interfaces import ConditionBundle
        sched = ScaleAdaptiveNoise(num_train_timesteps=1000, use_gabor_energy=False,
                                   bridge_mode=True, enabled=True)
        x0 = torch.randn(2, 1, 8, 8)
        noise = torch.randn(2, 1, 8, 8)
        src = torch.zeros(2, 1, 8, 8)
        t = torch.tensor([500, 500])
        c1 = ConditionBundle(maps={"gabor_energy": torch.zeros(2, 1, 8, 8)})
        c2 = ConditionBundle(maps={"gabor_energy": torch.ones(2, 1, 8, 8)})
        out1 = sched.add_noise(x0, noise, t, c1, x_source=src)
        out2 = sched.add_noise(x0, noise, t, c2, x_source=src)
        assert torch.allclose(out1, out2), "use_for_noise=False must ignore gabor_energy"

    def test_use_for_noise_true_responds_to_gabor_energy(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise
        from src.model.interfaces import ConditionBundle
        sched = ScaleAdaptiveNoise(num_train_timesteps=1000, use_gabor_energy=True,
                                   bridge_mode=True, enabled=True)
        x0 = torch.randn(2, 1, 8, 8)
        noise = torch.randn(2, 1, 8, 8)
        src = torch.zeros(2, 1, 8, 8)
        t = torch.tensor([500, 500])
        c1 = ConditionBundle(maps={"gabor_energy": torch.zeros(2, 1, 8, 8)})
        c2 = ConditionBundle(maps={"gabor_energy": torch.ones(2, 1, 8, 8)})
        out1 = sched.add_noise(x0, noise, t, c1, x_source=src)
        out2 = sched.add_noise(x0, noise, t, c2, x_source=src)
        assert not torch.allclose(out1, out2), "use_for_noise=True should change output"

    def test_use_for_hotspot_route_controls_hotspot(self):
        """gabor.use_for_hotspot controls HotspotPrior.use_gabor_energy at build time."""
        cfg_on = _base_cfg(
            gabor={"enabled": True, "use_for_hotspot": True},
            hotspot_prior={"enabled": True, "use_gabor_energy": True},
        )
        cfg_off = _base_cfg(
            gabor={"enabled": True, "use_for_hotspot": False},
            hotspot_prior={"enabled": True, "use_gabor_energy": True},  # explicit but gated
        )
        m_on = SLMFBBDM.from_config(cfg_on)
        m_off = SLMFBBDM.from_config(cfg_off)
        assert m_on.priors["hotspot_prior"].use_gabor_energy is True
        assert m_off.priors["hotspot_prior"].use_gabor_energy is False, \
            "use_for_hotspot=False must force CT-only HotspotPrior"

    def test_use_for_hotspot_false_invariant_to_gabor_energy(self):
        """A CT-only HotspotPrior must not change when gabor_energy changes."""
        from src.model.priors.hotspot import HotspotPrior
        from src.model.interfaces import ConditionBundle
        hp = HotspotPrior(use_gabor_energy=False, enabled=True)
        torch.manual_seed(0)
        batch = {"ct": torch.randn(2, 1, 16, 16)}
        t = torch.zeros(2, dtype=torch.long)
        c1 = ConditionBundle(maps={"gabor_energy": torch.zeros(2, 1, 16, 16)})
        c2 = ConditionBundle(maps={"gabor_energy": torch.ones(2, 1, 16, 16)})
        out1 = hp(batch, t, c1).maps["hotspot_prior"]
        out2 = hp(batch, t, c2).maps["hotspot_prior"]
        assert torch.allclose(out1, out2)

    def test_all_routes_off_forward_and_sample_run(self):
        """enabled=true with all four routes false must still forward + sample."""
        cfg = _base_cfg(gabor={
            "enabled": True, "inject_adapter": False, "use_for_noise": False,
            "use_for_hotspot": False, "use_for_loss": False,
        })
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        loss, logs = m(batch)
        assert torch.isfinite(loss).item()
        # gabor route logs present
        for k in ("enabled", "inject_adapter", "use_for_noise", "use_for_hotspot", "use_for_loss"):
            assert f"module/gabor_{k}" in logs
        out = m.sample(batch, num_steps=2)
        assert torch.isfinite(out["synthetic_pet"]).all().item()


# ---------------------------------------------------------------------------
# 2. sample() returns the terminal pred_x0
# ---------------------------------------------------------------------------

class TestSampleEndpoint:

    def test_sample_output_is_finite(self):
        cfg = _base_cfg()
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        out = m.sample(batch, num_steps=3)
        assert out["synthetic_pet"].shape == batch["pet"].shape
        assert torch.isfinite(out["synthetic_pet"]).all().item()

    def test_sample_reproducible_with_seed(self):
        cfg = _base_cfg()
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        torch.manual_seed(123)
        a = m.sample(batch, num_steps=3)["synthetic_pet"]
        torch.manual_seed(123)
        b = m.sample(batch, num_steps=3)["synthetic_pet"]
        assert torch.allclose(a, b), "fixed seed must make sampling reproducible"

    def test_sample_returns_pred_x0_not_noise(self):
        """With steps=1 the output is the model's clean estimate of x_T, not x_T itself."""
        cfg = _base_cfg()
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        torch.manual_seed(7)
        out = m.sample(batch, num_steps=1)["synthetic_pet"].clone()
        # The initial state x_T for BBDM is m_T·CT + σ_T·ε; a single step should
        # already move the output away from that raw noisy state in general.
        torch.manual_seed(7)
        from src.model.slmf_bbdm import _add_noise
        cond = m.build_condition_bundle(batch, torch.zeros(2, dtype=torch.long))
        x_t = _add_noise(m.noise_schedule, torch.zeros_like(batch["pet"]),
                         torch.randn_like(batch["pet"]),
                         torch.full((2,), m.noise_schedule.num_train_timesteps - 1, dtype=torch.long),
                         cond, batch["ct"])
        # Output should not equal the raw initial noisy state (model denoised it).
        assert not torch.allclose(out, x_t, atol=1e-4)

    def test_sample_heteroscedastic_logvar_matches_output(self):
        cfg = _base_cfg()
        cfg["model"]["enable_heteroscedastic"] = True
        m = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(H=32)
        out = m.sample(batch, num_steps=2)
        assert "logvar" in out, "heteroscedastic model must return logvar"
        assert out["logvar"].shape == out["synthetic_pet"].shape
        assert torch.isfinite(out["logvar"]).all().item()


# ---------------------------------------------------------------------------
# 3. HotspotPriorLoss target_mode
# ---------------------------------------------------------------------------

class TestHotspotTargetMode:

    def _ctx_with_far_hotspot(self):
        """mask at centre; PET has a bright pixel far away."""
        from src.model.interfaces import LossContext, ConditionBundle
        H = 32
        pet = torch.full((1, 1, H, H), -0.5)
        pet[0, 0, 3, 3] = 1.0  # far-away bright pixel (top-left)
        pet[0, 0, 20, 20] = 0.3  # dim pixel inside mask area
        mask = torch.zeros(1, 1, H, H)
        mask[0, 0, 18:22, 18:22] = 1.0  # lesion bottom-right
        batch = {"pet": pet, "mask": mask, "ct": torch.zeros(1, 1, H, H)}
        tau = torch.tensor([0.5])
        return LossContext(
            model_pred=pet, loss_target=pet, target_pet=pet, pred_x0=pet,
            timesteps=torch.tensor([500]), tau=tau, batch=batch, condition=ConditionBundle(),
        )

    def test_mask_only_excludes_far_hotspot(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        loss = HotspotPriorLoss(target_mode="mask_only", enabled=True)
        target = loss._target(self._ctx_with_far_hotspot())
        # The far bright pixel at (3,3) must NOT be in the target
        assert target[0, 0, 3, 3].item() == 0.0
        # The mask region must be in the target
        assert target[0, 0, 20, 20].item() == 1.0

    def test_legacy_includes_far_hotspot(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        loss = HotspotPriorLoss(target_mode="legacy", pet_threshold_quantile=0.9, enabled=True)
        target = loss._target(self._ctx_with_far_hotspot())
        # legacy mode uses full-image top percentile → the bright pixel enters
        assert target[0, 0, 3, 3].item() == 1.0

    def test_mask_plus_local_uptake_excludes_far_hotspot(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        loss = HotspotPriorLoss(target_mode="mask_plus_local_uptake",
                                local_uptake_radius=5, enabled=True)
        target = loss._target(self._ctx_with_far_hotspot())
        assert target[0, 0, 3, 3].item() == 0.0, "local uptake must stay inside dilated ROI"

    def test_invalid_target_mode_raises(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        with pytest.raises(ValueError):
            HotspotPriorLoss(target_mode="bogus")

    def test_default_target_mode_is_mask_only(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        assert HotspotPriorLoss().target_mode == "mask_only"


# ---------------------------------------------------------------------------
# 4. PNG baseline: no trainable OrganPrior, no DICOM/SUV/organ paths
# ---------------------------------------------------------------------------

class TestPNGBaselineClean:

    def _png_model(self):
        cfg = _base_cfg(
            gabor={"enabled": True, "inject_adapter": False, "use_for_noise": False,
                   "use_for_hotspot": False, "use_for_loss": False},
            organ_prior={"enabled": False},
            hotspot_prior={"enabled": False},
        )
        return SLMFBBDM.from_config(cfg)

    def test_no_organ_prior_trainable_params(self):
        m = self._png_model()
        assert "organ_prior" in m.priors
        op = m.priors["organ_prior"]
        params = list(op.parameters())
        assert len(params) == 0, "OrganPrior (NoOp) must have zero trainable parameters"

    def test_png_baseline_config_validation_passes(self):
        # The shipped PNG baseline config must pass validation.
        cfg_path = os.path.join(os.path.dirname(__file__), "..",
                                "configs", "experiments", "slmf_png_baseline.yaml")
        if not os.path.exists(cfg_path):
            pytest.skip("slmf_png_baseline.yaml not present")
        from src.model.config_utils import load_full_config
        cfg = load_full_config(cfg_path)
        status = validate_png_baseline_config(cfg)  # must not raise
        assert status["data_mode"] == "png"

    def test_png_config_rejects_roi_suv(self):
        cfg = _base_cfg()
        cfg["data"]["mode"] = "png"
        cfg["losses"] = {"roi_suv": {"enabled": True}}
        with pytest.raises(PNGBaselineConfigError):
            validate_png_baseline_config(cfg)

    def test_png_config_rejects_organ_prior(self):
        cfg = _base_cfg()
        cfg["data"]["mode"] = "png"
        cfg["modules"]["organ_prior"] = {"enabled": True}
        with pytest.raises(PNGBaselineConfigError):
            validate_png_baseline_config(cfg)

    def test_png_config_rejects_metadata(self):
        cfg = _base_cfg()
        cfg["data"]["mode"] = "png"
        cfg["metadata"] = {"enabled": True}
        with pytest.raises(PNGBaselineConfigError):
            validate_png_baseline_config(cfg)

    def test_non_png_mode_skips_validation(self):
        cfg = _base_cfg()
        cfg["data"]["mode"] = "dicom"
        cfg["losses"] = {"roi_suv": {"enabled": True}}
        # Should NOT raise (data mode is not png)
        status = validate_png_baseline_config(cfg)
        assert status["data_mode"] == "dicom"
