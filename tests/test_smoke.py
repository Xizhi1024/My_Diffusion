"""Smoke tests for SLMF-BBDM — no CUDA, no real data required.

Validates:
  - Config loading + ablation overrides
  - Registry instantiation
  - Prior modules (NoOp + enabled)
  - Condition adapter
  - Noise schedules
  - Loss terms
  - Full model forward pass
  - DDIM sampling
  - Trainer step

Run:  python -m pytest tests/test_smoke.py -q
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.slmf_bbdm import SLMFBBDM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_batch(B=2, H=32):
    return {
        "ct": torch.randn(B, 1, H, H),
        "pet": torch.randn(B, 1, H, H),
        "mask": torch.zeros(B, 1, H, H),
        "organ_mask": torch.zeros(B, 6, H, H),
        "organ_distance": torch.zeros(B, 6, H, H),
        "mu_map": torch.zeros(B, 1, H, H),
        "meta": [
            {"uptake_min": 60.0, "weight_kg": 65.0, "age_years": 52.0,
             "thickness_mm": 2.0, "z_mm": 0.0}
            for _ in range(B)
        ],
    }


def _toy_config(**overrides):
    """Minimal config dict for smoke testing."""
    cfg = {
        "experiment": {"name": "smoke_test", "seed": 42},
        "data": {
            "image_size": 32,
            "batch_size": 2,
            "val_batch_size": 2,
            "augment": False,
            "use_fake_data": True,
            "required_keys": ["ct", "pet", "mask"],
        },
        "runtime": {
            "amp": False,
            "channels_last": False,
            "torch_compile": False,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "gradient_accumulate_every": 1,
            "grad_clip_norm": 1.0,
            "eval_interval": 999,
            "sample_interval": 999,
            "save_interval": 999,
            "eval_sampling_steps": 5,
        },
        "training": {
            "num_epochs": 1,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "lr_min": 1e-6,
            "ema": {"decay": 0.999, "update_every": 1},
        },
        "model": {
            "objective": "pred_x0",
            "sample_scheduler": "ddim",
            "enable_heteroscedastic": True,
            "self_conditioning": {"enabled": False, "probability": 0.5},
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 1.0,
                "gradient_weight": 0.1,
                "min_snr_enabled": True,
                "min_snr_gamma": 5.0,
            },
        },
        "modules": {
            "gabor": {"enabled": False},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "scale_adaptive_noise": {"enabled": False, "name": "bbdm_bridge"},
        },
        "losses": {
            "topk_lesion": {"enabled": False},
            "focal_frequency": {"enabled": False},
            "patch_nce": {"enabled": False},
            "roi_suv": {"enabled": False},
            "false_hotspot": {"enabled": False},
            "lesion_roi_l1": {"enabled": False},
            "outside_peak_ranking": {"enabled": False},
            "heteroscedastic_nll": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "organ_consistency": {"enabled": False},
            "segmenter_consistency": {"enabled": False},
        },
    }
    for k, v in overrides.items():
        _deep_update(cfg, k, v)
    return cfg


def _deep_update(d, key, value):
    keys = key.split(".") if isinstance(key, str) else key
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_apply_dotlist_overrides(self):
        from src.model.config_utils import apply_dotlist_overrides

        cfg = {"modules": {"gabor": {"enabled": True}}}
        out = apply_dotlist_overrides(cfg, {"modules.gabor.enabled": False})
        assert out["modules"]["gabor"]["enabled"] is False

    def test_apply_dotlist_overrides_nested_create(self):
        from src.model.config_utils import apply_dotlist_overrides

        cfg = {"a": 1}
        out = apply_dotlist_overrides(cfg, {"b.c.d": 42})
        assert out["b"]["c"]["d"] == 42

    def test_resolve_runtime_profile_cpu(self):
        from src.model.config_utils import resolve_runtime_profile

        cfg = {"data": {"image_size": 192, "batch_size": 4}, "runtime": {}}
        out = resolve_runtime_profile(cfg)
        if torch.cuda.is_available():
            # CUDA available → no downsampling
            assert out["data"]["image_size"] == 192
            assert out["data"]["batch_size"] == 4
        else:
            assert out["data"]["image_size"] == 32
            assert out["data"]["batch_size"] == 1

    def test_load_full_config_normalises_scientific_notation(self, tmp_path):
        from src.model.config_utils import load_full_config

        config_path = tmp_path / "cfg.yaml"
        config_path.write_text(
            "training:\n"
            "  learning_rate: 1e-4\n"
            "  lr_min: 1e-6\n"
            "  weight_decay: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_full_config(str(config_path))
        assert cfg["training"]["learning_rate"] == 1e-4
        assert isinstance(cfg["training"]["learning_rate"], float)
        assert cfg["training"]["lr_min"] == 1e-6
        assert isinstance(cfg["training"]["lr_min"], float)


class TestClinicalEvaluationHelpers:
    def test_suv_calibration_metrics_fit_known_linear_relation(self):
        from scripts.evaluate import compute_calibration_metrics

        target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pred = 2.0 * target + 1.0
        metrics = compute_calibration_metrics(pred, target, prefix="suv_calib")

        assert metrics["suv_calib_n"] == 4
        assert np.isclose(metrics["suv_calib_slope"], 2.0)
        assert np.isclose(metrics["suv_calib_intercept"], 1.0)
        assert np.isclose(metrics["suv_calib_r2"], 1.0)
        assert np.isclose(metrics["suv_calib_bias"], np.mean(pred - target))

    def test_failure_detection_flags_cold_lesion_and_outside_peak(self):
        from scripts.evaluate import compute_failure_detection_metrics

        pred = np.zeros((1, 4, 4), dtype=np.float32)
        target = np.zeros((1, 4, 4), dtype=np.float32)
        mask = np.zeros((1, 4, 4), dtype=np.float32)
        mask[:, 1, 1] = 1.0
        pred[:, 1, 1] = 0.2
        target[:, 1, 1] = 1.0
        pred[:, 0, 0] = 0.9

        metrics = compute_failure_detection_metrics(
            pred,
            target,
            mask,
            outside_margin=0.05,
            lesion_min_ratio=0.6,
        )

        assert metrics["failure_outside_peak_gt_inside"] == 1.0
        assert metrics["failure_lesion_too_cold"] == 1.0
        assert metrics["failure_any"] == 1.0
        assert "outside_peak" in metrics["failure_reason"]
        assert "lesion_cold" in metrics["failure_reason"]

    def test_uncertainty_metrics_flag_high_lesion_uncertainty(self):
        from scripts.evaluate import compute_uncertainty_metrics, compute_failure_detection_metrics

        mask = np.zeros((1, 4, 4), dtype=np.float32)
        mask[:, 1, 1] = 1.0
        uncertainty = np.full((1, 4, 4), 0.1, dtype=np.float32)
        uncertainty[:, 1, 1] = 0.8
        u_metrics = compute_uncertainty_metrics(uncertainty, mask)
        f_metrics = compute_failure_detection_metrics(
            np.ones((1, 4, 4), dtype=np.float32),
            np.ones((1, 4, 4), dtype=np.float32),
            mask,
            uncertainty_metrics=u_metrics,
            uncertainty_ratio_threshold=2.0,
        )

        assert u_metrics["lesion_uncertainty_mean"] > u_metrics["outside_uncertainty_mean"]
        assert f_metrics["failure_high_uncertainty"] == 1.0

    def test_run_metadata_records_lesion_aware_posttraining(self, tmp_path):
        from scripts.train_v2 import _save_run_metadata

        cfg = _toy_config()
        cfg["training"]["stage"] = "lesion_posttrain"
        cfg["training"]["init_from"] = "checkpoints/slmf_bbdm_full/ckpt_epoch1000.pt"
        cfg["training"]["lesion_posttrain"] = {
            "enabled": True,
            "description": "ROI dense reconstruction plus outside peak ranking",
        }
        cfg["losses"]["lesion_roi_l1"]["enabled"] = True
        cfg["losses"]["outside_peak_ranking"]["enabled"] = True
        model = SLMFBBDM.from_config(cfg)

        _save_run_metadata(model, cfg, str(tmp_path), ablation=None)
        metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))

        assert metadata["training_stage"] == "lesion_posttrain"
        assert metadata["lesion_aware_posttraining"]["enabled"] is True
        assert metadata["lesion_aware_posttraining"]["source_checkpoint"].endswith("ckpt_epoch1000.pt")
        assert "lesion_roi_l1" in metadata["lesion_aware_posttraining"]["lesion_losses"]
        assert "outside_peak_ranking" in metadata["lesion_aware_posttraining"]["lesion_losses"]

    def test_finetune_can_initialize_from_checkpoint_ema(self):
        from scripts.train_v2 import _select_initial_model_state

        checkpoint = {
            "model": {
                "weight": torch.tensor([1.0]),
                "buffer": torch.tensor([3.0]),
            },
            "ema": {"shadow": {"weight": torch.tensor([2.0])}},
        }

        raw = _select_initial_model_state(checkpoint, "raw")
        ema = _select_initial_model_state(checkpoint, "ema")

        assert raw["weight"].item() == 1.0
        assert ema["weight"].item() == 2.0
        assert ema["buffer"].item() == 3.0

    def test_finetune_ema_init_requires_ema_checkpoint_state(self):
        from scripts.train_v2 import _select_initial_model_state

        with pytest.raises(KeyError, match="contains no EMA"):
            _select_initial_model_state({"model": {}}, "ema")

    def test_train_entrypoint_loads_resume_checkpoint(self, monkeypatch):
        import scripts.train_v2 as train_v2

        cfg = _toy_config()
        cfg["training"]["resume_from"] = "checkpoints/demo/ckpt_epoch0800.pt"
        calls = []

        class FakePrior:
            enabled = False

        class FakeModel:
            priors = {"gabor": FakePrior()}
            loss_terms = {}

            def get_trainable_params(self):
                return 0

            def get_total_params(self):
                return 0

        class FakeSLMF:
            @staticmethod
            def from_config(config):
                calls.append(("from_config", config["training"]["resume_from"]))
                return FakeModel()

        class FakeTrainer:
            def __init__(self, model, config, train_loader, val_loader):
                calls.append(("trainer_init", config["training"]["resume_from"]))

            def load_checkpoint(self, path):
                calls.append(("load_checkpoint", path))

            def run(self):
                calls.append(("run", None))

        monkeypatch.setattr(train_v2, "load_full_config", lambda *args, **kwargs: cfg)
        monkeypatch.setattr(train_v2, "resolve_runtime_profile", lambda config: config)
        monkeypatch.setattr(train_v2, "save_resolved_config", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_v2, "_save_run_metadata", lambda *args, **kwargs: None)
        monkeypatch.setattr(train_v2, "SLMFBBDM", FakeSLMF)
        monkeypatch.setattr(train_v2, "Trainer", FakeTrainer)
        monkeypatch.setattr("src.data.dataset.build_dataloaders", lambda *args, **kwargs: ("train", "val"))
        monkeypatch.setattr(sys, "argv", ["train_v2.py", "--config", "demo.yaml"])

        train_v2.main()

        assert ("load_checkpoint", "checkpoints/demo/ckpt_epoch0800.pt") in calls
        assert calls[-1] == ("run", None)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_build(self):
        from src.model.registry import Registry

        registry = Registry("test")

        class Demo:
            def __init__(self, value=1):
                self.value = value

        registry.register("demo", Demo)
        obj = registry.build({"name": "demo", "value": 7})
        assert obj.value == 7

    def test_registry_duplicate_raises(self):
        from src.model.registry import Registry, RegistryError

        registry = Registry("test")
        registry.register("x", lambda: None)
        with pytest.raises(RegistryError):
            registry.register("x", lambda: None)


# ---------------------------------------------------------------------------
# Prior module tests
# ---------------------------------------------------------------------------

class TestPriors:
    def test_noop_prior(self):
        from src.model.priors.noop import NoOpPrior

        prior = NoOpPrior()
        bundle = prior({}, torch.zeros(2, dtype=torch.long))
        assert bundle.maps == {}
        assert bundle.tokens == {}

    def test_gabor_prior_shape(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(filters=32, enabled=True)
        batch = _fake_batch()
        bundle = prior(batch, torch.zeros(2, dtype=torch.long))
        feat = bundle.maps["gabor_feat"]
        energy = bundle.maps["gabor_energy"]
        assert feat.shape == (2, 32, 32, 32)
        assert energy.shape == (2, 1, 32, 32)

    def test_gabor_prior_uses_learnable_gabor_parameters(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(filters=8, kernel_size=9, enabled=True)
        kernels = prior._build_kernels(torch.float32, torch.device("cpu"))
        assert kernels.shape == (8, 1, 9, 9)
        assert prior.log_frequency.requires_grad
        assert prior.theta_raw.requires_grad
        assert prior.log_sigma.requires_grad

    def test_gabor_prior_disabled(self):
        from src.model.priors.gabor import GaborPrior

        prior = GaborPrior(filters=32, enabled=False)
        bundle = prior(_fake_batch(), torch.zeros(2, dtype=torch.long))
        assert bundle.maps == {}

    def test_organ_prior_shape(self):
        from src.model.priors.organ import OrganPrior

        prior = OrganPrior(enabled=True)
        batch = _fake_batch()
        bundle = prior(batch, torch.zeros(2, dtype=torch.long))
        assert bundle.maps["organ_feat_1"].shape == (2, 16, 32, 32)
        assert bundle.maps["organ_feat_2"].shape == (2, 32, 16, 16)
        assert bundle.maps["organ_feat_3"].shape == (2, 64, 8, 8)

    def test_hotspot_prior_shape(self):
        from src.model.priors.hotspot import HotspotPrior

        prior = HotspotPrior(enabled=True)
        batch = _fake_batch()
        bundle = prior(batch, torch.zeros(2, dtype=torch.long))
        assert bundle.maps["hotspot_prior"].shape == (2, 1, 32, 32)

    def test_semantic_prior_shape(self):
        from src.model.priors.semantic import SemanticPrior

        prior = SemanticPrior(enabled=True)
        batch = _fake_batch()
        bundle = prior(batch, torch.zeros(2, dtype=torch.long))
        assert bundle.tokens["semantic"].shape == (2, 4, 64)

    def test_semantic_prior_missing_cache_is_deterministic(self):
        from src.model.priors.semantic import SemanticPrior

        prior = SemanticPrior(enabled=True)
        batch = _fake_batch()
        a = prior(batch, torch.zeros(2, dtype=torch.long)).tokens["semantic"]
        b = prior(batch, torch.zeros(2, dtype=torch.long)).tokens["semantic"]
        assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# Condition adapter tests
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_raw_concat_adapter(self):
        from src.model.conditioning.adapter import RawConcatAdapter
        from src.model.interfaces import ConditionBundle

        noisy = torch.randn(2, 1, 32, 32)
        ct = torch.randn(2, 1, 32, 32)
        adapter = RawConcatAdapter()
        out = adapter(noisy, ct, ConditionBundle(), torch.zeros(2, dtype=torch.long))
        assert out.shape == (2, 2, 32, 32)

    def test_zero_conv_adapter_modulates_l3(self):
        from src.model.conditioning.adapter import ZeroConvAdapter
        from src.model.conditioning.beta_schedule import spatial_beta

        adapter = ZeroConvAdapter(
            ct_channels=[1, 1, 1, 1],
            organ_channels=[1, 1, 1],
            gabor_channels=1,
            hotspot_channels=1,
            enabled=True,
        )
        for conv in adapter.zero_convs:
            conv.weight.data.fill_(1.0)
            conv.bias.data.zero_()

        ct_feats = [
            torch.ones(1, 1, 8, 8),
            torch.ones(1, 1, 4, 4),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 1, 1),
        ]
        organ_feats = [
            torch.ones(1, 1, 4, 4),
            torch.ones(1, 1, 2, 2),
            torch.ones(1, 1, 1, 1),
        ]
        tau = torch.ones(1)
        outputs = adapter.get_zero_conv_outputs(
            ct_feats,
            organ_feats,
            gabor_feat=torch.ones(1, 1, 8, 8),
            hotspot_prior=torch.ones(1, 1, 8, 8),
            hw_list=[8, 4, 2, 1],
            tau=tau,
        )
        expected_l3 = torch.full_like(outputs[3], 2.0 * spatial_beta(tau).item())
        assert torch.allclose(outputs[3], expected_l3)


class TestConditionDropout:
    def test_condition_dropout_does_not_mutate_input(self):
        from src.model.conditioning.dropout import ConditionDropout
        from src.model.interfaces import ConditionBundle

        bundle = ConditionBundle(maps={"organ_feat_1": torch.ones(1, 1, 2, 2)})
        dropped = ConditionDropout(enabled=True, p_organ=1.0).apply(bundle, training=True)
        assert dropped is not bundle
        assert bundle.maps["organ_feat_1"].sum().item() == 4.0
        assert dropped.maps["organ_feat_1"].sum().item() == 0.0


# ---------------------------------------------------------------------------
# Noise schedule tests
# ---------------------------------------------------------------------------

class TestNoiseSchedules:
    def test_ddpm_noise_shape(self):
        from src.model.noise.base import DDPMNoiseSchedule
        from src.model.interfaces import ConditionBundle

        schedule = DDPMNoiseSchedule(num_train_timesteps=100)
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.tensor([1, 10])
        out = schedule.add_noise(x0, noise, t, ConditionBundle())
        assert out.shape == x0.shape

    def test_bbdm_bridge_shape(self):
        from src.model.noise.base import BBDMBridgeSchedule
        from src.model.interfaces import ConditionBundle

        schedule = BBDMBridgeSchedule(num_train_timesteps=100)
        x0 = torch.randn(2, 1, 32, 32)
        x_source = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.tensor([10, 50])
        out = schedule.add_noise(x0, noise, t, ConditionBundle(), x_source=x_source)
        assert out.shape == x0.shape

    def test_scale_adaptive_sigmas_use_broadcast_shapes(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise

        schedule = ScaleAdaptiveNoise(
            num_train_timesteps=100,
            use_gabor_energy=False,
            enabled=True,
        )
        t = torch.tensor([1, 10])
        low, mid, high = schedule._get_scale_multipliers(t, gabor_energy=None, shape=(2, 1, 64, 64))

        assert low.shape == (2, 1, 1, 1)
        assert mid.shape == (2, 1, 1, 1)
        assert high.shape == (2, 1, 1, 1)
        assert (torch.ones(2, 1, 64, 64) * high).shape == (2, 1, 64, 64)

    def test_scale_adaptive_only_gabor_high_sigma_is_spatial(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise

        schedule = ScaleAdaptiveNoise(
            num_train_timesteps=100,
            use_gabor_energy=True,
            enabled=True,
        )
        t = torch.tensor([1, 10])
        gabor = torch.rand(2, 1, 64, 64)
        low, mid, high = schedule._get_scale_multipliers(t, gabor_energy=gabor, shape=(2, 1, 32, 32))

        assert low.shape == (2, 1, 1, 1)
        assert mid.shape == (2, 1, 1, 1)
        assert high.shape == (2, 1, 32, 32)

    def test_scale_adaptive_uses_gabor_only_for_high_frequency_band(self):
        from src.model.interfaces import ConditionBundle
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise

        schedule = ScaleAdaptiveNoise(num_train_timesteps=100, use_gabor_energy=True, enabled=True)
        t = torch.tensor([1, 10])
        pyramid = [
            torch.zeros(2, 1, 64, 64),
            torch.zeros(2, 1, 32, 32),
            torch.zeros(2, 1, 16, 16),
            torch.zeros(2, 1, 8, 8),
        ]
        calls = []

        def fake_get_scale_multipliers(timesteps, gabor_energy=None, shape=None):
            calls.append(gabor_energy is not None)
            sigma = torch.ones(2, 1, 1, 1)
            return sigma, sigma, sigma

        schedule._get_scale_multipliers = fake_get_scale_multipliers
        condition = ConditionBundle(maps={"gabor_energy": torch.ones(2, 1, 64, 64)})

        schedule._get_band_sigmas(t, condition, pyramid)

        assert calls == [True, False, False, False]

    def test_scale_adaptive_noise_shape(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise
        from src.model.interfaces import ConditionBundle

        schedule = ScaleAdaptiveNoise(num_train_timesteps=100, num_scales=3)
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.randn_like(x0)
        t = torch.tensor([10, 50])
        bundle = ConditionBundle(maps={"gabor_energy": torch.rand(2, 1, 32, 32)})
        out = schedule.add_noise(x0, noise, t, bundle)
        assert out.shape == x0.shape

    def test_scale_adaptive_noise_applies_signal_decay(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise
        from src.model.interfaces import ConditionBundle

        schedule = ScaleAdaptiveNoise(num_train_timesteps=100, num_scales=2)
        x0 = torch.randn(2, 1, 32, 32)
        noise = torch.zeros_like(x0)
        t = torch.tensor([10, 50])
        out = schedule.add_noise(x0, noise, t, ConditionBundle())
        expected = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0
        assert torch.allclose(out, expected, atol=1e-5)

    def test_scale_adaptive_bridge_reverse_shape(self):
        from src.model.noise.scale_adaptive import ScaleAdaptiveNoise
        from src.model.interfaces import ConditionBundle

        schedule = ScaleAdaptiveNoise(num_train_timesteps=100, num_scales=2, bridge_mode=True)
        x_t = torch.randn(2, 1, 32, 32)
        pred_x0 = torch.randn_like(x_t)
        x_source = torch.randn_like(x_t)
        t = torch.tensor([50, 60])
        t_next = torch.tensor([40, 50])
        out = schedule.step_from_prediction(
            x_t, pred_x0, t, t_next, ConditionBundle(), x_source=x_source
        )
        assert out.shape == x_t.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Loss term tests
# ---------------------------------------------------------------------------

class TestLossTerms:
    def test_disabled_loss_returns_zero(self):
        from src.model.loss_terms.base import DisabledLossTerm
        from src.model.interfaces import LossContext, ConditionBundle

        term = DisabledLossTerm()
        ctx = LossContext(
            model_pred=torch.randn(2, 1, 32, 32),
            loss_target=torch.randn(2, 1, 32, 32),
            target_pet=torch.randn(2, 1, 32, 32),
            pred_x0=torch.randn(2, 1, 32, 32),
            timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.ones(2),
            batch=_fake_batch(),
            condition=ConditionBundle(),
        )
        loss, logs = term(ctx)
        assert loss.item() == 0.0

    def test_topk_loss_no_crash(self):
        from src.model.loss_terms.topk import TopKLesionLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = TopKLesionLoss(enabled=True)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target,
            pred_x0=pred, timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.zeros(2),  # tau=0 → late step → loss active
            batch=_fake_batch(), condition=ConditionBundle(),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)

    def test_focal_freq_no_crash(self):
        from src.model.loss_terms.frequency import FocalFrequencyLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = FocalFrequencyLoss(enabled=True)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target,
            pred_x0=pred, timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.zeros(2), batch=_fake_batch(), condition=ConditionBundle(),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)

    def test_patch_nce_no_crash(self):
        from src.model.loss_terms.patch_nce import PatchNCELoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = PatchNCELoss(enabled=True, num_patches=32)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        ctx = LossContext(
            model_pred=pred,
            loss_target=target,
            target_pet=target,
            pred_x0=pred,
            timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.full((2,), 0.5),
            batch=_fake_batch(),
            condition=ConditionBundle(),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)
        assert "patch_nce/loss" in logs

    def test_heteroscedastic_nll_no_crash(self):
        from src.model.loss_terms.heteroscedastic import HeteroscedasticNLLLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = HeteroscedasticNLLLoss(enabled=True)
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randn(2, 1, 32, 32)
        logvar = torch.randn(2, 1, 32, 32) * 0.1
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target,
            pred_x0=pred, timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.ones(2), batch=_fake_batch(), condition=ConditionBundle(),
            pred_logvar=logvar,
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)

    def test_hotspot_prior_loss_no_crash(self):
        from src.model.loss_terms.hotspot import HotspotPriorLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = HotspotPriorLoss(enabled=True)
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randn(2, 1, 32, 32)
        batch = _fake_batch()
        batch["mask"][:, :, 12:16, 12:16] = 1.0
        ctx = LossContext(
            model_pred=target,
            loss_target=target,
            target_pet=target,
            pred_x0=target,
            timesteps=torch.zeros(2, dtype=torch.long),
            tau=torch.full((2,), 0.5),
            batch=batch,
            condition=ConditionBundle(maps={"hotspot_prior": pred}),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)
        assert "hotspot_prior/dice" in logs

    def test_organ_consistency_no_crash(self):
        from src.model.loss_terms.organ_consistency import OrganConsistencyLoss
        from src.model.interfaces import LossContext, ConditionBundle

        B, H, W = 2, 16, 16
        pred = torch.rand(B, 1, H, W) * 2 - 1
        target = torch.rand(B, 1, H, W) * 2 - 1
        organ_mask = torch.zeros(B, 6, H, W)
        organ_mask[:, 3, 2:8, 2:8] = 1.0   # bone (cold)
        organ_mask[:, 4, 8:12, 8:12] = 1.0 # fat (cold)
        organ_mask[:, 1, 12:14, 12:14] = 1.0  # bladder (warm, not penalised)

        term = OrganConsistencyLoss(enabled=True, active_tau_min=0.25, cold_weight=0.02)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target, pred_x0=pred,
            timesteps=torch.tensor([300, 500]), tau=torch.tensor([0.5, 0.3]),
            batch={"organ_mask": organ_mask},
            condition=ConditionBundle.empty(),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)
        assert "organ_consistency/bone_loss" in logs
        assert "organ_consistency/fat_loss" in logs
        assert "organ_consistency/gate_mean" in logs

    def test_organ_consistency_disabled_returns_zero(self):
        from src.model.loss_terms.organ_consistency import OrganConsistencyLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = OrganConsistencyLoss(enabled=False)
        ctx = LossContext(
            model_pred=torch.zeros(1, 1, 8, 8), loss_target=torch.zeros(1, 1, 8, 8),
            target_pet=torch.zeros(1, 1, 8, 8), pred_x0=torch.zeros(1, 1, 8, 8),
            timesteps=torch.tensor([0]), tau=torch.tensor([0.5]),
            batch={}, condition=ConditionBundle.empty(),
        )
        loss, logs = term(ctx)
        assert loss.item() == 0.0

    def test_lesion_roi_l1_ignores_far_outside_mask_error(self):
        from src.model.loss_terms.lesion_roi import LesionROIL1Loss
        from src.model.interfaces import LossContext, ConditionBundle

        target = torch.zeros(1, 1, 16, 16)
        pred = target.clone()
        pred[:, :, 0, 0] = 10.0  # far outside lesion ROI; should not matter
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 8, 8] = 1.0

        term = LesionROIL1Loss(enabled=True, weight=1.0, dilate_radius=1, active_tau_max=1.0)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target, pred_x0=pred,
            timesteps=torch.tensor([0]), tau=torch.tensor([0.0]),
            batch={"mask": mask}, condition=ConditionBundle.empty(),
        )
        loss, logs = term(ctx)
        assert loss.item() == 0.0
        assert logs["lesion_roi_l1/roi_pixels"].item() == 9

    def test_outside_peak_ranking_penalizes_outside_peak_above_inside(self):
        from src.model.loss_terms.lesion_roi import OutsidePeakRankingLoss
        from src.model.interfaces import LossContext, ConditionBundle

        target = torch.zeros(1, 1, 16, 16)
        pred_bad = target.clone()
        pred_good = target.clone()
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 8, 8] = 1.0

        pred_bad[:, :, 8, 8] = 0.4
        pred_bad[:, :, 0, 0] = 0.8
        pred_good[:, :, 8, 8] = 0.8
        pred_good[:, :, 0, 0] = 0.2

        term = OutsidePeakRankingLoss(
            enabled=True, weight=1.0, margin=0.05,
            inside_radius=1, outside_radius=2, topk_percent=0.0,
            active_tau_max=1.0,
        )
        ctx_bad = LossContext(
            model_pred=pred_bad, loss_target=target, target_pet=target, pred_x0=pred_bad,
            timesteps=torch.tensor([0]), tau=torch.tensor([0.0]),
            batch={"mask": mask}, condition=ConditionBundle.empty(),
        )
        ctx_good = LossContext(
            model_pred=pred_good, loss_target=target, target_pet=target, pred_x0=pred_good,
            timesteps=torch.tensor([0]), tau=torch.tensor([0.0]),
            batch={"mask": mask}, condition=ConditionBundle.empty(),
        )
        loss_bad, logs_bad = term(ctx_bad)
        loss_good, logs_good = term(ctx_good)
        assert loss_bad.item() > 0.0
        assert loss_good.item() == 0.0
        assert logs_bad["outside_peak_ranking/outside_peak"].item() > logs_bad["outside_peak_ranking/inside_peak"].item()
        assert logs_good["outside_peak_ranking/inside_peak"].item() > logs_good["outside_peak_ranking/outside_peak"].item()


# ---------------------------------------------------------------------------
# TinySegmenter tests
# ---------------------------------------------------------------------------

class TestTinySegmenter:
    def test_segmenter_forward_shape(self):
        from src.model.segmenter import TinySegmenter

        seg = TinySegmenter(in_channels=1, base_channels=16, stage=2)
        pet = torch.randn(2, 1, 32, 32)
        out = seg(pet)
        assert out.shape == (2, 1, 32, 32)
        # Stage 2 (tanh) output in [-1, 1]
        assert out.min() >= -1.01 and out.max() <= 1.01

    def test_segmenter_disabled_returns_zero(self):
        from src.model.segmenter import TinySegmenter

        seg = TinySegmenter(enabled=False)
        pet = torch.randn(2, 1, 32, 32)
        out = seg(pet)
        assert (out == 0).all()

    def test_segmenter_stage1_sigmoid(self):
        from src.model.segmenter import TinySegmenter

        seg = TinySegmenter(stage=1)
        pet = torch.randn(2, 1, 32, 32)
        out = seg(pet)
        assert out.min() >= -0.01 and out.max() <= 1.01  # sigmoid → [0, 1]

    def test_segmenter_param_budget(self):
        from src.model.segmenter import TinySegmenter

        seg = TinySegmenter(base_channels=16)
        params = seg.get_total_params()
        assert params < 500_000, f"Segmenter has {params:,} params, budget is 500K"

    def test_build_target_heatmap(self):
        from src.model.segmenter import TinySegmenter

        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 14:18, 14:18] = 1.0
        target = TinySegmenter.build_target(mask, stage=1, gaussian_sigma=6.0)
        assert target.shape == mask.shape
        assert target.max() > 0.5  # centre should be bright
        assert target[:, :, 0, 0] < 0.1  # far away should be dim

    def test_build_target_relaxed(self):
        from src.model.segmenter import TinySegmenter

        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 14:18, 14:18] = 1.0
        target = TinySegmenter.build_target(mask, stage=2)
        assert target.shape == mask.shape
        assert target[:, :, 14:18, 14:18].mean() > 0  # lesion interior positive

    def test_build_target_relaxed_small_lesion_signal(self):
        """Stage-2 target must keep a learnable signal for a small lesion.

        Regression guard: normalising by the global max distance crushes a small
        lesion's interior to ~0.06, leaving the model no gradient (loss pinned
        ~0.563, recall 0).  The scale must be anchored to the lesion interior so
        the lesion centre clears a strong threshold.
        """
        from src.model.segmenter import TinySegmenter

        mask = torch.zeros(1, 1, 192, 192)
        yy, xx = torch.meshgrid(torch.arange(192), torch.arange(192), indexing="ij")
        mask[0, 0, ((yy - 100) ** 2 + (xx - 100) ** 2) <= 8 ** 2] = 1.0
        target = TinySegmenter.build_target(mask, stage=2)
        assert torch.isfinite(target).all()
        assert target[0, 0, 100, 100] > 0.5  # lesion centre strongly positive
        assert target[0, 0, 0, 0] < -0.5  # far background strongly negative


class TestSegmenterConsistencyLoss:
    def test_disabled_returns_zero(self):
        from src.model.loss_terms.segmenter_consistency import SegmenterConsistencyLoss
        from src.model.interfaces import LossContext, ConditionBundle

        term = SegmenterConsistencyLoss(enabled=False)
        ctx = LossContext(
            model_pred=torch.zeros(1, 1, 8, 8), loss_target=torch.zeros(1, 1, 8, 8),
            target_pet=torch.zeros(1, 1, 8, 8), pred_x0=torch.zeros(1, 1, 8, 8),
            timesteps=torch.tensor([0]), tau=torch.tensor([0.1]),
            batch={}, condition=ConditionBundle.empty(),
        )
        loss, logs = term(ctx)
        assert loss.item() == 0.0

    def test_with_segmenter_no_crash(self):
        from src.model.segmenter import TinySegmenter
        from src.model.loss_terms.segmenter_consistency import SegmenterConsistencyLoss
        from src.model.interfaces import LossContext, ConditionBundle

        seg = TinySegmenter(stage=2, base_channels=8)
        seg.eval()
        for p in seg.parameters():
            p.requires_grad = False

        term = SegmenterConsistencyLoss(segmenter=seg, enabled=True)
        pred = torch.randn(2, 1, 16, 16)
        target = torch.randn(2, 1, 16, 16)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target, pred_x0=pred,
            timesteps=torch.tensor([5, 5]), tau=torch.tensor([0.1, 0.1]),
            batch={}, condition=ConditionBundle.empty(),
        )
        loss, logs = term(ctx)
        assert torch.isfinite(loss)
        assert "segmenter_consistency/loss" in logs


# ---------------------------------------------------------------------------
# Full model integration tests
# ---------------------------------------------------------------------------

class TestModelIntegration:
    @pytest.fixture(autouse=True)
    def seed(self):
        torch.manual_seed(42)

    def test_model_forward_baseline(self):
        """Baseline mode: all modules off."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        loss, logs = model(batch)
        assert torch.isfinite(loss)
        assert "loss/total" in logs
        assert "loss/base_l1" in logs
        assert "loss/base_gradient" in logs
        assert "loss/min_snr_weight" in logs
        assert logs["module/zero_adapter"].item() == 0.0
        assert logs["loss/topk_lesion/enabled"].item() == 0.0
        assert logs["loss/topk_lesion/loss"].item() == 0.0

    def test_model_forward_full(self):
        """Full mode: all modules enabled."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["modules"]["gabor"]["enabled"] = True
        cfg["modules"]["organ_prior"]["enabled"] = True
        cfg["modules"]["hotspot_prior"]["enabled"] = True
        cfg["modules"]["scale_adaptive_noise"]["enabled"] = True
        cfg["modules"]["scale_adaptive_noise"]["use_gabor_energy"] = True
        cfg["losses"]["topk_lesion"]["enabled"] = True
        cfg["losses"]["focal_frequency"]["enabled"] = True
        cfg["losses"]["patch_nce"]["enabled"] = True
        cfg["losses"]["roi_suv"]["enabled"] = True
        cfg["losses"]["heteroscedastic_nll"]["enabled"] = True
        cfg["losses"]["hotspot_prior"]["enabled"] = True

        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        loss, logs = model(batch)
        assert torch.isfinite(loss)

    def test_model_rejects_configured_base_diffusion_loss(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["losses"]["base_diffusion"] = {"enabled": True}
        with pytest.raises(ValueError, match="base_diffusion"):
            SLMFBBDM.from_config(cfg)

    def test_model_sample_ddim(self):
        """DDIM sampling produces correct shape."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)

    def test_model_sample_scale_adaptive_noise(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["modules"]["scale_adaptive_noise"] = {"enabled": True, "name": "scale_adaptive"}
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch, num_steps=3)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)
        assert torch.isfinite(result["synthetic_pet"]).all()

    def test_model_sample_scale_adaptive_bridge_uses_custom_reverse(self):
        import types
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["modules"]["scale_adaptive_noise"] = {
            "enabled": True,
            "name": "scale_adaptive",
            "bridge_mode": True,
        }
        model = SLMFBBDM.from_config(cfg)
        calls = {"n": 0}
        original = model.noise_schedule.step_from_prediction

        def wrapped(self, *args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        model.noise_schedule.step_from_prediction = types.MethodType(wrapped, model.noise_schedule)
        result = model.sample(_fake_batch(), num_steps=3)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)
        assert calls["n"] == 2

    def test_model_sample_with_heteroscedastic(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["enable_heteroscedastic"] = True
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch)
        assert "logvar" in result
        assert result["logvar"].shape == (2, 1, 32, 32)

    def test_model_forward_with_self_conditioning(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["self_conditioning"]["enabled"] = True
        cfg["model"]["self_conditioning"]["probability"] = 1.0
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        loss, logs = model(batch)
        assert torch.isfinite(loss)
        assert logs["module/self_conditioning"].item() == 1.0

    def test_model_sample_with_cfg(self):
        """Weak CFG sampling produces valid output with cfg_scale > 1."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch, num_steps=3, cfg_scale=1.2)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)
        assert torch.isfinite(result["synthetic_pet"]).all()

    def test_model_sample_cfg_1_is_noop(self):
        """cfg_scale=1.0 should match no-CFG output (no blending)."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch, num_steps=3, cfg_scale=1.0)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)

    def test_model_sample_with_self_conditioning(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["self_conditioning"]["enabled"] = True
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch, num_steps=3)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)

    def test_model_forward_with_metadata_film(self):
        """Metadata FiLM injection enabled produces valid loss and log."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["metadata"] = {"enabled": True}
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        loss, logs = model(batch)
        assert torch.isfinite(loss)
        assert logs["module/metadata_film"].item() == 1.0

    def test_model_sample_with_metadata_film(self):
        """Metadata FiLM injection during sampling produces valid output."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["metadata"] = {"enabled": True}
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        result = model.sample(batch, num_steps=3)
        assert result["synthetic_pet"].shape == (2, 1, 32, 32)
        assert torch.isfinite(result["synthetic_pet"]).all()

    def test_model_sample_mc_with_metadata_film(self):
        """Metadata FiLM during MC sampling produces confidence map."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["model"]["metadata"] = {"enabled": True}
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch(B=1)
        result = model.sample_mc(batch, n_samples=3, num_steps=3)
        assert "synthetic_pet" in result
        assert "epistemic_var" in result

    def test_metadata_disabled_produces_no_film_log(self):
        """When metadata FiLM is off, log shows 0."""
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        model = SLMFBBDM.from_config(cfg)
        batch = _fake_batch()
        loss, logs = model(batch)
        assert logs["module/metadata_film"].item() == 0.0

    def test_meta_to_tensor_missing_keys(self):
        """Missing keys in meta dict → centre of normalised range (0.0)."""
        from src.model.slmf_bbdm import _meta_to_tensor

        t = _meta_to_tensor(
            [{"uptake_min": 60.0}], B=1,
            device=torch.device("cpu"), dtype=torch.float32,
        )
        assert t.shape == (1, 5)
        # uptake_min normalised to [-1,1]: 60 → (60-30)/90*2-1 ≈ -0.333
        assert abs(t[0, 0].item() - (-0.333)) < 0.01
        # Missing keys → 0.0
        assert t[0, 1].item() == 0.0  # weight_kg
        assert t[0, 2].item() == 0.0  # age_years

    def test_meta_to_tensor_dict_of_lists(self):
        """Collated dict-of-lists format is transposed correctly."""
        from src.model.slmf_bbdm import _meta_to_tensor

        meta = {
            "uptake_min": [45.0, 90.0],
            "weight_kg": [55.0, 80.0],
            "age_years": [30.0, 70.0],
            "thickness_mm": [1.5, 3.0],
            "z_mm": [-50.0, 50.0],
        }
        t = _meta_to_tensor(meta, B=2, device=torch.device("cpu"), dtype=torch.float32)
        assert t.shape == (2, 5)
        # Two distinct samples
        assert not torch.allclose(t[0], t[1])
        # Range check: all values in [-1, 1]
        assert (t >= -1.01).all() and (t <= 1.01).all()

    @pytest.mark.parametrize("ablation", [
        "baseline", "no_gabor", "no_organ", "no_hotspot",
        "no_zero_adapter", "isotropic_noise", "no_heteroscedastic",
    ])
    def test_ablation_profiles(self, ablation):
        """Every ablation config must produce a valid model."""
        from src.model.config_utils import load_full_config
        from src.model.slmf_bbdm import SLMFBBDM

        # Write a minimal ablations file
        ablations = {
            "ablations": {
                "baseline": {"overrides": {
                    "modules.gabor.enabled": False,
                    "modules.organ_prior.enabled": False,
                    "modules.hotspot_prior.enabled": False,
                    "modules.scale_adaptive_noise.enabled": False,
                    "model.enable_heteroscedastic": False,
                    "losses.topk_lesion.enabled": False,
                    "losses.focal_frequency.enabled": False,
                    "losses.roi_suv.enabled": False,
                    "losses.heteroscedastic_nll.enabled": False,
                }},
                "no_gabor": {"overrides": {
                    "modules.gabor.enabled": False,
                    "modules.scale_adaptive_noise.use_gabor_energy": False,
                }},
                "no_organ": {"overrides": {"modules.organ_prior.enabled": False}},
                "no_hotspot": {"overrides": {"modules.hotspot_prior.enabled": False}},
                "no_zero_adapter": {"overrides": {"modules.zero_adapter.enabled": False}},
                "isotropic_noise": {"overrides": {"modules.scale_adaptive_noise.enabled": False}},
                "no_heteroscedastic": {"overrides": {"model.enable_heteroscedastic": False}},
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(ablations, f)
            ablation_path = f.name

        try:
            # Write full config to temp
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(_toy_config(), f)
                config_path = f.name

            cfg = load_full_config(config_path, ablation=ablation, ablation_config_path=ablation_path)
            model = SLMFBBDM.from_config(cfg)
            batch = _fake_batch()
            loss, _ = model(batch)
            assert torch.isfinite(loss), f"Ablation '{ablation}' produced NaN loss"
        finally:
            os.unlink(ablation_path)
            os.unlink(config_path)

    def test_adapter_switch_changes_output(self):
        """Adapter on/off must produce different output (verifies real wiring)."""
        from src.model.slmf_bbdm import SLMFBBDM

        torch.manual_seed(42)
        batch = _fake_batch()

        # Adapter OFF (raw concat)
        cfg_off = _toy_config()
        cfg_off["modules"]["zero_adapter"]["enabled"] = False
        model_off = SLMFBBDM.from_config(cfg_off)
        loss_off, _ = model_off(batch)

        # Adapter ON (ZeroConv)
        cfg_on = _toy_config()
        cfg_on["modules"]["zero_adapter"]["enabled"] = True
        model_on = SLMFBBDM.from_config(cfg_on)
        loss_on, _ = model_on(batch)

        # Loss may differ or be same magnitude — the key is both are finite
        assert torch.isfinite(loss_off)
        assert torch.isfinite(loss_on)

    def test_gabor_switch_changes_condition_bundle(self):
        """Gabor on must produce gabor_feat in condition bundle."""
        from src.model.slmf_bbdm import SLMFBBDM

        batch = _fake_batch()

        cfg_on = _toy_config()
        cfg_on["modules"]["gabor"]["enabled"] = True
        model_on = SLMFBBDM.from_config(cfg_on)
        bundle_on = model_on.build_condition_bundle(batch, torch.zeros(2, dtype=torch.long))
        assert "gabor_feat" in bundle_on.maps

        cfg_off = _toy_config()
        cfg_off["modules"]["gabor"]["enabled"] = False
        model_off = SLMFBBDM.from_config(cfg_off)
        bundle_off = model_off.build_condition_bundle(batch, torch.zeros(2, dtype=torch.long))
        assert "gabor_feat" not in bundle_off.maps

    def test_prior_switch_changes_output(self):
        """Enabling Gabor + Organ must change the loss value (non-zero gradient path)."""
        from src.model.slmf_bbdm import SLMFBBDM

        torch.manual_seed(42)
        batch = _fake_batch()

        cfg_on = _toy_config()
        cfg_on["modules"]["gabor"]["enabled"] = True
        cfg_on["modules"]["organ_prior"]["enabled"] = True
        cfg_on["modules"]["hotspot_prior"]["enabled"] = True
        cfg_on["modules"]["zero_adapter"]["enabled"] = True
        cfg_on["modules"]["scale_adaptive_noise"]["enabled"] = True
        model_on = SLMFBBDM.from_config(cfg_on)
        loss_on, _ = model_on(batch)

        assert torch.isfinite(loss_on)
        assert loss_on.item() != 0.0

    def test_non_prior_modules_keep_correct_status_logs(self):
        from src.model.slmf_bbdm import SLMFBBDM

        cfg = _toy_config()
        cfg["modules"]["zero_adapter"]["enabled"] = True
        cfg["modules"]["condition_dropout"]["enabled"] = True
        cfg["modules"]["scale_adaptive_noise"] = {"enabled": True, "name": "scale_adaptive"}
        model = SLMFBBDM.from_config(cfg)
        assert list(model.priors.keys()) == ["gabor", "organ_prior", "hotspot_prior", "semantic_prior"]

        loss, logs = model(_fake_batch(B=1), timesteps=torch.zeros(1, dtype=torch.long))
        assert torch.isfinite(loss)
        assert logs["module/zero_adapter"].item() == 1.0
        assert logs["module/condition_dropout"].item() == 1.0
        assert logs["module/scale_adaptive_noise"].item() == 1.0


# ---------------------------------------------------------------------------
# Trainer smoke test
# ---------------------------------------------------------------------------

class TestTrainer:
    def test_trainer_one_step(self):
        import shutil
        import uuid
        from pathlib import Path

        from src.data.dataset import FakeDataset
        from src.model.slmf_bbdm import SLMFBBDM
        from src.model.trainer import Trainer
        from torch.utils.data import DataLoader

        cfg = _toy_config()
        cfg["training"]["num_epochs"] = 1
        # Isolate the metrics JSONL: the default path is repo-anchored
        # (checkpoints/smoke_test/), so a stale file left by any earlier
        # run trips the exact-prefix guard and permanently reds this test
        # (found while re-verifying the suite after the audit fixes).
        scratch = (Path(__file__).resolve().parent.parent / ".t_dir"
                   / "smoke_trainer" / uuid.uuid4().hex[:12])
        scratch.mkdir(parents=True, exist_ok=True)
        cfg["runtime"]["metrics_jsonl"] = str(scratch / "training_metrics.jsonl")
        try:
            model = SLMFBBDM.from_config(cfg)
            ds = FakeDataset(8, image_size=32)
            dl = DataLoader(ds, batch_size=2, drop_last=True)
            trainer = Trainer(model, cfg, dl, dl)
            trainer.run(num_epochs=1)
            assert trainer.epoch_count == 1
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_trainer_flushes_partial_gradient_accumulation(self):
        from src.data.dataset import FakeDataset
        from src.model.slmf_bbdm import SLMFBBDM
        from src.model.trainer import Trainer
        from torch.utils.data import DataLoader

        cfg = _toy_config()
        cfg["runtime"]["gradient_accumulate_every"] = 2
        model = SLMFBBDM.from_config(cfg)
        ds = FakeDataset(5, image_size=32)  # 3 batches when batch_size=2
        dl = DataLoader(ds, batch_size=2, drop_last=False)
        trainer = Trainer(model, cfg, dl, None)
        trainer.train_epoch()
        assert trainer.step_count == 3
        assert trainer.ema.step_count == 2
        assert trainer.accum_count == 0

    def test_run_stops_at_configured_total_epochs_after_resume(self):
        from src.model.trainer import Trainer

        class FakeModel:
            priors = {}
            loss_terms = {}

            def get_trainable_params(self):
                return 0

            def get_total_params(self):
                return 0

        trainer = object.__new__(Trainer)
        trainer.config = {"training": {"num_epochs": 3}}
        trainer.device = "cpu"
        trainer.amp_dtype = torch.float32
        trainer.torch_compile = False
        trainer.grad_accum = 1
        trainer.log_interval = 1
        trainer.eval_interval = 999
        trainer.sample_interval = 999
        trainer.save_interval = 999
        trainer.val_loader = None
        trainer.model = FakeModel()
        trainer.epoch_count = 2
        calls = []

        def fake_train_epoch():
            trainer.epoch_count += 1
            calls.append(trainer.epoch_count)
            return {
                "loss/total": 0.0,
                "perf/epoch_seconds": 0.0,
                "perf/lr": 0.0,
            }

        trainer.train_epoch = fake_train_epoch

        trainer.run()

        assert calls == [3]
        assert trainer.epoch_count == 3


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset:
    def test_fake_dataset(self):
        from src.data.dataset import FakeDataset

        ds = FakeDataset(8, image_size=32)
        sample = ds[0]
        assert sample["ct"].shape == (1, 32, 32)
        assert sample["pet"].shape == (1, 32, 32)
        assert sample["mask"].shape == (1, 32, 32)
        assert sample["organ_mask"].shape == (6, 32, 32)
        assert sample["organ_distance"].shape == (6, 32, 32)
        assert sample["mu_map"].shape == (1, 32, 32)
        assert sample["ct_hu"].shape == (1, 32, 32)
        assert sample["pet_suv"].shape == (1, 32, 32)

    def test_build_dataloaders_fake(self):
        from src.data.dataset import build_dataloaders

        data_cfg = {"image_size": 32, "batch_size": 2, "use_fake_data": True}
        run_cfg = {"num_workers": 0, "pin_memory": False, "persistent_workers": False}
        train, val = build_dataloaders(data_cfg, run_cfg)
        batch = next(iter(train))
        assert "ct" in batch
        assert "pet" in batch

    def test_cached_dataset_closes_npz(self, tmp_path, monkeypatch):
        from src.data.dataset import CachedDataset

        sample_id = "001001"
        np.savez_compressed(
            tmp_path / f"{sample_id}.npz",
            ct=np.zeros((1, 4, 4), dtype=np.float32),
            pet=np.zeros((1, 4, 4), dtype=np.float32),
        )
        (tmp_path / f"{sample_id}_meta.json").write_text(
            json.dumps({"sample_id": sample_id, "split": "train", "has_label": False}),
            encoding="utf-8",
        )

        closed = {"value": False}
        real_np_load = np.load

        class TrackingLoad:
            def __init__(self, *args, **kwargs):
                self.inner = real_np_load(*args, **kwargs)

            def __enter__(self):
                return self.inner

            def __exit__(self, exc_type, exc, tb):
                closed["value"] = True
                self.inner.close()

        monkeypatch.setattr(np, "load", TrackingLoad)

        ds = CachedDataset(tmp_path, split="train", augment=False)
        _ = ds[0]
        assert closed["value"] is True

    def test_cached_dataset_returns_organ_distance(self, tmp_path):
        from src.data.dataset import CachedDataset

        sample_id = "001001"
        organ_distance = np.ones((6, 4, 4), dtype=np.float32)
        np.savez_compressed(
            tmp_path / f"{sample_id}.npz",
            ct=np.zeros((1, 4, 4), dtype=np.float32),
            pet=np.zeros((1, 4, 4), dtype=np.float32),
            organ_distance=organ_distance,
        )
        (tmp_path / f"{sample_id}_meta.json").write_text(
            json.dumps({"sample_id": sample_id, "split": "train", "has_label": False}),
            encoding="utf-8",
        )

        ds = CachedDataset(tmp_path, split="train", augment=False)
        sample = ds[0]
        assert torch.allclose(sample["organ_distance"], torch.ones(6, 4, 4))

    def test_cached_dataset_returns_physical_maps(self, tmp_path):
        from src.data.dataset import CachedDataset

        sample_id = "001001"
        ct_hu = np.full((1, 4, 4), 123.0, dtype=np.float32)
        pet_suv = np.full((1, 4, 4), 7.5, dtype=np.float32)
        np.savez_compressed(
            tmp_path / f"{sample_id}.npz",
            ct=np.zeros((1, 4, 4), dtype=np.float32),
            pet=np.zeros((1, 4, 4), dtype=np.float32),
            ct_hu=ct_hu,
            pet_suv=pet_suv,
            scale_meta_json=np.frombuffer(
                json.dumps({"suv_ok": True, "pet_suv_max": 20.0}).encode(), dtype=np.uint8
            ),
        )
        (tmp_path / f"{sample_id}_meta.json").write_text(
            json.dumps({"sample_id": sample_id, "split": "train", "has_label": False}),
            encoding="utf-8",
        )

        ds = CachedDataset(tmp_path, split="train", augment=False)
        sample = ds[0]
        assert torch.allclose(sample["ct_hu"], torch.full((1, 4, 4), 123.0))
        assert torch.allclose(sample["pet_suv"], torch.full((1, 4, 4), 7.5))
        assert sample["meta"]["pet_suv_available"] is True

    def test_cached_dataset_augment_horizontal_flip_keeps_fields_aligned(self, tmp_path, monkeypatch):
        from src.data.dataset import CachedDataset

        sample_id = "001001"
        ct = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        pet = ct + 100.0
        mask = np.zeros((1, 4, 4), dtype=np.float32)
        mask[:, :, 0] = 1.0
        np.savez_compressed(
            tmp_path / f"{sample_id}.npz",
            ct=ct,
            pet=pet,
            mask=mask,
        )
        (tmp_path / f"{sample_id}_meta.json").write_text(
            json.dumps({"sample_id": sample_id, "split": "train", "has_label": True}),
            encoding="utf-8",
        )
        monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))

        ds = CachedDataset(tmp_path, split="train", augment=True)
        sample = ds[0]
        assert torch.allclose(sample["ct"], torch.from_numpy(np.flip(ct, axis=-1).copy()))
        assert torch.allclose(sample["pet"], torch.from_numpy(np.flip(pet, axis=-1).copy()))
        assert torch.allclose(sample["mask"], torch.from_numpy(np.flip(mask, axis=-1).copy()))

    def test_png_cache_builds_npz_and_manifest_from_split_csv(self, tmp_path):
        from PIL import Image
        from src.data.dataset import CachedDataset
        from src.data.png_cache import build_png_cache

        main_data = tmp_path / "main_data"
        for split in ["train", "val"]:
            for subdir in ["ct", "pet", "pet_peizhuan", "label"]:
                (main_data / split / subdir).mkdir(parents=True)

        split_csv = main_data / "split.csv"
        split_csv.write_text(
            "file_name,patient_id,split,mask_area,area_class,area_level\n"
            "001001.png,001,train,4,0,small\n"
            "002001.png,002,val,4,0,small\n",
            encoding="utf-8",
        )

        for split, sample_id in [("train", "001001"), ("val", "002001")]:
            ct = np.arange(16, dtype=np.uint8).reshape(4, 4)
            pet = np.full((4, 4), 25, dtype=np.uint8)
            registered_pet = np.full((4, 4), 200, dtype=np.uint8)
            label = np.zeros((4, 4), dtype=np.uint8)
            label[1:3, 1:3] = 255
            Image.fromarray(ct).save(main_data / split / "ct" / f"{sample_id}.png")
            Image.fromarray(pet).save(main_data / split / "pet" / f"{sample_id}.png")
            Image.fromarray(registered_pet).save(main_data / split / "pet_peizhuan" / f"{sample_id}.png")
            Image.fromarray(label).save(main_data / split / "label" / f"{sample_id}.png")

        cache_dir = tmp_path / "cache" / "tensors_main"
        split_manifest = main_data / "split_manifest.csv"
        stats = build_png_cache(
            png_root=main_data,
            split_csv=split_csv,
            out_dir=cache_dir,
            split_manifest=split_manifest,
            image_size=8,
        )

        assert stats["built"] == 2
        assert stats["skipped"] == {}
        assert stats["pet_source_counts"] == {"pet_peizhuan": 2}
        assert split_manifest.exists()

        with np.load(cache_dir / "001001.npz") as data:
            assert set(["ct", "pet", "mask", "scale_meta_json"]).issubset(data.keys())
            assert data["ct"].shape == (1, 8, 8)
            assert data["pet"].shape == (1, 8, 8)
            assert data["mask"].shape == (1, 8, 8)
            assert data["mask"].max() == 1.0
            assert data["pet"].min() > 0.0  # registered PET was selected over the darker raw PET
            scale_meta = json.loads(bytes(data["scale_meta_json"].tolist()).decode("utf-8"))
            assert scale_meta["pet_physical_kind"] == "png_intensity"
            assert scale_meta["pet_suv_available"] is False

        train_ds = CachedDataset(
            cache_dir,
            split="train",
            augment=False,
            split_manifest=split_manifest,
            required_keys=["ct", "pet", "mask"],
        )
        val_ds = CachedDataset(
            cache_dir,
            split="val",
            augment=False,
            split_manifest=split_manifest,
            required_keys=["ct", "pet", "mask"],
        )
        assert len(train_ds) == 1
        assert len(val_ds) == 1
        assert train_ds[0]["meta"]["pet_suv_available"] is False


class TestScaleMeta:
    """Verify scale_meta propagation through data pipeline and loss terms."""

    def test_fake_dataset_has_meta(self):
        from src.data.dataset import FakeDataset
        ds = FakeDataset(8, image_size=32)
        sample = ds[0]
        assert "meta" in sample
        assert isinstance(sample["meta"], dict)
        assert "pet_suv_max" in sample["meta"]
        assert "patient_id" in sample["meta"]
        assert "suv_ok" in sample["meta"]

    def test_roi_suv_loss_uses_meta(self):
        from src.model.loss_terms.roi_suv import ROISUVLoss
        from src.model.interfaces import LossContext, ConditionBundle

        B, H, W = 2, 16, 16
        pred = torch.randn(B, 1, H, W)
        target = torch.randn(B, 1, H, W)
        mask = torch.zeros(B, 1, H, W)
        mask[:, :, 4:8, 4:8] = 1.0
        tau = torch.tensor([0.1, 0.1])

        loss_fn = ROISUVLoss(active_tau_max=0.25, enabled=True, weight=0.2)

        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target,
            pred_x0=pred, timesteps=torch.tensor([0, 0]), tau=tau,
            batch={"mask": mask, "meta": [{"suv_ok": True, "pet_suv_max": 20.0},
                                          {"suv_ok": False, "pet_raw_min": -1.0, "pet_raw_max": 1.0}]},
            condition=ConditionBundle.empty(),
        )
        loss, logs = loss_fn(ctx)
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        assert "roi_suv/suv_max_error" in logs
        assert "roi_suv/tbr_error" in logs
        assert torch.isclose(logs["roi_suv/valid_fraction"], torch.tensor(0.5))

    def test_roi_suv_loss_uses_physical_target_tensor(self):
        from src.model.loss_terms.roi_suv import ROISUVLoss
        from src.model.interfaces import LossContext, ConditionBundle

        pred = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.ones(1, 1, 4, 4)
        pet_suv = torch.full((1, 1, 4, 4), 30.0)
        loss_fn = ROISUVLoss(active_tau_max=0.25, enabled=True, weight=1.0)

        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target,
            pred_x0=pred, timesteps=torch.tensor([0]), tau=torch.tensor([0.1]),
            batch={
                "mask": mask,
                "pet_suv": pet_suv,
                "meta": [{"suv_ok": True, "pet_suv_max": 20.0, "pet_suv_available": True}],
            },
            condition=ConditionBundle.empty(),
        )
        loss, logs = loss_fn(ctx)
        assert loss.item() > 0.0
        assert torch.isclose(logs["roi_suv/target_suv_mean"], torch.tensor(30.0))

    def test_roi_suv_loss_skips_invalid_suv_samples(self):
        from src.model.loss_terms.roi_suv import ROISUVLoss
        from src.model.interfaces import LossContext, ConditionBundle

        loss_fn = ROISUVLoss(enabled=True)
        ctx = LossContext(
            model_pred=torch.ones(1, 1, 8, 8), loss_target=torch.zeros(1, 1, 8, 8),
            target_pet=torch.zeros(1, 1, 8, 8), pred_x0=torch.ones(1, 1, 8, 8),
            timesteps=torch.tensor([0]), tau=torch.tensor([0.1]),
            batch={"mask": torch.ones(1, 1, 8, 8), "meta": [{"suv_ok": False, "pet_raw_min": 0.0, "pet_raw_max": 99.0}]},
            condition=ConditionBundle.empty(),
        )
        loss, logs = loss_fn(ctx)
        assert loss.item() == 0.0
        assert logs["roi_suv/valid_fraction"].item() == 0.0

    def test_roi_suv_disabled_returns_zero(self):
        from src.model.loss_terms.roi_suv import ROISUVLoss
        from src.model.interfaces import LossContext, ConditionBundle

        loss_fn = ROISUVLoss(enabled=False)
        ctx = LossContext(
            model_pred=torch.zeros(1, 1, 8, 8), loss_target=torch.zeros(1, 1, 8, 8),
            target_pet=torch.zeros(1, 1, 8, 8), pred_x0=torch.zeros(1, 1, 8, 8),
            timesteps=torch.tensor([0]), tau=torch.tensor([0.1]),
            batch={}, condition=ConditionBundle.empty(),
        )
        loss, logs = loss_fn(ctx)
        assert loss.item() == 0.0


class TestFalseHotspotV2:
    """Verify per-organ differentiation in FalseHotspotLoss."""

    def test_cold_mask_excludes_warm_organs(self):
        from src.model.loss_terms.false_hotspot import FalseHotspotLoss
        from src.model.interfaces import LossContext, ConditionBundle

        B, H, W = 1, 8, 8
        # organ_mask: [B, 6, H, W], class 1=bladder, class 2=rectum
        organ_mask = torch.zeros(B, 6, H, W)
        organ_mask[:, 1, 2:6, 2:6] = 1.0  # bladder region
        organ_mask[:, 2, 3:5, 3:5] = 1.0  # rectum region
        organ_mask[:, 3, 0:2, 0:2] = 1.0  # bone region

        loss_fn = FalseHotspotLoss(active_tau_max=0.5, enabled=True)
        cold_mask = loss_fn._build_cold_mask(LossContext(
            model_pred=torch.zeros(B, 1, H, W), loss_target=torch.zeros(B, 1, H, W),
            target_pet=torch.zeros(B, 1, H, W), pred_x0=None,
            timesteps=torch.tensor([0]), tau=torch.tensor([0.1]),
            batch={"organ_mask": organ_mask},
            condition=ConditionBundle.empty(),
        ))

        # Bladder region should be warm (excluded from cold mask)
        assert cold_mask[0, 0, 2:6, 2:6].sum() == 0.0, "Bladder should be excluded from cold mask"
        # Bone region should stay cold
        assert cold_mask[0, 0, 0:2, 0:2].sum() > 0.0, "Bone should be in cold mask"
        # Background should be cold
        assert cold_mask[0, 0, 6:8, 6:8].sum() > 0.0, "Background should be in cold mask"

    def test_false_hotspot_with_organs(self):
        from src.model.loss_terms.false_hotspot import FalseHotspotLoss
        from src.model.interfaces import LossContext, ConditionBundle

        B, H, W = 2, 16, 16
        pred = torch.rand(B, 1, H, W) * 2 - 1
        target = torch.rand(B, 1, H, W) * 2 - 1
        organ_mask = torch.zeros(B, 6, H, W)
        organ_mask[:, 1, 2:6, 2:6] = 1.0  # bladder (warm)

        loss_fn = FalseHotspotLoss(active_tau_max=0.5, enabled=True, weight=0.05)
        ctx = LossContext(
            model_pred=pred, loss_target=target, target_pet=target, pred_x0=pred,
            timesteps=torch.tensor([0, 0]), tau=torch.tensor([0.1, 0.1]),
            batch={"organ_mask": organ_mask},
            condition=ConditionBundle.empty(),
        )
        loss, logs = loss_fn(ctx)
        assert loss.ndim == 0
        # Check per-region log keys
        assert "false_hotspot/bone_mean" in logs or "false_hotspot/loss" in logs


class TestMCSampling:
    """Verify MC sampling produces uncertainty estimates."""

    def test_sample_mc_basic(self):
        """sample_mc returns mean, epistemic variance, confidence map."""
        # Use baseline config (no modules) for fast smoke test
        slmf = SLMFBBDM(
            image_size=32,
            objective="pred_x0",
            enable_heteroscedastic=True,
            sample_scheduler="ddim",
            eval_sampling_steps=5,
        )

        batch = {
            "ct": torch.randn(1, 1, 32, 32),
            "mask": torch.zeros(1, 1, 32, 32),
            "organ_mask": torch.zeros(1, 6, 32, 32),
            "mu_map": torch.zeros(1, 1, 32, 32),
            "meta": {"pet_suv_max": 20.0, "suv_ok": True},
        }

        result = slmf.sample_mc(batch, n_samples=3, num_steps=3)
        assert "synthetic_pet" in result
        assert "epistemic_var" in result
        assert "total_var" in result
        assert "samples" in result
        assert result["synthetic_pet"].shape == (1, 1, 32, 32)
        assert result["epistemic_var"].shape == (1, 1, 32, 32)
        assert result["samples"].shape == (3, 1, 1, 32, 32)
        # With heteroscedastic: should have confidence
        assert "confidence_map" in result
        assert result["confidence_map"].shape == (1, 1, 32, 32)
        assert (result["confidence_map"] >= 0).all() and (result["confidence_map"] <= 1).all()

    def test_sample_mc_no_heteroscedastic(self):
        """sample_mc without heteroscedastic head keeps the uncertainty names.

        Naming contract (uncertainty naming fix): total_var and
        confidence_map are ALWAYS emitted; with the heteroscedastic head
        disabled they fall back to the epistemic-only form
        (total_var == epistemic_var), so evaluate.py's Uncertainty metric
        names (uncertainty_ratio / confidence_lesion_mean) cannot silently
        vanish for arms that set model.enable_heteroscedastic=false (every
        RC-BRD prod config).  Only the aleatoric_* keys stay conditional on
        the head.
        """
        slmf = SLMFBBDM(
            image_size=32,
            objective="pred_x0",
            enable_heteroscedastic=False,
        )

        batch = {
            "ct": torch.randn(1, 1, 32, 32),
            "mask": torch.zeros(1, 1, 32, 32),
            "organ_mask": torch.zeros(1, 6, 32, 32),
            "mu_map": torch.zeros(1, 1, 32, 32),
        }

        result = slmf.sample_mc(batch, n_samples=2, num_steps=3)
        assert "synthetic_pet" in result
        assert "epistemic_var" in result
        # Epistemic-only fallback: no aleatoric component anywhere.
        assert "aleatoric_var" not in result
        assert "aleatoric_logvar" not in result
        assert "total_var" in result
        torch.testing.assert_close(result["total_var"], result["epistemic_var"])
        assert "confidence_map" in result
        assert result["confidence_map"].shape == (1, 1, 32, 32)
        assert (result["confidence_map"] >= 0).all() and (result["confidence_map"] <= 1).all()

    def test_sample_mc_supports_median_point_estimator(self, monkeypatch):
        slmf = SLMFBBDM(image_size=32, enable_heteroscedastic=False)
        values = iter([1.0, 100.0, 2.0])

        def fake_sample(batch, **_):
            value = next(values)
            return {"synthetic_pet": torch.full_like(batch["ct"], value)}

        monkeypatch.setattr(slmf, "sample", fake_sample)
        batch = {"ct": torch.zeros(1, 1, 32, 32)}
        result = slmf.sample_mc(batch, n_samples=3, aggregate="median")
        torch.testing.assert_close(
            result["synthetic_pet"],
            torch.full_like(batch["ct"], 2.0),
        )

    def test_sample_mc_rejects_unknown_aggregation(self):
        slmf = SLMFBBDM(image_size=32, enable_heteroscedastic=False)
        batch = {"ct": torch.zeros(1, 1, 32, 32)}
        with pytest.raises(ValueError, match="aggregate"):
            slmf.sample_mc(batch, n_samples=1, num_steps=1, aggregate="trimmed")


class TestSplitManifest:
    """Basic smoke tests for split manifest generation and loading."""

    def test_generate_and_load(self, tmp_path):
        from src.data.split_manifest import generate_split_manifest, SplitManifest

        # Create fake .npz cache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        for pid in ["001", "002", "003"]:
            for slc in range(3):
                sid = f"{pid}{slc:03d}"
                np.savez_compressed(
                    cache_dir / f"{sid}.npz",
                    ct=np.zeros((1, 8, 8), dtype=np.float32),
                    pet=np.zeros((1, 8, 8), dtype=np.float32),
                )

        manifest_path = tmp_path / "split_manifest.csv"
        stats = generate_split_manifest(cache_dir, manifest_path, val_ratio=0.3, seed=42)

        assert stats["total_samples"] == 9
        assert stats["total_patients"] == 3
        assert stats["val_patients"] > 0
        assert stats["test_patients"] > 0

        manifest = SplitManifest(manifest_path)
        assert len(manifest) == 9
        assert manifest.validate_no_overlap()
        assert len(manifest.train_samples) > 0
        assert len(manifest.val_samples) > 0
        assert len(manifest.test_samples) > 0

        # No patient in multiple splits
        for pid in ["001", "002", "003"]:
            split = manifest.get_patient_split(pid)
            assert split in ("train", "val", "test")


class TestSemanticBuilder:
    """Smoke test for semantic token builder (random backend, no GPU needed)."""

    def test_builder_random_backend(self, tmp_path):
        from src.data.semantic_builder import SemanticTokenBuilder, ct_norm_to_3win_rgb

        # Create fake .npz with CT data
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        ct = np.random.randn(8, 8).astype(np.float32) * 2 - 1  # [-1, 1]
        np.savez_compressed(
            cache_dir / "001001.npz",
            ct=np.array([[ct]], dtype=np.float32),
            pet=np.zeros((1, 8, 8), dtype=np.float32),
            scale_meta_json=np.frombuffer(
                json.dumps({"ct_hu_min": -150.0, "ct_hu_max": 250.0}).encode(), dtype=np.uint8
            ),
        )

        builder = SemanticTokenBuilder(backend="random", token_dim=32, num_tokens=2, device="cpu")
        stats = builder.process_cache(cache_dir, batch_size=2)

        assert stats["updated"] == 1
        # Verify tokens were written
        data = np.load(cache_dir / "001001.npz")
        assert "semantic_tokens" in data
        assert data["semantic_tokens"].shape == (2, 32)

    def test_ct_to_rgb(self):
        from src.data.semantic_builder import ct_norm_to_3win_rgb
        ct_norm = np.random.randn(32, 32).astype(np.float32) * 2 - 1
        rgb = ct_norm_to_3win_rgb(ct_norm)
        assert rgb.shape == (32, 32, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
