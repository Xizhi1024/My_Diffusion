"""Unit tests for artifact-gated block screening and promotion."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


GATES = {
    "failure_any_mean": {"direction": "lower", "max_delta": 0.0625},
    "false_hotspot_density_mean": {
        "direction": "lower",
        "max_ratio": 1.25,
        "epsilon": 1e-6,
    },
    "stripe_excess_mean": {"direction": "lower", "max_delta": 0.05},
    "ssim_mean": {"direction": "higher", "max_delta": 0.03},
}


def _metrics(**updates):
    values = {
        "lesion_peak_error_norm_mean": 0.08,
        "lesion_centroid_distance_mean": 3.0,
        "failure_any_mean": 0.18,
        "false_hotspot_density_mean": 1e-4,
        "stripe_excess_mean": 0.02,
        "ssim_mean": 0.95,
        "mae_mean": 0.06,
        "lesion_boundary_gradient_mae_norm_mean": 0.08,
        "anatomy_edge_gradient_mae_norm_mean": 0.07,
        "directional_spectrum_error_norm_mean": 0.02,
    }
    values.update(updates)
    return values


def test_metric_value_accepts_flat_evaluator_keys_and_rejects_missing():
    from scripts.run_frequency_ablations import metric_value

    assert metric_value({"ssim_mean": 0.95}, "ssim_mean") == pytest.approx(0.95)
    with pytest.raises(KeyError, match="ssim_mean"):
        metric_value({}, "ssim_mean")


def test_hard_gates_compare_candidates_to_r0_reference():
    from scripts.run_frequency_ablations import passes_hard_gates

    reference = _metrics()
    safe = _metrics(failure_any_mean=0.20, ssim_mean=0.93)
    unsafe = _metrics(failure_any_mean=0.30, stripe_excess_mean=0.20)
    assert passes_hard_gates(safe, reference, GATES)[0]
    passed, reasons = passes_hard_gates(unsafe, reference, GATES)
    assert not passed
    assert any("failure_any_mean" in reason for reason in reasons)
    assert any("stripe_excess_mean" in reason for reason in reasons)


def test_composite_score_prefers_lesion_fidelity_without_ignoring_image_quality():
    from scripts.run_frequency_ablations import composite_score

    good = _metrics(
        lesion_peak_error_norm_mean=0.04,
        lesion_centroid_distance_mean=2.0,
        failure_any_mean=0.10,
        ssim_mean=0.96,
    )
    bad = _metrics(
        lesion_peak_error_norm_mean=0.20,
        lesion_centroid_distance_mean=8.0,
        failure_any_mean=0.30,
        ssim_mean=0.97,
    )
    assert composite_score(good) > composite_score(bad)


def test_composite_score_rewards_boundary_and_directional_fidelity():
    from scripts.run_frequency_ablations import composite_score

    good = _metrics(
        lesion_boundary_gradient_mae_norm_mean=0.02,
        anatomy_edge_gradient_mae_norm_mean=0.03,
        directional_spectrum_error_norm_mean=0.01,
    )
    bad = _metrics(
        lesion_boundary_gradient_mae_norm_mean=0.20,
        anatomy_edge_gradient_mae_norm_mean=0.18,
        directional_spectrum_error_norm_mean=0.12,
    )
    assert composite_score(good) > composite_score(bad)


def test_select_promotions_excludes_reference_and_applies_top_k():
    from scripts.run_frequency_ablations import select_promotions

    records = [
        {"id": "R0", "metrics": _metrics()},
        {"id": "R2", "metrics": _metrics(lesion_peak_error_norm_mean=0.06)},
        {"id": "F1", "checkpoint_epoch": 40, "metrics": _metrics(lesion_peak_error_norm_mean=0.04)},
        {"id": "F2", "metrics": _metrics(failure_any_mean=0.40)},
    ]
    promoted, ranked = select_promotions(
        records, reference_id="R0", gates=GATES, top_k=1
    )
    assert promoted == ["F1"]
    assert ranked[0]["id"] == "F1"
    assert ranked[0]["checkpoint_epoch"] == 40
    rejected = next(row for row in ranked if row["id"] == "F2")
    assert rejected["gate_passed"] is False


def test_command_builders_pin_experiment_seed_subset_ema_and_fresh_training():
    from scripts.run_frequency_ablations import (
        build_eval_command,
        build_mean_pretrain_command,
        build_train_command,
    )

    train = build_train_command(
        python="python",
        config="base.yaml",
        ablation_config="ablations.yaml",
        preset="freq_f3",
        experiment="freq_screen_f3",
        epochs=50,
        seed=42,
        eval_interval=10,
    )
    command_text = " ".join(train)
    assert "scripts/train_v2.py" in command_text
    assert "--ablation freq_f3" in command_text
    assert "experiment.name=freq_screen_f3" in command_text
    assert "training.num_epochs=50" in command_text
    assert "training.resume_from=null" in command_text
    assert "runtime.early_stopping.enabled=false" in command_text

    evaluate = build_eval_command(
        python="python",
        config="resolved.yaml",
        checkpoint="ckpt_best_combined.pt",
        output="result.json",
        split="val",
        max_samples=16,
        seed=42,
        mc_steps=20,
    )
    eval_text = " ".join(evaluate)
    assert "--weights ema" in eval_text
    assert "--max-samples 16" in eval_text
    assert "--seed 42" in eval_text
    assert "--mc-steps 20" in eval_text

    mean = build_mean_pretrain_command(
        python="python",
        config="base.yaml",
        output_dir="checkpoints/freq_mean_v2",
        epochs=30,
        seed=42,
        overrides={"data.num_workers": 2},
    )
    mean_text = " ".join(mean)
    assert "scripts/pretrain_conditional_mean.py" in mean_text
    assert "--output-dir checkpoints/freq_mean_v2" in mean_text
    assert "--epochs 30" in mean_text
    assert "data.num_workers=2" in mean_text


def test_dry_run_manifest_contains_all_screen_blocks_and_no_promotions_yet():
    from scripts.run_frequency_ablations import build_execution_manifest

    plan = {
        "base_config": "base.yaml",
        "ablation_config": "ablations.yaml",
        "output_dir": "results/frequency_ablations",
        "variants": [
            {"id": "R0", "preset": "freq_r0"},
            {"id": "R2", "preset": "freq_r2"},
            {"id": "F1", "preset": "freq_f1"},
            {"id": "F2", "preset": "freq_f2"},
            {"id": "F3", "preset": "freq_f3"},
            {"id": "F4", "preset": "freq_f4"},
        ],
        "screen": {
            "epochs": 50,
            "eval_interval": 10,
            "max_samples": 16,
            "seed": 42,
            "mc_steps": 20,
            "experiment_prefix": "freq_screen",
        },
    }
    manifest = build_execution_manifest(plan, python="python", stage="screen")
    assert [run["id"] for run in manifest["screen_runs"]] == [
        "R0", "R2", "F1", "F2", "F3", "F4"
    ]
    assert all(run["train_command"] and run["eval_command"] for run in manifest["screen_runs"])
    assert manifest["promotion_runs"] == []
    assert manifest["mean_run"] is None


def test_v2_manifest_runs_mean_first_and_applies_common_train_overrides():
    from scripts.run_frequency_ablations import build_execution_manifest

    plan = {
        "base_config": "base.yaml",
        "ablation_config": "ablations.yaml",
        "output_dir": "results/frequency_ablations_v2",
        "mean_pretrain": {
            "enabled": True,
            "experiment": "freq_mean_pretrain_v2",
            "output_dir": "checkpoints/freq_mean_pretrain_v2",
            "checkpoint": "checkpoints/freq_mean_pretrain_v2/mean_best.pt",
            "epochs": 30,
            "seed": 42,
        },
        "common_train_overrides": {
            "model.initialization_seed": 4242,
            "modules.conditional_mean.checkpoint": "checkpoints/freq_mean_pretrain_v2/mean_best.pt",
            "modules.conditional_mean.freeze": True,
            "modules.conditional_mean.loss_weight": 0.0,
        },
        "variants": [{"id": "R0", "preset": "freq_r0"}],
        "screen": {
            "epochs": 50,
            "eval_interval": 10,
            "max_samples": 16,
            "seed": 42,
            "mc_steps": 20,
            "experiment_prefix": "freq_v2_screen",
        },
    }
    manifest = build_execution_manifest(plan, python="python", stage="all")
    assert manifest["mean_run"]["checkpoint"].endswith("mean_best.pt")
    assert "scripts/pretrain_conditional_mean.py" in " ".join(
        manifest["mean_run"]["command"]
    )
    train_text = " ".join(manifest["screen_runs"][0]["train_command"])
    assert "model.initialization_seed=4242" in train_text
    assert "modules.conditional_mean.freeze=true" in train_text
    assert "modules.conditional_mean.loss_weight=0.0" in train_text


def test_v2_plan_keeps_v1_outputs_separate_and_freezes_shared_mean():
    from pathlib import Path

    import yaml

    plan_path = Path("configs/experiments/frequency_ablation_plan_v2.yaml")
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)

    assert plan["output_dir"] == "results/frequency_ablations_v2"
    assert plan["mean_pretrain"]["epochs"] == 30
    assert plan["common_train_overrides"]["model.initialization_seed"] == 4242
    assert plan["common_train_overrides"]["modules.conditional_mean.freeze"] is True
    assert plan["common_train_overrides"]["modules.conditional_mean.loss_weight"] == 0.0
    assert plan["screen"]["experiment_prefix"] == "freq_v2_screen"
    assert plan["promote"]["experiment_prefix"] == "freq_v2_full"


def test_v3_plan_reuses_frozen_mean_and_pins_fixed_promotions():
    import yaml

    plan = yaml.safe_load(
        Path("configs/experiments/boundary_reliable_ablation_plan_v3.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert plan["reference_id"] == "R2"
    assert plan["output_dir"] == "results/boundary_reliable_ablations_v3"
    assert plan["mean_pretrain"]["enabled"] is False
    assert plan["mean_pretrain"]["checkpoint"].endswith(
        "freq_mean_pretrain_v2/mean_best.pt"
    )
    assert plan["common_train_overrides"]["modules.conditional_mean.freeze"] is True
    assert plan["screen"]["epochs"] == 50
    assert plan["screen"]["eval_interval"] == 10
    assert plan["promote"]["epochs"] == 300
    assert plan["promote"]["eval_interval"] == 20
    assert plan["promote"]["early_stopping"] is False
    assert plan["top_k"] == 2
    assert [variant["id"] for variant in plan["variants"]] == [
        "R2", "F2", "F4", "BR-B", "BR-C", "BR-D", "BR-E", "BR-F"
    ]


def test_v3_promotion_command_explicitly_disables_early_stopping():
    import yaml

    from scripts.run_frequency_ablations import build_execution_manifest

    plan = yaml.safe_load(
        Path("configs/experiments/boundary_reliable_ablation_plan_v3.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_execution_manifest(
        plan, python="python", stage="promote", promoted_ids=["BR-F"]
    )
    command = " ".join(manifest["promotion_runs"][0]["train_command"])
    assert "training.num_epochs=300" in command
    assert "runtime.eval_interval=20" in command
    assert "runtime.early_stopping.enabled=false" in command


def test_checkpoint_epoch_requires_integer_epoch_metadata(tmp_path):
    import torch

    from scripts.run_frequency_ablations import checkpoint_epoch

    valid = tmp_path / "valid.pt"
    missing = tmp_path / "missing.pt"
    torch.save({"epoch": 120}, valid)
    torch.save({"model": {}}, missing)
    assert checkpoint_epoch(valid) == 120
    with pytest.raises(ValueError, match="epoch"):
        checkpoint_epoch(missing)


def test_ranking_json_sanitizes_unavailable_clinical_nan(tmp_path):
    import json
    import math

    from scripts.run_frequency_ablations import _write_rankings

    ranked = [{
        "id": "F1",
        "preset": "freq_f1",
        "gate_passed": True,
        "gate_reasons": [],
        "score": 0.5,
        "metrics": {**_metrics(), "suv_calib_slope": float("nan")},
    }]
    _write_rankings(tmp_path, ["F1"], ranked)
    payload = json.loads((tmp_path / "promotion_decision.json").read_text(encoding="utf-8"))
    assert payload["ranked"][0]["metrics"]["suv_calib_slope"] is None


@pytest.mark.parametrize(
    ("preset", "mode", "inject_wavelet", "ct_reliability", "content", "subbands", "direction", "boundary"),
    [
        ("br_r2", None, None, None, None, None, None, False),
        ("br_f2", "legacy", False, None, None, None, None, False),
        ("br_f4", "legacy", True, None, None, None, None, False),
        ("br_b", "boundary_reliable", True, False, False, False, False, False),
        ("br_c", "boundary_reliable", True, True, False, False, False, False),
        ("br_d", "boundary_reliable", True, True, True, True, False, False),
        ("br_e", "boundary_reliable", True, True, True, True, True, False),
        ("br_f", "boundary_reliable", True, True, True, True, True, True),
    ],
)
def test_boundary_reliable_presets_are_distinct_and_keep_wavelet_backbone_off(
    preset, mode, inject_wavelet, ct_reliability, content, subbands, direction, boundary
):
    from src.model.config_utils import load_full_config, validate_png_baseline_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_boundary_reliable.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    validate_png_baseline_config(cfg)
    model = SLMFBBDM.from_config(cfg)
    assert model.wavelet_unet_enabled is False
    if mode is None:
        assert model.residual_frequency_enabled is False
    else:
        assert model.residual_frequency_mode == mode
        assert model.residual_preconditioner.inject_wavelet is inject_wavelet
    if mode == "boundary_reliable":
        assert model.residual_preconditioner.use_ct_reliability is ct_reliability
        assert model.residual_preconditioner.use_content_reliability is content
        assert model.residual_preconditioner.use_subband_gates is subbands
        assert model.residual_preconditioner.use_directional_reliability is direction
    assert model.loss_terms["boundary_frequency"].enabled is boundary
    assert model.loss_terms["frequency_gate_tv"].enabled is boundary


@pytest.mark.parametrize(
    "preset",
    ["br_r2", "br_f2", "br_f4", "br_b", "br_c", "br_d", "br_e", "br_f"],
)
def test_every_boundary_reliable_preset_completes_a_real_forward(preset):
    import torch

    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_boundary_reliable.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    model = SLMFBBDM.from_config(cfg)
    batch = {
        "ct": torch.randn(1, 1, 32, 32),
        "pet": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
    }
    batch["mask"][:, :, 14:18, 14:18] = 1
    loss, _ = model(batch, timesteps=torch.tensor([250]))
    assert torch.isfinite(loss)


def test_zero_initialized_br_b_is_initially_equivalent_to_paired_r2():
    import torch

    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    def build(preset):
        cfg = load_full_config(
            "configs/experiments/slmf_png_boundary_reliable.yaml",
            ablation=preset,
            ablation_config_path="configs/experiments/ablations.yaml",
            overrides=[
                "data.image_size=32",
                "model.initialization_seed=4242",
                "runtime.eval_sampling_steps=2",
            ],
        )
        torch.manual_seed(123)
        return SLMFBBDM.from_config(cfg).eval()

    r2 = build("br_r2")
    br_b = build("br_b")
    batch = {
        "ct": torch.randn(1, 1, 32, 32),
        "pet": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
    }
    timestep = torch.tensor([250])
    torch.manual_seed(999)
    r2_loss, _ = r2(batch, timesteps=timestep)
    torch.manual_seed(999)
    br_b_loss, _ = br_b(batch, timesteps=timestep)

    assert torch.allclose(r2_loss, br_b_loss, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    ("preset", "mean", "bridge", "frequency", "inject_wavelet", "gabor_gate"),
    [
        ("freq_r0", False, False, False, None, None),
        ("freq_r2", True, True, False, None, None),
        ("freq_f1", True, True, True, True, False),
        ("freq_f2", True, True, True, False, True),
        ("freq_f3", True, True, True, True, True),
        ("freq_f4", True, True, True, True, True),
    ],
)
def test_frequency_presets_resolve_to_valid_distinct_models(
    preset, mean, bridge, frequency, inject_wavelet, gabor_gate
):
    from src.model.config_utils import load_full_config, validate_png_baseline_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_residual_frequency.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
    )
    validate_png_baseline_config(cfg)
    model = SLMFBBDM.from_config(cfg)
    assert model.conditional_mean_enabled is mean
    assert model.residual_bridge_enabled is bridge
    assert model.residual_frequency_enabled is frequency
    if frequency:
        assert model.residual_preconditioner.inject_wavelet is inject_wavelet
        assert model.residual_preconditioner.use_gabor_gate is gabor_gate
    if preset == "freq_f4":
        assert model.loss_terms["residual_wavelet"].enabled
        assert model.loss_terms["gabor_consistency"].enabled


@pytest.mark.parametrize("preset", ["freq_r0", "freq_r2", "freq_f1", "freq_f2", "freq_f3", "freq_f4"])
def test_every_frequency_preset_completes_a_real_model_forward(preset):
    import torch

    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_residual_frequency.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    model = SLMFBBDM.from_config(cfg)
    batch = {
        "ct": torch.randn(1, 1, 32, 32),
        "pet": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
    }
    batch["mask"][:, :, 14:18, 14:18] = 1
    loss, _ = model(batch, timesteps=torch.tensor([250]))
    assert torch.isfinite(loss)
    if preset == "freq_f2":
        sampled = model.sample(batch, num_steps=2)["synthetic_pet"]
        assert sampled.shape == batch["pet"].shape
        assert torch.isfinite(sampled).all()
