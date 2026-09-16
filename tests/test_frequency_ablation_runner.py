"""Unit tests for artifact-gated block screening and promotion."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# The v4 preset tests build the full production model whose frozen
# conditional-mean checkpoint is a local/cloud research artifact that is
# deliberately not part of the git tree.  Skip (not fail) on code-only
# checkouts; machines that do have the checkpoint still verify the presets.
_V4_MEAN_CHECKPOINT = Path("checkpoints/freq_mean_pretrain_v2/mean_best.pt")
requires_v4_mean_checkpoint = pytest.mark.skipif(
    not _V4_MEAN_CHECKPOINT.is_file(),
    reason=(
        "requires the pretrained conditional-mean checkpoint "
        f"({_V4_MEAN_CHECKPOINT}), which is not part of the git checkout"
    ),
)

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


def test_hard_gates_support_absolute_limits_without_reference_metric():
    from scripts.run_frequency_ablations import passes_hard_gates

    gates = {
        "failure_any_mean": {"direction": "lower", "max_value": 0.25},
        "ssim_mean": {"direction": "higher", "min_value": 0.90},
    }
    assert passes_hard_gates(
        {"failure_any_mean": 0.20, "ssim_mean": 0.94}, {}, gates
    )[0]
    passed, reasons = passes_hard_gates(
        {"failure_any_mean": 0.30, "ssim_mean": 0.89}, {}, gates
    )
    assert not passed
    assert len(reasons) == 2


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


def test_v4_composite_score_uses_robust_topq_peak_when_available():
    from scripts.run_frequency_ablations import composite_score

    common = _metrics(lesion_peak_error_norm_mean=0.05)
    good = {**common, "lesion_topq_peak_error_norm_mean": 0.02}
    bad = {**common, "lesion_topq_peak_error_norm_mean": 0.20}

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
    assert "runtime.eval_interval=10" in command_text
    assert "runtime.save_interval=10" in command_text
    assert "runtime.sample_interval=10" in command_text

    decoupled_train = build_train_command(
        python="python",
        config="base.yaml",
        ablation_config="ablations.yaml",
        preset="freq_f3",
        experiment="freq_promote_f3",
        epochs=300,
        seed=42,
        eval_interval=20,
        save_interval=50,
        sample_interval=20,
    )
    decoupled_text = " ".join(decoupled_train)
    assert "runtime.eval_interval=20" in decoupled_text
    assert "runtime.save_interval=50" in decoupled_text
    assert "runtime.sample_interval=20" in decoupled_text

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


def test_manifest_merges_variant_overrides_and_requires_final_checkpoint():
    from scripts.run_frequency_ablations import build_execution_manifest

    plan = {
        "base_config": "base.yaml",
        "ablation_config": "ablations.yaml",
        "output_dir": "results/v4",
        "common_train_overrides": {"model.initialization_seed": 4242},
        "variants": [{
            "id": "G2",
            "preset": "br_v4_d3",
            "overrides": {"modules.residual_frequency.use_gabor_agreement": True},
        }],
        "promote": {
            "epochs": 300,
            "eval_interval": 20,
            "max_samples": 64,
            "seed": 42,
            "mc_steps": 20,
            "experiment_prefix": "br_v4_full",
            "require_final_checkpoint": True,
        },
    }
    manifest = build_execution_manifest(
        plan, python="python", stage="promote", promoted_ids=["G2"]
    )
    run = manifest["promotion_runs"][0]
    command = " ".join(run["train_command"])

    assert "model.initialization_seed=4242" in command
    assert "modules.residual_frequency.use_gabor_agreement=true" in command
    assert run["completion_checkpoint"].endswith("ckpt_epoch0300.pt")


def test_run_entries_rejects_training_without_required_final_checkpoint(
    tmp_path, monkeypatch
):
    import json

    from scripts import run_frequency_ablations as runner

    best = tmp_path / "ckpt_best_combined.pt"
    final = tmp_path / "ckpt_epoch0300.pt"
    result = tmp_path / "result.json"

    def fake_execute(command):
        best.write_bytes(b"best")

    monkeypatch.setattr(runner, "_execute", fake_execute)
    entry = {
        "id": "G2",
        "preset": "br_v4_d3",
        "experiment": "br_v4_full_g2",
        "checkpoint": str(best),
        "completion_checkpoint": str(final),
        "result": str(result),
        "train_command": ["train"],
        "eval_command": ["evaluate"],
    }
    result.write_text(json.dumps(_metrics()), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="final checkpoint"):
        runner._run_entries([entry], force=False)


def test_v4_plan_has_two_fixed_64_sample_stages_and_exact_promotions():
    import yaml

    plan = yaml.safe_load(
        Path("configs/experiments/boundary_reliable_ablation_plan_v4.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert plan["output_dir"] == "results/boundary_reliable_ablations_v4"
    assert [row["id"] for row in plan["stage_a"]["variants"]] == [
        "D0", "D1", "D3", "D4"
    ]
    assert plan["stage_a"]["max_samples"] == 64
    assert plan["stage_b"]["max_samples"] == 64
    assert plan["promote"]["epochs"] == 300
    assert plan["promote"]["early_stopping"] is False
    assert plan["promote"]["require_final_checkpoint"] is True
    assert plan["stage_b"]["top_k"] == 2


@requires_v4_mean_checkpoint
@pytest.mark.parametrize(
    ("preset", "peak", "floor_l2", "floor_l1"),
    [
        ("br_v4_d0", False, 0.0, 0.0),
        ("br_v4_d1", True, 0.0, 0.0),
        ("br_v4_d3", True, 0.25, 0.50),
        ("br_v4_d4", True, 0.0, 1.0),
    ],
)
def test_v4_d_presets_change_only_peak_and_ct_floor_routes(
    preset, peak, floor_l2, floor_l1
):
    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_boundary_reliable_v4.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=["data.image_size=32", "runtime.eval_sampling_steps=2"],
    )
    model = SLMFBBDM.from_config(cfg)

    assert model.loss_terms["normalized_lesion_peak"].enabled is peak
    assert model.residual_preconditioner.ct_reliability_floors.tolist() == pytest.approx(
        [floor_l2, floor_l1]
    )
    assert model.residual_preconditioner.use_gabor_agreement is False
    assert model.residual_preconditioner.use_directional_reliability is False


def test_v4_stage_b_templates_inherit_selected_d_and_keep_g_routes_distinct():
    import yaml

    from scripts.run_boundary_reliable_v4 import build_stage_b_variants

    plan = yaml.safe_load(
        Path("configs/experiments/boundary_reliable_ablation_plan_v4.yaml").read_text(
            encoding="utf-8"
        )
    )
    variants = build_stage_b_variants(plan, selected_d_id="D4")

    assert [row["id"] for row in variants] == ["G0", "G1", "G2"]
    assert {row["preset"] for row in variants} == {"br_v4_d4"}
    routes = {row["id"]: row["overrides"] for row in variants}
    assert routes["G0"]["modules.residual_frequency.use_directional_reliability"] is False
    assert routes["G0"]["modules.residual_frequency.use_gabor_agreement"] is False
    assert routes["G1"]["modules.residual_frequency.use_directional_reliability"] is True
    assert routes["G1"]["modules.residual_frequency.use_gabor_agreement"] is False
    assert routes["G2"]["modules.residual_frequency.use_directional_reliability"] is False
    assert routes["G2"]["modules.residual_frequency.use_gabor_agreement"] is True


def test_v4_dry_run_builds_four_stage_a_and_three_dynamic_stage_b_runs():
    import yaml

    from scripts.run_boundary_reliable_v4 import build_v4_dry_run_manifest

    plan = yaml.safe_load(
        Path("configs/experiments/boundary_reliable_ablation_plan_v4.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_v4_dry_run_manifest(plan, python="python")

    assert len(manifest["stage_a_runs"]) == 4
    assert len(manifest["stage_b_runs"]) == 3
    assert len(manifest["promotion_runs"]) == 2
    paths = str(manifest)
    assert "boundary_reliable_ablations_v4" in paths
    assert "boundary_reliable_ablations_v3" not in paths


def test_v5_plan_pins_two_stages_gates_and_exact_final_promotion():
    import yaml

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert plan["base_config"] == (
        "configs/experiments/slmf_png_spectral_router_v5.yaml"
    )
    assert plan["ablation_config"] == "configs/experiments/ablations.yaml"
    assert plan["output_dir"] == "results/spectral_router_ablations_v5"
    assert plan["mean_pretrain"] == {
        "enabled": False,
        "checkpoint": "checkpoints/freq_mean_pretrain_v2/mean_best.pt",
    }
    assert plan["common_train_overrides"] == {
        "model.initialization_seed": 4242,
        "modules.conditional_mean.checkpoint": (
            "checkpoints/freq_mean_pretrain_v2/mean_best.pt"
        ),
        "modules.conditional_mean.freeze": True,
        "modules.conditional_mean.loss_weight": 0.0,
    }

    stage_a = plan["stage_a"]
    assert stage_a["reference_id"] == "S0"
    assert stage_a["top_k"] == 1
    assert stage_a["dry_run_selected_evidence"] == "S3"
    assert [row["id"] for row in stage_a["variants"]] == ["S0", "S1", "S2", "S3"]
    assert [row["preset"] for row in stage_a["variants"]] == [
        "sr_v5_s0",
        "sr_v5_s1",
        "sr_v5_s2",
        "sr_v5_s3",
    ]

    stage_b = plan["stage_b"]
    assert stage_b["reference_id"] == "T_native"
    assert stage_b["top_k"] == 3
    assert [row["id"] for row in stage_b["variants"]] == [
        "N0", "T_legacy", "T_native", "T_fixed", "C1", "C_no_null",
    ]
    assert all(
        row["overrides"]["modules.residual_frequency.mode"]
        == "spectral_evidence_router"
        for row in stage_b["variants"]
    )
    expected_gates = {
        "lesion_topq_peak_error_norm_mean": {
            "direction": "lower",
            "max_value": 0.080,
        },
        "lesion_peak_error_norm_mean": {"direction": "lower", "max_value": 0.20},
        "lesion_centroid_distance_mean": {
            "direction": "lower",
            "max_delta": 0.25,
        },
        "directional_spectrum_error_norm_mean": {
            "direction": "lower",
            "max_delta": 0.005,
        },
        "failure_any_mean": {
            "direction": "lower",
            "max_value": 0.0625,
            "max_delta": 0.0625,
        },
        "false_hotspot_density_mean": {
            "direction": "lower",
            "max_ratio": 1.25,
            "epsilon": 1e-6,
        },
        "stripe_excess_mean": {
            "direction": "lower",
            "max_value": 0.20,
            "max_delta": 0.05,
        },
        "mae_mean": {"direction": "lower", "max_delta": 0.002},
        "ssim_mean": {
            "direction": "higher",
            "max_delta": 0.03,
        },
    }
    assert stage_b["hard_gates"] == expected_gates

    for stage in (stage_a, stage_b):
        assert stage["epochs"] == 50
        assert stage["eval_interval"] == 10
        assert stage["split"] == "val"
        assert stage["max_samples"] == 64
        assert stage["seed"] == 42
        assert stage["mc_steps"] == 20
        assert stage["early_stopping"] is False

    promote = plan["promote"]
    assert promote == {
        "experiment_prefix": "sr_v5_full",
        "epochs": 300,
        "eval_interval": 20,
        "save_interval": 50,
        "sample_interval": 20,
        "split": "val",
        "max_samples": 64,
        "seed": 42,
        "mc_steps": 20,
        "early_stopping": False,
        "require_final_checkpoint": True,
        "trajectory_checkpoints": [50, 100, 150, 200, 250, 300],
    }
    assert plan["paired_comparison"] == {
        "seed": 42,
        "resamples": 10000,
        "metrics": {
            "small_lesion_topq_peak_error_norm_mean": "lower",
            "small_lesion_signed_bias_mean": "lower",
            "small_lesion_underestimate_rate": "lower",
            "lesion_topq_peak_error_norm_mean": "lower",
            "lesion_peak_error_norm_mean": "lower",
            "lesion_mean_error_norm_mean": "lower",
            "lesion_centroid_distance_mean": "lower",
            "lesion_boundary_gradient_mae_norm_mean": "lower",
            "anatomy_edge_gradient_mae_norm_mean": "lower",
            "directional_spectrum_error_norm_mean": "lower",
            "failure_any_mean": "lower",
            "false_hotspot_density_mean": "lower",
            "stripe_excess_mean": "lower",
            "mae_mean": "lower",
            "ssim_mean": "higher",
        },
    }


def test_v5_stage_b_variants_inherit_evidence_and_keep_six_routes_distinct():
    import yaml

    from scripts.run_spectral_router_v5 import (
        CROSS_ENABLED_KEY,
        FIXED_PRIOR_KEY,
        HARD_NULL_KEY,
        POLICY_KEY,
        build_stage_b_variants,
    )

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    variants = build_stage_b_variants(plan, selected_evidence_id="S3")

    assert [row["id"] for row in variants] == [
        "N0", "T_legacy", "T_native", "T_fixed", "C1", "C_no_null",
    ]
    assert {row["preset"] for row in variants} == {"sr_v5_s3"}
    assert {row["inherited_evidence"] for row in variants} == {"S3"}
    routes = {row["id"]: row["overrides"] for row in variants}
    assert len({
        json.dumps(route, sort_keys=True)
        for route in routes.values()
    }) == 6
    assert all(
        route["modules.residual_frequency.mode"] == "spectral_evidence_router"
        for route in routes.values()
    )
    # N0: hard all-null
    assert routes["N0"][CROSS_ENABLED_KEY] is True
    assert routes["N0"][HARD_NULL_KEY] is True
    # T_legacy: old cross_level_enabled=false
    assert routes["T_legacy"][CROSS_ENABLED_KEY] is False
    assert routes["T_legacy"][HARD_NULL_KEY] is False
    # T_native: native_only policy
    assert routes["T_native"][CROSS_ENABLED_KEY] is True
    assert routes["T_native"][POLICY_KEY] == "native_only"
    # T_fixed: fixed_prior policy
    assert routes["T_fixed"][CROSS_ENABLED_KEY] is True
    assert routes["T_fixed"][POLICY_KEY] == "fixed_prior"
    assert routes["T_fixed"][FIXED_PRIOR_KEY] == [0.05, 0.05, 0.90]
    # C1: learned policy — DCT/Gabor descriptors are inherited from the
    # Stage A evidence preset, not duplicated in overrides.
    assert routes["C1"][CROSS_ENABLED_KEY] is True
    assert routes["C1"][HARD_NULL_KEY] is False
    assert routes["C1"][POLICY_KEY] == "learned"
    assert (
        "modules.residual_frequency.dct_descriptor.enabled"
        not in routes["C1"]
    )
    assert (
        "modules.residual_frequency.gabor_descriptor.enabled"
        not in routes["C1"]
    )
    # C_no_null: learned_no_null policy
    assert routes["C_no_null"][CROSS_ENABLED_KEY] is True
    assert routes["C_no_null"][POLICY_KEY] == "learned_no_null"
    assert routes["C_no_null"][FIXED_PRIOR_KEY] == [0.5, 0.5]


def test_v5_stage_b_s0_keeps_identical_evidence_across_all_routes():
    import yaml

    from scripts.run_spectral_router_v5 import build_stage_b_variants

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    variants = build_stage_b_variants(plan, selected_evidence_id="S0")

    assert [row["id"] for row in variants] == [
        "N0", "T_legacy", "T_native", "T_fixed", "C1", "C_no_null",
    ]
    routes = {row["id"]: row["overrides"] for row in variants}
    assert len({
        json.dumps(route, sort_keys=True)
        for route in routes.values()
    }) == 6
    assert all(
        route["modules.residual_frequency.mode"] == "spectral_evidence_router"
        for route in routes.values()
    )
    for selected_evidence_id in ("S0", "S1", "S2", "S3"):
        inherited = build_stage_b_variants(
            plan, selected_evidence_id=selected_evidence_id
        )
        for row in inherited:
            assert (
                "modules.residual_frequency.dct_descriptor.enabled"
                not in row["overrides"]
            )
            assert (
                "modules.residual_frequency.gabor_descriptor.enabled"
                not in row["overrides"]
            )


def test_v5_evidence_identity_namespaces_stage_b_and_promotion_artifacts():
    import yaml

    from scripts.run_spectral_router_v5 import _build_entries, build_stage_b_variants

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )

    manifests = {}
    for evidence_id in ("S1", "S3"):
        variants = build_stage_b_variants(plan, evidence_id)
        manifests[evidence_id] = {
            "stage_b": _build_entries(
                plan,
                variants,
                plan["stage_b"],
                "python",
                "stage_b",
                evidence_id=evidence_id,
            ),
            "promote": _build_entries(
                plan,
                variants[:1],
                plan["promote"],
                "python",
                "promote",
                evidence_id=evidence_id,
            ),
        }

    for phase in ("stage_b", "promote"):
        s1 = manifests["S1"][phase][0]
        s3 = manifests["S3"][phase][0]
        assert s1["id"] == s3["id"] == "N0"
        for key in ("experiment", "checkpoint", "resolved_config", "result"):
            assert s1[key] != s3[key]
            assert "evidence-s1" in s1[key]
            assert "evidence-s3" in s3[key]
    assert (
        manifests["S1"]["promote"][0]["completion_checkpoint"]
        != manifests["S3"]["promote"][0]["completion_checkpoint"]
    )


def test_v5_stage_b_promotion_excludes_reference_and_non_promotable_routes():
    from scripts.run_spectral_router_v5 import _select_eligible_stage_b_routes

    ranked = [
        {"id": "T_native", "score": 10.0, "gate_passed": True},
        {"id": "C1", "score": 9.0, "gate_passed": True},
        {"id": "N0", "score": 8.0, "gate_passed": True},
        {"id": "T_legacy", "score": 7.0, "gate_passed": True},
        {"id": "T_fixed", "score": 6.0, "gate_passed": True},
        {"id": "C_no_null", "score": 5.0, "gate_passed": True},
    ]

    promoted = _select_eligible_stage_b_routes(
        ranked, reference_id="T_native", top_k=3
    )

    # reference excluded, N0 and T_legacy excluded (non-promotable)
    assert "T_native" not in promoted
    assert "N0" not in promoted
    assert "T_legacy" not in promoted
    # C1, T_fixed, C_no_null are eligible
    assert promoted == ["C1", "T_fixed", "C_no_null"]
    assert len(promoted) <= 3


def test_v5_promotion_always_includes_reference_and_caps_candidates():
    from scripts.run_spectral_router_v5 import _promotion_route_ids

    assert _promotion_route_ids(
        [], reference_id="T_native", top_k=3
    ) == {"T_native"}
    assert _promotion_route_ids(
        ["C1", "N0", "T_fixed", "C_no_null", "extra"],
        reference_id="T_native",
        top_k=3,
    ) == {"T_native", "C1", "T_fixed", "C_no_null"}


def test_v5_stage_b_screen_gates_use_matched_reference_for_image_quality():
    import yaml

    from scripts.run_frequency_ablations import passes_hard_gates

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml")
        .read_text(encoding="utf-8")
    )
    gates = plan["stage_b"]["hard_gates"]
    reference = {
        "lesion_topq_peak_error_norm_mean": 0.1230,
        "lesion_peak_error_norm_mean": 0.0726,
        "lesion_centroid_distance_mean": 4.1420,
        "directional_spectrum_error_norm_mean": 0.0107,
        "failure_any_mean": 0.3438,
        "false_hotspot_density_mean": 0.000651,
        "stripe_excess_mean": -0.0290,
        "mae_mean": 0.0516,
        "ssim_mean": 0.9058,
    }
    c1_like_candidate = {
        "lesion_topq_peak_error_norm_mean": 0.0587,
        "lesion_peak_error_norm_mean": 0.0920,
        "lesion_centroid_distance_mean": 4.1009,
        "directional_spectrum_error_norm_mean": 0.0152,
        "failure_any_mean": 0.015625,
        "false_hotspot_density_mean": 0.000588,
        "stripe_excess_mean": 0.0198,
        "mae_mean": 0.0419,
        "ssim_mean": 0.9267,
    }

    passed, reasons = passes_hard_gates(
        c1_like_candidate, reference, gates
    )
    assert passed is True
    assert reasons == []

    unsafe_candidate = dict(c1_like_candidate, failure_any_mean=0.25)
    passed, reasons = passes_hard_gates(unsafe_candidate, reference, gates)
    assert passed is False
    assert any("failure_any_mean" in reason for reason in reasons)


def test_v5_exact_epoch_validation_reads_checkpoint_metadata(tmp_path):
    import torch

    from scripts.run_spectral_router_v5 import validate_checkpoint_epoch

    wrong = tmp_path / "ckpt_epoch0300.pt"
    exact = tmp_path / "another" / "ckpt_epoch0300.pt"
    exact.parent.mkdir()
    torch.save({"epoch": 299, "model": {"weight": torch.ones(1)}}, wrong)
    torch.save({"epoch": 300}, exact)

    with pytest.raises(ValueError, match="expected exact epoch 300.*found 299"):
        validate_checkpoint_epoch(wrong, expected_epoch=300)
    assert validate_checkpoint_epoch(exact, expected_epoch=300) == 300


def test_v5_exact_epoch_validation_happens_before_evaluation(tmp_path, monkeypatch):
    import torch

    import scripts.run_frequency_ablations as shared_runner
    from scripts.run_spectral_router_v5 import _validate_promotion_checkpoint

    checkpoint = tmp_path / "ckpt_epoch0300.pt"
    result = tmp_path / "result.json"
    torch.save({"epoch": 120}, checkpoint)
    entry = {
        "id": "C1",
        "preset": "sr_v5_s3",
        "experiment": "sr_v5_full_evidence-s3_c1",
        "checkpoint": str(checkpoint),
        "completion_checkpoint": str(checkpoint),
        "required_checkpoint_epoch": 300,
        "result": str(result),
        "train_command": ["train"],
        "eval_command": ["evaluate"],
    }
    executed = []
    monkeypatch.setattr(shared_runner, "_execute", executed.append)

    with pytest.raises(ValueError, match="expected exact epoch 300"):
        shared_runner._run_entries(
            [entry],
            force=False,
            checkpoint_validator=_validate_promotion_checkpoint,
        )

    assert executed == []
    assert not result.exists()


def test_v5_stage_b_decision_rejects_mixed_evidence_provenance(tmp_path):
    import json
    import yaml

    from scripts.run_spectral_router_v5 import _load_stage_b_decision

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    plan["output_dir"] = str(tmp_path)
    decision_dir = tmp_path / "stage_b" / "evidence-s1"
    decision_dir.mkdir(parents=True)
    (decision_dir / "promotion_decision.json").write_text(
        json.dumps({
            "selected_evidence_id": "S3",
            "promoted": ["C1"],
            "ranked": [],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        _load_stage_b_decision(plan, tmp_path, "S1")


def test_v5_dry_run_manifest_is_exact_deterministic_and_v5_only():
    import yaml

    from scripts.run_spectral_router_v5 import build_v5_dry_run_manifest

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_v5_dry_run_manifest(plan, python="python")

    assert manifest["selected_evidence_template"] == "S3"
    assert len(manifest["stage_a_runs"]) == 4
    assert len(manifest["stage_b_runs"]) == 6
    assert len(manifest["promotion_runs"]) == 4  # T_native + up to 3 candidates
    assert [row["id"] for row in manifest["stage_a_runs"]] == [
        "S0",
        "S1",
        "S2",
        "S3",
    ]
    assert [row["id"] for row in manifest["stage_b_runs"]] == [
        "N0",
        "T_legacy",
        "T_native",
        "T_fixed",
        "C1",
        "C_no_null",
    ]
    # Promotion always includes T_native reference in dry-run
    promo_ids = [row["id"] for row in manifest["promotion_runs"]]
    assert "T_native" in promo_ids
    assert "N0" not in promo_ids
    assert "T_legacy" not in promo_ids
    manifest_text = str(manifest)
    assert "spectral_router_ablations_v5" in manifest_text
    assert "boundary_reliable_ablations_v4" not in manifest_text
    assert "boundary_reliable_ablations_v3" not in manifest_text
    for run in (
        manifest["stage_a_runs"]
        + manifest["stage_b_runs"]
        + manifest["promotion_runs"]
    ):
        train_text = " ".join(run["train_command"])
        eval_text = " ".join(run["eval_command"])
        assert "model.initialization_seed=4242" in train_text
        assert (
            "modules.conditional_mean.checkpoint="
            "checkpoints/freq_mean_pretrain_v2/mean_best.pt"
        ) in train_text
        assert "modules.conditional_mean.freeze=true" in train_text
        assert "--max-samples 64" in eval_text
        assert "--seed 42" in eval_text
        assert "--mc-steps 20" in eval_text
    for run in manifest["promotion_runs"]:
        train_text = " ".join(run["train_command"])
        assert "training.num_epochs=300" in train_text
        assert run["checkpoint"] == run["completion_checkpoint"]
        assert run["checkpoint"].endswith("ckpt_epoch0300.pt")


@pytest.mark.parametrize(
    ("stage", "expected_counts"),
    [
        ("stage-a", (4, 0, 0)),
        ("stage-b", (0, 6, 0)),
        ("promote", (0, 0, 4)),
        ("all", (4, 6, 4)),
    ],
)
def test_v5_dry_run_manifest_respects_requested_stage(stage, expected_counts):
    import yaml

    from scripts.run_spectral_router_v5 import build_v5_dry_run_manifest

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_ablation_plan_v5.yaml").read_text(
            encoding="utf-8"
        )
    )

    manifest = build_v5_dry_run_manifest(plan, python="python", stage=stage)

    assert (
        len(manifest["stage_a_runs"]),
        len(manifest["stage_b_runs"]),
        len(manifest["promotion_runs"]),
    ) == expected_counts


def test_v5_final_comparison_requires_300_epoch_reference_in_promotion_records(tmp_path):
    from scripts.run_spectral_router_v5 import _write_promotion_outputs

    plan = {
        "output_dir": str(tmp_path),
        "stage_b": {"reference_id": "T_native", "hard_gates": {}},
        "paired_comparison": {"metrics": {}},
    }
    stage_b_decision = {"ranked": [{"id": "T_native", "metrics": {}}]}

    with pytest.raises(ValueError, match="not trained to 300 epochs"):
        _write_promotion_outputs(
            plan,
            [{"id": "C1", "metrics": {}}],
            stage_b_decision,
        )


def test_v5_reference_only_promotion_writes_valid_empty_comparison(tmp_path):
    import json

    from scripts.run_spectral_router_v5 import _write_promotion_outputs

    plan = {
        "output_dir": str(tmp_path),
        "stage_b": {"reference_id": "T_native", "hard_gates": {}},
        "paired_comparison": {"metrics": {}, "seed": 42, "resamples": 10},
    }
    _write_promotion_outputs(
        plan,
        [{"id": "T_native", "metrics": {}}],
        {"selected_evidence_id": "S3", "promoted": []},
    )

    final_decision = json.loads(
        (tmp_path / "final_gate_decision.json").read_text(encoding="utf-8")
    )
    paired = json.loads(
        (tmp_path / "paired_comparison.json").read_text(encoding="utf-8")
    )
    assert final_decision["accepted"] == []
    assert [row["id"] for row in final_decision["ranked"]] == ["T_native"]
    assert paired["comparisons"] == []


def test_v5_promotion_can_use_stricter_endpoint_gates(tmp_path):
    import json

    from scripts.run_spectral_router_v5 import _write_promotion_outputs

    plan = {
        "output_dir": str(tmp_path),
        "stage_b": {
            "reference_id": "T_native",
            "hard_gates": {"score": {"direction": "lower", "max_value": 0.8}},
        },
        "promote": {
            "hard_gates": {"score": {"direction": "lower", "max_value": 0.4}}
        },
        "paired_comparison": {"metrics": {}, "seed": 42, "resamples": 10},
    }
    promote_dir = tmp_path / "promote"
    promote_dir.mkdir()
    for result_id in ("t_native", "c1"):
        (promote_dir / f"{result_id}.json").write_text(
            json.dumps({"per_patient": {}}), encoding="utf-8"
        )
    _write_promotion_outputs(
        plan,
        [
            {"id": "T_native", "metrics": {"score": 0.5}},
            {"id": "C1", "metrics": {"score": 0.5}},
        ],
        {"promoted": ["C1"]},
    )

    decision = json.loads(
        (tmp_path / "final_gate_decision.json").read_text(encoding="utf-8")
    )
    assert decision["accepted"] == []
    assert all(row["gate_passed"] is False for row in decision["ranked"])
    assert all("limit 0.4" in row["gate_reasons"][0] for row in decision["ranked"])


def test_v6_separates_mechanism_screen_from_endpoint_gates():
    import yaml

    plan = yaml.safe_load(
        Path("configs/experiments/spectral_router_repair_plan_v6.yaml").read_text(
            encoding="utf-8"
        )
    )
    screen_peak = plan["stage_b"]["hard_gates"][
        "lesion_topq_peak_error_norm_mean"
    ]
    endpoint_peak = plan["promote"]["hard_gates"][
        "lesion_topq_peak_error_norm_mean"
    ]

    assert screen_peak == {"direction": "lower", "max_ratio": 0.95}
    assert endpoint_peak == {
        "direction": "lower",
        "max_value": 0.12,
        "max_ratio": 0.90,
    }


def test_frequency_runner_can_evaluate_exact_final_checkpoint(tmp_path):
    from scripts.run_frequency_ablations import _run_manifest_entry

    plan = {
        "base_config": "base.yaml",
        "ablation_config": "ablations.yaml",
        "output_dir": str(tmp_path),
    }
    settings = {
        "experiment_prefix": "exact",
        "epochs": 30,
        "seed": 42,
        "eval_interval": 10,
        "max_samples": 64,
        "mc_steps": 20,
        "require_final_checkpoint": True,
        "evaluate_final_checkpoint": True,
    }

    entry = _run_manifest_entry(
        plan,
        {"id": "CTRL", "preset": "sr_v5_s3"},
        settings,
        python="python",
        phase="screen",
    )

    # Path separators differ between Windows and POSIX checkouts; compare
    # normalised so the contract holds on both.
    assert str(entry["checkpoint"]).replace("\\", "/").endswith(
        "exact_ctrl/ckpt_epoch0030.pt"
    )
    checkpoint_arg = entry["eval_command"].index("--checkpoint") + 1
    assert entry["eval_command"][checkpoint_arg] == entry["checkpoint"]


def test_frequency_runner_writes_optional_paired_results(tmp_path):
    import json

    from scripts.run_frequency_ablations import _write_paired_results

    result_dir = tmp_path / "screen"
    result_dir.mkdir()
    entries = []
    for result_id, values in (("CTRL", (1.0, 2.0)), ("COLD", (0.5, 1.5))):
        path = result_dir / f"{result_id.lower()}.json"
        path.write_text(
            json.dumps(
                {
                    "per_patient": {
                        "p0": {"score": values[0]},
                        "p1": {"score": values[1]},
                    }
                }
            ),
            encoding="utf-8",
        )
        entries.append({"id": result_id, "result": str(path)})
    plan = {
        "output_dir": str(tmp_path),
        "paired_comparison": {
            "seed": 42,
            "resamples": 10,
            "metrics": {"score": "lower"},
        },
    }

    _write_paired_results(plan, entries, phase="screen")

    report = json.loads(
        (tmp_path / "screen_paired_comparison.json").read_text(encoding="utf-8")
    )
    comparison = report["comparisons"][0]
    assert {comparison["left_id"], comparison["right_id"]} == {"CTRL", "COLD"}
    assert comparison["metrics"]["score"]["paired_patients"] == 2


def test_v7_cold_bias_plan_is_a_common_checkpoint_causal_screen():
    import yaml

    plan = yaml.safe_load(
        Path(
            "configs/experiments/spectral_router_cold_bias_plan_v7.yaml"
        ).read_text(encoding="utf-8")
    )
    variants = {row["id"]: row for row in plan["variants"]}
    common = plan["common_train_overrides"]

    assert set(variants) == {
        "CTRL",
        "COLD",
        "CT_RELAX",
        "COLD_ALIGN",
        "COLD_RANK",
    }
    assert common["training.init_from"].endswith(
        "sr_v6_confirm_evidence-s3_c1/ckpt_epoch0100.pt"
    )
    assert common[
        "modules.residual_frequency.cross_level_router.native_warmup_epochs"
    ] == 0
    assert plan["screen"]["require_final_checkpoint"] is True
    assert plan["screen"]["evaluate_final_checkpoint"] is True
    assert variants["CTRL"].get("overrides") is None
    assert variants["COLD"]["overrides"] == {
        "losses.normalized_lesion_peak.weight": 0.20,
        "losses.normalized_lesion_peak.cold_weight": 2.0,
        "losses.normalized_lesion_peak.active_tau_max": 0.55,
    }
    assert variants["CT_RELAX"]["overrides"] == {
        "modules.residual_frequency.ct_reliability_floor_l2": 0.10,
        "modules.residual_frequency.ct_reliability_floor_l1": 0.25,
    }
    assert plan["paired_comparison"]["metrics"][
        "small_lesion_cold_bias_norm_mean"
    ] == "lower"
    assert plan["promotion_hard_gates"][
        "small_lesion_topq_peak_error_norm_mean"
    ]["max_ratio"] == 1.0


def test_frequency_runner_writes_optional_final_gate_results(tmp_path):
    import json

    from scripts.run_frequency_ablations import _write_final_gate_results

    plan = {
        "output_dir": str(tmp_path),
        "reference_id": "CTRL",
        "top_k": 1,
        "promotion_hard_gates": {
            "lesion_topq_peak_error_norm_mean": {
                "direction": "lower",
                "max_value": 0.12,
            }
        },
    }
    common_metrics = {
        "lesion_peak_error_norm_mean": 0.10,
        "lesion_centroid_distance_mean": 3.0,
        "failure_any_mean": 0.10,
        "false_hotspot_density_mean": 0.0001,
        "stripe_excess_mean": 0.0,
        "ssim_mean": 0.95,
        "mae_mean": 0.02,
        "lesion_boundary_gradient_mae_norm_mean": 0.10,
        "anatomy_edge_gradient_mae_norm_mean": 0.05,
        "directional_spectrum_error_norm_mean": 0.01,
    }
    records = [
        {
            "id": "CTRL",
            "preset": "base",
            "metrics": {
                **common_metrics,
                "lesion_topq_peak_error_norm_mean": 0.15,
            },
        },
        {
            "id": "GOOD",
            "preset": "candidate",
            "metrics": {
                **common_metrics,
                "lesion_topq_peak_error_norm_mean": 0.11,
            },
        },
    ]

    _write_final_gate_results(plan, records)

    decision = json.loads(
        (tmp_path / "final_gate_decision.json").read_text(encoding="utf-8")
    )
    assert decision["accepted"] == ["GOOD"]
    assert decision["reference_id"] == "CTRL"
    assert (tmp_path / "promotion" / "leaderboard.csv").exists()


def test_v8_final_calibration_uses_v7_ema_and_independent_confirmation_seed():
    import yaml

    plan = yaml.safe_load(
        Path(
            "configs/experiments/spectral_router_final_calibration_plan_v8.yaml"
        ).read_text(encoding="utf-8")
    )
    common = plan["common_train_overrides"]
    variants = {row["id"]: row for row in plan["variants"]}

    assert common["training.init_from"].endswith(
        "sr_v7_confirm_cold_rank/ckpt_epoch0050.pt"
    )
    assert common["training.init_weights"] == "ema"
    assert common["training.learning_rate"] == pytest.approx(2e-5)
    assert set(variants) == {
        "CAL_CTRL",
        "PEAK25",
        "PEAK_WIDE",
        "BALANCED",
        "TOPK18",
    }
    assert plan["screen"]["epochs"] == 12
    assert plan["screen"]["seed"] == 42
    assert plan["promote"]["epochs"] == 12
    assert plan["promote"]["seed"] == 43
    assert plan["hard_gates"]["lesion_topq_peak_error_norm_mean"] == {
        "direction": "lower",
        "max_ratio": 0.98,
    }
    assert plan["promotion_hard_gates"][
        "lesion_topq_peak_error_norm_mean"
    ] == {"direction": "lower", "max_value": 0.12}


def test_v5_final_comparison_uses_300_epoch_reference_only(tmp_path):
    import json

    from scripts.run_spectral_router_v5 import _write_promotion_outputs

    plan = {
        "output_dir": str(tmp_path),
        "stage_b": {"reference_id": "T_native", "hard_gates": {}},
        "paired_comparison": {
            "seed": 42,
            "resamples": 10,
            "metrics": {"score": "lower"},
        },
    }
    stage_b_decision = {"ranked": [{"id": "T_native", "metrics": {}}]}
    per_patient = {"p0": {"score": 1.0}, "p1": {"score": 2.0}}
    promote_dir = tmp_path / "promote"
    promote_dir.mkdir()
    (promote_dir / "t_native.json").write_text(
        json.dumps({"per_patient": per_patient}), encoding="utf-8"
    )
    (promote_dir / "c1.json").write_text(
        json.dumps({"per_patient": {"p0": {"score": 0.5}, "p1": {"score": 1.5}}}),
        encoding="utf-8",
    )
    promotion_records = [
        {
            "id": "T_native",
            "metrics": {"score": {"mean": 1.0}},
            "experiment": "sr_v5_full_evidence-s3_t_native",
        },
        {
            "id": "C1",
            "metrics": {"score": {"mean": 0.5}},
            "experiment": "sr_v5_full_evidence-s3_c1",
        },
    ]
    _write_promotion_outputs(plan, promotion_records, stage_b_decision)

    report = json.loads(
        (tmp_path / "paired_comparison.json").read_text(encoding="utf-8")
    )
    assert [
        {row["left_id"], row["right_id"]} for row in report["comparisons"]
    ] == [{"C1", "T_native"}]


@pytest.mark.parametrize(
    ("preset", "mode", "dct_enabled", "gabor_enabled"),
    [
        ("sr_v5_s0", "boundary_reliable", False, False),
        ("sr_v5_s1", "spectral_evidence_router", False, True),
        ("sr_v5_s2", "spectral_evidence_router", True, False),
        ("sr_v5_s3", "spectral_evidence_router", True, True),
    ],
)
def test_v5_spectral_router_presets_resolve_exact_factors_and_forward(
    preset, mode, dct_enabled, gabor_enabled
):
    import torch

    from src.model.config_utils import load_full_config
    from src.model.slmf_bbdm import SLMFBBDM

    cfg = load_full_config(
        "configs/experiments/slmf_png_spectral_router_v5.yaml",
        ablation=preset,
        ablation_config_path="configs/experiments/ablations.yaml",
        overrides=[
            "data.image_size=32",
            "runtime.eval_sampling_steps=2",
            "modules.conditional_mean.freeze=false",
            "modules.conditional_mean.checkpoint=null",
        ],
    )
    model = SLMFBBDM.from_config(cfg)
    frequency = cfg["modules"]["residual_frequency"]

    assert model.residual_frequency_mode == mode
    assert frequency["dct_descriptor"]["enabled"] is dct_enabled
    assert frequency["gabor_descriptor"]["enabled"] is gabor_enabled
    assert frequency["cross_level_router"]["enabled"] is False
    assert cfg["losses"]["spectral_router_regularization"]["enabled"] is (
        mode == "spectral_evidence_router"
    )
    if mode == "spectral_evidence_router":
        assert model.residual_preconditioner.dct_enabled is dct_enabled
        assert model.residual_preconditioner.gabor_enabled is gabor_enabled
        assert model.residual_preconditioner.cross_level_enabled is False
    else:
        assert model.residual_preconditioner.use_gabor_agreement is False
        assert model.residual_preconditioner.use_directional_reliability is False

    direct_routes = (
        "inject_adapter",
        "use_for_noise",
        "use_for_hotspot",
        "use_for_loss",
    )
    for route in direct_routes:
        assert cfg["modules"]["gabor"][route] is False
    assert cfg["losses"]["normalized_lesion_peak"]["enabled"] is True
    assert cfg["losses"]["frequency_gate_tv"]["enabled"] is False  # V5 base: gate-TV off

    batch = {
        "ct": torch.randn(1, 1, 32, 32),
        "pet": torch.randn(1, 1, 32, 32),
        "mask": torch.zeros(1, 1, 32, 32),
    }
    batch["mask"][:, :, 14:18, 14:18] = 1
    loss, logs = model(batch, timesteps=torch.tensor([250]))
    assert torch.isfinite(loss)
    # gate-TV disabled → DisabledLossTerm; only enabled+loss keys, no "available"
    assert logs["loss/frequency_gate_tv/enabled"].item() == 0
    assert logs["loss/frequency_gate_tv/loss"].item() == 0
    assert "loss/frequency_gate_tv/available" not in logs


def test_v5_data_paths_are_portable_and_overrideable():
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs/experiments/slmf_png_spectral_router_v5.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    split_manifest = Path(cfg["data"]["split_manifest"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    assert not split_manifest.is_absolute()
    assert not cache_dir.is_absolute()
    assert split_manifest.suffix.lower() == ".csv"

    from scripts.run_spectral_router_v5 import build_v5_dry_run_manifest

    plan = yaml.safe_load(
        (repo_root / "configs/experiments/spectral_router_ablation_plan_v5.yaml")
        .read_text(encoding="utf-8")
    )
    plan["common_train_overrides"].update(
        {
            "data.split_manifest": "/cloud/data/main_data/split_manifest.csv",
            "data.cache_dir": "/cloud/data/cache/tensors_main",
            "modules.conditional_mean.checkpoint": (
                "/cloud/artifacts/freq_mean_pretrain_v2/mean_best.pt"
            ),
        }
    )
    manifest = build_v5_dry_run_manifest(plan, python="python")
    for phase in ("stage_a_runs", "stage_b_runs", "promotion_runs"):
        for entry in manifest[phase]:
            command = entry["train_command"]
            assert (
                "data.split_manifest=/cloud/data/main_data/split_manifest.csv"
                in command
            )
            assert "data.cache_dir=/cloud/data/cache/tensors_main" in command
            assert (
                "modules.conditional_mean.checkpoint="
                "/cloud/artifacts/freq_mean_pretrain_v2/mean_best.pt"
            ) in command


def test_v5_trajectory_requires_every_whitelisted_checkpoint(
    tmp_path, monkeypatch
):
    from scripts import evaluate_v5_checkpoint_trajectory as trajectory_module

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    config_path = experiment_dir / "resolved_config.yaml"
    config_path.write_text("experiment: {name: test}\n", encoding="utf-8")
    for epoch in (50, 100, 200, 300):
        (experiment_dir / f"ckpt_epoch{epoch:04d}.pt").touch()

    def reject_evaluation(*args, **kwargs):
        raise AssertionError("checkpoint validation must happen before evaluation")

    monkeypatch.setattr(
        trajectory_module, "evaluate_checkpoint", reject_evaluation
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        trajectory_module.collect_trajectory(
            experiment_dir,
            config_path,
            whitelist_epochs=[50, 100, 150, 200, 250, 300],
        )

    message = str(exc_info.value)
    assert "ckpt_epoch0150" in message
    assert "ckpt_epoch0250" in message


def test_v5_trajectory_collects_exact_whitelist_and_best_checkpoints(
    tmp_path, monkeypatch
):
    from scripts import evaluate_v5_checkpoint_trajectory as trajectory_module

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    config_path = experiment_dir / "resolved_config.yaml"
    config_path.write_text("experiment: {name: test}\n", encoding="utf-8")
    for label in (
        "ckpt_epoch0050",
        "ckpt_epoch0100",
        "ckpt_epoch0150",
        "ckpt_best_lesion",
        "ckpt_best_combined",
    ):
        (experiment_dir / f"{label}.pt").touch()

    evaluated = []

    def fake_evaluation(config, checkpoint, **kwargs):
        evaluated.append(checkpoint.stem)
        return {"aggregate": {"mae_mean": 0.1}}

    monkeypatch.setattr(
        trajectory_module, "evaluate_checkpoint", fake_evaluation
    )
    trajectory = trajectory_module.collect_trajectory(
        experiment_dir,
        config_path,
        whitelist_epochs=[50, 100],
    )

    assert set(trajectory["results"]) == {
        "ckpt_epoch0050",
        "ckpt_epoch0100",
        "ckpt_best_lesion",
        "ckpt_best_combined",
    }
    assert set(evaluated) == set(trajectory["results"])


def test_v5_trajectory_cli_help_runs_from_repo_root():
    import subprocess

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_v5_checkpoint_trajectory.py",
            "--help",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--experiment-dir" in completed.stdout
    assert "--config" in completed.stdout
    assert "--epochs" in completed.stdout


def test_paired_report_is_deterministic_and_handles_small_intersections(tmp_path):
    import json

    from scripts.compare_v4_results import compare_result_pair

    left = {
        "per_patient": {
            f"p{i}": {"lesion_topq_peak_error_norm_mean": float(i) / 100.0}
            for i in range(5)
        }
    }
    right = {
        "per_patient": {
            f"p{i}": {"lesion_topq_peak_error_norm_mean": float(i + 1) / 100.0}
            for i in range(5)
        }
    }
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left), encoding="utf-8")
    right_path.write_text(json.dumps(right), encoding="utf-8")

    first = compare_result_pair(
        left_path,
        right_path,
        {"lesion_topq_peak_error_norm_mean": "lower"},
        seed=42,
        resamples=10_000,
    )
    second = compare_result_pair(
        left_path,
        right_path,
        {"lesion_topq_peak_error_norm_mean": "lower"},
        seed=42,
        resamples=10_000,
    )
    row = first["metrics"]["lesion_topq_peak_error_norm_mean"]

    assert first == second
    assert row["paired_patients"] == 5
    assert row["wins"] == 5
    assert row["losses"] == 0
    assert row["median_difference"] == pytest.approx(-0.01)
    assert len(row["bootstrap_95_ci"]) == 2

    right["per_patient"] = {"p0": right["per_patient"]["p0"]}
    right_path.write_text(json.dumps(right), encoding="utf-8")
    small = compare_result_pair(
        left_path,
        right_path,
        {"lesion_topq_peak_error_norm_mean": "lower"},
    )
    assert small["metrics"]["lesion_topq_peak_error_norm_mean"]["bootstrap_95_ci"] is None


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


def test_v5_t0_c1_rescue_entries_are_matched_exact_300_epoch_runs():
    import yaml

    from scripts.run_v5_t0_c1_300 import build_rescue_entries

    plan = yaml.safe_load(
        Path(
            "configs/experiments/spectral_router_ablation_plan_v5.yaml"
        ).read_text(encoding="utf-8")
    )
    entries = build_rescue_entries(plan, python="python")

    # Legacy rescue uses T_legacy (the old cross_level_enabled=false)
    # and C1 (the learned 3-way route) for backward diagnostic compatibility.
    assert [entry["id"] for entry in entries] == ["T_legacy", "C1"]
    assert all(entry["selected_evidence_id"] == "S3" for entry in entries)
    assert all(entry["required_checkpoint_epoch"] == 300 for entry in entries)
    assert all(
        entry["checkpoint"].endswith("ckpt_epoch0300.pt") for entry in entries
    )
    assert entries[0]["experiment"] == "sr_v5_full_evidence-s3_t_legacy"
    assert entries[1]["experiment"] == "sr_v5_full_evidence-s3_c1"

    tlegacy_command = " ".join(entries[0]["train_command"])
    c1_command = " ".join(entries[1]["train_command"])
    for command in (tlegacy_command, c1_command):
        assert "training.num_epochs=300" in command
        assert "training.resume_from=null" in command
        assert "training.init_from=null" in command
        assert "model.initialization_seed=4242" in command
    assert "cross_level_router.enabled=false" in tlegacy_command
    assert "cross_level_router.enabled=true" in c1_command


def test_v5_t0_c1_one_click_pixi_tasks_are_declared():
    pixi = Path("pixi.toml").read_text(encoding="utf-8")
    assert 'train-spectral-router-v5-t0-c1 = "python scripts/run_v5_t0_c1_300.py"' in pixi
    assert (
        'dry-run-spectral-router-v5-t0-c1 = "python scripts/run_v5_t0_c1_300.py --dry-run"'
        in pixi
    )


def test_v5_t0_c1_script_executes_as_a_direct_dry_run():
    import subprocess

    completed = subprocess.run(
        [sys.executable, "scripts/run_v5_t0_c1_300.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "sr_v5_full_evidence-s3_t_legacy" in completed.stdout
    assert "sr_v5_full_evidence-s3_c1" in completed.stdout
    assert "ckpt_epoch0300.pt" in completed.stdout
