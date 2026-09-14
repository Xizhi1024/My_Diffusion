"""End-to-end CPU smoke test for the prior-anchored adaptive router.

Exercises the full exploratory chain on synthetic tensors, without CUDA, the
PNG cache, or a real mean checkpoint:

  config load -> direct-PNG prior load -> model construction ->
  one train step -> one optimiser step -> epoch phase update ->
  model-state roundtrip -> four-epoch Trainer run -> metrics JSONL.

It does not touch the formal 100-epoch cloud contract and writes nothing
outside ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.frequency.h3_native_null_schedule import (  # noqa: E402
    H3_V2_BAND_ORDER,
    H3_V2_PIPELINE_ID,
    H3_V2_ROUTE_ORDER,
)
from src.model.frequency.prior_anchor_schedule import (  # noqa: E402
    DIRECT_PNG_PREVIEW_PATH_POLICY,
    DIRECT_PNG_PREVIEW_PIPELINE_ID,
)


NUM_TIMESTEPS = 50


def _preview_payload(steps: int = NUM_TIMESTEPS) -> dict:
    """A schema-valid direct-PNG preview prior with a monotonic active mass."""
    # Per-band monotonic non-increasing curves in [0, 1].
    bases = [1.0, 0.85, 0.95, 0.60, 0.45, 0.25]
    rows: dict[str, list[float]] = {}
    for band, base in zip(H3_V2_BAND_ORDER, bases):
        curve = torch.linspace(base, 0.0, steps).clamp(min=0.0).tolist()
        rows[band] = curve
    partition_hash = "b" * 64
    return {
        "schema_version": 1,
        "pipeline_id": DIRECT_PNG_PREVIEW_PIPELINE_ID,
        "generated_at_utc": "2026-07-26T00:00:00+00:00",
        "decision": "PREVIEW_ONLY",
        "preview_only": True,
        "inference_schedule_allowed": False,
        "production_activation_allowed": False,
        "formal_h3_v2_claim_allowed": False,
        "reason": "cpu smoke test",
        "paths": {
            "png_root": "Data/data",
            "manifest": "main_data/split_manifest.csv",
            "dataset_contract": "configs/dataset_contract_stage0a_v1.json",
            "mean_checkpoint": "checkpoints/mean/model.pt",
            "output_dir": "artifacts/preview",
        },
        "path_policy": DIRECT_PNG_PREVIEW_PATH_POLICY,
        "input_contract": {
            "source": "raw_png_direct",
            "cache_read": False,
            "cache_written": False,
            "manifest_file_sha256": "a" * 64,
            "mechanism_partition_sha256": partition_hash,
            "image_size": 192,
            "ct_normalization": "grayscale_uint8/127.5-1",
            "pet_normalization": "(255-grayscale_uint8)/127.5-1",
            "mask_normalization": "nearest_resize_then_uint8>127",
            "physical_split_policy": "index_all_split_folders_by_sample_id",
            "raw_png_fingerprint": {
                "algorithm": "sha256",
                "canonicalization": "smoke",
                "sha256": "c" * 64,
                "file_count": 12,
            },
        },
        "dataset_contract": {
            "contract_name": "ct_pet_png_dataset_contract",
            "contract_sha256": "d" * 64,
            "checks": {
                "raw_png_file_count": True,
                "raw_png_combined_sha256": True,
                "manifest_file_sha256": True,
            },
        },
        "checkpoint": {
            "file_sha256": "e" * 64,
            "epoch": 1,
            "mean_config": {"pathology_policy": "excluded"},
            "mechanism_partition_sha256": partition_hash,
            "has_data_lineage": True,
            "checkpoint_lineage_verified_for_current_png_run": False,
        },
        "quality_control": {
            "empty_lesion_masks": 0,
            "zero_clean_band_norm_samples": {band: 0 for band in H3_V2_BAND_ORDER},
        },
        "analysis": {
            "analysis_seed": 17,
            "num_train_timesteps": steps,
            "m_schedule": "linear",
            "sigma_scale": 1.0,
            "band_order": list(H3_V2_BAND_ORDER),
            "patient_weighting": "equal_patient_weight",
            "noise_identity": "smoke",
            "recoverability": "smoke",
            "active_transform": "clip((recoverability-0.5)/0.5,0,1)",
            "shape_constraint": "per-band isotonic non-increasing over timestep",
            "dense_curve_method": (
                "exact A=<x,x>, B=<e,e>, C=<x,e> sufficient-statistic "
                "evaluation; no timestep interpolation"
            ),
        },
        "route_mapping_preview": {
            "route_order": list(H3_V2_ROUTE_ORDER),
            "formula": "[a,0,1-a]",
            "shallow_probability": "structurally_zero",
        },
        "crossings": {},
        "preview_native_active_mass": rows,
        "outputs": {
            "curve_csv": "artifacts/preview/h3_prior_curves.csv",
            "patient_npz": "artifacts/preview/h3_prior_patient_curves.npz",
            "plot_png": "artifacts/preview/h3_prior_curves.png",
        },
    }


def _write_preview(root: Path, steps: int = NUM_TIMESTEPS) -> Path:
    path = root / "artifacts" / "preview" / "h3_prior_preview.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_preview_payload(steps), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _router_config(repo_root: Path, prior_path: Path) -> dict:
    prior_relative = prior_path.relative_to(repo_root).as_posix()
    prior_digest = __import__("hashlib").sha256(prior_path.read_bytes()).hexdigest()
    return {
        "enabled": True,
        "hidden_channels": 8,
        "hard_all_null": False,
        "policy": "prior_anchored_learned",
        "fixed_prior": [0.05, 0.05, 0.90],
        "h3_schedule_path": prior_relative,
        "h3_schedule_sha256": prior_digest,
        "h3_schedule_source": "direct_png_preview",
        "h3_repository_root": str(repo_root),
        "h3_allow_unverified_preview_lineage": True,
        "initial_null_probability": 0.90,
        "active_logit_delta_max": 2.0,
        "initial_destination_native_probability": 0.95,
        # Compressed 100-epoch phase contract so the smoke test can visit every
        # phase on a tiny budget.  The ratios match the cloud schedule.
        "prior_warmup_epochs": 1,
        "prior_active_ramp_epochs": 1,
        "prior_destination_warmup_epochs": 3,
        "prior_destination_ramp_epochs": 1,
        "prior_anchor_decay_end_epoch": 5,
        "prior_anchor_final_scale": 0.10,
        "phase_observation_epochs": [1, 2, 3, 4, 5],
    }


def _smoke_config(repo_root: Path, prior_path: Path) -> dict:
    return {
        "experiment": {"name": "prior_anchored_smoke", "seed": 42},
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
            "log_interval": 1,
            "eval_interval": 999,
            "sample_interval": 999,
            "save_interval": 999,
            "eval_sampling_steps": 4,
            "gradient_diagnostics": True,
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
            "enable_heteroscedastic": False,
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
            "conditional_mean": {
                "enabled": True,
                "levels": 2,
                "base_channels": 8,
                "loss_weight": 0.0,
                "detach_bridge": True,
                "freeze": False,
            },
            "residual_bridge": {"enabled": True},
            "wavelet_unet": {"enabled": False},
            "residual_frequency": {
                "enabled": True,
                "mode": "spectral_evidence_router",
                # output_channels defaults to [256, 256, 128, 64] to match the
                # BBDMUNet skip channels (base_channels=64).  Do not shrink it.
                "band_scales": [0.5, 0.25],
                "use_noise_release": True,
                "use_ct_reliability": True,
                "use_content_reliability": True,
                "use_subband_gates": True,
                "ct_reliability_floor_l2": 0.25,
                "ct_reliability_floor_l1": 0.50,
                "gate_max": 0.25,
                "snr_center": 0.0,
                "snr_temperature": 2.0,
                "cross_temperature": 1.0,
                "content_hidden_channels": 8,
                "amplitude_delta_min": -0.05,
                "amplitude_delta_max": 0.10,
                "dct_descriptor": {"enabled": False},
                "gabor_descriptor": {"enabled": False},
                "cross_level_router": _router_config(repo_root, prior_path),
            },
            "gabor": {"enabled": False},
            "organ_prior": {"enabled": False},
            "hotspot_prior": {"enabled": False},
            "semantic_prior": {"enabled": False},
            "zero_adapter": {"enabled": False},
            "condition_dropout": {"enabled": False},
            "bbdm_bridge": {
                "name": "bbdm_bridge",
                "num_train_timesteps": NUM_TIMESTEPS,
                "m_schedule": "linear",
            },
        },
        "losses": {
            "lesion_roi_l1": {"enabled": False},
            "topk_lesion": {"enabled": False},
            "normalized_lesion_peak": {"enabled": False},
            "spectral_router_regularization": {
                "enabled": True,
                "weight": 1.0,
                "prior_anchor_weight": 1.0e-2,
                "monotonic_weight": 2.0e-3,
                "curvature_weight": 2.0e-4,
                "budget_weight": 2.0e-3,
                "shallow_cost_weight": 1.0e-3,
                "artifact_safety_weight": 0.0,
            },
        },
    }


def _build_model(repo_root: Path, prior_path: Path):
    from src.model.slmf_bbdm import SLMFBBDM

    return SLMFBBDM.from_config(_smoke_config(repo_root, prior_path))


def test_prior_anchored_router_end_to_end_on_cpu(tmp_path: Path) -> None:
    repo_root = tmp_path
    prior_path = _write_preview(repo_root)
    model = _build_model(repo_root, prior_path)

    router = model.residual_preconditioner
    assert router._effective_policy == "prior_anchored_learned"
    # The H3 prior buffer must be loaded and shaped [2, 3, T].
    assert router._h3_native_active_mass.shape == (2, 3, NUM_TIMESTEPS)

    # --- epoch phase update ---
    router.set_training_epoch(0)
    assert router._prior_active_progress.item() == pytest.approx(0.0)
    assert router._prior_destination_progress.item() == pytest.approx(0.0)
    router.set_training_epoch(5)
    assert router._prior_active_progress.item() == pytest.approx(1.0)
    assert router._prior_destination_progress.item() == pytest.approx(1.0)
    assert router._prior_anchor_scale.item() == pytest.approx(0.10)

    # --- one train step (warmup: adaptive heads bypassed) ---
    router.set_training_epoch(0)
    batch = {
        "ct": torch.randn(2, 1, 32, 32),
        "pet": torch.randn(2, 1, 32, 32),
        "mask": torch.zeros(2, 1, 32, 32),
        "organ_mask": torch.zeros(2, 6, 32, 32),
        "organ_distance": torch.zeros(2, 6, 32, 32),
        "mu_map": torch.zeros(2, 1, 32, 32),
        "meta": [{"pet_suv_max": 20.0, "suv_ok": True} for _ in range(2)],
    }
    loss, logs = model(batch)
    assert torch.isfinite(loss)
    # Warmup: destination head must receive no gradient signal.
    assert logs["frequency/prior_anchor_shallow_mass"].item() == pytest.approx(0.0)
    assert logs["frequency/prior_anchor_native_mass"].item() >= 0.0
    assert logs["frequency/route_native_mass"].item() >= 0.0
    assert logs["frequency/route_shallow_mass"].item() == pytest.approx(0.0)
    assert logs["frequency/route_null_mass"].item() >= 0.0
    native = logs["frequency/prior_anchor_native_mass"].item()
    shallow = logs["frequency/prior_anchor_shallow_mass"].item()
    null = logs["frequency/prior_anchor_null_mass"].item()
    assert native + shallow + null == pytest.approx(1.0, abs=1e-5)
    canonical = [
        logs["frequency/route_native_mass"].item(),
        logs["frequency/route_shallow_mass"].item(),
        logs["frequency/route_null_mass"].item(),
    ]
    assert canonical == pytest.approx([native, shallow, null], abs=1e-7)
    assert sum(canonical) == pytest.approx(1.0, abs=1e-5)

    # --- one optimiser step (full adaptive release) ---
    router.set_training_epoch(5)
    loss, logs = model(batch)
    loss.backward()
    prior_active_params = list(router.prior_active_heads.parameters())
    prior_dest_params = list(router.prior_destination_heads.parameters())
    assert any(p.grad is not None for p in prior_active_params)
    assert any(p.grad is not None for p in prior_dest_params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # --- weights-only-safe model-state roundtrip restores learned weights ---
    router.set_training_epoch(4)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"state": "{}"},
        "ema": {"shadow": {}, "step_count": 0, "decay": 0.999},
        "epoch": 4,
        "step": 16,
        "config": _smoke_config(repo_root, prior_path),
        "monitoring": {},
        "data_lineage": {"cache_metadata_sha256": "f" * 64},
    }
    ckpt_path = tmp_path / "ckpt_epoch0004.pt"
    torch.save(state, ckpt_path)
    resumed_payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    resumed_model = _build_model(repo_root, prior_path)
    resumed_model.load_state_dict(resumed_payload["model"])
    resumed_router = resumed_model.residual_preconditioner
    resumed_router.set_training_epoch(4)
    assert resumed_router._prior_active_progress.item() == pytest.approx(1.0)
    assert resumed_router._prior_destination_progress.item() == pytest.approx(1.0)
    assert resumed_router._prior_anchor_scale.item() == pytest.approx(
        router._prior_anchor_scale.item()
    )

    # --- trainer metrics JSONL + summary chain ---
    from src.data.dataset import FakeDataset
    from src.model.trainer import Trainer
    from torch.utils.data import DataLoader

    cfg = _smoke_config(repo_root, prior_path)
    cfg["training"]["num_epochs"] = 4
    cfg["training"]["checkpoint_dir"] = str(tmp_path / "checkpoints")
    cfg["runtime"]["sample_dir"] = str(tmp_path / "samples")
    cfg["runtime"]["save_interval"] = 1
    cfg["runtime"]["eval_interval"] = 999
    metrics_path = tmp_path / "training_metrics.jsonl"
    # Absolute path so the Trainer writes inside tmp_path, never the repo.
    cfg["runtime"]["training_metrics_jsonl"] = str(metrics_path)
    # Rebuild a fresh model so the trainer owns its parameters.
    trainer_model = _build_model(repo_root, prior_path)
    dl = DataLoader(FakeDataset(8, image_size=32), batch_size=2, drop_last=True)
    trainer = Trainer(trainer_model, cfg, dl, dl, device="cpu")
    trainer.run(num_epochs=4)

    assert metrics_path.is_file()
    records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["epoch"] for record in records] == [1, 2, 3, 4]
    # One record per epoch, no duplicates after the run completes.
    assert len(records) == len({record["epoch"] for record in records})
    # Router diagnostics are nested under train/ and must be present.
    train_keys = set(records[-1]["train"].keys())
    assert "frequency/prior_anchor_active_progress" in train_keys
    assert "frequency/prior_anchor_native_mass" in train_keys
    assert "frequency/prior_anchor_null_mass" in train_keys
    assert "frequency/route_native_mass" in train_keys
    assert "frequency/route_shallow_mass" in train_keys
    assert "frequency/route_null_mass" in train_keys
    assert "grad/prior_active_final" in train_keys
    # By epoch 4 the compressed schedule has fully released both heads.
    assert records[-1]["train"]["frequency/prior_anchor_active_progress"] == (
        pytest.approx(1.0)
    )
    assert records[-1]["train"]["frequency/prior_anchor_destination_progress"] == (
        pytest.approx(1.0)
    )
    # Nothing was written into the repository working tree.
    assert not Path("training_metrics.jsonl").exists()
    assert not Path("checkpoints/prior_anchored_smoke").exists()


def test_background_repair_freezes_router_and_forces_native_destination(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path
    prior_path = _write_preview(repo_root)
    config = _smoke_config(repo_root, prior_path)
    frequency = config["modules"]["residual_frequency"]
    frequency["freeze"] = True
    frequency["cross_level_router"]["destination_mode"] = "native_only"
    config["losses"]["spectral_router_regularization"]["enabled"] = False
    config["losses"]["nonlesion_body_lowpass"] = {
        "enabled": True,
        "weight": 0.2,
        "sigma": 2.0,
        "body_threshold": 0.03,
        "lesion_exclusion_radius": 4,
        "tail_quantile": 0.75,
        "tail_weight": 2.0,
    }

    from src.model.slmf_bbdm import SLMFBBDM

    model = SLMFBBDM.from_config(config)
    router = model.residual_preconditioner

    assert model.residual_frequency_frozen is True
    assert model.residual_frequency_destination_mode == "native_only"
    assert router.inference_destination_mode == "native_only"
    assert all(not parameter.requires_grad for parameter in router.parameters())
    assert model.loss_terms["nonlesion_body_lowpass"].enabled is True

    model.train()
    assert router.training is False


def test_smoke_prior_load_is_fail_closed_against_hash_tamper(tmp_path: Path) -> None:
    """The smoke prior loader must reject a tampered SHA-256."""
    from src.model.frequency.prior_anchor_schedule import load_prior_anchor_schedule

    prior_path = _write_preview(tmp_path)
    digest = __import__("hashlib").sha256(prior_path.read_bytes()).hexdigest()
    tampered = "0" * 64
    assert tampered != digest
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_prior_anchor_schedule(
            prior_path,
            schedule_source="direct_png_preview",
            expected_file_sha256=tampered,
            expected_num_train_timesteps=NUM_TIMESTEPS,
            repository_root=tmp_path,
            allow_unverified_preview_lineage=True,
        )


def test_prior_destination_inference_interventions_preserve_availability(
    tmp_path: Path,
) -> None:
    prior_path = _write_preview(tmp_path)
    model = _build_model(tmp_path, prior_path)
    router = model.residual_preconditioner
    router.set_training_epoch(5)

    residual = torch.randn(3, 1, 32, 32)
    ct = torch.randn_like(residual)
    timestep = torch.tensor([20, 20, 20])
    state_before = {
        key: value.detach().clone()
        for key, value in router.state_dict().items()
    }

    router.set_inference_destination_intervention("learned")
    _, learned = router(
        residual,
        timestep,
        model.noise_schedule,
        ct,
    )
    learned_active = learned["route_active"].detach().clone()

    router.set_inference_destination_intervention("native_only")
    _, native_only = router(
        residual,
        timestep,
        model.noise_schedule,
        ct,
    )
    torch.testing.assert_close(native_only["route_active"], learned_active)
    assert torch.count_nonzero(
        native_only["route_conditional_shallow"]
    ).item() == 0
    torch.testing.assert_close(
        native_only["routes_l2"][..., 0],
        learned_active[:, 0],
    )
    torch.testing.assert_close(
        native_only["routes_l1"][..., 0],
        learned_active[:, 1],
    )

    fixed_l2 = (0.20, 0.30, 0.40)
    fixed_l1 = (0.05, 0.06, 0.07)
    router.set_inference_destination_intervention(
        "fixed",
        fixed_l2=fixed_l2,
        fixed_l1=fixed_l1,
    )
    _, fixed = router(
        residual,
        timestep,
        model.noise_schedule,
        ct,
    )
    torch.testing.assert_close(fixed["route_active"], learned_active)
    torch.testing.assert_close(
        fixed["route_conditional_shallow"][:, 0],
        torch.tensor(fixed_l2).view(1, 3).expand(3, 3),
    )
    torch.testing.assert_close(
        fixed["route_conditional_shallow"][:, 1],
        torch.tensor(fixed_l1).view(1, 3).expand(3, 3),
    )
    torch.testing.assert_close(
        fixed["routes_l2"].sum(dim=-1),
        torch.ones(3, 3),
    )
    torch.testing.assert_close(
        fixed["routes_l1"].sum(dim=-1),
        torch.ones(3, 3),
    )

    router.set_inference_destination_intervention(
        "shuffled_destination",
        shuffle_seed=17,
    )
    _, shuffled = router(
        residual,
        timestep,
        model.noise_schedule,
        ct,
    )
    torch.testing.assert_close(shuffled["route_active"], learned_active)
    assert shuffled["routes_l2"].shape == learned["routes_l2"].shape
    assert router.inference_destination_intervention() == {
        "mode": "shuffled_destination",
        "shuffle_seed": 17,
    }

    state_after = router.state_dict()
    assert state_after.keys() == state_before.keys()
    for key, value in state_before.items():
        torch.testing.assert_close(state_after[key], value)


def test_router_background_suite_runs_all_interventions_on_cpu(
    tmp_path: Path,
) -> None:
    from scripts.eval_router_background_suite import _main, build_parser

    prior_path = _write_preview(tmp_path)
    config = _smoke_config(tmp_path, prior_path)
    config["data"]["use_fake_data"] = True
    config["data"]["require_cache_lineage"] = False
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    model = _build_model(tmp_path, prior_path)
    model.residual_preconditioner.set_training_epoch(5)
    checkpoint_path = tmp_path / "ckpt_epoch0005.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": 5,
            "step": 10,
            "config": config,
        },
        checkpoint_path,
    )
    fixed_values = {
        "frequency/prior_anchor_l2_lh_conditional_shallow": 0.20,
        "frequency/prior_anchor_l2_hl_conditional_shallow": 0.30,
        "frequency/prior_anchor_l2_hh_conditional_shallow": 0.40,
        "frequency/prior_anchor_l1_lh_conditional_shallow": 0.05,
        "frequency/prior_anchor_l1_hl_conditional_shallow": 0.06,
        "frequency/prior_anchor_l1_hh_conditional_shallow": 0.07,
    }
    metrics_path = tmp_path / "training_metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"epoch": 5, "train": fixed_values}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "suite"
    args = build_parser().parse_args(
        [
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
            "--weights",
            "raw",
            "--device",
            "cpu",
            "--max-samples",
            "2",
            "--batch-size",
            "2",
            "--steps",
            "1",
            "--modes",
            "learned",
            "--frequency-modes",
            "full,detail_off,ll_off,all_frequency_off",
            "--grid-samples",
            "0",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert _main(args) == 0
    summary = json.loads(
        (output_dir / "suite_summary.json").read_text(encoding="utf-8")
    )
    assert summary["manifest"]["schema_version"] == 2
    assert summary["manifest"]["frequency_modes"] == [
        "full",
        "detail_off",
        "ll_off",
        "all_frequency_off",
    ]
    assert len(summary["runs"]) == 4
    assert {
        row["frequency_mode"] for row in summary["runs"]
    } == {
        "full",
        "detail_off",
        "ll_off",
        "all_frequency_off",
    }
    assert {row["mode"] for row in summary["runs"]} == {"learned"}
    assert (output_dir / "paired_comparisons_vs_learned.csv").is_file()
    assert (
        output_dir / "paired_comparisons_vs_learned_full.csv"
    ).is_file()
