"""Training entry point for SLMF-BBDM.

Usage:
    python scripts/train_v2.py --config configs/experiments/slmf_full.yaml
    python scripts/train_v2.py --config configs/experiments/slmf_full.yaml --ablation no_gabor
    python scripts/train_v2.py --config configs/experiments/slmf_full.yaml --override modules.gabor.enabled=false
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.lineage import (
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.model.config_utils import (
    load_full_config,
    save_resolved_config,
    resolve_runtime_profile,
    validate_png_baseline_config,
    log_startup_status,
)
from src.model.slmf_bbdm import SLMFBBDM
from src.model.trainer import Trainer, resolve_checkpoint_dir


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_initial_model_state(checkpoint: dict, weights: str = "raw") -> dict:
    """Select raw or EMA model weights for optimizer-fresh fine-tuning."""
    raw_state = checkpoint.get("model", checkpoint)
    if weights == "raw":
        return raw_state
    if weights != "ema":
        raise ValueError("training.init_weights must be 'raw' or 'ema'")
    for key in ("ema_model", "model_ema"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, dict) and isinstance(ema_state.get("shadow"), dict):
        merged = dict(raw_state)
        merged.update(ema_state["shadow"])
        return merged
    raise KeyError("EMA init requested, but checkpoint contains no EMA weights")


def _save_run_metadata(
    model: SLMFBBDM,
    config: dict,
    ckpt_dir: str,
    ablation: str | None,
    data_lineage: dict | None = None,
) -> None:
    """Persist the experiment/module state needed to reproduce an ablation run."""
    os.makedirs(ckpt_dir, exist_ok=True)
    enabled_modules = {
        name: bool(prior.enabled)
        for name, prior in model.priors.items()
    }
    enabled_modules.update({
        "zero_adapter": bool(model.zero_adapter_enabled),
        "condition_dropout": bool(model.condition_dropout.enabled),
        "scale_adaptive_noise": bool(
            getattr(model.noise_schedule, "name", "") == "scale_adaptive_noise"
            and model.noise_schedule.enabled
        ),
        "metadata_film": bool(model.meta_enabled),
        "segmenter": bool(model.segmenter_enabled),
        "conditional_mean": bool(model.conditional_mean_enabled),
        "residual_bridge": bool(model.residual_bridge_enabled),
        "residual_frequency": bool(model.residual_frequency_enabled),
    })
    enabled_losses = {
        name: bool(loss.enabled)
        for name, loss in model.loss_terms.items()
    }
    training_cfg = config.get("training", {})
    lesion_loss_names = (
        "topk_lesion",
        "lesion_roi_l1",
        "outside_peak_ranking",
        "roi_suv",
        "false_hotspot",
        "hotspot_prior",
    )
    enabled_lesion_losses = [
        name
        for name in lesion_loss_names
        if enabled_losses.get(name, False)
    ]
    training_stage = training_cfg.get("stage", "base")
    posttrain_cfg = training_cfg.get("lesion_posttrain", {})
    lesion_posttrain_enabled = bool(
        posttrain_cfg.get("enabled", training_stage == "lesion_posttrain")
    )

    metadata = {
        "experiment": config.get("experiment", {}).get("name", "slmf_bbdm"),
        "ablation": ablation,
        "training_stage": training_stage,
        "init_from": training_cfg.get("init_from"),
        "init_weights": training_cfg.get("init_weights", "raw"),
        "resume_from": training_cfg.get("resume_from"),
        "enabled_modules": enabled_modules,
        "enabled_losses": enabled_losses,
        "lesion_aware_posttraining": {
            "enabled": lesion_posttrain_enabled,
            "source_checkpoint": training_cfg.get("init_from"),
            "description": posttrain_cfg.get(
                "description",
                "Dense lesion ROI reconstruction with outside-peak failure suppression.",
            ),
            "lesion_losses": enabled_lesion_losses,
        },
        "trainable_parameters": model.get_trainable_params(),
        "total_parameters": model.get_total_params(),
        "data_lineage": data_lineage,
    }
    with open(os.path.join(ckpt_dir, "run_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="SLMF-BBDM Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--ablation", type=str, default=None, help="Ablation preset name")
    parser.add_argument("--ablation-config", type=str, default="configs/experiments/ablations.yaml")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides (repeatable)")
    args = parser.parse_args()

    # 1. Load config with overrides
    config = load_full_config(
        args.config,
        ablation=args.ablation,
        ablation_config_path=args.ablation_config,
        overrides=args.override if args.override else None,
    )

    # Formal experiment configs must never be silently converted into the
    # CPU-only smoke profile below.
    require_cuda = bool(config.get("runtime", {}).get("require_cuda", False))
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.require_cuda=true, but CUDA is unavailable; refusing the "
            "CPU smoke/debug fallback"
        )

    # 1.5 Set random seed for reproducibility (must happen before model/dataloader init)
    seed = config.get("experiment", {}).get("seed", 42)
    _set_seed(seed)

    # 2. Resolve runtime profile (CPU fallback if no CUDA)
    config = resolve_runtime_profile(config)

    # 2.5 Validate PNG baseline config + log startup status
    #     Fails fast if data.mode=png but a DICOM/SUV/organ path is still on.
    startup_status = validate_png_baseline_config(config)
    log_startup_status(startup_status)
    data_lineage = load_checkpoint_data_lineage(config)
    if data_lineage is not None:
        print(
            "Verified cache lineage: "
            f"{data_lineage['cache_metadata_sha256']}"
        )

    # 3. Save resolved config
    ckpt_dir = resolve_checkpoint_dir(config)
    save_resolved_config(config, ckpt_dir)

    # 4. Build model
    print("Building SLMF-BBDM model...")
    model = SLMFBBDM.from_config(config)
    _save_run_metadata(
        model,
        config,
        ckpt_dir,
        args.ablation,
        data_lineage=data_lineage,
    )

    # 4.5 Optionally initialise weights from a checkpoint (fine-tuning).
    #     Only model weights are loaded; optimizer / scheduler / EMA start fresh
    #     so the new loss landscape isn't dragged by stale momentum.
    training_cfg = config.get("training", {})
    init_from = training_cfg.get("init_from")
    resume_from = training_cfg.get("resume_from")
    if init_from and resume_from:
        raise ValueError("Use either training.init_from or training.resume_from, not both.")
    if init_from:
        if not os.path.exists(init_from):
            raise FileNotFoundError(f"training.init_from checkpoint not found: {init_from}")
        init_weights = str(training_cfg.get("init_weights", "raw"))
        print(
            f"Loading {init_weights} model weights from {init_from} "
            "(optimizer/EMA fresh)..."
        )
        ckpt = torch.load(init_from, map_location="cpu", weights_only=True)
        validate_checkpoint_data_lineage(
            ckpt,
            data_lineage,
            required=bool(
                config.get("data", {}).get("require_cache_lineage", False)
            ),
            context=f"training.init_from checkpoint {init_from}",
        )
        model.load_state_dict(_select_initial_model_state(ckpt, init_weights))
        print("  loaded.")

    enabled_priors = [n for n, p in model.priors.items() if p.enabled]
    enabled_losses = [n for n, loss in model.loss_terms.items() if loss.enabled]

    print(f"  Stage: {config.get('training', {}).get('stage', 'base')}")
    print(f"  Priors: {enabled_priors if enabled_priors else '(none)'}")
    print(f"  Losses: {enabled_losses if enabled_losses else '(base only)'}")
    print(f"  Trainable: {model.get_trainable_params():,} / Total: {model.get_total_params():,}")

    # 5. Build DataLoaders
    from src.data.dataset import build_dataloaders

    data_cfg = config.get("data", {})
    run_cfg = config.get("runtime", {})
    train_loader, val_loader = build_dataloaders(data_cfg, run_cfg)

    # 6. Train
    trainer = Trainer(model, config, train_loader, val_loader)
    if resume_from:
        trainer.load_checkpoint(resume_from)
    trainer.run()


if __name__ == "__main__":
    main()
