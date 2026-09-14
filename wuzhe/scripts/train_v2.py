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

from src.model.config_utils import load_full_config, save_resolved_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM
from src.model.trainer import Trainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_run_metadata(model: SLMFBBDM, config: dict, ckpt_dir: str, ablation: str | None) -> None:
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
    })
    metadata = {
        "experiment": config.get("experiment", {}).get("name", "slmf_bbdm"),
        "ablation": ablation,
        "enabled_modules": enabled_modules,
        "enabled_losses": {
            name: bool(loss.enabled)
            for name, loss in model.loss_terms.items()
        },
        "trainable_parameters": model.get_trainable_params(),
        "total_parameters": model.get_total_params(),
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

    # 1.5 Set random seed for reproducibility (must happen before model/dataloader init)
    seed = config.get("experiment", {}).get("seed", 42)
    _set_seed(seed)

    # 2. Resolve runtime profile (CPU fallback if no CUDA)
    config = resolve_runtime_profile(config)

    # 3. Save resolved config
    exp_name = config.get("experiment", {}).get("name", "slmf_bbdm")
    ckpt_dir = os.path.join("checkpoints", exp_name)
    save_resolved_config(config, ckpt_dir)

    # 4. Build model
    print("Building SLMF-BBDM model...")
    model = SLMFBBDM.from_config(config)
    _save_run_metadata(model, config, ckpt_dir, args.ablation)

    enabled_priors = [n for n, p in model.priors.items() if p.enabled]
    enabled_losses = [n for n, loss in model.loss_terms.items() if loss.enabled]

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
    trainer.run()


if __name__ == "__main__":
    main()
