"""Pretrain the CT-conditioned PET mean used by residual diffusion variants."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import build_dataloaders
from src.data.lineage import load_checkpoint_data_lineage
from src.model.config_utils import (
    load_full_config,
    log_startup_status,
    resolve_runtime_profile,
    save_resolved_config,
    validate_png_baseline_config,
)
from src.model.mean_predictor import build_mean_predictor
from src.model.mean_pretraining import MeanPretrainer


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base experiment YAML")
    parser.add_argument("--output-dir", required=True, help="M0 checkpoint directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="key=value config override; repeatable",
    )
    args = parser.parse_args()

    config = load_full_config(
        args.config,
        overrides=args.override if args.override else None,
    )
    config = resolve_runtime_profile(config)
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("experiment", {}).get("seed", 42)
    )
    config.setdefault("experiment", {})["seed"] = seed
    _set_seed(seed)
    status = validate_png_baseline_config(config)
    log_startup_status(status)
    data_lineage = load_checkpoint_data_lineage(config)
    if data_lineage is not None:
        print(
            "Verified cache lineage: "
            f"{data_lineage['cache_metadata_sha256']}"
        )

    mean_cfg = config.get("modules", {}).get("conditional_mean", {})
    if not mean_cfg.get("enabled", False):
        raise ValueError("M0 requires modules.conditional_mean.enabled=true")
    predictor = build_mean_predictor(mean_cfg)

    train_loader, val_loader = build_dataloaders(
        config.get("data", {}), config.get("runtime", {})
    )
    if val_loader is None:
        raise ValueError("M0 requires a validation split for mean_best.pt selection")

    output_dir = Path(args.output_dir)
    save_resolved_config(config, str(output_dir))
    trainer = MeanPretrainer(
        predictor,
        train_loader,
        val_loader,
        config,
        device=config.get("runtime", {}).get("device")
        if config.get("runtime", {}).get("device") not in {None, "auto"}
        else None,
    )
    print(
        f"Pretraining conditional PET mean for {args.epochs} epochs on "
        f"{trainer.device} -> {output_dir}"
    )
    trainer.run(args.epochs, output_dir)
    print(f"Best conditional mean checkpoint: {output_dir / 'mean_best.pt'}")


if __name__ == "__main__":
    main()
