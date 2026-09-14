"""Train CT-to-PET comparison models without touching the main SLMF-BBDM code.

Example:
    python comparison_experiments/train_comparison.py --config comparison_experiments/configs/pix2pix.yaml
    python comparison_experiments/train_comparison.py --config comparison_experiments/configs/cpdm.yaml --override training.num_epochs=50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from comparison_experiments.models import build_comparison_model
from comparison_experiments.png_dataset import build_png_dataloaders
from comparison_experiments.trainers import OriginalProtocolTrainer
from src.model.config_utils import apply_dotlist_overrides, resolve_runtime_profile, save_resolved_config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in raw for ch in [".", "e", "E"]):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _load_config(path: str, overrides: list[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if overrides:
        override_dict = {}
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must be key=value, got: {item}")
            key, value = item.split("=", 1)
            override_dict[key] = _parse_value(value)
        config = apply_dotlist_overrides(config, override_dict)
    return config


def _save_model_metadata(model, config: Dict[str, Any], ckpt_dir: Path) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment": config.get("experiment", {}).get("name", "comparison"),
        "comparison_model": config.get("comparison_model", config.get("model", {})),
        "trainable_parameters": model.get_trainable_params(),
        "total_parameters": model.get_total_params(),
        "interface": {
            "forward": "loss, logs = model(batch)",
            "sample": "model.sample(batch)['synthetic_pet']",
            "input_keys": ["ct", "pet"],
            "output_key": "synthetic_pet",
        },
    }
    with open(ckpt_dir / "comparison_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SLMF-aligned comparison models")
    parser.add_argument("--config", required=True, help="Comparison YAML config")
    parser.add_argument("--override", action="append", default=[], help="key=value overrides")
    args = parser.parse_args()

    config = _load_config(args.config, args.override)
    config = resolve_runtime_profile(config)
    _set_seed(config.get("experiment", {}).get("seed", 42))

    exp_name = config.get("experiment", {}).get("name", "comparison")
    ckpt_dir = Path("checkpoints") / exp_name
    save_resolved_config(config, str(ckpt_dir))

    model = build_comparison_model(config)
    _save_model_metadata(model, config, ckpt_dir)

    print(f"Building comparison model: {model.model_name}")
    print(f"  Trainable: {model.get_trainable_params():,} / Total: {model.get_total_params():,}")

    train_loader, val_loader = build_png_dataloaders(config.get("data", {}), config.get("runtime", {}))
    device_cfg = config.get("runtime", {}).get("device", None)
    if device_cfg in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_cfg

    trainer = OriginalProtocolTrainer(model, config, train_loader, val_loader, device=device)
    trainer.run()


if __name__ == "__main__":
    main()
