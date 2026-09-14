"""Config loading, dotlist override, and ablation resolution.

Supports:
  --config configs/experiments/slmf_full.yaml
  --ablation no_gabor              (looks up ablations.yaml)
  --override modules.gabor.enabled=false  (repeatable)
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_INT_PATTERN = re.compile(r"^[+-]?\d+$")
_FLOAT_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)|(?:\d+\.\d*|\.\d+))$"
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return _normalize_scalar_strings(yaml.safe_load(fh))


def _normalize_scalar_strings(value: Any) -> Any:
    """Recursively convert numeric-looking strings into Python scalars.

    PyYAML may keep scientific notation like ``1e-4`` as a string under
    ``safe_load``. Training configs use this notation heavily, so we normalise
    after load and after CLI overrides.
    """
    if isinstance(value, dict):
        return {k: _normalize_scalar_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_scalar_strings(v) for v in value]
    if isinstance(value, str):
        if _INT_PATTERN.fullmatch(value):
            return int(value)
        if _FLOAT_PATTERN.fullmatch(value):
            return float(value)
    return value


def apply_dotlist_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Set nested keys via dotted paths, e.g. ``modules.gabor.enabled``.

    Creates intermediate dicts as needed.  Returns a new (deep-copied) dict.
    """
    out = deepcopy(config)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        cursor = out
        for part in parts[:-1]:
            if part not in cursor:
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return out


def resolve_ablation_overrides(
    ablation_config_path: str,
    ablation_name: str,
) -> Dict[str, Any]:
    """Load an ablation preset and return its overrides dict.

    Expected YAML structure::

        ablations:
          baseline:
            description: "..."
            overrides:
              modules.gabor.enabled: false
    """
    ablations = _load_yaml(ablation_config_path)
    entry = ablations.get("ablations", {}).get(ablation_name)
    if entry is None:
        available = list(ablations.get("ablations", {}).keys())
        raise KeyError(
            f"Ablation '{ablation_name}' not found. Available: {available}"
        )
    return entry.get("overrides", {})


def load_full_config(
    config_path: str,
    ablation: Optional[str] = None,
    ablation_config_path: str = "configs/experiments/ablations.yaml",
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Top-level entry point: load YAML, apply ablation, apply CLI overrides."""

    cfg = _load_yaml(config_path)

    if ablation:
        ablation_overrides = resolve_ablation_overrides(ablation_config_path, ablation)
        cfg = apply_dotlist_overrides(cfg, ablation_overrides)

    if overrides:
        for raw in overrides:
            if "=" not in raw:
                raise ValueError(f"Override must be key=value, got '{raw}'")
            key, value_str = raw.split("=", 1)
            # Attempt YAML / JSON parse for non-string types
            try:
                value = yaml.safe_load(value_str)
            except Exception:
                value = value_str
            cfg = apply_dotlist_overrides(cfg, {key: _normalize_scalar_strings(value)})

    return cfg


def save_resolved_config(cfg: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "resolved_config.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, allow_unicode=True)


def resolve_runtime_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply CPU-safe defaults when no CUDA is available."""
    import torch

    cfg = deepcopy(cfg)
    if not torch.cuda.is_available():
        cfg.setdefault("data", {})["image_size"] = min(
            int(cfg.get("data", {}).get("image_size", 192)), 32
        )
        cfg["data"]["batch_size"] = 1
        cfg.setdefault("runtime", {})["num_workers"] = 0
        cfg["runtime"]["eval_sampling_steps"] = 2
        cfg["runtime"]["torch_compile"] = False
        print("CUDA not available. Running CPU smoke/debug profile only.")
    return cfg
