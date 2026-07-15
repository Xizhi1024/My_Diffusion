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


class PNGBaselineConfigError(ValueError):
    """Raised when a PNG-mode config activates a path that needs DICOM/SUV data."""


def _is_enabled(node: Any) -> bool:
    """Return True only when a config node explicitly enables something."""
    if isinstance(node, dict):
        return bool(node.get("enabled", False))
    return bool(node)


def validate_png_baseline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that a PNG-mode config does not activate DICOM/SUV/organ paths.

    Reads ``data.mode`` (the real field consumed by the startup checks). When it
    equals ``"png"`` the following must all be disabled, otherwise a
    :class:`PNGBaselineConfigError` is raised:

      - ``losses.roi_suv``        (needs physical SUV)
      - ``modules.organ_prior``   (needs TotalSegmentator masks)
      - ``losses.organ_consistency`` (needs organ masks)
      - ``metadata`` / ``model.metadata`` (FiLM needs DICOM metadata)
      - ``segmenter`` / ``model.segmenter``

    Returns a dict of startup status fields for logging.
    """
    data_cfg = config.get("data", {}) or {}
    data_mode = str(data_cfg.get("mode", "")).lower().strip()

    modules_cfg = config.get("modules", {}) or {}
    losses_cfg = config.get("losses", {}) or {}
    model_cfg = config.get("model", {}) or {}
    eval_cfg = config.get("evaluation", {}) or {}

    gabor_cfg = modules_cfg.get("gabor", {}) or {}
    gabor_routes = {
        "enabled": bool(gabor_cfg.get("enabled", False)),
        "inject_adapter": bool(gabor_cfg.get("inject_adapter", False)),
        "use_for_noise": bool(gabor_cfg.get("use_for_noise", False)),
        "use_for_hotspot": bool(gabor_cfg.get("use_for_hotspot", False)),
        "use_for_loss": bool(gabor_cfg.get("use_for_loss", False)),
    }

    status = {
        "data_mode": data_mode or "(unset)",
        "cache_dir": data_cfg.get("cache_dir", ""),
        "split_manifest": data_cfg.get("split_manifest", ""),
        "organ_prior_enabled": _is_enabled(modules_cfg.get("organ_prior")),
        "roi_suv_enabled": _is_enabled(losses_cfg.get("roi_suv")),
        "metadata_enabled": _is_enabled(config.get("metadata")) or _is_enabled(model_cfg.get("metadata")),
        "segmenter_enabled": _is_enabled(config.get("segmenter")) or _is_enabled(model_cfg.get("segmenter")),
        "gabor_routes": gabor_routes,
        "conditional_mean_enabled": _is_enabled(modules_cfg.get("conditional_mean")),
        "residual_bridge_enabled": _is_enabled(modules_cfg.get("residual_bridge")),
        "residual_frequency_enabled": _is_enabled(modules_cfg.get("residual_frequency")),
        "physical_suv_available": bool(eval_cfg.get("physical_suv_available", False)),
    }

    if data_mode != "png":
        # Not a PNG run — nothing to enforce. Callers may still log the status.
        return status

    # PNG mode: hard-fail on paths that need DICOM/SUV/organ ground truth.
    offenders = []
    if status["roi_suv_enabled"]:
        offenders.append(
            "losses.roi_suv is enabled but PNG cache has no physical SUV "
            "(suv_ok=False). Disable losses.roi_suv or switch data.mode."
        )
    if status["organ_prior_enabled"]:
        offenders.append(
            "modules.organ_prior is enabled but PNG cache has no organ_mask/"
            "organ_distance (TotalSegmentator unavailable). This would run "
            "biased convolutions on all-zero inputs. Disable modules.organ_prior."
        )
    if _is_enabled(losses_cfg.get("organ_consistency")):
        offenders.append(
            "losses.organ_consistency is enabled but PNG cache has no organ masks."
        )
    if status["metadata_enabled"]:
        offenders.append(
            "metadata FiLM is enabled but PNG cache has no DICOM metadata "
            "(uptake_min/weight_kg/age/...). Disable metadata.enabled."
        )
    if status["segmenter_enabled"]:
        offenders.append(
            "segmenter is enabled but requires a pretrained PET segmenter "
            "checkpoint; not part of the PNG baseline."
        )
    if offenders:
        msg = "PNG baseline config validation failed:\n  - " + "\n  - ".join(offenders)
        raise PNGBaselineConfigError(msg)

    return status


def log_startup_status(status: Dict[str, Any]) -> None:
    """Pretty-print the startup status dict produced by validate_png_baseline_config."""
    print("\n" + "=" * 60)
    print("SLMF-BBDM startup status")
    print("=" * 60)
    print(f"  data_mode            : {status.get('data_mode')}")
    print(f"  cache_dir            : {status.get('cache_dir')}")
    print(f"  split_manifest       : {status.get('split_manifest')}")
    print(f"  organ_prior_enabled  : {status.get('organ_prior_enabled')}")
    print(f"  roi_suv_enabled      : {status.get('roi_suv_enabled')}")
    print(f"  metadata_enabled     : {status.get('metadata_enabled')}")
    print(f"  segmenter_enabled    : {status.get('segmenter_enabled')}")
    print(f"  conditional_mean     : {status.get('conditional_mean_enabled')}")
    print(f"  residual_bridge      : {status.get('residual_bridge_enabled')}")
    print(f"  residual_frequency   : {status.get('residual_frequency_enabled')}")
    print(f"  physical_suv_available: {status.get('physical_suv_available')}")
    routes = status.get("gabor_routes", {})
    if routes:
        print("  gabor routes         : " + ", ".join(
            f"{k}={'on' if v else 'off'}" for k, v in routes.items()
        ))
    print("=" * 60 + "\n")
