"""Stripe diagnosis for SLMF-BBDM.

Runs the same checkpoint / samples / seed under several Gabor routing
configurations and reports which route is the dominant stripe source.

Configurations (all share the same checkpoint weights via strict=False load):
  A. gabor_off        : modules.gabor.enabled=false                    (reference)
  B. adapter_only     : enabled=true, inject_adapter=true, rest=false
  C. noise_only       : enabled=true, use_for_noise=true,  rest=false  (needs scale_adaptive)
  D. hotspot_only     : enabled=true, use_for_hotspot=true, rest=false (needs hotspot_prior)
  E. current          : the config's own gabor routes (baseline)

For every (sample × config) we save CT / Target / Pred / Mask / Gabor energy /
Hotspot maps, plus a per-row metrics CSV and a Gabor-filter parameter dump.

Usage:
    python scripts/diagnose_stripes.py \
        --config configs/experiments/slmf_png_baseline.yaml \
        --checkpoint checkpoints/slmf_png_baseline/ckpt_best_combined.pt \
        --split val --sample-ids 001001 002003 \
        --seed 42 --output-dir results/diag_stripes
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config_utils import load_full_config, resolve_runtime_profile, apply_dotlist_overrides
from src.model.slmf_bbdm import SLMFBBDM


# ---------------------------------------------------------------------------
# Routing presets
# ---------------------------------------------------------------------------

ROUTING_PRESETS: Dict[str, Dict[str, bool]] = {
    "A_gabor_off":    {"modules.gabor.enabled": False},
    "B_adapter_only": {
        "modules.gabor.enabled": True,
        "modules.gabor.inject_adapter": True,
        "modules.gabor.use_for_noise": False,
        "modules.gabor.use_for_hotspot": False,
        "modules.gabor.use_for_loss": False,
    },
    "C_noise_only": {
        "modules.gabor.enabled": True,
        "modules.gabor.inject_adapter": False,
        "modules.gabor.use_for_noise": True,
        "modules.gabor.use_for_hotspot": False,
        "modules.gabor.use_for_loss": False,
        "modules.scale_adaptive_noise.enabled": True,
        "modules.scale_adaptive_noise.name": "scale_adaptive",
        "modules.scale_adaptive_noise.use_gabor_energy": True,
        "modules.scale_adaptive_noise.bridge_mode": True,
    },
    "D_hotspot_only": {
        "modules.gabor.enabled": True,
        "modules.gabor.inject_adapter": False,
        "modules.gabor.use_for_noise": False,
        "modules.gabor.use_for_hotspot": True,
        "modules.gabor.use_for_loss": False,
        "modules.hotspot_prior.enabled": True,
        "modules.hotspot_prior.use_gabor_energy": True,
    },
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _stripe_score(pred: np.ndarray) -> float:
    from scipy.ndimage import correlate
    if pred.ndim == 3:
        pred = pred[0]
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.T
    gx = correlate(pred, kx, mode="reflect")
    gy = correlate(pred, ky, mode="reflect")
    n_dirs = 8
    energies = np.empty(n_dirs)
    for i in range(n_dirs):
        theta = i * np.pi / n_dirs
        dg = gx * np.cos(theta) + gy * np.sin(theta)
        energies[i] = (dg * dg).sum()
    return float(energies.max() / max(energies.mean(), 1e-8))


def _high_freq_ratio(pred: np.ndarray) -> float:
    """Fraction of FFT energy in the top 25% frequencies."""
    if pred.ndim == 3:
        pred = pred[0]
    f = np.abs(np.fft.fft2(pred))
    f = np.fft.fftshift(f)
    H, W = f.shape
    cy, cx = H // 2, W // 2
    total = f.sum() + 1e-12
    # distance from center
    yy, xx = np.indices(f.shape)
    dist = np.hypot(yy - cy, xx - cx)
    radius = 0.25 * min(H, W)
    hf = f[dist > radius].sum()
    return float(hf / total)


def _lesion_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if mask.sum() <= 0:
        return {"lesion_peak_error": float("nan"), "lesion_mean_error": float("nan"),
                "outside_inside_peak_ratio": float("nan"), "lesion_centroid_distance": float("nan")}
    p_in = float((pred * mask).max())
    t_in = float((target * mask).max())
    ms = max(float(mask.sum()), 1.0)
    out["lesion_peak_error"] = abs(p_in - t_in)
    out["lesion_mean_error"] = abs(float((pred * mask).sum() / ms) - float((target * mask).sum() / ms))
    out_peak = float((pred * (1.0 - mask)).max())
    out["outside_inside_peak_ratio"] = out_peak / max(p_in, 1e-6)
    ys, xs = np.nonzero(mask)
    if len(xs) > 0:
        cy, cx = ys.mean(), xs.mean()
        pm = pred * mask
        py, px = np.unravel_index(np.argmax(pm), pm.shape)
        out["lesion_centroid_distance"] = float(np.hypot(py - cy, px - cx))
    else:
        out["lesion_centroid_distance"] = float("nan")
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model construction per routing preset
# ---------------------------------------------------------------------------

def build_model_with_routes(base_config: Dict[str, Any], preset_name: str) -> Tuple[SLMFBBDM, Dict[str, bool]]:
    cfg = deepcopy(base_config)
    if preset_name == "E_current":
        gabor_cfg = cfg.get("modules", {}).get("gabor", {})
        routes = {k: bool(gabor_cfg.get(k, False)) for k in
                  ("enabled", "inject_adapter", "use_for_noise", "use_for_hotspot", "use_for_loss")}
        return SLMFBBDM.from_config(cfg), routes
    overrides = ROUTING_PRESETS[preset_name]
    cfg = apply_dotlist_overrides(cfg, overrides)
    gabor_cfg = cfg.get("modules", {}).get("gabor", {})
    routes = {k: bool(gabor_cfg.get(k, False)) for k in
              ("enabled", "inject_adapter", "use_for_noise", "use_for_hotspot", "use_for_loss")}
    return SLMFBBDM.from_config(cfg), routes


def load_weights(model: SLMFBBDM, ckpt_path: Optional[str], device: str) -> str:
    """Load checkpoint with strict=False; return a short status string."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        return "no_checkpoint"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    parts = [f"loaded"]
    if missing:
        parts.append(f"{len(missing)} missing")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_panel(out_path: Path, panels: Dict[str, np.ndarray], vmin: float = -1.0, vmax: float = 1.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (title, img) in zip(axes, panels.items()):
        if img.ndim == 3:
            img = img[0]
        ax.imshow(img, cmap="hot" if "pet" in title.lower() or "gabor" in title.lower() or "hotspot" in title.lower() else "gray",
                  vmin=vmin, vmax=vmax if "pet" in title.lower() else None)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main diagnostic loop
# ---------------------------------------------------------------------------

def diagnose(config_path: str, checkpoint: Optional[str], split: str,
             sample_ids: List[str], seed: int, output_dir: str,
             num_steps: Optional[int], device: Optional[str],
             fake_data: bool = False) -> Dict[str, Any]:
    base_config = load_full_config(config_path)
    if fake_data:
        base_config.setdefault("data", {})["use_fake_data"] = True
    base_config = resolve_runtime_profile(base_config)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(output_dir)

    # Data
    from src.data.dataset import CachedDataset, FakeDataset
    from torch.utils.data import DataLoader
    data_cfg = base_config.get("data", {})
    if data_cfg.get("use_fake_data"):
        ds = FakeDataset(8, data_cfg.get("image_size", 192))
    else:
        split_manifest = data_cfg.get("split_manifest")
        split_manifest = Path(split_manifest) if split_manifest else None
        try:
            ds = CachedDataset(data_cfg["cache_dir"], split=split, augment=False, split_manifest=split_manifest)
        except ValueError:
            ds = CachedDataset(data_cfg["cache_dir"], split="train", augment=False, split_manifest=split_manifest)

    # Pick samples
    wanted = set(sample_ids) if sample_ids else None
    picked: List[Dict[str, Any]] = []
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    for batch in loader:
        sid = ""
        meta = batch.get("meta")
        if isinstance(meta, list) and meta:
            sid = str(meta[0].get("patient_id", "")) if isinstance(meta[0], dict) else ""
        if wanted and sid not in wanted:
            continue
        picked.append({k: (v[0] if torch.is_tensor(v) else v) for k, v in batch.items()})
        if wanted and len(picked) >= len(wanted):
            break
        if not wanted and len(picked) >= 4:
            break
    if not picked:
        picked = [{k: (v[0] if torch.is_tensor(v) else v) for k, v in b.items()} for b in loader]
        picked = picked[:4]

    preset_names = ["A_gabor_off", "B_adapter_only", "C_noise_only", "D_hotspot_only", "E_current"]
    rows: List[Dict[str, Any]] = []
    filter_dump: List[Dict[str, Any]] = []

    for preset in preset_names:
        try:
            model, routes = build_model_with_routes(base_config, preset)
        except Exception as exc:
            print(f"[{preset}] skipped (model build failed: {exc})")
            continue
        status = load_weights(model, checkpoint, device)
        model = model.to(device).eval()
        print(f"[{preset}] routes={routes}  ckpt={status}")

        # Gabor filter parameter dump (once, when gabor is enabled and has params)
        gabor_prior = model.priors["gabor"] if "gabor" in model.priors else None
        if gabor_prior is not None and hasattr(gabor_prior, "log_frequency") and preset == "E_current":
            import math
            with torch.no_grad():
                freq = gabor_prior.log_frequency.exp().cpu().numpy()
                theta = gabor_prior.theta_raw.cpu().numpy()
                sigma = gabor_prior.log_sigma.exp().cpu().numpy()
                gamma = (0.25 + 1.75 * torch.sigmoid(gabor_prior.gamma_raw)).cpu().numpy()
                phase = gabor_prior.phase.cpu().numpy()
            for i in range(len(freq)):
                filter_dump.append({
                    "preset": preset, "filter_idx": i,
                    "frequency": float(freq[i]),
                    "theta_rad": float(theta[i]),
                    "theta_deg": float(np.degrees(theta[i])),
                    "sigma": float(sigma[i]),
                    "gamma": float(gamma[i]),
                    "phase": float(phase[i]),
                })

        for si, sample in enumerate(picked):
            _set_seed(seed)
            batch = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            with torch.no_grad():
                cond = model.build_condition_bundle(batch, torch.zeros(batch["ct"].shape[0], dtype=torch.long, device=device))
                result = model.sample(batch, num_steps=num_steps)
            pred = result["synthetic_pet"][0, 0].float().cpu().numpy()
            target = batch["pet"][0, 0].float().cpu().numpy()
            ct = batch["ct"][0, 0].float().cpu().numpy()
            mask = batch["mask"][0, 0].float().cpu().numpy() if "mask" in batch else np.zeros_like(target)
            gabor_energy = cond.get_map("gabor_energy")
            gabor_energy_np = gabor_energy[0, 0].float().cpu().numpy() if gabor_energy is not None else np.zeros_like(target)
            gabor_feat = cond.get_map("gabor_feat")
            max_chan = int(gabor_feat[0].abs().reshape(gabor_feat.shape[1], -1).mean(1).argmax().item()) if gabor_feat is not None else -1
            hotspot = result.get("hotspot_prior")
            hotspot_np = hotspot[0, 0].float().cpu().numpy() if hotspot is not None else np.zeros_like(target)

            lm = _lesion_metrics(pred, target, mask)
            row = {
                "preset": preset,
                "sample_idx": si,
                **{f"route_{k}": v for k, v in routes.items()},
                **lm,
                "stripe_score": _stripe_score(pred),
                "target_stripe_score": _stripe_score(target),
                "high_freq_ratio": _high_freq_ratio(pred),
                "target_high_freq_ratio": _high_freq_ratio(target),
                "gabor_pred_corr": _corr(gabor_energy_np, pred),
                "gabor_max_channel": max_chan,
            }
            rows.append(row)

            # Panels
            tag = f"{preset}_s{si}"
            _save_panel(out_root / preset / f"{tag}.png", {
                "CT": ct, "Target PET": target, "Pred PET": pred,
                "Lesion mask": mask, "Gabor energy": gabor_energy_np,
                "Hotspot prior": hotspot_np,
            })

    # Write CSV + JSON
    out_root.mkdir(parents=True, exist_ok=True)
    import csv as _csv
    if rows:
        with open(out_root / "stripe_metrics.csv", "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out_root / "gabor_filters.json", "w", encoding="utf-8") as fh:
        json.dump(filter_dump, fh, indent=2)

    # Summary: which preset has the highest stripe score?
    summary: Dict[str, Any] = {"per_preset": {}}
    if rows:
        from collections import defaultdict
        per: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            per[r["preset"]].append(r["stripe_score"])
        summary["per_preset"] = {k: float(np.mean(v)) for k, v in per.items()}
        ref = summary["per_preset"].get("A_gabor_off")
        if ref is not None:
            # Exclude the reference itself from the delta search
            deltas = {k: v - ref for k, v in summary["per_preset"].items() if k != "A_gabor_off"}
            summary["stripe_delta_vs_A"] = deltas
            if deltas and max(deltas.values()) > 0.02:
                worst = max(deltas, key=deltas.get)
                summary["dominant_route"] = worst
                summary["note"] = (
                    f"{worst} shows the largest stripe-score increase over the "
                    f"gabor_off reference (Δ={deltas[worst]:.4f}) → likely the "
                    f"main stripe source."
                )
            else:
                summary["dominant_route"] = None
                summary["note"] = (
                    "no route raises stripe score by more than 0.02 over the "
                    "gabor_off reference on these samples."
                )
    with open(out_root / "stripe_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="SLMF-BBDM stripe diagnosis")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="val")
    ap.add_argument("--sample-ids", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="results/diag_stripes")
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--fake-data", action="store_true", help="Use FakeDataset (smoke test)")
    args = ap.parse_args()
    diagnose(args.config, args.checkpoint, args.split, args.sample_ids or [],
             args.seed, args.output_dir, args.num_steps, args.device, args.fake_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
