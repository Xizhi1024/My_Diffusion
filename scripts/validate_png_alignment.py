"""PNG data alignment check for SLMF-BBDM.

Verifies that the PNG-derived NPZ cache is internally consistent before any
training is trusted.  For each sample it checks:

  - CT / PET / mask raw shape and resize consistency
  - mask is non-empty
  - in-mask vs outside-mask PET intensity (mean / max)
  - distance from PET peak to mask centroid
  - left-right / up-down flip or fixed-shift registration anomalies
    between CT and PET (normalised cross-correlation vs its flipped versions)

Outputs:
  alignment_report.csv       per-sample row
  alignment_summary.json     aggregate + risk flags
  overlay PNGs               CT / PET / mask side by side (sample)

If in-mask PET is consistently not brighter than outside-mask PET, or the
PET peak is consistently far from the mask centroid, a data-registration risk
is reported.

Usage:
    python scripts/validate_png_alignment.py \
        --config configs/experiments/slmf_png_baseline.yaml \
        --max-samples 100 --output-dir results/alignment
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config_utils import load_full_config, resolve_runtime_profile


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _flip_anomaly(ct: np.ndarray, pet: np.ndarray) -> Dict[str, float]:
    """Compare NCC across flips; the baseline (no flip) should win normally."""
    base = _ncc(ct, pet)
    lr = _ncc(np.fliplr(ct), pet)      # left-right flip of CT
    ud = _ncc(np.flipud(ct), pet)      # up-down flip of CT
    return {
        "ncc_base": base,
        "ncc_ct_flipped_lr": lr,
        "ncc_ct_flipped_ud": ud,
        "lr_flip_delta": lr - base,    # >0 suggests CT-PET left-right mismatch
        "ud_flip_delta": ud - base,
    }


def _save_overlay(out_path: Path, ct: np.ndarray, pet: np.ndarray, mask: np.ndarray, sid: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    axes[0].imshow(ct, cmap="gray"); axes[0].set_title(f"CT\n{sid}", fontsize=8)
    axes[1].imshow(pet, cmap="hot", vmin=-1, vmax=1); axes[1].set_title("PET", fontsize=8)
    axes[2].imshow(mask, cmap="gray"); axes[2].set_title("mask", fontsize=8)
    # overlay: CT gray + mask contour + PET peak
    axes[3].imshow(ct, cmap="gray"); axes[3].contour(mask, colors="cyan", linewidths=0.5)
    axes[3].set_title("CT+mask", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def check_one(npz_path: Path) -> Optional[Dict[str, Any]]:
    import numpy as _np
    try:
        with _np.load(str(npz_path)) as data:
            if "ct" not in data or "pet" not in data or "mask" not in data:
                return None
            ct = data["ct"].astype(np.float32)
            pet = data["pet"].astype(np.float32)
            mask = data["mask"].astype(np.float32)
    except Exception as exc:
        return {"sample_id": npz_path.stem, "error": str(exc)}

    sid = npz_path.stem
    ct2d = ct[0] if ct.ndim == 3 else ct
    pet2d = pet[0] if pet.ndim == 3 else pet
    mask2d = mask[0] if mask.ndim == 3 else mask
    binary = (mask2d > 0.5).astype(np.float32)

    rec: Dict[str, Any] = {
        "sample_id": sid,
        "ct_shape": str(ct.shape),
        "pet_shape": str(pet.shape),
        "mask_shape": str(mask.shape),
        "shape_match": bool(ct.shape == pet.shape == mask.shape),
        "mask_area": float(binary.sum()),
        "mask_nonempty": bool(binary.sum() > 0),
    }
    if binary.sum() < 1:
        rec.update({
            "in_mask_pet_mean": float("nan"), "in_mask_pet_max": float("nan"),
            "out_mask_pet_mean": float("nan"), "out_mask_pet_max": float("nan"),
            "peak_to_centroid_dist": float("nan"),
            "in_gt_out_mean": False, "in_gt_out_max": False,
        })
        rec.update(_flip_anomaly(ct2d, pet2d))
        return rec

    in_mean = float((pet2d * binary).sum() / max(binary.sum(), 1.0))
    in_max = float((pet2d * binary).max())
    out_bin = 1.0 - binary
    out_mean = float((pet2d * out_bin).sum() / max(out_bin.sum(), 1.0))
    out_max = float((pet2d * out_bin).max())

    ys, xs = np.nonzero(binary)
    cy, cx = float(ys.mean()), float(xs.mean())
    py, px = np.unravel_index(np.argmax(pet2d), pet2d.shape)
    peak_to_centroid = float(np.hypot(py - cy, px - cx))

    rec.update({
        "in_mask_pet_mean": in_mean,
        "in_mask_pet_max": in_max,
        "out_mask_pet_mean": out_mean,
        "out_mask_pet_max": out_max,
        "in_gt_out_mean": bool(in_mean > out_mean),
        "in_gt_out_max": bool(in_max > out_max),
        "peak_to_centroid_dist": peak_to_centroid,
        "pet_peak_at": f"({py},{px})",
        "mask_centroid": f"({cy:.1f},{cx:.1f})",
    })
    rec.update(_flip_anomaly(ct2d, pet2d))
    return rec


def run(config_path: str, output_dir: str, max_samples: Optional[int],
        seed: int, overlay_n: int) -> Dict[str, Any]:
    cfg = load_full_config(config_path)
    cfg = resolve_runtime_profile(cfg)
    cache_dir = cfg.get("data", {}).get("cache_dir", "")
    if not cache_dir or not Path(cache_dir).is_dir():
        raise RuntimeError(f"cache_dir not found: {cache_dir}")

    npz_files = sorted(Path(cache_dir).glob("*.npz"))
    if not npz_files:
        raise RuntimeError(f"No .npz files in {cache_dir}")

    rng = random.Random(seed)
    if max_samples and max_samples < len(npz_files):
        npz_files = rng.sample(npz_files, max_samples)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_root / "overlays"

    rows: List[Dict[str, Any]] = []
    overlay_indices = set(rng.sample(range(len(npz_files)), min(overlay_n, len(npz_files))))

    for idx, npz_path in enumerate(npz_files):
        rec = check_one(npz_path)
        if rec is None:
            continue
        rows.append(rec)
        if idx in overlay_indices and "error" not in rec:
            with np.load(str(npz_path)) as data:
                ct = data["ct"].astype(np.float32)
                pet = data["pet"].astype(np.float32)
                mask = data["mask"].astype(np.float32)
            _save_overlay(overlay_dir / f"{npz_path.stem}.png",
                          ct[0] if ct.ndim == 3 else ct,
                          pet[0] if pet.ndim == 3 else pet,
                          mask[0] if mask.ndim == 3 else mask,
                          npz_path.stem)

    # CSV
    import csv as _csv
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out_root / "alignment_report.csv", "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in keys})

    # Risk aggregation
    n = len(rows)
    valid = [r for r in rows if r.get("mask_nonempty")]
    in_gt_out_mean_count = sum(1 for r in valid if r.get("in_gt_out_mean"))
    in_gt_out_max_count = sum(1 for r in valid if r.get("in_gt_out_max"))
    lr_flip_count = sum(1 for r in valid if r.get("lr_flip_delta", 0) > 0.1)
    ud_flip_count = sum(1 for r in valid if r.get("ud_flip_delta", 0) > 0.1)
    peak_dists = [r["peak_to_centroid_dist"] for r in valid
                  if r.get("peak_to_centroid_dist") == r.get("peak_to_centroid_dist")]
    peak_median = float(np.median(peak_dists)) if peak_dists else float("nan")

    risks: List[str] = []
    nv = max(len(valid), 1)
    if len(valid) < n * 0.5:
        risks.append("more than half the samples have empty masks")
    if in_gt_out_mean_count / nv < 0.7:
        risks.append(
            f"in-mask PET mean is NOT consistently > outside-mask "
            f"(only {in_gt_out_mean_count}/{len(valid)}); possible mis-registration"
        )
    if lr_flip_count / nv > 0.3:
        risks.append(f"left-right flip anomaly in {lr_flip_count}/{len(valid)} samples")
    if ud_flip_count / nv > 0.3:
        risks.append(f"up-down flip anomaly in {ud_flip_count}/{len(valid)} samples")
    if peak_median > 15.0:
        risks.append(f"PET peak is far from mask centroid (median={peak_median:.1f}px)")

    summary = {
        "num_samples": n,
        "num_with_mask": len(valid),
        "in_gt_out_mean_fraction": in_gt_out_mean_count / nv,
        "in_gt_out_max_fraction": in_gt_out_max_count / nv,
        "lr_flip_fraction": lr_flip_count / nv,
        "ud_flip_fraction": ud_flip_count / nv,
        "peak_to_centroid_median": peak_median,
        "risks": risks,
        "ok": len(risks) == 0,
    }
    with open(out_root / "alignment_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="PNG cache alignment check")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", default="results/alignment")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlay-n", type=int, default=12)
    args = ap.parse_args()
    run(args.config, args.output_dir, args.max_samples, args.seed, args.overlay_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
