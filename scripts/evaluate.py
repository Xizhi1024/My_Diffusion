"""Evaluation script for SLMF-BBDM: clinical and image-quality metrics.

Computes per-sample and aggregate:
  - Image quality: MAE, MSE, PSNR, SSIM
  - Clinical: SUVmax error, SUVmean error, TBR error (in physical SUV space)
  - False hotspot: count and mean intensity in background regions
  - Per-patient aggregation for patient-level statistics

Usage:
    python scripts/evaluate.py \\
        --config configs/experiments/slmf_full.yaml \\
        --checkpoint checkpoints/slmf_bbdm_full/ckpt_epoch0100.pt \\
        --split test \\
        --output results/eval_report.json

    # CPU-only smoke test
    python scripts/evaluate.py --config configs/experiments/slmf_baseline.yaml --fake-data
"""


import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import correlate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM
from src.model.loss_terms.roi_suv import _de_collate_meta


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def compute_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and mask.sum() > 0:
        return float(np.abs(pred - target)[mask > 0].mean())
    return float(np.abs(pred - target).mean())


def compute_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None and mask.sum() > 0:
        return float(((pred - target) ** 2)[mask > 0].mean())
    return float(((pred - target) ** 2).mean())


def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 2.0) -> float:
    mse = compute_mse(pred, target)
    if mse < 1e-12:
        return 100.0
    return float(20 * np.log10(max_val) - 10 * np.log10(mse))


def compute_ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Compute SSIM for a single 2D image (no extra dependencies).

    Pure numpy implementation using the standard SSIM formula with K1=0.01, K2=0.03.
    """
    H, W = pred.shape[-2], pred.shape[-1]
    if H < 7 or W < 7:
        return float("nan")

    p = pred.astype(np.float64).squeeze()
    t = target.astype(np.float64).squeeze()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 11x11 Gaussian window
    kernel_size = min(11, H, W)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    sigma = 1.5
    ax = np.arange(kernel_size) - kernel_size // 2
    gauss = np.exp(-0.5 * (ax / sigma) ** 2)
    gauss = gauss / gauss.sum()
    window = np.outer(gauss, gauss)

    # Convolve via correlate with mode='nearest'
    mu_p = correlate(p, window, mode='nearest')
    mu_t = correlate(t, window, mode='nearest')
    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = correlate(p ** 2, window, mode='nearest') - mu_p_sq
    sigma_t_sq = correlate(t ** 2, window, mode='nearest') - mu_t_sq
    sigma_pt = correlate(p * t, window, mode='nearest') - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2) + 1e-12)
    return float(ssim_map.mean())


def _meta_bool(value: Any) -> bool:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return False
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def _valid_suv_meta(meta: dict) -> bool:
    return _meta_bool(meta.get("suv_ok", False)) and meta.get("pet_suv_max") is not None


def _finite_pair_arrays(pred_values: np.ndarray, target_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_values, dtype=np.float64).reshape(-1)
    keep = np.isfinite(pred) & np.isfinite(target)
    return pred[keep], target[keep]


def compute_calibration_metrics(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    prefix: str = "suv_calib",
) -> Dict[str, float]:
    """Fit predicted SUV against target SUV and return calibration diagnostics."""
    pred, target = _finite_pair_arrays(pred_values, target_values)
    n = int(pred.size)
    metrics: Dict[str, float] = {f"{prefix}_n": float(n)}
    if n == 0:
        metrics.update({
            f"{prefix}_slope": float("nan"),
            f"{prefix}_intercept": float("nan"),
            f"{prefix}_r2": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_bias": float("nan"),
            f"{prefix}_limits_of_agreement_low": float("nan"),
            f"{prefix}_limits_of_agreement_high": float("nan"),
        })
        return metrics

    diff = pred - target
    metrics[f"{prefix}_mae"] = float(np.mean(np.abs(diff)))
    metrics[f"{prefix}_bias"] = float(np.mean(diff))
    diff_std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    metrics[f"{prefix}_limits_of_agreement_low"] = metrics[f"{prefix}_bias"] - 1.96 * diff_std
    metrics[f"{prefix}_limits_of_agreement_high"] = metrics[f"{prefix}_bias"] + 1.96 * diff_std

    if n < 2 or float(np.var(target)) < 1e-12:
        metrics[f"{prefix}_slope"] = float("nan")
        metrics[f"{prefix}_intercept"] = float("nan")
        metrics[f"{prefix}_r2"] = float("nan")
        return metrics

    slope, intercept = np.polyfit(target, pred, deg=1)
    fitted = slope * target + intercept
    ss_res = float(np.sum((pred - fitted) ** 2))
    ss_tot = float(np.sum((pred - pred.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    metrics[f"{prefix}_slope"] = float(slope)
    metrics[f"{prefix}_intercept"] = float(intercept)
    metrics[f"{prefix}_r2"] = float(r2)
    return metrics


def _positive_pet_for_detection(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if np.nanmin(x) < 0.0:
        return np.clip((x + 1.0) * 0.5, 0.0, None)
    return np.clip(x, 0.0, None)


def _masked_mean_np(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    if selected.size == 0:
        return float("nan")
    return float(np.mean(selected))


def compute_uncertainty_metrics(
    uncertainty: np.ndarray,
    lesion_mask: np.ndarray,
    confidence: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Summarise MC/variance maps inside and outside the lesion mask."""
    unc = np.asarray(uncertainty, dtype=np.float32)
    lesion = np.asarray(lesion_mask > 0.5)
    outside = ~lesion

    lesion_unc = _masked_mean_np(unc, lesion)
    outside_unc = _masked_mean_np(unc, outside)
    ratio = lesion_unc / max(outside_unc, 1e-8) if np.isfinite(lesion_unc) and np.isfinite(outside_unc) else float("nan")
    metrics = {
        "lesion_uncertainty_mean": lesion_unc,
        "outside_uncertainty_mean": outside_unc,
        "uncertainty_ratio": float(ratio),
    }
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32)
        metrics["confidence_lesion_mean"] = _masked_mean_np(conf, lesion)
    return metrics


def compute_failure_detection_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    lesion_mask: np.ndarray,
    uncertainty_metrics: Optional[Dict[str, float]] = None,
    outside_margin: float = 0.05,
    lesion_min_ratio: float = 0.6,
    uncertainty_ratio_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Flag clinically risky samples: off-mask peaks, cold lesions, or high uncertainty."""
    pred_pos = _positive_pet_for_detection(pred)
    target_pos = _positive_pet_for_detection(target)
    lesion = np.asarray(lesion_mask > 0.5)
    outside = ~lesion

    if not lesion.any():
        return {
            "inside_peak": float("nan"),
            "outside_peak": float("nan"),
            "outside_inside_peak_ratio": float("nan"),
            "pred_target_suvmax_ratio": float("nan"),
            "failure_outside_peak_gt_inside": 0.0,
            "failure_lesion_too_cold": 0.0,
            "failure_high_uncertainty": 0.0,
            "failure_any": 0.0,
            "failure_reason": "no_lesion_mask",
        }

    inside_peak = float(np.max(pred_pos[lesion]))
    target_peak = float(np.max(target_pos[lesion]))
    outside_peak = float(np.max(pred_pos[outside])) if outside.any() else 0.0
    outside_ratio = outside_peak / max(inside_peak, 1e-8)
    pred_target_ratio = inside_peak / max(target_peak, 1e-8)

    outside_fail = outside_peak > inside_peak + outside_margin
    cold_fail = pred_target_ratio < lesion_min_ratio
    uncertainty_ratio = float("nan")
    if uncertainty_metrics is not None:
        uncertainty_ratio = float(uncertainty_metrics.get("uncertainty_ratio", float("nan")))
    uncertainty_fail = np.isfinite(uncertainty_ratio) and uncertainty_ratio > uncertainty_ratio_threshold

    reasons = []
    if outside_fail:
        reasons.append("outside_peak")
    if cold_fail:
        reasons.append("lesion_cold")
    if uncertainty_fail:
        reasons.append("high_uncertainty")

    return {
        "inside_peak": inside_peak,
        "outside_peak": outside_peak,
        "outside_inside_peak_ratio": float(outside_ratio),
        "pred_target_suvmax_ratio": float(pred_target_ratio),
        "failure_outside_peak_gt_inside": float(outside_fail),
        "failure_lesion_too_cold": float(cold_fail),
        "failure_high_uncertainty": float(uncertainty_fail),
        "failure_any": float(bool(reasons)),
        "failure_reason": "|".join(reasons) if reasons else "ok",
    }


def _denormalise_pet_np(pet_norm: np.ndarray, meta: dict) -> np.ndarray:
    """Reverse [-1,1] -> physical SUV for a single sample."""
    if not _valid_suv_meta(meta):
        raise ValueError("SUV metrics require meta['suv_ok']=True and pet_suv_max")
    p01 = (pet_norm + 1.0) / 2.0
    suv_max = float(meta.get("pet_suv_max", 20.0))
    return p01 * suv_max


def compute_suv_metrics(
    pred: np.ndarray,         # [1, H, W] in [-1, 1]
    target: np.ndarray,       # [1, H, W] in [-1, 1]
    lesion_mask: np.ndarray,  # [1, H, W] binary
    organ_mask: np.ndarray,   # [6, H, W] one-hot
    meta: dict,
    target_suv: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute SUVmax, SUVmean, TBR errors in physical SUV space."""
    pred_suv = _denormalise_pet_np(pred, meta)
    if target_suv is None:
        target_suv = _denormalise_pet_np(target, meta)

    # SUVmax within lesion
    pred_max = (pred_suv * lesion_mask).max()
    target_max = (target_suv * lesion_mask).max()
    suv_max_error = float(abs(pred_max - target_max))

    # SUVmean within lesion
    mask_sum = max(lesion_mask.sum(), 1)
    pred_mean = (pred_suv * lesion_mask).sum() / mask_sum
    target_mean = (target_suv * lesion_mask).sum() / mask_sum
    suv_mean_error = float(abs(pred_mean - target_mean))

    # TBR: Tumour-to-Background Ratio
    organ_any = (organ_mask.sum(axis=0, keepdims=True) > 0).astype(np.float32)
    bg_mask = 1.0 - lesion_mask - organ_any
    bg_mask = np.maximum(bg_mask, 0.0)
    bg_sum = max(bg_mask.sum(), 1)
    pred_bg = (pred_suv * bg_mask).sum() / bg_sum
    target_bg = (target_suv * bg_mask).sum() / bg_sum
    pred_tbr = pred_mean / max(pred_bg, 1e-6)
    target_tbr = target_mean / max(target_bg, 1e-6)
    tbr_error = float(abs(pred_tbr - target_tbr))

    return {
        "suv_max_error": suv_max_error,
        "suv_mean_error": suv_mean_error,
        "tbr_error": tbr_error,
        "pred_suv_max": float(pred_max),
        "target_suv_max": float(target_max),
        "pred_suv_mean": float(pred_mean),
        "target_suv_mean": float(target_mean),
        "pred_tbr": float(pred_tbr),
        "target_tbr": float(target_tbr),
    }


def compute_false_hotspot_count(
    pred: np.ndarray,         # [1, H, W] in [-1, 1]
    organ_mask: np.ndarray,   # [6, H, W]
    threshold_percentile: float = 0.95,
) -> Dict[str, float]:
    """Count false hotspots in non-organ background."""
    organ_any = (organ_mask.sum(axis=0, keepdims=True) > 0).astype(np.float32)
    bg_mask = 1.0 - organ_any

    pred_bg = pred * bg_mask
    # Threshold: values above 95th percentile in background
    q = np.percentile(pred_bg[pred_bg > 0], threshold_percentile * 100) if pred_bg.max() > 0 else 0
    hotspots = (pred_bg > q).astype(np.float32)

    return {
        "false_hotspot_count": float(hotspots.sum()),
        "false_hotspot_density": float(hotspots.sum() / max(bg_mask.sum(), 1)),
        "false_hotspot_mean_intensity": float(pred_bg[hotspots > 0].mean()) if hotspots.sum() > 0 else 0.0,
    }


def compute_normalized_lesion_metrics(
    pred: np.ndarray,         # [1, H, W] in [-1, 1] (model output space)
    target: np.ndarray,       # [1, H, W] in [-1, 1]
    lesion_mask: np.ndarray,  # [1, H, W] binary
    organ_mask: np.ndarray,   # [C, H, W] one-hot (may be all-zero for PNG)
) -> Dict[str, float]:
    """Lesion intensity metrics in normalised [-1, 1] space.

    These do NOT require physical SUV calibration and are the primary
    lesion-fidelity metrics for the PNG baseline.  Emitted for every sample
    that has a non-empty lesion mask, regardless of ``suv_ok``.

    Never convert these values back to pseudo-SUV — that would fabricate
    physical units the PNG export does not have.
    """
    nan_metrics = {
        "lesion_peak_error_norm": float("nan"),
        "lesion_mean_error_norm": float("nan"),
        "lesion_to_background_ratio_norm": float("nan"),
        "lesion_centroid_distance": float("nan"),
    }
    if lesion_mask.sum() <= 0:
        return nan_metrics

    pred_max = float((pred * lesion_mask).max())
    target_max = float((target * lesion_mask).max())
    mask_sum = max(float(lesion_mask.sum()), 1.0)
    pred_mean = float((pred * lesion_mask).sum() / mask_sum)
    target_mean = float((target * lesion_mask).sum() / mask_sum)

    organ_any = (
        (organ_mask.sum(axis=0, keepdims=True) > 0).astype(np.float32)
        if organ_mask.shape[0] > 0 else np.zeros_like(lesion_mask)
    )
    bg_mask = np.maximum(1.0 - lesion_mask - organ_any, 0.0)
    bg_sum = max(float(bg_mask.sum()), 1.0)
    pred_bg = float((pred * bg_mask).sum() / bg_sum)
    target_bg = float((target * bg_mask).sum() / bg_sum)
    pred_tbr = pred_mean / max(abs(pred_bg), 1e-6)
    target_tbr = target_mean / max(abs(target_bg), 1e-6)

    # Distance (pixels) between the pred peak inside the mask and the mask centroid
    pred_masked = pred[0] * lesion_mask[0]
    ys, xs = np.nonzero(lesion_mask[0])
    if len(ys) > 0:
        cy, cx = float(ys.mean()), float(xs.mean())
        peak_idx = np.unravel_index(np.argmax(pred_masked), pred_masked.shape)
        py, px = float(peak_idx[0]), float(peak_idx[1])
        centroid_dist = float(np.hypot(py - cy, px - cx))
    else:
        centroid_dist = float("nan")

    return {
        "lesion_peak_error_norm": float(abs(pred_max - target_max)),
        "lesion_mean_error_norm": float(abs(pred_mean - target_mean)),
        "lesion_to_background_ratio_norm": float(abs(pred_tbr - target_tbr)),
        "lesion_centroid_distance": centroid_dist,
    }


def _append_calibration_summary(summary: Dict[str, Any], all_metrics: List[Dict[str, Any]]) -> None:
    pairs = [
        ("pred_suv_max", "target_suv_max", "suv_calib"),
        ("pred_suv_mean", "target_suv_mean", "suv_mean_calib"),
    ]
    for pred_key, target_key, prefix in pairs:
        pred_values = [
            row[pred_key]
            for row in all_metrics
            if pred_key in row and target_key in row
        ]
        target_values = [
            row[target_key]
            for row in all_metrics
            if pred_key in row and target_key in row
        ]
        summary.update(compute_calibration_metrics(np.asarray(pred_values), np.asarray(target_values), prefix=prefix))


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SLMFBBDM,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    amp: bool = True,
    save_samples: Optional[Path] = None,
    mc_samples: int = 1,
    mc_steps: Optional[int] = None,
    failure_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run full evaluation over a dataloader."""
    model.eval()
    model = model.to(device)

    results: Dict[str, List[float]] = defaultdict(list)
    patient_results: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    all_metrics: List[Dict[str, Any]] = []

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    failure_thresholds = failure_thresholds or {}

    for batch_idx, batch in enumerate(dataloader):
        batch_gpu = {
            k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        with torch.amp.autocast("cuda", enabled=amp and device == "cuda", dtype=amp_dtype):
            if mc_samples and mc_samples > 1:
                sample_out = model.sample_mc(batch_gpu, n_samples=mc_samples, num_steps=mc_steps, progress=False)
            else:
                sample_out = model.sample(batch_gpu, num_steps=mc_steps)

        synth_pet = sample_out["synthetic_pet"]  # [B, 1, H, W]
        uncertainty_map = sample_out.get("total_var", sample_out.get("epistemic_var"))
        confidence_map = sample_out.get("confidence_map")
        target_pet = batch_gpu["pet"]
        ct = batch_gpu["ct"]
        mask = batch_gpu.get("mask", torch.zeros_like(target_pet))
        organ_mask = batch_gpu.get("organ_mask", torch.zeros(target_pet.shape[0], 6, *target_pet.shape[2:]))
        pet_suv = batch_gpu.get("pet_suv")
        B = synth_pet.shape[0]
        meta_samples = _de_collate_meta(batch_gpu.get("meta"), B)

        for i in range(B):
            pred_np = _to_numpy(synth_pet[i])
            target_np = _to_numpy(target_pet[i])
            mask_np = _to_numpy(mask[i])
            organ_np = _to_numpy(organ_mask[i])
            meta = meta_samples[i] if i < len(meta_samples) else {}
            has_valid_suv = _valid_suv_meta(meta)
            uncertainty_metrics: Dict[str, float] = {}

            pid = meta.get("patient_id", f"sample_{batch_idx}_{i}")

            sample_metrics = {
                "sample_id": f"{batch_idx}_{i}",
                "patient_id": str(pid),
                "mae": compute_mae(pred_np, target_np),
                "mse": compute_mse(pred_np, target_np),
                "psnr": compute_psnr(pred_np, target_np),
                "ssim": compute_ssim(pred_np[0], target_np[0]),
                "suv_valid": float(has_valid_suv),
            }

            # Normalized-intensity lesion metrics: always emitted when a lesion
            # mask exists.  These are the PNG-baseline lesion-fidelity metrics
            # (no physical SUV needed).
            if mask_np.sum() > 0:
                sample_metrics.update(
                    compute_normalized_lesion_metrics(pred_np, target_np, mask_np, organ_np)
                )

            # Clinical SUV metrics are emitted only for validated SUV samples.
            if mask_np.sum() > 0 and has_valid_suv:
                target_suv_np = None
                if torch.is_tensor(pet_suv) and _meta_bool(meta.get("pet_suv_available", False)):
                    target_suv_np = _to_numpy(pet_suv[i])
                suv_m = compute_suv_metrics(pred_np, target_np, mask_np, organ_np, meta, target_suv=target_suv_np)
                sample_metrics.update(suv_m)

            if torch.is_tensor(uncertainty_map):
                conf_np = _to_numpy(confidence_map[i]) if torch.is_tensor(confidence_map) else None
                uncertainty_metrics = compute_uncertainty_metrics(_to_numpy(uncertainty_map[i]), mask_np, confidence=conf_np)
                sample_metrics.update(uncertainty_metrics)

            failure_pred = pred_np
            failure_target = target_np
            if has_valid_suv:
                try:
                    failure_pred = _denormalise_pet_np(pred_np, meta)
                    if torch.is_tensor(pet_suv) and _meta_bool(meta.get("pet_suv_available", False)):
                        failure_target = _to_numpy(pet_suv[i])
                    else:
                        failure_target = _denormalise_pet_np(target_np, meta)
                except ValueError:
                    failure_pred = pred_np
                    failure_target = target_np
            failure_metrics = compute_failure_detection_metrics(
                failure_pred,
                failure_target,
                mask_np,
                uncertainty_metrics=uncertainty_metrics or None,
                outside_margin=failure_thresholds.get("outside_margin", 0.05),
                lesion_min_ratio=failure_thresholds.get("lesion_min_ratio", 0.6),
                uncertainty_ratio_threshold=failure_thresholds.get("uncertainty_ratio_threshold", 2.0),
            )
            sample_metrics.update(failure_metrics)

            # False hotspots
            fh = compute_false_hotspot_count(pred_np, organ_np)
            sample_metrics.update(fh)

            all_metrics.append(sample_metrics)
            for k, v in sample_metrics.items():
                if isinstance(v, (int, float)) and not np.isnan(v):
                    results[k].append(v)
            patient_results[str(pid)].append(sample_metrics)

        if batch_idx % 10 == 0:
            print(f"  Evaluated {batch_idx + 1} batches ({len(all_metrics)} samples)")

    # ---- Aggregate statistics ----
    summary: Dict[str, Any] = {
        "num_samples": len(all_metrics),
        "num_patients": len(patient_results),
        # Derived from data: True only if at least one sample had suv_ok=True
        # AND a valid pet_suv_max.  PNG cache → always False.
        "physical_suv_available": bool(
            any(row.get("suv_valid", 0) > 0 for row in all_metrics)
        ),
    }

    for metric_name, values in results.items():
        vals = np.array([v for v in values if not np.isnan(v)])
        if len(vals) == 0:
            continue
        summary[f"{metric_name}_mean"] = float(vals.mean())
        summary[f"{metric_name}_std"] = float(vals.std())
        summary[f"{metric_name}_median"] = float(np.median(vals))
        summary[f"{metric_name}_min"] = float(vals.min())
        summary[f"{metric_name}_max"] = float(vals.max())

    # Per-patient aggregates
    patient_summary = {}
    for pid, p_metrics in patient_results.items():
        p_agg = {}
        for k in p_metrics[0]:
            vals = [m[k] for m in p_metrics if isinstance(m.get(k), (int, float)) and not np.isnan(m.get(k, float("nan")))]
            if vals:
                p_agg[f"{k}_mean"] = float(np.mean(vals))
        patient_summary[pid] = p_agg

    summary["per_patient"] = patient_summary
    _append_calibration_summary(summary, all_metrics)

    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(summary: Dict[str, Any]) -> None:
    """Pretty-print evaluation summary."""
    print("\n" + "=" * 60)
    print("SLMF-BBDM Evaluation Report")
    print("=" * 60)
    print(f"Samples: {summary['num_samples']}  |  Patients: {summary['num_patients']}")
    suv_avail = summary.get("physical_suv_available", False)
    print(f"physical_suv_available: {suv_avail}"
          + ("" if suv_avail else "  (PNG baseline: clinical SUV metrics omitted)"))
    print("-" * 60)

    sections = [
        ("Image Quality", ["mae", "mse", "psnr", "ssim"]),
        ("Normalized-Intensity Lesion ([-1,1] space, no SUV)",
         ["lesion_peak_error_norm", "lesion_mean_error_norm",
          "lesion_to_background_ratio_norm", "lesion_centroid_distance"]),
        ("Clinical SUV (lesion ROI)", ["suv_max_error", "suv_mean_error", "tbr_error",
                                        "pred_suv_max", "target_suv_max"]),
        ("SUV Calibration", ["suv_calib_slope", "suv_calib_intercept", "suv_calib_r2",
                             "suv_calib_mae", "suv_calib_bias"]),
        ("Uncertainty", ["lesion_uncertainty_mean", "outside_uncertainty_mean",
                         "uncertainty_ratio", "confidence_lesion_mean"]),
        ("Failure Detection", ["outside_inside_peak_ratio", "pred_target_suvmax_ratio",
                               "failure_outside_peak_gt_inside", "failure_lesion_too_cold",
                               "failure_high_uncertainty", "failure_any"]),
        ("False Hotspots", ["false_hotspot_count", "false_hotspot_density", "false_hotspot_mean_intensity"]),
    ]

    for section_name, keys in sections:
        print(f"\n{section_name}:")
        print(f"{'Metric':<35} {'Mean':>10} {'Std':>10} {'Median':>10}")
        print("-" * 65)
        for k in keys:
            mean_val = summary.get(f"{k}_mean")
            if mean_val is not None:
                std_val = summary.get(f"{k}_std", 0)
                med_val = summary.get(f"{k}_median", 0)
                print(f"{k:<35} {mean_val:>10.4f} {std_val:>10.4f} {med_val:>10.4f}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="SLMF-BBDM Evaluation")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    ap.add_argument("--fake-data", action="store_true", help="Use fake data for smoke test")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=None, help="Monte Carlo samples for uncertainty/failure analysis")
    ap.add_argument("--mc-steps", type=int, default=None, help="Sampling steps for evaluation/MC sampling")
    ap.add_argument("--failure-outside-margin", type=float, default=None)
    ap.add_argument("--failure-lesion-ratio", type=float, default=None)
    ap.add_argument("--failure-uncertainty-ratio", type=float, default=None)
    ap.add_argument(
        "--allow-train-fallback",
        action="store_true",
        help="If the requested split is absent, evaluate on train instead of failing.",
    )
    args = ap.parse_args()

    # Load config
    config = load_full_config(args.config)
    config = resolve_runtime_profile(config)
    if args.fake_data:
        config.setdefault("data", {})["use_fake_data"] = True
    eval_cfg = config.get("evaluation", {})
    failure_cfg = eval_cfg.get("failure_detection", {})
    mc_samples = args.mc_samples if args.mc_samples is not None else eval_cfg.get("mc_samples", 1)
    failure_outside_margin = (
        args.failure_outside_margin
        if args.failure_outside_margin is not None
        else failure_cfg.get("outside_margin", 0.05)
    )
    failure_lesion_ratio = (
        args.failure_lesion_ratio
        if args.failure_lesion_ratio is not None
        else failure_cfg.get("lesion_min_ratio", 0.6)
    )
    failure_uncertainty_ratio = (
        args.failure_uncertainty_ratio
        if args.failure_uncertainty_ratio is not None
        else failure_cfg.get("uncertainty_ratio_threshold", 2.0)
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    print("Building model...")
    model = SLMFBBDM.from_config(config)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
    else:
        print("WARNING: No checkpoint provided, using randomly initialised weights")

    # Build dataloader for the requested split
    from src.data.dataset import CachedDataset, FakeDataset, build_dataloaders
    data_cfg = config.get("data", {})
    run_cfg = config.get("runtime", {})
    run_cfg["num_workers"] = 0

    split_manifest = Path(data_cfg["split_manifest"]) if data_cfg.get("split_manifest") else None

    if data_cfg.get("use_fake_data", False):
        ds = FakeDataset(32, data_cfg.get("image_size", 192))
        loader = DataLoader(ds, batch_size=data_cfg.get("batch_size", 4), shuffle=False)
    elif data_cfg.get("cache_dir"):
        cache_dir = data_cfg["cache_dir"]
        try:
            ds = CachedDataset(cache_dir, split=args.split, augment=False, split_manifest=split_manifest)
        except ValueError:
            if not args.allow_train_fallback:
                raise RuntimeError(
                    f"No {args.split!r} samples found in {cache_dir}. "
                    "Refusing to fall back to train split; pass --allow-train-fallback "
                    "only for debugging."
                )
            print(f"[evaluate] No {args.split} samples in {cache_dir}, falling back to train split...")
            ds = CachedDataset(cache_dir, split="train", augment=False, split_manifest=split_manifest)
        loader = DataLoader(ds, batch_size=data_cfg.get("batch_size", 4), shuffle=False, num_workers=0)
    else:
        raise RuntimeError("No cache_dir configured and fake_data is not enabled")

    print(f"Evaluating on {len(loader.dataset)} samples...")

    # Run evaluation
    summary = evaluate(
        model, loader,
        device=device,
        amp=not args.no_amp,
        mc_samples=mc_samples,
        mc_steps=args.mc_steps,
        failure_thresholds={
            "outside_margin": failure_outside_margin,
            "lesion_min_ratio": failure_lesion_ratio,
            "uncertainty_ratio_threshold": failure_uncertainty_ratio,
        },
    )

    print_report(summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
