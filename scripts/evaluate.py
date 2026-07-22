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

    # Reproducible PNG baseline checkpoint evaluation (EMA + fixed 16 cases)
    python scripts/evaluate.py \\
        --config configs/experiments/slmf_png_baseline.yaml \\
        --checkpoint checkpoints/slmf_png_baseline/ckpt_best_combined.pt \\
        --weights ema --split val --max-samples 16 --seed 42 \\
        --output results/slmf_png_baseline_best_combined_eval.json

    # CPU-only smoke test
    python scripts/evaluate.py --config configs/experiments/slmf_baseline.yaml --fake-data
"""


import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy.ndimage import binary_dilation, binary_erosion, correlate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM
from src.model.loss_terms.roi_suv import _de_collate_meta
from src.model.trainer import (
    _compute_pet_sample_metrics,
    _stripe_score,
    _stratified_indices,
    _to_unit_interval,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_stripe_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compare directional-gradient anisotropy against the target image."""
    pred_score = _stripe_score(pred)
    target_score = _stripe_score(target)
    return {
        "stripe_score": float(pred_score),
        "target_stripe_score": float(target_score),
        "stripe_excess": float(pred_score - target_score),
    }


def _gradient_magnitude_np(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32).squeeze()
    dx = np.zeros_like(array)
    dy = np.zeros_like(array)
    dx[:, :-1] = array[:, 1:] - array[:, :-1]
    dy[:-1, :] = array[1:, :] - array[:-1, :]
    return np.sqrt(dx * dx + dy * dy)


def _boundary_ring_np(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = np.asarray(mask).squeeze() > 0.5
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return binary_dilation(binary, structure=structure) & ~binary_erosion(
        binary, structure=structure
    )


def _weighted_mean_np(values: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.asarray(weight, dtype=np.float64).sum())
    if denominator <= 1e-12:
        return float("nan")
    return float((np.asarray(values, dtype=np.float64) * weight).sum() / denominator)


def _normalized_edge_np(image: np.ndarray) -> np.ndarray:
    edge = _gradient_magnitude_np(image)
    maximum = float(edge.max())
    return edge / maximum if maximum > 1e-8 else np.zeros_like(edge)


def _gabor_orientation_spectrum(
    image: np.ndarray,
    orientations: int = 8,
    kernel_size: int = 9,
) -> np.ndarray:
    """Fixed quadrature-Gabor energy spectrum, independent of model weights."""
    array = np.asarray(image, dtype=np.float64).squeeze()
    coords = np.arange(kernel_size, dtype=np.float64) - kernel_size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    sigma = kernel_size / 4.0
    frequency = 0.2
    energies = []
    for index in range(orientations):
        theta = index * np.pi / orientations
        x_theta = xx * np.cos(theta) + yy * np.sin(theta)
        y_theta = -xx * np.sin(theta) + yy * np.cos(theta)
        envelope = np.exp(-(x_theta ** 2 + y_theta ** 2) / (2.0 * sigma ** 2))
        phase = 2.0 * np.pi * frequency * x_theta
        real = envelope * np.cos(phase)
        imag = envelope * np.sin(phase)
        real -= real.mean()
        imag -= imag.mean()
        real /= max(float(np.linalg.norm(real)), 1e-8)
        imag /= max(float(np.linalg.norm(imag)), 1e-8)
        response_real = correlate(array, real, mode="reflect")
        response_imag = correlate(array, imag, mode="reflect")
        energies.append(float(np.sqrt(response_real ** 2 + response_imag ** 2).mean()))
    spectrum = np.asarray(energies, dtype=np.float64)
    total = float(spectrum.sum())
    return spectrum / total if total > 1e-12 else np.zeros_like(spectrum)


def compute_directional_spectrum_error(
    pred: np.ndarray,
    target: np.ndarray,
    orientations: int = 8,
) -> float:
    """Compare phase-insensitive direction spectra instead of enforcing isotropy."""
    pred_spectrum = _gabor_orientation_spectrum(pred, orientations=orientations)
    target_spectrum = _gabor_orientation_spectrum(target, orientations=orientations)
    return float(np.abs(pred_spectrum - target_spectrum).mean())


def compute_boundary_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    ct: np.ndarray,
    lesion_mask: np.ndarray,
    organ_mask: np.ndarray,
    boundary_radius: int = 2,
) -> Dict[str, float]:
    """Normalized-space lesion, anatomy-consensus, and optional organ metrics."""
    pred_2d = np.asarray(pred, dtype=np.float32).squeeze()
    target_2d = np.asarray(target, dtype=np.float32).squeeze()
    absolute = np.abs(pred_2d - target_2d)
    gradient_error = np.abs(
        _gradient_magnitude_np(pred_2d) - _gradient_magnitude_np(target_2d)
    )

    lesion = np.asarray(lesion_mask).squeeze() > 0.5
    if lesion.any():
        lesion_ring = _boundary_ring_np(lesion, boundary_radius)
        lesion_intensity = _weighted_mean_np(absolute, lesion_ring)
        lesion_gradient = _weighted_mean_np(gradient_error, lesion_ring)
    else:
        lesion_intensity = float("nan")
        lesion_gradient = float("nan")

    anatomy_consensus = _normalized_edge_np(ct) * _normalized_edge_np(target)
    anatomy_gradient = _weighted_mean_np(gradient_error, anatomy_consensus)

    organs = np.asarray(organ_mask)
    organ_ring = np.zeros_like(pred_2d, dtype=bool)
    if organs.ndim == 2:
        organs = organs[None]
    for channel in organs:
        if np.any(channel > 0.5):
            organ_ring |= _boundary_ring_np(channel, boundary_radius)
    organ_gradient = _weighted_mean_np(gradient_error, organ_ring)

    return {
        "lesion_boundary_intensity_mae_norm": lesion_intensity,
        "lesion_boundary_gradient_mae_norm": lesion_gradient,
        "anatomy_edge_gradient_mae_norm": anatomy_gradient,
        "organ_boundary_gradient_mae_norm": organ_gradient,
        "directional_spectrum_error_norm": compute_directional_spectrum_error(
            pred, target
        ),
    }

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


def _positive_pet_for_detection(
    x: np.ndarray,
    *,
    model_space: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if model_space:
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
    model_space: bool = False,
) -> Dict[str, Any]:
    """Flag clinically risky samples: off-mask peaks, cold lesions, or high uncertainty."""
    pred_pos = _positive_pet_for_detection(pred, model_space=model_space)
    target_pos = _positive_pet_for_detection(target, model_space=model_space)
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
    topk_percent: float = 0.10,
    min_k: int = 3,
    max_k: int = 16,
) -> Dict[str, float]:
    """Lesion intensity metrics in normalised [-1, 1] space.

    These do NOT require physical SUV calibration and are the primary
    lesion-fidelity metrics for the PNG baseline.  Emitted for every sample
    that has a non-empty lesion mask, regardless of ``suv_ok``.

    Never convert these values back to pseudo-SUV — that would fabricate
    physical units the PNG export does not have.
    """
    if not 0.0 < topk_percent <= 1.0:
        raise ValueError("topk_percent must be in (0, 1]")
    if min_k < 1:
        raise ValueError("min_k must be at least 1")
    if max_k < min_k:
        raise ValueError("max_k must be greater than or equal to min_k")

    nan_metrics = {
        "lesion_peak_error_norm": float("nan"),
        "lesion_peak_signed_bias_norm": float("nan"),
        "lesion_topq_peak_pred_norm": float("nan"),
        "lesion_topq_peak_target_norm": float("nan"),
        "lesion_topq_peak_signed_bias_norm": float("nan"),
        "lesion_topq_cold_bias_norm": float("nan"),
        "lesion_topq_peak_error_norm": float("nan"),
        "lesion_peak_overestimated": float("nan"),
        "lesion_peak_underestimated": float("nan"),
        "lesion_topq_k": float("nan"),
        "lesion_size": float("nan"),
        "lesion_mean_error_norm": float("nan"),
        "lesion_to_background_ratio_norm": float("nan"),
        "lesion_centroid_distance": float("nan"),
        "lesion_core_fallback": float("nan"),
        "lesion_peak_to_boundary_distance": float("nan"),
        "lesion_core_topq_pred_norm": float("nan"),
        "lesion_core_topq_target_norm": float("nan"),
        "lesion_core_topq_error_norm": float("nan"),
        "lesion_ring_topq_pred_norm": float("nan"),
        "lesion_ring_topq_target_norm": float("nan"),
        "lesion_ring_topq_error_norm": float("nan"),
        "lesion_ring_core_ratio_pred": float("nan"),
        "lesion_ring_core_ratio_target": float("nan"),
        "lesion_ring_core_ratio_error": float("nan"),
    }
    lesion = lesion_mask > 0.5
    if not lesion.any():
        return nan_metrics

    pred_unit = _to_unit_interval(pred)
    target_unit = _to_unit_interval(target)
    pred_mean = float(pred_unit[lesion].mean())
    target_mean = float(target_unit[lesion].mean())

    organ_any = (
        organ_mask.sum(axis=0, keepdims=True) > 0.5
        if organ_mask.shape[0] > 0 else np.zeros_like(lesion, dtype=bool)
    )
    background = ~(lesion | organ_any)
    pred_bg = float(pred_unit[background].mean()) if background.any() else 0.0
    target_bg = float(target_unit[background].mean()) if background.any() else 0.0
    pred_tbr = pred_mean / max(pred_bg, 1e-6)
    target_tbr = target_mean / max(target_bg, 1e-6)

    peak_metrics = _compute_pet_sample_metrics(
        pred[0],
        target[0],
        lesion_mask[0],
        topk_percent=topk_percent,
        min_k=min_k,
        max_k=max_k,
    )
    assert peak_metrics is not None

    lesion_2d = lesion[0]
    structure = np.ones((3, 3), dtype=bool)
    core = binary_erosion(lesion_2d, structure=structure)
    core_fallback = not bool(core.any())
    if core_fallback:
        core = lesion_2d.copy()
    ring = binary_dilation(
        lesion_2d, structure=np.ones((5, 5), dtype=bool)
    ) & ~core

    def _topq_mean(image: np.ndarray, region: np.ndarray) -> float:
        values = image[0][region]
        if values.size == 0:
            return float("nan")
        k = min(
            max(int(np.ceil(values.size * topk_percent)), min_k),
            max_k,
            values.size,
        )
        return float(np.partition(values, -k)[-k:].mean())

    core_pred = _topq_mean(pred_unit, core)
    core_target = _topq_mean(target_unit, core)
    ring_pred = _topq_mean(pred_unit, ring)
    ring_target = _topq_mean(target_unit, ring)

    boundary = lesion_2d & ~binary_erosion(lesion_2d, structure=structure)
    lesion_yx = np.argwhere(lesion_2d)
    peak_local_index = int(np.argmax(pred_unit[0][lesion_2d]))
    peak_yx = lesion_yx[peak_local_index]
    boundary_yx = np.argwhere(boundary)
    peak_to_boundary = float(
        np.sqrt(((boundary_yx - peak_yx) ** 2).sum(axis=1)).min()
    )

    eps = 1e-6
    ring_core_pred = ring_pred / max(core_pred, eps)
    ring_core_target = ring_target / max(core_target, eps)

    return {
        **peak_metrics,
        "lesion_topq_cold_bias_norm": max(
            -float(peak_metrics["lesion_topq_peak_signed_bias_norm"]), 0.0
        ),
        "lesion_mean_error_norm": float(abs(pred_mean - target_mean)),
        "lesion_to_background_ratio_norm": float(abs(pred_tbr - target_tbr)),
        "lesion_core_fallback": float(core_fallback),
        "lesion_peak_to_boundary_distance": peak_to_boundary,
        "lesion_core_topq_pred_norm": core_pred,
        "lesion_core_topq_target_norm": core_target,
        "lesion_core_topq_error_norm": abs(core_pred - core_target),
        "lesion_ring_topq_pred_norm": ring_pred,
        "lesion_ring_topq_target_norm": ring_target,
        "lesion_ring_topq_error_norm": abs(ring_pred - ring_target),
        "lesion_ring_core_ratio_pred": ring_core_pred,
        "lesion_ring_core_ratio_target": ring_core_target,
        "lesion_ring_core_ratio_error": abs(ring_core_pred - ring_core_target),
    }


def _select_checkpoint_state(
    checkpoint: Dict[str, Any],
    weights: str = "ema",
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Select raw weights or overlay an old-style EMA shadow on model state."""
    raw_state = checkpoint.get("model", checkpoint)
    if weights == "raw":
        return raw_state, "model"
    if weights != "ema":
        raise ValueError(f"Unknown checkpoint weights: {weights!r}")

    for key in ("ema_model", "model_ema"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key

    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, dict):
        shadow = ema_state.get("shadow")
        if isinstance(shadow, dict):
            merged = dict(raw_state)
            merged.update(shadow)
            return merged, "ema.shadow"

    raise KeyError(
        "EMA weights requested, but checkpoint has no ema.shadow, ema_model, or model_ema"
    )


def _seed_evaluation(seed: int) -> None:
    """Seed every RNG used by the standalone evaluator."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_stratified_subset(dataset, count: int):
    """Choose deterministic lesion-area quantiles from an evaluation dataset."""
    total = len(dataset)
    if count <= 0:
        raise ValueError(f"Evaluation subset size must be positive, got {count}")
    if total <= count:
        return dataset, list(range(total))

    ranked: List[Tuple[float, int]] = []
    for index in range(total):
        sample = dataset[index]
        mask = sample.get("mask") if isinstance(sample, dict) else None
        if torch.is_tensor(mask):
            area = float(mask.sum().item())
        elif mask is not None:
            area = float(np.asarray(mask).sum())
        else:
            area = 0.0
        ranked.append((area, index))

    ranked.sort(key=lambda item: (item[0], item[1]))
    positions = _stratified_indices(total, count)
    indices = [ranked[position][1] for position in positions]
    return Subset(dataset, indices), indices


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


def _annotate_small_lesion_metrics(
    all_metrics: List[Dict[str, Any]],
    quantile: float = 0.25,
    underestimate_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Annotate the smallest lesion-area quantile without leaking model output.

    The threshold depends only on ground-truth mask area and is shared by every
    model evaluated on the same deterministic subset.  Derived keys are named so
    the normal per-patient aggregation produces the fields consumed by paired
    comparison plans.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("small-lesion quantile must be in (0, 1]")
    if not 0.0 <= underestimate_tolerance <= 1.0:
        raise ValueError("small-lesion underestimate tolerance must be in [0, 1]")

    areas = sorted(
        float(row["lesion_size"])
        for row in all_metrics
        if isinstance(row.get("lesion_size"), (int, float))
        and np.isfinite(float(row["lesion_size"]))
        and float(row["lesion_size"]) > 0.0
    )
    if not areas:
        return {
            "small_lesion_quantile": float(quantile),
            "small_lesion_underestimate_tolerance": float(
                underestimate_tolerance
            ),
            "small_lesion_area_threshold": None,
            "small_lesion_sample_count": 0,
        }

    cutoff_index = min(max(math.ceil(len(areas) * quantile) - 1, 0), len(areas) - 1)
    threshold = float(areas[cutoff_index])
    small_count = 0
    for row in all_metrics:
        area_value = row.get("lesion_size")
        is_small = (
            isinstance(area_value, (int, float))
            and np.isfinite(float(area_value))
            and 0.0 < float(area_value) <= threshold
        )
        row["small_lesion"] = float(is_small)
        if not is_small:
            continue
        small_count += 1
        topq_error = row.get("lesion_topq_peak_error_norm")
        signed_bias = row.get("lesion_topq_peak_signed_bias_norm")
        if isinstance(topq_error, (int, float)) and np.isfinite(float(topq_error)):
            row["small_lesion_topq_peak_error_norm"] = float(topq_error)
        if isinstance(signed_bias, (int, float)) and np.isfinite(float(signed_bias)):
            # Positive magnitude of cold bias; lower is unambiguously better.
            row["small_lesion_cold_bias_norm"] = max(-float(signed_bias), 0.0)
        if isinstance(signed_bias, (int, float)) and np.isfinite(float(signed_bias)):
            row["small_lesion_underestimate"] = float(
                float(signed_bias) < -underestimate_tolerance
            )

    return {
        "small_lesion_quantile": float(quantile),
        "small_lesion_underestimate_tolerance": float(underestimate_tolerance),
        "small_lesion_area_threshold": threshold,
        "small_lesion_sample_count": int(small_count),
    }


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
    small_lesion_quantile: float = 0.25,
    small_lesion_underestimate_tolerance: float = 0.05,
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
            ct_np = _to_numpy(ct[i])
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
            sample_metrics.update(compute_stripe_metrics(pred_np, target_np))
            sample_metrics.update(
                compute_boundary_metrics(
                    pred_np,
                    target_np,
                    ct_np,
                    mask_np,
                    organ_np,
                )
            )

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
            failure_model_space = True
            if has_valid_suv:
                try:
                    failure_pred = _denormalise_pet_np(pred_np, meta)
                    if torch.is_tensor(pet_suv) and _meta_bool(meta.get("pet_suv_available", False)):
                        failure_target = _to_numpy(pet_suv[i])
                    else:
                        failure_target = _denormalise_pet_np(target_np, meta)
                    failure_model_space = False
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
                model_space=failure_model_space,
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

    # Derive mask-area-only small-lesion strata before global and patient-level
    # aggregation. patient_results holds the same row objects as all_metrics.
    small_lesion_info = _annotate_small_lesion_metrics(
        all_metrics,
        quantile=small_lesion_quantile,
        underestimate_tolerance=small_lesion_underestimate_tolerance,
    )
    for row in all_metrics:
        for key, value in row.items():
            if (
                key.startswith("small_lesion_")
                and isinstance(value, (int, float))
                and np.isfinite(float(value))
            ):
                results[key].append(float(value))

    # ---- Aggregate statistics ----
    summary: Dict[str, Any] = {
        "num_samples": len(all_metrics),
        "num_patients": len(patient_results),
        # Derived from data: True only if at least one sample had suv_ok=True
        # AND a valid pet_suv_max.  PNG cache → always False.
        "physical_suv_available": bool(
            any(row.get("suv_valid", 0) > 0 for row in all_metrics)
        ),
        **small_lesion_info,
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
        ("Small-Lesion Stratum (mask-area bottom quantile)",
         ["small_lesion_topq_peak_error_norm", "small_lesion_cold_bias_norm",
          "small_lesion_underestimate"]),
        ("Clinical SUV (lesion ROI)", ["suv_max_error", "suv_mean_error", "tbr_error",
                                        "pred_suv_max", "target_suv_max"]),
        ("SUV Calibration", ["suv_calib_slope", "suv_calib_intercept", "suv_calib_r2",
                             "suv_calib_mae", "suv_calib_bias"]),
        ("Uncertainty", ["lesion_uncertainty_mean", "outside_uncertainty_mean",
                         "uncertainty_ratio", "confidence_lesion_mean"]),
        ("Failure Detection", ["outside_inside_peak_ratio", "pred_target_suvmax_ratio",
                               "failure_outside_peak_gt_inside", "failure_lesion_too_cold",
                               "failure_high_uncertainty", "failure_any"]),
        ("Directional Artifacts", ["stripe_score", "target_stripe_score", "stripe_excess"]),
        ("Boundary Fidelity", ["lesion_boundary_intensity_mae_norm",
                               "lesion_boundary_gradient_mae_norm",
                               "anatomy_edge_gradient_mae_norm",
                               "organ_boundary_gradient_mae_norm",
                               "directional_spectrum_error_norm"]),
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
    ap.add_argument(
        "--weights",
        choices=["ema", "raw"],
        default="ema",
        help="Checkpoint weights to evaluate (default: EMA, matching trainer model selection)",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    ap.add_argument("--fake-data", action="store_true", help="Use fake data for smoke test")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None, help="Deterministic evaluation seed")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate a lesion-area-stratified subset of this size",
    )
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=None, help="Monte Carlo samples for uncertainty/failure analysis")
    ap.add_argument("--mc-steps", type=int, default=None, help="Sampling steps for evaluation/MC sampling")
    ap.add_argument("--failure-outside-margin", type=float, default=None)
    ap.add_argument("--failure-lesion-ratio", type=float, default=None)
    ap.add_argument("--small-lesion-quantile", type=float, default=None)
    ap.add_argument(
        "--small-lesion-underestimate-tolerance", type=float, default=None
    )
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
    run_cfg = config.get("runtime", {})
    eval_cfg = config.get("evaluation", {})
    failure_cfg = eval_cfg.get("failure_detection", {})
    eval_seed = int(
        args.seed
        if args.seed is not None
        else run_cfg.get("eval_seed", config.get("experiment", {}).get("seed", 42))
    )
    _seed_evaluation(eval_seed)
    print(f"Evaluation seed: {eval_seed}")
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
    small_lesion_quantile = (
        args.small_lesion_quantile
        if args.small_lesion_quantile is not None
        else eval_cfg.get("small_lesion_quantile", 0.25)
    )
    small_lesion_underestimate_tolerance = (
        args.small_lesion_underestimate_tolerance
        if args.small_lesion_underestimate_tolerance is not None
        else eval_cfg.get("small_lesion_underestimate_tolerance", 0.05)
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build model
    print("Building model...")
    model = SLMFBBDM.from_config(config)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state, state_source = _select_checkpoint_state(ckpt, weights=args.weights)
        model.load_state_dict(state)
        print(f"Checkpoint weights: {state_source}")
    else:
        print("WARNING: No checkpoint provided, using randomly initialised weights")

    # Build dataloader for the requested split
    from src.data.dataset import CachedDataset, FakeDataset, build_dataloaders
    data_cfg = config.get("data", {})
    run_cfg["num_workers"] = 0

    split_manifest = Path(data_cfg["split_manifest"]) if data_cfg.get("split_manifest") else None

    if data_cfg.get("use_fake_data", False):
        ds = FakeDataset(32, data_cfg.get("image_size", 192))
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
    else:
        raise RuntimeError("No cache_dir configured and fake_data is not enabled")

    original_count = len(ds)
    if args.max_samples is not None:
        ds, selected_indices = _build_stratified_subset(ds, args.max_samples)
        print(
            f"Using lesion-area-stratified subset: {len(ds)} samples "
            f"from {original_count} candidates (indices={selected_indices})"
        )

    eval_batch_size = data_cfg.get("val_batch_size", data_cfg.get("batch_size", 4))
    loader = DataLoader(ds, batch_size=eval_batch_size, shuffle=False, num_workers=0)

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
        small_lesion_quantile=small_lesion_quantile,
        small_lesion_underestimate_tolerance=(
            small_lesion_underestimate_tolerance
        ),
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
