"""Sample the paired PNG dataset and quantify CT/PET frequency-geometry links.

The current PNG dataset provides CT, PET and a lesion mask, but no organ
segmentation.  Consequently, this script reports the CT body foreground and
PET high-uptake connected components as *proxies*; it never labels those
components as named organs.  True organ size/distance requires organ masks.

Default sampling is patient-balanced (up to two slices per patient).  The
authoritative split comes from ``main_data/split_manifest.csv``; the physical
``Data/data/train`` and ``Data/data/val`` folders are treated only as file
stores because their older split assignment differs from the manifest.

Outputs:
  sample_metrics.csv       one wide row per sampled slice
  component_metrics.csv    CT-body/PET-uptake component geometry
  band_summary.csv         patient-level summary for each Haar band
  association_summary.csv  predeclared patient-level correlations + FDR
  summary.json             compact machine-readable overview
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import ConstantInputWarning, pearsonr, spearmanr


BANDS = (
    "ll2",
    "l2_lh",
    "l2_hl",
    "l2_hh",
    "l1_lh",
    "l1_hl",
    "l1_hh",
)


def _index_pngs(root: Path, modality: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for old_split in ("train", "val", "test"):
        folder = root / old_split / modality
        if not folder.exists():
            continue
        for path in folder.glob("*.png"):
            if path.stem in result:
                raise RuntimeError(f"duplicate {modality} sample: {path.stem}")
            result[path.stem] = path
    return result


def _read_image(path: Path, size: int, *, invert: bool = False) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        value = np.asarray(image, dtype=np.float32) / 255.0
    return 1.0 - value if invert else value


def _read_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8) > 127


def _largest_body(ct: np.ndarray) -> np.ndarray:
    foreground = ndimage.binary_closing(ct > 0.025, iterations=2)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return np.ones_like(foreground, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = labels == int(sizes.argmax())
    body = ndimage.binary_fill_holes(body)
    return ndimage.binary_dilation(body, iterations=2).astype(bool)


def _haar2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    a = image[0::2, 0::2]
    b = image[0::2, 1::2]
    c = image[1::2, 0::2]
    d = image[1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh)


def _resize_float(image: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    pil = pil.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def _frequency_energy_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    size = int(image.shape[0])
    ll1, details1 = _haar2(image)
    ll2, details2 = _haar2(ll1)
    result = {
        "ll2": _resize_float(np.abs(ll2), size),
    }
    for level, details in (("l1", details1), ("l2", details2)):
        for band, value in zip(("lh", "hl", "hh"), details, strict=True):
            local_energy = ndimage.uniform_filter(np.abs(value), size=3, mode="nearest")
            result[f"{level}_{band}"] = _resize_float(local_energy, size)
    return result


def _centroid(mask_or_weight: np.ndarray) -> tuple[float, float]:
    weight = np.asarray(mask_or_weight, dtype=np.float64)
    total = float(weight.sum())
    if total <= 1e-12:
        return float("nan"), float("nan")
    yy, xx = np.indices(weight.shape, dtype=np.float64)
    return float((xx * weight).sum() / total), float((yy * weight).sum() / total)


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    if not np.all(np.isfinite((*a, *b))):
        return float("nan")
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _weighted_spread(mask_or_weight: np.ndarray) -> float:
    weight = np.asarray(mask_or_weight, dtype=np.float64)
    total = float(weight.sum())
    if total <= 1e-12:
        return float("nan")
    cx, cy = _centroid(weight)
    yy, xx = np.indices(weight.shape, dtype=np.float64)
    variance = ((xx - cx) ** 2 + (yy - cy) ** 2) * weight
    return float(np.sqrt(variance.sum() / total))


def _ring(mask: np.ndarray, body: np.ndarray, radius: int = 12) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=radius) & ~mask & body


def _safe_mean(value: np.ndarray, roi: np.ndarray) -> float:
    selected = value[roi]
    return float(selected.mean()) if selected.size else float("nan")


def _safe_corr(x: Iterable[float], y: Iterable[float], method: str) -> tuple[float, float, int]:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if x_arr.size < 4 or np.ptp(x_arr) <= 1e-12 or np.ptp(y_arr) <= 1e-12:
        return float("nan"), float("nan"), int(x_arr.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = pearsonr(x_arr, y_arr) if method == "pearson" else spearmanr(x_arr, y_arr)
    return float(result.statistic), float(result.pvalue), int(x_arr.size)


def _spatial_correlations(
    ct_energy: np.ndarray, pet_energy: np.ndarray, roi: np.ndarray
) -> tuple[float, float]:
    x = ct_energy[roi].astype(np.float64)
    y = pet_energy[roi].astype(np.float64)
    if x.size < 8:
        return float("nan"), float("nan")
    x_scale = np.median(x[x > 0]) if np.any(x > 0) else 1.0
    y_scale = np.median(y[y > 0]) if np.any(y > 0) else 1.0
    log_x = np.log1p(x / max(float(x_scale), 1e-8))
    log_y = np.log1p(y / max(float(y_scale), 1e-8))
    pearson, _, _ = _safe_corr(log_x, log_y, "pearson")
    spearman, _, _ = _safe_corr(x, y, "spearman")
    return pearson, spearman


def _radial_fft(image: np.ndarray, body: np.ndarray, bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    value = image.astype(np.float64).copy()
    if body.any():
        value -= float(value[body].mean())
    value *= body
    window = np.outer(np.hanning(value.shape[0]), np.hanning(value.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(value * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(value.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(value.shape[1]))
    yy, xx = np.meshgrid(fy, fx, indexing="ij")
    radius = np.sqrt(xx * xx + yy * yy)
    radius /= max(float(radius.max()), 1e-12)
    edges = np.linspace(0.0, 1.0, bins + 1)
    profile = np.zeros(bins, dtype=np.float64)
    for index in range(bins):
        selected = (radius >= edges[index]) & (radius < edges[index + 1])
        profile[index] = float(power[selected].mean()) if selected.any() else 0.0
    profile /= max(float(profile.sum()), 1e-12)
    return profile, radius


def _fft_band_proportions(profile: np.ndarray) -> tuple[float, float, float]:
    count = int(profile.size)
    low_end = max(1, round(0.20 * count))
    mid_end = max(low_end + 1, round(0.50 * count))
    return (
        float(profile[:low_end].sum()),
        float(profile[low_end:mid_end].sum()),
        float(profile[mid_end:].sum()),
    )


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else float("nan")


def _jensen_shannon_distance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x / max(float(x.sum()), 1e-12)
    y = y / max(float(y.sum()), 1e-12)
    midpoint = 0.5 * (x + y)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        valid = left > 0
        return float(np.sum(left[valid] * np.log(left[valid] / np.maximum(right[valid], 1e-12))))

    return float(np.sqrt(max(0.5 * kl(x, midpoint) + 0.5 * kl(y, midpoint), 0.0)))


def _uptake_components(
    pet: np.ndarray,
    body: np.ndarray,
    lesion_centroid: tuple[float, float],
    max_components: int = 5,
) -> tuple[list[dict[str, float]], float, np.ndarray, int]:
    body_values = pet[body]
    if body_values.size == 0:
        return [], float("nan"), np.zeros_like(body), 0
    q25, q75, q90 = np.quantile(body_values, (0.25, 0.75, 0.90))
    threshold = max(0.04, float(q75 + 1.5 * (q75 - q25)), float(0.60 * q90))
    support = (pet >= threshold) & body
    support = ndimage.binary_opening(support, iterations=1)
    labels, count = ndimage.label(support)
    components: list[dict[str, float]] = []
    for label in range(1, count + 1):
        region = labels == label
        area = int(region.sum())
        if area < 4:
            continue
        centroid = _centroid(region)
        components.append(
            {
                "area_px": float(area),
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "lesion_centroid_distance_px": _distance(centroid, lesion_centroid),
                "mean_uptake": _safe_mean(pet, region),
                "max_uptake": float(pet[region].max()),
            }
        )
    components.sort(key=lambda item: item["area_px"], reverse=True)
    return components[:max_components], threshold, support, len(components)


def _pairwise_centroid_stats(components: list[dict[str, float]]) -> tuple[float, float, float]:
    distances = []
    for left in range(len(components)):
        for right in range(left + 1, len(components)):
            a = (components[left]["centroid_x"], components[left]["centroid_y"])
            b = (components[right]["centroid_x"], components[right]["centroid_y"])
            distances.append(_distance(a, b))
    if not distances:
        return float("nan"), float("nan"), float("nan")
    return float(np.min(distances)), float(np.mean(distances)), float(np.max(distances))


def _balanced_sample(
    rows: list[dict[str, str]], slices_per_patient: int, seed: int
) -> list[dict[str, str]]:
    if slices_per_patient <= 0:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["patient_id"]].append(row)
    rng = np.random.default_rng(seed)
    sampled = []
    for patient_id in sorted(grouped):
        patient_rows = sorted(grouped[patient_id], key=lambda item: int(item["slice_id"]))
        count = min(slices_per_patient, len(patient_rows))
        indices = np.sort(rng.choice(len(patient_rows), count, replace=False))
        sampled.extend(patient_rows[int(index)] for index in indices)
    return sampled


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.flatnonzero(np.isfinite(pvalues))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    ranked = pvalues[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def _association_rows(patient_df: pd.DataFrame) -> list[dict[str, float | str | int]]:
    pairs: list[tuple[str, str, str]] = [
        ("ct_body_area_px", "pet_uptake_area_px", "body size vs PET uptake support"),
        ("lesion_area_px", "pet_uptake_area_px", "lesion size vs PET uptake support"),
        (
            "lesion_to_body_centroid_px",
            "lesion_to_nearest_pet_component_px",
            "lesion eccentricity vs nearest PET component",
        ),
        ("ct_fft_low", "pet_fft_low", "CT/PET low radial FFT proportion"),
        ("ct_fft_mid", "pet_fft_mid", "CT/PET mid radial FFT proportion"),
        ("ct_fft_high", "pet_fft_high", "CT/PET high radial FFT proportion"),
    ]
    for band in BANDS:
        pairs.extend(
            (
                (
                    f"ct_{band}_mean_energy",
                    f"pet_{band}_mean_energy",
                    f"{band}: CT/PET whole-body mean energy",
                ),
                (
                    f"ct_{band}_lesion_energy",
                    f"pet_{band}_lesion_energy",
                    f"{band}: CT/PET lesion energy",
                ),
                (
                    f"ct_{band}_lesion_ring_ratio",
                    f"pet_{band}_lesion_ring_ratio",
                    f"{band}: CT/PET lesion-to-ring energy ratio",
                ),
                (
                    "lesion_area_px",
                    f"ct_{band}_lesion_energy",
                    f"{band}: lesion size vs CT lesion energy",
                ),
                (
                    "lesion_area_px",
                    f"pet_{band}_lesion_energy",
                    f"{band}: lesion size vs PET lesion energy",
                ),
                (
                    "lesion_area_px",
                    f"{band}_ct_pet_centroid_distance_px",
                    f"{band}: lesion size vs CT/PET energy-centroid distance",
                ),
            )
        )
    result: list[dict[str, float | str | int]] = []
    for x_name, y_name, label in pairs:
        pearson, pearson_p, count = _safe_corr(patient_df[x_name], patient_df[y_name], "pearson")
        spearman, spearman_p, _ = _safe_corr(patient_df[x_name], patient_df[y_name], "spearman")
        result.append(
            {
                "association": label,
                "x": x_name,
                "y": y_name,
                "patients": count,
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
            }
        )
    adjusted = _bh_fdr(np.asarray([row["spearman_p"] for row in result], dtype=float))
    for row, qvalue in zip(result, adjusted, strict=True):
        row["spearman_fdr_bh"] = float(qvalue)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--slices-per-patient", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or root / "results" / "ct_pet_frequency_geometry"
    output.mkdir(parents=True, exist_ok=True)
    data_root = root / "Data" / "data"
    manifest_path = root / "main_data" / "split_manifest.csv"
    image_index = {name: _index_pngs(data_root, name) for name in ("ct", "pet", "label")}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    missing = {
        modality: [row["sample_id"] for row in manifest if row["sample_id"] not in paths]
        for modality, paths in image_index.items()
    }
    if any(missing.values()):
        raise FileNotFoundError(f"missing files for manifest rows: {missing}")
    sampled = _balanced_sample(manifest, args.slices_per_patient, args.seed)

    sample_rows: list[dict[str, float | str | int]] = []
    component_rows: list[dict[str, float | str | int]] = []
    diagonal = math.sqrt(2.0) * args.size
    for index, row in enumerate(sampled, start=1):
        sid = row["sample_id"]
        ct = _read_image(image_index["ct"][sid], args.size)
        pet = _read_image(image_index["pet"][sid], args.size, invert=True)
        lesion = _read_mask(image_index["label"][sid], args.size)
        body = _largest_body(ct) | lesion
        ring = _ring(lesion, body)
        body_centroid = _centroid(body)
        lesion_centroid = _centroid(lesion)
        lesion_area = int(lesion.sum())
        ct_maps = _frequency_energy_maps(ct)
        pet_maps = _frequency_energy_maps(pet)

        ct_profile, _ = _radial_fft(ct, body)
        pet_profile, _ = _radial_fft(pet, body)
        ct_fft = _fft_band_proportions(ct_profile)
        pet_fft = _fft_band_proportions(pet_profile)
        components, uptake_threshold, uptake_support, component_count_total = _uptake_components(
            pet, body, lesion_centroid
        )
        pair_min, pair_mean, pair_max = _pairwise_centroid_stats(components)
        nearest_component = min(
            (item["lesion_centroid_distance_px"] for item in components),
            default=float("nan"),
        )

        metrics: dict[str, float | str | int] = {
            "sample_id": sid,
            "patient_id": row["patient_id"],
            "slice_id": int(row["slice_id"]),
            "split": row["split"],
            "ct_body_area_px": int(body.sum()),
            "ct_body_fraction": float(body.mean()),
            "ct_body_centroid_x": body_centroid[0],
            "ct_body_centroid_y": body_centroid[1],
            "lesion_area_px": lesion_area,
            "lesion_fraction_full_image": float(lesion.mean()),
            "lesion_fraction_body": float(lesion.sum() / max(body.sum(), 1)),
            "lesion_equivalent_diameter_px": float(2.0 * math.sqrt(lesion_area / math.pi)),
            "lesion_spread_px": _weighted_spread(lesion),
            "lesion_centroid_x": lesion_centroid[0],
            "lesion_centroid_y": lesion_centroid[1],
            "lesion_to_body_centroid_px": _distance(lesion_centroid, body_centroid),
            "lesion_to_body_centroid_normalized": _distance(
                lesion_centroid, body_centroid
            ) / diagonal,
            "pet_uptake_threshold": uptake_threshold,
            "pet_uptake_area_px": int(uptake_support.sum()),
            "pet_uptake_fraction_body": float(uptake_support.sum() / max(body.sum(), 1)),
            "pet_uptake_component_count_total": component_count_total,
            "pet_uptake_component_count_top5": len(components),
            "lesion_to_nearest_pet_component_px": nearest_component,
            "pet_component_pairwise_centroid_min_px": pair_min,
            "pet_component_pairwise_centroid_mean_px": pair_mean,
            "pet_component_pairwise_centroid_max_px": pair_max,
            "ct_pet_fft_profile_cosine": _cosine(ct_profile, pet_profile),
            "ct_pet_fft_profile_cosine_without_low": _cosine(
                ct_profile[max(1, round(0.20 * ct_profile.size)) :],
                pet_profile[max(1, round(0.20 * pet_profile.size)) :],
            ),
            "ct_pet_fft_profile_js_distance": _jensen_shannon_distance(
                ct_profile, pet_profile
            ),
            "ct_fft_low": ct_fft[0],
            "ct_fft_mid": ct_fft[1],
            "ct_fft_high": ct_fft[2],
            "pet_fft_low": pet_fft[0],
            "pet_fft_mid": pet_fft[1],
            "pet_fft_high": pet_fft[2],
        }
        for rank, component in enumerate(components, start=1):
            component_rows.append(
                {
                    "sample_id": sid,
                    "patient_id": row["patient_id"],
                    "split": row["split"],
                    "component_type": "pet_high_uptake_proxy",
                    "rank_by_area": rank,
                    "body_centroid_distance_px": _distance(
                        (component["centroid_x"], component["centroid_y"]),
                        body_centroid,
                    ),
                    **component,
                }
            )
        component_rows.append(
            {
                "sample_id": sid,
                "patient_id": row["patient_id"],
                "split": row["split"],
                "component_type": "ct_body_foreground_proxy",
                "rank_by_area": 1,
                "area_px": float(body.sum()),
                "centroid_x": body_centroid[0],
                "centroid_y": body_centroid[1],
                    "lesion_centroid_distance_px": _distance(body_centroid, lesion_centroid),
                    "body_centroid_distance_px": 0.0,
                "mean_uptake": float("nan"),
                "max_uptake": float("nan"),
            }
        )

        for band in BANDS:
            ct_energy = ct_maps[band]
            pet_energy = pet_maps[band]
            ct_centroid = _centroid(ct_energy * body)
            pet_centroid = _centroid(pet_energy * body)
            spatial_pearson, spatial_spearman = _spatial_correlations(
                ct_energy, pet_energy, body
            )
            ct_lesion = _safe_mean(ct_energy, lesion)
            pet_lesion = _safe_mean(pet_energy, lesion)
            ct_ring = _safe_mean(ct_energy, ring)
            pet_ring = _safe_mean(pet_energy, ring)
            metrics.update(
                {
                    f"ct_{band}_mean_energy": _safe_mean(ct_energy, body),
                    f"pet_{band}_mean_energy": _safe_mean(pet_energy, body),
                    f"ct_{band}_lesion_energy": ct_lesion,
                    f"pet_{band}_lesion_energy": pet_lesion,
                    f"ct_{band}_ring_energy": ct_ring,
                    f"pet_{band}_ring_energy": pet_ring,
                    f"ct_{band}_lesion_ring_ratio": float(
                        ct_lesion / max(ct_ring, 1e-8)
                    ),
                    f"pet_{band}_lesion_ring_ratio": float(
                        pet_lesion / max(pet_ring, 1e-8)
                    ),
                    f"{band}_spatial_log_pearson": spatial_pearson,
                    f"{band}_spatial_spearman": spatial_spearman,
                    f"{band}_ct_centroid_x": ct_centroid[0],
                    f"{band}_ct_centroid_y": ct_centroid[1],
                    f"{band}_pet_centroid_x": pet_centroid[0],
                    f"{band}_pet_centroid_y": pet_centroid[1],
                    f"{band}_ct_pet_centroid_distance_px": _distance(
                        ct_centroid, pet_centroid
                    ),
                    f"{band}_ct_lesion_centroid_distance_px": _distance(
                        ct_centroid, lesion_centroid
                    ),
                    f"{band}_pet_lesion_centroid_distance_px": _distance(
                        pet_centroid, lesion_centroid
                    ),
                }
            )
        sample_rows.append(metrics)
        if index % 50 == 0 or index == len(sampled):
            print(f"[frequency-geometry] {index}/{len(sampled)}", flush=True)

    sample_df = pd.DataFrame(sample_rows)
    component_df = pd.DataFrame(component_rows)
    numeric_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    patient_df = sample_df.groupby("patient_id", as_index=False)[numeric_columns].mean()
    association_rows = _association_rows(patient_df)
    association_df = pd.DataFrame(association_rows)

    band_rows: list[dict[str, float | str | int]] = []
    for band in BANDS:
        energy_r, energy_p, count = _safe_corr(
            patient_df[f"ct_{band}_mean_energy"],
            patient_df[f"pet_{band}_mean_energy"],
            "spearman",
        )
        lesion_r, lesion_p, _ = _safe_corr(
            patient_df[f"ct_{band}_lesion_energy"],
            patient_df[f"pet_{band}_lesion_energy"],
            "spearman",
        )
        contrast_r, contrast_p, _ = _safe_corr(
            patient_df[f"ct_{band}_lesion_ring_ratio"],
            patient_df[f"pet_{band}_lesion_ring_ratio"],
            "spearman",
        )
        size_ct_r, size_ct_p, _ = _safe_corr(
            patient_df["lesion_area_px"], patient_df[f"ct_{band}_lesion_energy"], "spearman"
        )
        size_pet_r, size_pet_p, _ = _safe_corr(
            patient_df["lesion_area_px"], patient_df[f"pet_{band}_lesion_energy"], "spearman"
        )
        band_rows.append(
            {
                "band": band,
                "patients": count,
                "ct_pet_mean_energy_spearman": energy_r,
                "ct_pet_mean_energy_p": energy_p,
                "ct_pet_lesion_energy_spearman": lesion_r,
                "ct_pet_lesion_energy_p": lesion_p,
                "ct_pet_lesion_ring_ratio_spearman": contrast_r,
                "ct_pet_lesion_ring_ratio_p": contrast_p,
                "lesion_size_ct_energy_spearman": size_ct_r,
                "lesion_size_ct_energy_p": size_ct_p,
                "lesion_size_pet_energy_spearman": size_pet_r,
                "lesion_size_pet_energy_p": size_pet_p,
                "median_slice_spatial_spearman": float(
                    sample_df[f"{band}_spatial_spearman"].median()
                ),
                "median_ct_pet_centroid_distance_px": float(
                    sample_df[f"{band}_ct_pet_centroid_distance_px"].median()
                ),
                "median_ct_lesion_centroid_distance_px": float(
                    sample_df[f"{band}_ct_lesion_centroid_distance_px"].median()
                ),
                "median_pet_lesion_centroid_distance_px": float(
                    sample_df[f"{band}_pet_lesion_centroid_distance_px"].median()
                ),
            }
        )
    band_df = pd.DataFrame(band_rows)

    sample_df.to_csv(output / "sample_metrics.csv", index=False, encoding="utf-8-sig")
    component_df.to_csv(output / "component_metrics.csv", index=False, encoding="utf-8-sig")
    patient_df.to_csv(output / "patient_metrics.csv", index=False, encoding="utf-8-sig")
    band_df.to_csv(output / "band_summary.csv", index=False, encoding="utf-8-sig")
    association_df.to_csv(
        output / "association_summary.csv", index=False, encoding="utf-8-sig"
    )

    significant = association_df[
        association_df["spearman_fdr_bh"].notna()
        & (association_df["spearman_fdr_bh"] < 0.05)
    ].copy()
    significant["abs_rho"] = significant["spearman_rho"].abs()
    significant = significant.sort_values("abs_rho", ascending=False)
    summary = {
        "manifest": str(manifest_path),
        "sampling": {
            "seed": args.seed,
            "slices_per_patient": args.slices_per_patient,
            "sampled_slices": len(sample_df),
            "sampled_patients": int(sample_df["patient_id"].nunique()),
            "train_slices": int((sample_df["split"] == "train").sum()),
            "validation_slices": int((sample_df["split"] == "val").sum()),
        },
        "geometry": {
            "median_lesion_area_px": float(sample_df["lesion_area_px"].median()),
            "median_lesion_fraction_full_image": float(
                sample_df["lesion_fraction_full_image"].median()
            ),
            "median_lesion_to_body_centroid_px": float(
                sample_df["lesion_to_body_centroid_px"].median()
            ),
            "median_pet_uptake_component_count_top5": float(
                sample_df["pet_uptake_component_count_top5"].median()
            ),
            "median_pet_uptake_component_count_total": float(
                sample_df["pet_uptake_component_count_total"].median()
            ),
            "median_lesion_to_nearest_pet_component_px": float(
                sample_df["lesion_to_nearest_pet_component_px"].median()
            ),
        },
        "fft": {
            "median_ct_pet_profile_cosine": float(
                sample_df["ct_pet_fft_profile_cosine"].median()
            ),
            "median_ct_pet_profile_cosine_without_low": float(
                sample_df["ct_pet_fft_profile_cosine_without_low"].median()
            ),
            "median_ct_pet_profile_js_distance": float(
                sample_df["ct_pet_fft_profile_js_distance"].median()
            ),
            "patient_level_low_band_spearman": _safe_corr(
                patient_df["ct_fft_low"], patient_df["pet_fft_low"], "spearman"
            )[0],
            "patient_level_mid_band_spearman": _safe_corr(
                patient_df["ct_fft_mid"], patient_df["pet_fft_mid"], "spearman"
            )[0],
            "patient_level_high_band_spearman": _safe_corr(
                patient_df["ct_fft_high"], patient_df["pet_fft_high"], "spearman"
            )[0],
        },
        "strongest_fdr_significant_associations": significant[
            ["association", "spearman_rho", "spearman_fdr_bh"]
        ].head(15).to_dict(orient="records"),
        "limitations": [
            "No organ masks are present; body foreground and PET uptake components are proxies, not named organs.",
            "PNG intensities are normalized display values, not physical HU/SUV.",
            "Associations are descriptive and do not imply that CT determines PET uptake.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
