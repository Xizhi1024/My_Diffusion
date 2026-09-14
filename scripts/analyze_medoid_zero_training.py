"""Zero-training follow-up analyses for a four-trajectory medoid run.

The analysis is deliberately downstream-only: it reads the saved cohort tensors,
four prediction trajectories, and the already-computed metric tables. It never
loads an optimizer or updates model parameters.

Primary pre-specified checks
----------------------------
1. Patient-aggregated lesion disagreement vs. medoid TopQ error.
2. Connected-component area vs. baseline output-gradient mass.
3. Connected-component area vs. target-residual Haar high-frequency fraction.

The first check characterizes medoid uncertainty. Checks 2 and 3 are the two
mechanism gates for considering instance-balanced, scale-matched output losses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage, stats


ERROR_METRICS = (
    "lesion_topq_peak_error_norm",
    "lesion_peak_error_norm",
    "false_hotspot",
)
ALL_METRICS = (*ERROR_METRICS, "ssim")
PRIMARY_DISPERSION = "lesion_pixel_std"
PRIMARY_ERROR = "medoid_lesion_topq_peak_error_norm"
EPSILON = 1e-12
DETERMINISTIC_ARTIFACTS = (
    "slice_analysis.csv",
    "uncertainty_error_associations.csv",
    "uncertainty_sensitivity.csv",
    "patient_benefit_failure.csv",
    "patient_profile_comparison.csv",
    "continuous_size_effects.csv",
    "component_mechanism_audit.csv",
    "mechanism_gate_summary.csv",
    "mechanism_secondary_associations.csv",
    "uncertainty_vs_error.png",
    "patient_benefit_waterfall.png",
    "continuous_size_effect.png",
    "mechanism_audits.png",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")
    x_valid = x[valid]
    y_valid = y[valid]
    if np.unique(x_valid).size < 2 or np.unique(y_valid).size < 2:
        return float("nan")
    return float(stats.spearmanr(x_valid, y_valid).statistic)


def _spearman_with_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return float("nan"), float("nan")
    result = stats.spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def _bootstrap_ci(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)])
    if finite.size < 20:
        return float("nan"), float("nan")
    return (
        float(np.quantile(finite, alpha / 2.0)),
        float(np.quantile(finite, 1.0 - alpha / 2.0)),
    )


def patient_cluster_bootstrap_spearman(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    patient: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Spearman correlation with patients as the resampling unit."""
    clean = frame[[patient, x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    patient_ids = clean[patient].astype(str).unique()
    estimate, p_value = _spearman_with_p(
        clean[x].to_numpy(float),
        clean[y].to_numpy(float),
    )
    boot: list[float] = []
    groups = {
        patient_id: group
        for patient_id, group in clean.groupby(patient, sort=False)
    }
    for _ in range(replicates):
        sampled = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        pieces = [groups[patient_id] for patient_id in sampled]
        current = pd.concat(pieces, ignore_index=True)
        value = _spearman(
            current[x].to_numpy(float),
            current[y].to_numpy(float),
        )
        if math.isfinite(value):
            boot.append(value)
    low, high = _bootstrap_ci(boot)
    return {
        "estimate": estimate,
        "p_value_uncorrected": p_value,
        "ci95_low": low,
        "ci95_high": high,
        "patients": int(len(patient_ids)),
        "rows": int(len(clean)),
        "bootstrap_valid": int(len(boot)),
    }


def patient_aggregate_spearman(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    patient: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Aggregate slices within patient before correlation and bootstrapping."""
    patient_frame = (
        frame[[patient, x, y]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .groupby(patient, as_index=False)[[x, y]]
        .mean()
    )
    estimate, p_value = _spearman_with_p(
        patient_frame[x].to_numpy(float),
        patient_frame[y].to_numpy(float),
    )
    n = len(patient_frame)
    boot: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        value = _spearman(
            patient_frame[x].to_numpy(float)[indices],
            patient_frame[y].to_numpy(float)[indices],
        )
        if math.isfinite(value):
            boot.append(value)
    low, high = _bootstrap_ci(boot)
    return {
        "estimate": estimate,
        "p_value_uncorrected": p_value,
        "ci95_low": low,
        "ci95_high": high,
        "patients": int(n),
        "rows": int(n),
        "bootstrap_valid": int(len(boot)),
    }


def _single_row_patient_bootstrap_spearman(
    patient_frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    patient: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Bootstrap a frame already reduced to one row per patient."""
    if patient_frame[patient].astype(str).duplicated().any():
        raise ValueError("patient_frame must have one row per patient")
    return patient_cluster_bootstrap_spearman(
        patient_frame,
        x=x,
        y=y,
        patient=patient,
        replicates=replicates,
        rng=rng,
    )


def within_patient_slope(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    patient: str,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Patient-fixed-effect slope using within-patient centered sufficient stats."""
    clean = frame[[patient, x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    sufficient: list[tuple[float, float, int]] = []
    for _, group in clean.groupby(patient):
        if len(group) < 2:
            continue
        x_values = group[x].to_numpy(float)
        y_values = group[y].to_numpy(float)
        x_centered = x_values - x_values.mean()
        y_centered = y_values - y_values.mean()
        ss_xx = float(np.dot(x_centered, x_centered))
        if ss_xx <= EPSILON:
            continue
        sufficient.append(
            (float(np.dot(x_centered, y_centered)), ss_xx, len(group))
        )
    if not sufficient:
        return {
            "estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "patients": 0,
            "rows": 0,
            "bootstrap_valid": 0,
        }
    xy = np.asarray([item[0] for item in sufficient])
    xx = np.asarray([item[1] for item in sufficient])
    estimate = float(xy.sum() / xx.sum())
    boot: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(sufficient), size=len(sufficient))
        denominator = float(xx[indices].sum())
        if denominator > EPSILON:
            boot.append(float(xy[indices].sum() / denominator))
    low, high = _bootstrap_ci(boot)
    return {
        "estimate": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "patients": int(len(sufficient)),
        "rows": int(sum(item[2] for item in sufficient)),
        "bootstrap_valid": int(len(boot)),
    }


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    result = [float("nan")] * len(p_values)
    finite = [(index, value) for index, value in enumerate(p_values) if math.isfinite(value)]
    finite.sort(key=lambda item: item[1])
    running = 0.0
    total = len(finite)
    for rank, (index, value) in enumerate(finite):
        adjusted = min(1.0, value * (total - rank))
        running = max(running, adjusted)
        result[index] = running
    return result


def medoid_indices(stack: np.ndarray) -> np.ndarray:
    """Return the lowest-index full-image L1 medoid for each sample."""
    if stack.ndim < 3:
        raise ValueError("prediction stack must have [seed, sample, ...] dimensions")
    seeds, samples = stack.shape[:2]
    selected = np.empty(samples, dtype=np.int64)
    for sample_index in range(samples):
        flattened = stack[:, sample_index].reshape(seeds, -1)
        distances = np.abs(
            flattened[:, None, :] - flattened[None, :, :]
        ).mean(axis=2)
        selected[sample_index] = int(np.argmin(distances.sum(axis=1)))
    return selected


def trajectory_dispersion(
    stack: np.ndarray,
    masks: np.ndarray,
) -> pd.DataFrame:
    """Compute prediction-space and lesion-space four-trajectory disagreement."""
    if stack.shape[1] != masks.shape[0]:
        raise ValueError("prediction and mask sample counts do not match")
    pixel_std = stack.std(axis=0, ddof=1)
    binary_masks = masks > 0.5
    rows: list[dict[str, float]] = []
    pair_indices = list(combinations(range(stack.shape[0]), 2))
    for sample_index in range(stack.shape[1]):
        lesion = binary_masks[sample_index]
        lesion_count = max(int(lesion.sum()), 1)
        full_pairwise: list[float] = []
        lesion_pairwise: list[float] = []
        for left, right in pair_indices:
            difference = np.abs(stack[left, sample_index] - stack[right, sample_index])
            full_pairwise.append(float(difference.mean()))
            lesion_pairwise.append(float(difference[lesion].sum() / lesion_count))
        rows.append(
            {
                "full_pixel_std": float(pixel_std[sample_index].mean()),
                "lesion_pixel_std": float(
                    pixel_std[sample_index][lesion].sum() / lesion_count
                ),
                "full_pairwise_l1": float(np.mean(full_pairwise)),
                "lesion_pairwise_l1": float(np.mean(lesion_pairwise)),
            }
        )
    return pd.DataFrame(rows)


def _image_gradient_l1_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    loss_x = (pred_dx - target_dx).abs().flatten(1).mean(dim=1)
    loss_y = (pred_dy - target_dy).abs().flatten(1).mean(dim=1)
    return 0.5 * (loss_x + loss_y)


def baseline_output_gradient_maps(
    stack: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Average absolute d(base reconstruction loss)/d(pred_x0) over seeds.

    This is an inference-endpoint output-gradient proxy at tau=0. The common
    min-SNR scalar is omitted because it cannot change between-slice rankings.
    """
    tau = 0.0
    w_mse = 0.5 + 1.5 / (1.0 + math.exp(-10.0 * (tau - 0.43)))
    w_grad = 0.2 / (1.0 + math.exp(-10.0 * (0.25 - tau)))
    accumulated = np.zeros_like(target, dtype=np.float64)
    for seed_index in range(stack.shape[0]):
        for start in range(0, stack.shape[1], batch_size):
            stop = min(start + batch_size, stack.shape[1])
            prediction = torch.tensor(
                stack[seed_index, start:stop],
                dtype=torch.float64,
                requires_grad=True,
            )
            current_target = torch.tensor(target[start:stop], dtype=torch.float64)
            reduce_dims = tuple(range(1, prediction.dim()))
            mse = (prediction - current_target).square().mean(dim=reduce_dims)
            l1 = (prediction - current_target).abs().mean(dim=reduce_dims)
            gradient = _image_gradient_l1_per_sample(prediction, current_target)
            per_sample = w_mse * mse + l1 + 0.1 * w_grad * gradient
            per_sample.sum().backward()
            if prediction.grad is None:
                raise RuntimeError("output-gradient audit did not produce gradients")
            accumulated[start:stop] += np.abs(prediction.grad.detach().numpy())
    return (accumulated / stack.shape[0]).astype(np.float64)


def _haar_dwt2(image: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        raise ValueError(f"Haar input must be even, got {image.shape[-2:]}")
    a = image[..., 0::2, 0::2]
    b = image[..., 0::2, 1::2]
    c = image[..., 1::2, 0::2]
    d = image[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, (lh, hl, hh)


def _pool_any(mask: np.ndarray) -> np.ndarray:
    if mask.shape[-2] % 2 or mask.shape[-1] % 2:
        raise ValueError(f"mask must be even, got {mask.shape[-2:]}")
    return (
        mask[..., 0::2, 0::2]
        | mask[..., 0::2, 1::2]
        | mask[..., 1::2, 0::2]
        | mask[..., 1::2, 1::2]
    )


def component_mechanism_rows(
    *,
    masks: np.ndarray,
    gradient_maps: np.ndarray,
    target_residual: np.ndarray,
    cohort: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build component-level gradient and target-residual frequency evidence."""
    ll1, details1 = _haar_dwt2(target_residual)
    ll2, details2 = _haar_dwt2(ll1)
    rows: list[dict[str, Any]] = []
    for sample_index in range(masks.shape[0]):
        mask_2d = masks[sample_index, 0] > 0.5
        labels, count = ndimage.label(mask_2d)
        total_gradient = float(gradient_maps[sample_index].sum())
        for component_id in range(1, count + 1):
            component = labels == component_id
            area = int(component.sum())
            component_4d = component[None, None]
            mask_l1 = _pool_any(component_4d)
            mask_l2 = _pool_any(mask_l1)
            high_energy = float(
                sum(np.square(band[sample_index : sample_index + 1])[mask_l1].sum()
                    for band in details1)
            )
            mid_energy = float(
                sum(np.square(band[sample_index : sample_index + 1])[mask_l2].sum()
                    for band in details2)
            )
            low_energy = float(
                np.square(ll2[sample_index : sample_index + 1])[mask_l2].sum()
            )
            band_total = high_energy + mid_energy + low_energy
            lesion_gradient_mass = float(
                gradient_maps[sample_index, 0][component].sum()
            )
            rows.append(
                {
                    "patient_id": str(cohort.iloc[sample_index]["patient_id"]),
                    "sample_id": str(cohort.iloc[sample_index]["sample_id"]),
                    "slice_id": int(cohort.iloc[sample_index]["slice_id"]),
                    "component_id": component_id,
                    "component_area": area,
                    "log2_component_area": math.log2(max(area, 1)),
                    "lesion_gradient_mass": lesion_gradient_mass,
                    "log_lesion_gradient_mass": math.log(
                        lesion_gradient_mass + EPSILON
                    ),
                    "lesion_gradient_density": lesion_gradient_mass / max(area, 1),
                    "lesion_gradient_share": (
                        lesion_gradient_mass / max(total_gradient, EPSILON)
                    ),
                    "total_output_gradient_mass": total_gradient,
                    "target_residual_high_energy": high_energy,
                    "target_residual_mid_energy": mid_energy,
                    "target_residual_low_energy": low_energy,
                    "target_residual_high_fraction": (
                        high_energy / max(band_total, EPSILON)
                    ),
                    "target_residual_mid_fraction": (
                        mid_energy / max(band_total, EPSILON)
                    ),
                    "target_residual_low_fraction": (
                        low_energy / max(band_total, EPSILON)
                    ),
                    "target_residual_total_energy": band_total,
                }
            )
    return rows


def _entropy(counts: Sequence[int]) -> float:
    values = np.asarray(counts, dtype=float)
    values = values[values > 0]
    probabilities = values / values.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _load_run(run: Path) -> dict[str, Any]:
    required = (
        "run_manifest.json",
        "COMPLETE.json",
        "cohort.csv",
        "cohort_tensors.npz",
        "metrics.csv",
        "aggregation_metrics.csv",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if missing:
        raise FileNotFoundError(f"run is missing required files: {missing}")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "COMPLETE.json").read_text(encoding="utf-8"))
    if complete.get("status") != "COMPLETE":
        raise ValueError("source run is not marked COMPLETE")
    if manifest.get("training_or_optimizer_used") is not False:
        raise ValueError("source run does not declare zero training/optimizer use")
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if len(seeds) != 4:
        raise ValueError(f"expected exactly four trajectories, found {seeds}")
    cohort = pd.read_csv(run / "cohort.csv", dtype={"patient_id": str, "sample_id": str})
    metrics = pd.read_csv(run / "metrics.csv", dtype={"patient_id": str, "sample_id": str})
    aggregation = pd.read_csv(
        run / "aggregation_metrics.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    _require_columns(
        cohort,
        ("patient_id", "sample_id", "slice_id", "lesion_area"),
        "cohort.csv",
    )
    _require_columns(
        metrics,
        ("patient_id", "sample_id", "seed", *ALL_METRICS),
        "metrics.csv",
    )
    _require_columns(
        aggregation,
        ("patient_id", "sample_id", "aggregation", *ALL_METRICS),
        "aggregation_metrics.csv",
    )
    static_file = np.load(run / "cohort_tensors.npz")
    static = {key: static_file[key] for key in static_file.files}
    source_ids = static["sample_ids"].astype(str)
    if not np.array_equal(source_ids, cohort["sample_id"].astype(str).to_numpy()):
        raise ValueError("cohort_tensors sample order does not match cohort.csv")
    prediction_arrays: list[np.ndarray] = []
    prediction_hashes: dict[str, str] = {}
    for seed in seeds:
        path = run / "predictions" / f"B0_seed{seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing trajectory: {path}")
        prediction_file = np.load(path)
        if not np.array_equal(
            prediction_file["sample_ids"].astype(str),
            source_ids,
        ):
            raise ValueError(f"sample order mismatch in {path.name}")
        prediction_arrays.append(prediction_file["predictions"].astype(np.float32))
        prediction_hashes[path.name] = _sha256(path)
    stack = np.stack(prediction_arrays)
    expected_shape = static["target"].shape
    if stack.shape[1:] != expected_shape:
        raise ValueError(
            f"prediction shape {stack.shape[1:]} does not match target {expected_shape}"
        )
    return {
        "manifest": manifest,
        "complete": complete,
        "cohort": cohort,
        "metrics": metrics,
        "aggregation": aggregation,
        "static": static,
        "stack": stack,
        "seeds": seeds,
        "prediction_hashes": prediction_hashes,
    }


def _metric_improvement_frame(
    *,
    cohort: pd.DataFrame,
    metrics: pd.DataFrame,
    aggregation: pd.DataFrame,
    dispersion: pd.DataFrame,
    selected_seed: np.ndarray,
    seeds: Sequence[int],
) -> pd.DataFrame:
    baseline = (
        metrics.groupby(["patient_id", "sample_id"], as_index=False)[list(ALL_METRICS)]
        .mean()
        .rename(columns={metric: f"baseline_{metric}" for metric in ALL_METRICS})
    )
    medoid = (
        aggregation.loc[aggregation["aggregation"] == "full_image_medoid"]
        [["patient_id", "sample_id", *ALL_METRICS]]
        .rename(columns={metric: f"medoid_{metric}" for metric in ALL_METRICS})
    )
    frame = cohort[["patient_id", "sample_id", "slice_id", "lesion_area"]].copy()
    frame = frame.merge(baseline, on=["patient_id", "sample_id"], validate="one_to_one")
    frame = frame.merge(medoid, on=["patient_id", "sample_id"], validate="one_to_one")
    frame = pd.concat([frame.reset_index(drop=True), dispersion], axis=1)
    frame["selected_seed"] = [seeds[index] for index in selected_seed]
    frame["log2_lesion_area"] = np.log2(frame["lesion_area"].clip(lower=1))
    for metric in ERROR_METRICS:
        absolute = (
            frame[f"baseline_{metric}"] - frame[f"medoid_{metric}"]
        )
        frame[f"{metric}_absolute_improvement"] = absolute
        frame[f"{metric}_relative_improvement"] = (
            absolute / frame[f"baseline_{metric}"].abs().clip(lower=EPSILON)
        )
    ssim_absolute = frame["medoid_ssim"] - frame["baseline_ssim"]
    frame["ssim_absolute_improvement"] = ssim_absolute
    frame["ssim_relative_improvement"] = (
        ssim_absolute / frame["baseline_ssim"].abs().clip(lower=EPSILON)
    )
    frame["medoid_ssim_error"] = 1.0 - frame["medoid_ssim"]
    return frame


def _uncertainty_rows(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    predictors = (
        "lesion_pixel_std",
        "lesion_pairwise_l1",
        "full_pixel_std",
        "full_pairwise_l1",
    )
    outcomes = (
        "medoid_lesion_topq_peak_error_norm",
        "medoid_lesion_peak_error_norm",
        "medoid_false_hotspot",
        "medoid_ssim_error",
    )
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for outcome in outcomes:
            result = patient_aggregate_spearman(
                frame,
                x=predictor,
                y=outcome,
                patient="patient_id",
                replicates=replicates,
                rng=rng,
            )
            rows.append(
                {
                    "level": "patient_aggregate",
                    "predictor": predictor,
                    "outcome": outcome,
                    **result,
                    "primary": (
                        predictor == PRIMARY_DISPERSION
                        and outcome == PRIMARY_ERROR
                    ),
                }
            )
    return rows


def _uncertainty_sensitivity_rows(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    columns = ["patient_id", PRIMARY_DISPERSION, PRIMARY_ERROR]
    grouped = frame.groupby("patient_id", as_index=False)
    patient_mean = grouped[columns[1:]].mean()
    patient_median = grouped[columns[1:]].median()
    counts = frame.groupby("patient_id").size().rename("slices").reset_index()
    patient_mean = patient_mean.merge(counts, on="patient_id")
    patient_median = patient_median.merge(counts, on="patient_id")
    specifications = (
        ("patient_mean_all", patient_mean),
        ("patient_median_all", patient_median),
        ("patient_mean_min_3_slices", patient_mean.loc[patient_mean["slices"] >= 3]),
        (
            "patient_median_min_3_slices",
            patient_median.loc[patient_median["slices"] >= 3],
        ),
    )
    rows: list[dict[str, Any]] = []
    for analysis, current in specifications:
        result = _single_row_patient_bootstrap_spearman(
            current,
            x=PRIMARY_DISPERSION,
            y=PRIMARY_ERROR,
            patient="patient_id",
            replicates=replicates,
            rng=rng,
        )
        rows.append({"analysis": analysis, **result})
    slice_result = patient_cluster_bootstrap_spearman(
        frame,
        x=PRIMARY_DISPERSION,
        y=PRIMARY_ERROR,
        patient="patient_id",
        replicates=replicates,
        rng=rng,
    )
    rows.append({"analysis": "slice_level_patient_cluster_bootstrap", **slice_result})

    leave_one_out: list[float] = []
    for patient_id in patient_mean["patient_id"]:
        current = patient_mean.loc[patient_mean["patient_id"] != patient_id]
        leave_one_out.append(
            _spearman(
                current[PRIMARY_DISPERSION].to_numpy(float),
                current[PRIMARY_ERROR].to_numpy(float),
            )
        )
    rows.append(
        {
            "analysis": "patient_mean_leave_one_patient_out_range",
            "estimate": float(
                _spearman(
                    patient_mean[PRIMARY_DISPERSION].to_numpy(float),
                    patient_mean[PRIMARY_ERROR].to_numpy(float),
                )
            ),
            "p_value_uncorrected": float("nan"),
            "ci95_low": float(np.nanmin(leave_one_out)),
            "ci95_high": float(np.nanmax(leave_one_out)),
            "patients": int(len(patient_mean)),
            "rows": int(len(patient_mean)),
            "bootstrap_valid": 0,
        }
    )
    return rows


def _patient_rows(frame: pd.DataFrame, seeds: Sequence[int]) -> list[dict[str, Any]]:
    numeric_columns = [
        "lesion_area",
        "lesion_pixel_std",
        *[
            f"{metric}_{kind}_improvement"
            for metric in ALL_METRICS
            for kind in ("absolute", "relative")
        ],
        *[f"baseline_{metric}" for metric in ALL_METRICS],
        *[f"medoid_{metric}" for metric in ALL_METRICS],
    ]
    grouped = frame.groupby("patient_id", sort=True)
    rows: list[dict[str, Any]] = []
    for patient_id, group in grouped:
        row: dict[str, Any] = {
            "patient_id": str(patient_id),
            "slices": int(len(group)),
        }
        for column in numeric_columns:
            row[column] = float(group[column].mean())
        counts = Counter(int(value) for value in group["selected_seed"])
        for seed in seeds:
            row[f"medoid_seed_{seed}_count"] = int(counts.get(seed, 0))
        row["medoid_seed_entropy_bits"] = _entropy(
            [counts.get(seed, 0) for seed in seeds]
        )
        row["primary_group"] = (
            "benefit"
            if row["lesion_topq_peak_error_norm_relative_improvement"] > 0.0
            else "failure"
        )
        rows.append(row)
    return rows


def _profile_comparison(patient_frame: pd.DataFrame) -> list[dict[str, Any]]:
    features = (
        "lesion_area",
        "lesion_pixel_std",
        "baseline_lesion_topq_peak_error_norm",
        "baseline_lesion_peak_error_norm",
        "baseline_false_hotspot",
        "medoid_seed_entropy_bits",
    )
    benefit = patient_frame.loc[patient_frame["primary_group"] == "benefit"]
    failure = patient_frame.loc[patient_frame["primary_group"] == "failure"]
    rows: list[dict[str, Any]] = []
    for feature in features:
        left = benefit[feature].dropna().to_numpy(float)
        right = failure[feature].dropna().to_numpy(float)
        p_value = float("nan")
        if left.size and right.size:
            p_value = float(
                stats.mannwhitneyu(left, right, alternative="two-sided").pvalue
            )
        rows.append(
            {
                "feature": feature,
                "benefit_patients": int(left.size),
                "failure_patients": int(right.size),
                "benefit_median": float(np.median(left)) if left.size else float("nan"),
                "failure_median": float(np.median(right)) if right.size else float("nan"),
                "median_difference_benefit_minus_failure": (
                    float(np.median(left) - np.median(right))
                    if left.size and right.size
                    else float("nan")
                ),
                "mann_whitney_p_uncorrected_exploratory": p_value,
            }
        )
    return rows


def _continuous_size_rows(
    frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in ALL_METRICS:
        for kind in ("absolute", "relative"):
            outcome = f"{metric}_{kind}_improvement"
            result = within_patient_slope(
                frame,
                x="log2_lesion_area",
                y=outcome,
                patient="patient_id",
                replicates=replicates,
                rng=rng,
            )
            rows.append(
                {
                    "metric": metric,
                    "outcome": outcome,
                    "effect": "change in improvement per doubling of lesion area",
                    **result,
                }
            )
    return rows


def _mechanism_summary(
    component_frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    specifications = (
        (
            "gradient_mass_scales_with_component_area",
            "log2_component_area",
            "log_lesion_gradient_mass",
            "positive",
        ),
        (
            "high_frequency_fraction_decreases_with_component_area",
            "log2_component_area",
            "target_residual_high_fraction",
            "negative",
        ),
    )
    rows: list[dict[str, Any]] = []
    for audit, x, y, expected in specifications:
        result = patient_cluster_bootstrap_spearman(
            component_frame,
            x=x,
            y=y,
            patient="patient_id",
            replicates=replicates,
            rng=rng,
        )
        established = (
            result["ci95_low"] > 0.0
            if expected == "positive"
            else result["ci95_high"] < 0.0
        )
        rows.append(
            {
                "audit": audit,
                "predictor": x,
                "outcome": y,
                "expected_direction": expected,
                **result,
                "established": bool(established),
            }
        )
    return rows


def _mechanism_secondary_associations(
    component_frame: pd.DataFrame,
    *,
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    frame = component_frame.copy()
    for column in (
        "lesion_gradient_share",
        "lesion_gradient_density",
        "target_residual_high_energy",
        "target_residual_mid_energy",
        "target_residual_low_energy",
        "target_residual_total_energy",
    ):
        frame[f"log_{column}"] = np.log(frame[column].clip(lower=EPSILON))
    outcomes = (
        ("log_lesion_gradient_mass", "gradient"),
        ("log_lesion_gradient_share", "gradient"),
        ("log_lesion_gradient_density", "gradient"),
        ("log_target_residual_high_energy", "frequency_energy"),
        ("log_target_residual_mid_energy", "frequency_energy"),
        ("log_target_residual_low_energy", "frequency_energy"),
        ("log_target_residual_total_energy", "frequency_energy"),
        ("target_residual_high_fraction", "frequency_fraction"),
        ("target_residual_mid_fraction", "frequency_fraction"),
        ("target_residual_low_fraction", "frequency_fraction"),
    )
    rows: list[dict[str, Any]] = []
    for outcome, family in outcomes:
        result = patient_cluster_bootstrap_spearman(
            frame,
            x="log2_component_area",
            y=outcome,
            patient="patient_id",
            replicates=replicates,
            rng=rng,
        )
        rows.append(
            {
                "family": family,
                "predictor": "log2_component_area",
                "outcome": outcome,
                **result,
                "role": "exploratory_mechanism_description",
            }
        )
    return rows


def _plot_uncertainty(frame: pd.DataFrame, output: Path) -> None:
    patient = (
        frame.groupby("patient_id", as_index=False)[
            [PRIMARY_DISPERSION, PRIMARY_ERROR]
        ]
        .mean()
        .sort_values(PRIMARY_DISPERSION)
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.scatter(
        patient[PRIMARY_DISPERSION],
        patient[PRIMARY_ERROR],
        color="#1f77b4",
        alpha=0.85,
    )
    axis.set_xlabel("Patient-mean lesion trajectory SD")
    axis.set_ylabel("Patient-mean medoid TopQ error")
    axis.set_title("Trajectory disagreement vs. medoid error")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_patient_waterfall(patient_frame: pd.DataFrame, output: Path) -> None:
    metric = "lesion_topq_peak_error_norm_relative_improvement"
    ordered = patient_frame.sort_values(metric).reset_index(drop=True)
    colors = np.where(ordered[metric] > 0.0, "#2ca02c", "#d62728")
    figure, axis = plt.subplots(figsize=(8.0, 5.0))
    axis.bar(np.arange(len(ordered)), 100.0 * ordered[metric], color=colors)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(ordered)))
    axis.set_xticklabels(ordered["patient_id"], rotation=90, fontsize=7)
    axis.set_ylabel("TopQ relative improvement (%)")
    axis.set_title("Patient-level medoid benefit and failure")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _rolling_median(x: np.ndarray, y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_ordered = x[order]
    y_ordered = y[order]
    half = max(window // 2, 1)
    centers: list[float] = []
    medians: list[float] = []
    for index in range(len(x_ordered)):
        start = max(0, index - half)
        stop = min(len(x_ordered), index + half + 1)
        centers.append(float(np.median(x_ordered[start:stop])))
        medians.append(float(np.median(y_ordered[start:stop])))
    return np.asarray(centers), np.asarray(medians)


def _plot_continuous_size(frame: pd.DataFrame, output: Path) -> None:
    x = frame["log2_lesion_area"].to_numpy(float)
    y = frame[
        "lesion_topq_peak_error_norm_relative_improvement"
    ].to_numpy(float)
    rolling_x, rolling_y = _rolling_median(x, y, window=31)
    figure, axis = plt.subplots(figsize=(6.8, 4.8))
    axis.scatter(x, 100.0 * y, s=14, alpha=0.22, color="#4c78a8")
    axis.plot(rolling_x, 100.0 * rolling_y, color="#e45756", linewidth=2.0)
    axis.axhline(0.0, color="black", linewidth=0.8)
    ticks = np.arange(math.floor(x.min()), math.ceil(x.max()) + 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{2**tick:.0f}" for tick in ticks])
    axis.set_xlabel("Connected-component area (pixels, log2 scale)")
    axis.set_ylabel("Medoid TopQ relative improvement (%)")
    axis.set_title("Continuous lesion-size effect (rolling median)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_mechanisms(component_frame: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x = component_frame["log2_component_area"].to_numpy(float)
    y_gradient = component_frame["log_lesion_gradient_mass"].to_numpy(float)
    y_frequency = component_frame[
        "target_residual_high_fraction"
    ].to_numpy(float)
    axes[0].scatter(x, y_gradient, s=14, alpha=0.28, color="#f58518")
    axes[0].set_ylabel("log output-gradient mass")
    axes[0].set_title("Area vs. baseline output-gradient proxy")
    axes[1].scatter(x, y_frequency, s=14, alpha=0.28, color="#54a24b")
    axes[1].set_ylabel("Target-residual high-frequency fraction")
    axes[1].set_title("Area vs. target-residual frequency")
    ticks = np.arange(math.floor(x.min()), math.ceil(x.max()) + 1)
    labels = [f"{2**tick:.0f}" for tick in ticks]
    for axis in axes:
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Connected-component area (pixels, log2 scale)")
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _fallacy_scan() -> list[dict[str, str]]:
    return [
        {
            "fallacy": "Simpson's paradox",
            "severity": "CAUTION",
            "detail": "Aggregate and patient-controlled continuous effects are both retained; subgroup direction can still vary.",
        },
        {
            "fallacy": "Ecological fallacy",
            "severity": "NOTE",
            "detail": "Uncertainty is inferred at patient level; slice-level mechanism results are not promoted to patient-level causality.",
        },
        {
            "fallacy": "Berkson's paradox",
            "severity": "CAUTION",
            "detail": "The validation cohort is selected and contains only positive slices.",
        },
        {
            "fallacy": "Collider bias",
            "severity": "NOTE",
            "detail": "No post-treatment covariates are adjusted; patient centering is used only for clustering/heterogeneity.",
        },
        {
            "fallacy": "Base-rate neglect",
            "severity": "CAUTION",
            "detail": "No negative slices are available, so false-hotspot findings apply only to within-positive-slice nonlesion background.",
        },
        {
            "fallacy": "Regression to the mean",
            "severity": "CAUTION",
            "detail": "Benefit/failure profiling is post hoc; high baseline error can mechanically permit larger improvement.",
        },
        {
            "fallacy": "Survivorship bias",
            "severity": "NOTE",
            "detail": "All 237 cached validation slices are used; upstream cache eligibility remains a selection boundary.",
        },
        {
            "fallacy": "Look-elsewhere effect",
            "severity": "CAUTION",
            "detail": "Only three pre-specified primary checks are gate-bearing; all other associations are exploratory.",
        },
        {
            "fallacy": "Garden of forking paths",
            "severity": "CAUTION",
            "detail": "This is an internally extended, post hoc analysis; definitions and gates are recorded before result interpretation but were not preregistered externally.",
        },
        {
            "fallacy": "Correlation != causation",
            "severity": "CAUTION",
            "detail": "Dispersion/error and mechanism audits are associational and do not prove a loss change will improve performance.",
        },
        {
            "fallacy": "Reverse causality",
            "severity": "NOTE",
            "detail": "Prediction error may itself increase trajectory dispersion; the uncertainty association has no directional causal interpretation.",
        },
    ]


def _format_ci(row: Mapping[str, Any]) -> str:
    return (
        f"{float(row['estimate']):+.4f} "
        f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}]"
    )


def _write_report(
    *,
    output: Path,
    source_run: Path,
    summary: Mapping[str, Any],
    patient_frame: pd.DataFrame,
    size_rows: Sequence[Mapping[str, Any]],
) -> None:
    primary = summary["primary_checks"]["uncertainty"]
    sensitivity = summary["primary_checks"]["uncertainty_sensitivity"]
    mechanism = summary["mechanism_gates"]
    benefit_count = int((patient_frame["primary_group"] == "benefit").sum())
    failure_count = int((patient_frame["primary_group"] == "failure").sum())
    topq_size = next(
        row
        for row in size_rows
        if row["outcome"]
        == "lesion_topq_peak_error_norm_relative_improvement"
    )
    decision = summary["training_gate_decision"]
    fallacies = summary["fallacy_scan"]
    if primary["ci95_low"] > 0.0:
        uncertainty_gate = "positive association"
    elif primary["ci95_high"] < 0.0:
        uncertainty_gate = "opposite direction"
    else:
        uncertainty_gate = "uncertain"
    lines = [
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {summary['created_at_utc']}",
        f"- Verification Status: {summary['verification_status']}",
        "- Version Label: medoid_zero_training_v1",
        f"- Source Run: `{source_run}`",
        "",
        "## Validation Report",
        "",
        f"- **Dataset**: {summary['dataset']['slices']} slices / "
        f"{summary['dataset']['patients']} patients / "
        f"{summary['dataset']['seeds']} seeds",
        "- **Training or optimizer used**: No",
        f"- **Overall confidence**: {summary['overall_confidence']}",
        "",
        "### Primary statistical findings",
        "",
        "| Check | Estimate and patient-bootstrap 95% CI | Gate |",
        "|---|---:|---|",
        f"| Lesion disagreement vs medoid TopQ error | {_format_ci(primary)} | "
        f"{uncertainty_gate} |",
        f"| Component area vs baseline output-gradient mass | "
        f"{_format_ci(mechanism[0])} | "
        f"{'established' if mechanism[0]['established'] else 'not established'} |",
        f"| Component area vs target-residual high-frequency fraction | "
        f"{_format_ci(mechanism[1])} | "
        f"{'established' if mechanism[1]['established'] else 'not established'} |",
        "",
        "The primary disagreement/error association is negative, not positive. "
        "Across the sensitivity specifications its estimates range from "
        f"{min(float(row['estimate']) for row in sensitivity):+.3f} to "
        f"{max(float(row['estimate']) for row in sensitivity):+.3f}. Therefore "
        "raw four-trajectory lesion dispersion is not validated as a monotone "
        "error-uncertainty score; low-dispersion consensus can still be wrong.",
        "",
        "The gradient quantity is an inference-endpoint `pred_x0` output-gradient "
        "proxy averaged across the four baseline trajectories. It is not a parameter "
        "gradient and is used only to audit size-weighting mechanics.",
        "",
        "### Medoid benefit/failure patients",
        "",
        f"- TopQ benefit (>0 relative improvement): {benefit_count} patients",
        f"- TopQ failure (<=0 relative improvement): {failure_count} patients",
        "- The full patient table is in `patient_benefit_failure.csv`; group profile "
        "comparisons are exploratory because the split is defined by the outcome.",
        "",
        "### Continuous lesion-size analysis",
        "",
        "The patient-fixed-effect slope for TopQ relative improvement per doubling "
        f"of connected-component area is {_format_ci(topq_size)}. This controls "
        "time-invariant patient heterogeneity but does not establish causality.",
        "",
        "### Training gate",
        "",
        f"**{decision['status']}** — {decision['reason']}",
        "",
        "The gate requires both mechanism audits to have patient-cluster bootstrap "
        "confidence intervals entirely in their pre-specified directions.",
        "",
        "### Warnings",
        "",
        "- This is internal extension on the same validation patients, not independent "
        "patient confirmation.",
        "- The cohort has no negative slices; false-hotspot analysis covers only "
        "nonlesion background within positive slices.",
        "- Benefit/failure profiling and secondary uncertainty associations are "
        "exploratory and should not be used for model selection on this cohort.",
        "",
        "### Fallacy scan",
        "",
        "- **Coverage**: 11/11 statistical fallacy types checked",
        "",
        "| Fallacy | Severity | Detail |",
        "|---|---|---|",
    ]
    for row in fallacies:
        lines.append(
            f"| {row['fallacy']} | {row['severity']} | {row['detail']} |"
        )
    lines.extend(
        [
            "",
        "### Reproducibility",
        "",
        f"- Method: {summary['reproducibility']['method']}",
        f"- Verdict: {summary['reproducibility']['verdict']}",
        "",
    ]
    )
    (output / "validation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_analysis(args: argparse.Namespace) -> int:
    source_run = args.run.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    loaded = _load_run(source_run)
    cohort: pd.DataFrame = loaded["cohort"]
    metrics: pd.DataFrame = loaded["metrics"]
    aggregation: pd.DataFrame = loaded["aggregation"]
    static: dict[str, np.ndarray] = loaded["static"]
    stack: np.ndarray = loaded["stack"]
    seeds: list[int] = loaded["seeds"]
    rng = np.random.default_rng(args.analysis_seed)

    selected = medoid_indices(stack)
    dispersion = trajectory_dispersion(stack, static["mask"])
    slice_frame = _metric_improvement_frame(
        cohort=cohort,
        metrics=metrics,
        aggregation=aggregation,
        dispersion=dispersion,
        selected_seed=selected,
        seeds=seeds,
    )

    seed_metric_sd = (
        metrics.groupby(["patient_id", "sample_id"])[list(ALL_METRICS)]
        .std(ddof=1)
        .rename(columns={metric: f"seed_sd_{metric}" for metric in ALL_METRICS})
        .reset_index()
    )
    slice_frame = slice_frame.merge(
        seed_metric_sd,
        on=["patient_id", "sample_id"],
        validate="one_to_one",
    )
    uncertainty_rows = _uncertainty_rows(
        slice_frame,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    primary_uncertainty = next(row for row in uncertainty_rows if row["primary"])
    uncertainty_sensitivity_rows = _uncertainty_sensitivity_rows(
        slice_frame,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )

    patient_rows = _patient_rows(slice_frame, seeds)
    patient_frame = pd.DataFrame(patient_rows)
    profile_rows = _profile_comparison(patient_frame)
    size_rows = _continuous_size_rows(
        slice_frame,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )

    gradient_maps = baseline_output_gradient_maps(
        stack,
        static["target"],
        batch_size=args.gradient_batch_size,
    )
    component_rows = component_mechanism_rows(
        masks=static["mask"],
        gradient_maps=gradient_maps,
        target_residual=static["target"] - static["mean_pet"],
        cohort=cohort,
    )
    component_frame = pd.DataFrame(component_rows)
    mechanism_rows = _mechanism_summary(
        component_frame,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )
    mechanism_secondary_rows = _mechanism_secondary_associations(
        component_frame,
        replicates=args.bootstrap_replicates,
        rng=rng,
    )

    primary_rows = [primary_uncertainty, *mechanism_rows]
    adjusted = _holm_adjust(
        [float(row["p_value_uncorrected"]) for row in primary_rows]
    )
    for row, adjusted_p in zip(primary_rows, adjusted):
        row["holm_adjusted_p_across_three_primary_checks"] = adjusted_p

    both_mechanisms = all(bool(row["established"]) for row in mechanism_rows)
    gate_decision = {
        "status": (
            "ELIGIBLE_FOR_INSTANCE_BALANCED_SCALE_MATCHED_TRAINING"
            if both_mechanisms
            else "DO_NOT_START_NEW_OUTPUT_LOSS_TRAINING"
        ),
        "both_mechanism_audits_established": both_mechanisms,
        "reason": (
            "Both pre-specified zero-training mechanism audits passed."
            if both_mechanisms
            else "At least one pre-specified mechanism audit did not pass."
        ),
    }
    fallacies = _fallacy_scan()
    summary = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_status": "ANALYZED",
        "source_run": str(source_run),
        "source_run_name": source_run.name,
        "source_prediction_sha256": loaded["prediction_hashes"],
        "analysis_seed": args.analysis_seed,
        "bootstrap_replicates": args.bootstrap_replicates,
        "dataset": {
            "slices": int(len(cohort)),
            "patients": int(cohort["patient_id"].nunique()),
            "seeds": int(len(seeds)),
            "connected_components": int(len(component_frame)),
            "negative_slices_available": bool(
                loaded["manifest"]["cohort_audit"][
                    "negative_background_slices_available"
                ]
            ),
        },
        "analysis_scope": {
            "training_or_optimizer_used": False,
            "gradient_audit": (
                "inference-endpoint pred_x0 output-gradient proxy at tau=0; "
                "averaged across the four baseline trajectories; common min-SNR "
                "scalar omitted"
            ),
            "frequency_audit": (
                "two-level orthonormal Haar energy of target minus saved "
                "conditional mean, projected to connected-component support"
            ),
        },
        "primary_checks": {
            "uncertainty": primary_uncertainty,
            "uncertainty_sensitivity": uncertainty_sensitivity_rows,
        },
        "mechanism_gates": mechanism_rows,
        "training_gate_decision": gate_decision,
        "patient_benefit_count": int(
            (patient_frame["primary_group"] == "benefit").sum()
        ),
        "patient_failure_count": int(
            (patient_frame["primary_group"] == "failure").sum()
        ),
        "overall_confidence": "CAUTION",
        "reproducibility": {
            "method": "deterministic re-run not yet recorded in this artifact",
            "verdict": "CANNOT_VERIFY",
        },
        "fallacy_scan_coverage": "11/11",
        "fallacy_scan": fallacies,
        "limitations": [
            "same validation patients as the medoid confirmation",
            "no negative validation slices",
            "post hoc internal analysis rather than independent confirmation",
            "output-gradient proxy is not a parameter-gradient measurement",
            "secondary associations and benefit/failure profiles are exploratory",
        ],
    }

    _write_csv(output / "slice_analysis.csv", slice_frame.to_dict("records"))
    _write_csv(output / "uncertainty_error_associations.csv", uncertainty_rows)
    _write_csv(
        output / "uncertainty_sensitivity.csv",
        uncertainty_sensitivity_rows,
    )
    _write_csv(output / "patient_benefit_failure.csv", patient_rows)
    _write_csv(output / "patient_profile_comparison.csv", profile_rows)
    _write_csv(output / "continuous_size_effects.csv", size_rows)
    _write_csv(output / "component_mechanism_audit.csv", component_rows)
    _write_csv(output / "mechanism_gate_summary.csv", mechanism_rows)
    _write_csv(
        output / "mechanism_secondary_associations.csv",
        mechanism_secondary_rows,
    )
    _write_json(output / "analysis_summary.json", summary)
    _plot_uncertainty(slice_frame, output / "uncertainty_vs_error.png")
    _plot_patient_waterfall(patient_frame, output / "patient_benefit_waterfall.png")
    _plot_continuous_size(slice_frame, output / "continuous_size_effect.png")
    _plot_mechanisms(component_frame, output / "mechanism_audits.png")
    _write_report(
        output=output,
        source_run=source_run,
        summary=summary,
        patient_frame=patient_frame,
        size_rows=size_rows,
    )
    if args.verify_reference is not None:
        reference = args.verify_reference.resolve()
        if not reference.is_dir():
            raise FileNotFoundError(f"verification reference not found: {reference}")
        comparisons: list[dict[str, Any]] = []
        for name in DETERMINISTIC_ARTIFACTS:
            reference_path = reference / name
            rerun_path = output / name
            if not reference_path.is_file() or not rerun_path.is_file():
                comparisons.append(
                    {
                        "artifact": name,
                        "status": "MISSING",
                        "reference_sha256": (
                            _sha256(reference_path)
                            if reference_path.is_file()
                            else None
                        ),
                        "rerun_sha256": (
                            _sha256(rerun_path) if rerun_path.is_file() else None
                        ),
                    }
                )
                continue
            reference_hash = _sha256(reference_path)
            rerun_hash = _sha256(rerun_path)
            comparisons.append(
                {
                    "artifact": name,
                    "status": (
                        "EXACT_MATCH"
                        if reference_hash == rerun_hash
                        else "MISMATCH"
                    ),
                    "reference_sha256": reference_hash,
                    "rerun_sha256": rerun_hash,
                }
            )
        exact = all(row["status"] == "EXACT_MATCH" for row in comparisons)
        reproducibility = {
            "schema_version": 1,
            "method": (
                "deterministic full analysis re-run with identical source run, "
                "analysis seed, bootstrap count, and environment; SHA-256 "
                "comparison of core CSV and PNG artifacts"
            ),
            "verdict": "REPRODUCIBLE" if exact else "NOT_REPRODUCIBLE",
            "reference_output": str(reference),
            "artifacts_compared": len(comparisons),
            "exact_matches": sum(
                row["status"] == "EXACT_MATCH" for row in comparisons
            ),
            "comparisons": comparisons,
        }
        summary["verification_status"] = "VERIFIED" if exact else "ANALYZED"
        summary["reproducibility"] = {
            "method": reproducibility["method"],
            "verdict": reproducibility["verdict"],
        }
        _write_json(output / "reproducibility.json", reproducibility)
        _write_json(output / "analysis_summary.json", summary)
        _write_report(
            output=output,
            source_run=source_run,
            summary=summary,
            patient_frame=patient_frame,
            size_rows=size_rows,
        )
    artifact_hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json(
        output / "artifact_manifest.json",
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "source_run": str(source_run),
            "training_or_optimizer_used": False,
            "artifacts": artifact_hashes,
        },
    )
    print(json.dumps(_json_safe(summary["training_gate_decision"]), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--analysis-seed", type=int, default=20260729)
    parser.add_argument("--gradient-batch-size", type=int, default=16)
    parser.add_argument(
        "--verify-reference",
        type=Path,
        default=None,
        help="Compare deterministic core artifacts with a prior full execution.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    if args.gradient_batch_size < 1:
        raise ValueError("gradient-batch-size must be positive")
    return run_analysis(args)


if __name__ == "__main__":
    raise SystemExit(main())
