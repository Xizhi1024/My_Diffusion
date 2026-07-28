"""One-shot Stage-2/Stage-3 router and PET-background evaluation suite.

The suite keeps one checkpoint, EMA weights, dataset order, and noise seed
fixed while crossing destination interventions with DDIM step counts:

    learned | native_only | fixed | shuffled_destination
        x
    full | detail_off | ll_off | all_frequency_off
        x
    20 | 100 steps (configurable)

It also decomposes ``mean_pet + pred_residual`` and reports lesion recovery,
global-hotspot, non-lesion body low-pass, high-pass energy, quantile, stripe,
failure, and false-hotspot metrics.

Example (run from the repository root):

    pixi run python -u -m scripts.eval_router_background_suite `
      --config configs/experiments/slmf_png_prior_anchored_router_300e.yaml `
      --checkpoint results/prior_anchored_router_300e/checkpoints/ckpt_epoch0100.pt `
      --metrics results/prior_anchored_router_300e/checkpoints/training_metrics.jsonl `
      --split val --max-samples 64 --steps 20,100 `
      --output-dir results/router_background_suite/epoch0100

The fixed destination values are read from the checkpoint epoch's train block
in ``training_metrics.jsonl``.  They are not estimated from validation output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate import (  # noqa: E402
    _annotate_small_lesion_metrics,
    _build_stratified_subset,
    _de_collate_meta,
    _seed_evaluation,
    _select_checkpoint_state,
    _to_numpy,
    compute_boundary_metrics,
    compute_failure_detection_metrics,
    compute_false_hotspot_count,
    compute_mae,
    compute_mse,
    compute_normalized_lesion_metrics,
    compute_psnr,
    compute_ssim,
    compute_stripe_metrics,
    compute_target_relative_false_hotspots,
)
from src.data.lineage import (  # noqa: E402
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile  # noqa: E402
from src.model.slmf_bbdm import SLMFBBDM  # noqa: E402
from src.model.trainer import _to_unit_interval  # noqa: E402


DEFAULT_MODES = (
    "learned",
    "native_only",
    "fixed",
    "shuffled_destination",
)
FREQUENCY_INTERVENTIONS = (
    "full",
    "detail_off",
    "ll_off",
    "all_frequency_off",
)
DEFAULT_FREQUENCY_MODES = ("full",)
LOWER_IS_BETTER = {
    "lesion_topq_peak_error_norm": True,
    "lesion_peak_error_norm": True,
    "failure": True,
    "failure_any": True,
    "small_lesion_underestimate": True,
    "outside_inside_peak_ratio": True,
    "target_relative_false_hotspot_density": True,
    "target_relative_false_hotspot_components": True,
    "stripe_abs_excess": True,
    "body_nonlesion_lowpass_mae": True,
    "body_nonlesion_mae": True,
    "global_hotspot_distance_to_lesion": True,
    "lesion_peak_recovery_at_ratio": False,
    "lesion_topq_recovery_at_ratio": False,
    "global_hotspot_hit": False,
    "small_lesion_peak_recovery_at_ratio": False,
    "small_lesion_topq_recovery_at_ratio": False,
}
FIXED_KEY_TEMPLATE = (
    "frequency/prior_anchor_{level}_{band}_conditional_shallow"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _parse_csv_values(raw: str, *, cast, label_name: str) -> list[Any]:
    values = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError(f"{label_name} cannot be empty")
    return [cast(value) for value in values]


def _load_fixed_destination(
    metrics_path: Path,
    epoch: int,
) -> dict[str, list[float]]:
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Fixed destination requires training metrics: {metrics_path}"
        )
    selected: Mapping[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {metrics_path}:{line_number}"
                ) from exc
            if int(row.get("epoch", -1)) == int(epoch):
                train = row.get("train")
                if not isinstance(train, Mapping):
                    raise RuntimeError(
                        f"Epoch {epoch} has no train metric block"
                    )
                selected = train
                break
    if selected is None:
        raise RuntimeError(
            f"Epoch {epoch} was not found in {metrics_path}"
        )

    result: dict[str, list[float]] = {}
    missing: list[str] = []
    for level in ("l2", "l1"):
        values: list[float] = []
        for band in ("lh", "hl", "hh"):
            key = FIXED_KEY_TEMPLATE.format(level=level, band=band)
            value = _finite_or_none(selected.get(key))
            if value is None:
                missing.append(key)
            else:
                values.append(value)
        if len(values) == 3:
            result[level] = values
    if missing:
        raise RuntimeError(
            "Checkpoint-epoch metrics lack fixed-destination values: "
            + ", ".join(missing)
        )
    return result


def _largest_component(binary: np.ndarray) -> np.ndarray:
    components, count = label(binary)
    if count <= 0:
        return np.zeros_like(binary, dtype=bool)
    sizes = np.bincount(components.reshape(-1))
    sizes[0] = 0
    return components == int(np.argmax(sizes))


def _body_nonlesion_mask(
    ct: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    body_threshold: float,
    exclusion_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    ct_unit = _to_unit_interval(np.asarray(ct, dtype=np.float32))[0]
    body = ct_unit > float(body_threshold)
    body = binary_closing(body, structure=np.ones((5, 5), dtype=bool))
    body = binary_fill_holes(body)
    body = _largest_component(body)

    lesion = np.asarray(lesion_mask)[0] > 0.5
    if exclusion_radius > 0 and lesion.any():
        width = 2 * int(exclusion_radius) + 1
        excluded = binary_dilation(
            lesion,
            structure=np.ones((width, width), dtype=bool),
        )
    else:
        excluded = lesion
    return body, body & ~excluded


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[mask]
    return float(selected.mean()) if selected.size else float("nan")


def _masked_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    return _masked_mean(np.abs(prediction - target), mask)


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values)[mask]
    return (
        float(np.sqrt(np.mean(np.square(selected))))
        if selected.size
        else float("nan")
    )


def compute_body_background_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mean_pet: np.ndarray,
    pred_residual: np.ndarray,
    ct: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    lowpass_sigma: float,
    body_threshold: float,
    lesion_exclusion_radius: int,
) -> dict[str, float]:
    """Measure low/high-frequency fidelity inside CT body, outside the lesion."""
    body, region = _body_nonlesion_mask(
        ct,
        lesion_mask,
        body_threshold=body_threshold,
        exclusion_radius=lesion_exclusion_radius,
    )
    pred_unit = _to_unit_interval(pred)[0]
    target_unit = _to_unit_interval(target)[0]
    mean_unit = _to_unit_interval(mean_pet)[0]
    residual_unit_delta = np.asarray(pred_residual, dtype=np.float32)[0] * 0.5

    pred_low = gaussian_filter(pred_unit, sigma=lowpass_sigma, mode="reflect")
    target_low = gaussian_filter(
        target_unit, sigma=lowpass_sigma, mode="reflect"
    )
    mean_low = gaussian_filter(mean_unit, sigma=lowpass_sigma, mode="reflect")
    residual_low = gaussian_filter(
        residual_unit_delta,
        sigma=lowpass_sigma,
        mode="reflect",
    )
    pred_high = pred_unit - pred_low
    target_high = target_unit - target_low
    pred_high_rms = _masked_rms(pred_high, region)
    target_high_rms = _masked_rms(target_high, region)
    final_low_mae = _masked_mae(pred_low, target_low, region)
    mean_low_mae = _masked_mae(mean_low, target_low, region)

    metrics = {
        "body_fraction": float(body.mean()),
        "body_nonlesion_fraction": float(region.mean()),
        "body_nonlesion_pred_mean": _masked_mean(pred_unit, region),
        "body_nonlesion_target_mean": _masked_mean(target_unit, region),
        "body_nonlesion_mean_bias": _masked_mean(
            pred_unit - target_unit, region
        ),
        "body_nonlesion_mae": _masked_mae(pred_unit, target_unit, region),
        "body_nonlesion_lowpass_mae": final_low_mae,
        "mean_body_nonlesion_lowpass_mae": mean_low_mae,
        "body_nonlesion_lowpass_error_reduction_from_mean": (
            mean_low_mae - final_low_mae
        ),
        "body_nonlesion_lowpass_residual_rms": _masked_rms(
            residual_low, region
        ),
        "body_nonlesion_pred_highpass_rms": pred_high_rms,
        "body_nonlesion_target_highpass_rms": target_high_rms,
        "body_nonlesion_highpass_energy_ratio": (
            pred_high_rms / max(target_high_rms, 1e-8)
            if math.isfinite(pred_high_rms)
            and math.isfinite(target_high_rms)
            else float("nan")
        ),
    }
    pred_values = pred_unit[region]
    target_values = target_unit[region]
    for quantile in (10, 25, 50, 75, 90, 95):
        if pred_values.size and target_values.size:
            pred_q = float(np.percentile(pred_values, quantile))
            target_q = float(np.percentile(target_values, quantile))
            metrics[f"body_nonlesion_q{quantile}_bias"] = pred_q - target_q
            metrics[f"body_nonlesion_q{quantile}_abs_error"] = abs(
                pred_q - target_q
            )
        else:
            metrics[f"body_nonlesion_q{quantile}_bias"] = float("nan")
            metrics[f"body_nonlesion_q{quantile}_abs_error"] = float("nan")
    return metrics


def compute_global_hotspot_metrics(
    pred: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    hit_radius: int,
) -> dict[str, float]:
    pred_unit = _to_unit_interval(pred)[0]
    lesion = np.asarray(lesion_mask)[0] > 0.5
    if not lesion.any():
        return {
            "global_hotspot_hit": float("nan"),
            "global_hotspot_distance_to_lesion": float("nan"),
        }
    peak = np.unravel_index(int(np.argmax(pred_unit)), pred_unit.shape)
    if hit_radius > 0:
        width = 2 * int(hit_radius) + 1
        hit_region = binary_dilation(
            lesion,
            structure=np.ones((width, width), dtype=bool),
        )
    else:
        hit_region = lesion
    distances = distance_transform_edt(~lesion)
    return {
        "global_hotspot_hit": float(bool(hit_region[peak])),
        "global_hotspot_distance_to_lesion": float(distances[peak]),
    }


def compute_false_hotspot_components(
    pred: np.ndarray,
    target: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    excess_margin: float = 0.05,
    target_quantile: float = 0.99,
) -> dict[str, float]:
    pred_2d = np.asarray(pred, dtype=np.float32)[0]
    target_2d = np.asarray(target, dtype=np.float32)[0]
    outside = ~(np.asarray(lesion_mask)[0] > 0.5)
    if not outside.any():
        return {"target_relative_false_hotspot_components": 0.0}
    threshold = float(np.quantile(target_2d[outside], target_quantile))
    hotspot = (
        outside
        & ((pred_2d - target_2d) > excess_margin)
        & (pred_2d > threshold)
    )
    _, components = label(hotspot)
    return {"target_relative_false_hotspot_components": float(components)}


def _sample_identity(
    meta: Mapping[str, Any],
    *,
    batch_index: int,
    sample_index: int,
) -> tuple[str, str]:
    patient = str(meta.get("patient_id", f"sample_{batch_index}_{sample_index}"))
    slice_id = str(meta.get("slice_id", sample_index))
    sample_id = str(meta.get("sample_id", f"{patient}_{slice_id}"))
    return sample_id, patient


def _numeric_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            converted = _finite_or_none(value)
            if converted is not None:
                numeric[key].append(converted)
    summary: dict[str, Any] = {"num_samples": len(rows)}
    for key, values in numeric.items():
        array = np.asarray(values, dtype=np.float64)
        summary[f"{key}_mean"] = float(array.mean())
        summary[f"{key}_std"] = float(array.std())
        summary[f"{key}_median"] = float(np.median(array))
    patients = sorted({str(row["patient_id"]) for row in rows})
    summary["num_patients"] = len(patients)
    return summary


def _per_patient_metric(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_or_none(row.get(metric))
        if value is not None:
            grouped[str(row["patient_id"])].append(value)
    return {
        patient: float(np.mean(values))
        for patient, values in grouped.items()
        if values
    }


def _paired_effect(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    lower_is_better: bool,
    seed: int,
    bootstrap_samples: int = 5000,
) -> dict[str, Any] | None:
    candidate_patient = _per_patient_metric(candidate, metric)
    baseline_patient = _per_patient_metric(baseline, metric)
    patients = sorted(set(candidate_patient).intersection(baseline_patient))
    if not patients:
        return None
    effects = np.asarray(
        [
            (
                baseline_patient[patient] - candidate_patient[patient]
                if lower_is_better
                else candidate_patient[patient] - baseline_patient[patient]
            )
            for patient in patients
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    if effects.size == 1:
        low = high = float(effects[0])
    else:
        indices = rng.integers(
            0,
            effects.size,
            size=(bootstrap_samples, effects.size),
        )
        means = effects[indices].mean(axis=1)
        low, high = (
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        )
    return {
        "metric": metric,
        "patients": len(patients),
        "effect_definition": "positive means candidate better than learned",
        "mean_effect": float(effects.mean()),
        "ci95_low": low,
        "ci95_high": high,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _save_grid(
    path: Path,
    records: Sequence[Mapping[str, np.ndarray]],
    *,
    title: str,
    lowpass_sigma: float,
) -> None:
    if not records:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = (
        "CT",
        "Target PET",
        "Mean PET",
        "Pred residual",
        "Pred PET",
        "Target low-pass",
        "Pred low-pass",
        "Lesion mask",
    )
    figure, axes = plt.subplots(
        len(records),
        len(columns),
        figsize=(3 * len(columns), 3 * len(records)),
        squeeze=False,
    )
    residual_scale = max(
        float(np.max(np.abs(record["residual"])))
        for record in records
    )
    residual_scale = max(residual_scale, 1e-3)
    for row_index, record in enumerate(records):
        target_low = gaussian_filter(
            record["target"],
            sigma=lowpass_sigma,
            mode="reflect",
        )
        pred_low = gaussian_filter(
            record["pred"],
            sigma=lowpass_sigma,
            mode="reflect",
        )
        panels = (
            (record["ct"], "gray", -1.0, 1.0),
            (record["target"], "hot", -1.0, 1.0),
            (record["mean"], "hot", -1.0, 1.0),
            (
                record["residual"],
                "coolwarm",
                -residual_scale,
                residual_scale,
            ),
            (record["pred"], "hot", -1.0, 1.0),
            (target_low, "hot", -1.0, 1.0),
            (pred_low, "hot", -1.0, 1.0),
            (record["mask"], "gray", 0.0, 1.0),
        )
        for column_index, (image, cmap, vmin, vmax) in enumerate(panels):
            axis = axes[row_index, column_index]
            axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_index == 0:
                axis.set_title(columns[column_index])
            if column_index == 0:
                axis.set_ylabel(str(record["sample_id"]))
            axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


@torch.no_grad()
def evaluate_variant(
    model: SLMFBBDM,
    loader: DataLoader,
    *,
    device: str,
    amp: bool,
    steps: int,
    seed: int,
    lowpass_sigma: float,
    body_threshold: float,
    lesion_exclusion_radius: int,
    hotspot_hit_radius: int,
    recovery_ratio: float,
    small_lesion_quantile: float,
    small_lesion_underestimate_tolerance: float,
    grid_samples: int,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    _seed_evaluation(seed)
    model.eval()
    rows: list[dict[str, Any]] = []
    grid: list[dict[str, np.ndarray]] = []
    amp_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    device_type = torch.device(device).type

    for batch_index, batch in enumerate(loader):
        batch_gpu = {
            key: value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        with torch.amp.autocast(
            "cuda",
            enabled=amp and device_type == "cuda",
            dtype=amp_dtype,
        ):
            output = model.sample(batch_gpu, num_steps=steps, progress=False)
        pred = output["synthetic_pet"]
        mean_pet = output.get("mean_pet")
        residual = output.get("pred_residual")
        if not torch.is_tensor(mean_pet) or not torch.is_tensor(residual):
            raise RuntimeError(
                "The suite requires residual_bridge output "
                "(mean_pet and pred_residual)"
            )

        target = batch_gpu["pet"]
        ct = batch_gpu["ct"]
        mask = batch_gpu.get("mask", torch.zeros_like(target))
        organ = batch_gpu.get(
            "organ_mask",
            torch.zeros(
                target.shape[0],
                6,
                *target.shape[2:],
                device=target.device,
            ),
        )
        metadata = _de_collate_meta(batch_gpu.get("meta"), pred.shape[0])
        for index in range(pred.shape[0]):
            pred_np = _to_numpy(pred[index])
            target_np = _to_numpy(target[index])
            ct_np = _to_numpy(ct[index])
            mask_np = _to_numpy(mask[index])
            organ_np = _to_numpy(organ[index])
            mean_np = _to_numpy(mean_pet[index])
            residual_np = _to_numpy(residual[index])
            meta = metadata[index] if index < len(metadata) else {}
            sample_id, patient_id = _sample_identity(
                meta,
                batch_index=batch_index,
                sample_index=index,
            )
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "slice_id": str(meta.get("slice_id", index)),
                "mae": compute_mae(pred_np, target_np),
                "mse": compute_mse(pred_np, target_np),
                "psnr": compute_psnr(pred_np, target_np),
                "ssim": compute_ssim(pred_np[0], target_np[0]),
            }
            stripe = compute_stripe_metrics(pred_np, target_np)
            row.update(stripe)
            row["stripe_abs_excess"] = abs(stripe["stripe_excess"])
            row.update(
                compute_boundary_metrics(
                    pred_np,
                    target_np,
                    ct_np,
                    mask_np,
                    organ_np,
                )
            )
            row.update(
                compute_body_background_metrics(
                    pred_np,
                    target_np,
                    mean_np,
                    residual_np,
                    ct_np,
                    mask_np,
                    lowpass_sigma=lowpass_sigma,
                    body_threshold=body_threshold,
                    lesion_exclusion_radius=lesion_exclusion_radius,
                )
            )
            if mask_np.sum() > 0:
                lesion = compute_normalized_lesion_metrics(
                    pred_np,
                    target_np,
                    mask_np,
                    organ_np,
                )
                row.update(lesion)
                # compute_normalized_lesion_metrics exposes Top-Q prediction
                # and target, while the max-peak prediction/target are derived
                # directly here to keep the recovery definition explicit.
                pred_unit = _to_unit_interval(pred_np)[0]
                target_unit = _to_unit_interval(target_np)[0]
                valid = mask_np[0] > 0.5
                pred_peak = float(pred_unit[valid].max())
                target_peak = float(target_unit[valid].max())
                topq_pred = float(lesion["lesion_topq_peak_pred_norm"])
                topq_target = float(lesion["lesion_topq_peak_target_norm"])
                row["lesion_peak_pred_target_ratio"] = (
                    pred_peak / max(target_peak, 1e-8)
                )
                row["lesion_topq_pred_target_ratio"] = (
                    topq_pred / max(topq_target, 1e-8)
                )
                row["lesion_peak_recovery_at_ratio"] = float(
                    row["lesion_peak_pred_target_ratio"] >= recovery_ratio
                )
                row["lesion_topq_recovery_at_ratio"] = float(
                    row["lesion_topq_pred_target_ratio"] >= recovery_ratio
                )
                row.update(
                    compute_global_hotspot_metrics(
                        pred_np,
                        mask_np,
                        hit_radius=hotspot_hit_radius,
                    )
                )
            failure = compute_failure_detection_metrics(
                pred_np,
                target_np,
                mask_np,
                outside_margin=0.05,
                lesion_min_ratio=recovery_ratio,
                uncertainty_ratio_threshold=2.0,
                model_space=True,
            )
            row.update(failure)
            row.update(compute_false_hotspot_count(pred_np, organ_np))
            row.update(
                compute_target_relative_false_hotspots(
                    pred_np,
                    target_np,
                    mask_np,
                )
            )
            row.update(
                compute_false_hotspot_components(
                    pred_np,
                    target_np,
                    mask_np,
                )
            )
            rows.append(row)

            if len(grid) < grid_samples:
                grid.append(
                    {
                        "sample_id": sample_id,
                        "ct": ct_np[0],
                        "target": target_np[0],
                        "mean": mean_np[0],
                        "residual": residual_np[0],
                        "pred": pred_np[0],
                        "mask": mask_np[0],
                    }
                )
        if verbose:
            print(
                f"    batch {batch_index + 1}/{len(loader)} "
                f"({len(rows)} samples)",
                flush=True,
            )

    _annotate_small_lesion_metrics(
        rows,
        quantile=small_lesion_quantile,
        underestimate_tolerance=small_lesion_underestimate_tolerance,
    )
    for row in rows:
        if row.get("small_lesion") == 1.0:
            if "lesion_peak_recovery_at_ratio" in row:
                row["small_lesion_peak_recovery_at_ratio"] = row[
                    "lesion_peak_recovery_at_ratio"
                ]
            if "lesion_topq_recovery_at_ratio" in row:
                row["small_lesion_topq_recovery_at_ratio"] = row[
                    "lesion_topq_recovery_at_ratio"
                ]
    return rows, grid


def _load_dataset(
    config: Mapping[str, Any],
    *,
    split: str,
    max_samples: int | None,
    allow_train_fallback: bool,
):
    from src.data.dataset import CachedDataset, FakeDataset

    data = config.get("data", {})
    if data.get("use_fake_data", False):
        dataset = FakeDataset(
            32,
            int(data.get("image_size", 192)),
            seed=int(config.get("experiment", {}).get("seed", 42)),
        )
    else:
        cache_dir = data.get("cache_dir")
        if not cache_dir:
            raise RuntimeError("No data.cache_dir configured")
        manifest = (
            Path(data["split_manifest"])
            if data.get("split_manifest")
            else None
        )
        try:
            dataset = CachedDataset(
                cache_dir,
                split=split,
                augment=False,
                split_manifest=manifest,
            )
        except ValueError:
            if not allow_train_fallback:
                raise
            dataset = CachedDataset(
                cache_dir,
                split="train",
                augment=False,
                split_manifest=manifest,
            )
    selected_indices = list(range(len(dataset)))
    if max_samples is not None:
        dataset, selected_indices = _build_stratified_subset(
            dataset,
            max_samples,
        )
    return dataset, selected_indices


def _resolve_metrics_path(checkpoint: Path, value: str | None) -> Path:
    if value:
        return Path(value)
    return checkpoint.parent / "training_metrics.jsonl"


def _main(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    modes = _parse_csv_values(
        args.modes,
        cast=str,
        label_name="modes",
    )
    unknown = sorted(set(modes).difference(DEFAULT_MODES))
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}")
    frequency_modes = _parse_csv_values(
        args.frequency_modes,
        cast=str,
        label_name="frequency-modes",
    )
    unknown_frequency_modes = sorted(
        set(frequency_modes).difference(FREQUENCY_INTERVENTIONS)
    )
    if unknown_frequency_modes:
        raise ValueError(
            f"Unknown frequency modes: {unknown_frequency_modes}"
        )
    steps = _parse_csv_values(
        args.steps,
        cast=int,
        label_name="steps",
    )
    if any(value <= 0 for value in steps):
        raise ValueError("All step counts must be positive")

    config = resolve_runtime_profile(load_full_config(str(config_path)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_evaluation(args.seed)

    print("Building model...", flush=True)
    model = SLMFBBDM.from_config(config)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    validate_checkpoint_data_lineage(
        checkpoint,
        load_checkpoint_data_lineage(config),
        required=bool(config.get("data", {}).get("require_cache_lineage", False)),
        context=f"router-background suite checkpoint {checkpoint_path}",
    )
    state, state_source = _select_checkpoint_state(
        checkpoint,
        weights=args.weights,
    )
    model.load_state_dict(state)
    model = model.to(device)

    router = getattr(model, "residual_preconditioner", None)
    if router is None or not hasattr(
        router,
        "set_inference_destination_intervention",
    ):
        raise RuntimeError(
            "Checkpoint model has no prior-anchored destination intervention API"
        )
    if not hasattr(router, "set_inference_frequency_intervention"):
        raise RuntimeError(
            "Checkpoint model has no frequency-branch intervention API"
        )
    checkpoint_epoch = int(checkpoint.get("epoch", args.fixed_epoch or 0))
    fixed_epoch = (
        int(args.fixed_epoch)
        if args.fixed_epoch is not None
        else checkpoint_epoch
    )
    metrics_path = _resolve_metrics_path(checkpoint_path, args.metrics)
    fixed = None
    if "fixed" in modes:
        fixed = _load_fixed_destination(metrics_path, fixed_epoch)

    dataset, selected_indices = _load_dataset(
        config,
        split=args.split,
        max_samples=args.max_samples,
        allow_train_fallback=args.allow_train_fallback,
    )
    data_config = config.get("data", {})
    batch_size = int(
        args.batch_size
        or data_config.get("val_batch_size", data_config.get("batch_size", 4))
    )
    if "shuffled_destination" in modes and batch_size < 2:
        raise ValueError(
            "shuffled_destination requires --batch-size >= 2"
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 2,
        "scope": "exploratory_same_checkpoint_inference_intervention",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_weights": state_source,
        "metrics": str(metrics_path) if metrics_path.is_file() else None,
        "metrics_sha256": _sha256(metrics_path) if metrics_path.is_file() else None,
        "fixed_destination_epoch": fixed_epoch if fixed else None,
        "fixed_destination": fixed,
        "modes": modes,
        "frequency_modes": frequency_modes,
        "steps": steps,
        "seed": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "split": args.split,
        "selected_indices": selected_indices,
        "num_samples": len(dataset),
        "batch_size": batch_size,
        "lowpass_sigma": args.lowpass_sigma,
        "body_threshold_unit_interval": args.body_threshold,
        "lesion_exclusion_radius": args.lesion_exclusion_radius,
        "hotspot_hit_radius": args.hotspot_hit_radius,
        "recovery_ratio": args.recovery_ratio,
        "small_lesion_quantile": args.small_lesion_quantile,
        "device": device,
    }
    _write_json(output_dir / "run_manifest.json", _json_safe(manifest))

    total_units = sum(steps) * len(modes) * len(frequency_modes)
    completed_units = 0
    all_rows: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    run_summaries: list[dict[str, Any]] = []
    for mode in modes:
        intervention_args: dict[str, Any] = {
            "mode": mode,
            "shuffle_seed": args.shuffle_seed,
        }
        if mode == "fixed":
            assert fixed is not None
            intervention_args.update(
                fixed_l2=fixed["l2"],
                fixed_l1=fixed["l1"],
            )
        router.set_inference_destination_intervention(**intervention_args)
        for frequency_mode in frequency_modes:
            router.set_inference_frequency_intervention(frequency_mode)
            for step_count in steps:
                if frequency_mode == "full":
                    label_name = f"{mode}_steps{step_count:03d}"
                else:
                    label_name = (
                        f"{mode}_{frequency_mode}_steps{step_count:03d}"
                    )
                print(
                    f"\n[{label_name}] {len(dataset)} samples, "
                    f"batch={batch_size}",
                    flush=True,
                )
                run_started = time.perf_counter()
                rows, grid = evaluate_variant(
                    model,
                    loader,
                    device=device,
                    amp=not args.no_amp,
                    steps=step_count,
                    seed=args.seed,
                    lowpass_sigma=args.lowpass_sigma,
                    body_threshold=args.body_threshold,
                    lesion_exclusion_radius=args.lesion_exclusion_radius,
                    hotspot_hit_radius=args.hotspot_hit_radius,
                    recovery_ratio=args.recovery_ratio,
                    small_lesion_quantile=args.small_lesion_quantile,
                    small_lesion_underestimate_tolerance=(
                        args.small_lesion_underestimate_tolerance
                    ),
                    grid_samples=args.grid_samples,
                )
                run_seconds = time.perf_counter() - run_started
                completed_units += step_count
                elapsed = time.perf_counter() - started
                eta = (
                    elapsed
                    / completed_units
                    * (total_units - completed_units)
                    if completed_units
                    else float("nan")
                )
                summary = _numeric_summary(rows)
                summary.update(
                    {
                        "mode": mode,
                        "frequency_mode": frequency_mode,
                        "steps": step_count,
                        "seconds": run_seconds,
                        "seconds_per_sample": (
                            run_seconds / max(len(rows), 1)
                        ),
                        "intervention": {
                            "destination": (
                                router.inference_destination_intervention()
                            ),
                            "frequency": (
                                router.inference_frequency_intervention()
                            ),
                        },
                    }
                )
                run_dir = output_dir / label_name
                run_dir.mkdir(parents=True, exist_ok=True)
                with (run_dir / "samples.jsonl").open(
                    "w",
                    encoding="utf-8",
                ) as handle:
                    for row in rows:
                        handle.write(
                            json.dumps(
                                _json_safe(row),
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                _write_json(run_dir / "summary.json", _json_safe(summary))
                _save_grid(
                    run_dir / "decomposition_grid.png",
                    grid,
                    title=label_name,
                    lowpass_sigma=args.lowpass_sigma,
                )
                all_rows[(mode, frequency_mode, step_count)] = rows
                run_summaries.append(summary)
                print(
                    f"  completed in {run_seconds:.1f}s; "
                    f"estimated remaining {max(eta, 0.0):.1f}s",
                    flush=True,
                )

    comparisons: list[dict[str, Any]] = []
    for mode in modes:
        for frequency_index, frequency_mode in enumerate(frequency_modes):
            if mode == "learned" and frequency_mode == "full":
                continue
            for step_count in steps:
                candidate = all_rows[(mode, frequency_mode, step_count)]
                baseline = all_rows.get(("learned", "full", step_count))
                if baseline is None:
                    continue
                for metric_index, (metric, lower) in enumerate(
                    LOWER_IS_BETTER.items()
                ):
                    effect = _paired_effect(
                        candidate,
                        baseline,
                        metric=metric,
                        lower_is_better=lower,
                        seed=(
                            args.seed
                            + step_count * 100
                            + frequency_index * len(LOWER_IS_BETTER)
                            + metric_index
                        ),
                    )
                    if effect is not None:
                        comparisons.append(
                            {
                                "candidate_mode": mode,
                                "candidate_frequency_mode": frequency_mode,
                                "baseline_mode": "learned",
                                "baseline_frequency_mode": "full",
                                "steps": step_count,
                                **effect,
                            }
                        )

    total_seconds = time.perf_counter() - started
    suite_summary = {
        "manifest": manifest,
        "runs": run_summaries,
        "paired_comparisons_vs_learned": comparisons,
        "paired_comparisons_vs_learned_full": comparisons,
        "total_seconds": total_seconds,
        "training_epoch_seconds_reference": args.seconds_per_epoch,
        "training_epoch_time_equivalent": (
            total_seconds / args.seconds_per_epoch
            if args.seconds_per_epoch > 0
            else None
        ),
        "interpretation": {
            "positive_paired_effect": "candidate better than learned",
            "fixed_scope": (
                "checkpoint-epoch train diagnostics; not validation-fitted"
            ),
            "global_hotspot_hit": (
                "lesion-dominant hotspot hit, not general lesion detection"
            ),
        },
    }
    _write_json(output_dir / "suite_summary.json", _json_safe(suite_summary))
    _write_csv(output_dir / "run_summary.csv", run_summaries)
    _write_csv(output_dir / "paired_comparisons_vs_learned.csv", comparisons)
    _write_csv(
        output_dir / "paired_comparisons_vs_learned_full.csv",
        comparisons,
    )
    print(
        f"\nSuite complete: {total_seconds:.1f}s "
        f"(~{total_seconds / max(args.seconds_per_epoch, 1e-8):.1f} "
        f"training epochs at {args.seconds_per_epoch:.1f}s/epoch)",
        flush=True,
    )
    print(f"Results: {output_dir}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage-2/Stage-3 router/background evaluation in one command."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--metrics",
        default=None,
        help="training_metrics.jsonl; defaults to checkpoint directory",
    )
    parser.add_argument("--fixed-epoch", type=int, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", default="20,100")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument(
        "--frequency-modes",
        default=",".join(DEFAULT_FREQUENCY_MODES),
        help=(
            "Comma-separated frequency interventions: "
            + ",".join(FREQUENCY_INTERVENTIONS)
        ),
    )
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=20260728)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-train-fallback", action="store_true")
    parser.add_argument("--lowpass-sigma", type=float, default=4.0)
    parser.add_argument("--body-threshold", type=float, default=0.03)
    parser.add_argument("--lesion-exclusion-radius", type=int, default=8)
    parser.add_argument("--hotspot-hit-radius", type=int, default=3)
    parser.add_argument("--recovery-ratio", type=float, default=0.6)
    parser.add_argument("--small-lesion-quantile", type=float, default=0.25)
    parser.add_argument(
        "--small-lesion-underestimate-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--grid-samples", type=int, default=4)
    parser.add_argument("--seconds-per-epoch", type=float, default=25.0)
    parser.add_argument(
        "--output-dir",
        default="results/router_background_suite",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.lowpass_sigma <= 0:
        raise ValueError("--lowpass-sigma must be positive")
    if not 0.0 <= args.body_threshold <= 1.0:
        raise ValueError("--body-threshold must be in [0, 1]")
    if args.lesion_exclusion_radius < 0 or args.hotspot_hit_radius < 0:
        raise ValueError("mask radii must be non-negative")
    if not 0.0 < args.recovery_ratio <= 1.0:
        raise ValueError("--recovery-ratio must be in (0, 1]")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
