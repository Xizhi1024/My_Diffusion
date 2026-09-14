"""No-training B0-B8 action-axis screen for a fixed router checkpoint.

The runner is intentionally fail-closed:

* the checkpoint SHA-256 must match the user-supplied value;
* the validation cohort is deterministic and patient-balanced;
* every intervention reuses the same RNG seed and sample order;
* B0 is repeated byte-for-byte before the screen proceeds;
* no optimizer, backward pass, or checkpoint write exists in this script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_router_background_suite import (  # noqa: E402
    _body_nonlesion_mask,
    compute_body_background_metrics,
)
from scripts.evaluate import (  # noqa: E402
    _seed_evaluation,
    _select_checkpoint_state,
    _to_numpy,
    compute_normalized_lesion_metrics,
    compute_ssim,
    compute_target_relative_false_hotspots,
)
from src.data.lineage import (  # noqa: E402
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile  # noqa: E402
from src.model.frequency.haar import haar_dwt2  # noqa: E402
from src.model.slmf_bbdm import SLMFBBDM  # noqa: E402
from src.model.trainer import _to_unit_interval  # noqa: E402


SCHEMA_VERSION = 1
BAND_NAMES = ("LH", "HL", "HH")
LEVEL_NAMES = ("L2", "L1")
PRIMARY_METRICS = (
    "lesion_topq_peak_error_norm",
    "lesion_peak_error_norm",
    "lesion_roi_l1",
    "background_mae",
    "false_hotspot",
    "ssim",
)
LOWER_IS_BETTER = {
    "lesion_topq_peak_error_norm": True,
    "lesion_peak_error_norm": True,
    "lesion_roi_l1": True,
    "background_mae": True,
    "false_hotspot": True,
    "ssim": False,
}
VARIANTS: dict[str, dict[str, Any]] = {
    "B0": {
        "label": "learned_full",
        "frequency_mode": "full",
    },
    "B1": {
        "label": "frequency_all_off",
        "frequency_mode": "all_frequency_off",
    },
    "B2": {
        "label": "detail_off",
        "frequency_mode": "detail_off",
    },
    "B3": {
        "label": "ll_off",
        "frequency_mode": "ll_off",
    },
    "B4": {
        "label": "all_native",
        "frequency_mode": "full",
        "route_action": "native",
    },
    "B5": {
        "label": "all_shallow",
        "frequency_mode": "full",
        "route_action": "shallow",
    },
    "B6": {
        "label": "detail_gain_0p5",
        "frequency_mode": "full",
        "detail_gain": 0.5,
    },
    "B7": {
        "label": "detail_gain_1p5",
        "frequency_mode": "full",
        "detail_gain": 1.5,
    },
    "B8": {
        "label": "detail_late_only",
        "frequency_mode": "full",
        "logsnr_min": 0.0,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit_sha": commit,
        "worktree_dirty": None if status is None else bool(status),
        "worktree_status_porcelain": status,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(set(seeds)) < 4:
        raise ValueError("--seeds must contain at least four distinct seeds")
    return seeds


def _parse_variants(raw: str) -> list[str]:
    variants = [part.strip().upper() for part in raw.split(",") if part.strip()]
    if not variants:
        raise ValueError("--variants must contain at least B0")
    if len(set(variants)) != len(variants):
        raise ValueError("--variants must not contain duplicates")
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if "B0" not in variants:
        raise ValueError("--variants must include B0 as the paired baseline")
    return variants


def _entry_value(entry: Any, key: str, fallback: Any) -> Any:
    return getattr(entry, key, fallback)


def _slice_sort_key(record: Mapping[str, Any]) -> tuple[int, str]:
    raw = str(record["slice_id"])
    try:
        return int(raw), raw
    except ValueError:
        return 0, raw


def _pick_record(
    available: Sequence[Mapping[str, Any]],
    preference: Sequence[str],
) -> Mapping[str, Any]:
    for category in preference:
        candidates = [
            record for record in available if record["stratum"] == category
        ]
        if not candidates:
            continue
        if category == "small_lesion":
            return min(candidates, key=lambda row: (row["lesion_area"], row["index"]))
        if category == "large_lesion":
            return max(candidates, key=lambda row: (row["lesion_area"], -row["index"]))
        ordered = sorted(candidates, key=_slice_sort_key)
        return ordered[len(ordered) // 2]
    return min(available, key=lambda row: row["index"])


def build_patient_balanced_subset(
    dataset,
    *,
    count: int,
    max_per_patient: int,
    allow_positive_only_cohort: bool = False,
    all_validation: bool = False,
) -> tuple[Subset, list[dict[str, Any]]]:
    """Select deterministic lesion/background slices with a hard patient cap."""
    if count <= 0:
        raise ValueError("count must be positive")
    if max_per_patient <= 0:
        raise ValueError("max_per_patient must be positive")

    records: list[dict[str, Any]] = []
    positive_areas: list[float] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        mask = sample.get("mask")
        area = float(mask.sum().item()) if torch.is_tensor(mask) else 0.0
        if area > 0.0:
            positive_areas.append(area)
        entry = dataset.entries[index]
        records.append(
            {
                "index": index,
                "patient_id": str(_entry_value(entry, "patient_id", "")),
                "sample_id": str(_entry_value(entry, "sample_id", index)),
                "slice_id": str(_entry_value(entry, "slice_id", index)),
                "lesion_area": area,
            }
        )
    if not records:
        raise ValueError("validation dataset is empty")

    small_cut = (
        float(np.quantile(positive_areas, 0.25))
        if positive_areas
        else 0.0
    )
    large_cut = (
        float(np.quantile(positive_areas, 0.75))
        if positive_areas
        else 0.0
    )
    for record in records:
        area = float(record["lesion_area"])
        if area <= 0.0:
            record["stratum"] = "background"
        elif area <= small_cut:
            record["stratum"] = "small_lesion"
        elif area >= large_cut:
            record["stratum"] = "large_lesion"
        else:
            record["stratum"] = "middle_lesion"
    available_strata = {str(row["stratum"]) for row in records}
    background_available = "background" in available_strata
    required_strata = {
        "small_lesion",
        "large_lesion",
    }
    if background_available or not allow_positive_only_cohort:
        required_strata.add("background")
    missing_strata = sorted(required_strata.difference(available_strata))
    if missing_strata:
        raise ValueError(
            "Validation data cannot satisfy required cohort strata: "
            + ", ".join(missing_strata)
        )
    if all_validation:
        selected = [dict(record, selection_round=0) for record in records]
        indices = [int(row["index"]) for row in selected]
        return Subset(dataset, indices), selected

    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(record)
    patients = sorted(by_patient)
    capacity = sum(
        min(max_per_patient, len(by_patient[patient])) for patient in patients
    )
    if capacity < count:
        raise ValueError(
            f"Patient cap permits only {capacity} samples, fewer than {count}"
        )

    rotations = (
        ("small_lesion", "large_lesion", "background", "middle_lesion"),
        ("large_lesion", "background", "small_lesion", "middle_lesion"),
        ("background", "small_lesion", "large_lesion", "middle_lesion"),
    )
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    # Seed the cohort with every required stratum before round-robin filling.
    seed_strata = [
        category
        for category in ("small_lesion", "large_lesion", "background")
        if category in required_strata
    ]
    for category in seed_strata:
        candidates = [
            row for row in records if row["stratum"] == category
        ]
        candidates.sort(
            key=lambda row: (
                sum(
                    selected_row["patient_id"] == row["patient_id"]
                    for selected_row in selected
                ),
                row["patient_id"],
                row["index"],
            )
        )
        chosen = dict(_pick_record(candidates, (category,)))
        chosen["selection_round"] = 0
        selected.append(chosen)
        selected_indices.add(int(chosen["index"]))
    for selection_round in range(max_per_patient):
        for patient_index, patient in enumerate(patients):
            if len(selected) >= count:
                break
            already = sum(
                row["patient_id"] == patient for row in selected
            )
            if already >= selection_round + 1:
                continue
            available = [
                row
                for row in by_patient[patient]
                if int(row["index"]) not in selected_indices
            ]
            if not available:
                continue
            preference = rotations[
                (patient_index + selection_round) % len(rotations)
            ]
            chosen = dict(_pick_record(available, preference))
            chosen["selection_round"] = selection_round + 1
            selected.append(chosen)
            selected_indices.add(int(chosen["index"]))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"Selection produced {len(selected)} samples instead of {count}"
        )
    counts = defaultdict(int)
    for row in selected:
        counts[str(row["patient_id"])] += 1
    if max(counts.values()) > max_per_patient:
        raise AssertionError("patient cap was violated")
    selected_strata = {str(row["stratum"]) for row in selected}
    if not required_strata.issubset(selected_strata):
        raise AssertionError("required cohort strata were not preserved")
    indices = [int(row["index"]) for row in selected]
    return Subset(dataset, indices), selected


def _configure_variant(router, variant_id: str) -> dict[str, Any]:
    spec = dict(VARIANTS[variant_id])
    router.set_inference_destination_intervention("learned")
    router.set_inference_route_action_intervention()
    router.set_inference_frequency_intervention(spec["frequency_mode"])
    router.set_inference_frequency_gain(
        detail_gain=float(spec.get("detail_gain", 1.0)),
        ll_gain=float(spec.get("ll_gain", 1.0)),
    )
    router.set_inference_frequency_window(
        logsnr_min=spec.get("logsnr_min"),
        logsnr_max=spec.get("logsnr_max"),
    )
    action = spec.get("route_action")
    if action is not None:
        actions = (str(action),) * 3
        router.set_inference_route_action_intervention(
            actions_l2=actions,
            actions_l1=actions,
        )
    return {
        "variant_id": variant_id,
        **spec,
        "destination": router.inference_destination_intervention(),
        "route_action": router.inference_route_action_intervention(),
        "frequency": router.inference_frequency_intervention(),
    }


def _metric_row(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    ct: np.ndarray,
    mask: np.ndarray,
    mean_pet: np.ndarray,
    pred_residual: np.ndarray,
    body_threshold: float,
    lesion_exclusion_radius: int,
    lowpass_sigma: float,
) -> dict[str, float]:
    result = {
        metric: float("nan") for metric in PRIMARY_METRICS
    }
    if float(mask.sum()) > 0.0:
        lesion = compute_normalized_lesion_metrics(
            prediction,
            target,
            mask,
            np.zeros((6, *target.shape[1:]), dtype=np.float32),
        )
        for key in (
            "lesion_topq_peak_error_norm",
            "lesion_peak_error_norm",
            "lesion_roi_l1",
        ):
            result[key] = float(lesion[key])
    background = compute_body_background_metrics(
        prediction,
        target,
        mean_pet,
        pred_residual,
        ct,
        mask,
        lowpass_sigma=lowpass_sigma,
        body_threshold=body_threshold,
        lesion_exclusion_radius=lesion_exclusion_radius,
    )
    result["background_mae"] = float(background["body_nonlesion_mae"])
    hotspots = compute_target_relative_false_hotspots(
        prediction,
        target,
        mask,
    )
    result["false_hotspot"] = float(
        hotspots["target_relative_false_hotspot_density"]
    )
    result["ssim"] = float(compute_ssim(prediction[0], target[0]))
    return result


def _resized_mask(
    mask: np.ndarray,
    *,
    size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None].to(device)
    return F.interpolate(tensor, size=size, mode="nearest")[0, 0] > 0.5


@torch.no_grad()
def evaluate_run(
    *,
    model: SLMFBBDM,
    loader: DataLoader,
    cohort: Sequence[Mapping[str, Any]],
    device: torch.device,
    seed: int,
    steps: int,
    amp: bool,
    body_threshold: float,
    lesion_exclusion_radius: int,
    lowpass_sigma: float,
    return_frequency_trace: bool = True,
) -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    _seed_evaluation(seed)
    model.eval()
    metrics: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    static: dict[str, list[np.ndarray]] = defaultdict(list)
    q_rows: list[dict[str, Any]] = []
    offset = 0
    device_type = device.type
    amp_dtype = (
        torch.bfloat16
        if device_type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )
    for batch in loader:
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
            output = model.sample(
                batch_gpu,
                num_steps=steps,
                progress=False,
                return_frequency_trace=return_frequency_trace,
            )
        prediction = output["synthetic_pet"]
        mean_pet = output["mean_pet"]
        pred_residual = output["pred_residual"]
        target = batch_gpu["pet"]
        ct = batch_gpu["ct"]
        mask = batch_gpu.get("mask", torch.zeros_like(target))
        batch_size = int(prediction.shape[0])
        current_cohort = cohort[offset : offset + batch_size]
        for index, cohort_row in enumerate(current_cohort):
            pred_np = _to_numpy(prediction[index]).astype(np.float32)
            target_np = _to_numpy(target[index]).astype(np.float32)
            ct_np = _to_numpy(ct[index]).astype(np.float32)
            mask_np = _to_numpy(mask[index]).astype(np.float32)
            mean_np = _to_numpy(mean_pet[index]).astype(np.float32)
            residual_np = _to_numpy(pred_residual[index]).astype(np.float32)
            row: dict[str, Any] = {
                "sample_id": str(cohort_row["sample_id"]),
                "patient_id": str(cohort_row["patient_id"]),
                "slice_id": str(cohort_row["slice_id"]),
                "stratum": str(cohort_row["stratum"]),
            }
            row.update(
                _metric_row(
                    prediction=pred_np,
                    target=target_np,
                    ct=ct_np,
                    mask=mask_np,
                    mean_pet=mean_np,
                    pred_residual=residual_np,
                    body_threshold=body_threshold,
                    lesion_exclusion_radius=lesion_exclusion_radius,
                    lowpass_sigma=lowpass_sigma,
                )
            )
            metrics.append(row)
            predictions.append(pred_np)
            static["target"].append(target_np)
            static["ct"].append(ct_np)
            static["mask"].append(mask_np)
            static["mean_pet"].append(mean_np)

        for trace in output.get("frequency_trace", []):
            q = trace.get("route_conditional_shallow")
            active = trace.get("route_active")
            logsnr = trace.get("inference_frequency_bridge_logsnr")
            spatial = (
                trace.get("route_spatial_conditional_shallow_l2"),
                trace.get("route_spatial_conditional_shallow_l1"),
            )
            if not torch.is_tensor(q):
                continue
            for index, cohort_row in enumerate(current_cohort):
                mask_np = _to_numpy(mask[index])[0] > 0.5
                ct_np = _to_numpy(ct[index])
                _, background_np = _body_nonlesion_mask(
                    ct_np,
                    _to_numpy(mask[index]),
                    body_threshold=body_threshold,
                    exclusion_radius=lesion_exclusion_radius,
                )
                for level_index, level_name in enumerate(LEVEL_NAMES):
                    spatial_level = spatial[level_index]
                    lesion_small = background_small = None
                    if torch.is_tensor(spatial_level):
                        size = tuple(int(v) for v in spatial_level.shape[-2:])
                        lesion_small = _resized_mask(
                            mask_np,
                            size=size,
                            device=torch.device("cpu"),
                        )
                        background_small = _resized_mask(
                            background_np,
                            size=size,
                            device=torch.device("cpu"),
                        )
                    for band_index, band_name in enumerate(BAND_NAMES):
                        record: dict[str, Any] = {
                            "sample_id": str(cohort_row["sample_id"]),
                            "patient_id": str(cohort_row["patient_id"]),
                            "slice_id": str(cohort_row["slice_id"]),
                            "stratum": str(cohort_row["stratum"]),
                            "timestep": int(trace["timestep"]),
                            "level": level_name,
                            "band": band_name,
                            "q_conditional_shallow": float(
                                q[index, level_index, band_index].item()
                            ),
                        }
                        if torch.is_tensor(active):
                            record["route_active"] = float(
                                active[index, level_index, band_index].item()
                            )
                        if torch.is_tensor(logsnr):
                            record["bridge_logsnr"] = float(
                                logsnr[index].item()
                            )
                        if torch.is_tensor(spatial_level):
                            spatial_map = spatial_level[
                                index, band_index
                            ].float()
                            if lesion_small is not None and lesion_small.any():
                                record["q_lesion"] = float(
                                    spatial_map[lesion_small].mean().item()
                                )
                            else:
                                record["q_lesion"] = None
                            if (
                                background_small is not None
                                and background_small.any()
                            ):
                                record["q_background"] = float(
                                    spatial_map[background_small].mean().item()
                                )
                            else:
                                record["q_background"] = None
                        q_rows.append(record)
        offset += batch_size
    static_arrays = {
        key: np.stack(values).astype(np.float32)
        for key, values in static.items()
    }
    return (
        metrics,
        np.stack(predictions).astype(np.float32),
        static_arrays,
        q_rows,
    )


def _region_effect_rows(
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    static: Mapping[str, np.ndarray],
    cohort: Sequence[Mapping[str, Any]],
    seed: int,
    variant_id: str,
    body_threshold: float,
    lesion_exclusion_radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, cohort_row in enumerate(cohort):
        difference = np.abs(
            _to_unit_interval(candidate[index])[0]
            - _to_unit_interval(baseline[index])[0]
        )
        lesion = static["mask"][index, 0] > 0.5
        _, background = _body_nonlesion_mask(
            static["ct"][index],
            static["mask"][index],
            body_threshold=body_threshold,
            exclusion_radius=lesion_exclusion_radius,
        )
        rows.append(
            {
                "variant_id": variant_id,
                "seed": seed,
                "sample_id": cohort_row["sample_id"],
                "patient_id": cohort_row["patient_id"],
                "slice_id": cohort_row["slice_id"],
                "lesion_output_l1_unit": (
                    float(difference[lesion].mean())
                    if lesion.any()
                    else None
                ),
                "background_output_l1_unit": (
                    float(difference[background].mean())
                    if background.any()
                    else None
                ),
                "global_output_l1_unit": float(difference.mean()),
            }
        )
    return rows


def _sampling_variation_rows(
    *,
    baseline_by_seed: Mapping[int, np.ndarray],
    static: Mapping[str, np.ndarray],
    cohort: Sequence[Mapping[str, Any]],
    body_threshold: float,
    lesion_exclusion_radius: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for first_seed, second_seed in itertools.combinations(
        sorted(baseline_by_seed),
        2,
    ):
        rows.extend(
            _region_effect_rows(
                candidate=baseline_by_seed[first_seed],
                baseline=baseline_by_seed[second_seed],
                static=static,
                cohort=cohort,
                seed=first_seed,
                variant_id=f"sampling_{first_seed}_vs_{second_seed}",
                body_threshold=body_threshold,
                lesion_exclusion_radius=lesion_exclusion_radius,
            )
        )
        for row in rows[-len(cohort) :]:
            row["seed_1"] = first_seed
            row["seed_2"] = second_seed
    return rows


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _metric_rows_equal(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> bool:
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right):
        if left_row.keys() != right_row.keys():
            return False
        for key in left_row:
            left_value = left_row[key]
            right_value = right_row[key]
            if isinstance(left_value, (int, float)) and isinstance(
                right_value,
                (int, float),
            ):
                if math.isnan(float(left_value)) and math.isnan(
                    float(right_value)
                ):
                    continue
                if float(left_value) != float(right_value):
                    return False
            elif left_value != right_value:
                return False
    return True


def _paired_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    by_key = {
        (str(row["variant_id"]), int(row["seed"]), str(row["sample_id"])): row
        for row in rows
    }
    variants = sorted(
        {str(row["variant_id"]) for row in rows if row["variant_id"] != "B0"}
    )
    rng = np.random.default_rng(bootstrap_seed)
    results: list[dict[str, Any]] = []
    for variant in variants:
        for metric in PRIMARY_METRICS:
            lower = LOWER_IS_BETTER[metric]
            patient_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
            baseline_values: list[float] = []
            for seed in seeds:
                baseline_rows = [
                    row
                    for row in rows
                    if row["variant_id"] == "B0" and int(row["seed"]) == seed
                ]
                for baseline in baseline_rows:
                    candidate = by_key.get(
                        (variant, seed, str(baseline["sample_id"]))
                    )
                    if candidate is None:
                        continue
                    base_value = _finite(baseline.get(metric))
                    candidate_value = _finite(candidate.get(metric))
                    if base_value is None or candidate_value is None:
                        continue
                    effect = (
                        base_value - candidate_value
                        if lower
                        else candidate_value - base_value
                    )
                    patient_seed[(str(baseline["patient_id"]), seed)].append(
                        effect
                    )
                    baseline_values.append(base_value)
            patients = sorted({patient for patient, _ in patient_seed})
            if not patients:
                continue
            seed_effects = []
            patient_effects = []
            for seed in seeds:
                current = [
                    float(np.mean(patient_seed[(patient, seed)]))
                    for patient in patients
                    if (patient, seed) in patient_seed
                ]
                if current:
                    seed_effects.append(float(np.mean(current)))
            for patient in patients:
                current = [
                    float(np.mean(patient_seed[(patient, seed)]))
                    for seed in seeds
                    if (patient, seed) in patient_seed
                ]
                if current:
                    patient_effects.append(float(np.mean(current)))
            patient_array = np.asarray(patient_effects, dtype=np.float64)
            indices = rng.integers(
                0,
                patient_array.size,
                size=(bootstrap_replicates, patient_array.size),
            )
            bootstrap = patient_array[indices].mean(axis=1)
            mean_effect = float(patient_array.mean())
            baseline_mean = float(np.mean(baseline_values))
            mcse = (
                float(np.std(seed_effects, ddof=1) / math.sqrt(len(seed_effects)))
                if len(seed_effects) > 1
                else float("nan")
            )
            results.append(
                {
                    "variant_id": variant,
                    "metric": metric,
                    "patients": len(patient_effects),
                    "seeds": len(seed_effects),
                    "baseline_mean": baseline_mean,
                    "mean_effect": mean_effect,
                    "relative_improvement_percent": (
                        100.0 * mean_effect / max(abs(baseline_mean), 1e-12)
                    ),
                    "monte_carlo_se": mcse,
                    "effect_gt_2_mcse": (
                        bool(mean_effect > 2.0 * mcse)
                        if math.isfinite(mcse)
                        else False
                    ),
                    "patient_bootstrap_ci95_low": float(
                        np.quantile(bootstrap, 0.025)
                    ),
                    "patient_bootstrap_ci95_high": float(
                        np.quantile(bootstrap, 0.975)
                    ),
                    "effect_definition": "positive means variant better than B0",
                }
            )
    return results


def _medoid(stack: np.ndarray) -> np.ndarray:
    """Choose each sample's full-image L1 medoid using generated images only."""
    seeds, samples = stack.shape[:2]
    result = np.empty_like(stack[0])
    for sample_index in range(samples):
        candidates = stack[:, sample_index]
        flattened = candidates.reshape(seeds, -1)
        distances = np.abs(
            flattened[:, None, :] - flattened[None, :, :]
        ).mean(axis=2)
        index = int(np.argmin(distances.sum(axis=1)))
        result[sample_index] = candidates[index]
    return result


def _aggregation_rows(
    *,
    baseline_by_seed: Mapping[int, np.ndarray],
    static: Mapping[str, np.ndarray],
    cohort: Sequence[Mapping[str, Any]],
    body_threshold: float,
    lesion_exclusion_radius: int,
    lowpass_sigma: float,
) -> list[dict[str, Any]]:
    ordered_seeds = sorted(baseline_by_seed)
    stack = np.stack([baseline_by_seed[seed] for seed in ordered_seeds])
    aggregates = {
        "pixel_mean": stack.mean(axis=0),
        "pixel_median": np.median(stack, axis=0),
        "full_image_medoid": _medoid(stack),
    }
    rows: list[dict[str, Any]] = []
    for method, predictions in aggregates.items():
        for index, cohort_row in enumerate(cohort):
            row: dict[str, Any] = {
                "aggregation": method,
                "sample_id": cohort_row["sample_id"],
                "patient_id": cohort_row["patient_id"],
                "slice_id": cohort_row["slice_id"],
            }
            row.update(
                _metric_row(
                    prediction=predictions[index],
                    target=static["target"][index],
                    ct=static["ct"][index],
                    mask=static["mask"][index],
                    mean_pet=static["mean_pet"][index],
                    pred_residual=(
                        predictions[index] - static["mean_pet"][index]
                    ),
                    body_threshold=body_threshold,
                    lesion_exclusion_radius=lesion_exclusion_radius,
                    lowpass_sigma=lowpass_sigma,
                )
            )
            rows.append(row)
    return rows


def _band_tensors(image: torch.Tensor) -> dict[tuple[str, str], torch.Tensor]:
    ll1, details1 = haar_dwt2(image)
    _, details2 = haar_dwt2(ll1)
    result = {}
    for band_index, band in enumerate(BAND_NAMES):
        result[("L2", band)] = details2[band_index]
        result[("L1", band)] = details1[band_index]
    return result


@torch.no_grad()
def _snr_rows(
    *,
    residual: np.ndarray,
    cohort: Sequence[Mapping[str, Any]],
    schedule,
    steps: int,
    noise_replicates: int,
    seed: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    residual_tensor = torch.from_numpy(residual).to(device)
    clean_bands = _band_tensors(residual_tensor)
    signal_energy = {
        key: value.float().square().flatten(1).mean(dim=1)
        for key, value in clean_bands.items()
    }
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    noise_energy = {key: 0.0 for key in clean_bands}
    for _ in range(noise_replicates):
        noise = torch.randn(
            residual_tensor.shape,
            generator=generator,
            device=device,
            dtype=residual_tensor.dtype,
        )
        for key, value in _band_tensors(noise).items():
            noise_energy[key] += float(value.float().square().mean().item())
    noise_energy = {
        key: value / noise_replicates for key, value in noise_energy.items()
    }

    timesteps = torch.linspace(
        schedule.num_train_timesteps - 1,
        0,
        steps,
        device=device,
        dtype=torch.long,
    )
    sample_rows: list[dict[str, Any]] = []
    for timestep in timesteps.tolist():
        t = int(timestep)
        m = float(schedule.m_t[t].item())
        sigma = float(schedule.sigma_t[t].item())
        for (level, band), energies in signal_energy.items():
            snr = (
                (1.0 - m) ** 2
                * energies
                / max(sigma**2 * noise_energy[(level, band)], 1e-12)
            )
            for index, cohort_row in enumerate(cohort):
                sample_rows.append(
                    {
                        "patient_id": cohort_row["patient_id"],
                        "sample_id": cohort_row["sample_id"],
                        "slice_id": cohort_row["slice_id"],
                        "stratum": cohort_row["stratum"],
                        "timestep": t,
                        "level": level,
                        "band": band,
                        "snr": float(snr[index].item()),
                        "m_t": m,
                        "sigma_t": sigma,
                        "noise_band_energy_expectation": noise_energy[
                            (level, band)
                        ],
                    }
                )
    grouped: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    for row in sample_rows:
        grouped[
            (
                str(row["patient_id"]),
                int(row["timestep"]),
                str(row["level"]),
                str(row["band"]),
            )
        ].append(float(row["snr"]))
    patient_rows = [
        {
            "patient_id": patient,
            "timestep": timestep,
            "level": level,
            "band": band,
            "snr_mean": float(np.mean(values)),
            "snr_median": float(np.median(values)),
            "slices": len(values),
        }
        for (patient, timestep, level, band), values in sorted(grouped.items())
    ]
    return sample_rows, patient_rows


def _main(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_hash = _sha256(checkpoint_path)
    expected_hash = args.expected_checkpoint_sha256.lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            "Checkpoint SHA-256 mismatch: "
            f"expected {expected_hash}, observed {actual_hash}"
        )
    if output.exists() and any(output.iterdir()) and not args.resume_empty:
        raise RuntimeError(
            f"Output directory is not empty: {output}. "
            "Use a fresh path; diagnostic results are never overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)

    seeds = _parse_seeds(args.seeds)
    selected_variants = _parse_variants(args.variants)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    _seed_evaluation(seeds[0])

    config = resolve_runtime_profile(load_full_config(str(config_path)))
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
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
        context="B0-B8 action-axis checkpoint",
    )
    state, state_source = _select_checkpoint_state(
        checkpoint,
        weights=args.weights,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    router = model.residual_preconditioner
    required_methods = (
        "set_inference_frequency_gain",
        "set_inference_frequency_window",
        "set_inference_frequency_intervention",
        "set_inference_route_action_intervention",
    )
    missing = [name for name in required_methods if not hasattr(router, name)]
    if missing:
        raise RuntimeError(f"Diagnostic router API missing: {missing}")

    from src.data.dataset import CachedDataset

    data = config["data"]
    dataset = CachedDataset(
        data["cache_dir"],
        split=args.split,
        augment=False,
        split_manifest=Path(data["split_manifest"]),
        required_keys=list(data.get("required_keys", ("ct", "pet", "mask"))),
    )
    subset, cohort = build_patient_balanced_subset(
        dataset,
        count=args.samples,
        max_per_patient=args.max_per_patient,
        allow_positive_only_cohort=args.allow_positive_only_cohort,
        all_validation=args.all_validation,
    )
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    _write_csv(output / "cohort.csv", cohort)
    stratum_counts: dict[str, int] = defaultdict(int)
    for row in cohort:
        stratum_counts[str(row["stratum"])] += 1
    has_background_slices = stratum_counts.get("background", 0) > 0
    cohort_audit = {
        "selection_mode": (
            "all_validation" if args.all_validation else "patient_balanced"
        ),
        "requested_samples": (
            "all_validation" if args.all_validation else args.samples
        ),
        "selected_samples": len(cohort),
        "patients": len({row["patient_id"] for row in cohort}),
        "max_slices_per_patient": (
            None if args.all_validation else args.max_per_patient
        ),
        "stratum_counts": dict(sorted(stratum_counts.items())),
        "negative_background_slices_available": has_background_slices,
        "positive_only_cohort_override_enabled": (
            args.allow_positive_only_cohort
        ),
        "within_slice_nonlesion_background_metrics_available": True,
        "within_slice_background_definition": (
            "body mask minus dilated lesion mask"
        ),
        "lesion_slice_vs_background_slice_q_comparison": (
            "AVAILABLE"
            if has_background_slices
            else "UNAVAILABLE_NO_NEGATIVE_VALIDATION_SLICES"
        ),
    }
    _write_json(output / "cohort_audit.json", cohort_audit)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "fixed_checkpoint_no_training_full_validation_confirmation"
            if args.all_validation and selected_variants == ["B0"]
            else "fixed_checkpoint_no_training_action_axis_screen"
        ),
        "git": _git_state(ROOT),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": actual_hash,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_weights": state_source,
        "split": args.split,
        "samples": len(cohort),
        "all_validation": args.all_validation,
        "max_per_patient": (
            None if args.all_validation else args.max_per_patient
        ),
        "patients": len({row["patient_id"] for row in cohort}),
        "cohort_audit": cohort_audit,
        "data_limitations": (
            []
            if has_background_slices
            else [
                "The validation cache contains no negative slices. "
                "Background MAE, false-hotspot, and output-effect background "
                "terms use within-slice nonlesion body pixels; lesion-slice "
                "versus background-slice q comparison is not identifiable."
            ]
        ),
        "seeds": seeds,
        "ddim_steps": args.steps,
        "variants": {
            variant: VARIANTS[variant] for variant in selected_variants
        },
        "q_trace_enabled": not args.skip_q_trace,
        "band_snr_audit_enabled": not args.skip_band_snr,
        "device": str(device),
        "amp": not args.no_amp,
        "training_or_optimizer_used": False,
    }
    _write_json(output / "run_manifest.json", _json_safe(manifest))

    # Exact reproducibility gate: run B0 twice before any screen result is used.
    print(
        f"[repro 1/2] B0 seed={seeds[0]} samples={len(cohort)}",
        flush=True,
    )
    _configure_variant(router, "B0")
    first_metrics, first_pred, first_static, first_q = evaluate_run(
        model=model,
        loader=loader,
        cohort=cohort,
        device=device,
        seed=seeds[0],
        steps=args.steps,
        amp=not args.no_amp,
        body_threshold=args.body_threshold,
        lesion_exclusion_radius=args.lesion_exclusion_radius,
        lowpass_sigma=args.lowpass_sigma,
        return_frequency_trace=not args.skip_q_trace,
    )
    print(f"[repro 2/2] B0 seed={seeds[0]}", flush=True)
    second_metrics, second_pred, _, _ = evaluate_run(
        model=model,
        loader=loader,
        cohort=cohort,
        device=device,
        seed=seeds[0],
        steps=args.steps,
        amp=not args.no_amp,
        body_threshold=args.body_threshold,
        lesion_exclusion_radius=args.lesion_exclusion_radius,
        lowpass_sigma=args.lowpass_sigma,
        return_frequency_trace=not args.skip_q_trace,
    )
    exact = bool(np.array_equal(first_pred, second_pred))
    metrics_equal = _metric_rows_equal(first_metrics, second_metrics)
    reproduction = {
        "seed": seeds[0],
        "same_config_and_sample_order": True,
        "array_equal": exact,
        "max_absolute_difference": float(
            np.max(np.abs(first_pred - second_pred))
        ),
        "metric_rows_equal": metrics_equal,
    }
    _write_json(output / "reproducibility.json", reproduction)
    if not exact or not metrics_equal:
        raise RuntimeError(
            "B0 exact reproducibility gate failed; B0-B8 was not continued"
        )
    print("[repro PASS] byte-identical B0 outputs and metrics", flush=True)

    all_metric_rows: list[dict[str, Any]] = []
    all_q_rows: list[dict[str, Any]] = []
    action_effect_rows: list[dict[str, Any]] = []
    baseline_by_seed: dict[int, np.ndarray] = {}
    static = first_static
    total_runs = len(seeds) * len(selected_variants)
    completed_runs = 0
    for seed in seeds:
        baseline_pred: np.ndarray | None = None
        for variant_id in selected_variants:
            completed_runs += 1
            print(
                f"[screen {completed_runs}/{total_runs}] "
                f"{variant_id} seed={seed}",
                flush=True,
            )
            intervention = _configure_variant(router, variant_id)
            if variant_id == "B0" and seed == seeds[0]:
                metrics = first_metrics
                predictions = first_pred
                q_rows = first_q
            else:
                metrics, predictions, current_static, q_rows = evaluate_run(
                    model=model,
                    loader=loader,
                    cohort=cohort,
                    device=device,
                    seed=seed,
                    steps=args.steps,
                    amp=not args.no_amp,
                    body_threshold=args.body_threshold,
                    lesion_exclusion_radius=args.lesion_exclusion_radius,
                    lowpass_sigma=args.lowpass_sigma,
                    return_frequency_trace=not args.skip_q_trace,
                )
                for key in static:
                    if not np.array_equal(static[key], current_static[key]):
                        raise RuntimeError(
                            f"Static cohort tensor changed across runs: {key}"
                        )
            for row in metrics:
                row.update(
                    {
                        "variant_id": variant_id,
                        "variant_label": VARIANTS[variant_id]["label"],
                        "seed": seed,
                    }
                )
            for row in q_rows:
                row.update(
                    {
                        "variant_id": variant_id,
                        "variant_label": VARIANTS[variant_id]["label"],
                        "seed": seed,
                    }
                )
            all_metric_rows.extend(metrics)
            all_q_rows.extend(q_rows)
            prediction_path = (
                output
                / "predictions"
                / f"{variant_id}_seed{seed}.npz"
            )
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                prediction_path,
                predictions=predictions,
                sample_ids=np.asarray(
                    [row["sample_id"] for row in cohort],
                    dtype=str,
                ),
            )
            if variant_id == "B0":
                baseline_pred = predictions
                baseline_by_seed[seed] = predictions
                if seed == seeds[0]:
                    np.savez_compressed(
                        output / "cohort_tensors.npz",
                        **static,
                        sample_ids=np.asarray(
                            [row["sample_id"] for row in cohort],
                            dtype=str,
                        ),
                    )
            else:
                assert baseline_pred is not None
                action_effect_rows.extend(
                    _region_effect_rows(
                        candidate=predictions,
                        baseline=baseline_pred,
                        static=static,
                        cohort=cohort,
                        seed=seed,
                        variant_id=variant_id,
                        body_threshold=args.body_threshold,
                        lesion_exclusion_radius=args.lesion_exclusion_radius,
                    )
                )
            _write_json(
                output
                / "interventions"
                / f"{variant_id}_seed{seed}.json",
                _json_safe(intervention),
            )

    print("[analysis] action effects, sampling variation, aggregation", flush=True)
    sampling_rows = _sampling_variation_rows(
        baseline_by_seed=baseline_by_seed,
        static=static,
        cohort=cohort,
        body_threshold=args.body_threshold,
        lesion_exclusion_radius=args.lesion_exclusion_radius,
    )
    paired = _paired_summary(
        all_metric_rows,
        seeds=seeds,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.analysis_seed,
    )
    aggregation = _aggregation_rows(
        baseline_by_seed=baseline_by_seed,
        static=static,
        cohort=cohort,
        body_threshold=args.body_threshold,
        lesion_exclusion_radius=args.lesion_exclusion_radius,
        lowpass_sigma=args.lowpass_sigma,
    )
    if args.skip_band_snr:
        print("[analysis] band SNR audit skipped by request", flush=True)
        snr_sample: list[dict[str, Any]] = []
        snr_patient: list[dict[str, Any]] = []
    else:
        print("[analysis] band SNR audit", flush=True)
        snr_sample, snr_patient = _snr_rows(
            residual=static["target"] - static["mean_pet"],
            cohort=cohort,
            schedule=model.noise_schedule,
            steps=args.steps,
            noise_replicates=args.noise_replicates,
            seed=args.analysis_seed,
            device=device,
        )

    _write_csv(output / "metrics.csv", all_metric_rows)
    _write_csv(output / "q_trace.csv", all_q_rows)
    _write_csv(output / "action_output_effects.csv", action_effect_rows)
    _write_csv(output / "sampling_variation.csv", sampling_rows)
    _write_csv(output / "paired_summary.csv", paired)
    _write_csv(output / "aggregation_metrics.csv", aggregation)
    _write_csv(output / "band_snr_sample.csv", snr_sample)
    _write_csv(output / "band_snr_patient.csv", snr_patient)
    completion = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "seconds": time.perf_counter() - started,
        "output": str(output),
        "reproducibility": reproduction,
        "metric_rows": len(all_metric_rows),
        "q_rows": len(all_q_rows),
        "training_or_optimizer_used": False,
    }
    _write_json(output / "COMPLETE.json", _json_safe(completion))
    print(json.dumps(completion, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variants to run; B0 is always required.",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--max-per-patient", type=int, default=3)
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help=(
            "Use every sample in the validation split and disable the "
            "per-patient slice cap. Patient-level analysis must still weight "
            "patients equally."
        ),
    )
    parser.add_argument(
        "--allow-positive-only-cohort",
        action="store_true",
        help=(
            "Permit a validation cohort with no negative slices. Small and "
            "large lesion strata remain mandatory; background metrics use "
            "within-slice nonlesion body pixels and the missing slice-level "
            "q comparison is recorded as unavailable."
        ),
    )
    parser.add_argument("--seeds", default="42,43,44,45")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--body-threshold", type=float, default=0.03)
    parser.add_argument("--lesion-exclusion-radius", type=int, default=8)
    parser.add_argument("--lowpass-sigma", type=float, default=4.0)
    parser.add_argument(
        "--skip-q-trace",
        action="store_true",
        help="Do not collect the large per-timestep router trace.",
    )
    parser.add_argument(
        "--skip-band-snr",
        action="store_true",
        help="Do not repeat the residual-band SNR audit.",
    )
    parser.add_argument("--noise-replicates", type=int, default=16)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--analysis-seed", type=int, default=20260728)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-empty",
        action="store_true",
        help="Allow an existing but empty output directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.steps <= 0 or args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("steps, samples, and batch-size must be positive")
    if args.noise_replicates <= 0 or args.bootstrap_replicates <= 0:
        raise ValueError(
            "noise-replicates and bootstrap-replicates must be positive"
        )
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
