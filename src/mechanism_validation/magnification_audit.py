"""Application layer for the frozen lesion-magnification mechanism audit.

The geometry and statistical estimators live in :mod:`magnification`.  This
module composes them into paired inference tables and a predeclared gate while
remaining independent of checkpoint and configuration loading.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from scripts.evaluate import (
    compute_normalized_lesion_metrics,
    compute_target_relative_false_hotspots,
)
from src.mechanism_validation.magnification import (
    PatientBootstrapResult,
    backproject_crop,
    bootstrap_patient_balanced_slope,
    couple_noise_to_crop,
    crop_resize,
    lesion_crop_box,
)


@dataclass(frozen=True)
class CohortArrays:
    """Identity table and aligned model-space arrays for one frozen cohort."""

    cohort: pd.DataFrame
    ct: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    sample_ids: np.ndarray


@dataclass(frozen=True)
class AuditTables:
    """Machine-readable tables emitted by a magnification audit."""

    sample_metrics: pd.DataFrame
    patient_metrics: pd.DataFrame
    roundtrip_metrics: pd.DataFrame


def validate_cohort(data: CohortArrays) -> None:
    """Validate shape, identity, finiteness, and lesion-area alignment."""

    required = {"patient_id", "sample_id", "slice_id", "lesion_area"}
    missing = required.difference(data.cohort.columns)
    if missing:
        raise ValueError(f"cohort is missing required columns: {sorted(missing)}")

    arrays = {
        "ct": np.asarray(data.ct),
        "target": np.asarray(data.target),
        "mask": np.asarray(data.mask),
    }
    shapes = {name: value.shape for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"ct, target, and mask shape mismatch: {shapes}")
    shape = arrays["ct"].shape
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(
            "cohort arrays must have shape [N,1,H,W], "
            f"received {shape}"
        )

    sample_ids = np.asarray(data.sample_ids).astype(str).reshape(-1)
    if len(data.cohort) != shape[0] or len(sample_ids) != shape[0]:
        raise ValueError(
            "cohort, sample_ids, and array shape disagree: "
            f"{len(data.cohort)}, {len(sample_ids)}, {shape[0]}"
        )
    table_ids = data.cohort["sample_id"].astype(str).to_numpy()
    if not np.array_equal(table_ids, sample_ids):
        raise ValueError("cohort sample_ids do not align with tensor sample_ids")
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise ValueError("sample_ids must be unique")

    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    lesion_areas = (arrays["mask"] > 0.5).sum(axis=(1, 2, 3)).astype(float)
    if bool((lesion_areas <= 0).any()):
        raise ValueError("every audit sample must contain a non-empty lesion")
    declared_areas = data.cohort["lesion_area"].to_numpy(dtype=float)
    if not np.allclose(lesion_areas, declared_areas, rtol=0.0, atol=1.0e-6):
        raise ValueError("cohort lesion_area does not match mask pixel counts")


def _normal_noise(
    reference: torch.Tensor,
    *,
    seed: int,
    sample_index: int,
) -> torch.Tensor:
    """Draw batch-size-independent noise for one sample and trajectory seed."""

    generator = torch.Generator(device=reference.device)
    derived_seed = (int(seed) + 1_000_003 * int(sample_index)) % (2**63 - 1)
    generator.manual_seed(derived_seed)
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _lesion_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    pred_np = pred.detach().cpu().float().numpy()
    target_np = target.detach().cpu().float().numpy()
    mask_np = mask.detach().cpu().float().numpy()
    height, width = pred_np.shape[-2:]
    return compute_normalized_lesion_metrics(
        pred_np,
        target_np,
        mask_np,
        np.zeros((0, height, width), dtype=np.float32),
    )


def _context_hotspot_density(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    metrics = compute_target_relative_false_hotspots(
        pred.detach().cpu().float().numpy(),
        target.detach().cpu().float().numpy(),
        mask.detach().cpu().float().numpy(),
    )
    return float(metrics["target_relative_false_hotspot_density"])


def _relative_reduction(full_error: float, zoom_error: float) -> float:
    denominator = max(float(full_error), 1.0e-6)
    return float((float(full_error) - float(zoom_error)) / denominator)


def _sample_model(
    model: Any,
    ct: torch.Tensor,
    noise: torch.Tensor,
    *,
    num_steps: int,
    amp: bool,
) -> torch.Tensor:
    with torch.autocast(
        device_type=ct.device.type,
        enabled=amp,
    ):
        result = model.sample(
            {"ct": ct},
            num_steps=num_steps,
            initial_noise=noise,
        )
    if not isinstance(result, Mapping) or "synthetic_pet" not in result:
        raise TypeError("model.sample must return a mapping with 'synthetic_pet'")
    prediction = result["synthetic_pet"]
    if not torch.is_tensor(prediction) or prediction.shape != ct.shape:
        raise ValueError(
            "synthetic_pet shape must match CT shape: "
            f"{getattr(prediction, 'shape', None)} != {ct.shape}"
        )
    return prediction


def _patient_table(sample_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in sample_metrics.columns
        if (
            "error" in column
            or "improvement" in column
            or "hotspot" in column
        )
    ]
    rows: list[dict[str, Any]] = []
    for (patient_id, crop_size), group in sample_metrics.groupby(
        ["patient_id", "crop_size"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "patient_id": str(patient_id),
            "crop_size": int(crop_size),
            "sample_count": int(group["sample_id"].nunique()),
            "seed_count": int(group["seed"].nunique()),
        }
        for column in metric_columns:
            row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def run_frozen_magnification_audit(
    *,
    model: Any,
    cohort: CohortArrays,
    device: torch.device,
    seeds: Sequence[int],
    crop_sizes: Sequence[int],
    input_size: int,
    num_steps: int,
    batch_size: int,
    amp: bool = False,
) -> AuditTables:
    """Run paired full/zoom inference and interpolation-only controls."""

    validate_cohort(cohort)
    if input_size <= 0 or num_steps <= 0 or batch_size <= 0:
        raise ValueError("input_size, num_steps, and batch_size must be positive")
    if cohort.ct.shape[-2:] != (input_size, input_size):
        raise ValueError(
            "input_size does not match cohort arrays: "
            f"{input_size} != {cohort.ct.shape[-2:]}"
        )
    normalized_seeds = tuple(int(seed) for seed in seeds)
    normalized_crops = tuple(int(size) for size in crop_sizes)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    if not normalized_crops or len(set(normalized_crops)) != len(normalized_crops):
        raise ValueError("crop_sizes must be a non-empty unique sequence")
    if any(size <= 0 or size > input_size for size in normalized_crops):
        raise ValueError("every crop_size must be in [1, input_size]")

    if hasattr(model, "eval"):
        model.eval()

    sample_rows: list[dict[str, Any]] = []
    roundtrip_rows: list[dict[str, Any]] = []
    height_width = (input_size, input_size)

    with torch.inference_mode():
        for start in range(0, len(cohort.cohort), batch_size):
            stop = min(start + batch_size, len(cohort.cohort))
            ct = torch.as_tensor(
                cohort.ct[start:stop],
                dtype=torch.float32,
                device=device,
            )
            target = torch.as_tensor(
                cohort.target[start:stop],
                dtype=torch.float32,
                device=device,
            )
            mask = torch.as_tensor(
                cohort.mask[start:stop],
                dtype=torch.float32,
                device=device,
            )
            boxes_by_crop = {
                crop_size: [
                    lesion_crop_box(mask[index], crop_size)
                    for index in range(stop - start)
                ]
                for crop_size in normalized_crops
            }

            for crop_size, boxes in boxes_by_crop.items():
                for local_index, box in enumerate(boxes):
                    target_zoom = crop_resize(
                        target[local_index],
                        box,
                        output_size=input_size,
                        mode="bilinear",
                    )
                    roundtrip = backproject_crop(
                        target_zoom,
                        box,
                        canvas_size=height_width,
                        mode="bilinear",
                        base=target[local_index],
                    )
                    metrics = _lesion_metrics(
                        roundtrip,
                        target[local_index],
                        mask[local_index],
                    )
                    identity = cohort.cohort.iloc[start + local_index]
                    roundtrip_rows.append(
                        {
                            "patient_id": str(identity["patient_id"]),
                            "sample_id": str(identity["sample_id"]),
                            "slice_id": identity["slice_id"],
                            "crop_size": crop_size,
                            "magnification_factor": input_size / crop_size,
                            "lesion_area": float(identity["lesion_area"]),
                            "log2_lesion_area": float(
                                np.log2(identity["lesion_area"])
                            ),
                            "lesion_topq_peak_error_norm": float(
                                metrics["lesion_topq_peak_error_norm"]
                            ),
                            "lesion_peak_error_norm": float(
                                metrics["lesion_peak_error_norm"]
                            ),
                            "lesion_mean_error_norm": float(
                                metrics["lesion_mean_error_norm"]
                            ),
                        }
                    )

            for seed in normalized_seeds:
                full_noise = torch.cat(
                    [
                        _normal_noise(
                            ct[local_index : local_index + 1],
                            seed=seed,
                            sample_index=start + local_index,
                        )
                        for local_index in range(stop - start)
                    ],
                    dim=0,
                )
                full_prediction = _sample_model(
                    model,
                    ct,
                    full_noise,
                    num_steps=num_steps,
                    amp=amp,
                )
                full_metrics = [
                    _lesion_metrics(
                        full_prediction[index],
                        target[index],
                        mask[index],
                    )
                    for index in range(stop - start)
                ]

                for crop_size, boxes in boxes_by_crop.items():
                    zoom_ct = torch.stack(
                        [
                            crop_resize(
                                ct[index],
                                box,
                                output_size=input_size,
                                mode="bilinear",
                            )
                            for index, box in enumerate(boxes)
                        ]
                    )
                    zoom_noise = torch.stack(
                        [
                            couple_noise_to_crop(
                                full_noise[index],
                                box,
                                output_size=input_size,
                            )
                            for index, box in enumerate(boxes)
                        ]
                    )
                    zoom_prediction = _sample_model(
                        model,
                        zoom_ct,
                        zoom_noise,
                        num_steps=num_steps,
                        amp=amp,
                    )

                    for local_index, box in enumerate(boxes):
                        projected = backproject_crop(
                            zoom_prediction[local_index],
                            box,
                            canvas_size=height_width,
                            mode="bilinear",
                            base=full_prediction[local_index],
                        )
                        zoom_metrics = _lesion_metrics(
                            projected,
                            target[local_index],
                            mask[local_index],
                        )
                        full = full_metrics[local_index]
                        crop_slice = (
                            ...,
                            slice(box.top, box.bottom),
                            slice(box.left, box.right),
                        )
                        context_full = _context_hotspot_density(
                            full_prediction[local_index][crop_slice],
                            target[local_index][crop_slice],
                            mask[local_index][crop_slice],
                        )
                        context_zoom = _context_hotspot_density(
                            projected[crop_slice],
                            target[local_index][crop_slice],
                            mask[local_index][crop_slice],
                        )
                        identity = cohort.cohort.iloc[start + local_index]
                        full_topq = float(
                            full["lesion_topq_peak_error_norm"]
                        )
                        zoom_topq = float(
                            zoom_metrics["lesion_topq_peak_error_norm"]
                        )
                        row = {
                            "patient_id": str(identity["patient_id"]),
                            "sample_id": str(identity["sample_id"]),
                            "slice_id": identity["slice_id"],
                            "seed": seed,
                            "crop_size": crop_size,
                            "magnification_factor": input_size / crop_size,
                            "lesion_area": float(identity["lesion_area"]),
                            "log2_lesion_area": float(
                                np.log2(identity["lesion_area"])
                            ),
                            "lesion_topq_peak_error_norm_full": full_topq,
                            "lesion_topq_peak_error_norm_zoom": zoom_topq,
                            "lesion_topq_absolute_improvement": (
                                full_topq - zoom_topq
                            ),
                            "lesion_topq_relative_improvement": (
                                _relative_reduction(full_topq, zoom_topq)
                            ),
                            "lesion_peak_error_norm_full": float(
                                full["lesion_peak_error_norm"]
                            ),
                            "lesion_peak_error_norm_zoom": float(
                                zoom_metrics["lesion_peak_error_norm"]
                            ),
                            "lesion_mean_error_norm_full": float(
                                full["lesion_mean_error_norm"]
                            ),
                            "lesion_mean_error_norm_zoom": float(
                                zoom_metrics["lesion_mean_error_norm"]
                            ),
                            "context_hotspot_density_full": context_full,
                            "context_hotspot_density_zoom": context_zoom,
                        }
                        sample_rows.append(row)

    sample_metrics = pd.DataFrame(sample_rows)
    return AuditTables(
        sample_metrics=sample_metrics,
        patient_metrics=_patient_table(sample_metrics),
        roundtrip_metrics=pd.DataFrame(roundtrip_rows),
    )


def compute_gate_statistics(
    tables: AuditTables,
    *,
    crop_sizes: Sequence[int],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[int, dict[str, Any]]:
    """Estimate all predeclared gate statistics for each crop independently."""

    estimates: dict[int, dict[str, Any]] = {}
    for crop_index, crop_size in enumerate(crop_sizes):
        rows = tables.sample_metrics[
            tables.sample_metrics["crop_size"] == int(crop_size)
        ]
        if rows.empty:
            raise ValueError(f"no sample metrics for crop_size={crop_size}")
        area_q25 = float(
            rows.drop_duplicates("sample_id")["lesion_area"].quantile(0.25)
        )
        log_area_q25 = float(np.log2(area_q25))
        seed_offset = 10_000 * crop_index
        relative = bootstrap_patient_balanced_slope(
            rows["patient_id"],
            rows["log2_lesion_area"],
            rows["lesion_topq_relative_improvement"],
            seed=bootstrap_seed + seed_offset,
            replicates=bootstrap_replicates,
            x_reference=log_area_q25,
        )
        absolute = bootstrap_patient_balanced_slope(
            rows["patient_id"],
            rows["log2_lesion_area"],
            rows["lesion_topq_absolute_improvement"],
            seed=bootstrap_seed + seed_offset + 1,
            replicates=bootstrap_replicates,
            x_reference=log_area_q25,
        )

        patient_context = rows.groupby("patient_id")[
            [
                "context_hotspot_density_full",
                "context_hotspot_density_zoom",
            ]
        ].mean()
        context_full = float(
            patient_context["context_hotspot_density_full"].mean()
        )
        context_zoom = float(
            patient_context["context_hotspot_density_zoom"].mean()
        )
        if context_full <= 1.0e-12:
            context_worsening = 0.0 if context_zoom <= 1.0e-12 else float("inf")
        else:
            context_worsening = (context_zoom - context_full) / context_full

        roundtrip_rows = tables.roundtrip_metrics[
            tables.roundtrip_metrics["crop_size"] == int(crop_size)
        ]
        roundtrip = bootstrap_patient_balanced_slope(
            roundtrip_rows["patient_id"],
            roundtrip_rows["log2_lesion_area"],
            roundtrip_rows["lesion_topq_peak_error_norm"],
            seed=bootstrap_seed + seed_offset + 2,
            replicates=bootstrap_replicates,
            x_reference=log_area_q25,
        )
        absolute_gain = absolute.reference_prediction
        roundtrip_fraction = (
            roundtrip.reference_prediction / max(absolute_gain, 1.0e-12)
            if absolute_gain > 0.0
            else float("inf")
        )
        estimates[int(crop_size)] = {
            "area_q25": area_q25,
            "relative_topq": relative,
            "absolute_topq": absolute,
            "roundtrip_topq": roundtrip,
            "context_hotspot_relative_worsening": float(context_worsening),
            "roundtrip_fraction_of_absolute_gain": float(roundtrip_fraction),
        }
    return estimates


def _finite_le(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value <= threshold)


def _finite_lt(value: float, threshold: float) -> bool:
    return bool(np.isfinite(value) and value < threshold)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _gate_result(statistics: Mapping[str, Any]) -> dict[str, Any]:
    relative = statistics["relative_topq"]
    checks = {
        "area_slope_ci95_high_below_zero": _finite_lt(
            relative.slope_ci95_high,
            0.0,
        ),
        "q25_relative_improvement_at_least_3pct": bool(
            np.isfinite(relative.reference_prediction)
            and relative.reference_prediction >= 0.03
        ),
        "q25_relative_improvement_ci95_low_above_zero": bool(
            np.isfinite(relative.reference_ci95_low)
            and relative.reference_ci95_low > 0.0
        ),
        "context_hotspot_worsening_within_5pct": _finite_le(
            statistics["context_hotspot_relative_worsening"],
            0.05,
        ),
        "roundtrip_fraction_within_25pct": _finite_le(
            statistics["roundtrip_fraction_of_absolute_gain"],
            0.25,
        ),
    }
    serialized = {
        key: asdict(value)
        if isinstance(value, PatientBootstrapResult)
        else value
        for key, value in statistics.items()
    }
    return {
        "would_pass": all(checks.values()),
        "checks": checks,
        "statistics": _json_safe(serialized),
    }


def build_primary_gate(
    statistics_by_crop: Mapping[int, Mapping[str, Any]],
    *,
    primary_crop_size: int,
) -> dict[str, Any]:
    """Apply the gate to the primary crop; sensitivities cannot rescue it."""

    if primary_crop_size not in statistics_by_crop:
        raise ValueError(
            f"primary_crop_size={primary_crop_size} has no statistics"
        )
    results = {
        str(crop_size): _gate_result(statistics)
        for crop_size, statistics in sorted(statistics_by_crop.items())
    }
    primary = results[str(primary_crop_size)]
    sensitivity = {
        crop_size: result
        for crop_size, result in results.items()
        if int(crop_size) != int(primary_crop_size)
    }
    return {
        "decision": "PASS" if primary["would_pass"] else "FAIL",
        "primary_crop_size": int(primary_crop_size),
        "primary_result": primary,
        "sensitivity_results": sensitivity,
        "sensitivity_can_rescue_primary": False,
        "thresholds": {
            "area_slope_ci95_high": 0.0,
            "q25_relative_improvement": 0.03,
            "q25_relative_improvement_ci95_low": 0.0,
            "context_hotspot_relative_worsening": 0.05,
            "roundtrip_fraction_of_absolute_gain": 0.25,
        },
    }


def write_audit_artifacts(
    output: Path,
    *,
    tables: AuditTables,
    manifest: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> None:
    """Write the fixed audit schema to a new, non-overwriting directory."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tables.sample_metrics.to_csv(output / "sample_metrics.csv", index=False)
    tables.patient_metrics.to_csv(output / "patient_metrics.csv", index=False)
    tables.roundtrip_metrics.to_csv(
        output / "roundtrip_metrics.csv",
        index=False,
    )
    (output / "audit_manifest.json").write_text(
        json.dumps(
            _json_safe(dict(manifest)),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (output / "gate.json").write_text(
        json.dumps(
            _json_safe(dict(gate)),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
