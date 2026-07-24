"""Uncertainty-aware H4-v2 evidence, calibration, and locked probes.

The functions in this module are deliberately split into fit/apply paths.
Population statistics and model coefficients are fitted on development
mechanism-train patients; confidence and subgroup thresholds are frozen from
development calibration.  A future external cohort only calls the apply path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.mechanism_validation.common import (
    bootstrap_mean,
    canonical_json_sha256,
    sign_flip_p,
)


BANDS = ("l2_lh", "l2_hl", "l2_hh", "l1_lh", "l1_hl", "l1_hh")
MODEL_ORDER = (
    "no_route",
    "h3_fixed_schedule",
    "original_evidence",
    "uncertainty_aware",
)
RIDGE_ALPHA = 1.0
NEIGHBOR_RADIUS = 1
PATIENT_SHRINKAGE_K = 60.0
CONFIDENCE_QUANTILE = 0.25
MIN_CONFIRMATION_PATIENTS = 40
MIN_SUBGROUP_PATIENTS = 10
MIN_ACTIVE_COVERAGE = 0.50
SUBGROUP_MARGIN_FRACTION = 0.05


def _as_id(value: Any, width: int = 3) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(width) if text.isdigit() else text


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["patient_id"] = result["patient_id"].map(
        lambda value: _as_id(value, 3)
    )
    result["sample_id"] = result["sample_id"].map(
        lambda value: _as_id(value, 6)
    )
    if "slice_id" not in result:
        result["slice_id"] = result["sample_id"].map(
            lambda value: int(str(value)[-3:])
            if str(value)[-3:].isdigit()
            else -1
        )
    return result


def add_context_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Add adjacent-slice and same-orientation cross-scale evidence context."""

    frame = normalize_ids(rows)
    required = {
        "patient_id",
        "sample_id",
        "band",
        "timestep",
        "log_snr",
        "noise_calibrated_evidence",
        "recoverability",
        "partition",
        "slice_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H4-v2 rows missing columns: {sorted(missing)}")
    if not set(frame["band"]).issubset(BANDS):
        raise ValueError("H4-v2 rows contain an unsupported band")
    frame = frame.sort_values(
        ["patient_id", "band", "timestep", "slice_id", "sample_id"]
    ).reset_index(drop=True)
    lookup = {
        (
            str(row.patient_id),
            str(row.band),
            int(row.timestep),
            int(row.slice_id),
        ): float(row.noise_calibrated_evidence)
        for row in frame.itertuples(index=False)
    }
    peer_lookup = {
        (
            str(row.patient_id),
            str(row.sample_id),
            int(row.timestep),
            str(row.band).split("_", 1)[1],
            str(row.band).split("_", 1)[0],
        ): float(row.noise_calibrated_evidence)
        for row in frame.itertuples(index=False)
    }
    context_median: list[float] = []
    context_dispersion: list[float] = []
    context_count: list[int] = []
    neighbor_count: list[int] = []
    scale_peer_present: list[float] = []
    for row in frame.itertuples(index=False):
        patient = str(row.patient_id)
        band = str(row.band)
        timestep = int(row.timestep)
        slice_id = int(row.slice_id)
        values = [float(row.noise_calibrated_evidence)]
        neighbors = 0
        for delta in range(-NEIGHBOR_RADIUS, NEIGHBOR_RADIUS + 1):
            if delta == 0:
                continue
            candidate = lookup.get((patient, band, timestep, slice_id + delta))
            if candidate is not None:
                values.append(candidate)
                neighbors += 1
        level, orientation = band.split("_", 1)
        other_level = "l1" if level == "l2" else "l2"
        peer = peer_lookup.get(
            (
                patient,
                str(row.sample_id),
                timestep,
                orientation,
                other_level,
            )
        )
        if peer is not None:
            values.append(peer)
        array = np.asarray(values, dtype=np.float64)
        median = float(np.median(array))
        dispersion = float(np.median(np.abs(array - median)))
        context_median.append(median)
        context_dispersion.append(dispersion)
        context_count.append(int(array.size))
        neighbor_count.append(neighbors)
        scale_peer_present.append(float(peer is not None))
    frame["context_evidence"] = context_median
    frame["context_dispersion"] = context_dispersion
    frame["context_count"] = context_count
    frame["neighbor_count"] = neighbor_count
    frame["scale_peer_present"] = scale_peer_present
    return frame


def fit_population_stats(
    context: pd.DataFrame,
    *,
    partition: str = "mechanism_train",
) -> dict[str, dict[str, float]]:
    selected = context[context["partition"] == partition]
    if selected.empty:
        raise ValueError(f"No rows in fit partition {partition!r}")
    result: dict[str, dict[str, float]] = {}
    for (band, timestep), group in selected.groupby(
        ["band", "timestep"], sort=True
    ):
        values = group["context_evidence"].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = max(1.4826 * mad, 1e-6)
        result[f"{band}|{int(timestep)}"] = {
            "median": median,
            "scale": scale,
        }
    return result


def apply_hierarchical_calibration(
    context: pd.DataFrame,
    population_stats: Mapping[str, Mapping[str, float]],
    *,
    shrinkage_k: float = PATIENT_SHRINKAGE_K,
) -> pd.DataFrame:
    """Apply population band/time scaling plus shrunk patient-band centering."""

    if shrinkage_k <= 0:
        raise ValueError("shrinkage_k must be positive")
    frame = context.copy()
    population_median = []
    population_scale = []
    for row in frame.itertuples(index=False):
        key = f"{row.band}|{int(row.timestep)}"
        stats = population_stats.get(key)
        if not isinstance(stats, Mapping):
            raise KeyError(f"Missing frozen population statistics for {key}")
        population_median.append(float(stats["median"]))
        population_scale.append(max(float(stats["scale"]), 1e-6))
    frame["population_median"] = population_median
    frame["population_scale"] = population_scale
    frame["population_z"] = (
        frame["context_evidence"] - frame["population_median"]
    ) / frame["population_scale"]
    patient_offsets = (
        frame.groupby(["patient_id", "band"], sort=False)["population_z"]
        .agg(["median", "count"])
        .reset_index()
    )
    patient_offsets["weight"] = patient_offsets["count"] / (
        patient_offsets["count"] + shrinkage_k
    )
    patient_offsets["shrunk_offset"] = (
        patient_offsets["median"] * patient_offsets["weight"]
    )
    frame = frame.merge(
        patient_offsets[
            ["patient_id", "band", "shrunk_offset", "count"]
        ].rename(columns={"count": "patient_band_observations"}),
        on=["patient_id", "band"],
        how="left",
        validate="many_to_one",
    )
    frame["hierarchical_evidence"] = (
        frame["population_z"] - frame["shrunk_offset"]
    )
    stability = np.exp(
        -frame["context_dispersion"].to_numpy(dtype=np.float64)
        / frame["population_scale"].to_numpy(dtype=np.float64)
    )
    count_score = np.minimum(
        frame["context_count"].to_numpy(dtype=np.float64) / 4.0,
        1.0,
    )
    frame["local_evidence_confidence"] = np.clip(
        stability * count_score,
        0.0,
        1.0,
    )
    frame["patient_band_time_confidence"] = frame.groupby(
        ["patient_id", "band", "timestep"],
        sort=False,
    )["local_evidence_confidence"].transform("median")
    frame["evidence_confidence"] = np.sqrt(
        frame["local_evidence_confidence"]
        * frame["patient_band_time_confidence"]
    ).clip(0.0, 1.0)
    frame["confidence_weighted_evidence"] = (
        frame["hierarchical_evidence"] * frame["evidence_confidence"]
    )
    return frame


def fit_original_standardization(
    frame: pd.DataFrame,
    *,
    partition: str = "mechanism_train",
) -> dict[str, float]:
    values = frame.loc[
        frame["partition"] == partition,
        "noise_calibrated_evidence",
    ].to_numpy(dtype=np.float64)
    if not values.size:
        raise ValueError("No original evidence values for standardization")
    return {
        "mean": float(values.mean()),
        "scale": max(float(values.std()), 1e-8),
    }


def _feature_matrix(
    frame: pd.DataFrame,
    model_name: str,
    original_standardization: Mapping[str, float],
) -> tuple[np.ndarray, list[str]]:
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unknown H4-v2 comparator {model_name!r}")
    if model_name in ("no_route", "h3_fixed_schedule"):
        raise ValueError(
            f"{model_name} is a frozen lookup comparator, not a ridge model"
        )
    band_index = {band: index for index, band in enumerate(BANDS)}
    one_hot = np.zeros((len(frame), len(BANDS) - 1), dtype=np.float64)
    for row_index, band in enumerate(frame["band"].astype(str)):
        index = band_index[band]
        if index:
            one_hot[row_index, index - 1] = 1.0
    log_snr = frame["log_snr"].to_numpy(dtype=np.float64) / 20.0
    if model_name == "original_evidence":
        mean = float(original_standardization["mean"])
        scale = max(float(original_standardization["scale"]), 1e-8)
        evidence = (
            frame["noise_calibrated_evidence"].to_numpy(dtype=np.float64)
            - mean
        ) / scale
        columns = [
            evidence[:, None],
            (evidence * log_snr)[:, None],
            evidence[:, None] * one_hot,
        ]
        names = [
            "single_slice_noise_calibrated_evidence",
            "single_slice_evidence_x_bridge_log_snr",
            *[
                f"single_slice_evidence_x_band_{band}"
                for band in BANDS[1:]
            ],
        ]
    else:
        confidence = frame["evidence_confidence"].to_numpy(dtype=np.float64)
        weighted = (
            frame["hierarchical_evidence"].to_numpy(dtype=np.float64)
            * confidence
        )
        columns = [
            weighted[:, None],
            (weighted * log_snr)[:, None],
            weighted[:, None] * one_hot,
        ]
        names = [
            "confidence_weighted_hierarchical_context_evidence",
            "weighted_context_evidence_x_bridge_log_snr",
            *[
                f"weighted_context_evidence_x_band_{band}"
                for band in BANDS[1:]
            ],
        ]
    return np.concatenate(columns, axis=1), names


def _pack_ridge(
    model: Ridge,
    feature_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "type": "ridge",
        "alpha": float(model.alpha),
        "feature_names": list(feature_names),
        "coef": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "intercept": float(model.intercept_),
    }


def fit_comparators(
    calibrated: pd.DataFrame,
    original_standardization: Mapping[str, float],
    *,
    partition: str = "mechanism_train",
    alpha: float = RIDGE_ALPHA,
) -> dict[str, dict[str, Any]]:
    train = calibrated[calibrated["partition"] == partition]
    target = train["recoverability"].to_numpy(dtype=np.float64)
    if not target.size:
        raise ValueError("No mechanism-train rows for H4-v2 models")
    global_mean = float(target.mean())
    result: dict[str, dict[str, Any]] = {
        "no_route": {
            "type": "frozen_band_mean",
            "global_mean": global_mean,
            "band_means": {
                str(band): float(group["recoverability"].mean())
                for band, group in train.groupby("band", sort=True)
            },
            "time_conditioned": False,
            "evidence_conditioned": False,
        },
        "h3_fixed_schedule": {
            "type": "frozen_band_timestep_schedule",
            "global_mean": global_mean,
            "schedule": {
                f"{band}|{int(timestep)}": float(
                    group["recoverability"].mean()
                )
                for (band, timestep), group in train.groupby(
                    ["band", "timestep"], sort=True
                )
            },
            "source": (
                "mechanism_train band/timestep recoverability at the "
                "pre-registered H3 bridge log-SNR grid"
            ),
            "time_conditioned": True,
            "evidence_conditioned": False,
        },
    }
    h3_prediction = _apply_fixed_comparator(
        train,
        result["h3_fixed_schedule"],
    )
    correction_target = target - h3_prediction
    for name in ("original_evidence", "uncertainty_aware"):
        features, feature_names = _feature_matrix(
            train,
            name,
            original_standardization,
        )
        model = Ridge(alpha=alpha, fit_intercept=False).fit(
            features,
            correction_target,
        )
        result[name] = _pack_ridge(model, feature_names)
        result[name]["prediction_role"] = (
            "additive_correction_to_h3_fixed_schedule"
        )
    return result


def _predict_packed(features: np.ndarray, model: Mapping[str, Any]) -> np.ndarray:
    coefficients = np.asarray(model["coef"], dtype=np.float64)
    if features.shape[1] != coefficients.shape[0]:
        raise ValueError("Frozen model feature dimension mismatch")
    return features @ coefficients + float(model["intercept"])


def _apply_fixed_comparator(
    frame: pd.DataFrame,
    model: Mapping[str, Any],
) -> np.ndarray:
    model_type = str(model.get("type", ""))
    fallback = float(model["global_mean"])
    if model_type == "frozen_band_mean":
        band_means = model["band_means"]
        return np.asarray(
            [
                float(band_means.get(str(band), fallback))
                for band in frame["band"]
            ],
            dtype=np.float64,
        )
    if model_type == "frozen_band_timestep_schedule":
        schedule = model["schedule"]
        return np.asarray(
            [
                float(
                    schedule.get(
                        f"{row.band}|{int(row.timestep)}",
                        fallback,
                    )
                )
                for row in frame.itertuples(index=False)
            ],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported frozen comparator type {model_type!r}")


def apply_comparators(
    calibrated: pd.DataFrame,
    *,
    models: Mapping[str, Mapping[str, Any]],
    original_standardization: Mapping[str, float],
    confidence_threshold: float,
) -> pd.DataFrame:
    frame = calibrated.copy()
    raw_predictions: dict[str, np.ndarray] = {
        name: _apply_fixed_comparator(frame, models[name])
        for name in ("no_route", "h3_fixed_schedule")
    }
    for name in ("original_evidence", "uncertainty_aware"):
        features, feature_names = _feature_matrix(
            frame,
            name,
            original_standardization,
        )
        model = models[name]
        if list(model["feature_names"]) != feature_names:
            raise ValueError(f"Frozen feature order mismatch for {name}")
        raw_predictions[name] = (
            raw_predictions["h3_fixed_schedule"]
            + _predict_packed(features, model)
        )
    active = (
        frame["evidence_confidence"].to_numpy(dtype=np.float64)
        >= confidence_threshold
    )
    raw_predictions["uncertainty_aware"] = np.where(
        active,
        raw_predictions["uncertainty_aware"],
        raw_predictions["h3_fixed_schedule"],
    )
    for name, values in raw_predictions.items():
        frame[f"prediction_{name}"] = values
    frame["router_active"] = active.astype(float)
    frame["router_abstained"] = (~active).astype(float)
    return frame


def patient_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (partition, patient), group in predictions.groupby(
        ["partition", "patient_id"], sort=True
    ):
        target = group["recoverability"].to_numpy(dtype=np.float64)
        record: dict[str, Any] = {
            "partition": str(partition),
            "patient_id": str(patient),
            "rows": int(len(group)),
            "active_fraction": float(group["router_active"].mean()),
            "mean_confidence": float(group["evidence_confidence"].mean()),
        }
        for name in MODEL_ORDER:
            prediction = group[f"prediction_{name}"].to_numpy(
                dtype=np.float64
            )
            record[f"mse_{name}"] = float(
                np.mean((target - prediction) ** 2)
            )
        records.append(record)
    return pd.DataFrame(records)


def fit_patient_attributes(
    h1_sample_rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = normalize_ids(h1_sample_rows)
    contrast_key = "pet_ll2_lesion_ring_log_ratio"
    required = {"partition", "patient_id", "sample_id", "mask_area", contrast_key}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"H1 attributes missing columns: {sorted(missing)}")
    return (
        frame.groupby(["partition", "patient_id"], as_index=False)
        .agg(
            slice_count=("sample_id", "nunique"),
            mean_mask_area=("mask_area", "mean"),
            mean_pet_ll2_log_contrast=(contrast_key, "mean"),
        )
        .sort_values(["partition", "patient_id"])
        .reset_index(drop=True)
    )


def freeze_subgroup_spec(
    attributes: pd.DataFrame,
    patient_error_frame: pd.DataFrame,
) -> dict[str, Any]:
    train = attributes[attributes["partition"] == "mechanism_train"]
    calibration_errors = patient_error_frame[
        patient_error_frame["partition"] == "calibration"
    ]
    if train["patient_id"].nunique() < 20 or len(calibration_errors) < 5:
        raise ValueError("Insufficient development patients for subgroup spec")
    h3_reference = calibration_errors[
        "mse_h3_fixed_schedule"
    ].to_numpy(dtype=np.float64)
    margin = SUBGROUP_MARGIN_FRACTION * float(np.median(h3_reference))
    return {
        "few_slices": {
            "field": "slice_count",
            "operator": "<=",
            "threshold": float(train["slice_count"].quantile(0.25)),
        },
        "small_lesion": {
            "field": "mean_mask_area",
            "operator": "<=",
            "threshold": float(train["mean_mask_area"].quantile(0.25)),
        },
        "low_contrast": {
            "field": "mean_pet_ll2_log_contrast",
            "operator": "<=",
            "threshold": float(
                train["mean_pet_ll2_log_contrast"].quantile(0.25)
            ),
        },
        "noninferiority_margin": margin,
        "margin_fraction_of_calibration_median_h3_mse": (
            SUBGROUP_MARGIN_FRACTION
        ),
        "minimum_patients_per_subgroup": MIN_SUBGROUP_PATIENTS,
    }


def assign_subgroups(
    attributes: pd.DataFrame,
    subgroup_spec: Mapping[str, Any],
) -> pd.DataFrame:
    frame = attributes.copy()
    for name in ("few_slices", "small_lesion", "low_contrast"):
        spec = subgroup_spec[name]
        if spec.get("operator") != "<=":
            raise ValueError(f"Unsupported subgroup operator for {name}")
        frame[name] = (
            frame[str(spec["field"])] <= float(spec["threshold"])
        )
    return frame


def effect_statistics(
    patient_error_frame: pd.DataFrame,
    *,
    partition: str,
    left: str,
    right: str,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    selected = patient_error_frame[
        patient_error_frame["partition"] == partition
    ]
    effect = (
        selected[f"mse_{right}"].to_numpy(dtype=np.float64)
        - selected[f"mse_{left}"].to_numpy(dtype=np.float64)
    )
    result = bootstrap_mean(effect, seed=seed, replicates=replicates)
    result.update(
        {
            "effect": f"MSE({right}) - MSE({left}); positive favors {left}",
            "sign_flip_p": sign_flip_p(
                effect,
                seed=seed + 1,
                replicates=replicates,
            ),
        }
    )
    return result


def development_diagnostics(
    patient_error_frame: pd.DataFrame,
    *,
    partition: str = "validation",
    seed: int = 20260730,
) -> dict[str, Any]:
    return {
        "partition": partition,
        "exploratory_only": True,
        "h3_vs_no_route": effect_statistics(
            patient_error_frame,
            partition=partition,
            left="h3_fixed_schedule",
            right="no_route",
            seed=seed,
        ),
        "original_vs_h3": effect_statistics(
            patient_error_frame,
            partition=partition,
            left="original_evidence",
            right="h3_fixed_schedule",
            seed=seed + 100,
        ),
        "uncertainty_vs_h3": effect_statistics(
            patient_error_frame,
            partition=partition,
            left="uncertainty_aware",
            right="h3_fixed_schedule",
            seed=seed + 200,
        ),
        "uncertainty_vs_original": effect_statistics(
            patient_error_frame,
            partition=partition,
            left="uncertainty_aware",
            right="original_evidence",
            seed=seed + 300,
        ),
    }


def seal_probe(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("probe_sha256", None)
    return {**body, "probe_sha256": canonical_json_sha256(body)}


def validate_probe(payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    claimed = str(body.pop("probe_sha256", ""))
    computed = canonical_json_sha256(body)
    if not claimed or claimed != computed:
        raise ValueError(
            f"H4-v2 probe self-hash mismatch: {claimed!r} != {computed!r}"
        )
    if payload.get("schema_version") != 2:
        raise ValueError("H4-v2 probe schema_version must be 2")
    if tuple(payload.get("bands", ())) != BANDS:
        raise ValueError("H4-v2 frozen band order mismatch")
    if tuple(payload.get("model_order", ())) != MODEL_ORDER:
        raise ValueError("H4-v2 comparator order mismatch")


def patient_set_sha256(patient_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(str(value) for value in patient_ids)).encode("utf-8")
    ).hexdigest()
