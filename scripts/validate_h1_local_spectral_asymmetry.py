"""Stage 0B / H1: local-only CT/PET spectral asymmetry validation.

H1 has two independently falsifiable parts:

1. CT multiscale band responses localize annotated PET-lesion support.
2. Given lesion geometry, CT band amplitudes do not provide recoverable
   incremental prediction of PET lesion-band amplitudes.

The authoritative split is the locked Stage-0A dataset contract.  Manifest
train patients are deterministically divided into mechanism-train and
calibration; manifest val is untouched until all directions, band choices,
ridge hyperparameters, size cut points, and null thresholds are frozen.
All uncertainty intervals and outcome comparisons are patient-level.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


SCHEMA_VERSION = 1
IMAGE_SIZE = 192
PARTITION_SEED = 42
ANALYSIS_SEED = 20260723
CALIBRATION_FRACTION = 0.20
BOOTSTRAP_REPLICATES = 10_000
PERMUTATION_REPLICATES = 10_000
REGRESSION_NULL_REPLICATES = 2_000
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
RING_RADIUS = 12
BANDS = ("ll2", "l2_lh", "l2_hl", "l2_hh", "l1_lh", "l1_hl", "l1_hh")
SUPPORT_CANDIDATES = BANDS


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ()
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("contract_sha256", "")
    actual = _canonical_sha256(payload)
    if claimed != actual:
        raise RuntimeError(
            f"dataset contract self-hash mismatch: claimed={claimed}, actual={actual}"
        )
    payload["contract_sha256"] = claimed
    if payload.get("contract_status") != "LOCKED":
        raise RuntimeError(f"dataset contract is not LOCKED: {payload.get('contract_status')}")
    if not payload.get("claim_boundary", {}).get("dataset_hypothesis_allowed", False):
        raise RuntimeError("dataset contract does not allow dataset-hypothesis analysis")
    return payload


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _index_pngs(root: Path, modality: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        folder = root / split / modality
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.png")):
            if path.stem in result:
                raise RuntimeError(f"duplicate {modality} sample_id: {path.stem}")
            result[path.stem] = path
    return result


def _read_image(path: Path, *, nearest: bool = False, invert: bool = False) -> np.ndarray:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE), resample)
        value = np.asarray(image, dtype=np.float32) / 255.0
    return 1.0 - value if invert else value


def _read_mask(path: Path) -> np.ndarray:
    value = _read_image(path, nearest=True, invert=False)
    return value > 0.5


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


def _resize_float(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(image.astype(np.float32), mode="F")
    pil = pil.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def _band_energy_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    ll1, detail1 = _haar2(image)
    ll2, detail2 = _haar2(ll1)
    result = {"ll2": _resize_float(np.abs(ll2))}
    for level, details in (("l1", detail1), ("l2", detail2)):
        for name, value in zip(("lh", "hl", "hh"), details, strict=True):
            local = ndimage.uniform_filter(np.abs(value), size=3, mode="nearest")
            result[f"{level}_{name}"] = _resize_float(local)
    return result


def _safe_auc(mask: np.ndarray, ring: np.ndarray, score: np.ndarray) -> float:
    selected = mask | ring
    labels = mask[selected].astype(np.uint8)
    values = score[selected].astype(np.float64)
    if labels.size < 4 or np.unique(labels).size != 2 or np.ptp(values) <= 1e-12:
        return 0.5
    return float(roc_auc_score(labels, values))


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        return float("nan"), float("nan")
    return float(xx.mean()), float(yy.mean())


def _patient_partition(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    train_patients = sorted({row["patient_id"] for row in rows if row["split"] == "train"})
    shuffled = np.asarray(train_patients, dtype=object)
    np.random.default_rng(PARTITION_SEED).shuffle(shuffled)
    calibration_count = max(1, round(CALIBRATION_FRACTION * len(shuffled)))
    calibration = set(shuffled[:calibration_count].tolist())
    result = {
        patient_id: ("calibration" if patient_id in calibration else "mechanism_train")
        for patient_id in train_patients
    }
    result.update(
        {
            row["patient_id"]: "validation"
            for row in rows
            if row["split"] == "val"
        }
    )
    return result


def _bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def _sign_flip_p(effect: np.ndarray, seed: int, *, two_sided: bool = False) -> float:
    effect = np.asarray(effect, dtype=np.float64)
    effect = effect[np.isfinite(effect)]
    if effect.size == 0:
        return float("nan")
    observed = float(effect.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATION_REPLICATES):
        candidate = float((effect * rng.choice((-1.0, 1.0), size=effect.size)).mean())
        if (abs(candidate) >= abs(observed)) if two_sided else (candidate >= observed):
            exceed += 1
    return float((exceed + 1) / (PERMUTATION_REPLICATES + 1))


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = rankdata(np.asarray(left, dtype=np.float64), method="average")
    right_rank = rankdata(np.asarray(right, dtype=np.float64), method="average")
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return float(left_rank @ right_rank / denominator) if denominator > 1e-12 else 0.0


def _bootstrap_rank_corr(
    left: np.ndarray, right: np.ndarray, seed: int
) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    observed = _rank_corr(left, right)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, left.size, size=left.size)
        values[index] = _rank_corr(left[selected], right[selected])
    low, high = np.quantile(values, (0.025, 0.975))
    return observed, float(low), float(high)


def _corr_permutation_p(left: np.ndarray, right: np.ndarray, seed: int) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    observed = abs(_rank_corr(left, right))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(PERMUTATION_REPLICATES):
        if abs(_rank_corr(left, rng.permutation(right))) >= observed:
            exceed += 1
    return float((exceed + 1) / (PERMUTATION_REPLICATES + 1))


def _bh_adjust(rows: list[dict[str, Any]], key: str = "permutation_p") -> None:
    valid = [(index, float(row[key])) for index, row in enumerate(rows) if np.isfinite(row[key])]
    if not valid:
        return
    ordered = sorted(valid, key=lambda item: item[1])
    count = len(ordered)
    adjusted = [1.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        _, p_value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, p_value * count / rank)
        adjusted[reverse_index] = running
    for (original_index, _), value in zip(ordered, adjusted, strict=True):
        rows[original_index]["permutation_fdr_bh"] = float(min(value, 1.0))


def _aggregate_patients(sample_df: pd.DataFrame) -> pd.DataFrame:
    numeric = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    patient = sample_df.groupby(["patient_id", "partition"], as_index=False)[numeric].mean()
    return patient


def _support_statistics(
    patient_df: pd.DataFrame,
    directions: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = ("raw_ct", *BANDS)
    for partition_index, partition in enumerate(("mechanism_train", "calibration", "validation")):
        current = patient_df[patient_df["partition"] == partition]
        partition_rows: list[dict[str, Any]] = []
        for band_index, band in enumerate(candidates):
            raw = current[f"support_auc_{band}"].to_numpy(dtype=np.float64)
            directional = raw if directions[band] > 0 else 1.0 - raw
            estimate, low, high = _bootstrap_mean(
                directional, ANALYSIS_SEED + partition_index * 100 + band_index
            )
            effect = directional - 0.5
            partition_rows.append(
                {
                    "partition": partition,
                    "band": band,
                    "direction": "+" if directions[band] > 0 else "-",
                    "patients": len(current),
                    "directional_auc": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "permutation_p": _sign_flip_p(
                        effect,
                        ANALYSIS_SEED + 1_000 + partition_index * 100 + band_index,
                    ),
                }
            )
        _bh_adjust(partition_rows)
        rows.extend(partition_rows)
    return rows


def _paired_comparison(left: np.ndarray, right: np.ndarray, seed: int) -> dict[str, Any]:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    estimate, low, high = _bootstrap_mean(difference, seed)
    return {
        "patients": int(np.isfinite(difference).sum()),
        "mean_difference": estimate,
        "ci95_low": low,
        "ci95_high": high,
        "permutation_p": _sign_flip_p(difference, seed + 10_000),
    }


def _standardize_fit(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = value.mean(axis=0)
    scale = value.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return mean, scale


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
    *,
    x_mean: np.ndarray | None = None,
    x_scale: np.ndarray | None = None,
    y_mean: np.ndarray | None = None,
    y_scale: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if x_mean is None or x_scale is None:
        x_mean, x_scale = _standardize_fit(train_x)
    if y_mean is None or y_scale is None:
        y_mean, y_scale = _standardize_fit(train_y)
    model = Ridge(alpha=alpha)
    model.fit((train_x - x_mean) / x_scale, (train_y - y_mean) / y_scale)
    prediction = model.predict((test_x - x_mean) / x_scale)
    return prediction, (x_mean, x_scale, y_mean, y_scale)


def _select_alpha(
    train_x: np.ndarray,
    train_y: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    results: list[dict[str, float]] = []
    for alpha in RIDGE_ALPHAS:
        prediction, (_, _, y_mean, y_scale) = _ridge_predict(
            train_x, train_y, calibration_x, alpha
        )
        target = (calibration_y - y_mean) / y_scale
        mse = float(np.mean((target - prediction) ** 2))
        results.append({"alpha": alpha, "calibration_mse": mse})
    selected = min(results, key=lambda row: (row["calibration_mse"], row["alpha"]))
    return float(selected["alpha"]), results


def _skill(target: np.ndarray, full: np.ndarray, baseline: np.ndarray) -> float:
    full_error = float(np.mean((target - full) ** 2))
    baseline_error = float(np.mean((target - baseline) ** 2))
    return 1.0 - full_error / max(baseline_error, 1e-12)


def _skill_bootstrap(
    target: np.ndarray, full: np.ndarray, baseline: np.ndarray, seed: int
) -> tuple[float, float, float]:
    observed = _skill(target, full, baseline)
    rng = np.random.default_rng(seed)
    values = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, target.shape[0], size=target.shape[0])
        values[index] = _skill(target[selected], full[selected], baseline[selected])
    low, high = np.quantile(values, (0.025, 0.975))
    return observed, float(low), float(high)


def _regression_arrays(patient_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    geometry = ["log_mask_area", "mask_centroid_x", "mask_centroid_y"]
    ct_features: list[str] = []
    targets: list[str] = []
    for band in BANDS:
        ct_features.extend(
            [
                f"log_ct_{band}_lesion",
                f"log_ct_{band}_ring",
                f"log_ct_{band}_whole",
            ]
        )
        targets.append(f"log_pet_{band}_lesion")
    missing = [name for name in (*geometry, *ct_features, *targets) if name not in patient_df]
    if missing:
        raise KeyError(f"missing regression columns: {missing}")
    return geometry, ct_features, targets


def _amplitude_regression(patient_df: pd.DataFrame) -> dict[str, Any]:
    geometry, ct_features, targets = _regression_arrays(patient_df)
    train = patient_df[patient_df["partition"] == "mechanism_train"]
    calibration = patient_df[patient_df["partition"] == "calibration"]
    validation = patient_df[patient_df["partition"] == "validation"]

    train_base = train[geometry].to_numpy(dtype=np.float64)
    train_full = train[[*geometry, *ct_features]].to_numpy(dtype=np.float64)
    train_y = train[targets].to_numpy(dtype=np.float64)
    cal_base = calibration[geometry].to_numpy(dtype=np.float64)
    cal_full = calibration[[*geometry, *ct_features]].to_numpy(dtype=np.float64)
    cal_y = calibration[targets].to_numpy(dtype=np.float64)

    base_alpha, base_grid = _select_alpha(train_base, train_y, cal_base, cal_y)
    full_alpha, full_grid = _select_alpha(train_full, train_y, cal_full, cal_y)
    base_cal, (_, _, y_mean, y_scale) = _ridge_predict(
        train_base, train_y, cal_base, base_alpha
    )
    full_cal, _ = _ridge_predict(train_full, train_y, cal_full, full_alpha)
    cal_target = (cal_y - y_mean) / y_scale
    calibration_skill = _skill(cal_target, full_cal, base_cal)

    rng = np.random.default_rng(ANALYSIS_SEED + 30_000)
    calibration_null = np.empty(REGRESSION_NULL_REPLICATES, dtype=np.float64)
    geometry_count = len(geometry)
    for index in range(REGRESSION_NULL_REPLICATES):
        permuted = train_full.copy()
        order = rng.permutation(len(permuted))
        permuted[:, geometry_count:] = permuted[order, geometry_count:]
        permuted_alpha, _ = _select_alpha(permuted, train_y, cal_full, cal_y)
        permuted_prediction, _ = _ridge_predict(
            permuted, train_y, cal_full, permuted_alpha
        )
        calibration_null[index] = _skill(cal_target, permuted_prediction, base_cal)
    null_threshold = float(np.quantile(calibration_null, 0.95))

    development = patient_df[patient_df["partition"].isin(("mechanism_train", "calibration"))]
    dev_base = development[geometry].to_numpy(dtype=np.float64)
    dev_full = development[[*geometry, *ct_features]].to_numpy(dtype=np.float64)
    dev_y = development[targets].to_numpy(dtype=np.float64)
    val_base = validation[geometry].to_numpy(dtype=np.float64)
    val_full = validation[[*geometry, *ct_features]].to_numpy(dtype=np.float64)
    val_y = validation[targets].to_numpy(dtype=np.float64)

    base_val, (_, _, dev_y_mean, dev_y_scale) = _ridge_predict(
        dev_base, dev_y, val_base, base_alpha
    )
    full_val, _ = _ridge_predict(dev_full, dev_y, val_full, full_alpha)
    val_target = (val_y - dev_y_mean) / dev_y_scale
    val_skill, val_low, val_high = _skill_bootstrap(
        val_target, full_val, base_val, ANALYSIS_SEED + 40_000
    )

    validation_null = np.empty(REGRESSION_NULL_REPLICATES, dtype=np.float64)
    rng = np.random.default_rng(ANALYSIS_SEED + 50_000)
    for index in range(REGRESSION_NULL_REPLICATES):
        permuted = dev_full.copy()
        order = rng.permutation(len(permuted))
        permuted[:, geometry_count:] = permuted[order, geometry_count:]
        prediction, _ = _ridge_predict(permuted, dev_y, val_full, full_alpha)
        validation_null[index] = _skill(val_target, prediction, base_val)
    permutation_p = float(
        (1 + np.sum(validation_null >= val_skill)) / (REGRESSION_NULL_REPLICATES + 1)
    )

    if val_high <= null_threshold and permutation_p >= 0.05:
        status = "PASS_NO_RECOVERABLE_INCREMENT"
    elif val_low > null_threshold and permutation_p < 0.05:
        status = "FAIL_RECOVERABLE_INCREMENT"
    else:
        status = "INCONCLUSIVE"

    per_band: list[dict[str, Any]] = []
    for band_index, band in enumerate(BANDS):
        skill, low, high = _skill_bootstrap(
            val_target[:, band_index],
            full_val[:, band_index],
            base_val[:, band_index],
            ANALYSIS_SEED + 60_000 + band_index,
        )
        per_band.append(
            {
                "band": band,
                "validation_incremental_skill": skill,
                "ci95_low": low,
                "ci95_high": high,
            }
        )

    return {
        "definition": "incremental standardized MSE skill of geometry+CT bands over geometry-only baseline",
        "train_patients": len(train),
        "calibration_patients": len(calibration),
        "validation_patients": len(validation),
        "geometry_features": geometry,
        "ct_features": ct_features,
        "targets": targets,
        "baseline_alpha": base_alpha,
        "full_alpha": full_alpha,
        "baseline_alpha_grid": base_grid,
        "full_alpha_grid": full_grid,
        "calibration_skill": calibration_skill,
        "calibration_permutation_null_q95": null_threshold,
        "calibration_null_replicates": REGRESSION_NULL_REPLICATES,
        "validation_incremental_skill": val_skill,
        "validation_ci95_low": val_low,
        "validation_ci95_high": val_high,
        "validation_permutation_p": permutation_p,
        "validation_null_replicates": REGRESSION_NULL_REPLICATES,
        "status": status,
        "per_band": per_band,
    }


def run(
    root: Path,
    output: Path,
    *,
    raw_root_override: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = root / "results" / "mechanism_validation" / "00_data_audit" / "dataset_contract.json"
    contract = _load_contract(contract_path)
    manifest_path = root / contract["manifest"]["path"]
    raw_root = (
        raw_root_override.resolve()
        if raw_root_override is not None
        else root / contract["raw_png"]["root"]
    )
    manifest = _read_manifest(manifest_path)
    partition = _patient_partition(manifest)

    indexes = {
        "ct": _index_pngs(raw_root, "ct"),
        "pet": _index_pngs(raw_root, "pet"),
        "mask": _index_pngs(raw_root, "label"),
    }
    manifest_ids = {row["sample_id"] for row in manifest}
    for modality, index in indexes.items():
        if set(index) != manifest_ids:
            raise RuntimeError(
                f"{modality} index differs from manifest: missing={len(manifest_ids-set(index))}, "
                f"extra={len(set(index)-manifest_ids)}"
            )
    raw_hash_entries = [
        f"{sample_id}|{modality}|{_sha256_file(path)}"
        for modality, index in indexes.items()
        for sample_id, path in index.items()
    ]
    observed_raw_sha256 = hashlib.sha256(
        "\n".join(sorted(raw_hash_entries)).encode("utf-8")
    ).hexdigest()
    expected_raw_sha256 = contract["raw_png"]["combined_sha256"]
    if observed_raw_sha256 != expected_raw_sha256:
        raise RuntimeError(
            "raw PNG fingerprint differs from locked contract: "
            f"observed={observed_raw_sha256}, expected={expected_raw_sha256}"
        )

    partition_rows = [
        {
            "patient_id": patient_id,
            "manifest_split": "val" if role == "validation" else "train",
            "mechanism_partition": role,
        }
        for patient_id, role in sorted(partition.items())
    ]
    _write_csv(output / "patient_partition.csv", partition_rows)

    sample_rows: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(manifest, key=lambda item: item["sample_id"])):
        sample_id = row["sample_id"]
        ct = _read_image(indexes["ct"][sample_id])
        pet = _read_image(indexes["pet"][sample_id], invert=True)
        mask = _read_mask(indexes["mask"][sample_id])
        ring = ndimage.binary_dilation(mask, iterations=RING_RADIUS) & ~mask
        if not mask.any() or not ring.any():
            raise RuntimeError(f"empty lesion or ring: {sample_id}")
        ct_maps = _band_energy_maps(ct)
        pet_maps = _band_energy_maps(pet)
        cx, cy = _centroid(mask)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "patient_id": row["patient_id"],
            "manifest_split": row["split"],
            "partition": partition[row["patient_id"]],
            "mask_area": int(mask.sum()),
            "mask_centroid_x": cx / (IMAGE_SIZE - 1),
            "mask_centroid_y": cy / (IMAGE_SIZE - 1),
            "support_auc_raw_ct": _safe_auc(mask, ring, ct),
        }
        eps = 1e-8
        for band in BANDS:
            ct_map = ct_maps[band]
            pet_map = pet_maps[band]
            ct_lesion = float(ct_map[mask].mean())
            ct_ring = float(ct_map[ring].mean())
            ct_whole = float(ct_map.mean())
            pet_lesion = float(pet_map[mask].mean())
            pet_ring = float(pet_map[ring].mean())
            record.update(
                {
                    f"support_auc_{band}": _safe_auc(mask, ring, ct_map),
                    f"ct_{band}_lesion": ct_lesion,
                    f"ct_{band}_ring": ct_ring,
                    f"ct_{band}_whole": ct_whole,
                    f"ct_{band}_lesion_ring_log_ratio": math.log((ct_lesion + eps) / (ct_ring + eps)),
                    f"pet_{band}_lesion": pet_lesion,
                    f"pet_{band}_ring": pet_ring,
                    f"pet_{band}_whole": float(pet_map.mean()),
                    f"pet_{band}_lesion_ring_log_ratio": math.log((pet_lesion + eps) / (pet_ring + eps)),
                }
            )
        sample_rows.append(record)
        if (index + 1) % 200 == 0:
            print(f"processed {index + 1}/{len(manifest)} samples", flush=True)

    sample_df = pd.DataFrame(sample_rows)
    patient_df = _aggregate_patients(sample_df)
    patient_df["log_mask_area"] = np.log(patient_df["mask_area"].clip(lower=1.0))
    for band in BANDS:
        for modality in ("ct", "pet"):
            for region in ("lesion", "ring", "whole"):
                source = f"{modality}_{band}_{region}"
                patient_df[f"log_{source}"] = np.log(patient_df[source].clip(lower=1e-8))

    train = patient_df[patient_df["partition"] == "mechanism_train"]
    directions: dict[str, int] = {}
    for band in ("raw_ct", *BANDS):
        mean_auc = float(train[f"support_auc_{band}"].mean())
        directions[band] = 1 if mean_auc >= 0.5 else -1

    for band in ("raw_ct", *BANDS):
        source = patient_df[f"support_auc_{band}"].to_numpy(dtype=np.float64)
        patient_df[f"directional_auc_{band}"] = source if directions[band] > 0 else 1.0 - source

    calibration = patient_df[patient_df["partition"] == "calibration"]
    selected_band = max(
        SUPPORT_CANDIDATES,
        key=lambda band: (
            float(calibration[f"directional_auc_{band}"].mean()),
            -SUPPORT_CANDIDATES.index(band),
        ),
    )

    support_rows = _support_statistics(patient_df, directions)
    validation = patient_df[patient_df["partition"] == "validation"]
    selected_validation = next(
        row
        for row in support_rows
        if row["partition"] == "validation" and row["band"] == selected_band
    )
    support_vs_raw = _paired_comparison(
        validation[f"directional_auc_{selected_band}"].to_numpy(),
        validation["directional_auc_raw_ct"].to_numpy(),
        ANALYSIS_SEED + 20_000,
    )
    support_pass = bool(
        selected_validation["ci95_low"] > 0.5
        and selected_validation["permutation_p"] < 0.05
    )

    train_patient_area = patient_df.loc[
        patient_df["partition"] == "mechanism_train", "mask_area"
    ].to_numpy(dtype=np.float64)
    small_max, medium_max = np.quantile(train_patient_area, (1 / 3, 2 / 3))
    patient_df["size_stratum"] = np.where(
        patient_df["mask_area"] <= small_max,
        "small",
        np.where(patient_df["mask_area"] <= medium_max, "medium", "large"),
    )
    size_rows: list[dict[str, Any]] = []
    for partition_index, role in enumerate(("mechanism_train", "calibration", "validation")):
        for stratum_index, stratum in enumerate(("small", "medium", "large")):
            current = patient_df[
                (patient_df["partition"] == role) & (patient_df["size_stratum"] == stratum)
            ]
            values = current[f"directional_auc_{selected_band}"].to_numpy(dtype=np.float64)
            estimate, low, high = _bootstrap_mean(
                values, ANALYSIS_SEED + 70_000 + partition_index * 100 + stratum_index
            )
            size_rows.append(
                {
                    "partition": role,
                    "size_stratum": stratum,
                    "patients": len(current),
                    "selected_band": selected_band,
                    "directional_auc": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "permutation_p": _sign_flip_p(
                        values - 0.5,
                        ANALYSIS_SEED + 80_000 + partition_index * 100 + stratum_index,
                    ),
                }
            )

    correlation_rows: list[dict[str, Any]] = []
    for partition_index, role in enumerate(("development", "validation")):
        current = (
            patient_df[patient_df["partition"].isin(("mechanism_train", "calibration"))]
            if role == "development"
            else validation
        )
        role_rows: list[dict[str, Any]] = []
        for band_index, band in enumerate(BANDS):
            left = current[f"ct_{band}_lesion"].to_numpy(dtype=np.float64)
            right = current[f"pet_{band}_lesion"].to_numpy(dtype=np.float64)
            rho, low, high = _bootstrap_rank_corr(
                left, right, ANALYSIS_SEED + 90_000 + partition_index * 100 + band_index
            )
            role_rows.append(
                {
                    "partition": role,
                    "band": band,
                    "patients": len(current),
                    "spearman_rho": rho,
                    "ci95_low": low,
                    "ci95_high": high,
                    "permutation_p": _corr_permutation_p(
                        left,
                        right,
                        ANALYSIS_SEED + 100_000 + partition_index * 100 + band_index,
                    ),
                }
            )
        _bh_adjust(role_rows)
        correlation_rows.extend(role_rows)

    regression = _amplitude_regression(patient_df)
    amplitude_status = regression["status"]
    if not support_pass:
        h1_decision = "FAIL"
    elif amplitude_status == "FAIL_RECOVERABLE_INCREMENT":
        h1_decision = "FAIL"
    elif amplitude_status == "PASS_NO_RECOVERABLE_INCREMENT":
        h1_decision = "PASS"
    else:
        h1_decision = "INCONCLUSIVE"

    mask_nonempty_count = int((sample_df["mask_area"] > 0).sum())
    presence_identifiability = (
        "NOT_IDENTIFIABLE_ALL_SLICES_POSITIVE"
        if mask_nonempty_count == len(sample_df)
        else "IDENTIFIABLE"
    )

    sample_df.to_csv(output / "sample_band_metrics.csv", index=False, encoding="utf-8-sig")
    patient_df.to_csv(output / "patient_band_metrics.csv", index=False, encoding="utf-8-sig")
    _write_csv(output / "support_band_statistics.csv", support_rows)
    _write_csv(output / "size_stratified_support.csv", size_rows)
    _write_csv(output / "amplitude_correlations.csv", correlation_rows)
    _write_json(output / "amplitude_regression.json", regression)

    analysis_spec = {
        "schema_version": SCHEMA_VERSION,
        "dataset_contract_sha256": contract["contract_sha256"],
        "audited_raw_root": _relative(raw_root, root),
        "raw_png_combined_sha256": observed_raw_sha256,
        "partition_seed": PARTITION_SEED,
        "analysis_seed": ANALYSIS_SEED,
        "mechanism_train_fraction": 1.0 - CALIBRATION_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "permutation_replicates": PERMUTATION_REPLICATES,
        "regression_null_replicates": REGRESSION_NULL_REPLICATES,
        "support_rule": "calibration selects one train-oriented CT band; validation patient bootstrap lower CI > 0.5 and sign-flip p < 0.05",
        "amplitude_rule": {
            "metric": "incremental MSE skill of geometry+CT over geometry-only baseline",
            "threshold_source": "95th percentile of mechanism-train->calibration patient-permutation null",
            "pass": "validation upper bootstrap CI <= frozen null threshold and permutation p >= 0.05",
            "fail": "validation lower bootstrap CI > frozen null threshold and permutation p < 0.05",
            "otherwise": "INCONCLUSIVE",
        },
        "h1_rule": "PASS only if support PASS and amplitude PASS; FAIL if either is positively falsified; otherwise INCONCLUSIVE",
        "ridge_alphas": RIDGE_ALPHAS,
        "ring_radius_px": RING_RADIUS,
        "bands": BANDS,
    }
    _write_json(output / "analysis_spec.json", analysis_spec)

    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "00B_H1_local_spectral_asymmetry",
        "scope": "local_raw_png_only",
        "cloud_actions_performed": False,
        "dataset_contract_sha256": contract["contract_sha256"],
        "audited_raw_root": _relative(raw_root, root),
        "raw_png_combined_sha256": observed_raw_sha256,
        "decision": h1_decision,
        "H1_support_localization": {
            "status": "PASS" if support_pass else "FAIL",
            "selected_band": selected_band,
            "direction": "+" if directions[selected_band] > 0 else "-",
            "validation_directional_auc": selected_validation["directional_auc"],
            "validation_ci95": [selected_validation["ci95_low"], selected_validation["ci95_high"]],
            "validation_permutation_p": selected_validation["permutation_p"],
            "secondary_selected_band_minus_raw_ct": support_vs_raw,
        },
        "H1_pet_band_amplitude": {
            "status": amplitude_status,
            "validation_incremental_skill": regression["validation_incremental_skill"],
            "validation_ci95": [
                regression["validation_ci95_low"],
                regression["validation_ci95_high"],
            ],
            "calibration_null_q95": regression["calibration_permutation_null_q95"],
            "validation_permutation_p": regression["validation_permutation_p"],
            "wording": "incremental predictive evidence only; non-significance alone is not interpreted as inability",
        },
        "lesion_presence_prediction": presence_identifiability,
        "size_thresholds_from_mechanism_train_patient_mean_areas": {
            "small_max": float(small_max),
            "medium_max": float(medium_max),
        },
        "next_stage_allowed": h1_decision == "PASS",
        "stop_rule": (
            "H1 passed; subsequent local-only data validation may proceed."
            if h1_decision == "PASS"
            else "Do not promote the H1 mechanism story or corresponding module; resolve or accept this gate before proceeding."
        ),
    }
    _write_json(output / "decision.json", decision)

    report = [
        "# Stage 0B / H1 local spectral asymmetry",
        "",
        f"Decision: **{h1_decision}**. No cache, checkpoint, model, training, or cloud action was used.",
        "",
        f"Locked dataset contract: `{contract['contract_sha256']}`.",
        "",
        "## Primary gates",
        "",
        f"- CT support localization: **{'PASS' if support_pass else 'FAIL'}**. Calibration selected "
        f"`{selected_band}` with direction `{'+' if directions[selected_band] > 0 else '-'}`; "
        f"validation patient-level AUC {selected_validation['directional_auc']:.4f}, 95% CI "
        f"[{selected_validation['ci95_low']:.4f}, {selected_validation['ci95_high']:.4f}], "
        f"permutation p={selected_validation['permutation_p']:.6f}.",
        f"- PET band-amplitude increment: **{amplitude_status}**. Validation skill "
        f"{regression['validation_incremental_skill']:.4f}, 95% CI "
        f"[{regression['validation_ci95_low']:.4f}, {regression['validation_ci95_high']:.4f}]; "
        f"calibration null q95={regression['calibration_permutation_null_q95']:.4f}; "
        f"permutation p={regression['validation_permutation_p']:.6f}.",
        "",
        "## Guardrails",
        "",
        "- Associations and predictive probes are not causal or mutual-information claims.",
        "- All thresholds, directions, feature choices, and hyperparameters were frozen before validation.",
        "- Lesion presence is not identifiable because every manifest slice has a non-empty mask.",
        "- PNG values are display intensities, not physical HU or SUV.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_json(
        output / "execution_metadata.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script": _relative(Path(__file__), root),
            "script_sha256": _sha256_file(Path(__file__)),
            "python": sys.version,
            "command": " ".join(sys.argv),
        },
    )
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/00B_h1_spectral_asymmetry"),
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help=(
            "Physical CT/PET/label PNG root override. Its complete byte "
            "fingerprint must match the locked Stage-0A contract."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    raw_root = (
        None
        if args.raw_root is None
        else (
            args.raw_root.resolve()
            if args.raw_root.is_absolute()
            else (root / args.raw_root).resolve()
        )
    )
    decision = run(root, output, raw_root_override=raw_root)
    print(json.dumps(decision, ensure_ascii=False, indent=2, default=_json_default))
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
