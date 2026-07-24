"""Patient-level probe: can CT frequency bands localize the annotated PET lesion?

The probe is deliberately lightweight.  It is not a segmentation benchmark and
does not tune anything on the repository validation patients.  A deterministic
subset of the training patients is held out for threshold calibration, while
``main_data/split_manifest.csv`` remains the authoritative train/validation
split.  The physical train/val folders under ``Data/data`` are searched only as
file stores because their older split assignment does not match that manifest.

Outputs are written to ``results/ct_frequency_lesion_probe``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class Sample:
    sample_id: str
    patient_id: str
    split: str
    ct: np.ndarray  # float16 [H,W], converted to float32 when used
    mask: np.ndarray  # bool [H,W]
    body: np.ndarray  # bool [H,W]


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


def _read_ct(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _read_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.NEAREST)
        return np.asarray(image, dtype=np.uint8) > 127


def _read_pet_uptake(path: Path, size: int) -> np.ndarray:
    # Data/data PET PNGs use a white canvas and dark uptake, hence the inversion.
    with Image.open(path) as image:
        image = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return 1.0 - np.asarray(image, dtype=np.float32) / 255.0


def _largest_body(ct: np.ndarray) -> np.ndarray:
    foreground = ct > 0.025
    foreground = ndimage.binary_closing(foreground, iterations=2)
    labels, count = ndimage.label(foreground)
    if count == 0:
        return np.ones_like(foreground, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    body = labels == int(sizes.argmax())
    body = ndimage.binary_fill_holes(body)
    body = ndimage.binary_dilation(body, iterations=2)
    return body.astype(bool)


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


def _spectral_maps(ct: np.ndarray) -> dict[str, np.ndarray]:
    size = int(ct.shape[0])
    ll1, d1 = _haar2(ct)
    ll2, d2 = _haar2(ll1)
    maps: dict[str, np.ndarray] = {
        "raw": ct.astype(np.float32),
        "ll2": _resize_float(ll2, size),
    }
    for level, details in (("l1", d1), ("l2", d2)):
        for band, value in zip(("lh", "hl", "hh"), details, strict=True):
            signed = _resize_float(value, size)
            energy = _resize_float(
                ndimage.uniform_filter(np.abs(value), size=3, mode="nearest"), size
            )
            maps[f"{level}_{band}"] = signed
            maps[f"{level}_{band}_energy"] = energy
    return maps


def _coords(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx = xx / max(size - 1, 1) * 2.0 - 1.0
    yy = yy / max(size - 1, 1) * 2.0 - 1.0
    return np.stack((xx, yy, xx * xx, yy * yy, xx * yy), axis=-1)


def _feature_array(kind: str, ct: np.ndarray, coord: np.ndarray) -> np.ndarray:
    maps = _spectral_maps(ct)
    l1 = [maps[f"l1_{b}"] for b in ("lh", "hl", "hh")]
    l1e = [maps[f"l1_{b}_energy"] for b in ("lh", "hl", "hh")]
    l2 = [maps[f"l2_{b}"] for b in ("lh", "hl", "hh")]
    l2e = [maps[f"l2_{b}_energy"] for b in ("lh", "hl", "hh")]
    if kind == "coords":
        return coord
    if kind == "raw_ct":
        return maps["raw"][..., None]
    if kind == "local_patch":
        # Capacity-matched spatial control: raw CT samples around each pixel,
        # without an explicit frequency transform.
        offsets = (
            (0, 0),
            (-2, 0), (2, 0), (0, -2), (0, 2),
            (-4, 0), (4, 0), (0, -4), (0, 4),
            (-3, -3), (-3, 3), (3, -3), (3, 3),
        )
        pad = 4
        padded = np.pad(ct, pad, mode="reflect")
        height, width = ct.shape
        patches = [
            padded[pad + dy : pad + dy + height, pad + dx : pad + dx + width]
            for dy, dx in offsets
        ]
        return np.stack(patches, axis=-1).astype(np.float32)
    if kind == "ll2":
        return maps["ll2"][..., None]
    if kind == "l2_details":
        return np.stack(l2 + l2e, axis=-1)
    if kind == "l1_details":
        return np.stack(l1 + l1e, axis=-1)
    all_frequency = np.stack([maps["ll2"], *l2, *l2e, *l1, *l1e], axis=-1)
    if kind == "all_frequency":
        return all_frequency
    if kind == "coords_all_frequency":
        return np.concatenate((coord, all_frequency), axis=-1)
    raise KeyError(kind)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return (
        float(average_precision_score(y, score))
        if np.unique(y).size == 2
        else float("nan")
    )


def _ring(mask: np.ndarray, body: np.ndarray, radius: int = 12) -> np.ndarray:
    return ndimage.binary_dilation(mask, iterations=radius) & ~mask & body


def _sample_training_indices(
    sample: Sample, rng: np.random.Generator, max_positive: int = 160
) -> np.ndarray:
    positive = np.flatnonzero(sample.mask.ravel())
    if positive.size > max_positive:
        positive = rng.choice(positive, max_positive, replace=False)
    ring = np.flatnonzero(_ring(sample.mask, sample.body).ravel())
    other = np.flatnonzero((sample.body & ~sample.mask & ~_ring(sample.mask, sample.body)).ravel())
    target_negative = max(3 * positive.size, 1)
    ring_n = min(ring.size, target_negative // 2)
    other_n = min(other.size, target_negative - ring_n)
    neg_parts = []
    if ring_n:
        neg_parts.append(rng.choice(ring, ring_n, replace=False))
    if other_n:
        neg_parts.append(rng.choice(other, other_n, replace=False))
    return np.concatenate((positive, *neg_parts)).astype(np.int64)


def _fit_probe(
    kind: str,
    fit_samples: list[Sample],
    train_indices: dict[str, np.ndarray],
    coord: np.ndarray,
    seed: int,
) -> HistGradientBoostingClassifier:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sample in fit_samples:
        feature = _feature_array(kind, sample.ct.astype(np.float32), coord)
        index = train_indices[sample.sample_id]
        xs.append(feature.reshape(-1, feature.shape[-1])[index])
        ys.append(sample.mask.ravel()[index].astype(np.uint8))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    positive_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    weight = np.where(y == 1, positive_weight, 1.0)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=seed,
    )
    model.fit(x, y, sample_weight=weight)
    return model


def _predict_probe(
    model: HistGradientBoostingClassifier,
    kind: str,
    sample: Sample,
    coord: np.ndarray,
) -> np.ndarray:
    feature = _feature_array(kind, sample.ct.astype(np.float32), coord)
    score = model.predict_proba(feature.reshape(-1, feature.shape[-1]))[:, 1]
    return score.reshape(sample.mask.shape).astype(np.float32)


def _calibrate_threshold(
    samples: list[Sample], predict: Callable[[Sample], np.ndarray]
) -> tuple[float, float]:
    ys: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for sample in samples:
        score = predict(sample)
        roi = sample.body
        ys.append(sample.mask[roi])
        scores.append(score[roi])
    y = np.concatenate(ys)
    score = np.concatenate(scores)
    quantiles = np.linspace(0.90, 0.99995, 240)
    candidates = np.unique(np.quantile(score, quantiles))
    best_threshold, best_dice = 0.5, -1.0
    positive = int(y.sum())
    for threshold in candidates:
        pred = score >= threshold
        tp = int(np.logical_and(pred, y).sum())
        dice = 2.0 * tp / max(int(pred.sum()) + positive, 1)
        if dice > best_dice:
            best_threshold, best_dice = float(threshold), float(dice)
    return best_threshold, best_dice


def _evaluate(
    samples: list[Sample],
    predict: Callable[[Sample], np.ndarray],
    threshold: float,
    size_thresholds: tuple[float, float],
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    full_y: list[np.ndarray] = []
    full_score: list[np.ndarray] = []
    ring_y: list[np.ndarray] = []
    ring_score: list[np.ndarray] = []
    topk_dice: list[float] = []
    patient: dict[str, dict[str, list[np.ndarray]]] = {}
    strata: dict[str, dict[str, list[np.ndarray] | list[float]]] = {
        name: {"y": [], "s": [], "ry": [], "rs": [], "topk": []}
        for name in ("small", "medium", "large")
    }

    for sample in samples:
        score = predict(sample)
        roi = sample.body
        y = sample.mask[roi]
        s = score[roi]
        full_y.append(y)
        full_score.append(s)

        ring_roi = sample.mask | _ring(sample.mask, sample.body)
        ring_y.append(sample.mask[ring_roi])
        ring_score.append(score[ring_roi])

        candidate = np.flatnonzero(roi.ravel())
        k = int(sample.mask.sum())
        if k > 0 and candidate.size >= k:
            chosen_local = np.argpartition(score.ravel()[candidate], -k)[-k:]
            chosen = candidate[chosen_local]
            overlap = int(sample.mask.ravel()[chosen].sum())
            topk_dice.append(float(overlap / k))
            sample_topk = float(overlap / k)
        else:
            sample_topk = float("nan")

        area = float(sample.mask.sum())
        if area <= size_thresholds[0]:
            stratum = "small"
        elif area <= size_thresholds[1]:
            stratum = "medium"
        else:
            stratum = "large"
        strata[stratum]["y"].append(y)
        strata[stratum]["s"].append(s)
        strata[stratum]["ry"].append(sample.mask[ring_roi])
        strata[stratum]["rs"].append(score[ring_roi])
        strata[stratum]["topk"].append(sample_topk)

        bucket = patient.setdefault(sample.patient_id, {"y": [], "s": [], "ry": [], "rs": []})
        bucket["y"].append(y)
        bucket["s"].append(s)
        bucket["ry"].append(sample.mask[ring_roi])
        bucket["rs"].append(score[ring_roi])

    y = np.concatenate(full_y)
    score = np.concatenate(full_score)
    pred = score >= threshold
    tp = int(np.logical_and(pred, y).sum())
    fp = int(np.logical_and(pred, ~y).sum())
    fn = int(np.logical_and(~pred, y).sum())
    patient_ap = []
    patient_ring_auc = []
    for bucket in patient.values():
        py, ps = np.concatenate(bucket["y"]), np.concatenate(bucket["s"])
        pry, prs = np.concatenate(bucket["ry"]), np.concatenate(bucket["rs"])
        patient_ap.append(_safe_ap(py, ps))
        patient_ring_auc.append(_safe_auc(pry, prs))
    overall = {
        "pixel_auroc": _safe_auc(y, score),
        "pixel_ap": _safe_ap(y, score),
        "pixel_prevalence": float(y.mean()),
        "dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "ring_auroc": _safe_auc(np.concatenate(ring_y), np.concatenate(ring_score)),
        "ring_ap": _safe_ap(np.concatenate(ring_y), np.concatenate(ring_score)),
        "topk_dice_mean": float(np.mean(topk_dice)),
        "patient_macro_ap": float(np.nanmean(patient_ap)),
        "patient_macro_ring_auroc": float(np.nanmean(patient_ring_auc)),
    }
    stratum_rows: list[dict[str, float | str]] = []
    for name, bucket in strata.items():
        sy, ss = np.concatenate(bucket["y"]), np.concatenate(bucket["s"])
        sry, srs = np.concatenate(bucket["ry"]), np.concatenate(bucket["rs"])
        spred = ss >= threshold
        stp = int(np.logical_and(spred, sy).sum())
        sfp = int(np.logical_and(spred, ~sy).sum())
        sfn = int(np.logical_and(~spred, sy).sum())
        stratum_rows.append(
            {
                "size_stratum": name,
                "slices": int(len(bucket["y"])),
                "pixel_ap": _safe_ap(sy, ss),
                "dice": float(2 * stp / max(2 * stp + sfp + sfn, 1)),
                "ring_auroc": _safe_auc(sry, srs),
                "ring_ap": _safe_ap(sry, srs),
                "topk_dice_mean": float(np.nanmean(bucket["topk"])),
            }
        )
    return overall, stratum_rows


def _band_energy_diagnostics(
    fit_samples: list[Sample], val_samples: list[Sample]
) -> list[dict[str, float | str]]:
    bands = [
        "ll2",
        "l2_lh_energy",
        "l2_hl_energy",
        "l2_hh_energy",
        "l1_lh_energy",
        "l1_hl_energy",
        "l1_hh_energy",
    ]

    def collect(samples: list[Sample], band: str, limit: int) -> tuple[np.ndarray, np.ndarray]:
        ys: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for sample in samples:
            value = _spectral_maps(sample.ct.astype(np.float32))[band]
            roi = sample.mask | _ring(sample.mask, sample.body)
            index = np.flatnonzero(roi.ravel())
            if index.size > limit:
                # Deterministic spatial thinning, independent of scores and labels.
                index = index[np.linspace(0, index.size - 1, limit).astype(int)]
            ys.append(sample.mask.ravel()[index])
            scores.append(value.ravel()[index])
        return np.concatenate(ys), np.concatenate(scores)

    rows: list[dict[str, float | str]] = []
    for band in bands:
        fit_y, fit_score = collect(fit_samples, band, 1000)
        direction = 1.0 if _safe_auc(fit_y, fit_score) >= 0.5 else -1.0
        val_y, val_score = collect(val_samples, band, 2000)
        oriented = direction * val_score
        rows.append(
            {
                "band": band,
                "train_selected_direction": "+" if direction > 0 else "-",
                "val_ring_auroc": _safe_auc(val_y, oriented),
                "val_ring_ap": _safe_ap(val_y, oriented),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or root / "results" / "ct_frequency_lesion_probe"
    output.mkdir(parents=True, exist_ok=True)
    data_root = root / "Data" / "data"
    manifest_path = root / "main_data" / "split_manifest.csv"

    indices = {name: _index_pngs(data_root, name) for name in ("ct", "pet", "label")}
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    missing = {
        name: [row["sample_id"] for row in manifest if row["sample_id"] not in index]
        for name, index in indices.items()
    }
    if any(missing.values()):
        raise FileNotFoundError(f"missing PNGs relative to manifest: {missing}")

    samples: list[Sample] = []
    pet_ring_y: list[np.ndarray] = []
    pet_ring_score: list[np.ndarray] = []
    for row in manifest:
        sid = row["sample_id"]
        ct = _read_ct(indices["ct"][sid], args.size)
        mask = _read_mask(indices["label"][sid], args.size)
        body = _largest_body(ct)
        body |= mask
        samples.append(
            Sample(sid, row["patient_id"], row["split"], ct.astype(np.float16), mask, body)
        )
        if row["split"] == "val":
            pet = _read_pet_uptake(indices["pet"][sid], args.size)
            roi = mask | _ring(mask, body)
            pet_ring_y.append(mask[roi])
            pet_ring_score.append(pet[roi])

    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    train_patients = sorted({sample.patient_id for sample in train_samples})
    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(train_patients, dtype=object)
    rng.shuffle(shuffled)
    n_calibration = max(1, round(0.20 * len(shuffled)))
    calibration_patients = set(shuffled[:n_calibration].tolist())
    fit_samples = [sample for sample in train_samples if sample.patient_id not in calibration_patients]
    calibration_samples = [sample for sample in train_samples if sample.patient_id in calibration_patients]

    train_indices = {
        sample.sample_id: _sample_training_indices(sample, rng) for sample in fit_samples
    }
    coord = _coords(args.size)
    fit_areas = np.asarray([sample.mask.sum() for sample in fit_samples], dtype=np.float64)
    size_thresholds = (
        float(np.quantile(fit_areas, 1.0 / 3.0)),
        float(np.quantile(fit_areas, 2.0 / 3.0)),
    )
    size_results: list[dict[str, float | str]] = []

    # No-CT anatomical atlas baseline, fitted only on probe-fit patients.
    atlas = np.mean([sample.mask.astype(np.float32) for sample in fit_samples], axis=0)
    atlas = ndimage.gaussian_filter(atlas, sigma=3.0)
    atlas_predict = lambda sample: atlas  # noqa: E731
    atlas_threshold, atlas_calibration_dice = _calibrate_threshold(
        calibration_samples, atlas_predict
    )
    atlas_metrics, atlas_size_rows = _evaluate(
        val_samples, atlas_predict, atlas_threshold, size_thresholds
    )
    results: list[dict[str, float | str]] = [{
        "probe": "atlas_no_ct",
        "calibration_threshold": atlas_threshold,
        "calibration_dice": atlas_calibration_dice,
        **atlas_metrics,
    }]
    size_results.extend({"probe": "atlas_no_ct", **row} for row in atlas_size_rows)

    kinds = (
        "coords",
        "raw_ct",
        "local_patch",
        "ll2",
        "l2_details",
        "l1_details",
        "all_frequency",
        "coords_all_frequency",
    )
    for kind in kinds:
        print(f"[probe] fitting {kind}", flush=True)
        model = _fit_probe(kind, fit_samples, train_indices, coord, args.seed)
        predict = lambda sample, m=model, k=kind: _predict_probe(m, k, sample, coord)
        threshold, calibration_dice = _calibrate_threshold(calibration_samples, predict)
        metrics, size_rows = _evaluate(
            val_samples, predict, threshold, size_thresholds
        )
        row = {
            "probe": kind,
            "calibration_threshold": threshold,
            "calibration_dice": calibration_dice,
            **metrics,
        }
        results.append(row)
        size_results.extend({"probe": kind, **item} for item in size_rows)
        print(
            f"[probe] {kind}: AP={row['pixel_ap']:.4f} "
            f"ring_AUC={row['ring_auroc']:.4f} Dice={row['dice']:.4f}",
            flush=True,
        )

    band_rows = _band_energy_diagnostics(fit_samples, val_samples)
    resized_areas = np.asarray([sample.mask.sum() for sample in samples], dtype=np.float64)
    resized_fractions = resized_areas / float(args.size * args.size)
    summary = {
        "manifest": str(manifest_path),
        "image_size": args.size,
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(val_samples),
        "train_patients": len(train_patients),
        "probe_fit_patients": len({sample.patient_id for sample in fit_samples}),
        "probe_calibration_patients": len(calibration_patients),
        "validation_patients": len({sample.patient_id for sample in val_samples}),
        "mask_area_resized": {
            "min": float(resized_areas.min()),
            "q25": float(np.quantile(resized_areas, 0.25)),
            "median": float(np.median(resized_areas)),
            "q75": float(np.quantile(resized_areas, 0.75)),
            "max": float(resized_areas.max()),
            "median_fraction": float(np.median(resized_fractions)),
        },
        "probe_fit_area_tertile_thresholds": {
            "small_max": size_thresholds[0],
            "medium_max": size_thresholds[1],
        },
        "pet_inverted_intensity_val_ring_auroc": _safe_auc(
            np.concatenate(pet_ring_y), np.concatenate(pet_ring_score)
        ),
        "interpretation_guardrail": (
            "This is a supervised predictability probe on annotated tumor masks, not proof "
            "that CT frequency bands determine PET uptake or generalize externally."
        ),
    }

    fieldnames = list(results[0].keys())
    with (output / "probe_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with (output / "band_energy_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(band_rows[0].keys()))
        writer.writeheader()
        writer.writerows(band_rows)
    with (output / "probe_metrics_by_size.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(size_results[0].keys()))
        writer.writeheader()
        writer.writerows(size_results)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "probes": results,
                "probes_by_size": size_results,
                "bands": band_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[done] {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
