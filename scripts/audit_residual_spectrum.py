"""FR-6.1 (DESIGN S10 row 2, [计划] S3.3 Stage S0): residual spectrum audit.

With the frozen strong mean mu_phi: r0 = y - mu(x) -> two-level orthonormal
Haar ([计划] S3.2) -> per-stratum band energies and coverage, for s in
{small_lesion, non_small_lesion, area_matched_background, whole} and
B = B_detail (the 6 detail bands, [裁决] S4 scope B) plus a with-LL2 control:

    E_{b,s} = E_i || M_{i,s} . recon_b(r0_i) ||^2             ([计划] S3.3)
    coverage_{B,s} = sum_{b in B} E_{b,s} / (sum_{b in B_cand} E_{b,s} + eps)

Gate ([裁决] S4): coverage(B_detail, small_lesion) >= floor AND
coverage(B_detail, whole) >= floor; floor = 0.60 is the preregistered
ENGINEERING floor, not a theoretical constant ([计划] S3.3), reported with
0.50/0.70 sensitivity. Any miss, or an empty small-lesion stratum ->
gate=false and ll2_must_enter_r0=true ("LL2 必须进入 R0 候选"); the verdict
lives in the JSON and the process still exits 0 (DESIGN S10).

口径 (documented; cross-check docs/残差均值诊断_能量分解.md): energies are
measured in the IMAGE domain on the orthogonal per-band synthesis
recon_b = haar_inverse2(band b only) - recon_LL2 vs the rest reproduces that
report's historical low/detail split, and with the all-ones mask E_{b,whole}
equals the band coefficient energy (Parseval); psd[b] = E_{b,whole}/n_b;
noise_floor = median over items of the per-coefficient HH1 mean-square
energy; var_ratio = Var(r0)/Var(y) ([审计] S3). small-lesion =
4-connected components with area <= q25_train, q25 = percentile(train
component areas, 25) frozen from TRAIN masks only (no train masks ->
ValueError, [计划] S5.3 leakage line). area-matched background: per
component (area A, raster order), side r = ceil(sqrt(A)); a seeded center is
drawn among positions whose r x r square (top-left = center - r//2) is
in-image and lesion-free; the ROI is its A pixels nearest the center
(Euclidean, raster ties); ROIs merge as a pixel union.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model.mean_predictor import FullImagePETPredictor  # noqa: E402
from src.model.rc_brd import (  # noqa: E402
    BAND_NAMES,
    DETAIL_BANDS,
    haar_forward2,
    haar_inverse2,
    mean_weights_sha256,
)

SCHEMA_VERSION = 1
STRATA = ("small_lesion", "non_small_lesion", "area_matched_background", "whole")
GATED_STRATA = ("small_lesion", "whole")  # [裁决] S4: the two gated coverages
DETAIL_SCOPE = "B_detail"                 # scope B of [裁决] S4 (6 detail bands)
COVERAGE_FLOOR = 0.60                     # preregistered engineering floor, [计划] S3.3
SENSITIVITY_FLOORS = (0.50, 0.60, 0.70)   # [计划] S3.3 sensitivity ladder
COVERAGE_EPS = 1e-12                      # +eps of the coverage denominator
Q25_PERCENTILE = 25.0                     # [计划] S5.3 small-lesion threshold
TRAIN_SPLIT, BATCH_SIZE = "train", 8  # split tag / CPU batch
GATE_RULE = ("coverage(B_detail, small_lesion) >= floor AND coverage(B_detail, whole) >= floor "
             "-> gate=true; floor=0.60 is the preregistered ENGINEERING floor, not a theoretical "
             "constant ([计划] S3.3); any miss, or an empty small-lesion stratum, -> gate=false "
             "and ll2_must_enter_r0=true: LL2 必须进入 R0 候选 ([裁决] S4 fail-closed); the "
             "gate verdict lives in this JSON and the process still exits 0 (DESIGN S10)")
MeanFn = Callable[[torch.Tensor], torch.Tensor]

@dataclass(frozen=True)
class AuditItem:
    """One audited slice: 2D float arrays plus split/patient identity."""

    item_id: str
    patient_id: str
    split: str
    ct: np.ndarray
    pet: np.ndarray
    mask: np.ndarray
# --- Data sources (injectable; synthetic is the exact-oracle test channel) ---

def _disk(rr: np.ndarray, cc: np.ndarray, center: tuple[int, int], area: int) -> np.ndarray:
    """Boolean disk of exactly area pixels nearest to center (raster ties)."""
    dist = (rr - center[0]) ** 2 + (cc - center[1]) ** 2
    selected = np.unravel_index(np.argsort(dist.ravel(), kind="stable")[:area], dist.shape)
    out = np.zeros(dist.shape, dtype=bool)
    out[selected] = True
    return out
def make_synthetic_items(spec: dict[str, Any]) -> list[AuditItem]:
    """Seeded items with an exactly known band mix (test oracle): pet = ll2
    amplitude * ones + hh1 amplitude * (2x2 checkerboard) + boost * lesion +
    gradient * row/H; the constant term is purely LL2 and the checkerboard
    purely HH1, so with the identity mean and ct_amplitude=0: coverage(B_detail,
    s) = hh1^2/(ll2^2+hh1^2) for EVERY stratum, psd LL2 = 16*ll2^2,
    psd/noise-floor HH1 = 4*hh1^2, peak [ll2-hh1, ll2+hh1]; gradient/ct
    amplitude > 0 breaks the closed form."""
    rng = np.random.default_rng(int(spec.get("seed", 0)))
    size = int(spec.get("image_size", 64))
    if size % 4:
        raise ValueError(f"image_size must be divisible by 4, got {size}")
    a_low, a_det = float(spec.get("ll2_amplitude", 0.5)), float(spec.get("hh1_amplitude", 0.5))
    boost, grad = float(spec.get("lesion_boost", 0.0)), float(spec.get("gradient_amplitude", 0.0))
    rows, cols = np.indices((size, size))
    pet = (a_low + a_det * (1.0 - 2.0 * ((rows + cols) % 2.0))
           + grad * (rows / size)).astype(np.float32)
    ct = (float(spec.get("ct_amplitude", 0.0)) * rng.standard_normal((size, size))).astype(np.float32)
    items: list[AuditItem] = []
    plans = ((TRAIN_SPLIT, "n_train", "train_lesion_areas", (4, 9, 16, 25)),
             ("test", "n_test", "test_lesion_areas", (36, 49)))
    for split, n_key, area_key, defaults in plans:
        areas = [int(a) for a in spec.get(area_key, defaults)]
        for k in range(int(spec.get(n_key, 1))):
            mask = np.zeros((size, size), dtype=np.float32)
            for j, area in enumerate(areas):
                mask[_disk(rows, cols, (6 + 13 * (j % 4), 6 + 13 * (j // 4)), area)] = 1.0
            items.append(AuditItem(f"{split}_{k:02d}", f"synth_{split}_{k:02d}", split,
                                   ct, (pet + boost * mask).astype(np.float32), mask))
    return items
def _require_arrays(data: Any, path: Path) -> dict[str, np.ndarray]:
    """ct/pet/mask arrays of one npz item as [H,W] float32 (fail-closed)."""
    missing = [key for key in ("ct", "pet", "mask") if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing arrays {missing}")
    out: dict[str, np.ndarray] = {}
    for key in ("ct", "pet", "mask"):
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"{path}: array {key!r} must be [H,W] or [1,H,W], got {arr.shape}")
        out[key] = arr
    return out
def _npz_scalar(data: Any, key: str, default: str) -> str:
    if key not in data.files:
        return default
    return str(np.asarray(data[key]).reshape(()).item())
def load_npz_items(directory: Path) -> list[AuditItem]:
    """npz-dir: one *.npz per slice with ct/pet/mask ([H,W] or [1,H,W]);
    optional 0-d strings "split" (default "test"), "patient_id" (stem)."""
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise ValueError(f"no *.npz audit items in {directory}")
    items: list[AuditItem] = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            arrays = _require_arrays(data, path)
            split = _npz_scalar(data, "split", "test")
            patient = _npz_scalar(data, "patient_id", path.stem)
        items.append(AuditItem(path.stem, patient, split,
                               arrays["ct"], arrays["pet"], arrays["mask"]))
    return items
def load_cache_items(directory: Path) -> list[AuditItem]:
    """data-cache (cache/tensors layout): <stem>.npz with ct/pet/mask [1,H,W]
    plus <stem>_meta.json; split "train" stays train, val/test count as test."""
    files = sorted(Path(directory).glob("*.npz"))
    if not files:
        raise ValueError(f"no *.npz audit items in {directory}")
    items: list[AuditItem] = []
    for path in files:
        split, patient = "test", path.stem
        meta_path = path.with_name(f"{path.stem}_meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            split = TRAIN_SPLIT if str(meta.get("split")) == TRAIN_SPLIT else "test"
            patient = str(meta.get("patient_id", path.stem))
        with np.load(path, allow_pickle=False) as data:
            arrays = _require_arrays(data, path)
        items.append(AuditItem(path.stem, patient, split,
                               arrays["ct"], arrays["pet"], arrays["mask"]))
    return items
def _png_array(path: Path, image_size: int, *, mask: bool) -> np.ndarray:
    from PIL import Image  # lazy: same dependency as src/data/png_cache.py
    with Image.open(path) as img:
        if mask or img.mode not in {"L", "I", "I;16", "I;16B", "I;16L", "F"}:
            img = img.convert("L")
        if image_size > 0 and img.size != (image_size, image_size):
            resample = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
            img = img.resize((image_size, image_size), resample)
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if mask:
        return (arr > 0.0).astype(np.float32)  # png_cache mask rule
    unit = arr / 255.0 if arr.max() > 1.0 else arr  # png_cache "auto" unit range
    return (np.clip(unit, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)
def load_png_items(png_root: Path, image_size: int = 0) -> list[AuditItem]:
    """png-root ([计划] PNG route): <root>/<split>/ct/*.png + same-named pet
    (or pet_peizhuan) and label PNGs; "train" dir is train; patient=stem[:3]."""
    root = Path(png_root)
    if not root.is_dir():
        raise ValueError(f"png root not found: {root}")
    items: list[AuditItem] = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        split = TRAIN_SPLIT if split_dir.name == TRAIN_SPLIT else "test"
        pet_dir = next((split_dir / d for d in ("pet", "pet_peizhuan")
                       if (split_dir / d).is_dir()), None)
        if pet_dir is None:
            raise ValueError(f"{split_dir}: neither pet/ nor pet_peizhuan/ exists")
        for ct_path in sorted((split_dir / "ct").glob("*.png")):
            pet_path, label_path = pet_dir / ct_path.name, split_dir / "label" / ct_path.name
            if not (pet_path.is_file() and label_path.is_file()):
                continue
            items.append(AuditItem(ct_path.stem, ct_path.stem[:3], split,
                                   _png_array(ct_path, image_size, mask=False),
                                   _png_array(pet_path, image_size, mask=False),
                                   _png_array(label_path, image_size, mask=True)))
    if not items:
        raise ValueError(f"no aligned ct/pet/label PNG triples under {root}")
    return items
# --- Strata: components, frozen q25, area-matched background ([计划] S5.3/S3.3) ---

def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Boolean 4-connected components of a 2D binary mask, raster order."""
    active = mask > 0.5
    visited = np.zeros(active.shape, dtype=bool)
    height, width = active.shape
    components: list[np.ndarray] = []
    for seed_row, seed_col in zip(*np.nonzero(active)):
        if visited[seed_row, seed_col]:
            continue
        visited[seed_row, seed_col] = True
        stack, pixels = [(int(seed_row), int(seed_col))], []
        while stack:
            r, c = stack.pop()
            pixels.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < height and 0 <= nc < width and active[nr, nc]
                        and not visited[nr, nc]):
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        component = np.zeros(active.shape, dtype=bool)
        component[tuple(zip(*pixels))] = True
        components.append(component)
    return components
def freeze_q25(items: Sequence[AuditItem]) -> float:
    """q25 = percentile(train component areas, 25); TRAIN masks only
    ([计划] S5.3); no train components -> ValueError (leakage fail-closed)."""
    areas: list[int] = []
    for item in items:
        if item.split == TRAIN_SPLIT:
            areas.extend(int(comp.sum()) for comp in connected_components(item.mask))
    if not areas:
        raise ValueError("q25 cannot be frozen: no train-split lesion masks provided; "
                         "using test masks would break the [计划] S5.3 leakage rule")
    return float(np.percentile(np.asarray(areas, dtype=float), Q25_PERCENTILE))
def _sample_area_matched_roi(lesion: np.ndarray, area: int,
                             rng: np.random.Generator) -> np.ndarray:
    """One seeded area-matched background ROI (rule in the module docstring)."""
    side = int(np.ceil(np.sqrt(area)))
    height, width = lesion.shape
    pooled = torch.nn.functional.max_pool2d(  # centers whose square touches a lesion pixel
        torch.from_numpy(lesion.astype(np.float32))[None, None], side, stride=1,
        padding=side // 2)[0, 0].numpy()[:height, :width] > 0.5
    lo = side // 2
    valid = np.zeros(lesion.shape, dtype=bool)
    valid[lo:height - side + lo + 1, lo:width - side + lo + 1] = True  # in-bounds square
    valid &= ~pooled
    candidates = np.argwhere(valid)  # raster order
    roi = np.zeros(lesion.shape, dtype=bool)
    if candidates.size == 0:
        return roi  # lesion too large to mirror: skipped (documented union rule)
    ci, cj = (int(v) for v in candidates[int(rng.integers(len(candidates)))])
    rows, cols = np.meshgrid(np.arange(ci - lo, ci - lo + side),
                             np.arange(cj - lo, cj - lo + side), indexing="ij")
    dist = (rows - ci) ** 2 + (cols - cj) ** 2
    selected = np.unravel_index(np.argsort(dist.ravel(), kind="stable")[:area], dist.shape)
    roi[selected[0] + ci - lo, selected[1] + cj - lo] = True  # offset into the image
    return roi
def build_stratum_masks(item: AuditItem, q25: float,
                        rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Image-domain stratum masks of one item ([计划] S3.3 strata)."""
    lesion = item.mask > 0.5
    small = np.zeros(lesion.shape, dtype=bool)
    large = np.zeros(lesion.shape, dtype=bool)
    background = np.zeros(lesion.shape, dtype=bool)
    for component in connected_components(lesion):
        if int(component.sum()) <= q25:  # [计划] S5.3: A_l <= q25_train
            small |= component
        else:
            large |= component
        background |= _sample_area_matched_roi(lesion, int(component.sum()), rng)
    return {"small_lesion": small, "non_small_lesion": large,
            "area_matched_background": background}

# --- Means (identity test channel; checkpoint channel mirrors slmf_bbdm) ---

def identity_mean(ct: torch.Tensor) -> torch.Tensor:
    """mu(x) = x (synthetic-identity test mean)."""
    return ct
def load_mean_checkpoint(path: Path) -> tuple[MeanFn, dict[str, Any]]:
    """FullImagePETPredictor mean from mean/comparison/full_model state dicts,
    mirroring slmf_bbdm conditional-mean loading without importing it: {"model":
    state, "format_version" in (1,2)} -> mean; raw state dict as-is; uniform
    "generator."/"mean_predictor." prefix stripped; mismatch fails closed."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get("model"), dict) else checkpoint
    if not (isinstance(state, dict) and state
            and all(isinstance(v, torch.Tensor) for v in state.values())):
        raise ValueError(f"mean checkpoint {path} is not a tensor state dict")
    fmt = "raw"
    for prefix, name in (("generator.", "comparison"), ("mean_predictor.", "full_model")):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: v for key, v in state.items()}
            fmt = name
            break
    if fmt == "raw" and isinstance(checkpoint, dict) and "model" in checkpoint:
        fmt = "mean"
        if checkpoint.get("format_version") not in (1, 2):
            raise ValueError("mean checkpoint format_version must be 1 or 2 (slmf_bbdm semantics)")
    first = state.get("down1.block.0.weight")
    if not isinstance(first, torch.Tensor) or first.ndim != 4:
        raise ValueError(f"mean state dict of {path} lacks down1.block.0.weight "
                         "(FullImagePETPredictor layout)")
    model = FullImagePETPredictor(in_channels=int(first.shape[1]),
                                  base_channels=int(first.shape[0]))
    model.load_state_dict(state, strict=True)
    model.eval()

    def mean_fn(ct: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(ct)["mean_pet"]

    return mean_fn, {"kind": "checkpoint", "checkpoint": str(path), "format": fmt,
                     "state_sha256": mean_weights_sha256(state)}
# --- Core audit ([计划] S3.3 E_{b,s}/coverage; [裁决] S4 gate) ---
def _pop_variance(acc: list[float]) -> float:
    # population variance from (sum, sum-of-squares, count) accumulators
    if acc[2] == 0.0:
        return 0.0
    mean = acc[0] / acc[2]
    return max(acc[1] / acc[2] - mean * mean, 0.0)
def _update_peak(peak: list, tensor: torch.Tensor, mask: torch.Tensor) -> None:
    values = tensor[mask]
    if values.numel():
        low, high = float(values.min().item()), float(values.max().item())
        peak[0] = low if peak[0] is None else min(peak[0], low)
        peak[1] = high if peak[1] is None else max(peak[1], high)
def _gate_at(cov: dict[str, float | None], floor: float) -> tuple[bool, bool, list[str]]:
    """[裁决] S4 rule at one floor -> (gate, ll2_must_enter_r0, empty strata)."""
    empty = [s for s in GATED_STRATA if cov.get(s) is None]
    gate = all(cov.get(s) is not None and cov[s] >= floor for s in GATED_STRATA)
    return gate, (not gate), empty
def _sensitivity_at(cov: dict[str, float | None], floor: float) -> dict[str, Any]:
    gate, ll2, empty = _gate_at(cov, floor)
    values = [cov[s] for s in GATED_STRATA if cov.get(s) is not None]
    return {"floor": floor, "gate": gate, "ll2_must_enter_r0": ll2,
            "coverage_min": min(values) if values else None, "empty_strata": empty}
def compute_residual_spectrum(items: Sequence[AuditItem], mean_fn: MeanFn, *,
                              seed: int = 0, batch_size: int = BATCH_SIZE) -> dict[str, Any]:
    """S0 core: items + mean_fn -> spectrum report ([计划] S3.3; [裁决] S4)."""
    q25 = freeze_q25(items)
    rng = np.random.default_rng(int(seed))
    energy = {b: {s: 0.0 for s in STRATA} for b in BAND_NAMES}
    counts = {s: 0 for s in STRATA}
    n_coef = {b: 0 for b in BAND_NAMES}
    hh1_items: list[float] = []
    acc_r0: list[float] = [0.0, 0.0, 0.0]
    acc_pet: list[float] = [0.0, 0.0, 0.0]
    peak = {k: [None, None] for k in ("all_lesion", "small_lesion", "non_small_lesion")}
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        chunk = items[start:start + step]
        strata = [build_stratum_masks(it, q25, rng) for it in chunk]
        ct = torch.from_numpy(np.stack([it.ct for it in chunk])).float()[:, None]
        pet = torch.from_numpy(np.stack([it.pet for it in chunk])).float()[:, None]
        residual = pet - mean_fn(ct)  # r0 = y - mu(x), [计划] S3.1
        bands = haar_forward2(residual)
        masks = {s: torch.from_numpy(np.stack([st[s] for st in strata]).astype(np.float32))[:, None]
                 for s in STRATA[:-1]}
        counts["whole"] += residual.numel()
        for s, m in masks.items():
            counts[s] += int(m.sum().item())
        for band in BAND_NAMES:
            solo = {n: (bands[n] if n == band else torch.zeros_like(bands[n]))
                    for n in BAND_NAMES}
            recon = haar_inverse2(solo)  # orthogonal image-domain band contribution
            n_coef[band] += bands[band].numel()
            squared = recon.pow(2)
            energy[band]["whole"] += float(squared.sum().item())
            if band == "HH1":  # noise floor: per-coefficient HH1 power, per item
                n_b1 = (recon.shape[-2] // 2) * (recon.shape[-1] // 2)
                hh1_items.extend((squared.sum(dim=(1, 2, 3)) / n_b1).tolist())
            for s, m in masks.items():
                energy[band][s] += float(squared.mul(m).sum().item())
        for acc, t in ((acc_r0, residual), (acc_pet, pet)):
            acc[0], acc[2] = acc[0] + float(t.sum().item()), acc[2] + t.numel()
            acc[1] += float(t.pow(2).sum().item())
        small, large = masks["small_lesion"].bool(), masks["non_small_lesion"].bool()
        _update_peak(peak["small_lesion"], residual, small)
        _update_peak(peak["non_small_lesion"], residual, large)
        _update_peak(peak["all_lesion"], residual, small | large)
    psd = {b: energy[b]["whole"] / n_coef[b] for b in BAND_NAMES}
    var_pet = _pop_variance(acc_pet)
    var_ratio = _pop_variance(acc_r0) / var_pet if var_pet > 0.0 else None
    coverage: dict[str, dict[str, Any]] = {}
    for scope, band_set in ((DETAIL_SCOPE, DETAIL_BANDS), ("with_ll2", BAND_NAMES)):
        coverage[scope] = {}
        for s in STRATA:
            if counts[s] == 0:
                coverage[scope][s] = None  # empty stratum: undefined, fails closed
            else:
                num = sum(energy[b][s] for b in band_set)
                den = sum(energy[b][s] for b in BAND_NAMES) + COVERAGE_EPS
                coverage[scope][s] = num / den
    gate, ll2, empty = _gate_at(coverage[DETAIL_SCOPE], COVERAGE_FLOOR)
    sensitivity = {f"{fl:.2f}": _sensitivity_at(coverage[DETAIL_SCOPE], fl)
                   for fl in SENSITIVITY_FLOORS}
    peaks = {k: ({"min": v[0], "max": v[1]} if v[0] is not None else None)
             for k, v in peak.items()}
    return {
        "schema_version": SCHEMA_VERSION, "gate": gate, "ll2_must_enter_r0": ll2,
        "floor": COVERAGE_FLOOR, "gate_rule": GATE_RULE, "gate_empty_strata": empty,
        "coverage": {DETAIL_SCOPE: coverage[DETAIL_SCOPE], "with_ll2": coverage["with_ll2"],
                     "B_detail_bands": list(DETAIL_BANDS), "with_ll2_bands": list(BAND_NAMES),
                     "q25_train_pixels": q25, "epsilon": COVERAGE_EPS,
                     "stratum_pixel_counts": dict(counts),
                     "per_band_energy": {b: dict(energy[b]) for b in BAND_NAMES}},
        "sensitivity": sensitivity, "var_ratio": var_ratio, "psd": psd,
        "psd_basis": "mean-square energy per Haar coefficient (whole-image, Parseval)",
        "noise_floor": {"band": "HH1", "value": float(np.median(hh1_items)) if hh1_items else None,
                        "proxy": "median over items of per-coefficient HH1 mean-square energy"},
        "lesion_peak_residual_range": peaks,
        "n_items": {"total": len(items),
                    "train": sum(1 for it in items if it.split == TRAIN_SPLIT),
                    "test": sum(1 for it in items if it.split != TRAIN_SPLIT)},
        "image_size": [int(items[0].ct.shape[0]), int(items[0].ct.shape[1])],
        "seed": int(seed)}
# --- Orchestration + CLI (DESIGN S10 row 2) ---
def _slug(fold: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(fold)) or "0"
def run_audit(*, source: str = "synthetic", synthetic_spec: str = "",
              npz_dir: str | None = None, data_cache: str | None = None,
              png_root: str | None = None, mean: str | None = None,
              mean_checkpoint: str | None = None, fold: str = "0",
              out: str = "results", seed: int = 0) -> int:
    """Load items + mean, audit, write JSON; a failed gate still returns 0."""
    if source == "synthetic":
        spec = json.loads(synthetic_spec) if synthetic_spec.strip() else {}
        items, source_info = make_synthetic_items(spec), {"source": "synthetic", "spec": spec}
    else:
        loaders = {"npz-dir": (load_npz_items, npz_dir),
                   "data-cache": (load_cache_items, data_cache),
                   "png-root": (load_png_items, png_root)}
        if source not in loaders:
            raise ValueError(f"unknown source {source!r} (synthetic|npz-dir|data-cache|png-root)")
        loader, value = loaders[source]
        if not value:
            raise ValueError(f"--{source} is required with --source {source}")
        items, source_info = loader(Path(value)), {"source": source, "path": value}
    if mean_checkpoint:
        mean_fn, mean_info = load_mean_checkpoint(Path(mean_checkpoint))
    elif mean == "synthetic-identity":
        mean_fn, mean_info = identity_mean, {"kind": "synthetic-identity"}
    else:
        raise ValueError("pass --mean synthetic-identity or --mean-checkpoint PATH")
    report = compute_residual_spectrum(items, mean_fn, seed=seed)
    report["fold"], report["source"], report["mean"] = str(fold), source_info, mean_info
    out_path = Path(out)
    if out_path.suffix.lower() != ".json":  # directory mode -> DESIGN S10 file name
        out_path = out_path / f"residual_spectrum_fold{_slug(fold)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"residual spectrum fold={fold} gate={report['gate']} "
          f"ll2_must_enter_r0={report['ll2_must_enter_r0']} "
          f"coverage_detail_whole={report['coverage'][DETAIL_SCOPE]['whole']} -> {out_path}")
    return 0
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="S0 residual spectrum audit ([计划] S3.3, [裁决] S4; FR-6.1)")
    parser.add_argument("--source", choices=("synthetic", "npz-dir", "data-cache", "png-root"),
                        default="synthetic")
    parser.add_argument("--synthetic-spec", default="",
                        help="JSON spec for --source synthetic (see make_synthetic_items)")
    for flag in ("--npz-dir", "--data-cache", "--png-root"):
        parser.add_argument(flag)
    mean_group = parser.add_mutually_exclusive_group(required=True)
    mean_group.add_argument("--mean", help="synthetic-identity identity test mean")
    mean_group.add_argument("--mean-checkpoint", help="FullImagePETPredictor mean weights")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--out", required=True, help="output JSON file or directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        return run_audit(source=args.source, synthetic_spec=args.synthetic_spec,
                         npz_dir=args.npz_dir, data_cache=args.data_cache,
                         png_root=args.png_root, mean=args.mean,
                         mean_checkpoint=args.mean_checkpoint, fold=args.fold,
                         out=args.out, seed=args.seed)
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"residual spectrum audit failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
