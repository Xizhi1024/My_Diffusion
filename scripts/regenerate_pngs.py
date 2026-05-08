"""
Regenerate PNGs from DICOM, ensuring perfect CT-PET-label alignment.

Algorithm:
  1. For each patient, read DICOMs from all 4 modality dirs.
  2. Match slices by z-coordinate (tolerance 0.01 mm).
  3. Only keep slices that have BOTH CT image AND PET image.
  4. Generate PNGs with windowed 8-bit values, named {patient_id}{idx:03d}.png.

Usage:
  python scripts/regenerate_pngs.py --data-root Data --split train
  python scripts/regenerate_pngs.py --data-root Data --split test
  python scripts/regenerate_pngs.py --data-root Data --split all
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pydicom
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
Z_TOLERANCE = 3.5  # mm — matching tolerance (just over 1 slice thickness, ~3.27mm)
CT_HU_MIN = -150.0
CT_HU_MAX = 250.0
PET_SUV_MAX = 20.0

_DICOM_EXTENSIONS = {".dcm", ".dicom"}

# ---------------------------------------------------------------------------
# DICOM helpers
# ---------------------------------------------------------------------------


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _read_z(ds) -> Optional[float]:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None:
        return None
    try:
        return float(ipp[2])
    except (TypeError, ValueError, IndexError):
        return None


def _read_dicom_pixels(path: str) -> Tuple[pydicom.Dataset, np.ndarray]:
    """Return (dataset, float32 pixels with Slope/Intercept applied)."""
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    if pixels.ndim == 3:
        if pixels.shape[0] == 1:
            pixels = pixels[0]
        elif pixels.shape[-1] == 1:
            pixels = pixels[..., 0]
        else:
            pixels = pixels[0]
    slope = _safe_float(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", None), 0.0)
    return ds, (pixels * slope + intercept).astype(np.float32)


# ---------------------------------------------------------------------------
# SUV computation (PET)
# ---------------------------------------------------------------------------


def _parse_dcm_datetime(date_str, time_str):
    from datetime import datetime

    if not date_str or not time_str:
        return None
    date_str, time_str = str(date_str).strip(), str(time_str).strip()
    if not date_str or len(date_str) != 8:
        return None
    time_main = time_str.split(".")[0].ljust(6, "0")
    try:
        return datetime.strptime(f"{date_str}{time_main[:6]}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _compute_suv(ds, activity_conc: np.ndarray) -> Tuple[np.ndarray, bool]:
    import math

    units = str(getattr(ds, "Units", "")).upper()
    if units.startswith("SUV") or units == "GML":
        return activity_conc, True

    if units in {"BQML", "BQCC", ""}:
        activity_bqml = activity_conc
    elif units in {"KBQML", "KBQCC"}:
        activity_bqml = activity_conc * 1000.0
    else:
        return activity_conc, False

    weight_kg = _safe_float(getattr(ds, "PatientWeight", None), -1.0)
    if weight_kg <= 0:
        return activity_conc, False

    rad_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not rad_seq:
        return activity_conc, False
    rad = rad_seq[0]

    total_dose = _safe_float(getattr(rad, "RadionuclideTotalDose", None), -1.0)
    if total_dose <= 0:
        return activity_conc, False

    half_life = _safe_float(getattr(rad, "RadionuclideHalfLife", None), -1.0)

    start_dt = None
    raw_start = getattr(rad, "RadiopharmaceuticalStartDateTime", None)
    if raw_start:
        start_dt = _parse_dcm_datetime(str(raw_start)[:8], str(raw_start)[8:])
    if start_dt is None:
        start_dt = _parse_dcm_datetime(
            getattr(ds, "SeriesDate", None) or getattr(ds, "AcquisitionDate", None),
            getattr(rad, "RadiopharmaceuticalStartTime", None),
        )

    acq_dt = None
    raw_acq = getattr(ds, "AcquisitionDateTime", None)
    if raw_acq:
        acq_dt = _parse_dcm_datetime(str(raw_acq)[:8], str(raw_acq)[8:])
    if acq_dt is None:
        acq_dt = _parse_dcm_datetime(
            getattr(ds, "AcquisitionDate", None) or getattr(ds, "SeriesDate", None),
            getattr(ds, "AcquisitionTime", None) or getattr(ds, "SeriesTime", None),
        )

    decay_seconds = 0.0
    if start_dt is not None and acq_dt is not None:
        decay_seconds = max(0.0, (acq_dt - start_dt).total_seconds())

    decayed_dose = total_dose
    if half_life > 0 and decay_seconds > 0:
        decayed_dose = total_dose * math.exp(-math.log(2.0) * decay_seconds / half_life)
    if decayed_dose <= 0:
        return activity_conc, False

    suv = activity_bqml * (weight_kg * 1000.0) / decayed_dose
    return suv.astype(np.float32), True


# ---------------------------------------------------------------------------
# PNG generation per modality
# ---------------------------------------------------------------------------


def _array_to_png_u8(arr: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    """Clip to [vmin, vmax], normalise to [0, 255], return 'L' mode PIL Image."""
    clipped = np.clip(arr, vmin, vmax)
    u8 = ((clipped - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    return Image.fromarray(u8, mode="L")


def _process_ct_image(dcm_path: str) -> Image.Image:
    _, hu = _read_dicom_pixels(dcm_path)
    return _array_to_png_u8(hu, CT_HU_MIN, CT_HU_MAX)


def _process_pet_image(dcm_path: str) -> Image.Image:
    ds, activity = _read_dicom_pixels(dcm_path)
    suv, suv_ok = _compute_suv(ds, activity)
    if suv_ok:
        values = suv
        vmin, vmax = 0.0, PET_SUV_MAX
    else:
        values = activity
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax - vmin < 1e-8:
            vmax = vmin + 1.0
    return _array_to_png_u8(values, vmin, vmax)


def _process_label(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[..., 0]
    # Label should be binary; threshold at 0.5
    binary = (arr > 0.5).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


# ---------------------------------------------------------------------------
# Core: scan DICOM directory → {patient_id: {z: path}}
# ---------------------------------------------------------------------------


def _scan_dcm_dir(root: Path) -> Dict[str, List[Tuple[float, str]]]:
    """Return {patient_id: [(z, full_dcm_path), ...]} sorted by z descending.

    Skips files without valid z-coordinate.
    """
    result: Dict[str, List[Tuple[float, str]]] = {}
    if not root.is_dir():
        return result

    for patient_dir in sorted(root.iterdir()):
        if not patient_dir.is_dir():
            continue
        pid = patient_dir.name
        entries: List[Tuple[float, str]] = []
        for entry in sorted(patient_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in _DICOM_EXTENSIONS:
                continue
            try:
                ds = pydicom.dcmread(str(entry), stop_before_pixels=True, force=True)
                z = _read_z(ds)
                if z is None:
                    continue
                entries.append((z, str(entry)))
            except Exception as exc:
                print(f"  WARNING: bad DICOM {entry}: {exc}")
        if entries:
            entries.sort(key=lambda x: x[0], reverse=True)  # z_desc
            result[pid] = entries
    return result


def _find_closest_z(
    target_z: float,
    candidates: List[Tuple[float, str]],
    tolerance: float,
) -> Optional[float]:
    """Return the z-value from *candidates* closest to *target_z* within *tolerance*."""
    best_z = None
    best_dist = float("inf")
    for z, _ in candidates:
        dist = abs(z - target_z)
        if dist < best_dist and dist <= tolerance:
            best_dist = dist
            best_z = z
    return best_z


# ---------------------------------------------------------------------------
# Main: match & generate
# ---------------------------------------------------------------------------


def regenerate_split(data_root: Path, split: str) -> dict:
    """Regenerate PNGs for one split ('train' or 'test').

    Returns stats dict.
    """
    sub = f"{split}_data"

    ct_img_dir = data_root / "part_CT" / sub / "ImageSet" / "DICOM"
    ct_lbl_dir = data_root / "part_CT" / sub / "LabelSet" / "DICOM"
    pet_img_dir = data_root / "part_PET" / sub / "ImageSet" / "DICOM"
    pet_lbl_dir = data_root / "part_PET" / sub / "LabelSet" / "DICOM"

    ct_img_out = data_root / "part_CT" / sub / "ImageSet" / "PNG"
    ct_lbl_out = data_root / "part_CT" / sub / "LabelSet" / "PNG"
    pet_img_out = data_root / "part_PET" / sub / "ImageSet" / "PNG"
    pet_lbl_out = data_root / "part_PET" / sub / "LabelSet" / "PNG"

    print(f"\n{'='*60}")
    print(f"Processing split: {split}")
    print(f"{'='*60}")

    print(f"Scanning DICOM directories ...")
    ct_img = _scan_dcm_dir(ct_img_dir)
    ct_lbl = _scan_dcm_dir(ct_lbl_dir)
    pet_img = _scan_dcm_dir(pet_img_dir)
    pet_lbl = _scan_dcm_dir(pet_lbl_dir)

    all_patients = sorted(
        set(ct_img) | set(ct_lbl) | set(pet_img) | set(pet_lbl)
    )
    print(f"  CT  ImageSet: {len(ct_img)} patients, {sum(len(m) for m in ct_img.values())} slices")
    print(f"  CT  LabelSet: {len(ct_lbl)} patients, {sum(len(m) for m in ct_lbl.values())} slices")
    print(f"  PET ImageSet: {len(pet_img)} patients, {sum(len(m) for m in pet_img.values())} slices")
    print(f"  PET LabelSet: {len(pet_lbl)} patients, {sum(len(m) for m in pet_lbl.values())} slices")
    print(f"  Union patients: {len(all_patients)}")

    # Clear output directories
    for out_dir in [ct_img_out, ct_lbl_out, pet_img_out, pet_lbl_out]:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.png"):
            old.unlink()
        print(f"  Cleared: {out_dir}")

    # Per-patient matching
    stats = {
        "total_patients": len(all_patients),
        "generated_patients": 0,
        "total_slices_before_match": 0,
        "total_slices_after_match": 0,
        "skipped_no_ct_img": 0,
        "skipped_no_pet_img": 0,
        "mismatch_details": [],
    }

    for pid in all_patients:
        ct_list = ct_img.get(pid, [])  # [(z, path), ...] sorted z_desc
        cl_list = ct_lbl.get(pid, [])
        pt_list = pet_img.get(pid, [])
        pl_list = pet_lbl.get(pid, [])

        n_before = len(ct_list) + len(cl_list) + len(pt_list) + len(pl_list)
        stats["total_slices_before_match"] += n_before

        if not ct_list or not pt_list:
            if not ct_list:
                stats["skipped_no_ct_img"] += 1
            if not pt_list:
                stats["skipped_no_pet_img"] += 1
            stats["mismatch_details"].append(
                f"{pid}: CT_img={len(ct_list)} CT_lbl={len(cl_list)} "
                f"PET_img={len(pt_list)} PET_lbl={len(pl_list)} → 0 matched slices"
            )
            continue

        # Match CT ↔ PET by z-coordinate within tolerance.
        # For each CT slice, find closest PET slice within Z_TOLERANCE mm.
        # Build path lookup dicts keyed by raw z for fast retrieval.
        pt_path_by_z = {z: p for z, p in pt_list}
        cl_path_by_z = {z: p for z, p in cl_list}
        pl_path_by_z = {z: p for z, p in pl_list}

        matched: List[Tuple[float, float]] = []  # [(ct_z, pet_z), ...]
        for ct_z, _ in ct_list:
            pet_z = _find_closest_z(ct_z, pt_list, Z_TOLERANCE)
            if pet_z is not None:
                matched.append((ct_z, pet_z))

        if not matched:
            stats["mismatch_details"].append(
                f"{pid}: CT_img={len(ct_list)} CT_lbl={len(cl_list)} "
                f"PET_img={len(pt_list)} PET_lbl={len(pl_list)} → 0 matched slices"
            )
            continue

        # Build z→path lookup for CT images
        ct_path_by_z = {z: p for z, p in ct_list}

        # Generate PNGs for matched slices
        ct_lbl_missing = 0
        pet_lbl_missing = 0
        for idx, (ct_z, pet_z) in enumerate(matched, start=1):
            sid = f"{pid}{idx:03d}"

            # CT image (mandatory)
            ct_png = _process_ct_image(ct_path_by_z[ct_z])
            ct_png.save(ct_img_out / f"{sid}.png")

            # PET image (mandatory)
            pt_png = _process_pet_image(pt_path_by_z[pet_z])
            pt_png.save(pet_img_out / f"{sid}.png")

            # CT label (match by z within tolerance)
            ct_lbl_z = _find_closest_z(ct_z, cl_list, Z_TOLERANCE)
            if ct_lbl_z is not None:
                ct_lbl_png = _process_label(cl_path_by_z[ct_lbl_z])
                ct_lbl_png.save(ct_lbl_out / f"{sid}.png")
            else:
                ct_lbl_missing += 1

            # PET label (match by z within tolerance)
            pet_lbl_z = _find_closest_z(pet_z, pl_list, Z_TOLERANCE)
            if pet_lbl_z is not None:
                pet_lbl_png = _process_label(pl_path_by_z[pet_lbl_z])
                pet_lbl_png.save(pet_lbl_out / f"{sid}.png")
            else:
                pet_lbl_missing += 1

        n_matched = len(matched)
        stats["total_slices_after_match"] += n_matched
        stats["generated_patients"] += 1

        # Report per-patient alignment quality
        if (len(ct_list) != len(pt_list)
                or len(ct_list) != len(cl_list)
                or len(pt_list) != len(pl_list)
                or ct_lbl_missing > 0
                or pet_lbl_missing > 0):
            stats["mismatch_details"].append(
                f"{pid}: CT_img={len(ct_list)} CT_lbl={len(cl_list)} "
                f"PET_img={len(pt_list)} PET_lbl={len(pl_list)} "
                f"→ {n_matched} matched slices "
                f"CT_lbl_missing={ct_lbl_missing} PET_lbl_missing={pet_lbl_missing}"
            )

    # Print stats
    print(f"\n  Generated patients: {stats['generated_patients']}/{stats['total_patients']}")
    print(f"  Total slices (before match): {stats['total_slices_before_match']}")
    print(f"  Total slices (after match):  {stats['total_slices_after_match']}")
    if stats["mismatch_details"]:
        print(f"\n  Mismatch/quality details:")
        for detail in stats["mismatch_details"][:30]:
            print(f"    {detail}")
        if len(stats["mismatch_details"]) > 30:
            print(f"    ... and {len(stats['mismatch_details']) - 30} more")

    # Verify output
    for label, out_dir in [("CT ImageSet", ct_img_out),
                            ("CT LabelSet", ct_lbl_out),
                            ("PET ImageSet", pet_img_out),
                            ("PET LabelSet", pet_lbl_out)]:
        n = len(list(out_dir.glob("*.png")))
        print(f"  Output {label}: {n} PNGs")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Regenerate PNGs from DICOM with z-coordinate matching"
    )
    ap.add_argument("--data-root", required=True, type=Path,
                    help="Root Data directory (contains part_CT/, part_PET/)")
    ap.add_argument("--split", default="all",
                    choices=["train", "test", "all"],
                    help="Which split to process (default: all)")
    ap.add_argument("--ct-hu-min", type=float, default=-150.0,
                    help="CT HU window lower bound")
    ap.add_argument("--ct-hu-max", type=float, default=250.0,
                    help="CT HU window upper bound")
    ap.add_argument("--pet-suv-max", type=float, default=20.0,
                    help="PET SUV window upper bound")
    args = ap.parse_args(argv)

    global CT_HU_MIN, CT_HU_MAX, PET_SUV_MAX
    CT_HU_MIN = args.ct_hu_min
    CT_HU_MAX = args.ct_hu_max
    PET_SUV_MAX = args.pet_suv_max

    if not args.data_root.is_dir():
        print(f"ERROR: data-root not found: {args.data_root}")
        return 1

    splits = ["train", "test"] if args.split == "all" else [args.split]
    all_stats = {}
    for split in splits:
        all_stats[split] = regenerate_split(args.data_root, split)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for split, stats in all_stats.items():
        print(f"  {split}: {stats['total_slices_after_match']} slices "
              f"from {stats['generated_patients']} patients")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
