"""Organ prior preprocessing: TotalSegmentator inference + distance transform.

Usage:
    # Run TotalSegmentator on all CT DICOMs and write organ data to .npz cache
    python -m src.data.organ_preprocess \\
        --cache-dir cache/tensors \\
        --total-segmentator-path /path/to/Totalsegmentator \\
        --gpu

    # With custom organ class mapping
    python -m src.data.organ_preprocess \\
        --cache-dir cache/tensors \\
        --organ-map configs/organ_mapping.yaml

Output per .npz:
    organ_mask      [6, H, W]  float32  -- one-hot per organ class
    organ_distance  [6, H, W]  float32  -- signed distance transform per organ

Target organ classes (fixed 6-class scheme):
    0: uterus / pelvic_region     (TS: uterus, vagina, ovary, or bbox fallback)
    1: bladder                    (TS: urinary_bladder)
    2: rectum                     (TS: rectum, colon, small_bowel)
    3: bone                       (TS: hip_left, hip_right, sacrum, vertebra, rib)
    4: fat                        (TS: subcutaneous_fat, torso_fat)
    5: muscle_or_other            (TS: gluteus, iliopsoas, autochthon, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# TotalSegmentator class → 6-organ mapping
# ---------------------------------------------------------------------------
# TotalSegmentator v2 "total" task label indices.
# Update these to match your installed TotalSegmentator version.
# Run: Totalsegmentator --task total --list_labels  to see all indices.

TOTALSEGMENTATOR_V2_MAPPING: Dict[int, int] = {
    # class_0: uterus / pelvic_region
    55: 0,   # uterus (varies by TS version)
    56: 0,   # vagina
    57: 0,   # ovary_left
    58: 0,   # ovary_right
    # class_1: bladder
    20: 1,   # urinary_bladder
    # class_2: rectum / bowel
    38: 2,   # rectum
    21: 2,   # colon
    22: 2,   # small_bowel
    23: 2,   # duodenum
    # class_3: bone
    1: 3,    # femur_left
    2: 3,    # femur_right
    3: 3,    # hip_left
    4: 3,    # hip_right
    5: 3,    # sacrum
    6: 3,    # vertebra_L5
    7: 3,    # vertebra_L4
    8: 3,    # vertebra_L3
    9: 3,    # vertebra_L2
    10: 3,   # vertebra_L1
    # class_4: fat
    11: 4,   # subcutaneous_fat
    12: 4,   # torso_fat
    # class_5: muscle or other (catch-all)
    13: 5,   # gluteus_maximus_left
    14: 5,   # gluteus_maximus_right
    15: 5,   # gluteus_medius_left
    16: 5,   # gluteus_medius_right
    17: 5,   # gluteus_minimus_left
    18: 5,   # gluteus_minimus_right
    19: 5,   # iliopsoas_left
    24: 5,   # iliopsoas_right
    25: 5,   # autochthon_left
    26: 5,   # autochthon_right
}

NUM_ORGAN_CLASSES = 6

ORGAN_CLASS_NAMES = [
    "uterus_pelvic",
    "bladder",
    "rectum_bowel",
    "bone",
    "fat",
    "muscle_other",
]


# ---------------------------------------------------------------------------
# Distance transform
# ---------------------------------------------------------------------------

def _distance_transform_2d(mask: np.ndarray) -> np.ndarray:
    """Signed distance transform: positive inside, negative outside, normalised."""
    from scipy.ndimage import distance_transform_edt
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros_like(mask, dtype=np.float32)
    d_in = distance_transform_edt(mask_bool).astype(np.float32)
    d_out = distance_transform_edt(~mask_bool).astype(np.float32)
    signed = d_in - d_out
    # Normalise to roughly [-1, 1] with soft clamping
    max_abs = max(abs(signed.max()), abs(signed.min()), 1.0)
    return np.tanh(signed / (max_abs * 0.5)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main preprocessing logic
# ---------------------------------------------------------------------------

def _try_import_totalsegmentator():
    """Attempt to import TotalSegmentator. Returns the class or None."""
    try:
        from totalsegmentator.python_api import Totalsegmentator
        return Totalsegmentator
    except ImportError:
        return None


def _try_import_nibabel():
    try:
        import nibabel as nib
        return nib
    except ImportError:
        return None


def _render_organ_mask_and_distance(
    ts_seg: np.ndarray,       # [H, W] int, TotalSegmentator label map
    mapping: Dict[int, int],
    image_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert TS label map → organ_mask [6,H,W] + organ_distance [6,H,W]."""
    ts_seg_resized = _resize_np(ts_seg.astype(np.float32), image_size)
    ts_seg_int = np.round(ts_seg_resized).astype(np.int32)

    organ_mask = np.zeros((NUM_ORGAN_CLASSES, image_size, image_size), dtype=np.float32)
    organ_distance = np.zeros((NUM_ORGAN_CLASSES, image_size, image_size), dtype=np.float32)

    for ts_label, organ_class in mapping.items():
        if organ_class >= NUM_ORGAN_CLASSES:
            continue
        m = (ts_seg_int == ts_label).astype(np.float32)
        organ_mask[organ_class] = np.maximum(organ_mask[organ_class], m)
        organ_distance[organ_class] = np.maximum(
            organ_distance[organ_class], _distance_transform_2d(m)
        )

    return organ_mask, organ_distance


def _resize_np(arr: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.squeeze(0).squeeze(0).numpy()


def _build_ct_volume_from_cache(cache_dir: Path, patient_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict, List[Path]]]:
    """Reconstruct CT HU volume from cached .npz files for one patient.

    Returns (volume [D,H,W], affine, meta_dict, sorted_npz_paths) or None.
    The npz_paths list maps volume axis-0 to .npz files for correct slice correspondence.
    """
    nib = _try_import_nibabel()
    if nib is None:
        print("[organ_preprocess] nibabel not available; cannot reconstruct CT volume")
        return None

    # Find all .npz for this patient, sorted by slice_id
    npz_files = sorted(
        [p for p in cache_dir.glob("*.npz") if p.stem.startswith(patient_id)],
        key=lambda p: p.stem,
    )
    if not npz_files:
        return None

    slices_hu = []
    slice_meta = {}
    for npz_path in npz_files:
        data = np.load(npz_path)
        # Read scale_meta to get HU range
        scale_meta = {}
        if "scale_meta_json" in data:
            try:
                json_bytes = bytes(data["scale_meta_json"].tolist())
                scale_meta = json.loads(json_bytes.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        slice_meta = scale_meta

        # Reverse CT normalisation: [-1,1] -> [0,1] -> HU
        ct_raw = data["ct"]
        ct_norm = np.squeeze(ct_raw)
        if ct_norm.ndim != 2:
            continue  # skip malformed entries
        ct_01 = (ct_norm + 1.0) / 2.0
        hu_min = scale_meta.get("ct_hu_min", -150.0)
        hu_max = scale_meta.get("ct_hu_max", 250.0)
        ct_hu = ct_01 * (hu_max - hu_min) + hu_min
        slices_hu.append(ct_hu.astype(np.float32))

    if not slices_hu:
        return None

    volume = np.stack(slices_hu, axis=0)  # [D, H, W]
    # Best-effort affine: assume 1mm in-plane, slice spacing from cache size.
    # When true DICOM spacing is available via scale_meta, it should be used instead.
    in_plane_mm = float(slice_meta.get("pixel_spacing", 1.0)) if slice_meta else 1.0
    slice_spacing = float(slice_meta.get("slice_thickness", 3.0)) if slice_meta else 3.0
    affine = np.array([
        [in_plane_mm, 0, 0, 0],
        [0, in_plane_mm, 0, 0],
        [0, 0, slice_spacing, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    # Return sorted file list so caller can map volume axis-0 back to .npz files
    return volume, affine, slice_meta, npz_files


def _infer_organ_data_for_sample(
    ts_seg: Optional[np.ndarray],
    image_size: int,
    mapping: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get organ_mask and organ_distance for one sample from TS segmentation.

    When ts_seg is None (TS failed or not available), returns zeros.
    """
    if ts_seg is not None:
        return _render_organ_mask_and_distance(ts_seg, mapping, image_size)
    return (
        np.zeros((NUM_ORGAN_CLASSES, image_size, image_size), dtype=np.float32),
        np.zeros((NUM_ORGAN_CLASSES, image_size, image_size), dtype=np.float32),
    )


def process_cache_with_organ_prior(
    cache_dir: Path,
    total_segmentator_binary: Optional[str] = None,
    mapping: Optional[Dict[int, int]] = None,
    gpu: bool = True,
    image_size: int = 192,
    stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run TotalSegmentator per patient, write organ data into existing .npz files.

    Strategy:
      1. Group .npz files by patient_id (from scale_meta or filename prefix)
      2. Reconstruct CT HU volume per patient
      3. Run TotalSegmentator once per patient
      4. Extract 2D organ mask per slice from the 3D TS output
      5. Compute distance transform, write back to each slice's .npz
    """
    if mapping is None:
        mapping = TOTALSEGMENTATOR_V2_MAPPING

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    npz_files = sorted(cache_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No .npz files in {cache_dir}")

    stats_out: Dict[str, Any] = {"total_npz": len(npz_files), "updated": 0, "skipped": 0,
                                  "errors": [], "organ_coverage": {}}

    # ---- Group by patient_id ----
    patient_slices: Dict[str, List[Path]] = {}
    for npz_path in npz_files:
        # Try to read patient_id from scale_meta; fall back to filename prefix
        pid = npz_path.stem[:3]  # First 3 chars = patient ID in 001002 format
        patient_slices.setdefault(pid, []).append(npz_path)

    # ---- Per-patient TotalSegmentator inference ----
    TS = _try_import_totalsegmentator()
    nib = _try_import_nibabel()

    for pid, slice_paths in patient_slices.items():
        print(f"[organ_preprocess] Patient {pid}: {len(slice_paths)} slices")

        # Reconstruct CT volume
        vol_result = _build_ct_volume_from_cache(cache_dir, pid)
        if vol_result is None:
            for p in slice_paths:
                _write_zeros_organ(p)
            stats_out["skipped"] += len(slice_paths)
            continue

        ct_volume, affine, meta, sorted_paths = vol_result

        # Build volume-path index for correct slice correspondence
        vol_path_to_idx = {str(p): i for i, p in enumerate(sorted_paths)}

        # Run TotalSegmentator
        ts_seg_3d = None
        if TS is not None and nib is not None:
            try:
                nii = nib.Nifti1Image(ct_volume, affine)
                ts = TS(task="total", device="gpu" if gpu else "cpu", fast=True, verbose=False)
                # Save to temp file (TS API requires file path or Nifti)
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                    nib.save(nii, tmp.name)
                    tmp_path = tmp.name
                ts_seg_result = ts.segment(tmp_path)
                ts_seg_3d = ts_seg_result.get_fdata().astype(np.int32)  # [D, H, W]
                os.unlink(tmp_path)
                print(f"  TotalSegmentator OK. Unique labels: {np.unique(ts_seg_3d)}")
            except Exception as exc:
                stats_out.setdefault("errors", []).append(f"{pid}: TS failed: {exc}")
                print(f"  WARNING: TotalSegmentator failed: {exc}")
        else:
            print(f"  TotalSegmentator not available; writing zero organ data")

        # ---- Write per-slice organ data ----
        for npz_path in slice_paths:
            try:
                # Use the volume-path index for accurate slice correspondence
                slc_idx = vol_path_to_idx.get(str(npz_path), None)

                # Extract 2D organ seg from 3D
                ts_seg_2d = None
                if ts_seg_3d is not None and slc_idx is not None and 0 <= slc_idx < ts_seg_3d.shape[0]:
                    ts_seg_2d = ts_seg_3d[slc_idx]  # [H, W] (TS native res)
                elif ts_seg_3d is not None and slc_idx is not None:
                    # Slice index out of range — use nearest available
                    nearest = min(max(0, slc_idx), ts_seg_3d.shape[0] - 1)
                    ts_seg_2d = ts_seg_3d[nearest]

                organ_mask, organ_distance = _infer_organ_data_for_sample(
                    ts_seg_2d, image_size, mapping
                )

                _update_npz_with_organ(npz_path, organ_mask, organ_distance)
                stats_out["updated"] += 1

                # Track coverage
                for c in range(NUM_ORGAN_CLASSES):
                    cov = float(organ_mask[c].sum() / organ_mask[c].size)
                    stats_out["organ_coverage"].setdefault(ORGAN_CLASS_NAMES[c], []).append(cov)

            except Exception as exc:
                stats_out.setdefault("errors", []).append(f"{npz_path.name}: {exc}")

    # ---- Summary ----
    for cname in ORGAN_CLASS_NAMES:
        covs = stats_out["organ_coverage"].get(cname, [])
        if covs:
            stats_out[f"organ_coverage_{cname}_mean"] = float(np.mean(covs))
            stats_out[f"organ_coverage_{cname}_zero_pct"] = float(np.mean([1.0 if c < 1e-6 else 0.0 for c in covs]) * 100)

    return stats_out


def _write_zeros_organ(npz_path: Path) -> None:
    """Write zero organ data into .npz (used when TS is unavailable)."""
    img_size = 192
    # Try to read existing image size from cache
    try:
        data = np.load(npz_path)
        img_size = data["ct"].shape[1]
    except (KeyError, OSError):
        pass
    _update_npz_with_organ(
        npz_path,
        np.zeros((NUM_ORGAN_CLASSES, img_size, img_size), dtype=np.float32),
        np.zeros((NUM_ORGAN_CLASSES, img_size, img_size), dtype=np.float32),
    )


def _update_npz_with_organ(
    npz_path: Path,
    organ_mask: np.ndarray,
    organ_distance: np.ndarray,
) -> None:
    """Append organ_mask and organ_distance to existing .npz."""
    existing = dict(np.load(npz_path))
    existing["organ_mask"] = organ_mask.astype(np.float32)
    existing["organ_distance"] = organ_distance.astype(np.float32)
    np.savez_compressed(str(npz_path), **existing)


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def generate_organ_report(stats: Dict[str, Any]) -> str:
    """Generate a human-readable quality report for organ preprocessing."""
    lines = [
        "# Organ Prior Preprocessing Report",
        "",
        f"- Total .npz files: {stats.get('total_npz', 0)}",
        f"- Updated: {stats.get('updated', 0)}",
        f"- Skipped (no TS available): {stats.get('skipped', 0)}",
        f"- Errors: {len(stats.get('errors', []))}",
        "",
        "## Per-Organ Coverage",
        "",
        "| Organ Class | Mean Coverage | Zero-Mask % |",
        "|-------------|---------------|-------------|",
    ]
    for cname in ORGAN_CLASS_NAMES:
        mean_cov = stats.get(f"organ_coverage_{cname}_mean", "N/A")
        zero_pct = stats.get(f"organ_coverage_{cname}_zero_pct", "N/A")
        if isinstance(mean_cov, float):
            mean_cov = f"{mean_cov:.4f}"
        if isinstance(zero_pct, float):
            zero_pct = f"{zero_pct:.1f}%"
        lines.append(f"| {cname} | {mean_cov} | {zero_pct} |")

    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    # Flag organs with >50% zero masks
    warnings = []
    for cname in ORGAN_CLASS_NAMES:
        zero_pct = stats.get(f"organ_coverage_{cname}_zero_pct", 0)
        if isinstance(zero_pct, float) and zero_pct > 50.0:
            warnings.append(f"- **{cname}**: {zero_pct:.1f}% slices have zero coverage (mask may be empty)")
    if warnings:
        lines.extend(warnings)
    else:
        lines.append("(no warnings)")

    if stats.get("errors"):
        lines.append("")
        lines.append("## Errors")
        for err in stats["errors"][:20]:
            lines.append(f"- {err}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Organ prior preprocessing for .npz cache")
    ap.add_argument("--cache-dir", required=True, type=Path, help="Path to .npz cache directory")
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--gpu", action="store_true", default=True)
    ap.add_argument("--cpu", dest="gpu", action="store_false")
    ap.add_argument("--organ-map", type=Path, default=None, help="JSON file with custom TS->organ mapping")
    ap.add_argument("--output-report", type=Path, default=None, help="Write quality report to file")
    ap.add_argument("--dry-run", action="store_true", help="Scan only, no writes")
    args = ap.parse_args(argv)

    mapping = TOTALSEGMENTATOR_V2_MAPPING
    if args.organ_map and args.organ_map.exists():
        with open(args.organ_map, "r", encoding="utf-8") as f:
            raw = json.load(f)
            mapping = {int(k): int(v) for k, v in raw.items()}

    if args.dry_run:
        npz_files = sorted(args.cache_dir.glob("*.npz"))
        patients = set(p.stem[:3] for p in npz_files)
        print(f"[organ_preprocess] DRY RUN: {len(npz_files)} .npz files, {len(patients)} patients")
        print(f"  Organ mapping: {len(mapping)} TS labels -> {NUM_ORGAN_CLASSES} classes")
        TS = _try_import_totalsegmentator()
        print(f"  TotalSegmentator: {'AVAILABLE' if TS else 'NOT FOUND'}")
        nib = _try_import_nibabel()
        print(f"  nibabel: {'AVAILABLE' if nib else 'NOT FOUND'}")
        return 0

    stats = process_cache_with_organ_prior(
        cache_dir=args.cache_dir,
        mapping=mapping,
        gpu=args.gpu,
        image_size=args.image_size,
    )

    report = generate_organ_report(stats)
    print(report)

    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report, encoding="utf-8")
        print(f"Report saved to {args.output_report}")

    print(f"\nDone. {stats['updated']} files updated, {stats['skipped']} skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
