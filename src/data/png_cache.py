"""Build NPZ training cache from split-organized PNG data.

Expected layout by default:

    main_data/
      split.csv
      train/ct/*.png
      train/pet_peizhuan/*.png
      train/pet/*.png
      train/label/*.png
      val/...

The output .npz files are compatible with :class:`src.data.dataset.CachedDataset`.
PNG intensity is not treated as physical SUV/HU; SUV-specific losses should be
disabled unless the PNG export is backed by a known physical calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

_CACHE_EXT = ".npz"
_SPLITS = {"train", "val", "test"}
_MANIFEST_COLUMNS = ["sample_id", "patient_id", "slice_id", "split", "cache_path"]
_IMAGE_EXTS = (".png", ".PNG")


def _sample_id_from_file_name(file_name: str) -> str:
    return Path(str(file_name).strip()).stem


def _parse_sample_id(sample_id: str) -> Tuple[str, int]:
    patient_id = sample_id[:3] if len(sample_id) >= 3 else sample_id
    try:
        slice_id = int(sample_id[3:6]) if len(sample_id) >= 6 else 0
    except ValueError:
        slice_id = 0
    return patient_id, slice_id


def _read_split_rows(split_csv: Path) -> List[Dict[str, str]]:
    with open(split_csv, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for row in reader:
            rows.append({str(k): ("" if v is None else str(v).strip()) for k, v in row.items()})
        return rows


def _candidate_file_names(file_name: str) -> Iterable[str]:
    name = str(file_name).strip()
    if not name:
        return []
    path = Path(name)
    if path.suffix:
        return [path.name]
    return [f"{path.name}{ext}" for ext in _IMAGE_EXTS]


def _find_png(base_dir: Path, file_name: str) -> Optional[Path]:
    for candidate in _candidate_file_names(file_name):
        path = base_dir / candidate
        if path.exists():
            return path
    return None


def _image_resample_filter(mask: bool) -> int:
    if mask:
        return Image.Resampling.NEAREST
    return Image.Resampling.BILINEAR


def _load_png_array(path: Path, image_size: int, *, mask: bool = False) -> Tuple[np.ndarray, str]:
    with Image.open(path) as img:
        original_mode = img.mode
        if mask:
            img = img.convert("L")
        elif img.mode not in {"L", "I", "I;16", "I;16B", "I;16L", "F"}:
            img = img.convert("L")
        if image_size > 0 and img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), _image_resample_filter(mask))
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported PNG shape {arr.shape} for {path}")
    return arr, original_mode


def _normalise_unit(arr: np.ndarray, mode: str, normalization: str) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if normalization == "minmax":
        v_min = float(arr.min())
        v_max = float(arr.max())
        return np.zeros_like(arr, dtype=np.float32) if v_max <= v_min else (arr - v_min) / (v_max - v_min)

    if normalization != "auto":
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    v_min = float(arr.min())
    v_max = float(arr.max())
    if v_min >= 0.0 and v_max <= 1.0:
        unit = arr
    elif v_min >= 0.0 and v_max <= 255.0:
        unit = arr / 255.0
    elif v_min >= 0.0 and v_max <= 65535.0:
        unit = arr / 65535.0
    else:
        unit = _normalise_unit(arr, mode, "minmax")
    return np.clip(unit.astype(np.float32), 0.0, 1.0)


def _normalise_image(arr: np.ndarray, mode: str, normalization: str) -> np.ndarray:
    unit = _normalise_unit(arr, mode, normalization)
    return (unit * 2.0 - 1.0).astype(np.float32)


def _normalise_mask(arr: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        mask = arr > 0.0
    else:
        limit = threshold if arr.max() <= 1.0 else threshold * 255.0
        mask = arr > limit
    return mask.astype(np.float32)


def _json_bytes(payload: Dict[str, Any]) -> np.ndarray:
    return np.frombuffer(json.dumps(payload, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)


def _write_split_manifest(rows: List[Dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _empty_stats() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "built": 0,
        "skipped": {},
        "pet_source_counts": {},
        "manifest_rows": 0,
        "errors": [],
    }


def build_png_cache(
    *,
    png_root: str | Path,
    split_csv: str | Path | None = None,
    out_dir: str | Path,
    split_manifest: str | Path | None = None,
    image_size: int = 192,
    ct_subdir: str = "ct",
    pet_subdirs: Sequence[str] = ("pet_peizhuan", "pet"),
    label_subdir: str = "label",
    normalization: str = "auto",
    mask_threshold: float = 0.0,
    pet_invert: bool = False,
    allow_missing_mask: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """Convert split-organized PNG slices into CachedDataset-compatible NPZ files."""
    png_root = Path(png_root)
    split_csv = Path(split_csv) if split_csv is not None else png_root / "split.csv"
    out_dir = Path(out_dir)
    split_manifest = Path(split_manifest) if split_manifest is not None else png_root / "split_manifest.csv"

    if not split_csv.exists():
        raise FileNotFoundError(f"split_csv not found: {split_csv}")
    if not png_root.exists():
        raise FileNotFoundError(f"png_root not found: {png_root}")
    if not pet_subdirs:
        raise ValueError("pet_subdirs must contain at least one directory name")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_split_rows(split_csv)
    skipped: Counter[str] = Counter()
    pet_source_counts: Counter[str] = Counter()
    errors: List[str] = []
    manifest_rows: List[Dict[str, str]] = []

    for row in rows:
        sample_file = row.get("file_name", "") or row.get("sample_id", "")
        sample_id = _sample_id_from_file_name(sample_file)
        split = row.get("split", "").lower()
        patient_id = row.get("patient_id", "") or _parse_sample_id(sample_id)[0]
        parsed_patient, slice_id = _parse_sample_id(sample_id)
        patient_id = patient_id or parsed_patient

        if not sample_id:
            skipped["invalid_row"] += 1
            continue
        if split not in _SPLITS:
            skipped["invalid_split"] += 1
            if len(errors) < 50:
                errors.append(f"{sample_id}: invalid split {split!r}")
            continue

        split_dir = png_root / split
        ct_path = _find_png(split_dir / ct_subdir, sample_file)
        if ct_path is None:
            skipped["missing_ct_png"] += 1
            if len(errors) < 50:
                errors.append(f"{sample_id}: missing CT PNG under {split_dir / ct_subdir}")
            continue

        pet_path = None
        pet_source = ""
        for pet_subdir in pet_subdirs:
            pet_path = _find_png(split_dir / pet_subdir, sample_file)
            if pet_path is not None:
                pet_source = pet_subdir
                break
        if pet_path is None:
            skipped["missing_pet_png"] += 1
            if len(errors) < 50:
                errors.append(f"{sample_id}: missing PET PNG under {split_dir / list(pet_subdirs)[0]}")
            continue

        label_path = _find_png(split_dir / label_subdir, sample_file)
        if label_path is None and not allow_missing_mask:
            skipped["missing_label_png"] += 1
            if len(errors) < 50:
                errors.append(f"{sample_id}: missing label PNG under {split_dir / label_subdir}")
            continue

        out_path = out_dir / f"{sample_id}{_CACHE_EXT}"
        if out_path.exists() and not overwrite:
            skipped["exists"] += 1
            # Audit P0-5: the split manifest is rewritten in full below and
            # CachedDataset treats it as authoritative, so skipped-but-existing
            # samples must still be listed or they silently vanish from
            # training/eval on an incremental --no-overwrite rerun.
            manifest_rows.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "slice_id": str(slice_id),
                "split": split,
                "cache_path": str(out_path),
            })
            pet_source_counts[pet_source] += 1
            continue

        try:
            ct_raw, ct_mode = _load_png_array(ct_path, image_size, mask=False)
            pet_raw, pet_mode = _load_png_array(pet_path, image_size, mask=False)
            if pet_invert:
                # pet_peizhuan PNGs are inverted grayscale on a white canvas:
                # background (outside body) = 255, hot lesion = darkest. Invert
                # (255 - x) so background -> 0 and hot lesion -> bright before
                # normalisation. Use only for white-canvas inverted PET.
                pet_raw = 255.0 - pet_raw
            ct = _normalise_image(ct_raw, ct_mode, normalization)[None, ...]
            pet = _normalise_image(pet_raw, pet_mode, normalization)[None, ...]

            has_mask = label_path is not None
            if has_mask:
                mask_raw, _ = _load_png_array(label_path, image_size, mask=True)
                mask = _normalise_mask(mask_raw, mask_threshold)[None, ...]
            else:
                mask = np.zeros_like(ct, dtype=np.float32)

            scale_meta = {
                "suv_ok": False,
                "pet_suv_available": False,
                "pet_raw_min": float(pet_raw.min()),
                "pet_raw_max": float(pet_raw.max()),
                "ct_raw_min": float(ct_raw.min()),
                "ct_raw_max": float(ct_raw.max()),
                "ct_physical_key": "",
                "pet_physical_key": "",
                "pet_physical_kind": "png_intensity",
                "png_normalization": normalization,
                "pet_invert": pet_invert,
                "patient_id": patient_id,
                "slice_id": int(slice_id),
            }
            payload = {
                "ct": ct.astype(np.float32),
                "pet": pet.astype(np.float32),
                "mask": mask.astype(np.float32),
                "mu_map": np.zeros_like(ct, dtype=np.float32),
                "scale_meta_json": _json_bytes(scale_meta),
            }
            np.savez_compressed(str(out_path), **payload)

            meta = {
                "sample_id": sample_id,
                "patient_id": patient_id,
                "slice_id": int(slice_id),
                "split": split,
                "source": "png",
                "has_label": bool(has_mask),
                "ct_png_path": str(ct_path),
                "pet_png_path": str(pet_path),
                "pet_source": pet_source,
                "label_png_path": str(label_path) if label_path is not None else "",
                "scale_meta": scale_meta,
            }
            with open(out_dir / f"{sample_id}_meta.json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, default=str)

            manifest_rows.append({
                "sample_id": sample_id,
                "patient_id": patient_id,
                "slice_id": str(slice_id),
                "split": split,
                "cache_path": str(out_path),
            })
            pet_source_counts[pet_source] += 1
        except Exception as exc:
            skipped["processing_error"] += 1
            if len(errors) < 50:
                errors.append(f"{sample_id}: {exc}")

    _write_split_manifest(manifest_rows, split_manifest)
    stats = _empty_stats()
    stats.update({
        "total_rows": len(rows),
        "built": len(manifest_rows) - int(skipped.get("exists", 0)),
        "manifest_rows": len(manifest_rows),
        "skipped": dict(skipped),
        "pet_source_counts": dict(pet_source_counts),
        "cache_dir": str(out_dir),
        "split_manifest": str(split_manifest),
        "errors": errors,
    })
    with open(split_manifest.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False, default=str)
    return stats


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build NPZ cache from split-organized PNG data")
    ap.add_argument("--png-root", required=True, type=Path, help="Root containing split.csv and train/val folders")
    ap.add_argument("--split-csv", type=Path, default=None, help="CSV with file_name, patient_id, split columns")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output .npz cache directory")
    ap.add_argument("--split-manifest", type=Path, default=None, help="Output split_manifest.csv")
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--ct-subdir", default="ct")
    ap.add_argument("--pet-subdirs", nargs="+", default=["pet_peizhuan", "pet"],
                    help="PET subdirectories in priority order")
    ap.add_argument("--label-subdir", default="label")
    ap.add_argument("--normalization", choices=["auto", "minmax"], default="auto")
    ap.add_argument("--mask-threshold", type=float, default=0.0)
    ap.add_argument("--pet-invert", action="store_true",
                    help="Invert PET (255-x): for white-canvas inverted PET where background=255 and hot lesion is dark")
    ap.add_argument("--allow-missing-mask", action="store_true")
    ap.add_argument("--no-overwrite", action="store_true")
    ap.add_argument("--allow-skips", action="store_true",
                    help="Return success even if some rows could not be converted")
    args = ap.parse_args(argv)

    stats = build_png_cache(
        png_root=args.png_root,
        split_csv=args.split_csv,
        out_dir=args.out_dir,
        split_manifest=args.split_manifest,
        image_size=args.image_size,
        ct_subdir=args.ct_subdir,
        pet_subdirs=args.pet_subdirs,
        label_subdir=args.label_subdir,
        normalization=args.normalization,
        mask_threshold=args.mask_threshold,
        pet_invert=args.pet_invert,
        allow_missing_mask=args.allow_missing_mask,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    if stats["skipped"] and not args.allow_skips:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
