"""
CT→PET medical dataset — offline preprocessing + fast online loading.

Architecture
------------
**Phase 1 — offline (run once):** ``CacheBuilder`` reads manifest CSV,
loads DICOM/PNG sources, computes HU / SUV, validates CT-PET spatial
alignment, normalises to [-1,1], and writes compact ``.npz`` cache files.
No DICOM parsing happens during training.

**Phase 2 — online (every epoch):** ``CachedDataset`` reads precomputed
``.npz`` tensors via ``np.load`` → ``torch.as_tensor``, applies
lightweight CPU augmentations, and feeds the GPU through pinned-memory
``DataLoader`` workers.

Usage
-----
.. code-block:: bash

   # 1) Build manifest (one-time)
   python scripts/build_exposed_dataset.py  \
       --raw-root Data --dicom-root Data --test-png-root Data  \
       --out-root cache/

   # 2) Preprocess raw → cache (one-time, heavy)
   python -m src.data.dataset preprocess  \
       --manifest cache/manifest_all.csv  \
       --out-dir cache/tensors

   # 3) Train (reads cache, lightweight)
   #    Set "cache_dir": "cache/tensors" in your config YAML
   python scripts/train_v2.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

_CACHE_EXT = ".npz"
_SAMPLE_ID_PATTERN: re.Pattern = re.compile(r"^(\d{3})(\d{3})$")

_CT_TARGET_RANGE = (-1.0, 1.0)
_PET_TARGET_RANGE = (-1.0, 1.0)

# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def _basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _parse_sample_id(sid: str) -> Tuple[Optional[str], Optional[int]]:
    m = _SAMPLE_ID_PATTERN.match(sid)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        if array.shape[0] == 1:
            return array[0]
        if array.shape[-1] == 1:
            return array[..., 0]
        return array[0]
    raise ValueError(f"Unsupported DICOM pixel shape: {array.shape}")


def _read_dicom_pixels(path: str) -> Tuple[Any, np.ndarray]:
    """Return ``(pydicom_dataset, float32_pixels)`` with Slope/Intercept applied."""
    import pydicom

    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    pixels = _ensure_2d(pixels)
    slope = _safe_float(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", None), 0.0)
    return ds, (pixels * slope + intercept).astype(np.float32)


# ---------------------------------------------------------------------------
# DICOM datetime parsing (shared by CacheBuilder)
# ---------------------------------------------------------------------------


def _parse_dcm_datetime(
    date_str: Optional[str], time_str: Optional[str]
) -> Optional[datetime]:
    if not date_str or not time_str:
        return None
    date_str, time_str = str(date_str).strip(), str(time_str).strip()
    if not date_str:
        return None
    time_main = time_str.split(".")[0].ljust(6, "0")
    frac = time_str[len(time_main):]
    if len(date_str) != 8:
        return None
    try:
        dt = datetime.strptime(f"{date_str}{time_main[:6]}", "%Y%m%d%H%M%S")
    except ValueError:
        return None
    if frac.startswith("."):
        try:
            dt = dt.replace(microsecond=int(frac[1:7].ljust(6, "0")))
        except ValueError:
            pass
    return dt


def compute_suv(ds: Any, activity_conc: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Attempt SUV computation from PET DICOM metadata.

    Returns ``(suv_array, suv_ok)``.  When *suv_ok* is ``False`` the
    returned array is the raw *activity_conc*.
    """
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

    # injection time
    start_dt = None
    raw_start = getattr(rad, "RadiopharmaceuticalStartDateTime", None)
    if raw_start:
        raw_start = str(raw_start)
        start_dt = _parse_dcm_datetime(raw_start[:8], raw_start[8:])
    if start_dt is None:
        start_dt = _parse_dcm_datetime(
            getattr(ds, "SeriesDate", None) or getattr(ds, "AcquisitionDate", None),
            getattr(rad, "RadiopharmaceuticalStartTime", None),
        )

    # acquisition time
    acq_dt = None
    raw_acq = getattr(ds, "AcquisitionDateTime", None)
    if raw_acq:
        raw_acq = str(raw_acq)
        acq_dt = _parse_dcm_datetime(raw_acq[:8], raw_acq[8:])
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


# ===================================================================
# Phase 1: Offline preprocessing
# ===================================================================


@dataclass
class PreprocessConfig:
    """Normalisation parameters that control the offline cache builder."""

    ct_hu_min: float = -150.0
    ct_hu_max: float = 250.0
    pet_suv_max: float = 20.0
    image_size: int = 128
    strict_suv: bool = False


class CacheBuilder:
    """One-shot offline pipeline: manifest CSV → normalised ``.npz`` cache.

    All heavy work — DICOM decode, HU conversion, SUV computation,
    spatial validation, resizing — runs **once** here.  The training
    ``Dataset`` only does cheap ``np.load`` + ``torch.as_tensor``.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        self._errors: List[str] = []
        self._stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def build(
        self,
        manifest_csv: Path,
        output_dir: Path,
        *,
        validate_spatial: bool = True,
        z_tolerance_mm: float = 2.0,
    ) -> Dict[str, Any]:
        """Read *manifest_csv*, convert every valid row, write ``.npz`` files.

        Returns a stats dict suitable for logging / report generation.
        """
        import csv as _csv

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(manifest_csv, "r", encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))

        if not rows:
            raise ValueError(f"Manifest is empty: {manifest_csv}")

        self._errors.clear()
        built = 0
        skipped: Dict[str, int] = defaultdict(int)

        for row in rows:
            if self._process_row(row, output_dir, validate_spatial, z_tolerance_mm, skipped):
                built += 1

        # spatial alignment summary
        ct_z_ok: Optional[int] = None
        if validate_spatial:
            ct_z_ok = 0
            for row in rows:
                cz = row.get("ct_dicom_z", "")
                pz = row.get("pet_dicom_z", "")
                if cz and pz:
                    if abs(float(cz) - float(pz)) <= z_tolerance_mm:
                        ct_z_ok += 1

        self._stats = {
            "total_rows": len(rows),
            "built": built,
            "skipped": dict(skipped),
            "spatially_aligned_pairs": ct_z_ok,
            "z_tolerance_mm": z_tolerance_mm if validate_spatial else None,
            "errors": self._errors[:50],
        }
        return self._stats

    # ------------------------------------------------------------------
    # per-row
    # ------------------------------------------------------------------

    def _process_row(
        self,
        row: Dict[str, str],
        out_dir: Path,
        validate: bool,
        z_tol: float,
        skipped: Dict[str, int],
    ) -> bool:
        sid = row.get("sample_id", "")
        split = row.get("split", "")
        if not sid or split not in {"train", "val", "test"}:
            skipped["invalid_split"] += 1
            return False

        ct_png = row.get("ct_png_path", "")
        pet_png = row.get("pet_png_path", "")
        ct_dcm = row.get("ct_dicom_path", "")
        pet_dcm = row.get("pet_dicom_path", "")
        ct_label = row.get("ct_label_png_path", "")
        pet_label = row.get("pet_label_png_path", "")

        has_dicom = bool(ct_dcm and pet_dcm
                         and Path(ct_dcm).exists() and Path(pet_dcm).exists())
        has_png = bool(ct_png and pet_png
                       and Path(ct_png).exists() and Path(pet_png).exists())

        if not has_dicom and not has_png:
            skipped["no_source"] += 1
            return False

        # spatial check (DICOM only)
        if validate and has_dicom:
            ct_z = _safe_float(row.get("ct_dicom_z"), None)
            pet_z = _safe_float(row.get("pet_dicom_z"), None)
            if (ct_z is not None and pet_z is not None
                    and abs(ct_z - pet_z) > z_tol):
                skipped["spatial_mismatch"] += 1
                return False

        # --- load & normalise ---
        try:
            if has_dicom:
                ct_tensor, ct_meta = self._process_ct_dicom(ct_dcm)
                pet_tensor, pet_meta = self._process_pet_dicom(pet_dcm)
            else:
                ct_tensor, ct_meta = self._process_ct_png(ct_png)
                pet_tensor, pet_meta = self._process_pet_png(pet_png)

            mask_tensor: Optional[torch.Tensor] = None
            mask_meta: Dict[str, Any] = {}
            label_path = ct_label or pet_label
            if label_path and Path(label_path).exists():
                mask_tensor, mask_meta = self._process_label_png(label_path)

        except Exception as exc:
            self._errors.append(f"{sid}: {exc}")
            skipped["processing_error"] += 1
            return False

        # --- write cache ---
        payload: Dict[str, np.ndarray] = {
            "ct": ct_tensor.numpy().astype(np.float32),
            "pet": pet_tensor.numpy().astype(np.float32),
        }
        if mask_tensor is not None:
            payload["mask"] = mask_tensor.numpy().astype(np.float32)

        np.savez_compressed(str(out_dir / f"{sid}{_CACHE_EXT}"), **payload)

        # lightweight per-sample sidecar
        meta = {
            "sample_id": sid,
            "patient_id": row.get("patient_id", ""),
            "slice_id": row.get("slice_id", ""),
            "split": split,
            "source": "dicom" if has_dicom else "png",
            **ct_meta,
            **pet_meta,
            **mask_meta,
        }
        with open(out_dir / f"{sid}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, default=str)

        return True

    # ------------------------------------------------------------------
    # modality processors
    # ------------------------------------------------------------------

    def _process_ct_dicom(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ds, hu = _read_dicom_pixels(path)
        tensor = self._normalize_ct(hu)
        meta = {
            "ct_sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "ct_series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "ct_instance_number": str(getattr(ds, "InstanceNumber", "")),
        }
        return tensor, meta

    def _process_ct_png(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = self._resize_to_tensor(arr)
        lo, hi = _CT_TARGET_RANGE
        tensor = tensor * (hi - lo) + lo
        return tensor, {"ct_source": "png"}

    def _process_pet_dicom(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        ds, activity = _read_dicom_pixels(path)
        suv, suv_ok = compute_suv(ds, activity)
        if not suv_ok:
            if self.cfg.strict_suv:
                raise RuntimeError(f"Cannot compute SUV for {path}")
            suv = activity
        tensor = self._normalize_pet(suv, is_suv=suv_ok)
        meta = {
            "pet_sop_uid": str(getattr(ds, "SOPInstanceUID", "")),
            "pet_series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "pet_instance_number": str(getattr(ds, "InstanceNumber", "")),
            "pet_suv_computed": suv_ok,
        }
        return tensor, meta

    def _process_pet_png(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = self._resize_to_tensor(arr)
        lo, hi = _PET_TARGET_RANGE
        tensor = tensor * (hi - lo) + lo
        return tensor, {"pet_source": "png"}

    def _process_label_png(self, path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        mask = torch.from_numpy(arr).unsqueeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="nearest",
        ).squeeze(0)
        mask = (mask > 0.5).float()
        return mask, {"has_label": True, "label_source": "png"}

    # ------------------------------------------------------------------
    # normalisation
    # ------------------------------------------------------------------

    def _normalize_ct(self, hu: np.ndarray) -> torch.Tensor:
        hu = np.clip(hu, self.cfg.ct_hu_min, self.cfg.ct_hu_max)
        arr_01 = (hu - self.cfg.ct_hu_min) / (self.cfg.ct_hu_max - self.cfg.ct_hu_min)
        arr_01 = np.clip(arr_01, 0.0, 1.0)
        tensor = self._resize_to_tensor(arr_01)
        lo, hi = _CT_TARGET_RANGE
        return tensor * (hi - lo) + lo

    def _normalize_pet(self, values: np.ndarray, *, is_suv: bool) -> torch.Tensor:
        if is_suv:
            arr_01 = np.clip(values / self.cfg.pet_suv_max, 0.0, 1.0)
        else:
            v_min, v_max = float(values.min()), float(values.max())
            if v_max - v_min < 1e-8:
                arr_01 = np.zeros_like(values, dtype=np.float32)
            else:
                arr_01 = (values - v_min) / (v_max - v_min)
        tensor = self._resize_to_tensor(arr_01)
        lo, hi = _PET_TARGET_RANGE
        return tensor * (hi - lo) + lo

    @staticmethod
    def _resize_to_tensor(arr: np.ndarray, size: int = 128) -> torch.Tensor:
        """arr: (H, W) float → tensor (1, size, size)."""
        t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(
            t, size=(size, size), mode="bilinear", align_corners=False,
        )
        return t.squeeze(0)


# ===================================================================
# Phase 2: Online dataset (fast — no DICOM at runtime)
# ===================================================================


@dataclass
class SampleEntry:
    """Lightweight index entry pointing into the cache."""

    sample_id: str
    patient_id: str
    slice_id: int
    split: str
    cache_path: Path
    has_mask: bool = False
    patient_slice_order: int = 0  # per-patient ordinal for 2.5D neighbour lookup


class CachedDataset(Dataset):
    """Fast training dataset backed by preprocessed ``.npz`` cache.

    ``__getitem__`` does **no** DICOM parsing, **no** PIL decode, and
    **no** online normalisation.  Everything is precomputed by
    ``CacheBuilder``, so the hot path is::

        np.load → torch.as_tensor → augment → pin_memory → GPU

    Parameters
    ----------
    cache_dir:
        Directory containing ``{sample_id}.npz`` files.
    split:
        Which split to serve (``"train"``, ``"val"``, ``"test"``).
    augment:
        Enable horizontal-flip augmentation (train only).
    neighbor_slices:
        Number of adjacent axial slices to stack as extra channels.
        0 = single slice; 2 = ±1 neighbour → 3-channel input.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str = "train",
        *,
        augment: bool = False,
        neighbor_slices: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.augment = augment
        self.neighbor_slices = int(neighbor_slices)

        self.entries: List[SampleEntry] = []
        self._patient_slice_map: Dict[str, Dict[int, SampleEntry]] = defaultdict(dict)

        for npz_path in sorted(self.cache_dir.glob(f"*{_CACHE_EXT}")):
            sid = _basename(str(npz_path))
            pid, slc = _parse_sample_id(sid)

            meta_path = self.cache_dir / f"{sid}_meta.json"
            entry_split = split
            has_mask = False
            if meta_path.exists():
                try:
                    meta = json.loads(open(meta_path, encoding="utf-8").read())
                    entry_split = meta.get("split", split)
                    has_mask = meta.get("has_label", False)
                except (json.JSONDecodeError, OSError):
                    pass

            if entry_split != self.split:
                continue

            entry = SampleEntry(
                sample_id=sid,
                patient_id=pid or sid,
                slice_id=slc or 0,
                split=entry_split,
                cache_path=npz_path,
                has_mask=has_mask,
            )
            self.entries.append(entry)
            if pid is not None and slc is not None:
                self._patient_slice_map[pid][slc] = entry

        # assign per-patient slice ordinals for 2.5D neighbour lookup
        if self.neighbor_slices > 0:
            for pid, slc_map in self._patient_slice_map.items():
                for order, (_, e) in enumerate(sorted(slc_map.items())):
                    e.patient_slice_order = order

        if not self.entries:
            raise ValueError(
                f"No cached samples for split={self.split!r} in {self.cache_dir}"
            )

        self._aug_fn: Optional[Callable] = (
            T.RandomHorizontalFlip(p=0.5) if augment else None
        )
        n_masked = sum(1 for e in self.entries if e.has_mask)
        print(
            f"[CachedDataset] split={self.split!r}  "
            f"samples={len(self.entries)}  masked={n_masked}  "
            f"neighbors={self.neighbor_slices}"
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]

        data = np.load(entry.cache_path)
        ct = torch.from_numpy(data["ct"].copy()).float()   # (1, H, W) in [-1,1]
        pet = torch.from_numpy(data["pet"].copy()).float()  # (1, H, W) in [-1,1]
        mask = torch.zeros(1, ct.shape[1], ct.shape[2])
        if entry.has_mask and "mask" in data:
            mask = torch.from_numpy(data["mask"].copy()).float()

        if self.neighbor_slices > 0:
            ct = self._stack_neighbors(entry, "ct")
            pet = self._stack_neighbors(entry, "pet")

        if self._aug_fn is not None:
            stacked = torch.cat([ct, pet, mask], dim=0)
            stacked = self._aug_fn(stacked)
            ct, pet, mask = stacked[0:1], stacked[1:2], stacked[2:3]

        return {"ct": ct, "pet": pet, "mask": mask}

    # ------------------------------------------------------------------
    # 2.5D neighbour stacking
    # ------------------------------------------------------------------

    def _stack_neighbors(self, entry: SampleEntry, modality: str) -> torch.Tensor:
        pid = entry.patient_id
        if pid not in self._patient_slice_map:
            return self._load_single(entry, modality)

        neighbors = self._patient_slice_map[pid]
        sorted_slices = sorted(neighbors.items())
        center_idx = next(
            (i for i, (_, e) in enumerate(sorted_slices)
             if e.sample_id == entry.sample_id), None
        )
        if center_idx is None:
            return self._load_single(entry, modality)

        channels = []
        N = self.neighbor_slices
        for offset in range(-N, N + 1):
            idx = max(0, min(len(sorted_slices) - 1, center_idx + offset))
            channels.append(self._load_single(sorted_slices[idx][1], modality))
        return torch.cat(channels, dim=0)

    @staticmethod
    def _load_single(entry: SampleEntry, modality: str) -> torch.Tensor:
        data = np.load(entry.cache_path)
        return torch.from_numpy(data[modality].copy()).float()


# ===================================================================
# DataLoader factory (backward-compatible config interface)
# ===================================================================


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Build train+val ``DataLoader`` from a configuration dict.

    Supports two modes:

    * **Cache mode** (recommended): ``cache_dir`` is set in config.
      Trains on preprocessed ``.npz`` files — no DICOM at runtime.
    * **Legacy PNG mode** (fallback): ``train_pet_path`` / ``train_ct_path``
      are set.  Uses online DICOM reading (slower; kept for compat).

    Returns ``(train_loader, val_loader)``.  Each batch dict contains
    ``ct``, ``pet``, and ``mask`` keys.  Training loops that only access
    ``ct`` / ``pet`` are unaffected.
    """
    cache_dir = config.get("cache_dir", None)

    if cache_dir and Path(cache_dir).is_dir():
        return _dataloaders_from_cache(config)
    return _dataloaders_from_png(config)


def _dataloaders_from_cache(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    cache_dir = config["cache_dir"]
    num_w = int(config.get("num_workers", 4))
    bs = int(config["batch_size"])
    vbs = int(config.get("val_batch_size", bs))
    seed = int(config.get("seed", 42))
    neighbors = int(config.get("neighbor_slices", 0))

    train_ds = CachedDataset(
        cache_dir, split="train",
        augment=config.get("augment", True),
        neighbor_slices=neighbors,
    )
    val_ds = CachedDataset(
        cache_dir, split="val",
        augment=False,
        neighbor_slices=neighbors,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        num_workers=num_w, pin_memory=True,
        generator=g, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=vbs, shuffle=False,
        num_workers=num_w, pin_memory=True,
    )
    return train_loader, val_loader


def _dataloaders_from_png(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Legacy PNG+DICOM loader path.

    The legacy ``MedicalImageDataset`` / ``PairedDataset`` classes have
    been superseded by the offline-preprocessing pipeline.  To use the
    current pipeline:

        1. Run ``python -m src.data.dataset preprocess`` to build a cache.
        2. Set ``cache_dir`` in your config YAML.
        3. Training will use ``CachedDataset`` automatically.
    """
    raise NotImplementedError(
        "Legacy PNG-mode loading has been removed. "
        "Run 'python -m src.data.dataset preprocess --manifest <csv> --out-dir <dir>' "
        "to build a cache, then set 'cache_dir' in your config."
    )


# ===================================================================
# CLI: offline preprocessing entry point
# ===================================================================


def _cli_preprocess(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Offline preprocessing: manifest CSV → .npz cache",
    )
    ap.add_argument(
        "--manifest", required=True, type=Path,
        help="Path to manifest_all.csv (from build_exposed_dataset.py)",
    )
    ap.add_argument(
        "--out-dir", required=True, type=Path,
        help="Output directory for .npz cache files",
    )
    ap.add_argument("--ct-hu-min", type=float, default=-150.0)
    ap.add_argument("--ct-hu-max", type=float, default=250.0)
    ap.add_argument("--pet-suv-max", type=float, default=20.0)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--no-spatial-check", action="store_true",
                     help="Skip CT-PET z-coordinate alignment check")
    ap.add_argument("--z-tolerance-mm", type=float, default=2.0)
    args = ap.parse_args(argv)

    cfg = PreprocessConfig(
        ct_hu_min=args.ct_hu_min,
        ct_hu_max=args.ct_hu_max,
        pet_suv_max=args.pet_suv_max,
        image_size=args.image_size,
    )
    builder = CacheBuilder(cfg)
    print(f"[preprocess] Building cache from {args.manifest} → {args.out_dir} ...")
    stats = builder.build(
        args.manifest, args.out_dir,
        validate_spatial=not args.no_spatial_check,
        z_tolerance_mm=args.z_tolerance_mm,
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    print(f"[preprocess] Done.  {stats['built']}/{stats['total_rows']} samples cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_preprocess())
