"""Dataset and DataLoader builders for SLMF-BBDM.

Supports:
  - NPZ cached tensor datasets (preprocessed offline)
  - Fake (random) dataset for smoke tests
  - Optional keys (organ_mask, mu_map, semantic_tokens, meta) with zero-fallback
  - Config-driven DataLoader with pin_memory, persistent_workers, prefetch_factor

Architecture:
  Phase 1 (offline): CacheBuilder reads DICOM → computes HU/SUV → writes .npz
  Phase 2 (online):  CachedDataset np.load → torch.as_tensor → augment → GPU
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

_CACHE_EXT = ".npz"
_SAMPLE_ID_PATTERN: re.Pattern = re.compile(r"^(\d{3})(\d{3})$")
_CT_TARGET_RANGE = (-1.0, 1.0)
_PET_TARGET_RANGE = (-1.0, 1.0)


def _epoch_sample_seed(base_seed: int, epoch: int, sample_index: int) -> int:
    """Return a stable, platform-independent seed for one sample in one epoch."""
    # Keep the result in torch.Generator.manual_seed's signed 63-bit range.
    mixed = (
        int(base_seed)
        + 0x4F1BBCDCBFA54001 * (int(epoch) + 1)
        + 0x369DEA0F31A53F85 * (int(sample_index) + 1)
    )
    return mixed & ((1 << 63) - 1)


def _split_epoch_index(index: Any) -> Tuple[int, int]:
    """Accept ordinary indices and epoch-stamped indices from our sampler."""
    if isinstance(index, tuple) and len(index) == 2:
        epoch, sample_index = index
        return int(epoch), int(sample_index)
    return 0, int(index)


class EpochShuffleSampler(Sampler[Tuple[int, int]]):
    """Shuffle with an explicit RNG and stamp every index with its epoch.

    Stamping the index lets dataset augmentation depend only on
    ``(seed, epoch, sample_index)``.  It therefore remains exactly reproducible
    across checkpoint resume even when DataLoader persistent workers are used.
    """

    def __init__(
        self,
        data_source: Dataset,
        *,
        generator: torch.Generator,
    ) -> None:
        self.data_source = data_source
        self.generator = generator
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[Tuple[int, int]]:
        indices = torch.randperm(
            len(self.data_source),
            generator=self.generator,
        ).tolist()
        epoch = self.epoch
        return iter((epoch, int(index)) for index in indices)

    def __len__(self) -> int:
        return len(self.data_source)


# ---------------------------------------------------------------------------
# Utilities
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


def _parse_age_string(age_str: str) -> Optional[float]:
    """Parse DICOM AS string like '052Y', '006M', '010W' → years."""
    if not age_str:
        return None
    s = str(age_str).strip()
    if len(s) < 2:
        return None
    try:
        val = float(s[:-1])
    except ValueError:
        return None
    unit = s[-1].upper()
    if unit == "Y":
        return val
    elif unit == "M":
        return val / 12.0
    elif unit == "W":
        return val / 52.0
    elif unit == "D":
        return val / 365.0
    return None


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
    import pydicom
    ds = pydicom.dcmread(path)
    pixels = ds.pixel_array.astype(np.float32)
    pixels = _ensure_2d(pixels)
    slope = _safe_float(getattr(ds, "RescaleSlope", None), 1.0)
    intercept = _safe_float(getattr(ds, "RescaleIntercept", None), 0.0)
    return ds, (pixels * slope + intercept).astype(np.float32)


def _parse_dcm_datetime(date_str: Optional[str], time_str: Optional[str]) -> Optional[datetime]:
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


def compute_suv(ds: Any, activity_conc: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
    """Compute SUV from PET DICOM. Returns (suv, suv_ok, meta_dict)."""
    units = str(getattr(ds, "Units", "")).upper()
    if units.startswith("SUV") or units == "GML":
        return activity_conc, True, {}
    if units in {"BQML", "BQCC", ""}:
        activity_bqml = activity_conc
    elif units in {"KBQML", "KBQCC"}:
        activity_bqml = activity_conc * 1000.0
    else:
        return activity_conc, False, {}

    weight_kg = _safe_float(getattr(ds, "PatientWeight", None), -1.0)
    if weight_kg <= 0:
        return activity_conc, False, {}
    rad_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not rad_seq:
        return activity_conc, False, {}
    rad = rad_seq[0]
    total_dose = _safe_float(getattr(rad, "RadionuclideTotalDose", None), -1.0)
    if total_dose <= 0:
        return activity_conc, False, {}
    half_life = _safe_float(getattr(rad, "RadionuclideHalfLife", None), -1.0)

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
        return activity_conc, False, {}
    suv = activity_bqml * (weight_kg * 1000.0) / decayed_dose
    meta = {
        "uptake_min": decay_seconds / 60.0,
        "weight_kg": float(weight_kg),
    }
    return suv.astype(np.float32), True, meta


# ---------------------------------------------------------------------------
# Phase 1: Offline preprocessing (CacheBuilder)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    ct_hu_min: float = -200.0
    ct_hu_max: float = 500.0
    pet_suv_max: float = 50.0
    image_size: int = 192
    strict_suv: bool = True


class CacheBuilder:
    """Offline pipeline: manifest CSV → normalised .npz cache."""

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.cfg = config or PreprocessConfig()
        self._errors: List[str] = []
        self._stats: Dict[str, Any] = {}

    def build(self, manifest_csv: Path, output_dir: Path, *, validate_spatial: bool = True, z_tolerance_mm: float = 2.0) -> Dict[str, Any]:
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
        self._stats = {
            "total_rows": len(rows), "built": built, "skipped": dict(skipped),
            "errors": self._errors[:50],
        }
        return self._stats

    def _process_row(self, row, out_dir, validate, z_tol, skipped):
        sid = row.get("sample_id", "")
        split = row.get("split", "")
        if not sid or split not in {"train", "val", "test"}:
            skipped["invalid_split"] += 1
            return False
        ct_dcm = row.get("ct_dicom_path", "")
        pet_dcm = row.get("pet_dicom_path", "")
        has_dicom = bool(ct_dcm and pet_dcm and Path(ct_dcm).exists() and Path(pet_dcm).exists())
        if not has_dicom:
            skipped["no_source"] += 1
            return False
        if validate:
            ct_z = _safe_float(row.get("ct_dicom_z"), None)
            pet_z = _safe_float(row.get("pet_dicom_z"), None)
            if (ct_z is not None and pet_z is not None and abs(ct_z - pet_z) > z_tol):
                skipped["spatial_mismatch"] += 1
                return False
        try:
            ds_ct, ct_hu = _read_dicom_pixels(ct_dcm)
            mu_map = 0.096 * (ct_hu / 1000.0) + 0.096
            mu_tensor = _resize_to_tensor(mu_map, self.cfg.image_size)
            ct_hu_tensor = _resize_to_tensor(ct_hu, self.cfg.image_size)
            ct_hu_clip = np.clip(ct_hu, self.cfg.ct_hu_min, self.cfg.ct_hu_max)
            ct_norm = (ct_hu_clip - self.cfg.ct_hu_min) / (self.cfg.ct_hu_max - self.cfg.ct_hu_min)
            ct_tensor = _resize_to_tensor(ct_norm, self.cfg.image_size)
            ct_tensor = ct_tensor * (_CT_TARGET_RANGE[1] - _CT_TARGET_RANGE[0]) + _CT_TARGET_RANGE[0]

            ds_pet, activity = _read_dicom_pixels(pet_dcm)
            suv, suv_ok, suv_meta = compute_suv(ds_pet, activity)
            pet_activity_tensor = None
            pet_suv_tensor = None
            if not suv_ok:
                if self.cfg.strict_suv:
                    raise RuntimeError(f"Cannot compute SUV for {pet_dcm}")
                suv = activity
            if suv_ok:
                pet_suv_tensor = _resize_to_tensor(suv, self.cfg.image_size)
                pet_norm = np.clip(suv / self.cfg.pet_suv_max, 0.0, 1.0)
            else:
                pet_activity_tensor = _resize_to_tensor(suv, self.cfg.image_size)
                v_min, v_max = float(suv.min()), float(suv.max())
                pet_norm = (suv - v_min) / max(v_max - v_min, 1e-8)
            pet_tensor = _resize_to_tensor(pet_norm, self.cfg.image_size)
            pet_tensor = pet_tensor * (_PET_TARGET_RANGE[1] - _PET_TARGET_RANGE[0]) + _PET_TARGET_RANGE[0]

            mask_tensor = None
            label_path = row.get("pet_label_png_path", "") or row.get("ct_label_png_path", "")
            if label_path and Path(label_path).exists():
                img = Image.open(label_path).convert("L")
                arr = np.array(img, dtype=np.float32) / 255.0
                m = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                m = F.interpolate(m, size=(self.cfg.image_size, self.cfg.image_size), mode="nearest").squeeze(0)
                mask_tensor = (m > 0.5).float()
        except Exception as exc:
            self._errors.append(f"{sid}: {exc}")
            skipped["processing_error"] += 1
            return False

        # ---- DICOM metadata for model conditioning ----
        thickness = getattr(ds_ct, "SliceThickness", None)
        if thickness is not None:
            thickness = float(str(thickness))
        z_loc = getattr(ds_ct, "SliceLocation", None)
        if z_loc is not None:
            z_loc = float(str(z_loc))
        age_str = getattr(ds_ct, "PatientAge", None) or getattr(ds_pet, "PatientAge", None)
        age_years = _parse_age_string(str(age_str)) if age_str else None

        # ---- scale_meta: preserves physical-to-normalised mapping for loss terms ----
        pid, slc = _parse_sample_id(sid)
        scale_meta = {
            "pet_suv_max": float(self.cfg.pet_suv_max),
            "suv_ok": bool(suv_ok),
            "pet_raw_min": float(suv.min()),
            "pet_raw_max": float(suv.max()),
            "ct_hu_min": float(self.cfg.ct_hu_min),
            "ct_hu_max": float(self.cfg.ct_hu_max),
            "ct_physical_key": "ct_hu",
            "pet_physical_key": "pet_suv" if suv_ok else "pet_activity",
            "pet_physical_kind": "suv" if suv_ok else "activity",
            "pet_suv_available": bool(suv_ok),
            "patient_id": pid or sid,
            "slice_id": slc or 0,
        }
        # Inject DICOM metadata (None/absent = omitted from JSON)
        if suv_meta.get("uptake_min") is not None:
            scale_meta["uptake_min"] = float(suv_meta["uptake_min"])
        if suv_meta.get("weight_kg") is not None:
            scale_meta["weight_kg"] = float(suv_meta["weight_kg"])
        if age_years is not None:
            scale_meta["age_years"] = float(age_years)
        if thickness is not None:
            scale_meta["thickness_mm"] = float(thickness)
        if z_loc is not None:
            scale_meta["z_mm"] = float(z_loc)
        # Store as JSON string inside .npz (np.savez_compressed cannot nest dicts)
        import json as _json
        scale_meta_json = _json.dumps(scale_meta, ensure_ascii=False)

        payload = {
            "ct": ct_tensor.numpy().astype(np.float32),
            "ct_hu": ct_hu_tensor.numpy().astype(np.float32),
            "pet": pet_tensor.numpy().astype(np.float32),
            "mu_map": mu_tensor.numpy().astype(np.float32),
            "scale_meta_json": np.frombuffer(scale_meta_json.encode("utf-8"), dtype=np.uint8),
        }
        if pet_suv_tensor is not None:
            payload["pet_suv"] = pet_suv_tensor.numpy().astype(np.float32)
        if pet_activity_tensor is not None:
            payload["pet_activity"] = pet_activity_tensor.numpy().astype(np.float32)
        has_mask = False
        if mask_tensor is not None:
            payload["mask"] = mask_tensor.numpy().astype(np.float32)
            has_mask = True
        np.savez_compressed(str(out_dir / f"{sid}{_CACHE_EXT}"), **payload)
        meta = {
            "sample_id": sid, "split": split, "source": "dicom",
            "has_label": has_mask, "scale_meta": scale_meta,
        }
        with open(out_dir / f"{sid}_meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, default=str)
        return True


def _resize_to_tensor(arr: np.ndarray, size: int) -> torch.Tensor:
    t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0)


# ---------------------------------------------------------------------------
# Phase 2: Online datasets
# ---------------------------------------------------------------------------

@dataclass
class SampleEntry:
    sample_id: str
    patient_id: str
    slice_id: int
    split: str
    cache_path: Path
    has_mask: bool = False


class CachedDataset(Dataset):
    """Fast dataset backed by preprocessed .npz cache. No DICOM at runtime.

    When ``split_manifest`` is provided (recommended), uses patient-level split
    assignments from the manifest, guaranteeing no patient leaks across splits.
    Otherwise falls back to per-sample ``_meta.json`` split annotations.
    """

    def __init__(self, cache_dir: str | Path, split: str = "train", *, augment: bool = False,
                 split_manifest: Optional[Path] = None,
                 required_keys: Optional[List[str]] = None,
                 optional_keys: Optional[List[str]] = None,
                 augmentation_seed: int = 0):
        self.cache_dir = Path(cache_dir)
        self.split = split
        self.augment = augment
        self.augmentation_seed = int(augmentation_seed)
        self.entries: List[SampleEntry] = []
        self.required_keys = required_keys or ["ct", "pet"]
        self.optional_keys = optional_keys or []
        self._missing_key_counts: Dict[str, int] = {k: 0 for k in self.required_keys + self.optional_keys}

        # Load split manifest for strict patient-level splits
        manifest = None
        if split_manifest is not None and Path(split_manifest).exists():
            from .split_manifest import SplitManifest
            manifest = SplitManifest(Path(split_manifest))
            manifest.validate_no_overlap()

        for npz_path in sorted(self.cache_dir.glob(f"*{_CACHE_EXT}")):
            sid = _basename(str(npz_path))
            pid, slc = _parse_sample_id(sid)

            if manifest is not None:
                manifest_entry = manifest.get_entry(sid)
                if not manifest_entry:
                    continue  # skip samples not in manifest
                entry_split = manifest_entry["split"]
                # The manifest is authoritative.  In particular, external
                # cohorts may use site-prefixed IDs that cannot be recovered
                # safely from a cache filename pattern.
                pid = str(manifest_entry["patient_id"])
                raw_slice_id = manifest_entry.get("slice_id", "")
                try:
                    slc = int(float(raw_slice_id))
                except (TypeError, ValueError):
                    _, parsed_slice = _parse_sample_id(sid)
                    slc = parsed_slice if parsed_slice is not None else 0
            else:
                meta_path = self.cache_dir / f"{sid}_meta.json"
                entry_split = split
                if meta_path.exists():
                    try:
                        meta = json.loads(open(meta_path, encoding="utf-8").read())
                        entry_split = meta.get("split", split)
                    except (json.JSONDecodeError, OSError):
                        pass

            if entry_split != self.split:
                continue

            has_mask = False
            meta_path = self.cache_dir / f"{sid}_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(open(meta_path, encoding="utf-8").read())
                    has_mask = meta.get("has_label", False)
                except (json.JSONDecodeError, OSError):
                    pass

            self.entries.append(SampleEntry(
                sample_id=sid, patient_id=pid or sid, slice_id=slc or 0,
                split=entry_split, cache_path=npz_path, has_mask=has_mask,
            ))
        if not self.entries:
            raise ValueError(f"No cached samples for split={split!r} in {cache_dir}")
        patient_count = len(set(e.patient_id for e in self.entries))

        # ---- Key-presence scan (sample every Nth file to avoid I/O storm) ----
        scan_n = max(1, len(self.entries) // 20)
        scan_entries = self.entries[::scan_n]
        for entry in scan_entries:
            try:
                with np.load(entry.cache_path) as data:
                    keys = set(data.keys())
                for k in self.required_keys:
                    if k not in keys:
                        self._missing_key_counts[k] += 1
                for k in self.optional_keys:
                    if k not in keys:
                        self._missing_key_counts[k] += 1
            except (OSError, ValueError):
                pass

        print(f"[CachedDataset] split={split!r} samples={len(self.entries)} patients={patient_count} "
              f"masked={sum(1 for e in self.entries if e.has_mask)}"
              f"{' (manifest)' if manifest else ''}")
        # Warn about required key gaps
        for k in self.required_keys:
            missing = self._missing_key_counts.get(k, 0)
            if missing > 0:
                print(f"  WARNING: required key '{k}' missing in {missing}/{len(scan_entries)} scanned .npz files "
                      f"— downstream losses that depend on '{k}' may silently degrade")
        for k in self.optional_keys:
            missing = self._missing_key_counts.get(k, 0)
            if missing > 0:
                print(f"  optional key '{k}' missing in {missing}/{len(scan_entries)} scanned .npz files")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: Any) -> Dict[str, torch.Tensor]:
        epoch, idx = _split_epoch_index(idx)
        entry = self.entries[idx]
        with np.load(entry.cache_path) as data:
            missing_required = [k for k in self.required_keys if k not in data]
            if missing_required:
                raise KeyError(
                    f"Cached sample {entry.sample_id} is missing required key(s): "
                    f"{missing_required}. Rebuild the cache or remove them from data.required_keys."
                )
            ct = torch.from_numpy(data["ct"].copy()).float()
            pet = torch.from_numpy(data["pet"].copy()).float()
            H, W = ct.shape[1], ct.shape[2]

            mask = torch.zeros(1, H, W)
            if "mask" in data:
                mask = torch.from_numpy(data["mask"].copy()).float()

            # Read optional fields from .npz if present, else zero
            organ_mask = (
                torch.from_numpy(data["organ_mask"].copy()).float()
                if "organ_mask" in data
                else torch.zeros(6, H, W)
            )
            organ_distance = (
                torch.from_numpy(data["organ_distance"].copy()).float()
                if "organ_distance" in data
                else torch.zeros(6, H, W)
            )
            mu_map = (
                torch.from_numpy(data["mu_map"].copy()).float()
                if "mu_map" in data
                else torch.zeros(1, H, W)
            )
            has_ct_hu = "ct_hu" in data
            has_pet_suv = "pet_suv" in data
            has_pet_activity = "pet_activity" in data
            ct_hu = (
                torch.from_numpy(data["ct_hu"].copy()).float()
                if has_ct_hu
                else torch.zeros(1, H, W)
            )
            pet_suv = (
                torch.from_numpy(data["pet_suv"].copy()).float()
                if has_pet_suv
                else torch.zeros(1, H, W)
            )
            pet_activity = (
                torch.from_numpy(data["pet_activity"].copy()).float()
                if has_pet_activity
                else torch.zeros(1, H, W)
            )
            semantic_tokens = (
                torch.from_numpy(data["semantic_tokens"].copy()).float()
                if "semantic_tokens" in data
                else None
            )

            # Load scale_meta (stored as JSON bytes in .npz)
            scale_meta = {}
            if "scale_meta_json" in data:
                import json as _json
                try:
                    json_bytes = bytes(data["scale_meta_json"].tolist())
                    scale_meta = _json.loads(json_bytes.decode("utf-8"))
                except (UnicodeDecodeError, _json.JSONDecodeError):
                    pass

            scale_meta = dict(scale_meta)
            scale_meta.setdefault("ct_physical_key", "ct_hu" if has_ct_hu else "")
            scale_meta.setdefault("pet_physical_key", "pet_suv" if has_pet_suv else ("pet_activity" if has_pet_activity else ""))
            scale_meta.setdefault("pet_suv_available", bool(has_pet_suv))

        if self.augment:
            augment_generator = torch.Generator(device="cpu")
            augment_generator.manual_seed(
                _epoch_sample_seed(self.augmentation_seed, epoch, idx)
            )
            if torch.rand((), generator=augment_generator) < 0.5:
                stacked = torch.cat([ct, pet, mask, organ_mask, organ_distance, mu_map, ct_hu, pet_suv, pet_activity], dim=0)
                stacked = torch.flip(stacked, dims=(-1,))
                ct, pet, mask = stacked[0:1], stacked[1:2], stacked[2:3]
                organ_mask = stacked[3:9]
                organ_distance = stacked[9:15]
                mu_map = stacked[15:16]
                ct_hu = stacked[16:17]
                pet_suv = stacked[17:18]
                pet_activity = stacked[18:19]

        sample = {
            "ct": ct,
            "pet": pet,
            "mask": mask,
            "organ_mask": organ_mask,
            "organ_distance": organ_distance,
            "mu_map": mu_map,
            "ct_hu": ct_hu,
            "pet_suv": pet_suv,
            "pet_activity": pet_activity,
            "meta": scale_meta,
        }
        if semantic_tokens is not None:
            sample["semantic_tokens"] = semantic_tokens
        return sample


class FakeDataset(Dataset):
    """Synthetic dataset for smoke tests — no real data needed."""

    def __init__(
        self,
        num_samples: int = 32,
        image_size: int = 32,
        seed: int = 0,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: Any) -> Dict[str, torch.Tensor]:
        epoch, idx = _split_epoch_index(idx)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_epoch_sample_seed(self.seed, epoch, idx))
        H = W = self.image_size
        sample = {
            "ct": torch.randn(1, H, W, generator=generator) * 2 - 1,
            "pet": torch.randn(1, H, W, generator=generator) * 2 - 1,
            "mask": torch.zeros(1, H, W),
            "organ_mask": torch.zeros(6, H, W),
            "organ_distance": torch.zeros(6, H, W),
            "mu_map": torch.zeros(1, H, W),
            "ct_hu": torch.zeros(1, H, W),
            "pet_suv": torch.zeros(1, H, W),
            "pet_activity": torch.zeros(1, H, W),
            "meta": {"pet_suv_max": 20.0, "suv_ok": False, "pet_raw_min": -1.0, "pet_raw_max": 1.0,
                     "ct_hu_min": -150.0, "ct_hu_max": 250.0, "patient_id": f"fake_{idx}", "slice_id": 0,
                     "ct_physical_key": "", "pet_physical_key": "", "pet_suv_available": False,
                     "uptake_min": 60.0, "weight_kg": 65.0, "age_years": 52.0,
                     "thickness_mm": 2.0, "z_mm": 0.0},
        }
        cx = H // 2 + torch.randint(-3, 4, (1,), generator=generator).item()
        cy = W // 2 + torch.randint(-3, 4, (1,), generator=generator).item()
        sample["mask"][0, cx:cx+3, cy:cy+3] = 1.0
        return sample


# ---------------------------------------------------------------------------
# DataLoader factory (used by train_v2.py)
# ---------------------------------------------------------------------------

def build_dataloaders(data_cfg: Dict[str, Any], run_cfg: Dict[str, Any]) -> Tuple[DataLoader, Optional[DataLoader]]:
    image_size = data_cfg.get("image_size", 192)
    batch_size = data_cfg.get("batch_size", 4)
    val_batch_size = data_cfg.get("val_batch_size", 4)
    augment = data_cfg.get("augment", True)
    num_workers = run_cfg.get("num_workers", 4)
    pin_memory = run_cfg.get("pin_memory", True) and torch.cuda.is_available()
    persistent = run_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch = run_cfg.get("prefetch_factor", 2) if num_workers > 0 else None
    dataloader_seed = int(
        run_cfg.get(
            "dataloader_seed",
            run_cfg.get("eval_seed", run_cfg.get("seed", 42)),
        )
    )
    cache_dir = data_cfg.get("cache_dir", "")
    val_cache_dir = data_cfg.get("val_cache_dir", "")
    split_manifest_path = data_cfg.get("split_manifest", None)
    if split_manifest_path:
        split_manifest_path = Path(split_manifest_path)

    required_keys = data_cfg.get("required_keys", ["ct", "pet"])
    optional_keys = data_cfg.get("optional_keys", [])

    use_fake = data_cfg.get("use_fake_data", False)
    if use_fake:
        # Must be checked BEFORE the cache branch: once cache_dir exists,
        # the cache-first ordering would silently ignore use_fake_data=true,
        # turning a smoke test into a real training run on (possibly misaligned) data.
        print("[DataLoader] use_fake_data=True — using FakeDataset for smoke testing")
        train_ds = FakeDataset(32, image_size, seed=dataloader_seed)
        val_ds = FakeDataset(8, image_size, seed=dataloader_seed + 1)
    elif cache_dir and Path(cache_dir).is_dir():
        train_ds = CachedDataset(cache_dir, split="train", augment=augment,
                                 split_manifest=split_manifest_path,
                                 required_keys=required_keys, optional_keys=optional_keys,
                                 augmentation_seed=dataloader_seed)
        val_ds = None
        val_dir = val_cache_dir or cache_dir
        if Path(val_dir).is_dir():
            try:
                val_ds = CachedDataset(val_dir, split="val", augment=False,
                                       split_manifest=split_manifest_path,
                                       required_keys=required_keys, optional_keys=optional_keys,
                                       augmentation_seed=dataloader_seed + 1)
            except ValueError:
                print(f"[DataLoader] No val samples in {val_dir} — training without validation set")
    else:
        raise RuntimeError(
            f"Cache directory not found: '{cache_dir}'. "
            "To use fake/synthetic data for smoke testing, set 'data.use_fake_data: true' in your config."
        )

    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(dataloader_seed)
    worker_generator = torch.Generator(device="cpu")
    worker_generator.manual_seed(dataloader_seed + 1)
    train_sampler = EpochShuffleSampler(
        train_ds,
        generator=sampler_generator,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch,
        drop_last=True,
        generator=worker_generator,
    )
    val_loader = None
    if val_ds is not None:
        val_generator = torch.Generator(device="cpu")
        val_generator.manual_seed(dataloader_seed + 2)
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=min(num_workers, 1),
            pin_memory=pin_memory,
            drop_last=False,
            generator=val_generator,
        )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# CLI: preprocessing entry point
# ---------------------------------------------------------------------------

def _cli_preprocess(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline preprocessing: manifest CSV → .npz cache")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--ct-hu-min", type=float, default=-150.0)
    ap.add_argument("--ct-hu-max", type=float, default=250.0)
    ap.add_argument("--pet-suv-max", type=float, default=20.0)
    ap.add_argument("--image-size", type=int, default=192)
    ap.add_argument("--no-spatial-check", action="store_true")
    ap.add_argument("--z-tolerance-mm", type=float, default=2.0)
    ap.add_argument(
        "--allow-non-suv",
        action="store_true",
        help="Allow PET slices without valid SUV metadata; clinical SUV losses/metrics will ignore them.",
    )
    args = ap.parse_args(argv)
    cfg = PreprocessConfig(
        ct_hu_min=args.ct_hu_min,
        ct_hu_max=args.ct_hu_max,
        pet_suv_max=args.pet_suv_max,
        image_size=args.image_size,
        strict_suv=not args.allow_non_suv,
    )
    builder = CacheBuilder(cfg)
    print(f"[preprocess] Building cache from {args.manifest} → {args.out_dir} ...")
    stats = builder.build(args.manifest, args.out_dir, validate_spatial=not args.no_spatial_check, z_tolerance_mm=args.z_tolerance_mm)
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    print(f"[preprocess] Done. {stats['built']}/{stats['total_rows']} samples cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_preprocess())
