"""Lightweight PNG dataset for comparison experiments.

This module intentionally avoids torchvision so the comparison pipeline can run
in environments where torch is installed but torchvision is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
_CT_TARGET_RANGE = (-1.0, 1.0)
_PET_TARGET_RANGE = (-1.0, 1.0)


def _parse_sample_id(sample_id: str) -> Tuple[str, int]:
    if len(sample_id) >= 6 and sample_id[:6].isdigit():
        return sample_id[:3], int(sample_id[3:6])
    return sample_id, 0


def _read_png_unit(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    raw = np.asarray(img)
    if raw.ndim == 3:
        raw = np.asarray(img.convert("L"))
    arr = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        denom = float(np.iinfo(raw.dtype).max)
    else:
        denom = 1.0 if float(arr.max()) <= 1.0 else float(arr.max())
    return np.clip(arr / max(denom, 1.0), 0.0, 1.0).astype(np.float32)


def _unit_to_range(arr: np.ndarray, target_range: Tuple[float, float]) -> np.ndarray:
    lo, hi = target_range
    return (arr * (hi - lo) + lo).astype(np.float32)


def _resize_to_tensor(arr: np.ndarray, size: int, *, mode: str = "bilinear") -> torch.Tensor:
    tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        tensor = F.interpolate(tensor, size=(size, size), mode=mode)
    else:
        tensor = F.interpolate(tensor, size=(size, size), mode=mode, align_corners=False)
    return tensor.squeeze(0)


@dataclass
class PNGEntry:
    sample_id: str
    patient_id: str
    slice_id: int
    split: str
    ct_path: Path
    pet_path: Path
    label_path: Optional[Path]


class LightweightPNGSliceDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        image_size: int = 192,
        ct_dir: str = "ct",
        pet_dir: str = "pet_peizhuan",
        label_dir: str = "label",
        augment: bool = False,
        require_label: bool = True,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.require_label = bool(require_label)

        split_root = self.root / split
        ct_root = split_root / ct_dir
        pet_root = split_root / pet_dir
        label_root = split_root / label_dir if label_dir else None
        if not ct_root.is_dir():
            raise FileNotFoundError(f"CT PNG directory not found: {ct_root}")
        if not pet_root.is_dir():
            raise FileNotFoundError(f"PET PNG directory not found: {pet_root}")
        if self.require_label and (label_root is None or not label_root.is_dir()):
            raise FileNotFoundError(f"Label PNG directory not found: {label_root}")

        ct_files = self._scan(ct_root)
        pet_files = self._scan(pet_root)
        label_files = self._scan(label_root) if label_root and label_root.is_dir() else {}
        sample_ids = sorted(set(ct_files) & set(pet_files))
        if self.require_label:
            sample_ids = [sid for sid in sample_ids if sid in label_files]
        if not sample_ids:
            raise ValueError(f"No paired PNG samples for split={split!r} under {self.root}")

        self.entries = []
        for sid in sample_ids:
            pid, slc = _parse_sample_id(sid)
            self.entries.append(PNGEntry(sid, pid, slc, split, ct_files[sid], pet_files[sid], label_files.get(sid)))

        patient_count = len({entry.patient_id for entry in self.entries})
        print(
            f"[LightweightPNGSliceDataset] split={split!r} samples={len(self.entries)} "
            f"patients={patient_count} image_size={self.image_size} pet_dir={pet_dir!r}"
        )

    @staticmethod
    def _scan(root: Optional[Path]) -> Dict[str, Path]:
        if root is None or not root.is_dir():
            return {}
        out = {}
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
                if path.stem in out:
                    raise ValueError(f"Duplicate sample_id {path.stem!r} in {root}")
                out[path.stem] = path
        return out

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        ct_unit = _read_png_unit(entry.ct_path)
        pet_unit = _read_png_unit(entry.pet_path)
        ct_h, ct_w = int(ct_unit.shape[0]), int(ct_unit.shape[1])
        pet_h, pet_w = int(pet_unit.shape[0]), int(pet_unit.shape[1])

        ct = _resize_to_tensor(_unit_to_range(ct_unit, _CT_TARGET_RANGE), self.image_size)
        pet = _resize_to_tensor(_unit_to_range(pet_unit, _PET_TARGET_RANGE), self.image_size)

        if entry.label_path is not None:
            label_unit = _read_png_unit(entry.label_path)
            label_h, label_w = int(label_unit.shape[0]), int(label_unit.shape[1])
            mask = (_resize_to_tensor(label_unit, self.image_size, mode="nearest") > 0.5).float()
        else:
            label_h, label_w = 0, 0
            mask = torch.zeros(1, self.image_size, self.image_size)

        if self.augment and torch.rand(()) < 0.5:
            ct = torch.flip(ct, dims=(-1,))
            pet = torch.flip(pet, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        organ_mask = torch.zeros(6, self.image_size, self.image_size)
        organ_distance = torch.zeros(6, self.image_size, self.image_size)
        mu_map = torch.zeros(1, self.image_size, self.image_size)
        ct_hu = torch.zeros(1, self.image_size, self.image_size)
        pet_suv = torch.zeros(1, self.image_size, self.image_size)
        pet_activity = _resize_to_tensor(pet_unit, self.image_size)

        meta = {
            "sample_id": entry.sample_id,
            "patient_id": entry.patient_id,
            "slice_id": entry.slice_id,
            "split": entry.split,
            "source": "png",
            "ct_path": str(entry.ct_path),
            "pet_path": str(entry.pet_path),
            "label_path": str(entry.label_path) if entry.label_path is not None else "",
            "model_image_size": self.image_size,
            "ct_original_h": ct_h,
            "ct_original_w": ct_w,
            "pet_original_h": pet_h,
            "pet_original_w": pet_w,
            "label_original_h": label_h,
            "label_original_w": label_w,
            "pet_raw_min": float(pet_unit.min()),
            "pet_raw_max": float(pet_unit.max()),
            "pet_physical_kind": "png_unit",
            "ct_physical_key": "",
            "pet_physical_key": "pet_activity",
            "pet_suv_available": False,
            "suv_ok": False,
        }
        return {
            "ct": ct,
            "pet": pet,
            "mask": mask,
            "organ_mask": organ_mask,
            "organ_distance": organ_distance,
            "mu_map": mu_map,
            "ct_hu": ct_hu,
            "pet_suv": pet_suv,
            "pet_activity": pet_activity,
            "meta": meta,
        }


def build_png_dataloaders(data_cfg: Dict[str, Any], run_cfg: Dict[str, Any]):
    png_root = data_cfg.get("png_root")
    if not png_root:
        raise RuntimeError("comparison_experiments requires data.png_root for PNG experiments")
    image_size = int(data_cfg.get("image_size", 192))
    batch_size = int(data_cfg.get("batch_size", 4))
    val_batch_size = int(data_cfg.get("val_batch_size", batch_size))
    num_workers = int(run_cfg.get("num_workers", 0))
    pin_memory = bool(run_cfg.get("pin_memory", True)) and torch.cuda.is_available()
    persistent = bool(run_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch = int(run_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    train_ds = LightweightPNGSliceDataset(
        png_root,
        split=data_cfg.get("train_split_name", "train"),
        image_size=image_size,
        ct_dir=data_cfg.get("png_ct_dir", "ct"),
        pet_dir=data_cfg.get("png_pet_dir", "pet_peizhuan"),
        label_dir=data_cfg.get("png_label_dir", "label"),
        augment=bool(data_cfg.get("augment", True)),
        require_label=bool(data_cfg.get("png_require_label", True)),
    )
    val_ds = LightweightPNGSliceDataset(
        png_root,
        split=data_cfg.get("val_split_name", "val"),
        image_size=image_size,
        ct_dir=data_cfg.get("png_ct_dir", "ct"),
        pet_dir=data_cfg.get("png_pet_dir", "pet_peizhuan"),
        label_dir=data_cfg.get("png_label_dir", "label"),
        augment=False,
        require_label=bool(data_cfg.get("png_require_label", True)),
    )

    kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
        "drop_last": True,
    }
    if prefetch is not None:
        kwargs["prefetch_factor"] = prefetch
    train_loader = DataLoader(train_ds, **kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 1),
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader

