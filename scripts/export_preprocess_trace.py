import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import yaml

# Add project root to PYTHONPATH for local imports when running from scripts/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_input_path(path_str: str) -> str:
    """
    Normalize user input path.
    Supports:
    - Linux paths directly
    - Windows-style absolute path (e.g. D:\\foo\\bar.png) in WSL (/mnt/d/foo/bar.png)
    """
    raw = str(path_str).strip().strip('"').strip("'")
    candidates: List[str] = []

    # As-is
    candidates.append(raw)

    # Replace backslashes
    candidates.append(raw.replace("\\", "/"))

    # Windows drive to WSL style
    if re.match(r"^[A-Za-z]:[\\/].*", raw):
        drive = raw[0].lower()
        tail = raw[2:].replace("\\", "/").lstrip("/")
        candidates.append(f"/mnt/{drive}/{tail}")

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    tried = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(f"Cannot resolve path: {path_str}\nTried:\n{tried}")


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _save_gray(path: str, arr: np.ndarray):
    arr_u8 = _to_uint8_image(arr)
    Image.fromarray(arr_u8, mode="L").save(path)


def _tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
    }


def _array_stats(x: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
    }


def _find_matching_ct(
    pet_path: str,
    config: Dict[str, Any],
    ct_path_override: Optional[str] = None,
) -> str:
    if ct_path_override:
        return _normalize_input_path(ct_path_override)

    pet_path_obj = Path(pet_path)
    sample_id = pet_path_obj.stem
    ext = pet_path_obj.suffix

    # Prefer split-consistent CT folder first.
    preferred: List[str] = []
    fallback: List[str] = []
    train_pet = config.get("train_pet_path")
    train_ct = config.get("train_ct_path")
    val_pet = config.get("val_pet_path")
    val_ct = config.get("val_ct_path")

    abs_pet = os.path.abspath(pet_path)
    if train_pet and train_ct and abs_pet.startswith(os.path.abspath(train_pet)):
        preferred.append(train_ct)
    if val_pet and val_ct and abs_pet.startswith(os.path.abspath(val_pet)):
        preferred.append(val_ct)

    for d in [train_ct, val_ct]:
        if d and d not in preferred:
            fallback.append(d)

    search_dirs = preferred + fallback
    for d in search_dirs:
        if not d:
            continue
        c1 = os.path.join(d, sample_id + ext)
        if os.path.exists(c1):
            return os.path.abspath(c1)
        # fallback to any extension
        for fn in os.listdir(d):
            if Path(fn).stem == sample_id:
                return os.path.abspath(os.path.join(d, fn))

    raise FileNotFoundError(
        f"Cannot auto-find CT pair for PET sample_id={sample_id}. "
        "Please pass --ct-path explicitly."
    )


def _save_stage_bundle(
    out_dir: str,
    stage_name: str,
    tensor_01: torch.Tensor,
    tensor_m11: Optional[torch.Tensor] = None,
):
    """
    Save stage image/tensor files.
    tensor_01: [1,H,W] in [0,1]
    tensor_m11: [1,H,W] in [-1,1] (optional)
    """
    os.makedirs(out_dir, exist_ok=True)
    np_01 = tensor_01.squeeze(0).detach().cpu().numpy()
    _save_gray(os.path.join(out_dir, f"{stage_name}.png"), np_01)
    np.save(os.path.join(out_dir, f"{stage_name}.npy"), np_01.astype(np.float32))
    torch.save(tensor_01.detach().cpu(), os.path.join(out_dir, f"{stage_name}.pt"))

    if tensor_m11 is not None:
        np_m11 = tensor_m11.squeeze(0).detach().cpu().numpy()
        np_m11_vis = np.clip((np_m11 + 1.0) * 0.5, 0.0, 1.0)
        _save_gray(os.path.join(out_dir, f"{stage_name}_minus1_1_vis.png"), np_m11_vis)
        np.save(
            os.path.join(out_dir, f"{stage_name}_minus1_1.npy"),
            np_m11.astype(np.float32),
        )
        torch.save(
            tensor_m11.detach().cpu(),
            os.path.join(out_dir, f"{stage_name}_minus1_1.pt"),
        )


def _write_markdown_report(path: str, report: Dict[str, Any]):
    lines: List[str] = []
    lines.append("# Preprocess Trace Report")
    lines.append("")
    lines.append("## Input")
    lines.append(f"- PET path: `{report['input']['pet_path']}`")
    lines.append(f"- CT path: `{report['input']['ct_path']}`")
    lines.append(f"- Config: `{report['input']['config_path']}`")
    lines.append("")
    lines.append("## Effective Settings")
    for k, v in report["settings"].items():
        lines.append(f"- {k}: `{v}`")
    lines.append("")
    lines.append("## Stage Statistics")
    for name, stats in report["stages"].items():
        lines.append(f"### {name}")
        for k in ["shape", "dtype", "min", "max", "mean", "std"]:
            if k in stats:
                lines.append(f"- {k}: `{stats[k]}`")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def export_trace(
    config_path: str,
    pet_path: str,
    output_dir: str,
    ct_path: Optional[str] = None,
    apply_augment: bool = False,
    seed: int = 42,
):
    cfg = _load_config(config_path)
    pet_path = _normalize_input_path(pet_path)
    ct_path = _find_matching_ct(pet_path, cfg, ct_path_override=ct_path)

    image_size = int(cfg.get("image_size", 128))
    invert_pet = bool(cfg.get("invert_pet", False))
    augment_enabled = bool(cfg.get("augment", False))
    augment_applied = False

    os.makedirs(output_dir, exist_ok=True)
    tensor_dir = os.path.join(output_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)

    # Stage 0: raw images
    pet_raw_img = Image.open(pet_path).convert("L")
    ct_raw_img = Image.open(ct_path).convert("L")
    pet_raw_np = np.array(pet_raw_img, dtype=np.uint8)
    ct_raw_np = np.array(ct_raw_img, dtype=np.uint8)
    _save_gray(os.path.join(output_dir, "00_pet_raw_uint8.png"), pet_raw_np)
    _save_gray(os.path.join(output_dir, "00_ct_raw_uint8.png"), ct_raw_np)
    np.save(os.path.join(tensor_dir, "pet_raw_uint8.npy"), pet_raw_np)
    np.save(os.path.join(tensor_dir, "ct_raw_uint8.npy"), ct_raw_np)

    # Stage 1: same as dataset._load_png_pair transform
    transform_to_tensor = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.CenterCrop(image_size),
            T.ToTensor(),  # [1,H,W], [0,1]
        ]
    )
    pet_01 = transform_to_tensor(pet_raw_img)
    ct_01 = transform_to_tensor(ct_raw_img)

    # Optional augmentation (same logic as dataset.__getitem__)
    if apply_augment and augment_enabled:
        random.seed(seed)
        if random.random() > 0.5:
            pet_01 = T.functional.hflip(pet_01)
            ct_01 = T.functional.hflip(ct_01)
            augment_applied = True

    _save_stage_bundle(
        out_dir=tensor_dir,
        stage_name="01_pet_after_resize_crop_01",
        tensor_01=pet_01,
    )
    _save_stage_bundle(
        out_dir=tensor_dir,
        stage_name="01_ct_after_resize_crop_01",
        tensor_01=ct_01,
    )
    _save_gray(
        os.path.join(output_dir, "01_pet_after_resize_crop_01.png"),
        pet_01.squeeze(0).numpy(),
    )
    _save_gray(
        os.path.join(output_dir, "01_ct_after_resize_crop_01.png"),
        ct_01.squeeze(0).numpy(),
    )

    # Stage 2: PET invert (if enabled)
    if invert_pet:
        pet_after_invert_01 = 1.0 - pet_01
    else:
        pet_after_invert_01 = pet_01.clone()

    _save_stage_bundle(
        out_dir=tensor_dir,
        stage_name="02_pet_after_optional_invert_01",
        tensor_01=pet_after_invert_01,
    )
    _save_gray(
        os.path.join(output_dir, "02_pet_after_optional_invert_01.png"),
        pet_after_invert_01.squeeze(0).numpy(),
    )

    # Stage 3: model domain [-1, 1]
    pet_m11 = pet_after_invert_01 * 2.0 - 1.0
    ct_m11 = ct_01 * 2.0 - 1.0

    _save_stage_bundle(
        out_dir=tensor_dir,
        stage_name="03_pet_model_input",
        tensor_01=pet_after_invert_01,
        tensor_m11=pet_m11,
    )
    _save_stage_bundle(
        out_dir=tensor_dir,
        stage_name="03_ct_model_input",
        tensor_01=ct_01,
        tensor_m11=ct_m11,
    )
    _save_gray(
        os.path.join(output_dir, "03_pet_model_input_minus1_1_vis.png"),
        np.clip(((pet_m11.squeeze(0).numpy() + 1.0) * 0.5), 0.0, 1.0),
    )
    _save_gray(
        os.path.join(output_dir, "03_ct_model_input_minus1_1_vis.png"),
        np.clip(((ct_m11.squeeze(0).numpy() + 1.0) * 0.5), 0.0, 1.0),
    )

    # Stage 4: paired tensor [2,H,W] as DataLoader output sample
    pair_tensor = torch.cat([pet_m11, ct_m11], dim=0)
    torch.save(pair_tensor, os.path.join(tensor_dir, "04_pair_tensor_B2HW.pt"))
    np.save(
        os.path.join(tensor_dir, "04_pair_tensor_B2HW.npy"),
        pair_tensor.detach().cpu().numpy().astype(np.float32),
    )
    # visualization: left=PET, right=CT in [0,1]
    pair_vis = np.concatenate(
        [
            np.clip((pair_tensor[0].numpy() + 1.0) * 0.5, 0.0, 1.0),
            np.clip((pair_tensor[1].numpy() + 1.0) * 0.5, 0.0, 1.0),
        ],
        axis=1,
    )
    _save_gray(os.path.join(output_dir, "04_pair_pet_left_ct_right.png"), pair_vis)

    # Stage 5: trainer split simulation
    batch_like = pair_tensor.unsqueeze(0)  # [1,2,H,W]
    pet_split, ct_split = batch_like.chunk(2, dim=1)
    torch.save(pet_split, os.path.join(tensor_dir, "05_trainer_split_pet_B1HW.pt"))
    torch.save(ct_split, os.path.join(tensor_dir, "05_trainer_split_ct_B1HW.pt"))

    report: Dict[str, Any] = {
        "input": {
            "pet_path": pet_path,
            "ct_path": ct_path,
            "config_path": os.path.abspath(config_path),
        },
        "settings": {
            "image_size": image_size,
            "invert_pet": invert_pet,
            "config_augment": augment_enabled,
            "apply_augment_flag": bool(apply_augment),
            "augment_applied": augment_applied,
            "data_format": cfg.get("data_format", "png"),
            "use_dicom_mapping": cfg.get("use_dicom_mapping", False),
            "enable_dicom_hu_suv": cfg.get("enable_dicom_hu_suv", False),
        },
        "stages": {
            "00_pet_raw_uint8": _array_stats(pet_raw_np),
            "00_ct_raw_uint8": _array_stats(ct_raw_np),
            "01_pet_after_resize_crop_01": _tensor_stats(pet_01),
            "01_ct_after_resize_crop_01": _tensor_stats(ct_01),
            "02_pet_after_optional_invert_01": _tensor_stats(pet_after_invert_01),
            "03_pet_model_input_minus1_1": _tensor_stats(pet_m11),
            "03_ct_model_input_minus1_1": _tensor_stats(ct_m11),
            "04_pair_tensor_B2HW": _tensor_stats(pair_tensor),
            "05_trainer_split_pet_B1HW": _tensor_stats(pet_split),
            "05_trainer_split_ct_B1HW": _tensor_stats(ct_split),
        },
    }

    with open(os.path.join(output_dir, "trace_summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_markdown_report(os.path.join(output_dir, "trace_report.md"), report)

    print("=" * 72)
    print("Preprocess trace export finished")
    print(f"PET: {pet_path}")
    print(f"CT : {ct_path}")
    print(f"Output folder: {os.path.abspath(output_dir)}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Export step-by-step preprocess artifacts for one PET/CT pair."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--pet-path",
        type=str,
        required=True,
        help="PET image path (supports Windows style path).",
    )
    parser.add_argument(
        "--ct-path",
        type=str,
        default=None,
        help="Optional CT image path. If omitted, auto-match by basename.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="samples/preprocess_trace",
        help="Output directory for exported artifacts.",
    )
    parser.add_argument(
        "--apply-augment",
        action="store_true",
        help="Apply augmentation logic (random hflip) if config.augment=true.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for augmentation decision when --apply-augment is set.",
    )
    args = parser.parse_args()

    export_trace(
        config_path=args.config,
        pet_path=args.pet_path,
        ct_path=args.ct_path,
        output_dir=args.output_dir,
        apply_augment=args.apply_augment,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
