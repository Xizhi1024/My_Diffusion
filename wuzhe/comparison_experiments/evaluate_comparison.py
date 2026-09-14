"""Evaluate SLMF-aligned comparison models on PNG train/val splits.

The intended protocol is:
  1. Train on train split.
  2. Select ckpt_best.pt by validation loss.
  3. Compute final metrics on the validation split using that checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from comparison_experiments.models import build_comparison_model
from comparison_experiments.png_dataset import LightweightPNGSliceDataset


def _torch_load(path: str, device: str) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _de_collate_meta(meta: Any, batch_size: int) -> List[dict]:
    if isinstance(meta, list):
        return [dict(m) if isinstance(m, dict) else {} for m in meta]
    if isinstance(meta, dict):
        out = []
        for i in range(batch_size):
            item = {}
            for key, value in meta.items():
                if torch.is_tensor(value):
                    if value.ndim > 0 and value.shape[0] == batch_size:
                        item[key] = value[i].detach().cpu().item() if value[i].numel() == 1 else value[i].detach().cpu().tolist()
                    else:
                        item[key] = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
                elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                    item[key] = value[i]
                else:
                    item[key] = value
            out.append(item)
        return out
    return [{} for _ in range(batch_size)]


def _ssim_torch(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Small dependency-free SSIM for [B,1,H,W] tensors."""
    if pred.shape[-1] < 7 or pred.shape[-2] < 7:
        return torch.full((pred.shape[0],), float("nan"), device=pred.device)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = 11
    pad = kernel // 2
    mu_x = F.avg_pool2d(pred, kernel, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, kernel, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel, stride=1, padding=pad) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-12
    )
    return ssim_map.flatten(1).mean(dim=1)


def _sample_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    diff = pred - target
    mse = diff.square().mean().item()
    mae = diff.abs().mean().item()
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 100.0 if mse < 1e-12 else 20.0 * math.log10(2.0) - 10.0 * math.log10(mse)
    ssim = _ssim_torch(pred.unsqueeze(0), target.unsqueeze(0))[0].item()
    out = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }
    if mask.sum().item() > 0:
        roi = mask > 0.5
        roi_diff = diff[roi]
        out["roi_mae"] = roi_diff.abs().mean().item()
        out["roi_mse"] = roi_diff.square().mean().item()
        out["roi_rmse"] = math.sqrt(max(out["roi_mse"], 0.0))
    else:
        out["roi_mae"] = float("nan")
        out["roi_mse"] = float("nan")
        out["roi_rmse"] = float("nan")
    return out


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"num_samples": len(rows)}
    if not rows:
        return summary
    metric_keys = [k for k, v in rows[0].items() if isinstance(v, float)]
    for key in metric_keys:
        vals = np.array([row[key] for row in rows if isinstance(row.get(key), float) and not math.isnan(row[key])], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[f"{key}_mean"] = float(vals.mean())
        summary[f"{key}_std"] = float(vals.std())
        summary[f"{key}_median"] = float(np.median(vals))
        summary[f"{key}_min"] = float(vals.min())
        summary[f"{key}_max"] = float(vals.max())
    return summary


@torch.no_grad()
def evaluate(
    checkpoint: str,
    png_root: str,
    *,
    split: str = "val",
    pet_dir: str = "pet_peizhuan",
    image_size: int = 192,
    batch_size: int = 4,
    output_dir: str = "output/comparison_metrics",
    device: str | None = None,
) -> Dict[str, Any]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _torch_load(checkpoint, device)
    config = ckpt.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a training config; cannot rebuild comparison model.")
    model = build_comparison_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = LightweightPNGSliceDataset(
        png_root,
        split=split,
        image_size=image_size,
        ct_dir="ct",
        pet_dir=pet_dir,
        label_dir="label",
        augment=False,
        require_label=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    rows: List[Dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        pred = model.sample(batch_gpu)["synthetic_pet"].detach()
        target = batch_gpu["pet"]
        mask = batch_gpu.get("mask", torch.zeros_like(target))
        metas = _de_collate_meta(batch.get("meta"), pred.shape[0])
        for i in range(pred.shape[0]):
            row = {
                "sample_id": str(metas[i].get("sample_id", f"{batch_idx}_{i}")),
                "patient_id": str(metas[i].get("patient_id", "")),
            }
            row.update(_sample_metrics(pred[i].cpu(), target[i].cpu(), mask[i].cpu()))
            rows.append(row)
        if (batch_idx + 1) % 10 == 0:
            print(f"  evaluated {batch_idx + 1} batches / {len(rows)} samples")

    summary = _aggregate(rows)
    summary.update({
        "checkpoint": checkpoint,
        "png_root": png_root,
        "split": split,
        "pet_dir": pet_dir,
        "image_size": image_size,
    })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_per_sample.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate comparison model best checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--png-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--pet-dir", default="pet_peizhuan")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(
        args.checkpoint,
        args.png_root,
        split=args.split,
        pet_dir=args.pet_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
