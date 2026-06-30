"""Diagnose whether Gabor/Hotspot/Adapter paths carry stripe-like artifacts.

This is a read-only diagnostic script. It loads a trained SLMF-BBDM checkpoint,
samples a few masked validation cases, and saves:

  - CT / target PET / predicted PET / lesion mask / HotspotPrior
  - Gabor energy
  - ZeroConv Adapter injection norm maps for L0-L3 at several tau values
  - A CSV summary with prediction and injection statistics

Run from the project root, for example:

    pixi run python scripts/diagnose_gabor_adapter.py \
      --checkpoint checkpoints/slmf_bbdm_full_v4/ckpt_epoch0100.pt \
      --out-dir results/diag_gabor_adapter_v4_ep100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import CachedDataset
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM


def _load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _select_state_dict(ckpt: dict[str, Any], weights: str) -> tuple[dict[str, torch.Tensor], str]:
    if weights == "raw":
        if "model" not in ckpt:
            raise KeyError("Checkpoint has no raw model weights under key 'model'.")
        return ckpt["model"], "model"

    for key in ("ema_model", "model_ema", "ema"):
        if key in ckpt:
            return ckpt[key], key
    raise KeyError("--weights ema requested, but checkpoint has no EMA weights.")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _to_numpy(t: torch.Tensor):
    return t.detach().float().cpu().numpy()


def _show(ax, image, title: str, cmap: str = "gray", vmin=None, vmax=None) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _text_cell(ax, text: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10)


def _resize_norm_map(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    norm = t.detach().abs().mean(dim=1, keepdim=True)
    if norm.shape[-2:] != size:
        norm = F.interpolate(norm, size=size, mode="bilinear", align_corners=False)
    return norm[:, 0]


def _adapter_outputs_by_tau(
    model: SLMFBBDM,
    condition,
    tau_values: list[int],
) -> dict[int, list[torch.Tensor]]:
    if not model.zero_adapter_enabled:
        return {tv: [] for tv in tau_values}

    ct_feats = [condition.maps[f"ct_feat_{idx}"] for idx in range(4)]
    organ_feats = [
        condition.maps.get("organ_feat_1"),
        condition.maps.get("organ_feat_2"),
        condition.maps.get("organ_feat_3"),
        None,
    ]
    gabor_feat = condition.maps.get("gabor_feat")
    hotspot_prior = condition.maps.get("hotspot_prior")
    hw_list = [feat.shape[-1] for feat in ct_feats]

    outputs: dict[int, list[torch.Tensor]] = {}
    device = ct_feats[0].device
    for tv in tau_values:
        timesteps = torch.full((ct_feats[0].shape[0],), tv, dtype=torch.long, device=device)
        tau = model.noise_schedule.get_tau(timesteps)
        outputs[tv] = model.adapter.get_zero_conv_outputs(
            ct_feats,
            organ_feats,
            gabor_feat,
            hotspot_prior,
            hw_list,
            tau=tau,
        )
    return outputs


def _sample_indices(ds: CachedDataset, limit: int, sample_ids: list[str]) -> list[int]:
    if sample_ids:
        by_id = {entry.sample_id: idx for idx, entry in enumerate(ds.entries)}
        missing = [sid for sid in sample_ids if sid not in by_id]
        if missing:
            raise ValueError(f"Sample id(s) not found in split: {missing}")
        return [by_id[sid] for sid in sample_ids]

    masked = [idx for idx, entry in enumerate(ds.entries) if entry.has_mask]
    if not masked:
        print("WARNING: no masked entries found; falling back to first samples.")
        return list(range(min(limit, len(ds))))
    return masked[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Gabor and Adapter condition maps.")
    parser.add_argument("--config", default="configs/experiments/slmf_full.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out-dir", default="results/diag_gabor_adapter")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-id", action="append", default=[], help="Specific sample id; repeatable.")
    parser.add_argument("--num-steps", type=int, default=None, help="Sampling steps; default uses config.")
    parser.add_argument("--tau", type=int, action="append", default=[0, 500, 900])
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    args = parser.parse_args()

    cfg = resolve_runtime_profile(load_full_config(args.config))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {out_dir}")

    model = SLMFBBDM.from_config(cfg).to(device).eval()
    ckpt = _load_checkpoint(Path(args.checkpoint), map_location="cpu")
    state, state_key = _select_state_dict(ckpt, args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARNING: load_state_dict strict=False missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing:", missing[:10])
        if unexpected:
            print("  unexpected:", unexpected[:10])
    print(f"Loaded weights from ckpt['{state_key}']")

    data_cfg = cfg["data"]
    ds = CachedDataset(
        data_cfg["cache_dir"],
        split=args.split,
        augment=False,
        split_manifest=Path(data_cfg["split_manifest"]),
        required_keys=data_cfg.get("required_keys"),
        optional_keys=data_cfg.get("optional_keys"),
    )
    indices = _sample_indices(ds, args.num_samples, args.sample_id)
    loader = DataLoader(Subset(ds, indices), batch_size=1, shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = []
    tau_values = sorted(set(args.tau), reverse=False)

    for out_idx, batch in enumerate(loader):
        entry = ds.entries[indices[out_idx]]
        bg = _move_batch(batch, device)
        H, W = bg["ct"].shape[-2:]
        t0 = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            condition = model.build_condition_bundle(bg, t0)
            sample = model.sample(bg, num_steps=args.num_steps)
            pred = sample["synthetic_pet"]
            adapter_by_tau = _adapter_outputs_by_tau(model, condition, tau_values)

        ct_np = _to_numpy(batch["ct"][0, 0])
        target_np = _to_numpy(batch["pet"][0, 0])
        pred_np = _to_numpy(pred[0, 0])
        mask_np = _to_numpy(batch["mask"][0, 0])

        hotspot = condition.maps.get("hotspot_prior")
        hotspot_np = _to_numpy(hotspot[0, 0]) if hotspot is not None else None
        gabor = condition.maps.get("gabor_energy")
        gabor_np = _to_numpy(gabor[0, 0]) if gabor is not None else None

        fig, axes = plt.subplots(4, 5, figsize=(21, 15))
        _show(axes[0, 0], ct_np, "CT", "gray")
        _show(axes[0, 1], target_np, "Target PET", "hot", vmin=-1, vmax=1)
        _show(axes[0, 2], pred_np, "Pred PET", "hot", vmin=-1, vmax=1)
        _show(axes[0, 3], mask_np, "Lesion mask", "gray")
        if hotspot_np is not None:
            _show(axes[0, 4], hotspot_np, "HotspotPrior", "hot", vmin=0, vmax=1)
        else:
            _text_cell(axes[0, 4], "HotspotPrior\nmissing")

        if gabor_np is not None:
            _show(axes[1, 0], gabor_np, "Gabor energy", "magma", vmin=0, vmax=1)
        else:
            _text_cell(axes[1, 0], "Gabor energy\nmissing")

        row = {
            "sample_id": entry.sample_id,
            "patient_id": entry.patient_id,
            "slice_id": entry.slice_id,
            "pred_min": float(pred_np.min()),
            "pred_max": float(pred_np.max()),
            "target_min": float(target_np.min()),
            "target_max": float(target_np.max()),
            "mask_pixels": int((mask_np > 0.5).sum()),
        }
        mask_bool = mask_np > 0.5
        if mask_bool.any():
            row["pred_mask_max"] = float(pred_np[mask_bool].max())
            row["target_mask_max"] = float(target_np[mask_bool].max())
        else:
            row["pred_mask_max"] = ""
            row["target_mask_max"] = ""
        if gabor_np is not None:
            row["gabor_mean"] = float(gabor_np.mean())
            row["gabor_max"] = float(gabor_np.max())
        if hotspot_np is not None:
            row["hotspot_mean"] = float(hotspot_np.mean())
            row["hotspot_max"] = float(hotspot_np.max())

        for row_offset, tv in enumerate(tau_values[:3], start=1):
            if row_offset > 3:
                break
            outputs = adapter_by_tau.get(tv, [])
            if row_offset > 1:
                _text_cell(axes[row_offset, 0], f"Adapter\nT={tv}\ntau={tv / 1000:.2f}")
            for level in range(4):
                ax = axes[row_offset, level + 1]
                if level < len(outputs):
                    norm = _resize_norm_map(outputs[level], (H, W))
                    norm_np = _to_numpy(norm[0])
                    _show(
                        ax,
                        norm_np,
                        f"L{level} inj | T={tv}\nmax={norm_np.max():.3g}",
                        "viridis",
                    )
                    row[f"adapter_L{level}_T{tv}_mean"] = float(norm_np.mean())
                    row[f"adapter_L{level}_T{tv}_max"] = float(norm_np.max())
                else:
                    _text_cell(ax, f"L{level}\nmissing")

        title = (
            f"{entry.sample_id} pid={entry.patient_id} slice={entry.slice_id} | "
            f"weights={state_key} | pred[{pred_np.min():.2f},{pred_np.max():.2f}]"
        )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig_path = out_dir / f"diag_{out_idx:02d}_{entry.sample_id}_pid{entry.patient_id}.png"
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

        rows.append(row)
        print(f"[{out_idx}] saved {fig_path}")

    csv_path = out_dir / "diagnostics_summary.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary -> {csv_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
