"""Visualize model predictions vs ground truth on val split (masked samples)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM
from src.data.dataset import CachedDataset
from torch.utils.data import DataLoader, Subset

CKPT = "checkpoints/slmf_bbdm_full_v4/ckpt_epoch0100.pt"
CONFIG = "configs/experiments/slmf_full.yaml"
OUT  = Path("results/vis_v4_ep100_RAW")
#OUT = Path("results/vis_epoch1000"); OUT.mkdir(parents=True, exist_ok=True)

cfg = resolve_runtime_profile(load_full_config(CONFIG))
device = "cuda"
model = SLMFBBDM.from_config(cfg).to(device).eval()
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True)["model"])

d = cfg["data"]
ds = CachedDataset(d["cache_dir"], split="val", augment=False,
                   split_manifest=Path(d["split_manifest"]))
masked = [i for i, e in enumerate(ds.entries) if e.has_mask]
print(f"{len(masked)} masked val samples; visualizing first 8")
loader = DataLoader(Subset(ds, masked[:8]), batch_size=1)

for i, batch in enumerate(loader):
    bg = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        pred = model.sample(bg)["synthetic_pet"][0, 0].cpu().numpy()
    tgt = batch["pet"][0, 0].numpy()
    ct = batch["ct"][0, 0].numpy()
    msk = batch["mask"][0, 0].numpy()
    pid = ds.entries[masked[i]].patient_id
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(ct, cmap="gray");                ax[0].set_title("CT")
    ax[1].imshow(tgt, cmap="hot", vmin=-1, vmax=1); ax[1].set_title("Target PET")
    ax[2].imshow(pred, cmap="hot", vmin=-1, vmax=1); ax[2].set_title("Pred PET")
    ax[3].imshow(msk, cmap="gray");                ax[3].set_title("Lesion mask")
    fig.suptitle(f"pid={pid} | pred[{pred.min():.2f},{pred.max():.2f}] tgt[{tgt.min():.2f},{tgt.max():.2f}]")
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / f"sample_{i:02d}_pid{pid}.png", dpi=120); plt.close(fig)
    print(f"[{i}] pid={pid} pred_max={pred.max():.3f} (mask区域) tgt_max={tgt.max():.3f}")
print(f"saved to {OUT}/")
