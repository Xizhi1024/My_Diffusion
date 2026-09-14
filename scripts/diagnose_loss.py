"""Diagnose per-loss-term magnitudes on real val samples (masked)."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import torch
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM
from src.data.dataset import CachedDataset
from torch.utils.data import DataLoader, Subset

CKPT = "checkpoints/slmf_bbdm_full/ckpt_epoch0600.pt"
CONFIG = "configs/experiments/slmf_full.yaml"

cfg = resolve_runtime_profile(load_full_config(CONFIG))
device = "cuda"
model = SLMFBBDM.from_config(cfg).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True)["model"])
model.eval()

d = cfg["data"]
ds = CachedDataset(d["cache_dir"], split="val", augment=False,
                   split_manifest=Path(d["split_manifest"]))
masked = [i for i, e in enumerate(ds.entries) if e.has_mask][:8]
loader = DataLoader(Subset(ds, masked), batch_size=4, shuffle=False)
batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(iter(loader)).items()}

def run(label, **kw):
    torch.manual_seed(42)
    with torch.no_grad():
        loss, logs = model(batch, **kw)
    print(f"\n===== {label} =====  total={loss.item():.5f}")
    for k in sorted(logs):
        v = logs[k]
        if torch.is_tensor(v) and v.numel() == 1:
            print(f"  {k:45s} {v.item():>12.5f}")

run("随机τ (真实训练分布)")
run("τ=0 (最终去噪步, 病灶loss最激活)", timesteps=torch.zeros(4, dtype=torch.long, device=device))
