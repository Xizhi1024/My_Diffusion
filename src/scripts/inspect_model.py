"""Print model parameter counts and module status (dry-run, no training)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.slmf_bbdm import SLMFBBDM
from src.model.config_utils import load_full_config

cfg = load_full_config("configs/experiments/slmf_full.yaml")
m = SLMFBBDM.from_config(cfg)

print(f"Trainable: {m.get_trainable_params():,} / Total: {m.get_total_params():,}")
print(f"Priors:  {[(n, p.enabled) for n, p in m.priors.items()]}")
print(f"Losses:  {[(n, l.enabled) for n, l in m.loss_terms.items()]}")
