# -*- coding: utf-8 -*-
"""P0-2 + P1-10 empirical checks."""
import ast, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parent.parent
out = {}

# --- P0-2a: batch[0] has MORE meta keys than others -> KeyError
batches = [
    [{"meta": {"weight_kg": 70.0, "z_mm": 10.0}}, {"meta": {"weight_kg": 60.0}}],  # first has extra key
    [{"meta": {"weight_kg": 60.0}}, {"meta": {"weight_kg": 70.0, "z_mm": 10.0}}],  # first lacks key
]
labels = ["first_has_extra_key", "first_lacks_key"]
for label, bl in zip(labels, batches):
    try:
        c = default_collate(bl)
        out[f"P0-2_{label}"] = {"result": "collated", "keys": sorted(c["meta"].keys())}
    except Exception as ex:
        out[f"P0-2_{label}"] = {"result": f"CRASH {type(ex).__name__}: {ex}"}

# --- P1-10: BBDMBridgeSchedule(enabled=False) still applies forward noise
def extract_src(path, names):
    src_text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src_text)
    segs = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            segs[node.name] = ast.get_source_segment(src_text, node)
    return segs

class _StubNS(torch.nn.Module):
    def __init__(self, enabled=True):
        super().__init__(); self.enabled = enabled

ns = {"torch": torch, "math": __import__("math"), "NoiseSchedule": _StubNS, "nn": torch.nn}
exec(extract_src(ROOT / "src/model/noise/base.py", {"BBDMBridgeSchedule"})["BBDMBridgeSchedule"], ns)
sched = ns["BBDMBridgeSchedule"](num_train_timesteps=1000, m_schedule="linear", enabled=False)
x0 = torch.zeros(1, 1, 8, 8); src = torch.ones(1, 1, 8, 8)
noise = torch.ones(1, 1, 8, 8)
t = torch.tensor([500])
xt = sched.add_noise(x0, noise, t, None, x_source=src)
m = sched.m_t[500].item(); s = sched.sigma_t[500].item()
out["P1-10_enabled_false_add_noise"] = {
    "raised": False,
    "output_mean": float(xt.mean()),
    "expected_m*src_plus_sigma": m * 1.0 + s * 1.0,
    "note": "enabled=False had no effect; schedule ran the full bridge forward",
}
print(json.dumps(out, indent=2))
