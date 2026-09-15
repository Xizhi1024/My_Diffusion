# -*- coding: utf-8 -*-
"""E1 rerun: _meta_to_tensor with real collated dict-of-tensors (post-fix)."""
import ast, json, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
class MetaDS(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        return {"meta": {"uptake_min": 60.0 + i, "weight_kg": 70.0, "age_years": 55.0, "thickness_mm": 3.0, "z_mm": 10.0}}
batch = next(iter(DataLoader(MetaDS(), batch_size=4)))
meta = batch["meta"]

src_text = (ROOT / "src/model/slmf_bbdm.py").read_text(encoding="utf-8")
tree = ast.parse(src_text)
segs = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in {"_meta_to_tensor"}:
        segs[node.name] = ast.get_source_segment(src_text, node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id in {"_META_KEYS", "_META_NORM"}:
                segs[t.id] = ast.get_source_segment(src_text, node)
ns = {"torch": torch, "math": math}
for k in ("_META_KEYS", "_META_NORM", "_meta_to_tensor"):
    exec(segs[k], ns)
out = ns["_meta_to_tensor"](meta, 4, torch.device("cpu"), torch.float32)
# also: NaN sentinel treated as missing
out2 = ns["_meta_to_tensor"]({"uptake_min": torch.tensor([float("nan")]*4), "weight_kg": torch.tensor([70.0]*4)}, 4, torch.device("cpu"), torch.float32)
print(json.dumps({
    "result": "NO CRASH (fixed)",
    "shape": list(out.shape),
    "row0": [round(v, 3) for v in out[0].tolist()],
    "nan_sentinel_weight_kg_norm": round(out2[0, 1].item(), 3),
    "nan_sentinel_uptake_fills_center": round(out2[0, 0].item(), 3),
}, indent=2))
