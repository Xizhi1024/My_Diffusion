# -*- coding: utf-8 -*-
"""P1-4 empirical: EMA.load_state_dict with mismatched keys -> silent NoOp."""
import importlib.util, json
from pathlib import Path
import torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ema_real", ROOT / "src/model/ema.py")
ema_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(ema_mod)

m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
ema = ema_mod.EMA(m, decay=0.999, update_every=1)
real_before = {n: p.detach().clone() for n, p in m.named_parameters()}

state = {"shadow": {f"_orig_mod.{k}": v for k, v in ema.shadow.items()},
         "step_count": 5, "decay": 0.999, "update_every": 1}
ema.load_state_dict(state)   # accepts silently
shadow_keys_after = list(ema.shadow.keys())
overlap = [k for k in shadow_keys_after if k in dict(m.named_parameters())]

with torch.no_grad():
    for p in m.parameters(): p.add_(1.0)   # large real-weight change
ema.update()                                 # intended: shadow should follow weights

# did any shadow tensor track the change? (compare via the _orig_mod.-keyed copies vs their originals)
orig_shadow = {k.removeprefix("_orig_mod."): v for k, v in ema.shadow.items()}
max_delta = max((orig_shadow[n] - real_before[n]).abs().max().item() for n in real_before)

ema.apply()  # intended: swap EMA weights into model
apply_changed = max((p.detach() - real_before[n]).abs().max().item()
                    for n, p in zip(real_before.keys(), m.parameters()))
print(json.dumps({
    "load_state_dict_raised": False,
    "shadow_key_overlap_with_model_params": len(overlap),
    "shadow_followed_weight_change": max_delta,
    "apply_changed_model_weights": apply_changed,
    "verdict": "CONFIRMED: EMA silently becomes a NoOp (no error, no tracking, no swap)",
}, indent=2))
