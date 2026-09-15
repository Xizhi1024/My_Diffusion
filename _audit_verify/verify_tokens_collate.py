# -*- coding: utf-8 -*-
"""Red-team new finding: semantic_tokens conditional key collate risk."""
import json
from torch.utils.data._utils.collate import default_collate
b1 = {"ct": None, "semantic_tokens": None}
a = [{"ct": [1.0]}, {"ct": [2.0], "semantic_tokens": [0.5]}]   # first lacks tokens
b = [{"ct": [1.0], "semantic_tokens": [0.5]}, {"ct": [2.0]}]   # first has tokens
out = {}
try:
    c = default_collate(a); out["first_lacks_tokens"] = {"result": "collated", "keys": sorted(c.keys())}
except Exception as ex:
    out["first_lacks_tokens"] = {"result": f"CRASH {type(ex).__name__}: {ex}"}
try:
    c = default_collate(b); out["first_has_tokens"] = {"result": "collated", "keys": sorted(c.keys())}
except Exception as ex:
    out["first_has_tokens"] = {"result": f"CRASH {type(ex).__name__}: {ex}"}
print(json.dumps(out, indent=2))
