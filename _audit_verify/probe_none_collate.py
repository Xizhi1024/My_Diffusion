# -*- coding: utf-8 -*-
from torch.utils.data._utils.collate import default_collate
import json
# design probe: can we normalize heterogeneous meta by filling missing keys with None?
try:
    r = default_collate([{"a": 1.0, "b": None}, {"a": 2.0, "b": None}])
    print("all-None value collate:", r)
except Exception as ex:
    print("all-None FAILS:", type(ex).__name__, str(ex)[:100])
try:
    r = default_collate([{"a": 1.0, "b": 70.0}, {"a": 2.0, "b": None}])
    print("mixed None/float collate:", r)
except Exception as ex:
    print("mixed FAILS:", type(ex).__name__, str(ex)[:100])
