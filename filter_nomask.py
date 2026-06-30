"""过滤 split_manifest.csv：移除 npz 中无 mask 的样本（required key）。

同时检测 npz 损坏（BadZipFile）。结果写回 split_manifest.csv，原文件备份为 .bak。
"""
import csv
import os
import shutil

import numpy as np

manifest = "cache/split_manifest.csv"
cache_dir = "cache/tensors"

shutil.copy(manifest, manifest + ".bak")

with open(manifest) as f:
    rows = list(csv.DictReader(f))

keep = []
removed = []

for r in rows:
    sid = r["sample_id"]
    npz = os.path.join(cache_dir, sid + ".npz")
    if not os.path.exists(npz):
        removed.append((sid, "npz 不存在"))
        continue
    try:
        with np.load(npz) as d:
            has_mask = "mask" in d.files
    except Exception as e:
        removed.append((sid, f"读取失败: {type(e).__name__}: {e}"))
        continue
    if has_mask:
        keep.append(r)
    else:
        removed.append((sid, "无 mask"))

with open(manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(keep)

print(f"总计 {len(rows)} | 保留 {len(keep)} | 移除 {len(removed)}")
for sid, reason in removed:
    print(f"  {sid}: {reason}")
