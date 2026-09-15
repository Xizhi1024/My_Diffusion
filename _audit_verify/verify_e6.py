# -*- coding: utf-8 -*-
"""E6 re-run: png_cache --no-overwrite incremental manifest loss."""
import csv, json, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image
from src.data.png_cache import build_png_cache

tmp = ROOT / "_audit_verify_tmp"
if tmp.exists(): shutil.rmtree(tmp)
png_root = tmp / "png"; out = tmp / "cache"
for sub in ("ct", "pet", "label"):
    (png_root / "train" / sub).mkdir(parents=True)
ids = ["001001", "001002", "002001"]
for sid in ids:
    arr = (np.arange(64*64).reshape(64, 64) % 255).astype(np.uint8)
    Image.fromarray(arr).save(png_root / "train" / "ct" / f"{sid}.png")
    Image.fromarray(255 - arr).save(png_root / "train" / "pet" / f"{sid}.png")
    Image.fromarray((arr > 128).astype(np.uint8) * 255).save(png_root / "train" / "label" / f"{sid}.png")
(tmp / "split.csv").write_text("file_name,patient_id,split\n" + "".join(f"{sid},{sid[:3]},train\n" for sid in ids), encoding="utf-8")

mpath = png_root / "split_manifest.csv"   # <-- written next to png_root by default
s1 = build_png_cache(png_root=str(png_root), split_csv=str(tmp / "split.csv"), out_dir=str(out), overwrite=True)
m1 = list(csv.DictReader(mpath.open(encoding="utf-8")))
s2 = build_png_cache(png_root=str(png_root), split_csv=str(tmp / "split.csv"), out_dir=str(out), overwrite=False)
m2 = list(csv.DictReader(mpath.open(encoding="utf-8")))
print(json.dumps({
    "manifest_path": str(mpath),
    "first_run_built": s1["built"], "first_run_manifest_rows": len(m1),
    "second_run_built": s2["built"], "second_run_skipped": dict(s2["skipped"]),
    "second_run_manifest_rows": len(m2),
    "npz_files_still_on_disk": len(list(out.glob('*.npz'))),
    "claim_confirmed_manifest_shrinks": len(m2) < len(m1),
}, indent=2))
shutil.rmtree(tmp, ignore_errors=True)
