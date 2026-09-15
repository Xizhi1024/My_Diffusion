# -*- coding: utf-8 -*-
"""Empirical verification of audit claims (read-only wrt repo; writes only under _audit_verify_tmp/)."""
import ast, csv, json, math, shutil, sys, traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = {}

def extract_src(path, names):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    segs = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            segs[node.name] = ast.get_source_segment(Path(path).read_text(encoding="utf-8"), node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    segs[t.id] = ast.get_source_segment(Path(path).read_text(encoding="utf-8"), node)
    return segs

# ============ E1: P0-1 _meta_to_tensor vs default collate ============
class MetaDS(Dataset):
    def __len__(self): return 4
    def __getitem__(self, i):
        return {"meta": {"uptake_min": 60.0 + i, "weight_kg": 70.0, "age_years": 55.0, "thickness_mm": 3.0, "z_mm": 10.0}}

dl = DataLoader(MetaDS(), batch_size=4)
batch = next(iter(dl))
meta_collated = batch["meta"]
e1 = {"collated_value_type": type(next(iter(meta_collated.values()))).__name__}

segs = extract_src(ROOT / "src/model/slmf_bbdm.py", {"_META_KEYS", "_META_NORM", "_meta_to_tensor"})
ns = {"torch": torch}
exec(segs["_META_KEYS"], ns); exec(segs["_META_NORM"], ns); exec(segs["_meta_to_tensor"], ns)
e1["is_collated_branch_taken"] = False  # replicated check:
first_val = next(iter(meta_collated.values()))
e1["isinstance_list_tuple"] = isinstance(first_val, (list, tuple))
e1["len_eq_B"] = (hasattr(first_val, "__len__") and len(first_val) == 4)
try:
    ns["_meta_to_tensor"](meta_collated, 4, torch.device("cpu"), torch.float32)
    e1["result"] = "NO ERROR (claim refuted)"
except Exception as ex:
    e1["result"] = f"CRASH: {type(ex).__name__}: {ex}"
OUT["E1_meta_to_tensor"] = e1

# ============ E2: P1-7 bridge min-SNR arithmetic ============
class _StubNS(torch.nn.Module):
    def __init__(self, enabled=True):
        super().__init__(); self.enabled = enabled
ns2 = {"torch": torch, "math": math, "NoiseSchedule": _StubNS, "nn": torch.nn}
exec(extract_src(ROOT / "src/model/noise/base.py", {"BBDMBridgeSchedule"})["BBDMBridgeSchedule"], ns2)
sched = ns2["BBDMBridgeSchedule"](num_train_timesteps=1000, m_schedule="linear")
m = sched.m_t.clone(); s = sched.sigma_t.clone()
snr = (1.0 - m).square() / s.square().clamp_min(1e-8)
gamma = 5.0
w = torch.minimum(snr, torch.full_like(snr, gamma))
OUT["E2_min_snr"] = {
    "sigma_at_t0": float(s[0]), "sigma_at_t999": float(s[999]),
    "snr_t0": float(snr[0]), "snr_t999": float(snr[999]),
    "weight_t0": float(w[0]), "weight_t999": float(w[999]),
    "analytic_check_(1-u)/(2u)_at_t999": (1-0.999)/(2*0.999),
    "weight_ratio_t0_over_t999": float(w[0]/max(w[999], 1e-12)),
}

# ============ E3: P1-1 early stopping arithmetic (logic replicated verbatim from trainer.py:2136-2155) ============
def check_early_stopping(state, improved, patience=40, min_epochs=50):
    if not state["enabled"]: return False
    if improved or state["last_improve"] is None:
        state["last_improve"] = state["epoch"]; state["since"] = 0; return False
    state["since"] = state["epoch"] - state["last_improve"]
    if state["epoch"] < min_epochs: return False
    if state["since"] >= patience: return True
    return False
state = {"enabled": True, "last_improve": None, "since": 0, "epoch": 0}
stop_epoch = None
improvements = {50: True, 100: False, 150: False, 200: False}  # evals every 50; improve only at first
for epoch in range(1, 301):
    state["epoch"] = epoch
    if epoch % 50 == 0:
        if check_early_stopping(state, improvements[epoch]):
            stop_epoch = epoch; break
OUT["E3_early_stopping"] = {"stop_epoch": stop_epoch, "note": "patience=40, eval_interval=50, improve only at epoch 50"}

# ============ E4: P2-3 false_hotspot_count constancy ============
segs4 = extract_src(ROOT / "scripts/evaluate.py", {"compute_false_hotspot_count"})
ns4 = {"np": np}
exec(segs4["compute_false_hotspot_count"], ns4)
rng = np.random.default_rng(0)
organ = np.zeros((6, 64, 64), dtype=np.float32)
counts = []
for k in range(3):
    if k == 0: pred = rng.random((1, 64, 64)).astype(np.float32)          # uniform noise
    elif k == 1: pred = np.zeros((1, 64, 64), np.float32); pred[0, 32, 32] = 1.0  # one true hotspot
    else: pred = (rng.random((1, 64, 64)) ** 3).astype(np.float32)        # skewed dark image
    counts.append(ns4["compute_false_hotspot_count"](pred, organ)["false_hotspot_count"])
pos_frac = [round(c / (64*64), 4) for c in counts]
OUT["E4_false_hotspot"] = {"counts": counts, "fraction_of_image": pos_frac,
    "note": "3 completely different predictions give ~same count (fixed by construction)"}

# ============ E5: P1-20 patient aggregation drops keys ============
patient_results = {"p1": [
    {"mae": 0.1, "psnr": 20.0},                       # first slice: NO lesion keys
    {"mae": 0.2, "psnr": 19.0, "lesion_peak_error_norm": 0.5},
    {"mae": 0.15, "psnr": 19.5, "lesion_peak_error_norm": 0.4},
]}
agg = {}
for pid, p_metrics in patient_results.items():
    for k in p_metrics[0]:
        vals = [m[k] for m in p_metrics if isinstance(m.get(k), (int, float)) and not np.isnan(m.get(k, float("nan")))]
        if vals: agg[f"{k}_mean"] = float(np.mean(vals))
OUT["E5_patient_agg"] = {"aggregated_keys": sorted(agg.keys()),
    "lesion_key_present": "lesion_peak_error_norm_mean" in agg,
    "note": "lesion keys exist on rows 2-3 but are dropped because row 0 lacks them"}

# ============ E8: config trigger values ============
import yaml
cfg_info = {}
for y in sorted((ROOT / "configs/experiments").glob("*.yaml")):
    try:
        c = yaml.safe_load(y.read_text(encoding="utf-8"))
        rt = c.get("runtime", {}) or {}
        tr = c.get("training", {}) or {}
        ema = tr.get("ema", {}) or {}
        es = rt.get("early_stopping", {}) or {}
        cfg_info[y.name] = {
            "eval_interval": rt.get("eval_interval"), "sample_interval": rt.get("sample_interval"),
            "es_enabled": es.get("enabled"), "patience": es.get("patience"), "min_epochs": es.get("min_epochs"),
            "ema_decay": ema.get("decay"), "ema_update_every": ema.get("update_every"),
        }
    except Exception as ex:
        cfg_info[y.name] = {"error": str(ex)}
OUT["E8_configs"] = cfg_info

# ============ E9: pixi eval tasks lack --checkpoint ============
pixi = (ROOT / "pixi.toml").read_text(encoding="utf-8")
eval_lines = [l.strip() for l in pixi.splitlines() if l.strip().startswith("eval-")]
OUT["E9_pixi_eval_tasks"] = {l.split("=")[0].strip(): ("--checkpoint" in l) for l in eval_lines}

# ============ E10: config_utils string normalization ============
import importlib.util
spec = importlib.util.spec_from_file_location("cfgutils", ROOT / "src/model/config_utils.py")
cu = importlib.util.module_from_spec(spec); spec.loader.exec_module(cu)
OUT["E10_scalar_norm"] = {"input_042": cu._normalize_scalar_strings("042"),
                          "input_1e-4": cu._normalize_scalar_strings("1e-4"),
                          "type_042": type(cu._normalize_scalar_strings("042")).__name__}

# ============ E6: P0-5 png_cache --no-overwrite manifest loss (empirical) ============
tmp = ROOT / "_audit_verify_tmp"
if tmp.exists(): shutil.rmtree(tmp)
try:
    from PIL import Image
    from src.data.png_cache import build_png_cache
    png_root = tmp / "png"; out1 = tmp / "cache1"; out2 = tmp / "cache2"
    for split in ("train",):
        for sub in ("ct", "pet", "label"):
            (png_root / split / sub).mkdir(parents=True)
    ids = ["001001", "001002", "002001"]
    for sid in ids:
        arr = (np.arange(64*64).reshape(64, 64) % 255).astype(np.uint8)
        Image.fromarray(arr).save(png_root / "train" / "ct" / f"{sid}.png")
        Image.fromarray(255 - arr).save(png_root / "train" / "pet" / f"{sid}.png")
        Image.fromarray((arr > 128).astype(np.uint8) * 255).save(png_root / "train" / "label" / f"{sid}.png")
    (tmp / "split.csv").write_text("file_name,patient_id,split\n" + "".join(f"{sid},{sid[:3]},train\n" for sid in ids), encoding="utf-8")
    s1 = build_png_cache(png_root=str(png_root), split_csv=str(tmp / "split.csv"), out_dir=str(out1), overwrite=True)
    m1 = list(csv.DictReader((out1 / "split_manifest.csv").open(encoding="utf-8")))
    # rerun into the SAME dir with overwrite=False
    s2 = build_png_cache(png_root=str(png_root), split_csv=str(tmp / "split.csv"), out_dir=str(out1), overwrite=False)
    m2 = list(csv.DictReader((out1 / "split_manifest.csv").open(encoding="utf-8")))
    OUT["E6_png_cache_overwrite"] = {
        "first_run_manifest_rows": len(m1), "first_run_built": s1["built"],
        "second_run_built": s2["built"], "second_run_manifest_rows": len(m2),
        "claim_confirmed": len(m2) < len(m1),
    }
except Exception:
    OUT["E6_png_cache_overwrite"] = {"error": traceback.format_exc(limit=3)}

# ============ E7: P0-4 split fallback leak (empirical) ============
try:
    from src.data.dataset import CachedDataset
    cache = tmp / "npzcache"; cache.mkdir(parents=True, exist_ok=True)
    payload = {
        "ct": np.zeros((1, 32, 32), np.float32), "pet": np.zeros((1, 32, 32), np.float32),
        "mask": np.zeros((1, 32, 32), np.float32),
        "scale_meta_json": np.frombuffer(json.dumps({"patient_id": "001"}).encode(), dtype=np.uint8),
    }
    np.savez_compressed(str(cache / "001001.npz"), **payload)
    (cache / "001001_meta.json").write_text(json.dumps({"split": "test", "has_label": True}), encoding="utf-8")
    r = {}
    try:
        CachedDataset(str(cache), split="train"); r["with_meta_train_len"] = "constructed (leak!)"
    except ValueError as ex:
        r["with_meta_train_len"] = f"ValueError as expected: {str(ex)[:80]}"
    (cache / "001001_meta.json").unlink()
    try:
        ds = CachedDataset(str(cache), split="train")
        r["without_meta_train_len"] = len(ds)
        r["entry_split_assigned"] = ds.entries[0].split
    except ValueError as ex:
        r["without_meta_train_len"] = f"ValueError: {str(ex)[:80]}"
    r["claim_confirmed"] = (r.get("without_meta_train_len") == 1 and r.get("entry_split_assigned") == "train")
    OUT["E7_split_fallback_leak"] = r
except Exception:
    OUT["E7_split_fallback_leak"] = {"error": traceback.format_exc(limit=3)}

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(json.dumps(OUT, indent=2, ensure_ascii=False, default=str))
