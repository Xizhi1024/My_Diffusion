# -*- coding: utf-8 -*-
"""Consistency: no config trips the new guards (unknown losses, noise enabled=false, inert losses)."""
import ast, json, re
from pathlib import Path
import yaml
root = Path(__file__).resolve().parent.parent

# collect known loss names from _build_loss
src = (root / "src/model/slmf_bbdm.py").read_text(encoding="utf-8")
tree = ast.parse(src)
known = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_build_loss":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Name) and sub.left.id == "name":
                for comp in sub.comparators:
                    if isinstance(comp, ast.Constant):
                        known.add(comp.value)
issues = []
for cfg_path in sorted((root / "configs/experiments").glob("*.yaml")):
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    losses = cfg.get("losses") or {}
    for lname, lcfg in losses.items():
        if lname not in known and not isinstance(lcfg, bool):
            issues.append(f"{cfg_path.name}: unknown loss '{lname}'")
        if lname in known and isinstance(lcfg, dict):
            enabled = lcfg.get("enabled", True)
            # inert combos now guarded
            if lname == "segmenter_consistency" and enabled and not (cfg.get("segmenter", {}) or {}).get("enabled", False) and not ((cfg.get("model", {}) or {}).get("segmenter", {}) or {}).get("enabled", False):
                issues.append(f"{cfg_path.name}: segmenter_consistency without segmenter (would raise)")
            if lname == "heteroscedastic_nll" and enabled and not (cfg.get("model", {}) or {}).get("enable_heteroscedastic", True):
                issues.append(f"{cfg_path.name}: heteroscedastic_nll without heteroscedastic head (would raise)")
    modules = cfg.get("modules") or {}
    for mkey in ("bbdm_bridge", "scale_adaptive_noise", "noise"):
        m = modules.get(mkey)
        if isinstance(m, dict) and m.get("enabled", True) is False and "name" in m:
            issues.append(f"{cfg_path.name}: modules.{mkey}.enabled=false (would raise)")
    es = ((cfg.get("runtime") or {}).get("early_stopping") or {})
    ev = (cfg.get("runtime") or {}).get("eval_interval")
    if es.get("enabled") and ev and es.get("patience", 40) < ev:
        issues.append(f"{cfg_path.name}: patience {es.get('patience')} < eval_interval {ev} (would raise)")
print(json.dumps({"known_losses": sorted(known), "issues": issues}, indent=2, ensure_ascii=False))
