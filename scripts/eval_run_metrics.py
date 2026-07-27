"""Evaluate a prior-anchored router run vs the 100e no-op baseline.

Reads training_metrics.jsonl, finds the best epoch by lesion peak error, prints
the per-eval-epoch lesion metrics + the shallow-mass tail (did experiment D keep
the shallow branch alive?), and compares the best epoch head-to-head against the
100e no-op baseline (results/audit.json ep70). Prints a verdict.

The JSONL record is nested: {"epoch":..., "train":{...}, "eval":{...},
"validation": {mae, lesion_peak_error_norm, ...}}. Validation metrics live under
the "validation" block (keys WITHOUT the val/ prefix); route diagnostics (e.g.
route_shallow_mass) live under the "train" block.

Run on the cloud box:
  pixi run python -u -m scripts.eval_run_metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

# 100e no-op baseline, best epoch 70 (from results/audit.json). Same data, split,
# normalization, eval protocol — the only differences are the D fix (de-zeroed
# shallow projections + shallow_cost_weight=0) and the 300ep schedule.
BASELINE = {
    "name": "100e no-op (ep70, best)",
    "peak": 0.1330, "topq": 0.1810, "roi": 0.1594, "failure": 0.3125,
    "small_under": 0.50, "outside_inside": 1.460, "mae": 0.01136,
    "psnr": 32.695, "ssim": 0.9659, "stripe": 1.2014, "shallow": 9.10e-7,
}

# exact keys inside the record["validation"] block
VAL_KEYS = {
    "peak": "lesion_peak_error_norm",
    "topq": "lesion_topq_peak_error_norm",
    "roi": "lesion_roi_l1",
    "failure": "failure_rate",
    "small_under": "small_lesion_underestimate",
    "outside_inside": "outside_inside_peak_ratio",
    "mae": "mae",
    "psnr": "psnr",
    "ssim": "ssim",
    "stripe": "stripe_score",
}


def _find_metrics(name: str) -> Optional[str]:
    hits = glob.glob("results/prior_anchored_router_300e/**/training_metrics.jsonl", recursive=True)
    if hits:
        return hits[0]
    return name if name and os.path.exists(name) else None


def _val(row: Dict[str, Any]) -> Dict[str, Any]:
    v = row.get("validation")
    return v if isinstance(v, dict) else {}


def _find_value(node: Any, substr: str) -> Optional[float]:
    """Recursively search nested dicts for the first numeric value whose key
    contains substr (e.g. route_shallow_mass under train -> frequency/...)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if substr in k and isinstance(v, (int, float)):
                return float(v)
        for v in node.values():
            found = _find_value(v, substr)
            if found is not None:
                return found
    return None


def _read(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/prior_anchored_router_300e/checkpoints/training_metrics.jsonl")
    args = ap.parse_args()

    path = _find_metrics(args.metrics)
    if not path:
        raise SystemExit("training_metrics.jsonl not found under results/prior_anchored_router_300e/")
    print(f"[eval] metrics = {path}")

    rows = _read(path)
    eval_rows = [r for r in rows if _val(r).get("lesion_peak_error_norm") is not None]
    if not eval_rows:
        sample_keys = sorted(rows[0].keys()) if rows else "empty"
        raise SystemExit(
            f"no validation.lesion_peak_error_norm found across {len(rows)} rows. "
            f"Sample row top-level keys: {sample_keys}"
        )

    def epoch_of(r):
        return int(r.get("epoch") or 0)
    eval_rows.sort(key=epoch_of)

    print(f"\n{'ep':>4} {'peak':>7} {'topq':>7} {'roi':>7} {'fail':>6} {'smund':>6} {'o/i':>6} {'psnr':>7} {'stripe':>7} {'shallow':>8}")
    for r in eval_rows:
        v = _val(r)
        sh = _find_value(r, "route_shallow_mass")
        print("{:>4} {:>7.4f} {:>7.4f} {:>7.4f} {:>6.3f} {:>6.3f} {:>6.2f} {:>7.2f} {:>7.4f} {:>8.2e}".format(
            epoch_of(r),
            v.get("lesion_peak_error_norm", float("nan")),
            v.get("lesion_topq_peak_error_norm", float("nan")),
            v.get("lesion_roi_l1", float("nan")),
            v.get("failure_rate", float("nan")),
            v.get("small_lesion_underestimate", float("nan")),
            v.get("outside_inside_peak_ratio", float("nan")),
            v.get("psnr", float("nan")),
            v.get("stripe_score", float("nan")),
            sh if sh is not None else float("nan"),
        ))

    best = min(eval_rows, key=lambda r: _val(r)["lesion_peak_error_norm"])
    be = epoch_of(best)
    vb = _val(best)
    cur = {k: vb.get(sub, float("nan")) for k, sub in VAL_KEYS.items()}

    # shallow tail (last 30% of ALL rows)
    all_sorted = sorted(rows, key=epoch_of)
    tail = all_sorted[len(all_sorted) * 7 // 10:]
    shallow_vals = [(_find_value(r, "route_shallow_mass") or 0.0) for r in tail]
    shallow_tail = sum(shallow_vals) / max(len(shallow_vals), 1)

    print("\n=== best epoch (min lesion peak error) ===")
    print(f"epoch = {be}")
    print(f"{'metric':<16} {'baseline':>10} {'this run':>10} {'delta':>10} {'%':>8}")
    for k in ["peak", "topq", "roi", "failure", "small_under", "outside_inside", "mae", "psnr", "ssim", "stripe"]:
        b = BASELINE[k]
        c = cur[k]
        d = c - b
        pct = 100.0 * d / abs(b) if b else float("nan")
        arrow = "↓ better" if d < 0 else ("↑ worse" if d > 0 else "=")
        print(f"{k:<16} {b:>10.4f} {c:>10.4f} {d:>+10.4f} {pct:>+7.1f}%  {arrow}")
    print(f"\nshallow_mass tail (last 30% of epochs): {shallow_tail:.3e}  "
          f"(baseline ep70: {BASELINE['shallow']:.3e}; D-alive threshold 1e-3)")
    print(f"shallow sustained ≥1e-3: {'YES' if shallow_tail >= 1e-3 else 'NO'}")

    dpeak = cur["peak"] - BASELINE["peak"]
    ppeak = 100.0 * dpeak / BASELINE["peak"]
    ptopq = 100.0 * (cur["topq"] - BASELINE["topq"]) / BASELINE["topq"]
    print("\n=== verdict ===")
    print(f"peak: {cur['peak']:.4f} vs {BASELINE['peak']:.4f} ({ppeak:+.1f}%)")
    print(f"topq: {cur['topq']:.4f} vs {BASELINE['topq']:.4f} ({ptopq:+.1f}%)")
    if ppeak <= -5.0 and shallow_tail >= 1e-3:
        print("D HELPED: peak improved ≥5% AND shallow stayed alive.")
    elif ppeak >= 5.0:
        print("D HURT: peak worse ≥5%. Check outside_inside trend + whether shallow_mass overshot in late epochs.")
    else:
        print("D NEUTRAL: peak within ±5%. Judge by topq/small_under/ROI trend.")


if __name__ == "__main__":
    main()
