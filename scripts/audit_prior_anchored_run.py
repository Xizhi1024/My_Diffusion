#!/usr/bin/env python3
"""Read-only audit of one prior-anchored 100-epoch cloud run.

This script is a READ-ONLY auditor.  It never modifies training code,
``resolved_config.yaml``, the prior artifact, checkpoints, or
``training_metrics.jsonl``.  The only thing it writes is a single JSON
report inside the run's own ``observations/`` directory.  Everything else
is printed to stdout so the report can be copied back from the cloud box.

Boundary rules enforced by design:

* the cloud run directory is the single source of truth (no assumption that
  local PNG / cache / checkpoints match the cloud);
* every reported path is repository-relative;
* no Formal H3 runtime SHA is refreshed and no fail-closed gate is bypassed;
* this is an EXPLORATORY run -- the report must not be read as a claim of
  clinical efficacy, production readiness, or causal routing benefit, and
  A/B/C/D are not yet a valid paired comparison.

Usage (run on the cloud box, from the repo root)::

    pixi run python scripts/audit_prior_anchored_run.py \
        --run-dir results/prior_anchored_router_100e/runs/run-20260726T133123.363531Z

An optional ``--output`` overrides the default
``<run-dir>/observations/audit.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"ERROR: PyYAML is required to read resolved_config.yaml: {exc}")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]

TOTAL_EPOCHS = 100
PHASE_OBSERVATION_EPOCHS = (10, 20, 30, 40, 100)
TABLE_EPOCHS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
CHECKPOINT_RE = re.compile(r"^ckpt_epoch(\d{4})\.pt$")


# --------------------------------------------------------------------------- #
# Phase contract (must match the runner / summarizer exactly)
# --------------------------------------------------------------------------- #
def phase_for_epoch(epoch: int) -> str:
    if epoch <= 0:
        return "not_started"
    if epoch <= 10:
        return "prior_frozen"
    if epoch <= 20:
        return "active_ramp"
    if epoch <= 30:
        return "active_only_hold"
    if epoch <= 40:
        return "destination_ramp"
    if epoch <= TOTAL_EPOCHS:
        return "full_adaptive"
    return "out_of_contract"


PHASE_RANGES = {
    "prior_frozen": (1, 10),
    "active_ramp": (11, 20),
    "active_only_hold": (21, 30),
    "destination_ramp": (31, 40),
    "full_adaptive": (41, TOTAL_EPOCHS),
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value: Any, width: int = 9) -> str:
    if value is None:
        return "--".rjust(width)
    if isinstance(value, bool):
        return str(value).rjust(width)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            return ("NaN" if math.isnan(float(value)) else "Inf").rjust(width)
        av = abs(float(value))
        if av != 0.0 and (av < 1e-3 or av >= 1e6):
            text = f"{value:.2e}"
        elif float(value).is_integer():
            text = f"{int(value)}"
        else:
            text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text[:width].rjust(width)
    text = str(value)
    return text[:width].rjust(width)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mapping_at(data: Any, *keys: str) -> Any:
    cursor = data
    for key in keys:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return {}
        cursor = cursor[key]
    return cursor if isinstance(cursor, Mapping) else {}


# --------------------------------------------------------------------------- #
# Flatten one metrics record
# --------------------------------------------------------------------------- #
def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Collapse {train, eval, validation} sections to a flat metric dict.

    ``train`` keys already carry their own prefix (``loss/``, ``frequency/``,
    ``perf/``, ``grad/``) so they are kept as-is; ``validation`` keys already
    carry ``val/``; ``eval`` keys are namespaced under ``eval/`` to avoid any
    collision.
    """
    flat: dict[str, Any] = {}
    for top in ("epoch", "step", "phase", "schema_version"):
        if top in record:
            flat[top] = record[top]
    train = record.get("train")
    if isinstance(train, Mapping):
        for key, value in train.items():
            flat[str(key)] = value
    validation = record.get("validation")
    if isinstance(validation, Mapping):
        for key, value in validation.items():
            flat[str(key)] = value
    eval_section = record.get("eval")
    if isinstance(eval_section, Mapping):
        for key, value in eval_section.items():
            flat[f"eval/{key}"] = value
    return flat


# --------------------------------------------------------------------------- #
# Metrics jsonl reader + completeness contract
# --------------------------------------------------------------------------- #
def read_metrics(path: Path) -> dict[str, Any]:
    expected = list(range(1, TOTAL_EPOCHS + 1))
    base = {
        "path": repo_relative(path, ROOT),
        "records_by_epoch": {},
        "line_count": 0,
        "invalid_lines": [],
        "non_finite": [],
        "duplicate_epochs": [],
        "out_of_range_epochs": [],
        "missing_epochs": [],
        "epoch_sequence_valid": False,
        "step_strictly_increasing": True,
        "phase_matches": True,
        "phase_mismatches": [],
        "final_step": None,
        "latest_three": [],
        "all_keys": set(),
    }
    if not path.is_file():
        base["status"] = "NOT_WRITTEN"
        base["missing_epochs"] = expected
        return base

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    base["line_count"] = len(raw_lines)
    records: list[tuple[int, dict[str, Any]]] = []
    previous_step: int | None = None
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            base["invalid_lines"].append({"line": line_number, "error": "blank line"})
            continue
        try:
            payload = json.loads(
                raw,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON token {token}")
                ),
            )
            if not isinstance(payload, dict):
                raise ValueError("record is not a JSON object")
            epoch = payload.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                raise ValueError("record lacks integer epoch")
        except Exception as exc:
            base["invalid_lines"].append(
                {"line": line_number, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        records.append((epoch, payload))

        step = payload.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            base["step_strictly_increasing"] = False
        elif previous_step is not None and step <= previous_step:
            base["step_strictly_increasing"] = False
        if isinstance(step, int) and not isinstance(step, bool):
            previous_step = step

        expected_phase = phase_for_epoch(epoch)
        if payload.get("phase") != expected_phase:
            base["phase_matches"] = False
            base["phase_mismatches"].append(
                {"epoch": epoch, "expected": expected_phase, "observed": payload.get("phase")}
            )

        flat = flatten_record(payload)
        for key, value in flat.items():
            if key in ("epoch", "step", "phase", "schema_version"):
                continue
            base["all_keys"].add(key)
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                base["non_finite"].append(
                    {"epoch": epoch, "key": key, "value": repr(value)}
                )

    counts = Counter(epoch for epoch, _ in records)
    base["duplicate_epochs"] = sorted(e for e, c in counts.items() if c > 1)
    base["out_of_range_epochs"] = sorted(
        {e for e, _ in records if not 1 <= e <= TOTAL_EPOCHS}
    )
    by_epoch: dict[int, dict[str, Any]] = {}
    for epoch, payload in records:
        by_epoch.setdefault(epoch, payload)
    for epoch in expected:
        if epoch not in by_epoch:
            base["missing_epochs"].append(epoch)
    observed = [e for e, _ in records]
    base["epoch_sequence_valid"] = observed == expected
    base["records_by_epoch"] = {
        e: flatten_record(p) for e, p in by_epoch.items() if 1 <= e <= TOTAL_EPOCHS
    }
    final = by_epoch.get(TOTAL_EPOCHS)
    if isinstance(final, Mapping) and isinstance(final.get("step"), int):
        base["final_step"] = final["step"]
    # latest three records by epoch order
    latest_epochs = sorted(e for e in by_epoch if 1 <= e <= TOTAL_EPOCHS)[-3:]
    base["latest_three"] = [
        {"epoch": e, "step": by_epoch[e].get("step"), "phase": by_epoch[e].get("phase")}
        for e in latest_epochs
    ]

    clean = (
        not base["invalid_lines"]
        and not base["non_finite"]
        and not base["duplicate_epochs"]
        and not base["out_of_range_epochs"]
        and not base["missing_epochs"]
        and base["epoch_sequence_valid"]
        and base["step_strictly_increasing"]
        and base["phase_matches"]
        and len(records) == TOTAL_EPOCHS
    )
    base["status"] = "PASS" if clean else "PARTIAL"
    return base


# --------------------------------------------------------------------------- #
# Checkpoint inspection (weights_only=True, never executes untrusted code)
# --------------------------------------------------------------------------- #
def inspect_final_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "path": repo_relative(path, ROOT),
        "exists": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "loadable_weights_only": False,
        "epoch": None,
        "step": None,
        "has_required_keys": False,
        "missing_keys": [],
        "monitoring": None,
        "rng_loader_generators": None,
        "error": None,
    }
    if not path.is_file():
        info["error"] = "checkpoint file not found"
        return info
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    if not isinstance(payload, Mapping):
        info["error"] = "checkpoint payload is not a mapping"
        return info
    info["loadable_weights_only"] = True
    info["epoch"] = payload.get("epoch")
    info["step"] = payload.get("step")
    required = ("model", "optimizer", "scheduler", "ema", "step", "config", "rng")
    missing = [k for k in required if k not in payload]
    info["missing_keys"] = missing
    info["has_required_keys"] = not missing
    monitoring = payload.get("monitoring")
    if isinstance(monitoring, Mapping):
        info["monitoring"] = {
            "best_combined_score": monitoring.get("best_combined_score"),
            "best_lesion_score": monitoring.get("best_lesion_score"),
            "best_image_score": monitoring.get("best_image_score"),
            "last_combined_improvement_epoch": monitoring.get(
                "last_combined_improvement_epoch"
            ),
            "epochs_since_improve": monitoring.get("epochs_since_improve"),
        }
    rng = payload.get("rng")
    if isinstance(rng, Mapping):
        lg = rng.get("loader_generators")
        if isinstance(lg, Mapping):
            info["rng_loader_generators"] = {
                k: ("None" if v is None else type(v).__name__)
                for k, v in lg.items()
            }
    return info


# --------------------------------------------------------------------------- #
# Per-10-epoch tables
# --------------------------------------------------------------------------- #
TRAIN_KEYS = [
    "loss/total",
    "loss/base_diffusion",
    "perf/lr",
    "perf/epoch_seconds",
    "grad/route_final",
    "grad/prior_active_final",
    "grad/prior_destination_final",
    "grad/projection_final",
]
ROUTE_KEYS = [
    "frequency/route_native_mass",
    "frequency/route_shallow_mass",
    "frequency/route_null_mass",
    "frequency/prior_anchor_prior_active_mean",
    "frequency/prior_anchor_active_mean",
    "frequency/prior_anchor_prior_active_mae",
    "frequency/prior_anchor_active_delta_abs_mean",
    "frequency/prior_anchor_active_delta_abs_max",
    "frequency/prior_anchor_active_progress",
    "frequency/prior_anchor_destination_progress",
    "frequency/prior_anchor_anchor_scale",
    "frequency/prior_anchor_monotonic_violation",
    "frequency/prior_anchor_monotonic_violation_fraction",
    "frequency/prior_anchor_curvature",
    "frequency/prior_anchor_budget_deviation",
    "frequency/prior_anchor_shallow_cost",
]
VAL_KEYS = [
    "val/mae",
    "val/mse",
    "val/psnr",
    "val/ssim",
    "val/stripe_score",
    "val/lesion_roi_l1",
    "val/lesion_peak_error_norm",
    "val/lesion_topq_peak_error_norm",
    "val/lesion_centroid_distance",
    "val/outside_inside_peak_ratio",
    "val/false_hotspot_proxy",
    "val/failure_rate",
    "val/small_lesion_underestimate",
    "val/lesion_sample_count",
    "val/small_lesion_sample_count",
]


def table_for(
    records_by_epoch: Mapping[int, Mapping[str, Any]],
    keys: list[str],
    epochs: tuple[int, ...] = TABLE_EPOCHS,
) -> dict[str, Any]:
    present: list[str] = []
    absent: list[str] = []
    for key in keys:
        seen = any(
            key in records_by_epoch.get(e, {}) for e in epochs
        )
        (present if seen else absent).append(key)
    rows: list[dict[str, Any]] = []
    for epoch in epochs:
        flat = records_by_epoch.get(epoch, {})
        row: dict[str, Any] = {"epoch": epoch}
        for key in keys:
            row[key] = flat.get(key)
        rows.append(row)
    return {"keys_present": present, "keys_absent": absent, "rows": rows}


def render_table(
    title: str,
    keys: list[str],
    rows: list[dict[str, Any]],
    *,
    extra_cols: tuple[tuple[str, str], ...] = (),
    chunk_size: int = 8,
) -> str:
    """Render a compact fixed-width table, split into <= chunk_size metric columns.

    Wide tables (routing, validation) are split vertically into chunks so each
    block stays readable in a normal terminal; every block repeats the epoch
    column.  ``extra_cols`` is ((label, src_key), ...) appended to every block.
    """
    extra = list(extra_cols)

    def render_block(block_keys: list[str], subtitle: str) -> list[str]:
        header_cols = [("epoch", "epoch")] + [
            (k.split("/")[-1], k) for k in block_keys
        ] + extra
        labels = [label for label, _ in header_cols]
        widths = [max(len(label), 9) for label in labels]
        sep = "  ".join("-" * w for w in widths)
        head = "  ".join(label.rjust(w) for label, w in zip(labels, widths))
        out = [f"  [{title} :: {subtitle}]", "  " + head, "  " + sep]
        for row in rows:
            cells = [fmt(row.get(src), w) for (_, src), w in zip(header_cols, widths)]
            out.append("  " + "  ".join(cells))
        return out

    if len(keys) <= chunk_size:
        return "\n".join(render_block(keys, "all"))
    lines: list[str] = []
    for start in range(0, len(keys), chunk_size):
        chunk = keys[start:start + chunk_size]
        if lines:
            lines.append("")
        lines.extend(render_block(chunk, f"cols {start + 1}-{start + len(chunk)}"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Phase-behavior verification
# --------------------------------------------------------------------------- #
def verify_phase_behavior(
    records_by_epoch: Mapping[int, Mapping[str, Any]],
    router_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    def get(epoch: int, key: str) -> float | None:
        v = records_by_epoch.get(epoch, {}).get(key)
        return float(v) if is_number(v) else None

    findings: list[str] = []
    final_scale_cfg = router_cfg.get("prior_anchor_final_scale", 0.10)

    # mass sum across all epochs
    mass_sums: list[dict[str, Any]] = []
    mass_ok = True
    for epoch in sorted(records_by_epoch):
        native = get(epoch, "frequency/route_native_mass")
        shallow = get(epoch, "frequency/route_shallow_mass")
        null = get(epoch, "frequency/route_null_mass")
        if None in (native, shallow, null):
            continue
        total = native + shallow + null
        if abs(total - 1.0) > 0.05:
            mass_ok = False
            mass_sums.append({"epoch": epoch, "sum": round(total, 4)})
    if not mass_sums:
        mass_sums = [{"note": "all observed epochs within 0.05 of 1.0"}]

    # warmup: adaptive heads should not release during prior_frozen (epoch<=10)
    warmup_leak = []
    for epoch in range(1, 11):
        ap = get(epoch, "frequency/prior_anchor_active_progress")
        dp = get(epoch, "frequency/prior_anchor_destination_progress")
        if (ap is not None and ap > 0.05) or (dp is not None and dp > 0.05):
            warmup_leak.append({"epoch": epoch, "active_progress": ap, "dest_progress": dp})

    # anchor scale at epoch 100 vs configured final scale
    anchor_e100 = get(TOTAL_EPOCHS, "frequency/prior_anchor_anchor_scale")
    anchor_match = None
    if anchor_e100 is None:
        anchor_match = "anchor_scale not recorded at epoch 100"
    else:
        try:
            target = float(final_scale_cfg)
        except (TypeError, ValueError):
            target = 0.10
        anchor_match = abs(anchor_e100 - target) <= max(1e-3, 0.05 * abs(target))

    # route collapse scans
    null_epochs = [e for e in sorted(records_by_epoch)
                   if (get(e, "frequency/route_null_mass") or 0.0) > 0.95]
    native_epochs = [e for e in sorted(records_by_epoch)
                     if (get(e, "frequency/route_native_mass") or 0.0) > 0.95]
    shallow_epochs = [e for e in sorted(records_by_epoch)
                      if (get(e, "frequency/route_shallow_mass") or 0.0) > 0.95]

    # progress phase transitions
    def at(epoch: int, key: str) -> float | None:
        return get(epoch, key)

    transitions = {
        "active_progress": {
            "e10": at(10, "frequency/prior_anchor_active_progress"),
            "e20": at(20, "frequency/prior_anchor_active_progress"),
            "e30": at(30, "frequency/prior_anchor_active_progress"),
            "e100": at(100, "frequency/prior_anchor_active_progress"),
        },
        "destination_progress": {
            "e10": at(10, "frequency/prior_anchor_destination_progress"),
            "e30": at(30, "frequency/prior_anchor_destination_progress"),
            "e40": at(40, "frequency/prior_anchor_destination_progress"),
            "e100": at(100, "frequency/prior_anchor_destination_progress"),
        },
    }

    # post-release gradient presence
    def grad_nonzero(epoch: int, key: str) -> Any:
        v = get(epoch, key)
        if v is None:
            return None
        return v > 0.0 and math.isfinite(v)

    grad_presence = {
        "active_after_release_e20": grad_nonzero(20, "grad/prior_active_final"),
        "active_full_e100": grad_nonzero(100, "grad/prior_active_final"),
        "destination_after_release_e40": grad_nonzero(40, "grad/prior_destination_final"),
        "destination_full_e100": grad_nonzero(100, "grad/prior_destination_final"),
    }

    if not mass_ok:
        findings.append(
            f"route mass native+shallow+null deviates from 1.0 in {len(mass_sums)} epochs"
        )
    if warmup_leak:
        findings.append(
            f"adaptive heads appear to release during prior_frozen in "
            f"{len(warmup_leak)} epoch(s)"
        )
    if anchor_match is False:
        findings.append(
            f"epoch-100 anchor_scale ({anchor_e100}) != configured final_scale "
            f"({final_scale_cfg})"
        )
    if len(null_epochs) >= 5:
        findings.append(f"null_mass>0.95 sustained for {len(null_epochs)} epochs (possible null collapse)")
    if len(native_epochs) >= 20:
        findings.append(f"native_mass>0.95 for {len(native_epochs)} epochs (possible native collapse)")
    if len(shallow_epochs) >= 5:
        findings.append(f"shallow_mass>0.95 for {len(shallow_epochs)} epochs")

    return {
        "mass_within_unity": mass_ok,
        "mass_deviations": mass_sums[:20],
        "warmup_leak": warmup_leak,
        "anchor_e100": anchor_e100,
        "anchor_final_scale_cfg": final_scale_cfg,
        "anchor_matches_config": anchor_match,
        "null_mass_gt_0p95_epochs": null_epochs,
        "native_mass_gt_0p95_epochs": native_epochs[:30],
        "shallow_mass_gt_0p95_epochs": shallow_epochs[:30],
        "progress_transitions": transitions,
        "grad_presence_post_release": grad_presence,
        "findings": findings,
    }


# --------------------------------------------------------------------------- #
# Anomaly scan
# --------------------------------------------------------------------------- #
def detect_anomalies(records_by_epoch: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    epochs_with_validation = []

    def add(epoch: int, key: str, kind: str, detail: str) -> None:
        anomalies.append({"epoch": epoch, "key": key, "kind": kind, "detail": detail})

    # monotonic / curvature / budget sudden worsening
    spike_keys = (
        "frequency/prior_anchor_monotonic_violation",
        "frequency/prior_anchor_monotonic_violation_fraction",
        "frequency/prior_anchor_curvature",
        "frequency/prior_anchor_budget_deviation",
        "frequency/prior_anchor_prior_active_mae",
    )
    for key in spike_keys:
        series: list[tuple[int, float]] = []
        for epoch in sorted(records_by_epoch):
            v = records_by_epoch[epoch].get(key)
            if is_number(v):
                series.append((epoch, float(v)))
        for i in range(1, len(series)):
            prev = series[i - 1][1]
            cur_e, cur = series[i]
            if prev <= 0:
                continue
            ratio = cur / prev
            if ratio >= 5.0 and cur > 1e-6:
                add(cur_e, key, "spike", f"{prev:.4g} -> {cur:.4g} (x{ratio:.1f})")

    # validation NaN handled in read_metrics; here we look at val lesion sample counts
    for epoch in sorted(records_by_epoch):
        flat = records_by_epoch[epoch]
        if any(k.startswith("val/") for k in flat):
            epochs_with_validation.append(epoch)
        n_lesion = flat.get("val/lesion_sample_count")
        n_small = flat.get("val/small_lesion_sample_count")
        if is_number(n_lesion) and n_lesion < 5:
            add(epoch, "val/lesion_sample_count", "low_sample",
                f"only {n_lesion} lesion samples; do not draw lesion conclusions")
        if is_number(n_small) and n_small < 5:
            add(epoch, "val/small_lesion_sample_count", "low_sample",
                f"only {n_small} small-lesion samples; do not draw small-lesion conclusions")
        stripe = flat.get("val/stripe_score")
        if is_number(stripe) and stripe > 1.5:
            add(epoch, "val/stripe_score", "stripe_high", f"{stripe:.3f} > 1.5")

    return {
        "anomalies": anomalies,
        "epochs_with_validation": epochs_with_validation,
    }


# --------------------------------------------------------------------------- #
# Best validation epoch (recompute the trainer's combined score)
# --------------------------------------------------------------------------- #
def combined_score(
    val_metrics: Mapping[str, Any], alpha: float, stripe_penalty: float
) -> tuple[float | None, float | None, float | None, dict[str, float]]:
    def g(key: str) -> float:
        v = val_metrics.get(key)
        return float(v) if is_number(v) else 0.0

    peak = g("val/lesion_peak_error_norm")
    topq_raw = val_metrics.get("val/lesion_topq_peak_error_norm")
    topq = float(topq_raw) if is_number(topq_raw) else peak
    centroid = g("val/lesion_centroid_distance")
    fail = g("val/failure_rate")
    mae = g("val/mae")
    ssim = g("val/ssim")
    stripe = g("val/stripe_score")
    parts = {"peak": peak, "topq": topq, "centroid": centroid, "fail": fail,
             "mae": mae, "ssim": ssim, "stripe": stripe}
    lesion = -(topq + 0.5 * peak + 0.02 * centroid + 0.5 * fail)
    image = ssim - mae - 0.1 * max(stripe - 1.0, 0.0)
    combined = alpha * lesion + (1.0 - alpha) * image - stripe_penalty * max(stripe - 1.0, 0.0)
    has = any(is_number(val_metrics.get(k)) for k in
              ("val/lesion_peak_error_norm", "val/ssim", "val/mae"))
    if not has:
        return None, None, None, parts
    return lesion, image, combined, parts


def best_validation_epoch(
    records_by_epoch: Mapping[int, Mapping[str, Any]],
    alpha: float,
    stripe_penalty: float,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for epoch in sorted(records_by_epoch):
        flat = records_by_epoch[epoch]
        if not any(k.startswith("val/") for k in flat):
            continue
        lesion, image, combined, parts = combined_score(flat, alpha, stripe_penalty)
        if combined is None:
            continue
        scored.append({
            "epoch": epoch,
            "lesion_score": lesion,
            "image_score": image,
            "combined": combined,
            **{f"part_{k}": v for k, v in parts.items()},
        })
    if not scored:
        return {"best_epoch": None, "reason": "no validation records with usable metrics"}
    scored.sort(key=lambda r: r["combined"], reverse=True)
    best = scored[0]
    return {
        "best_epoch": best["epoch"],
        "best_combined": best["combined"],
        "best_lesion_score": best["lesion_score"],
        "best_image_score": best["image_score"],
        "parts": {
            k: best[f"part_{k}"]
            for k in ("peak", "topq", "centroid", "fail", "mae", "ssim", "stripe")
        },
        "scored_epochs": len(scored),
        "top5": [
            {"epoch": r["epoch"], "combined": r["combined"],
             "lesion": r["lesion_score"], "image": r["image_score"]}
            for r in scored[:5]
        ],
    }


# --------------------------------------------------------------------------- #
# Main audit
# --------------------------------------------------------------------------- #
def audit(run_dir: Path, output: Path | None) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    root = ROOT.resolve()
    report: dict[str, Any] = {
        "audit_pipeline_id": "PRIOR_ANCHORED_ROUTER_100E_AUDIT_V1",
        "generated_at_utc": utc_now(),
        "run_dir": repo_relative(run_dir, root),
        "exploratory_only": True,
        "no_causal_claims": True,
    }

    # ---- artifacts present? ----
    state_path = run_dir / "state.json"
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "training_metrics.jsonl"
    logs_path = run_dir / "logs"

    state = read_json(state_path) if state_path.is_file() else {}
    config: Any = {}
    config_read_error: str | None = None
    if config_path.is_file():
        try:
            config = read_yaml(config_path)
        except Exception as exc:
            config_read_error = f"{type(exc).__name__}: {exc}"

    training_cfg = mapping_at(config, "training")
    runtime_cfg = mapping_at(config, "runtime")
    router_cfg = mapping_at(config, "modules", "residual_frequency", "cross_level_router")
    run_meta = mapping_at(config, "prior_anchored_run")
    best_cfg = mapping_at(training_cfg, "best_checkpoint")
    alpha = float(best_cfg.get("combined_alpha", 0.5))
    stripe_penalty = float(best_cfg.get("stripe_penalty", 0.3))

    # Locate artifacts via state/config (repo-relative), then resolve.
    def resolve(rel_or_abs: Any) -> Path | None:
        if not isinstance(rel_or_abs, str) or not rel_or_abs:
            return None
        p = Path(rel_or_abs)
        if not p.is_absolute():
            p = (root / p)
        return p

    prior_in_state = state.get("prior_artifact") if isinstance(state, Mapping) else None
    prior_in_cfg = run_meta.get("prior_artifact_path")
    metrics_in_state = state.get("training_metrics") if isinstance(state, Mapping) else None
    metrics_in_cfg = training_cfg.get("training_metrics_jsonl")
    ckpt_in_state = state.get("latest_checkpoint") if isinstance(state, Mapping) else None
    ckpt_dir_in_cfg = training_cfg.get("checkpoint_dir")

    # Prefer the state.json-declared paths for SHA cross-check (state vouches
    # for these), then fall back to resolved_config, then to run-dir defaults.
    prior_path = resolve(prior_in_state) or resolve(prior_in_cfg)
    metrics_path = (
        resolve(metrics_in_state) or resolve(metrics_in_cfg)
        or (run_dir / "training_metrics.jsonl")
    )
    checkpoint_dir = (
        resolve(ckpt_dir_in_cfg) or (run_dir / "checkpoints")
    )
    final_ckpt = (
        resolve(ckpt_in_state)
        or (checkpoint_dir / f"ckpt_epoch{TOTAL_EPOCHS:04d}.pt")
    )

    # ---- SHA cross-check ----
    sha_checks: list[dict[str, Any]] = []

    def sha_pair(label: str, expected: Any, path: Path | None) -> None:
        rec: dict[str, Any] = {"artifact": label}
        rec["expected_sha256"] = expected
        rec["path"] = repo_relative(path, root) if path else None
        rec["exists"] = bool(path and path.is_file())
        rec["actual_sha256"] = sha256_file(path) if rec["exists"] else None
        rec["match"] = (
            rec["exists"]
            and isinstance(expected, str)
            and rec["actual_sha256"] == expected
        ) if expected is not None else rec["exists"]
        sha_checks.append(rec)

    if isinstance(state, Mapping):
        sha_pair("resolved_config", state.get("resolved_config_sha256"), config_path)
        sha_pair("prior_artifact", state.get("prior_artifact_sha256"), prior_path)
        sha_pair(
            "training_metrics",
            state.get("training_metrics_sha256"),
            metrics_path if metrics_path else None,
        )

    # ---- metrics ----
    metrics = read_metrics(metrics_path) if metrics_path and metrics_path.is_file() else read_metrics(run_dir / "training_metrics.jsonl")

    # ---- checkpoint ----
    ckpt_info = inspect_final_checkpoint(final_ckpt) if final_ckpt else {
        "path": None, "exists": False, "error": "checkpoint_dir unresolved"
    }

    # ---- tables ----
    records = metrics["records_by_epoch"]
    train_table = table_for(records, TRAIN_KEYS)
    route_table = table_for(records, ROUTE_KEYS)
    val_table = table_for(records, VAL_KEYS)

    # ---- phase behavior ----
    phase_report = verify_phase_behavior(records, router_cfg)

    # ---- anomalies ----
    anomaly_report = detect_anomalies(records)

    # ---- best validation epoch ----
    best_report = best_validation_epoch(records, alpha, stripe_penalty)

    # ---- completeness gate ----
    state_ok = (
        isinstance(state, Mapping)
        and state.get("status") == "COMPLETE"
        and state.get("latest_epoch") == TOTAL_EPOCHS
        and state.get("current_stage") in (None, "null")
    )
    sha_ok = all(item.get("match") for item in sha_checks if item.get("expected_sha256"))
    sha_artifacts_exist = all(item.get("exists") for item in sha_checks)
    metrics_clean = metrics["status"] == "PASS"
    ckpt_ok = (
        ckpt_info.get("exists")
        and ckpt_info.get("loadable_weights_only")
        and ckpt_info.get("has_required_keys")
        and ckpt_info.get("epoch") == TOTAL_EPOCHS
    )
    step_match = (
        isinstance(metrics["final_step"], int)
        and isinstance(ckpt_info.get("step"), int)
        and metrics["final_step"] == ckpt_info["step"]
    )
    phase_obs_present = all(e in records for e in PHASE_OBSERVATION_EPOCHS)
    eval_val_every_10 = all(
        isinstance(records.get(e, {}).get("epoch"), int) for e in range(10, TOTAL_EPOCHS + 1, 10)
    ) and all(
        any(k.startswith("val/") for k in records.get(e, {}))
        and any(k.startswith("eval/") for k in records.get(e, {}))
        for e in range(10, TOTAL_EPOCHS + 1, 10)
    )

    gate = {
        "state_COMPLETE_latest100_stage_null": state_ok,
        "sha_artifacts_exist": sha_artifacts_exist,
        "sha_match_where_declared": sha_ok,
        "metrics_PASS_no_dup_no_gap_no_nan": metrics_clean,
        "epoch_step_strictly_increasing": metrics["step_strictly_increasing"],
        "phase_matches_per_epoch": metrics["phase_matches"],
        "checkpoint_weights_only_loadable": ckpt_info.get("loadable_weights_only", False),
        "checkpoint_has_required_keys": ckpt_info.get("has_required_keys", False),
        "checkpoint_epoch_100": ckpt_info.get("epoch") == TOTAL_EPOCHS,
        "epoch100_metric_step_equals_checkpoint_step": step_match,
        "phase_observation_epochs_present": phase_obs_present,
        "eval_and_validation_every_10_epochs": eval_val_every_10,
    }
    complete = all(gate.values())

    # ---- final vs best ----
    best_epoch = best_report.get("best_epoch")
    ckpt_best_epoch = (
        ckpt_info.get("monitoring", {}) or {}
    ).get("last_combined_improvement_epoch")
    final_is_best = (
        best_epoch == TOTAL_EPOCHS
        and (ckpt_best_epoch in (None, TOTAL_EPOCHS))
    )

    # ---- verdict ----
    if complete and not anomaly_report["anomalies"]:
        verdict = "RUN_COMPLETE_NUMERICALLY_STABLE"
    elif complete:
        verdict = "RUN_COMPLETE_WITH_ANOMALIES_TO_INVESTIGATE"
    elif metrics.get("status") == "NOT_WRITTEN" or not records:
        verdict = "RESULT_UNUSABLE_RERUN"
    else:
        verdict = "INCOMPLETE_REVIEW_GATE"

    report.update({
        "state": state if isinstance(state, Mapping) else {"raw": state},
        "config_read_error": config_read_error,
        "config_paths": {
            "checkpoint_dir": repo_relative(checkpoint_dir, root),
            "final_checkpoint": repo_relative(final_ckpt, root) if final_ckpt else None,
            "training_metrics_jsonl": repo_relative(metrics_path, root) if metrics_path else None,
            "prior_artifact": repo_relative(prior_path, root) if prior_path else None,
            "samples_dir": runtime_cfg.get("sample_dir"),
            "logs_dir": repo_relative(logs_path, root) if logs_path.is_dir() else None,
        },
        "router_config": {
            "policy": router_cfg.get("policy"),
            "prior_warmup_epochs": router_cfg.get("prior_warmup_epochs"),
            "prior_active_ramp_epochs": router_cfg.get("prior_active_ramp_epochs"),
            "prior_destination_warmup_epochs": router_cfg.get("prior_destination_warmup_epochs"),
            "prior_destination_ramp_epochs": router_cfg.get("prior_destination_ramp_epochs"),
            "prior_anchor_decay_end_epoch": router_cfg.get("prior_anchor_decay_end_epoch"),
            "prior_anchor_final_scale": router_cfg.get("prior_anchor_final_scale"),
            "h3_schedule_source": router_cfg.get("h3_schedule_source"),
            "h3_allow_unverified_preview_lineage": router_cfg.get(
                "h3_allow_unverified_preview_lineage"
            ),
        },
        "best_checkpoint_config": {"combined_alpha": alpha, "stripe_penalty": stripe_penalty},
        "sha_checks": sha_checks,
        "metrics": {k: v for k, v in metrics.items() if k != "records_by_epoch"},
        "metrics_all_keys": sorted(metrics["all_keys"]),
        "checkpoint": ckpt_info,
        "train_table": train_table,
        "route_table": route_table,
        "val_table": val_table,
        "phase_behavior": phase_report,
        "anomalies": anomaly_report,
        "best_validation": best_report,
        "ckpt_tracked_best_epoch": ckpt_best_epoch,
        "final_epoch_is_best": final_is_best,
        "completeness_gate": gate,
        "verdict": verdict,
    })

    # ---- write observations/audit.json (the ONLY write) ----
    out_path = output or (run_dir / "observations" / "audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _make_serializable(report)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=False)
    report["audit_json_path"] = repo_relative(out_path, root)
    return report


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


# --------------------------------------------------------------------------- #
# stdout rendering (this is what gets copied back from the cloud)
# --------------------------------------------------------------------------- #
def print_report(report: dict[str, Any]) -> None:
    bar = "=" * 78
    print(bar)
    print("PRIOR-ANCHORED ROUTER 100E -- READ-ONLY AUDIT (exploratory)")
    print("no causal / clinical / production claims; A/B/C/D not yet paired")
    print(f"run_dir : {report['run_dir']}")
    print(f"generated: {report['generated_at_utc']}")
    print(bar)

    gate = report["completeness_gate"]
    verdict = report["verdict"]
    print("\n1) COMPLETENESS VERDICT")
    print(f"   -> {verdict}")
    for key, value in gate.items():
        mark = "OK " if value else "XX "
        print(f"   {mark} {key}")

    print("\n2) KEY EVIDENCE")
    state = report.get("state", {})
    print(f"   state.status            : {state.get('status')}")
    print(f"   state.latest_epoch      : {state.get('latest_epoch')}")
    print(f"   state.current_stage     : {state.get('current_stage')!r}")
    if report.get("config_read_error"):
        print(f"   resolved_config parse   : ERROR {report['config_read_error']}")
    ckpt = report["checkpoint"]
    print(f"   checkpoint epoch        : {ckpt.get('epoch')}  step: {ckpt.get('step')}")
    print(f"   checkpoint weights_only : {ckpt.get('loadable_weights_only')}  "
          f"sha256: {(ckpt.get('sha256') or '')[:16]}...")
    print(f"   metrics final_step      : {report['metrics'].get('final_step')}")
    print(f"   metrics status          : {report['metrics'].get('status')}  "
          f"records(line_count): {report['metrics'].get('line_count')}")
    print(f"   phase mismatches        : {len(report['metrics'].get('phase_mismatches', []))}")
    print(f"   non-finite metric hits  : {len(report['metrics'].get('non_finite', []))}")
    print(f"   duplicate / missing     : "
          f"{report['metrics'].get('duplicate_epochs')} / "
          f"{report['metrics'].get('missing_epochs')[:10]}")
    print("   SHA cross-check:")
    for item in report["sha_checks"]:
        exp = (item.get("expected_sha256") or "")[:16]
        act = (item.get("actual_sha256") or "")[:16]
        mark = "OK " if item.get("match") else ("?? " if item.get("expected_sha256") is None else "XX ")
        print(f"     {mark}{item['artifact']:<16} exp={exp:<16} act={act:<16} "
              f"({item.get('path')})")

    print("\n3) PER-10-EPOCH METRIC TABLES")
    train = report["train_table"]
    print(f"   train keys present: {len(train['keys_present'])}/{len(TRAIN_KEYS)}  "
          f"absent: {train['keys_absent']}")
    print(render_table("training", TRAIN_KEYS, train["rows"]))
    print()
    # discover extra lesion losses actually present
    extra_loss = sorted(k for k in report["metrics_all_keys"]
                        if k.startswith("loss/") and "lesion" in k or "peak" in k or "topk" in k)
    if extra_loss:
        print(f"   lesion/peak loss keys observed: {extra_loss}")
    route = report["route_table"]
    print(f"\n   route keys present: {len(route['keys_present'])}/{len(ROUTE_KEYS)}  "
          f"absent: {route['keys_absent']}")
    print(render_table("routing", ROUTE_KEYS, route["rows"]))
    val = report["val_table"]
    print(f"\n   validation keys present: {len(val['keys_present'])}/{len(VAL_KEYS)}  "
          f"absent: {val['keys_absent']}")
    print(render_table("validation", VAL_KEYS, val["rows"]))

    print("\n4) PHASE-BEHAVIOR (5 phases)")
    pb = report["phase_behavior"]
    print(f"   native+shallow+null within unity : {pb['mass_within_unity']}")
    print(f"   warmup leak (epoch<=10)          : {len(pb['warmup_leak'])} epoch(s)")
    print(f"   anchor_scale @ e100              : {pb['anchor_e100']}  "
          f"(cfg final_scale={pb['anchor_final_scale_cfg']}, match={pb['anchor_matches_config']})")
    null_n = len(pb["null_mass_gt_0p95_epochs"])
    native_n = len(pb["native_mass_gt_0p95_epochs"])
    shallow_n = len(pb["shallow_mass_gt_0p95_epochs"])
    print(f"   mass>0.95 sustained epochs       : null={null_n} native={native_n} shallow={shallow_n}")
    print(f"   progress transitions             : {pb['progress_transitions']}")
    print(f"   grad presence post-release       : {pb['grad_presence_post_release']}")
    if pb["findings"]:
        print("   findings:")
        for f in pb["findings"]:
            print(f"     - {f}")
    else:
        print("   findings: none flagged by automated checks")

    print("\n5) ANOMALY LIST")
    anoms = report["anomalies"]["anomalies"]
    if not anoms:
        print("   none flagged by automated scans")
    else:
        for a in anoms:
            print(f"   - epoch {a['epoch']:<4} {a['key']:<48} [{a['kind']}] {a['detail']}")

    print("\n6) BEST VALIDATION EPOCH")
    best = report["best_validation"]
    if best.get("best_epoch") is None:
        print(f"   {best.get('reason', 'no best epoch identified')}")
    else:
        print(f"   best_epoch = {best['best_epoch']}  "
              f"(combined={best['best_combined']:.4f}, "
              f"lesion={best['best_lesion_score']:.4f}, image={best['best_image_score']:.4f})")
        print(f"   scored_epochs = {best['scored_epochs']}")
        print("   why: highest recomputed combined score "
              "(alpha*lesion + (1-alpha)*image - stripe_penalty*max(stripe-1,0)); "
              "lesion rewards low topq/peak/centroid/fail, image rewards high SSIM / low MAE / low stripe)")
        print("   top-5 by combined:")
        for row in best["top5"]:
            print(f"     e{row['epoch']:<4} combined={row['combined']:.4f} "
                  f"lesion={row['lesion']:.4f} image={row['image']:.4f}")

    print("\n7) FINAL CHECKPOINT vs BEST EPOCH")
    print(f"   final checkpoint epoch        : 100")
    print(f"   recomputed best val epoch     : {best.get('best_epoch')}")
    print(f"   checkpoint-tracked best epoch : {report.get('ckpt_tracked_best_epoch')} "
          f"(from monitoring.last_combined_improvement_epoch)")
    print(f"   final == best                 : {report['final_epoch_is_best']}")

    print("\n8) TECHNICAL VERDICT")
    print(f"   -> {verdict}")
    if verdict == "RUN_COMPLETE_NUMERICALLY_STABLE":
        print("   run is complete and numerically stable; no automated anomaly flagged.")
    elif verdict == "RUN_COMPLETE_WITH_ANOMALIES_TO_INVESTIGATE":
        print("   run is complete but the anomaly list above must be reviewed before any use.")
    elif verdict == "INCOMPLETE_REVIEW_GATE":
        print("   completeness gate failed above; do not treat results as final.")
    else:
        print("   results are not usable as-is; rerun is required.")

    print("\n9) NEXT STEPS (suggested, non-binding)")
    print("   - cross-check every item in the anomaly list against raw logs before interpreting trends;")
    print("   - confirm small-lesion / lesion sample counts are adequate before any lesion claim;")
    print("   - do NOT promote the direct-PNG prior to a Formal H3 PASS;")
    print("   - for any comparison, run the paired A/B/C/D ablation; a single run proves only")
    print("     internal observability and training integrity, not routing benefit.")
    print(f"\n   audit.json written to: {report.get('audit_json_path')}")
    print(bar)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="repository-relative path to the run directory")
    parser.add_argument("--root", default=str(ROOT),
                        help="repository root (default: auto from script location)")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: <run-dir>/observations/audit.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Keep stdout from crashing on any stray non-ASCII value under cp936/cp1252
    # consoles; the report itself is ASCII, but paths or exceptions may not be.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = parse_args(argv)
    global ROOT
    ROOT = Path(args.root).resolve()
    run_dir = (ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return 2
    output = Path(args.output).resolve() if args.output else None
    report = audit(run_dir, output)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
