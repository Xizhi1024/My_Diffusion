#!/usr/bin/env python3
"""Scan cloud run directories under ``results/``, summarize each run, and
filter them into a leaderboard + promotion shortlist.

READ-ONLY by design:

  * never modifies training code, ``resolved_config.yaml``, ``state.json``,
    ``training_metrics.jsonl``, checkpoints, or any other run artifact;
  * the only writes go to one summary directory (default
    ``results/cloud_runs_summary/``);
  * every reported path is repository-relative;
  * checkpoint hashing / ``torch.load`` are OFF by default and only enabled
    by explicit flags (``--with-sha`` / ``--with-checkpoint``), so a default
    scan is fast and never touches large checkpoint files.

This is a CROSS-RUN scanner. For a single run's deep audit use
``scripts/audit_prior_anchored_run.py``; for a single run's completeness
contract use ``scripts/summarize_prior_anchored_run.py``.  This script reuses
the pure helpers from ``audit_prior_anchored_run.py`` and adds:

  * recursive run discovery (any dir holding ``state.json`` or
    ``training_metrics.jsonl``);
  * a generic metrics reader (``total_epochs`` inferred from state / config /
    observed records, not hard-coded to 100);
  * a per-run grade (``COMPLETE_STABLE`` / ``COMPLETE_ANOMALY`` /
    ``ROUTER_NOOP`` / ``INCOMPLETE`` / ``FAILED`` / ``EMPTY``);
  * a ``leaderboard.csv`` + ``runs.json`` + ``promoted.json`` + terminal table.

Boundary (matches memory [[run-env-cloud-win]] and
[[prior-anchored-router-experiment]]):

  * run on the CLOUD box -- the local dev box only carries FAILED stubs under
    ``results/prior_anchored_router_100e/runs/``, not real metrics/checkpoints;
  * exploratory prior-anchored runs are tagged ``ROUTER_NOOP`` when the route
    mass collapsed; that is a structural observation, not a causal claim, and
    does not by itself authorize A/B/C/D paired promotion.

Usage (cloud box, from the repo root)::

    pixi run python scripts/summarize_cloud_runs.py \\
        --runs-root results \\
        --filter "prior_anchored*" \\
        --sort-by best_combined \\
        --promote-top 5 \\
        --out results/cloud_runs_summary

A broad scan (all experiment buckets)::

    pixi run python scripts/summarize_cloud_runs.py --runs-root results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"ERROR: PyYAML is required to read resolved_config.yaml: {exc}")
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the audited pure helpers rather than duplicating them.  These are
# stable, side-effect-free functions; if audit_prior_anchored_run.py changes
# them incompatibly the import will fail loudly at startup.
from scripts.audit_prior_anchored_run import (  # noqa: E402
    best_validation_epoch,
    combined_score,
    detect_anomalies,
    flatten_record,
    fmt,
    is_number,
    read_json,
    read_yaml,
    repo_relative,
    sha256_file,
    utc_now,
)

SUMMARY_PIPELINE_ID = "CLOUD_RUNS_SUMMARY_V1"

# Validation keys used for the compact leaderboard columns.  These are the
# same keys audit_prior_anchored_run.py consumes, so a run graded here is
# consistent with a single-run audit.
FINAL_VAL_KEYS = (
    "val/mae",
    "val/ssim",
    "val/psnr",
    "val/stripe_score",
    "val/lesion_peak_error_norm",
    "val/lesion_topq_peak_error_norm",
    "val/lesion_centroid_distance",
    "val/failure_rate",
    "val/lesion_roi_l1",
)
# Router keys (prior-anchored runs only).  Missing on baseline / v5 runs.
ROUTER_KEYS = (
    "frequency/route_native_mass",
    "frequency/route_shallow_mass",
    "frequency/route_null_mass",
    "frequency/prior_anchor_active_delta_abs_mean",
    "frequency/prior_anchor_active_delta_abs_max",
    "frequency/prior_anchor_prior_active_mae",
    "frequency/prior_anchor_shallow_cost",
)

# Router no-op thresholds (see memory [[prior-anchored-router-experiment]]):
# the 100e audit saw shallow collapse to ~1e-6 and active_delta dominated by
# regularizers.  A run below both floors over its final epoch is structurally
# inert, independent of whether training completed.
ROUTER_NOOP_SHALLOW = 1.0e-3
ROUTER_NOOP_ACTIVE_DELTA = 1.0e-3


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def discover_runs(
    runs_root: Path,
    *,
    filter_glob: str | None,
) -> list[Path]:
    """Return every run directory under ``runs_root``.

    A run directory is any directory that contains ``state.json`` or
    ``training_metrics.jsonl``.  ``filter_glob`` is matched against the
    repository-relative POSIX path so callers can scope e.g.
    ``"prior_anchored*"`` or ``"mechanism_validation_v2/*"``.
    """
    runs_root = runs_root.resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    seen: set[Path] = set()
    runs: list[Path] = []
    for marker in ("state.json", "training_metrics.jsonl"):
        for path in runs_root.rglob(marker):
            run_dir = path.parent
            if run_dir in seen:
                continue
            seen.add(run_dir)
            runs.append(run_dir)
    # Filter on repo-relative POSIX path so globs are intuitive.
    root = ROOT.resolve()
    if filter_glob:
        from fnmatch import fnmatch

        runs = [
            r for r in runs
            if fnmatch(repo_relative(r, root), f"*{filter_glob}*")
            or fnmatch(r.name, f"*{filter_glob}*")
        ]
    runs.sort(key=lambda p: repo_relative(p, root))
    return runs


# --------------------------------------------------------------------------- #
# Generic metrics reader (total_epochs is inferred, not hard-coded)
# --------------------------------------------------------------------------- #
def read_metrics_generic(
    path: Path,
    *,
    total_epochs: int | None,
) -> dict[str, Any]:
    """Read a training_metrics.jsonl without assuming a fixed epoch count.

    Mirrors audit_prior_anchored_run.read_metrics but parametrizes
    ``total_epochs`` (pass ``None`` to skip the missing / sequence contract and
    only report what was observed).
    """
    base: dict[str, Any] = {
        "path": repo_relative(path, ROOT.resolve()),
        "records_by_epoch": {},
        "line_count": 0,
        "invalid_lines": [],
        "non_finite": [],
        "duplicate_epochs": [],
        "out_of_range_epochs": [],
        "missing_epochs": [],
        "epoch_sequence_valid": None,
        "step_strictly_increasing": True,
        "final_step": None,
        "latest_three": [],
        "all_keys": set(),
        "max_epoch_observed": None,
    }
    if total_epochs is not None:
        expected = list(range(1, total_epochs + 1))
    else:
        expected = None
    if not path.is_file():
        base["status"] = "NOT_WRITTEN"
        if expected is not None:
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

        flat = flatten_record(payload)
        for key, value in flat.items():
            if key in ("epoch", "step", "phase", "schema_version"):
                continue
            base["all_keys"].add(key)
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                base["non_finite"].append({"epoch": epoch, "key": key, "value": repr(value)})

    counts = Counter(epoch for epoch, _ in records)
    base["duplicate_epochs"] = sorted(e for e, c in counts.items() if c > 1)
    by_epoch: dict[int, dict[str, Any]] = {}
    for epoch, payload in records:
        by_epoch.setdefault(epoch, payload)
    observed_epochs = sorted(by_epoch)
    base["max_epoch_observed"] = observed_epochs[-1] if observed_epochs else None
    if expected is not None:
        base["out_of_range_epochs"] = sorted(
            {e for e in observed_epochs if not 1 <= e <= (total_epochs or 0)}
        )
        base["missing_epochs"] = [e for e in expected if e not in by_epoch]
        base["epoch_sequence_valid"] = observed_epochs == expected
    final = by_epoch.get(observed_epochs[-1]) if observed_epochs else None
    if isinstance(final, Mapping) and isinstance(final.get("step"), int):
        base["final_step"] = final["step"]
    base["records_by_epoch"] = {
        e: flatten_record(p) for e, p in by_epoch.items()
    }
    base["latest_three"] = [
        {"epoch": e, "step": by_epoch[e].get("step"), "phase": by_epoch[e].get("phase")}
        for e in observed_epochs[-3:]
    ]

    clean = (
        not base["invalid_lines"]
        and not base["non_finite"]
        and not base["duplicate_epochs"]
        and not base["out_of_range_epochs"]
        and not base["missing_epochs"]
        and base["epoch_sequence_valid"] is not False
        and base["step_strictly_increasing"]
    )
    base["status"] = "PASS" if clean else "PARTIAL"
    return base


# --------------------------------------------------------------------------- #
# Per-run classification
# --------------------------------------------------------------------------- #
def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return int(value)


def _infer_total_epochs(
    state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    max_epoch_observed: int | None,
) -> int | None:
    """Prefer state, then resolved_config training.num_epochs, then observed."""
    for source in (state, config):
        if not isinstance(source, Mapping):
            continue
        training = source.get("training")
        if isinstance(training, Mapping):
            candidate = _coerce_int(training.get("num_epochs"))
            if candidate is not None:
                return candidate
    if isinstance(state, Mapping):
        candidate = _coerce_int(state.get("total_epochs"))
        if candidate is not None:
            return candidate
    return max_epoch_observed


def _infer_experiment(
    run_dir: Path,
    state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
) -> str:
    """Pick the most informative experiment label for this run."""
    for source in (state, config):
        if isinstance(source, Mapping):
            pid = source.get("pipeline_id")
            if isinstance(pid, str) and pid:
                return pid
    # Fall back to the experiment bucket (two levels up: runs/<bucket>/<run>).
    parts = run_dir.relative_to(ROOT.resolve()).parts if run_dir.is_relative_to(ROOT.resolve()) else run_dir.parts
    if len(parts) >= 2:
        return parts[-3] if parts[-2] == "runs" else parts[-2]
    return run_dir.name


def _final_val_metrics(
    records_by_epoch: Mapping[int, Mapping[str, Any]],
) -> dict[str, float | None]:
    """Pull validation keys from the last epoch that has any val/ entry."""
    if not records_by_epoch:
        return {key: None for key in FINAL_VAL_KEYS}
    last_epoch = max(records_by_epoch)
    flat = records_by_epoch[last_epoch]
    if not any(str(k).startswith("val/") for k in flat):
        # scan backwards for the most recent epoch that produced validation
        for epoch in sorted(records_by_epoch, reverse=True):
            candidate = records_by_epoch[epoch]
            if any(str(k).startswith("val/") for k in candidate):
                flat = candidate
                last_epoch = epoch
                break
    out: dict[str, float | None] = {"_epoch": last_epoch}
    for key in FINAL_VAL_KEYS:
        value = flat.get(key)
        out[key] = float(value) if is_number(value) else None
    return out


def _router_state(
    records_by_epoch: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize prior-anchored router behavior from the final epoch.

    Returns ``{present: bool, ...final router metrics..., noop: bool}``.
    ``present=False`` on runs that never emit ``frequency/route_*`` keys
    (baseline / v5), so router grading is skipped for them.
    """
    out: dict[str, Any] = {"present": False}
    if not records_by_epoch:
        return out
    epochs_with_route = [
        e for e in sorted(records_by_epoch)
        if any(k.startswith("frequency/route_") for k in records_by_epoch[e])
    ]
    if not epochs_with_route:
        return out
    out["present"] = True
    final_epoch = epochs_with_route[-1]
    flat = records_by_epoch[final_epoch]
    for key in ROUTER_KEYS:
        value = flat.get(key)
        out[key] = float(value) if is_number(value) else None
    out["final_epoch"] = final_epoch
    shallow = out.get("frequency/route_shallow_mass")
    delta = out.get("frequency/prior_anchor_active_delta_abs_mean")
    # Mean over the full_adaptive tail (last 50% of observed epochs) so a
    # single noisy epoch does not flip the verdict.
    tail_start = epochs_with_route[len(epochs_with_route) // 2]
    shallow_tail = [
        float(records_by_epoch[e].get("frequency/route_shallow_mass"))
        for e in epochs_with_route[tail_start <= e:]  # noqa: E203
        if is_number(records_by_epoch[e].get("frequency/route_shallow_mass"))
    ]
    shallow_tail_mean = (
        sum(shallow_tail) / len(shallow_tail) if shallow_tail else None
    )
    out["shallow_mass_tail_mean"] = shallow_tail_mean
    out["noop"] = (
        (shallow is not None and shallow < ROUTER_NOOP_SHALLOW)
        and (shallow_tail_mean is not None and shallow_tail_mean < ROUTER_NOOP_SHALLOW)
        and (delta is not None and delta < ROUTER_NOOP_ACTIVE_DELTA)
    )
    return out


def _alpha_stripe_from_config(config: Mapping[str, Any] | None) -> tuple[float, float]:
    if not isinstance(config, Mapping):
        return 0.5, 0.3
    training = config.get("training")
    if not isinstance(training, Mapping):
        return 0.5, 0.3
    best = training.get("best_checkpoint")
    if not isinstance(best, Mapping):
        return 0.5, 0.3
    alpha = best.get("combined_alpha", 0.5)
    stripe = best.get("stripe_penalty", 0.3)
    try:
        return float(alpha), float(stripe)
    except (TypeError, ValueError):
        return 0.5, 0.3


def classify_run(
    run_dir: Path,
    *,
    root: Path,
    with_checkpoint: bool,
    with_sha: bool,
) -> dict[str, Any]:
    """Collect every summary field for one run.  Never raises."""
    root = root.resolve()
    run_rel = repo_relative(run_dir, root)
    state_path = run_dir / "state.json"
    config_path = run_dir / "resolved_config.yaml"
    metrics_path = run_dir / "training_metrics.jsonl"

    state: Mapping[str, Any] | None = None
    state_error: str | None = None
    if state_path.is_file():
        try:
            loaded = read_json(state_path)
            state = loaded if isinstance(loaded, Mapping) else None
        except Exception as exc:
            state_error = f"{type(exc).__name__}: {exc}"

    config: Mapping[str, Any] | None = None
    config_error: str | None = None
    if config_path.is_file():
        try:
            loaded = read_yaml(config_path)
            config = loaded if isinstance(loaded, Mapping) else None
        except Exception as exc:
            config_error = f"{type(exc).__name__}: {exc}"

    # Probe for the metrics path declared in state/config first (the runner
    # may write it outside the run dir), then fall back to the run-local file.
    metrics_resolved = metrics_path
    for source in (state, config):
        if not isinstance(source, Mapping):
            continue
        candidate = None
        training = source.get("training")
        if isinstance(training, Mapping):
            candidate = training.get("training_metrics_jsonl")
        if not candidate:
            candidate = source.get("training_metrics")
        if isinstance(candidate, str) and candidate:
            p = Path(candidate)
            resolved = p if p.is_absolute() else (root / p)
            if resolved.is_file():
                metrics_resolved = resolved
                break

    metrics = read_metrics_generic(metrics_resolved, total_epochs=None)
    total_epochs = _infer_total_epochs(
        state, config, metrics.get("max_epoch_observed")
    )
    # Re-read with the contract if we now know the epoch budget.
    if total_epochs is not None:
        metrics = read_metrics_generic(
            metrics_resolved, total_epochs=total_epochs
        )

    experiment = _infer_experiment(run_dir, state, config)
    alpha, stripe_penalty = _alpha_stripe_from_config(config)
    records = metrics["records_by_epoch"]
    final_val = _final_val_metrics(records)
    router = _router_state(records)
    best = best_validation_epoch(records, alpha, stripe_penalty)
    anomalies = detect_anomalies(records)
    anomaly_count = len(anomalies.get("anomalies", []))

    # ---- checkpoint (optional, slow) ----
    ckpt: dict[str, Any] = {"inspected": False}
    if with_checkpoint:
        ckpt = _inspect_checkpoint(run_dir, state, config, total_epochs)

    # ---- sha (optional, slow on large checkpoints) ----
    metrics_sha = (
        sha256_file(metrics_resolved)
        if with_sha and metrics_resolved.is_file() else None
    )
    config_sha = (
        sha256_file(config_path) if with_sha and config_path.is_file() else None
    )

    summary: dict[str, Any] = {
        "run_dir": run_rel,
        "run_id": run_dir.name,
        "experiment": experiment,
        "state_status": state.get("status") if isinstance(state, Mapping) else None,
        "state_error": state_error,
        "state_failure": (
            state.get("failure") if isinstance(state, Mapping) else None
        ),
        "config_present": config_path.is_file(),
        "config_error": config_error,
        "config_sha256": config_sha,
        "total_epochs": total_epochs,
        "latest_epoch_state": (
            state.get("latest_epoch") if isinstance(state, Mapping) else None
        ),
        "max_epoch_observed": metrics.get("max_epoch_observed"),
        "metrics_path": repo_relative(metrics_resolved, root) if metrics_resolved else None,
        "metrics_status": metrics.get("status"),
        "metrics_sha256": metrics_sha,
        "metrics_line_count": metrics.get("line_count"),
        "invalid_lines": len(metrics.get("invalid_lines", [])),
        "non_finite": len(metrics.get("non_finite", [])),
        "duplicate_epochs": metrics.get("duplicate_epochs", []),
        "missing_epochs_count": len(metrics.get("missing_epochs", [])),
        "epoch_sequence_valid": metrics.get("epoch_sequence_valid"),
        "step_strictly_increasing": metrics.get("step_strictly_increasing"),
        "anomaly_count": anomaly_count,
        "router_present": router.get("present", False),
        "router": router,
        "best_epoch": best.get("best_epoch"),
        "best_combined": best.get("best_combined"),
        "best_lesion_score": best.get("best_lesion_score"),
        "best_image_score": best.get("best_image_score"),
        "best_scored_epochs": best.get("scored_epochs"),
        "final_val_epoch": final_val.get("_epoch"),
        "final_val": {k: v for k, v in final_val.items() if k != "_epoch"},
        "checkpoint": ckpt,
        "alpha": alpha,
        "stripe_penalty": stripe_penalty,
    }
    summary["grade"], summary["grade_reasons"] = grade(summary)
    return summary


def _inspect_checkpoint(
    run_dir: Path,
    state: Mapping[str, Any] | None,
    config: Mapping[str, Any] | None,
    total_epochs: int | None,
) -> dict[str, Any]:
    """Load final checkpoint metadata (weights_only).  Optional / best-effort."""
    info: dict[str, Any] = {"inspected": True, "exists": False}
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        info["error"] = f"torch unavailable: {exc}"
        return info
    ckpt_dir = None
    if isinstance(config, Mapping):
        training = config.get("training")
        if isinstance(training, Mapping):
            cd = training.get("checkpoint_dir")
            if isinstance(cd, str) and cd:
                ckpt_dir = Path(cd)
                if not ckpt_dir.is_absolute():
                    ckpt_dir = (ROOT.resolve() / ckpt_dir)
    if ckpt_dir is None:
        ckpt_dir = run_dir / "checkpoints"
    final_epoch = total_epochs or 0
    candidates = []
    if final_epoch > 0:
        candidates.append(ckpt_dir / f"ckpt_epoch{final_epoch:04d}.pt")
    if isinstance(state, Mapping) and isinstance(state.get("latest_checkpoint"), str):
        candidates.append(Path(state["latest_checkpoint"]))
    # fall back to the lexically largest ckpt_epoch*.pt
    if ckpt_dir.is_dir():
        candidates.extend(sorted(ckpt_dir.glob("ckpt_epoch*.pt"), reverse=True))
    ckpt_path = next((p for p in candidates if p and p.is_file()), None)
    if ckpt_path is None:
        info["error"] = "no checkpoint file found"
        return info
    info["path"] = repo_relative(ckpt_path, ROOT.resolve())
    info["exists"] = True
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        info["error"] = f"torch.load failed: {type(exc).__name__}: {exc}"
        return info
    if isinstance(payload, Mapping):
        info["epoch"] = payload.get("epoch")
        info["step"] = payload.get("step")
        monitoring = payload.get("monitoring")
        if isinstance(monitoring, Mapping):
            info["monitoring"] = {
                "best_combined_score": monitoring.get("best_combined_score"),
                "last_combined_improvement_epoch": monitoring.get(
                    "last_combined_improvement_epoch"
                ),
                "epochs_since_improve": monitoring.get("epochs_since_improve"),
            }
    return info


def grade(summary: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return a (label, reasons) pair.  Reasons double as CLI breadcrumbs."""
    reasons: list[str] = []
    state_status = summary.get("state_status")
    metrics_status = summary.get("metrics_status")
    records_empty = not summary.get("max_epoch_observed")
    router = summary.get("router") or {}
    router_noop = bool(router.get("noop")) if router.get("present") else False

    if records_empty and metrics_status in (None, "NOT_WRITTEN"):
        if state_status == "FAILED":
            label = "FAILED"
            reasons.append("state.json reports FAILED and no metrics were written")
        else:
            label = "EMPTY"
            reasons.append("no training_metrics.jsonl and no records observed")
        return label, reasons

    if state_status == "FAILED":
        label = "FAILED"
        reasons.append("state.json reports FAILED")
        if summary.get("state_failure"):
            reasons.append(f"failure: {summary['state_failure']}")
        return label, reasons

    if metrics_status in (None, "NOT_WRITTEN"):
        label = "EMPTY"
        reasons.append("training_metrics.jsonl not written")
        return label, reasons

    # Completeness gate (relaxed: state may be absent on salvaged cloud runs).
    complete_checks = {
        "metrics_PASS": metrics_status == "PASS",
        "no_invalid_lines": summary.get("invalid_lines", 0) == 0,
        "no_non_finite": summary.get("non_finite", 0) == 0,
        "no_duplicate_epochs": len(summary.get("duplicate_epochs", [])) == 0,
        "epoch_sequence_valid": summary.get("epoch_sequence_valid") is not False,
        "step_strictly_increasing": summary.get("step_strictly_increasing") is True,
    }
    missing_epochs = summary.get("missing_epochs_count", 0)
    total = summary.get("total_epochs")
    if total is not None:
        complete_checks["all_epochs_present"] = missing_epochs == 0
    if isinstance(summary.get("latest_epoch_state"), int) and total is not None:
        complete_checks["state_latest_equals_total"] = (
            summary["latest_epoch_state"] == total
        )

    failed_checks = [k for k, v in complete_checks.items() if not v]
    if failed_checks:
        label = "INCOMPLETE"
        reasons.append(f"completeness gate failed: {', '.join(failed_checks)}")
        if router_noop:
            label = "ROUTER_NOOP"
            reasons.append(
                f"route mass collapsed (shallow={router.get('frequency/route_shallow_mass'):.2e}, "
                f"active_delta={router.get('frequency/prior_anchor_active_delta_abs_mean'):.2e})"
            )
        return label, reasons

    if summary.get("anomaly_count", 0) > 0:
        label = "COMPLETE_ANOMALY"
        reasons.append(f"{summary['anomaly_count']} anomaly(ies) flagged (spike / low-sample / stripe)")
    else:
        label = "COMPLETE_STABLE"

    if router_noop:
        # Orthogonal sub-label: complete but structurally inert router.
        label = "ROUTER_NOOP" if label == "COMPLETE_STABLE" else f"{label}+ROUTER_NOOP"
        reasons.append(
            f"router no-op at final epoch "
            f"(shallow={router.get('frequency/route_shallow_mass'):.2e} < "
            f"{ROUTER_NOOP_SHALLOW:.0e}, "
            f"active_delta={router.get('frequency/prior_anchor_active_delta_abs_mean'):.2e} < "
            f"{ROUTER_NOOP_ACTIVE_DELTA:.0e})"
        )
    return label, reasons


# --------------------------------------------------------------------------- #
# Leaderboard rendering + outputs
# --------------------------------------------------------------------------- #
SORT_KEYS = (
    "best_combined",
    "best_lesion_score",
    "best_image_score",
    "final_val/val/mae",
    "final_val/val/ssim",
    "best_epoch",
    "max_epoch_observed",
    "anomaly_count",
    "experiment",
    "run_id",
)


def _sort_value(summary: Mapping[str, Any], key: str) -> float:
    if key.startswith("final_val/"):
        return -float(summary.get("final_val", {}).get(key.split("/", 1)[1]) or 0.0)
    if key in ("best_combined", "best_lesion_score", "best_image_score",
               "best_epoch", "max_epoch_observed"):
        return -float(summary.get(key) or 0.0)
    if key == "anomaly_count":
        return float(summary.get(key) or 0)
    return 0.0


def _grade_rank(label: str) -> int:
    order = {
        "COMPLETE_STABLE": 0,
        "COMPLETE_ANOMALY": 1,
        "COMPLETE_ANOMALY+ROUTER_NOOP": 2,
        "ROUTER_NOOP": 3,
        "INCOMPLETE": 4,
        "FAILED": 5,
        "EMPTY": 6,
    }
    return order.get(label, 9)


def sort_summaries(
    summaries: list[dict[str, Any]],
    *,
    sort_by: str,
) -> list[dict[str, Any]]:
    """Stable sort: usable grades first, then by ``sort_by`` (descended)."""
    return sorted(
        summaries,
        key=lambda s: (
            _grade_rank(s.get("grade", "")),
            _sort_value(s, sort_by) if sort_by in SORT_KEYS else 0.0,
            s.get("run_id", ""),
        ),
    )


# Leaderboard columns: (csv_header, accessor_into_summary)
LEADERBOARD_COLUMNS: Sequence[tuple[str, str]] = (
    ("run_id", "run_id"),
    ("experiment", "experiment"),
    ("grade", "grade"),
    ("state_status", "state_status"),
    ("total_epochs", "total_epochs"),
    ("max_epoch_observed", "max_epoch_observed"),
    ("metrics_status", "metrics_status"),
    ("best_epoch", "best_epoch"),
    ("best_combined", "best_combined"),
    ("best_lesion", "best_lesion_score"),
    ("best_image", "best_image_score"),
    ("final_val_mae", "final_val/val/mae"),
    ("final_val_ssim", "final_val/val/ssim"),
    ("final_val_peak_err", "final_val/val/lesion_peak_error_norm"),
    ("final_val_stripe", "final_val/val/stripe_score"),
    ("final_shallow_mass", "router/frequency/route_shallow_mass"),
    ("final_native_mass", "router/frequency/route_native_mass"),
    ("final_null_mass", "router/frequency/route_null_mass"),
    ("final_active_delta", "router/frequency/prior_anchor_active_delta_abs_mean"),
    ("router_noop", "router/noop"),
    ("anomaly_count", "anomaly_count"),
    ("missing_epochs", "missing_epochs_count"),
    ("run_dir", "run_dir"),
)


def _cell(summary: Mapping[str, Any], accessor: str) -> Any:
    cursor: Any = summary
    for part in accessor.split("/"):
        if isinstance(cursor, Mapping) and part in cursor:
            cursor = cursor[part]
        else:
            return ""
    if isinstance(cursor, bool):
        return "true" if cursor else "false"
    if isinstance(cursor, float):
        return cursor if math.isfinite(cursor) else ""
    if isinstance(cursor, (list, dict, set)):
        return json.dumps(cursor, ensure_ascii=False, default=str)
    return cursor


def write_leaderboard_csv(
    summaries: list[Mapping[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header for header, _ in LEADERBOARD_COLUMNS])
        for summary in summaries:
            writer.writerow(
                [_cell(summary, accessor) for _, accessor in LEADERBOARD_COLUMNS]
            )


def _make_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_make_serializable(v) for v in obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def write_json_file(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_make_serializable(obj), handle, indent=2, ensure_ascii=False)


def build_promoted(
    summaries: list[Mapping[str, Any]],
    *,
    promote_top: int,
) -> dict[str, Any]:
    """Shortlist the top complete runs and explain why each was chosen."""
    usable = [s for s in summaries if s.get("grade") in ("COMPLETE_STABLE", "COMPLETE_ANOMALY")]
    # Among usable, prefer non-router-no-op runs; record no-op ones separately.
    candidates = [s for s in usable if not (s.get("router") or {}).get("noop")]
    noop_within_complete = [s for s in usable if (s.get("router") or {}).get("noop")]
    chosen = candidates[:promote_top]
    return {
        "promote_top": promote_top,
        "usable_complete_count": len(usable),
        "non_noop_candidate_count": len(candidates),
        "chosen": [
            {
                "run_id": s["run_id"],
                "run_dir": s["run_dir"],
                "experiment": s["experiment"],
                "grade": s["grade"],
                "best_epoch": s.get("best_epoch"),
                "best_combined": s.get("best_combined"),
                "best_lesion_score": s.get("best_lesion_score"),
                "best_image_score": s.get("best_image_score"),
                "final_val_mae": (s.get("final_val") or {}).get("val/mae"),
                "final_val_ssim": (s.get("final_val") or {}).get("val/ssim"),
                "router_present": s.get("router_present"),
                "reason": (
                    "highest recomputed combined score among complete, "
                    "numerically stable, non-no-op runs"
                ),
            }
            for s in chosen
        ],
        "complete_but_router_noop": [
            {
                "run_id": s["run_id"],
                "run_dir": s["run_dir"],
                "best_combined": s.get("best_combined"),
                "final_shallow_mass": (s.get("router") or {}).get(
                    "frequency/route_shallow_mass"
                ),
                "final_active_delta": (s.get("router") or {}).get(
                    "frequency/prior_anchor_active_delta_abs_mean"
                ),
            }
            for s in noop_within_complete
        ],
        "note": (
            "Promotion here only means 'complete and worth a single-run deep "
            "audit'. It is NOT causal / production authorization, and does not "
            "satisfy the A/B/C/D paired-ablation contract in "
            "configs/experiments/prior_anchored_paired_ablation_v1.yaml."
        ),
    }


def render_terminal_table(
    summaries: list[Mapping[str, Any]],
    *,
    top: int | None,
) -> str:
    """Compact fixed-width leaderboard for the console."""
    if not summaries:
        return "(no runs discovered)"
    shown = summaries if top is None else summaries[:top]
    cols = [
        ("run_id", 28),
        ("grade", 22),
        ("st", 8),
        ("max_ep", 7),
        ("best_ep", 7),
        ("combined", 9),
        ("val_mae", 8),
        ("val_ssim", 8),
        ("shallow", 9),
        ("delta", 9),
        ("anom", 5),
    ]
    header = "  ".join(label.rjust(width) for label, width in cols)
    sep = "  ".join("-" * width for _, width in cols)
    lines = [header, sep]

    def g(s: Mapping[str, Any], *path: str, width: int = 9) -> str:
        cursor: Any = s
        for p in path:
            if isinstance(cursor, Mapping) and p in cursor:
                cursor = cursor[p]
            else:
                return "--".rjust(width)
        return fmt(cursor, width)

    for s in shown:
        lines.append(
            "  ".join(
                [
                    str(s.get("run_id", ""))[:28].rjust(28),
                    str(s.get("grade", ""))[:22].rjust(22),
                    str(s.get("state_status") or "-")[:8].rjust(8),
                    g(s, "max_epoch_observed", width=7),
                    g(s, "best_epoch", width=7),
                    g(s, "best_combined", width=9),
                    g(s, "final_val", "val/mae", width=8),
                    g(s, "final_val", "val/ssim", width=8),
                    g(s, "router", "frequency/route_shallow_mass", width=9),
                    g(s, "router", "frequency/prior_anchor_active_delta_abs_mean", width=9),
                    g(s, "anomaly_count", width=5),
                ]
            )
        )
    return "\n".join(lines)


def grade_histogram(summaries: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    return dict(Counter(s.get("grade", "?") for s in summaries))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-root",
        default="results",
        help="root directory to scan for runs (default: results)",
    )
    parser.add_argument(
        "--root",
        default=str(ROOT),
        help="repository root (default: auto from script location)",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="glob filter on run path or experiment name "
        '(e.g. "prior_anchored*", "mechanism_validation_v2/*")',
    )
    parser.add_argument(
        "--sort-by",
        default="best_combined",
        choices=SORT_KEYS,
        help="leaderboard sort key within a grade bucket (default: best_combined)",
    )
    parser.add_argument(
        "--promote-top",
        type=int,
        default=5,
        help="number of runs to shortlist in promoted.json (default: 5)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="limit the terminal table to the top N runs (default: show all)",
    )
    parser.add_argument(
        "--out",
        default="results/cloud_runs_summary",
        help="output directory (default: results/cloud_runs_summary)",
    )
    parser.add_argument(
        "--with-checkpoint",
        action="store_true",
        help="inspect the final checkpoint (torch.load weights_only); slow",
    )
    parser.add_argument(
        "--with-sha",
        action="store_true",
        help="SHA-256 the metrics jsonl and resolved_config; slow on large files",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="do not write any file; only print to stdout",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    args = parse_args(argv)
    global ROOT
    ROOT = Path(args.root).resolve()
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root

    try:
        run_dirs = discover_runs(runs_root, filter_glob=args.filter)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"[scan] root={repo_relative(runs_root, ROOT)} filter={args.filter!r}")
    print(f"[scan] discovered {len(run_dirs)} run directory(ies)")

    summaries: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            summary = classify_run(
                run_dir,
                root=ROOT,
                with_checkpoint=args.with_checkpoint,
                with_sha=args.with_sha,
            )
        except Exception as exc:
            summary = {
                "run_dir": repo_relative(run_dir, ROOT),
                "run_id": run_dir.name,
                "experiment": "?",
                "grade": "EMPTY",
                "grade_reasons": [f"classify raised {type(exc).__name__}: {exc}"],
            }
        summaries.append(summary)

    summaries = sort_summaries(summaries, sort_by=args.sort_by)
    histogram = grade_histogram(summaries)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    promoted = build_promoted(summaries, promote_top=args.promote_top)
    report = {
        "pipeline_id": SUMMARY_PIPELINE_ID,
        "generated_at_utc": utc_now(),
        "runs_root": repo_relative(runs_root, ROOT),
        "filter": args.filter,
        "sort_by": args.sort_by,
        "run_count": len(summaries),
        "grade_histogram": histogram,
        "summaries": summaries,
        "promoted": promoted,
        "boundary": {
            "read_only": True,
            "exploratory_only": True,
            "no_causal_claims": True,
            "cloud_runtime_required": True,
            "note": (
                "Cross-run leaderboard for triage. Not a completeness contract "
                "(use summarize_prior_anchored_run.py) nor a deep audit (use "
                "audit_prior_anchored_run.py). Router no-op tags reflect the "
                "structural observation in memory [[prior-anchored-router-experiment]]."
            ),
        },
    }

    if not args.no_write:
        write_json_file(report, out_dir / "runs.json")
        write_json_file(promoted, out_dir / "promoted.json")
        write_leaderboard_csv(summaries, out_dir / "leaderboard.csv")
        # also a stable README so the directory is self-describing
        readme = out_dir / "README.md"
        readme.parent.mkdir(parents=True, exist_ok=True)
        readme.write_text(
            _render_readme(histogram, promoted, repo_relative(out_dir, ROOT)),
            encoding="utf-8",
        )

    bar = "=" * 78
    print(bar)
    print("CLOUD RUNS -- CROSS-RUN SUMMARY (exploratory, read-only)")
    print(f"runs_root : {repo_relative(runs_root, ROOT)}")
    print(f"sort_by   : {args.sort_by}   promote_top: {args.promote_top}")
    print(bar)
    print("\nGRADE HISTOGRAM")
    for label, count in sorted(histogram.items(), key=lambda kv: (_grade_rank(kv[0]), kv[0])):
        print(f"  {label:<28} {count}")
    print(f"  {'TOTAL':<28} {len(summaries)}")

    print("\nLEADERBOARD")
    print(render_terminal_table(summaries, top=args.top))

    print("\nPROMOTED SHORTLIST")
    if not promoted["chosen"]:
        print("  (no COMPLETE_STABLE / non-no-op run to promote)")
    else:
        for entry in promoted["chosen"]:
            print(
                f"  - {entry['run_id']:<28} grade={entry['grade']:<18} "
                f"best_epoch={entry['best_epoch']}  combined={entry['best_combined']}"
            )
    if promoted["complete_but_router_noop"]:
        print(
            f"\n  complete-but-router-no-op ({len(promoted['complete_but_router_noop'])}):"
        )
        for entry in promoted["complete_but_router_noop"]:
            print(
                f"    - {entry['run_id']:<28} combined={entry['best_combined']}  "
                f"shallow={entry['final_shallow_mass']}  delta={entry['final_active_delta']}"
            )

    print("\nNEXT STEPS (suggested, non-binding)")
    print("  - drill into any run with scripts/audit_prior_anchored_run.py --run-dir <run_dir>")
    print("  - confirm completeness contract with scripts/summarize_prior_anchored_run.py --run-dir <run_dir>")
    print("  - do NOT promote a router-no-op run to a paired ablation; it is structurally inert")
    print("    (see memory [[prior-anchored-router-experiment]] and [[artifact-safety-weight-dead-key]])")
    if not args.no_write:
        print(f"\n  artifacts:")
        print(f"    {repo_relative(out_dir, ROOT)}/runs.json")
        print(f"    {repo_relative(out_dir, ROOT)}/promoted.json")
        print(f"    {repo_relative(out_dir, ROOT)}/leaderboard.csv")
        print(f"    {repo_relative(out_dir, ROOT)}/README.md")
    print(bar)
    return 0


def _render_readme(
    histogram: Mapping[str, int],
    promoted: Mapping[str, Any],
    out_rel: str,
) -> str:
    lines = [
        "# Cloud runs summary",
        "",
        f"Generated by `scripts/summarize_cloud_runs.py`. Output dir: `{out_rel}`.",
        "",
        "This is a **cross-run triage** artifact, read-only and exploratory.",
        "It is NOT a completeness contract (use `summarize_prior_anchored_run.py`)",
        "and NOT a deep audit (use `audit_prior_anchored_run.py`).",
        "",
        "## Grade histogram",
        "",
        "| grade | count |",
        "|---|---|",
    ]
    for label, count in sorted(histogram.items(), key=lambda kv: (_grade_rank(kv[0]), kv[0])):
        lines.append(f"| `{label}` | {count} |")
    lines += [
        "",
        "## Files",
        "",
        "- `leaderboard.csv` -- one row per run, compact columns for spreadsheet sorting.",
        "- `runs.json` -- full per-run records (the source of truth for this directory).",
        "- `promoted.json` -- shortlist of complete, non-no-op runs + reasons.",
        "",
        "## Grade legend",
        "",
        "- `COMPLETE_STABLE` -- completeness gate passes and no anomaly flagged.",
        "- `COMPLETE_ANOMALY` -- gate passes but spikes / low-sample / stripe flagged.",
        "- `ROUTER_NOOP` -- route mass collapsed at the final epoch (prior-anchored runs).",
        "- `INCOMPLETE` -- completeness gate failed (dup/gap/non-finite/step regress).",
        "- `FAILED` -- `state.json` reports FAILED.",
        "- `EMPTY` -- no `training_metrics.jsonl` and no records.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
