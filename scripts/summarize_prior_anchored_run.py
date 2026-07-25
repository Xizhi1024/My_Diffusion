#!/usr/bin/env python3
"""Summarize one prior-anchored 100-epoch run without changing training state."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_prior_anchored_router_100e import (  # noqa: E402
    CHECKPOINT_PATTERN,
    PHASE_OBSERVATION_EPOCHS,
    PIPELINE_ID,
    SCHEMA_VERSION,
    TOTAL_EPOCHS,
    RunnerError,
    _checkpoint_dir_from_config,
    inspect_resume_checkpoint,
    normalize_repo_relative,
    portable_path,
    read_json,
    read_yaml,
    resolve_repo_path,
    sha256_file,
    utc_now,
    write_json_atomic,
)

SUMMARY_PIPELINE_ID = "PRIOR_ANCHORED_ROUTER_100E_SUMMARY_V1"


def phase_for_epoch(epoch: int) -> str:
    """Return the human-facing phase after the named epoch has completed."""

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


def _json_scalar(value: Any) -> int | float | bool | str | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _interesting_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, value in record.items():
        lowered = str(key).lower()
        if key == "epoch" or any(
            token in lowered
            for token in (
                "router",
                "route_",
                "frequency/route",
                "prior",
                "shallow",
                "monotonic",
                "curvature",
                "budget",
                "loss",
                "perf/lr",
                "phase",
                "progress",
            )
        ):
            scalar = _json_scalar(value)
            if scalar is not None:
                selected[str(key)] = scalar
    return selected


def read_metrics_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "NOT_WRITTEN",
            "record_count": 0,
            "epochs": [],
            "duplicate_epochs": [],
            "invalid_lines": [],
            "latest": None,
            "phase_observations": {},
        }
    records: list[tuple[int, int, dict[str, Any]]] = []
    invalid_lines: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("record is not an object")
            epoch = payload.get("epoch")
            if isinstance(epoch, bool) or not isinstance(epoch, int):
                raise ValueError("record lacks integer epoch")
        except Exception as exc:
            invalid_lines.append(
                {
                    "line": line_number,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        records.append((epoch, line_number, payload))
    counts = Counter(epoch for epoch, _, _ in records)
    duplicate_epochs = sorted(
        epoch for epoch, count in counts.items() if count > 1
    )
    by_epoch: dict[int, dict[str, Any]] = {}
    for epoch, _, payload in records:
        by_epoch[epoch] = payload
    epochs = sorted(by_epoch)
    latest = (
        {
            "epoch": epochs[-1],
            "phase": phase_for_epoch(epochs[-1]),
            "metrics": _interesting_metrics(by_epoch[epochs[-1]]),
        }
        if epochs
        else None
    )
    observations = {
        str(epoch): {
            "phase": phase_for_epoch(epoch),
            "metrics": _interesting_metrics(by_epoch[epoch]),
        }
        for epoch in PHASE_OBSERVATION_EPOCHS
        if epoch in by_epoch
    }
    return {
        "status": (
            "PASS"
            if not invalid_lines and not duplicate_epochs
            else "PARTIAL"
        ),
        "record_count": len(records),
        "epochs": epochs,
        "missing_epochs": [
            epoch for epoch in range(1, TOTAL_EPOCHS + 1)
            if epoch not in by_epoch
        ],
        "duplicate_epochs": duplicate_epochs,
        "invalid_lines": invalid_lines,
        "latest": latest,
        "phase_observations": observations,
    }


def inspect_checkpoints(
    checkpoint_dir: Path,
    *,
    config: Mapping[str, Any],
    root: Path = ROOT,
) -> dict[str, Any]:
    buckets: dict[int, list[Path]] = {}
    ignored: list[str] = []
    if checkpoint_dir.is_dir():
        for path in sorted(checkpoint_dir.glob("ckpt_epoch*.pt")):
            match = CHECKPOINT_PATTERN.fullmatch(path.name)
            if match is None:
                ignored.append(portable_path(path, root=root))
                continue
            buckets.setdefault(int(match.group(1)), []).append(path)
    records: list[dict[str, Any]] = []
    duplicate_epochs = sorted(
        epoch for epoch, paths in buckets.items() if len(paths) > 1
    )
    valid: dict[int, dict[str, Any]] = {}
    for epoch in sorted(buckets):
        paths = buckets[epoch]
        if len(paths) > 1:
            for path in paths:
                records.append(
                    {
                        "epoch_from_name": epoch,
                        "path": portable_path(path, root=root),
                        "status": "DUPLICATE",
                    }
                )
            continue
        path = paths[0]
        try:
            metadata = inspect_resume_checkpoint(
                path,
                expected_config=config,
            )
            record = {
                "epoch": int(metadata["epoch"]),
                "step": int(metadata["step"]),
                "phase": phase_for_epoch(int(metadata["epoch"])),
                "path": portable_path(path, root=root),
                "sha256": sha256_file(path),
                "status": "VALID",
            }
            valid[epoch] = record
            records.append(record)
        except Exception as exc:
            records.append(
                {
                    "epoch_from_name": epoch,
                    "path": portable_path(path, root=root),
                    "status": "INVALID",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    valid_epochs = sorted(valid)
    latest = valid[valid_epochs[-1]] if valid_epochs else None
    return {
        "checkpoint_dir": portable_path(checkpoint_dir, root=root),
        "records": records,
        "valid_epochs": valid_epochs,
        "invalid_count": sum(
            record["status"] == "INVALID" for record in records
        ),
        "duplicate_epochs": duplicate_epochs,
        "ignored_files": ignored,
        "latest": latest,
        "phase_observation_epochs": list(PHASE_OBSERVATION_EPOCHS),
        "missing_phase_checkpoints": [
            epoch for epoch in PHASE_OBSERVATION_EPOCHS
            if epoch not in valid
        ],
        "final_checkpoint_valid": TOTAL_EPOCHS in valid,
    }


def _prior_summary(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    metadata = config.get("prior_anchored_run", {})
    if not isinstance(metadata, Mapping):
        return {"status": "MISSING_RUN_METADATA"}
    path_value = metadata.get("prior_artifact_path")
    expected_hash = metadata.get("prior_artifact_sha256")
    if not isinstance(path_value, str):
        return {"status": "MISSING_PATH"}
    try:
        relative = normalize_repo_relative(path_value, root=root)
        path = resolve_repo_path(relative, root=root)
    except Exception as exc:
        return {
            "status": "INVALID_PATH",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not path.is_file():
        return {"status": "MISSING", "path": relative}
    observed_hash = sha256_file(path)
    payload: dict[str, Any] = {}
    try:
        payload = read_json(path)
    except Exception as exc:
        return {
            "status": "INVALID",
            "path": relative,
            "sha256": observed_hash,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "PASS" if observed_hash == expected_hash else "HASH_MISMATCH",
        "path": relative,
        "sha256": observed_hash,
        "expected_sha256": expected_hash,
        "pipeline_id": payload.get("pipeline_id"),
        "decision": payload.get("decision"),
        "preview_only": payload.get("preview_only"),
        "patients": payload.get("quality_control", {}).get("patients")
        if isinstance(payload.get("quality_control"), Mapping)
        else None,
        "crossings": payload.get("crossings"),
    }


def summarize_run(
    run_dir_value: str | Path,
    *,
    output: str | Path | None = None,
    root: Path = ROOT,
    write: bool = True,
) -> dict[str, Any]:
    run_relative = normalize_repo_relative(run_dir_value, root=root)
    run_dir = resolve_repo_path(run_relative, root=root)
    if not run_dir.is_dir():
        raise RunnerError(f"Run directory does not exist: {run_relative}")
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise RunnerError(f"Run lacks resolved_config.yaml: {run_relative}")
    config = read_yaml(config_path)
    run_metadata = config.get("prior_anchored_run", {})
    if (
        not isinstance(run_metadata, Mapping)
        or run_metadata.get("pipeline_id") != PIPELINE_ID
    ):
        raise RunnerError("Run config is not a prior-anchored 100e run")
    checkpoint_dir = _checkpoint_dir_from_config(config, root=root)
    checkpoints = inspect_checkpoints(
        checkpoint_dir,
        config=config,
        root=root,
    )
    runtime = config.get("runtime", {})
    metrics_value = (
        runtime.get("training_metrics_jsonl")
        if isinstance(runtime, Mapping)
        else None
    )
    if not isinstance(metrics_value, str):
        metrics_value = f"{run_relative}/training_metrics.jsonl"
    metrics_relative = normalize_repo_relative(metrics_value, root=root)
    metrics = read_metrics_jsonl(
        resolve_repo_path(metrics_relative, root=root)
    )
    state_path = run_dir / "state.json"
    state: dict[str, Any] | None
    state_error: str | None = None
    if state_path.is_file():
        try:
            state = read_json(state_path)
        except Exception as exc:
            state = None
            state_error = f"{type(exc).__name__}: {exc}"
    else:
        state = None
        state_error = "state.json is missing"
    final_valid = bool(checkpoints["final_checkpoint_valid"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": SUMMARY_PIPELINE_ID,
        "generated_at_utc": utc_now(),
        "decision": "COMPLETE" if final_valid else "INCOMPLETE",
        "scientific_claim_allowed": False,
        "reason": (
            "This artifact reports run completeness and router observability; "
            "it does not establish clinical, production, or causal validity."
        ),
        "run_dir": run_relative,
        "resolved_config": portable_path(config_path, root=root),
        "resolved_config_sha256": sha256_file(config_path),
        "configured_total_epochs": int(
            config.get("training", {}).get("num_epochs", -1)
        ),
        "runner_state": {
            "status": state.get("status") if state else None,
            "latest_epoch": state.get("latest_epoch") if state else None,
            "current_stage": state.get("current_stage") if state else None,
            "error": state_error,
        },
        "prior": _prior_summary(config, root=root),
        "checkpoints": checkpoints,
        "metrics_jsonl": {
            "path": metrics_relative,
            **metrics,
        },
        "phase_contract": [
            {
                "checkpoint_epoch": epoch,
                "phase": phase_for_epoch(epoch),
                "checkpoint_present": (
                    epoch in checkpoints["valid_epochs"]
                ),
                "metrics_present": str(epoch)
                in metrics["phase_observations"],
            }
            for epoch in PHASE_OBSERVATION_EPOCHS
        ],
    }
    if write:
        output_relative = normalize_repo_relative(
            output if output is not None else f"{run_relative}/summary.json",
            root=root,
        )
        output_path = resolve_repo_path(output_relative, root=root)
        write_json_atomic(output_path, summary)
    return summary


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = summarize_run(
            args.run_dir,
            output=args.output,
            write=not args.no_write,
        )
    except (RunnerError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
