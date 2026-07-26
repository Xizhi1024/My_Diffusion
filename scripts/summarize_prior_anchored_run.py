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
    validate_prior_artifact,
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


_METRIC_TOKENS = (
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
    # Validation / lesion observability required by the monitoring contract.
    "val/",
    "lesion",
    "uptake",
    "stripe",
    "native",
    "null",
)
# Trainer writes one record per epoch as
#   {"epoch": N, "step": S, "train": {...}, "eval": {...}, "validation": {...}}
# The summarizer must recurse into these sections; scanning only the top
# level would hide every router metric in a real run.
_METRIC_SECTIONS = ("train", "eval", "validation")
_CANONICAL_ROUTE_METRICS = frozenset(
    {
        "frequency/route_native_mass",
        "frequency/route_shallow_mass",
        "frequency/route_null_mass",
    }
)


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

    def _scan(key: str, value: Any, *, prefix: str = "") -> None:
        lowered = str(key).lower()
        if key == "epoch" and not prefix:
            scalar = _json_scalar(value)
            if scalar is not None:
                selected["epoch"] = scalar
            return
        if key == "step" and not prefix:
            scalar = _json_scalar(value)
            if scalar is not None:
                selected["step"] = scalar
            return
        if any(token in lowered for token in _METRIC_TOKENS):
            scalar = _json_scalar(value)
            if scalar is not None:
                selected[f"{prefix}{key}"] = scalar

    for key, value in record.items():
        if key in _METRIC_SECTIONS and isinstance(value, Mapping):
            for inner_key, inner_value in value.items():
                _scan(inner_key, inner_value, prefix=f"{key}/")
            continue
        _scan(key, value)
    return selected


def _metric_section_contract_errors(
    section: Any,
    *,
    label: str,
    allow_none: bool,
) -> list[str]:
    if section is None and allow_none:
        return []
    if not isinstance(section, Mapping) or not section:
        return [f"{label} must be a non-empty metrics object"]
    errors: list[str] = []
    for key, value in section.items():
        if not isinstance(key, str) or not key:
            errors.append(f"{label} contains an invalid metric name")
            continue
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(
                f"{label}.{key} must be a finite number or null"
            )
    return errors


def read_metrics_jsonl(
    path: Path,
    *,
    expected_total_epochs: int = TOTAL_EPOCHS,
) -> dict[str, Any]:
    expected_epochs = list(range(1, expected_total_epochs + 1))
    if not path.is_file():
        return {
            "status": "NOT_WRITTEN",
            "record_count": 0,
            "line_count": 0,
            "epochs": [],
            "expected_total_epochs": expected_total_epochs,
            "missing_epochs": expected_epochs,
            "out_of_range_epochs": [],
            "duplicate_epochs": [],
            "invalid_lines": [],
            "contract_errors": [],
            "epoch_sequence_valid": False,
            "final_step": None,
            "latest": None,
            "phase_observations": {},
        }
    records: list[tuple[int, int, dict[str, Any]]] = []
    invalid_lines: list[dict[str, Any]] = []
    contract_errors: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    valid_contract_lines: set[int] = set()
    previous_step: int | None = None
    for line_number, raw in enumerate(raw_lines, start=1):
        if not raw.strip():
            invalid_lines.append(
                {
                    "line": line_number,
                    "error": "ValueError: blank record",
                }
            )
            continue
        try:
            payload = json.loads(
                raw,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON value {value}")
                ),
            )
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

        row_errors: list[str] = []
        if payload.get("schema_version") != 1:
            row_errors.append("schema_version must be 1")
        step = payload.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step < 0
        ):
            row_errors.append("step must be a non-negative integer")
        else:
            if previous_step is not None and step <= previous_step:
                row_errors.append(
                    "step must increase strictly; "
                    f"found {step} after {previous_step}"
                )
            previous_step = step
        expected_phase = phase_for_epoch(epoch)
        if payload.get("phase") != expected_phase:
            row_errors.append(
                f"phase must be {expected_phase!r} for epoch {epoch}"
            )
        train = payload.get("train")
        row_errors.extend(
            _metric_section_contract_errors(
                train,
                label="train",
                allow_none=False,
            )
        )
        row_errors.extend(
            _metric_section_contract_errors(
                payload.get("eval"),
                label="eval",
                allow_none=True,
            )
        )
        row_errors.extend(
            _metric_section_contract_errors(
                payload.get("validation"),
                label="validation",
                allow_none=True,
            )
        )
        if epoch % 10 == 0 and (
            not isinstance(payload.get("eval"), Mapping)
            or not payload["eval"]
            or not isinstance(payload.get("validation"), Mapping)
            or not payload["validation"]
        ):
            row_errors.append(
                f"epoch {epoch} lacks its configured validation observation"
            )
        if (
            epoch in PHASE_OBSERVATION_EPOCHS
            and isinstance(train, Mapping)
            and (
                not _CANONICAL_ROUTE_METRICS.issubset(train)
                or any(
                    train[key] is None
                    for key in _CANONICAL_ROUTE_METRICS
                    if key in train
                )
            )
        ):
            row_errors.append(
                f"epoch {epoch} lacks canonical route-mass observations"
            )
        if row_errors:
            contract_errors.append(
                {
                    "line": line_number,
                    "epoch": epoch,
                    "errors": row_errors,
                }
            )
        else:
            valid_contract_lines.add(line_number)

    counts = Counter(epoch for epoch, _, _ in records)
    duplicate_epochs = sorted(
        epoch for epoch, count in counts.items() if count > 1
    )
    out_of_range_epochs = sorted(
        {
            epoch
            for epoch, _, _ in records
            if not 1 <= epoch <= expected_total_epochs
        }
    )
    by_epoch: dict[int, dict[str, Any]] = {}
    valid_by_epoch: dict[int, dict[str, Any]] = {}
    for epoch, line_number, payload in records:
        # A duplicate epoch is a data-integrity issue, not an overwrite.
        by_epoch.setdefault(epoch, payload)
        if (
            line_number in valid_contract_lines
            and 1 <= epoch <= expected_total_epochs
        ):
            valid_by_epoch.setdefault(epoch, payload)
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
            "metrics": _interesting_metrics(valid_by_epoch[epoch]),
        }
        for epoch in PHASE_OBSERVATION_EPOCHS
        if epoch in valid_by_epoch
    }
    missing_epochs = [
        epoch
        for epoch in range(1, expected_total_epochs + 1)
        if epoch not in by_epoch
    ]
    observed_sequence = [epoch for epoch, _, _ in records]
    epoch_sequence_valid = observed_sequence == expected_epochs
    final_payload = by_epoch.get(expected_total_epochs)
    final_step = (
        final_payload.get("step")
        if isinstance(final_payload, Mapping)
        and not isinstance(final_payload.get("step"), bool)
        and isinstance(final_payload.get("step"), int)
        else None
    )
    # PASS means exactly one well-formed record for every epoch, in order.
    clean = (
        not invalid_lines
        and not contract_errors
        and not duplicate_epochs
        and not out_of_range_epochs
        and not missing_epochs
        and len(records) == expected_total_epochs
        and epoch_sequence_valid
    )
    return {
        "status": "PASS" if clean else "PARTIAL",
        "record_count": len(records),
        "line_count": len(raw_lines),
        "epochs": epochs,
        "expected_total_epochs": expected_total_epochs,
        "missing_epochs": missing_epochs,
        "out_of_range_epochs": out_of_range_epochs,
        "duplicate_epochs": duplicate_epochs,
        "invalid_lines": invalid_lines,
        "contract_errors": contract_errors,
        "epoch_sequence_valid": epoch_sequence_valid,
        "final_step": final_step,
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
                root=root,
                verify_current_lineage=True,
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
    router = config.get("modules", {})
    if isinstance(router, Mapping):
        router = router.get("residual_frequency", {})
    if isinstance(router, Mapping):
        router = router.get("cross_level_router", {})
    if not isinstance(router, Mapping):
        router = {}
    preview_contract = {
        "policy": router.get("policy") == "prior_anchored_learned",
        "schedule_path": router.get("h3_schedule_path") == relative,
        "schedule_sha256": (
            isinstance(expected_hash, str)
            and router.get("h3_schedule_sha256") == expected_hash
        ),
        "schedule_source": (
            router.get("h3_schedule_source") == "direct_png_preview"
        ),
        "repository_root": router.get("h3_repository_root") == ".",
        "explicit_preview_opt_in": (
            router.get("h3_allow_unverified_preview_lineage") is True
        ),
        "run_scope": (
            metadata.get("scope")
            == "exploratory_direct_png_prior_anchor"
        ),
        "path_policy": (
            metadata.get("path_policy")
            == "repository_relative_posix_only"
        ),
        "preview_not_local_training": (
            metadata.get("local_training_performed") is False
        ),
        "cloud_runtime_required": (
            metadata.get("cloud_runtime_required") is True
        ),
    }
    payload: dict[str, Any]
    try:
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
        ):
            raise RunnerError(
                "Run config lacks a valid prior artifact SHA-256"
            )
        payload = validate_prior_artifact(
            path,
            expected_sha256=expected_hash,
            root=root,
            strict_runtime=True,
        )
        if not all(preview_contract.values()):
            failed = [
                name
                for name, passed in preview_contract.items()
                if not passed
            ]
            raise RunnerError(
                "Run config violates preview prior contract: "
                + ", ".join(failed)
            )
    except Exception as exc:
        return {
            "status": "INVALID",
            "path": relative,
            "sha256": observed_hash,
            "expected_sha256": expected_hash,
            "preview_contract": preview_contract,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "PASS",
        "path": relative,
        "sha256": observed_hash,
        "expected_sha256": expected_hash,
        "preview_contract": preview_contract,
        "pipeline_id": payload.get("pipeline_id"),
        "decision": payload.get("decision"),
        "preview_only": payload.get("preview_only"),
        "patients": payload.get("quality_control", {}).get("patients")
        if isinstance(payload.get("quality_control"), Mapping)
        else None,
        "crossings": payload.get("crossings"),
    }


def _runner_state_summary(
    state: Mapping[str, Any] | None,
    *,
    state_error: str | None,
    config_path: Path,
    prior: Mapping[str, Any],
    metrics_path: str,
    metrics_sha256: str | None,
    root: Path,
) -> dict[str, Any]:
    expected_config_path = portable_path(config_path, root=root)
    expected_config_hash = sha256_file(config_path)
    expected_prior_path = prior.get("path")
    expected_prior_hash = prior.get("sha256")
    state_is_complete = (
        state is not None and state.get("status") == "COMPLETE"
    )
    metrics_path_present = (
        state is not None and "training_metrics" in state
    )
    metrics_hash_present = (
        state is not None and "training_metrics_sha256" in state
    )
    metrics_fields_present = metrics_path_present and metrics_hash_present
    metrics_fields_absent = (
        not metrics_path_present and not metrics_hash_present
    )
    metrics_fields_allowed = (
        metrics_fields_present
        or (not state_is_complete and metrics_fields_absent)
    )
    checks = {
        "state_readable": state is not None and state_error is None,
        "status_complete": state_is_complete,
        "latest_epoch_100": (
            state is not None
            and not isinstance(state.get("latest_epoch"), bool)
            and state.get("latest_epoch") == TOTAL_EPOCHS
        ),
        "current_stage_empty": (
            state is not None
            and "current_stage" in state
            and state.get("current_stage") is None
        ),
        "resolved_config_path_matches": (
            state is not None
            and state.get("resolved_config") == expected_config_path
        ),
        "resolved_config_sha256_matches": (
            state is not None
            and state.get("resolved_config_sha256")
            == expected_config_hash
        ),
        "prior_artifact_path_matches": (
            state is not None
            and state.get("prior_artifact") == expected_prior_path
        ),
        "prior_artifact_sha256_matches": (
            state is not None
            and isinstance(expected_prior_hash, str)
            and state.get("prior_artifact_sha256")
            == expected_prior_hash
        ),
        "training_metrics_fields_complete": metrics_fields_allowed,
        "training_metrics_path_matches": (
            (
                state is not None
                and state.get("training_metrics") == metrics_path
            )
            if metrics_fields_present
            else not state_is_complete
        ),
        "training_metrics_sha256_matches": (
            (
                state is not None
                and isinstance(metrics_sha256, str)
                and state.get("training_metrics_sha256")
                == metrics_sha256
            )
            if metrics_fields_present
            else not state_is_complete
        ),
    }
    return {
        "status": state.get("status") if state else None,
        "latest_epoch": state.get("latest_epoch") if state else None,
        "current_stage": state.get("current_stage") if state else None,
        "error": state_error,
        "contract_valid": all(checks.values()),
        "checks": checks,
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
    metrics_path = resolve_repo_path(metrics_relative, root=root)
    metrics = read_metrics_jsonl(
        metrics_path,
        expected_total_epochs=TOTAL_EPOCHS,
    )
    metrics_sha256 = (
        sha256_file(metrics_path) if metrics_path.is_file() else None
    )
    state_path = run_dir / "state.json"
    state: Mapping[str, Any] | None
    state_error: str | None = None
    if state_path.is_file():
        try:
            state_payload = read_json(state_path)
            if not isinstance(state_payload, Mapping):
                raise ValueError("state.json root is not an object")
            state = state_payload
        except Exception as exc:
            state = None
            state_error = f"{type(exc).__name__}: {exc}"
    else:
        state = None
        state_error = "state.json is missing"

    training = config.get("training", {})
    configured_total_epochs = (
        training.get("num_epochs")
        if isinstance(training, Mapping)
        else None
    )
    run_total_epochs = run_metadata.get("total_epochs")
    configuration_is_100_epochs = (
        not isinstance(configured_total_epochs, bool)
        and configured_total_epochs == TOTAL_EPOCHS
        and not isinstance(run_total_epochs, bool)
        and run_total_epochs == TOTAL_EPOCHS
    )
    prior = _prior_summary(config, root=root)
    runner_state = _runner_state_summary(
        state,
        state_error=state_error,
        config_path=config_path,
        prior=prior,
        metrics_path=metrics_relative,
        metrics_sha256=metrics_sha256,
        root=root,
    )
    required_observations = {
        str(epoch) for epoch in PHASE_OBSERVATION_EPOCHS
    }
    observed_phase_epochs = set(metrics["phase_observations"])
    phase_observations_complete = (
        observed_phase_epochs == required_observations
        and all(
            metrics["phase_observations"][str(epoch)]["phase"]
            == phase_for_epoch(epoch)
            for epoch in PHASE_OBSERVATION_EPOCHS
        )
    )
    final_checkpoint_step = (
        checkpoints["latest"].get("step")
        if checkpoints["final_checkpoint_valid"]
        and isinstance(checkpoints.get("latest"), Mapping)
        and checkpoints["latest"].get("epoch") == TOTAL_EPOCHS
        else None
    )
    final_step_matches = (
        metrics["status"] == "PASS"
        and not isinstance(metrics.get("final_step"), bool)
        and isinstance(metrics.get("final_step"), int)
        and not isinstance(final_checkpoint_step, bool)
        and isinstance(final_checkpoint_step, int)
        and metrics["final_step"] == final_checkpoint_step
    )
    completion_checks = {
        "configuration_is_100_epochs": configuration_is_100_epochs,
        "runner_state_contract_valid": bool(
            runner_state["contract_valid"]
        ),
        "prior_artifact_contract_valid": prior.get("status") == "PASS",
        "final_checkpoint_valid": bool(
            checkpoints["final_checkpoint_valid"]
        ),
        "metrics_exactly_epochs_1_through_100": (
            metrics["status"] == "PASS"
        ),
        "metrics_final_step_matches_checkpoint": final_step_matches,
        "phase_observations_complete": phase_observations_complete,
    }
    complete = all(completion_checks.values())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": SUMMARY_PIPELINE_ID,
        "generated_at_utc": utc_now(),
        "decision": "COMPLETE" if complete else "INCOMPLETE",
        "scientific_claim_allowed": False,
        "reason": (
            "This artifact reports run completeness and router observability; "
            "it does not establish clinical, production, or causal validity."
        ),
        "run_dir": run_relative,
        "resolved_config": portable_path(config_path, root=root),
        "resolved_config_sha256": sha256_file(config_path),
        "configured_total_epochs": configured_total_epochs,
        "run_metadata_total_epochs": run_total_epochs,
        "completion_checks": completion_checks,
        "runner_state": runner_state,
        "prior": prior,
        "checkpoints": checkpoints,
        "metrics_jsonl": {
            "path": metrics_relative,
            "sha256": metrics_sha256,
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
