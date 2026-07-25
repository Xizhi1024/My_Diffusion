#!/usr/bin/env python3
"""Run the 100-epoch prior-anchored router experiment on the cloud.

The local-safe entry point is ``--preflight-only``.  It validates the static
contract without requiring CUDA, the cloud tensor cache, raw PNGs, or a mean
checkpoint.  A real launch selects a pathology-excluded mean checkpoint,
re-estimates the direct-PNG prior inside the run directory, writes a run-owned
resolved YAML, then invokes ``scripts/train_v2.py``.

All persisted project paths are repository-relative POSIX paths.  Absolute
paths are used only transiently for filesystem operations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION = 1
PIPELINE_ID = "PRIOR_ANCHORED_ROUTER_100E_CLOUD_V1"
TOTAL_EPOCHS = 100
PHASE_OBSERVATION_EPOCHS = (10, 20, 30, 40, 100)
DEFAULT_CONFIG = Path(
    "configs/experiments/slmf_png_prior_anchored_router_100e.yaml"
)
DEFAULT_RUNS_ROOT = Path("results/prior_anchored_router_100e/runs")
DEFAULT_MEAN_CANDIDATES = (
    Path("checkpoints/freq_mean_excluded_v1/mean_best.pt"),
    Path("checkpoints/freq_mean_pretrain_v2/mean_best.pt"),
)
PRIOR_FILENAME = "h3_prior_preview.json"
CHECKPOINT_PATTERN = re.compile(r"^ckpt_epoch(\d+)\.pt$")
COMPLETED_STATUSES = frozenset({"COMPLETE", "COMPLETED"})


class RunnerError(RuntimeError):
    """A fail-closed runner contract violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_text(value: str | os.PathLike[str]) -> str:
    text = os.fspath(value).strip().replace("\\", "/")
    if not text:
        raise ValueError("Repository-relative path must not be empty")
    if "\x00" in text:
        raise ValueError("Repository-relative path contains NUL")
    return text


def normalize_repo_relative(
    value: str | os.PathLike[str],
    *,
    root: Path = ROOT,
) -> str:
    """Return a normalized POSIX path and reject absolute/escaping paths."""

    text = _path_text(value)
    posix = PurePosixPath(text)
    windows = PureWindowsPath(text)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"Absolute project path is forbidden: {text}")
    if any(part == ".." for part in posix.parts):
        raise ValueError(f"Project path may not escape the repository: {text}")
    normalized = posix.as_posix()
    resolved = (root.resolve() / Path(*posix.parts)).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Project path may not escape the repository: {text}"
        ) from exc
    return normalized


def resolve_repo_path(
    value: str | os.PathLike[str],
    *,
    root: Path = ROOT,
) -> Path:
    relative = normalize_repo_relative(value, root=root)
    return (root.resolve() / Path(*PurePosixPath(relative).parts)).resolve()


def portable_path(path: Path, *, root: Path = ROOT) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Only paths under the repository may be persisted: {path}"
        ) from exc
    return PurePosixPath(*relative.parts).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise RunnerError(f"Expected a JSON object: {path}")
    return payload


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunnerError(f"Expected a YAML mapping: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        yaml.safe_dump(
            dict(payload),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mapping_at(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise RunnerError(f"Missing configuration mapping: {'.'.join(keys)}")
        current = current[key]
    if not isinstance(current, Mapping):
        raise RunnerError(f"Configuration value is not a mapping: {'.'.join(keys)}")
    return current


def _assert_config_path(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    root: Path,
    allow_null: bool = False,
) -> None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise RunnerError(f"Missing path field: {'.'.join(keys)}")
        current = current[key]
    if current is None and allow_null:
        return
    if not isinstance(current, str):
        raise RunnerError(f"Path field must be a string: {'.'.join(keys)}")
    normalize_repo_relative(current, root=root)


def validate_base_config(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    training = _mapping_at(config, "training")
    runtime = _mapping_at(config, "runtime")
    router = _mapping_at(
        config,
        "modules",
        "residual_frequency",
        "cross_level_router",
    )
    regularizer = _mapping_at(
        config,
        "losses",
        "spectral_router_regularization",
    )
    if int(training.get("num_epochs", -1)) != TOTAL_EPOCHS:
        raise RunnerError(f"training.num_epochs must be {TOTAL_EPOCHS}")
    if training.get("init_from") is not None:
        raise RunnerError("The cloud template must set training.init_from=null")
    if training.get("resume_from") is not None:
        raise RunnerError("The cloud template must set training.resume_from=null")
    if runtime.get("require_cuda") is not True:
        raise RunnerError("runtime.require_cuda must be true")
    if runtime.get("early_stopping", {}).get("enabled") is not False:
        raise RunnerError("Early stopping must remain disabled for the 100e budget")
    for key in ("eval_interval", "save_interval", "sample_interval"):
        if int(runtime.get(key, -1)) != 10:
            raise RunnerError(f"runtime.{key} must be 10")
    if router.get("policy") != "prior_anchored_learned":
        raise RunnerError(
            "cross_level_router.policy must be prior_anchored_learned"
        )
    expected_phase = {
        "prior_warmup_epochs": 10,
        "prior_active_ramp_epochs": 10,
        "prior_destination_warmup_epochs": 30,
        "prior_destination_ramp_epochs": 10,
        "prior_anchor_decay_end_epoch": 100,
    }
    observed_phase = {key: int(router.get(key, -1)) for key in expected_phase}
    if observed_phase != expected_phase:
        raise RunnerError(
            f"Router phase schedule mismatch: {observed_phase}"
        )
    observed_epochs = tuple(
        int(value) for value in router.get("phase_observation_epochs", ())
    )
    if observed_epochs != PHASE_OBSERVATION_EPOCHS:
        raise RunnerError(
            "phase_observation_epochs must be 10/20/30/40/100"
        )
    if router.get("h3_schedule_source") != "direct_png_preview":
        raise RunnerError(
            "h3_schedule_source must be direct_png_preview for this runner"
        )
    if router.get("h3_repository_root") != ".":
        raise RunnerError("h3_repository_root must be the relative path '.'")
    if router.get("h3_allow_unverified_preview_lineage") is not True:
        raise RunnerError(
            "Direct-PNG preview lineage must be explicitly acknowledged"
        )
    if float(regularizer.get("active_mass_weight", -1.0)) != 0.0:
        raise RunnerError(
            "Legacy active_mass_weight must be zero for a prior-anchored router"
        )
    if float(regularizer.get("artifact_safety_weight", -1.0)) != 0.0:
        raise RunnerError(
            "artifact_safety_weight must remain zero until independently validated"
        )
    path_fields = (
        ("data", "cache_dir"),
        ("data", "cache_lineage"),
        ("data", "dataset_contract"),
        ("data", "split_manifest"),
        ("modules", "conditional_mean", "checkpoint"),
        (
            "modules",
            "residual_frequency",
            "cross_level_router",
            "h3_schedule_path",
        ),
    )
    for keys in path_fields:
        _assert_config_path(
            config,
            keys,
            root=root,
            allow_null=keys[-1] == "h3_schedule_path",
        )
    return {
        "num_epochs": TOTAL_EPOCHS,
        "phase_observation_epochs": list(PHASE_OBSERVATION_EPOCHS),
        "policy": router["policy"],
        "relative_path_contract": "PASS",
    }


def build_static_preflight(
    *,
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG,
    runs_root: str | os.PathLike[str] = DEFAULT_RUNS_ROOT,
    png_root: str | os.PathLike[str] | None = None,
    manifest: str | os.PathLike[str] | None = None,
    dataset_contract: str | os.PathLike[str] | None = None,
    mean_checkpoint: str | os.PathLike[str] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    config_relative = normalize_repo_relative(config_path, root=root)
    runs_relative = normalize_repo_relative(runs_root, root=root)
    config = read_yaml(resolve_repo_path(config_relative, root=root))
    contract = validate_base_config(config, root=root)
    optional_paths = {
        "png_root": png_root,
        "manifest": manifest,
        "dataset_contract": dataset_contract,
        "mean_checkpoint": mean_checkpoint,
    }
    normalized_optional = {
        key: (
            normalize_repo_relative(value, root=root)
            if value is not None
            else None
        )
        for key, value in optional_paths.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "decision": "STATIC_PREFLIGHT_PASS",
        "cloud_runtime_started": False,
        "cloud_files_required_for_this_check": False,
        "config": config_relative,
        "runs_root": runs_relative,
        "overrides": normalized_optional,
        "mean_checkpoint_candidates": [
            normalize_repo_relative(path, root=root)
            for path in DEFAULT_MEAN_CANDIDATES
        ],
        "contract": contract,
        "next_action": (
            "Run without --preflight-only in the cloud repository; CUDA, raw "
            "PNGs, the cloud cache lineage, and a pathology-excluded mean "
            "checkpoint are checked there."
        ),
    }


def _checkpoint_is_pathology_excluded(path: Path) -> bool:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        return False
    mean_config = payload.get("mean_config", {})
    if not isinstance(mean_config, Mapping):
        return False
    if mean_config.get("pathology_policy") == "excluded":
        return True
    exclusion = mean_config.get("pathology_exclusion", {})
    return (
        isinstance(exclusion, Mapping)
        and exclusion.get("enabled") is True
    )


def select_mean_checkpoint(
    *,
    root: Path = ROOT,
    explicit: str | os.PathLike[str] | None = None,
    candidates: Sequence[str | os.PathLike[str]] = DEFAULT_MEAN_CANDIDATES,
) -> str:
    requested = (explicit,) if explicit is not None else tuple(candidates)
    failures: list[str] = []
    for value in requested:
        relative = normalize_repo_relative(value, root=root)
        path = resolve_repo_path(relative, root=root)
        if not path.is_file():
            failures.append(f"{relative}: missing")
            continue
        try:
            excluded = _checkpoint_is_pathology_excluded(path)
        except Exception as exc:
            failures.append(f"{relative}: unreadable ({type(exc).__name__})")
            continue
        if not excluded:
            failures.append(f"{relative}: not pathology-excluded")
            continue
        return relative
    raise RunnerError(
        "No usable pathology-excluded mean checkpoint was found: "
        + "; ".join(failures)
    )


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RunnerError(
            "CUDA is unavailable; real training is cloud-only. "
            "Use --preflight-only for local static validation."
        )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S.%fZ")


def _state_path(run_dir: Path) -> Path:
    return run_dir / "state.json"


def _config_path(run_dir: Path) -> Path:
    return run_dir / "resolved_config.yaml"


def _checkpoint_dir_from_config(
    config: Mapping[str, Any],
    *,
    root: Path,
) -> Path:
    checkpoint_dir = _mapping_at(config, "training").get("checkpoint_dir")
    if not isinstance(checkpoint_dir, str):
        raise RunnerError("Run config is missing training.checkpoint_dir")
    return resolve_repo_path(checkpoint_dir, root=root)


def update_state(
    run_dir: Path,
    *,
    root: Path = ROOT,
    **changes: Any,
) -> dict[str, Any]:
    path = _state_path(run_dir)
    if path.is_file():
        state = read_json(path)
    else:
        state = {
            "schema_version": SCHEMA_VERSION,
            "pipeline_id": PIPELINE_ID,
            "run_dir": portable_path(run_dir, root=root),
            "created_at_utc": utc_now(),
        }
    state.update(changes)
    state["updated_at_utc"] = utc_now()
    write_json_atomic(path, state)
    return state


def _run_status(run_dir: Path) -> str:
    state_path = _state_path(run_dir)
    if not state_path.is_file():
        return "UNKNOWN"
    try:
        return str(read_json(state_path).get("status", "UNKNOWN")).upper()
    except Exception:
        return "INVALID"


def select_latest_resumable_run(
    runs_root: str | os.PathLike[str] = DEFAULT_RUNS_ROOT,
    *,
    root: Path = ROOT,
) -> str:
    relative_root = normalize_repo_relative(runs_root, root=root)
    directory = resolve_repo_path(relative_root, root=root)
    if not directory.is_dir():
        raise RunnerError(f"Runs root does not exist: {relative_root}")
    candidates = [
        item
        for item in directory.iterdir()
        if item.is_dir()
        and _config_path(item).is_file()
        and _run_status(item) not in COMPLETED_STATUSES
    ]
    if not candidates:
        raise RunnerError(f"No incomplete run is available below {relative_root}")
    candidates.sort(
        key=lambda item: (
            _state_path(item).stat().st_mtime
            if _state_path(item).is_file()
            else item.stat().st_mtime,
            item.name,
        ),
        reverse=True,
    )
    return portable_path(candidates[0], root=root)


def prepare_fresh_run(
    run_relative: str | os.PathLike[str],
    *,
    root: Path = ROOT,
) -> Path:
    relative = normalize_repo_relative(run_relative, root=root)
    run_dir = resolve_repo_path(relative, root=root)
    if run_dir.exists():
        checkpoints = list(run_dir.rglob("*.pt")) if run_dir.is_dir() else []
        detail = (
            " Existing checkpoints were found; use --resume with the exact "
            "run directory or --resume-latest."
            if checkpoints
            else ""
        )
        raise RunnerError(f"Fresh run directory already exists: {relative}.{detail}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _logical_python_command(arguments: Sequence[str]) -> list[str]:
    return ["python", *arguments]


def execute_logged(
    arguments: Sequence[str],
    *,
    log_path: Path,
    root: Path = ROOT,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logical = _logical_python_command(arguments)
    print("+ " + subprocess.list2cmdline(logical), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RunnerError(
            f"Command failed with exit code {completed.returncode}; "
            f"see {portable_path(log_path, root=root)}"
        )


def validate_prior_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
    root: Path = ROOT,
    strict_runtime: bool = False,
) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("pipeline_id") != "H3_DIRECT_PNG_PRIOR_PREVIEW_V1":
        raise RunnerError("Direct-PNG prior pipeline_id mismatch")
    if payload.get("decision") != "PREVIEW_ONLY":
        raise RunnerError("Direct-PNG prior must retain PREVIEW_ONLY provenance")
    curves = payload.get("preview_native_active_mass")
    if not isinstance(curves, Mapping) or len(curves) != 6:
        raise RunnerError("Direct-PNG prior must contain six band curves")
    lengths = {
        len(values)
        for values in curves.values()
        if isinstance(values, list)
    }
    if lengths != {1000} or len(lengths) != 1:
        raise RunnerError("Every direct-PNG prior curve must contain 1000 values")
    if strict_runtime:
        from src.model.frequency.prior_anchor_schedule import (
            load_prior_anchor_schedule,
        )

        observed_sha256 = sha256_file(path)
        if expected_sha256 is not None and observed_sha256 != expected_sha256:
            raise RunnerError("Direct-PNG prior SHA-256 mismatch")
        load_prior_anchor_schedule(
            portable_path(path, root=root),
            schedule_source="direct_png_preview",
            expected_file_sha256=observed_sha256,
            expected_num_train_timesteps=1000,
            repository_root=root,
            allow_unverified_preview_lineage=True,
        )
    return payload


def estimate_cloud_prior(
    *,
    run_dir: Path,
    mean_checkpoint: str,
    manifest: str,
    dataset_contract: str,
    png_root: str | None,
    device: str,
    root: Path = ROOT,
) -> tuple[str, str, list[str]]:
    output_relative = portable_path(run_dir / "prior", root=root)
    arguments = [
        "-u",
        "scripts/estimate_h3_prior_from_png.py",
        "--root",
        ".",
        "--manifest",
        manifest,
        "--dataset-contract",
        dataset_contract,
        "--mean-checkpoint",
        mean_checkpoint,
        "--output-dir",
        output_relative,
        "--device",
        device,
    ]
    if png_root is not None:
        arguments.extend(["--png-root", png_root])
    execute_logged(
        arguments,
        log_path=run_dir / "logs" / "prior_estimation.log",
        root=root,
    )
    prior_path = run_dir / "prior" / PRIOR_FILENAME
    if not prior_path.is_file():
        raise RunnerError(
            f"Prior estimation did not produce {portable_path(prior_path, root=root)}"
        )
    prior_sha256 = sha256_file(prior_path)
    validate_prior_artifact(
        prior_path,
        expected_sha256=prior_sha256,
        root=root,
        strict_runtime=True,
    )
    return (
        portable_path(prior_path, root=root),
        prior_sha256,
        _logical_python_command(arguments),
    )


def materialize_run_config(
    base_config: Mapping[str, Any],
    *,
    run_dir: Path,
    mean_checkpoint: str,
    prior_path: str,
    prior_sha256: str,
    manifest: str | None = None,
    dataset_contract: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    run_relative = portable_path(run_dir, root=root)
    run_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_dir.name)
    config.setdefault("experiment", {})["name"] = (
        f"prior_anchored_router_100e_{run_slug}"
    )
    data = config.setdefault("data", {})
    if manifest is not None:
        data["split_manifest"] = normalize_repo_relative(manifest, root=root)
    if dataset_contract is not None:
        data["dataset_contract"] = normalize_repo_relative(
            dataset_contract,
            root=root,
        )
    data["require_cache_lineage"] = True
    data["use_fake_data"] = False
    training = config.setdefault("training", {})
    training["num_epochs"] = TOTAL_EPOCHS
    training["checkpoint_dir"] = f"{run_relative}/checkpoints"
    training["init_from"] = None
    training["resume_from"] = None
    runtime = config.setdefault("runtime", {})
    runtime["require_cuda"] = True
    runtime["sample_dir"] = f"{run_relative}/samples"
    runtime["training_metrics_jsonl"] = f"{run_relative}/training_metrics.jsonl"
    config["modules"]["conditional_mean"]["checkpoint"] = (
        normalize_repo_relative(mean_checkpoint, root=root)
    )
    router = config["modules"]["residual_frequency"]["cross_level_router"]
    router["policy"] = "prior_anchored_learned"
    router["h3_schedule_path"] = normalize_repo_relative(prior_path, root=root)
    router["h3_schedule_sha256"] = str(prior_sha256)
    router["h3_schedule_source"] = "direct_png_preview"
    router["h3_repository_root"] = "."
    router["h3_allow_unverified_preview_lineage"] = True
    config["prior_anchored_run"] = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "scope": "exploratory_direct_png_prior_anchor",
        "run_dir": run_relative,
        "prior_artifact_path": normalize_repo_relative(prior_path, root=root),
        "prior_artifact_sha256": str(prior_sha256),
        "mean_checkpoint": normalize_repo_relative(
            mean_checkpoint,
            root=root,
        ),
        "phase_observation_epochs": list(PHASE_OBSERVATION_EPOCHS),
        "total_epochs": TOTAL_EPOCHS,
        "path_policy": "repository_relative_posix_only",
        "local_training_performed": False,
        "cloud_runtime_required": True,
    }
    validate_base_config(config, root=root)
    for value in (
        training["checkpoint_dir"],
        runtime["sample_dir"],
        runtime["training_metrics_jsonl"],
        router["h3_schedule_path"],
        config["prior_anchored_run"]["run_dir"],
        config["prior_anchored_run"]["prior_artifact_path"],
        config["prior_anchored_run"]["mean_checkpoint"],
    ):
        normalize_repo_relative(value, root=root)
    return config


def _checkpoint_epoch_from_name(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def inspect_resume_checkpoint(
    path: Path,
    *,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    name_epoch = _checkpoint_epoch_from_name(path)
    if name_epoch is None:
        raise RunnerError(f"Not a numbered epoch checkpoint: {path.name}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RunnerError(
            f"Latest checkpoint is unreadable and will not be skipped: {path.name}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RunnerError(f"Checkpoint is not a mapping: {path.name}")
    payload_epoch = payload.get("epoch")
    if isinstance(payload_epoch, bool) or not isinstance(payload_epoch, int):
        raise RunnerError(f"Checkpoint lacks integer epoch metadata: {path.name}")
    if payload_epoch != name_epoch:
        raise RunnerError(
            f"Checkpoint filename epoch {name_epoch} != metadata {payload_epoch}"
        )
    if not 0 < payload_epoch <= TOTAL_EPOCHS:
        raise RunnerError(f"Checkpoint epoch is outside 1..{TOTAL_EPOCHS}")
    required = ("model", "optimizer", "scheduler", "ema", "step", "config")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RunnerError(
            f"Checkpoint cannot resume; missing state: {', '.join(missing)}"
        )
    checkpoint_config = payload["config"]
    if not isinstance(checkpoint_config, Mapping):
        raise RunnerError("Checkpoint config is not a mapping")
    expected_name = _mapping_at(expected_config, "experiment").get("name")
    observed_name = _mapping_at(checkpoint_config, "experiment").get("name")
    if observed_name != expected_name:
        raise RunnerError("Checkpoint experiment identity mismatch")
    observed_epochs = int(
        _mapping_at(checkpoint_config, "training").get("num_epochs", -1)
    )
    if observed_epochs != TOTAL_EPOCHS:
        raise RunnerError("Checkpoint was not created with a 100-epoch budget")
    expected_router = _mapping_at(
        expected_config,
        "modules",
        "residual_frequency",
        "cross_level_router",
    )
    observed_router = _mapping_at(
        checkpoint_config,
        "modules",
        "residual_frequency",
        "cross_level_router",
    )
    for key in (
        "policy",
        "h3_schedule_path",
        "h3_schedule_sha256",
        "h3_schedule_source",
        "h3_repository_root",
        "h3_allow_unverified_preview_lineage",
    ):
        if observed_router.get(key) != expected_router.get(key):
            raise RunnerError(f"Checkpoint router contract mismatch: {key}")
    if bool(_mapping_at(expected_config, "data").get("require_cache_lineage")):
        if not isinstance(payload.get("data_lineage"), Mapping):
            raise RunnerError("Strict run checkpoint is missing data_lineage")
    return {
        "epoch": payload_epoch,
        "step": int(payload["step"]),
        "path": path,
    }


def find_latest_resume_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    numbered: list[tuple[int, Path]] = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("ckpt_epoch*.pt"):
            epoch = _checkpoint_epoch_from_name(path)
            if epoch is not None:
                numbered.append((epoch, path))
    if not numbered:
        other_checkpoints = (
            list(checkpoint_dir.glob("*.pt"))
            if checkpoint_dir.is_dir()
            else []
        )
        if other_checkpoints:
            raise RunnerError(
                "Checkpoint files exist but no numbered resume checkpoint is "
                "available; refusing to restart from epoch zero."
            )
        return None
    numbered.sort(key=lambda item: item[0], reverse=True)
    return inspect_resume_checkpoint(
        numbered[0][1],
        expected_config=expected_config,
    )


def _next_log_path(run_dir: Path, prefix: str) -> Path:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    index = 1
    while (logs / f"{prefix}_{index:03d}.log").exists():
        index += 1
    return logs / f"{prefix}_{index:03d}.log"


def invoke_training(
    *,
    run_dir: Path,
    config_path: Path,
    resume_checkpoint: Path | None,
    root: Path = ROOT,
) -> list[str]:
    arguments = [
        "-u",
        "scripts/train_v2.py",
        "--config",
        portable_path(config_path, root=root),
    ]
    if resume_checkpoint is not None:
        arguments.extend(
            [
                "--override",
                "training.resume_from="
                + portable_path(resume_checkpoint, root=root),
            ]
        )
    execute_logged(
        arguments,
        log_path=_next_log_path(run_dir, "training"),
        root=root,
    )
    return _logical_python_command(arguments)


def _verify_run_inputs(config: Mapping[str, Any], *, root: Path) -> None:
    data = _mapping_at(config, "data")
    required = {
        "cache_dir": "directory",
        "cache_lineage": "file",
        "dataset_contract": "file",
        "split_manifest": "file",
    }
    missing: list[str] = []
    for key, kind in required.items():
        value = data.get(key)
        if not isinstance(value, str):
            missing.append(f"data.{key}: not configured")
            continue
        path = resolve_repo_path(value, root=root)
        exists = path.is_dir() if kind == "directory" else path.is_file()
        if not exists:
            missing.append(f"data.{key}: {value}")
    if missing:
        raise RunnerError(
            "Cloud runtime inputs are missing: " + "; ".join(missing)
        )


def _resume_context(
    run_dir: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = _config_path(run_dir)
    state_path = _state_path(run_dir)
    if not config_path.is_file() or not state_path.is_file():
        raise RunnerError("Resume run lacks resolved_config.yaml or state.json")
    config = read_yaml(config_path)
    validate_base_config(config, root=root)
    state = read_json(state_path)
    if state.get("pipeline_id") != PIPELINE_ID:
        raise RunnerError("Resume state pipeline identity mismatch")
    if state.get("run_dir") != portable_path(run_dir, root=root):
        raise RunnerError("Resume state run directory mismatch")
    expected_config_hash = state.get("resolved_config_sha256")
    if expected_config_hash != sha256_file(config_path):
        raise RunnerError("Run-owned resolved config changed after launch")
    prior = _mapping_at(config, "prior_anchored_run")
    prior_path = resolve_repo_path(
        str(prior.get("prior_artifact_path", "")),
        root=root,
    )
    if not prior_path.is_file():
        raise RunnerError("Run-owned prior artifact is missing")
    if sha256_file(prior_path) != prior.get("prior_artifact_sha256"):
        raise RunnerError("Run-owned prior artifact SHA-256 mismatch")
    validate_prior_artifact(
        prior_path,
        expected_sha256=str(prior.get("prior_artifact_sha256", "")),
        root=root,
        strict_runtime=True,
    )
    mean_path = resolve_repo_path(
        str(prior.get("mean_checkpoint", "")),
        root=root,
    )
    if not mean_path.is_file():
        raise RunnerError("Selected mean checkpoint is missing on resume")
    expected_mean_hash = state.get("mean_checkpoint_sha256")
    if expected_mean_hash and sha256_file(mean_path) != expected_mean_hash:
        raise RunnerError("Selected mean checkpoint changed after launch")
    return config, state


def run_cloud(args: argparse.Namespace, *, root: Path = ROOT) -> dict[str, Any]:
    _require_cuda()
    config_relative = normalize_repo_relative(args.config, root=root)
    runs_relative = normalize_repo_relative(args.runs_root, root=root)
    manifest_override = (
        normalize_repo_relative(args.manifest, root=root)
        if args.manifest is not None
        else None
    )
    contract_override = (
        normalize_repo_relative(args.dataset_contract, root=root)
        if args.dataset_contract is not None
        else None
    )
    png_override = (
        normalize_repo_relative(args.png_root, root=root)
        if args.png_root is not None
        else None
    )
    if args.resume_latest:
        run_relative = select_latest_resumable_run(
            runs_relative,
            root=root,
        )
        resume = True
    elif args.resume:
        if args.run_dir is None:
            raise RunnerError("--resume requires --run-dir")
        run_relative = normalize_repo_relative(args.run_dir, root=root)
        resume = True
    else:
        resume = False
        run_relative = (
            normalize_repo_relative(args.run_dir, root=root)
            if args.run_dir is not None
            else f"{runs_relative}/{_run_id()}"
        )

    if resume:
        run_dir = resolve_repo_path(run_relative, root=root)
        if not run_dir.is_dir():
            raise RunnerError(f"Resume run directory does not exist: {run_relative}")
        config, state = _resume_context(run_dir, root=root)
        _verify_run_inputs(config, root=root)
        checkpoint_dir = _checkpoint_dir_from_config(config, root=root)
        latest = find_latest_resume_checkpoint(
            checkpoint_dir,
            expected_config=config,
        )
        if latest is not None and int(latest["epoch"]) == TOTAL_EPOCHS:
            update_state(
                run_dir,
                root=root,
                status="COMPLETE",
                current_stage=None,
                latest_checkpoint=portable_path(latest["path"], root=root),
                latest_epoch=TOTAL_EPOCHS,
                note="Training was already complete; no restart was attempted.",
            )
            return {
                "decision": "ALREADY_COMPLETE",
                "run_dir": run_relative,
                "latest_epoch": TOTAL_EPOCHS,
            }
        resume_path = latest["path"] if latest is not None else None
        update_state(
            run_dir,
            root=root,
            status="RUNNING",
            current_stage="training_resume",
            resumed_at_utc=utc_now(),
            resume_from=(
                portable_path(resume_path, root=root)
                if resume_path is not None
                else None
            ),
            latest_epoch=(int(latest["epoch"]) if latest is not None else 0),
        )
    else:
        run_dir = prepare_fresh_run(run_relative, root=root)
        base_config = read_yaml(resolve_repo_path(config_relative, root=root))
        validate_base_config(base_config, root=root)
        update_state(
            run_dir,
            root=root,
            status="RUNNING",
            current_stage="cloud_input_preflight",
            total_epochs=TOTAL_EPOCHS,
            phase_observation_epochs=list(PHASE_OBSERVATION_EPOCHS),
        )
        mean_checkpoint = select_mean_checkpoint(
            root=root,
            explicit=args.mean_checkpoint,
        )
        mean_hash = sha256_file(
            resolve_repo_path(mean_checkpoint, root=root)
        )
        effective_manifest = (
            manifest_override
            or str(_mapping_at(base_config, "data")["split_manifest"])
        )
        effective_contract = (
            contract_override
            or str(_mapping_at(base_config, "data")["dataset_contract"])
        )
        update_state(
            run_dir,
            root=root,
            current_stage="direct_png_prior_estimation",
            mean_checkpoint=mean_checkpoint,
            mean_checkpoint_sha256=mean_hash,
        )
        prior_path, prior_hash, prior_command = estimate_cloud_prior(
            run_dir=run_dir,
            mean_checkpoint=mean_checkpoint,
            manifest=normalize_repo_relative(effective_manifest, root=root),
            dataset_contract=normalize_repo_relative(
                effective_contract,
                root=root,
            ),
            png_root=png_override,
            device=args.device,
            root=root,
        )
        config = materialize_run_config(
            base_config,
            run_dir=run_dir,
            mean_checkpoint=mean_checkpoint,
            prior_path=prior_path,
            prior_sha256=prior_hash,
            manifest=effective_manifest,
            dataset_contract=effective_contract,
            root=root,
        )
        _verify_run_inputs(config, root=root)
        config_path = _config_path(run_dir)
        write_yaml_atomic(config_path, config)
        config_hash = sha256_file(config_path)
        update_state(
            run_dir,
            root=root,
            current_stage="training",
            resolved_config=portable_path(config_path, root=root),
            resolved_config_sha256=config_hash,
            prior_artifact=prior_path,
            prior_artifact_sha256=prior_hash,
            prior_command=prior_command,
        )
        state = read_json(_state_path(run_dir))
        resume_path = None

    config_path = _config_path(run_dir)
    try:
        training_command = invoke_training(
            run_dir=run_dir,
            config_path=config_path,
            resume_checkpoint=resume_path,
            root=root,
        )
        checkpoint_dir = _checkpoint_dir_from_config(config, root=root)
        final_path = checkpoint_dir / f"ckpt_epoch{TOTAL_EPOCHS:04d}.pt"
        if not final_path.is_file():
            raise RunnerError(
                "Training returned successfully without ckpt_epoch0100.pt"
            )
        final_metadata = inspect_resume_checkpoint(
            final_path,
            expected_config=config,
        )
        update_state(
            run_dir,
            root=root,
            status="COMPLETE",
            current_stage=None,
            training_command=training_command,
            latest_checkpoint=portable_path(final_path, root=root),
            latest_epoch=int(final_metadata["epoch"]),
            completed_at_utc=utc_now(),
        )
    except KeyboardInterrupt:
        update_state(
            run_dir,
            root=root,
            status="INTERRUPTED",
            current_stage="training",
            interrupted_at_utc=utc_now(),
        )
        raise
    except Exception as exc:
        update_state(
            run_dir,
            root=root,
            status="FAILED",
            failure=f"{type(exc).__name__}: {exc}",
            failed_at_utc=utc_now(),
        )
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "decision": "COMPLETE",
        "run_dir": portable_path(run_dir, root=root),
        "final_checkpoint": portable_path(final_path, root=root),
        "final_epoch": TOTAL_EPOCHS,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG).replace("\\", "/"))
    parser.add_argument(
        "--runs-root",
        default=str(DEFAULT_RUNS_ROOT).replace("\\", "/"),
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-latest", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--mean-checkpoint")
    parser.add_argument("--png-root")
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-contract")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.resume and args.resume_latest:
        parser.error("--resume and --resume-latest are mutually exclusive")
    if args.resume_latest and args.run_dir is not None:
        parser.error("--resume-latest selects the run; omit --run-dir")
    if args.preflight_only and (args.resume or args.resume_latest):
        parser.error("--preflight-only cannot resume a run")
    if args.device != "cuda" and not args.preflight_only:
        parser.error("real runs require --device cuda")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.preflight_only:
            result = build_static_preflight(
                config_path=args.config,
                runs_root=args.runs_root,
                png_root=args.png_root,
                manifest=args.manifest,
                dataset_contract=args.dataset_contract,
                mean_checkpoint=args.mean_checkpoint,
            )
        else:
            result = run_cloud(args)
    except (RunnerError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
