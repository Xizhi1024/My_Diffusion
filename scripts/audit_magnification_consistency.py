"""Frozen full/zoom mechanism audit for small-lesion representation dilution.

This command consumes the immutable cohort exported by a completed medoid
confirmation run.  It never trains or updates the checkpoint.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate import _select_checkpoint_state
from src.mechanism_validation.magnification import lesion_crop_box
from src.mechanism_validation.magnification_audit import (
    CohortArrays,
    build_primary_gate,
    compute_gate_statistics,
    run_frozen_magnification_audit,
    validate_cohort,
    write_audit_artifacts,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, received {value!r}"
        ) from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("values must be unique")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON object from {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_source_cohort(source_run: Path) -> tuple[CohortArrays, dict[str, Any]]:
    """Load the fixed cohort while preserving zero-padded patient identities."""

    source_run = Path(source_run).resolve()
    required = [
        source_run / "COMPLETE.json",
        source_run / "run_manifest.json",
        source_run / "cohort.csv",
        source_run / "cohort_tensors.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "source run is incomplete; missing: " + ", ".join(missing)
        )
    completion = _read_json(source_run / "COMPLETE.json")
    if str(completion.get("status", "")).upper() != "COMPLETE":
        raise ValueError("source run COMPLETE.json does not declare completion")
    if completion.get("training_or_optimizer_used") is True:
        raise ValueError("source confirmation unexpectedly used training")
    source_manifest = _read_json(source_run / "run_manifest.json")
    cohort_table = pd.read_csv(
        source_run / "cohort.csv",
        dtype={"patient_id": str, "sample_id": str},
    )
    with np.load(
        source_run / "cohort_tensors.npz",
        allow_pickle=False,
    ) as archive:
        tensor_keys = {"ct", "target", "mask", "sample_ids"}
        missing_keys = tensor_keys.difference(archive.files)
        if missing_keys:
            raise ValueError(
                "cohort_tensors.npz is missing keys: "
                f"{sorted(missing_keys)}"
            )
        cohort = CohortArrays(
            cohort=cohort_table,
            ct=np.asarray(archive["ct"]).copy(),
            target=np.asarray(archive["target"]).copy(),
            mask=np.asarray(archive["mask"]).copy(),
            sample_ids=np.asarray(archive["sample_ids"]).astype(str).copy(),
        )
    validate_cohort(cohort)
    expected_samples = source_manifest.get("samples")
    if expected_samples is not None and int(expected_samples) != len(cohort_table):
        raise ValueError(
            "source manifest sample count does not match cohort: "
            f"{expected_samples} != {len(cohort_table)}"
        )
    expected_patients = source_manifest.get("patients")
    actual_patients = cohort_table["patient_id"].nunique()
    if expected_patients is not None and int(expected_patients) != actual_patients:
        raise ValueError(
            "source manifest patient count does not match cohort: "
            f"{expected_patients} != {actual_patients}"
        )
    return cohort, source_manifest


def _validate_crop_plan(
    cohort: CohortArrays,
    *,
    input_size: int,
    crop_sizes: Sequence[int],
) -> None:
    if cohort.ct.shape[-2:] != (input_size, input_size):
        raise ValueError(
            "input_size does not match cohort tensors: "
            f"{input_size} != {cohort.ct.shape[-2:]}"
        )
    for crop_size in crop_sizes:
        for index in range(len(cohort.cohort)):
            lesion_crop_box(
                torch.as_tensor(cohort.mask[index]),
                int(crop_size),
            )


def _validate_frozen_input_hashes(
    *,
    config: Path,
    checkpoint: Path,
    source_manifest: dict[str, Any],
) -> tuple[str, str]:
    config_hash = _sha256(config)
    checkpoint_hash = _sha256(checkpoint)
    expected_config = source_manifest.get("config_sha256")
    expected_checkpoint = source_manifest.get("checkpoint_sha256")
    if expected_config and config_hash != expected_config:
        raise ValueError(
            "config hash does not match the frozen source run: "
            f"{config_hash} != {expected_config}"
        )
    if expected_checkpoint and checkpoint_hash != expected_checkpoint:
        raise ValueError(
            "checkpoint hash does not match the frozen source run: "
            f"{checkpoint_hash} != {expected_checkpoint}"
        )
    return config_hash, checkpoint_hash


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {
            "commit_sha": run("rev-parse", "HEAD"),
            "worktree_dirty": bool(status),
            "worktree_status_porcelain": status or None,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "commit_sha": None,
            "worktree_dirty": None,
            "worktree_status_porcelain": None,
        }


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    return device


def _source_weights(
    requested: str,
    source_manifest: dict[str, Any],
) -> str:
    if requested == "source":
        requested = str(source_manifest.get("checkpoint_weights", "model"))
    aliases = {"model": "raw", "raw": "raw", "ema": "ema"}
    if requested not in aliases:
        raise ValueError(f"unsupported checkpoint weight source: {requested!r}")
    return aliases[requested]


def _load_frozen_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    weights: str,
) -> tuple[SLMFBBDM, dict[str, Any]]:
    config = resolve_runtime_profile(load_full_config(str(config_path)))
    construction_config = deepcopy(config)
    mean_config = (
        construction_config.setdefault("modules", {})
        .setdefault("conditional_mean", {})
    )
    mean_was_frozen = bool(mean_config.get("freeze", False))
    mean_config["checkpoint"] = None
    mean_config["freeze"] = False

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")
    state, state_source = _select_checkpoint_state(checkpoint, weights=weights)
    if not isinstance(state, dict):
        raise ValueError("selected checkpoint weights are not a state dict")
    if (
        bool(mean_config.get("enabled", False))
        and not any(key.startswith("mean_predictor.") for key in state)
    ):
        raise ValueError(
            "full checkpoint lacks mean_predictor weights required by "
            "the residual bridge"
        )

    model = SLMFBBDM.from_config(construction_config)
    model.load_state_dict(state, strict=True)
    model.mean_frozen = mean_was_frozen
    if mean_was_frozen and model.mean_predictor is not None:
        for parameter in model.mean_predictor.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    model.eval()
    metadata = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selected_weight_source": state_source,
        "conditional_mean_loaded_from_full_checkpoint": True,
    }
    return model, metadata


def _manifest(
    *,
    args: argparse.Namespace,
    repository_root: Path,
    cohort: CohortArrays,
    source_manifest: dict[str, Any],
    hashes: dict[str, str],
    device: torch.device,
    weight_source: str,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = len(cohort.cohort)
    trajectories = samples * len(args.seeds)
    diffusion_samples = trajectories * (1 + len(args.crop_sizes))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_checkpoint_full_zoom_mechanism_audit",
        "git": _git_provenance(repository_root),
        "inputs": {
            "config": str(args.config.resolve()),
            "config_sha256": hashes["config"],
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashes["checkpoint"],
            "source_run": str(args.source_run.resolve()),
            "source_manifest_sha256": hashes["source_manifest"],
            "cohort_csv_sha256": hashes["cohort_csv"],
            "cohort_tensors_sha256": hashes["cohort_tensors"],
        },
        "source_confirmation": {
            "schema_version": source_manifest.get("schema_version"),
            "checkpoint_epoch": source_manifest.get("checkpoint_epoch"),
            "checkpoint_weights": source_manifest.get("checkpoint_weights"),
        },
        "cohort": {
            "samples": samples,
            "patients": int(cohort.cohort["patient_id"].nunique()),
            "positive_lesion_slices_only": True,
        },
        "audit": {
            "seeds": list(args.seeds),
            "ddim_steps": args.ddim_steps,
            "input_size": args.input_size,
            "crop_sizes": list(args.crop_sizes),
            "primary_crop_size": args.primary_crop_size,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_replicates": args.bootstrap_replicates,
            "interpolation": {
                "ct_target_prediction": "bilinear_align_corners_false",
                "mask": "nearest",
                "noise": "bilinear_then_per_sample_standardization",
            },
            "estimated_trajectories": trajectories,
            "estimated_diffusion_samples": diffusion_samples,
        },
        "execution": {
            "dry_run": bool(args.dry_run),
            "training_or_optimizer_used": False,
            "medoid_rule_modified": False,
            "device": str(device),
            "amp": bool(args.amp and device.type == "cuda"),
            "batch_size": args.batch_size,
            "requested_weights": args.weights,
            "resolved_weights": weight_source,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "model": model_metadata,
    }


def _write_dry_run(output: Path, manifest: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output / "DRY_RUN.json").write_text(
        json.dumps(
            {
                "status": "validated_not_executed",
                "estimated_diffusion_samples": manifest["audit"][
                    "estimated_diffusion_samples"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write provenance without loading the model",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Run the long frozen-checkpoint audit",
    )
    parser.add_argument("--seeds", type=_parse_int_csv, default=(42, 43, 44, 45))
    parser.add_argument("--crop-sizes", type=_parse_int_csv, default=(96, 48))
    parser.add_argument("--primary-crop-size", type=int, default=48)
    parser.add_argument("--input-size", type=int, default=192)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--weights",
        choices=("source", "model", "raw", "ema"),
        default="source",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.source_run = args.source_run.resolve()
    args.output = args.output.resolve()
    for label, path in (
        ("config", args.config),
        ("checkpoint", args.checkpoint),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    if args.primary_crop_size not in args.crop_sizes:
        raise ValueError("primary_crop_size must be one of crop_sizes")
    if (
        args.input_size <= 0
        or args.ddim_steps <= 0
        or args.batch_size <= 0
        or args.bootstrap_replicates <= 0
    ):
        raise ValueError("numeric audit settings must be positive")

    cohort, source_manifest = load_source_cohort(args.source_run)
    _validate_crop_plan(
        cohort,
        input_size=args.input_size,
        crop_sizes=args.crop_sizes,
    )
    config_hash, checkpoint_hash = _validate_frozen_input_hashes(
        config=args.config,
        checkpoint=args.checkpoint,
        source_manifest=source_manifest,
    )
    hashes = {
        "config": config_hash,
        "checkpoint": checkpoint_hash,
        "source_manifest": _sha256(args.source_run / "run_manifest.json"),
        "cohort_csv": _sha256(args.source_run / "cohort.csv"),
        "cohort_tensors": _sha256(args.source_run / "cohort_tensors.npz"),
    }
    device = _resolve_device(args.device)
    weight_source = _source_weights(args.weights, source_manifest)
    manifest = _manifest(
        args=args,
        repository_root=REPOSITORY_ROOT,
        cohort=cohort,
        source_manifest=source_manifest,
        hashes=hashes,
        device=device,
        weight_source=weight_source,
    )
    if args.dry_run:
        _write_dry_run(args.output, manifest)
        return 0

    model, model_metadata = _load_frozen_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=device,
        weights=weight_source,
    )
    manifest["model"] = model_metadata
    tables = run_frozen_magnification_audit(
        model=model,
        cohort=cohort,
        device=device,
        seeds=args.seeds,
        crop_sizes=args.crop_sizes,
        input_size=args.input_size,
        num_steps=args.ddim_steps,
        batch_size=args.batch_size,
        amp=bool(args.amp and device.type == "cuda"),
    )
    statistics = compute_gate_statistics(
        tables,
        crop_sizes=args.crop_sizes,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    gate = build_primary_gate(
        statistics,
        primary_crop_size=args.primary_crop_size,
    )
    write_audit_artifacts(
        args.output,
        tables=tables,
        manifest=manifest,
        gate=gate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
