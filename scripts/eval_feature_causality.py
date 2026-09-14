"""Stage-A2 causal intervention audit CLI.

Runs the c2 lesion-intervention matrix (baseline, c2 lesion zero, background
replace, nonlesion same-area, shifted-mask sham, CT inpainting) on an existing
checkpoint, reusing the *same sampling noise per seed* across interventions.

Requires diffusion sampling, so this must run on a GPU — never on the training
GPU while the mainline training is active (queue it or use CUDA_VISIBLE_DEVICES).

Usage::

    python scripts/eval_feature_causality.py \\
        --config configs/experiments/slmf_png_xxx.yaml \\
        --checkpoint results/prior_anchored_router_300e/ckpt_best_lesion.pt \\
        --split val \\
        --output results/lesion_feature_emergence_v1/c2_causal/
        [--fake-data] [--max-samples N] [--num-seeds 4] [--device cuda:1]
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import socket
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data.dataset import CachedDataset, FakeDataset
from src.data.lineage import (
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.mechanism_validation.common import file_sha256, write_json
from src.mechanism_validation.feature_causality import (
    INTERVENTIONS,
    run_causality_audit,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM


def _select_checkpoint_state(checkpoint: dict, weights: str = "ema"):
    raw_state = checkpoint.get("model", checkpoint)
    if weights == "raw":
        return raw_state, "model"
    if weights != "ema":
        raise ValueError(f"Unknown checkpoint weights: {weights!r}")
    if checkpoint.get("checkpoint_weights") == "ema_materialized_as_model":
        return raw_state, "model (ema_materialized)"
    for key in ("ema_model", "model_ema"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state, key
    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, dict):
        shadow = ema_state.get("shadow")
        if isinstance(shadow, dict):
            merged = dict(raw_state)
            merged.update(shadow)
            return merged, "ema.shadow"
    raise KeyError("EMA weights requested, but checkpoint has no ema weights")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-A2 causal intervention audit")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--num-seeds", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=None, help="sampling steps (default: config)")
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument(
        "--interventions",
        nargs="+",
        default=None,
        choices=list(INTERVENTIONS.keys()),
        help="subset of interventions to run",
    )
    args = ap.parse_args()

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = Path(args.output).name
    out_dir = Path(args.output)
    # Fail-closed on output-directory reuse (stale COMPLETE.json hazard).
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"[causality] Refusing to write into non-empty run directory {out_dir}. "
            f"Use a unique run id or clear the directory first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_full_config(args.config)
    config = resolve_runtime_profile(config)
    if args.fake_data:
        config.setdefault("data", {})["use_fake_data"] = True
        config.setdefault("data", {})["image_size"] = 64
        config["data"]["require_cache_lineage"] = False
        config["data"].pop("cache_lineage", None)
        config.setdefault("modules", {}).setdefault("conditional_mean", {})[
            "enabled"
        ] = False
        config.setdefault("modules", {}).setdefault("mean_predictor", {})[
            "enabled"
        ] = False
        config.setdefault("modules", {}).setdefault("residual_bridge", {})[
            "enabled"
        ] = False
        config.setdefault("modules", {}).setdefault("residual_frequency", {})[
            "enabled"
        ] = False

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[causality] run_id={run_id} device={device} split={args.split}")

    model = SLMFBBDM.from_config(config)
    expected_lineage = load_checkpoint_data_lineage(config)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    validate_checkpoint_data_lineage(
        ckpt,
        expected_lineage,
        required=bool(config.get("data", {}).get("require_cache_lineage", False)),
        context=f"causality checkpoint {args.checkpoint}",
    )
    state, state_source = _select_checkpoint_state(ckpt, weights=args.weights)
    model.load_state_dict(state, strict=not args.fake_data)
    model.to(device)
    model.eval()
    print(f"[causality] weights={state_source}")

    data_cfg = config.get("data", {})
    if args.fake_data:
        ds = FakeDataset(16, data_cfg.get("image_size", 64), seed=0)
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    elif data_cfg.get("cache_dir"):
        split_manifest = Path(data_cfg["split_manifest"]) if data_cfg.get("split_manifest") else None
        ds = CachedDataset(
            data_cfg["cache_dir"], split=args.split, augment=False, split_manifest=split_manifest
        )
        if args.max_samples is not None:
            ds = torch.utils.data.Subset(ds, list(range(min(args.max_samples, len(ds)))))
        loader = torch.utils.data.DataLoader(ds, batch_size=data_cfg.get("val_batch_size", 4), shuffle=False)
    else:
        raise RuntimeError("No cache_dir and fake_data not enabled")

    print(f"[causality] loaded {len(ds)} samples")

    seeds = tuple(range(args.num_seeds))
    errors: list[str] = []
    try:
        decision = run_causality_audit(
            model,
            loader,
            device,
            out_dir,
            seeds=seeds,
            interventions=args.interventions,
            max_samples=args.max_samples,
            num_sampling_steps=args.num_steps,
        )
    except Exception as exc:
        print(f"[causality] FAILED: {exc!r}")
        errors.append(repr(exc))
        decision = None

    status = "FAILED" if errors else "COMPLETE"
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
    from src.mechanism_validation.common import canonical_json_sha256
    from src.mechanism_validation.feature_emergence import _patient_keys

    train_code_sha = str(ckpt.get("git_sha", "") or "").strip()
    eligibility = "exploratory" if train_code_sha else "restricted_no_train_lineage"
    manifest = {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "git_sha": _git_sha(),
        "checkpoint_path": str(Path(args.checkpoint)),
        "checkpoint_sha256": file_sha256(Path(args.checkpoint)),
        "checkpoint_train_code_sha": train_code_sha or "unknown",
        "checkpoint_state_source": state_source,
        "config_sha256": canonical_json_sha256(config),
        "cache_lineage_sha256": str((expected_lineage or {}).get("cache_metadata_sha256", "")),
        "split_manifest_sha256": _split_manifest_sha256(config),
        "patient_set_hash": _patient_set_hash(loader),
        "data_mode": "fake" if args.fake_data else "real",
        "eligibility": eligibility,
        "seeds": list(seeds),
        "command": " ".join(sys.argv),
        "started_at": started,
        "finished_at": finished,
        "exit_code": 0 if not errors else 1,
        "status": status,
        "errors": errors,
    }
    write_json(out_dir / "cloud_run_manifest.json", manifest)
    if errors:
        write_json(out_dir / "error_report.json", {"status": status, "errors": errors})
    else:
        write_json(out_dir / "causal_decision.json", decision)
        write_json(
            out_dir / "COMPLETE.json",
            {"run_id": run_id, "status": "COMPLETE", "finished_at": finished},
        )
    print(f"[causality] status={status}")
    return 0 if not errors else 1


def _split_manifest_sha256(config) -> str:
    manifest = config.get("data", {}).get("split_manifest", "")
    if not manifest:
        return ""
    path = Path(manifest)
    if not path.is_file():
        return ""
    try:
        return file_sha256(path)
    except Exception:
        return ""


def _patient_set_hash(loader) -> str:
    from src.mechanism_validation.common import canonical_json_sha256
    from src.mechanism_validation.feature_emergence import _patient_keys

    patients: set[str] = set()
    try:
        for batch in loader:
            patients.update(_patient_keys(batch))
    except Exception:
        return "error"
    if not patients:
        return "empty"
    return canonical_json_sha256(sorted(patients))


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
