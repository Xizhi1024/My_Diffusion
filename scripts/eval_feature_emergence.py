"""Stage-A (zero-training) representation audit CLI: A0 + A1 + A2.

Runs the feature-source provenance audit, the cross-modal (shared-weight
probe) linear-CKA analysis, and the channel-agnostic lesion-emergence analysis
on an existing checkpoint.  No diffusion sampling is needed — only the CT
encoder forward pass and bundle construction — so this can run on CPU.

Usage::

    python scripts/eval_feature_emergence.py \\
        --config configs/experiments/slmf_png_xxx.yaml \\
        --checkpoint results/prior_anchored_router_300e/ckpt_best_lesion.pt \\
        --split val \\
        --output results/lesion_feature_emergence_v1/c1_audit/
        [--fake-data] [--max-samples N] [--device cpu|cuda:1]
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
from src.mechanism_validation.feature_emergence import (
    cross_modal_similarity_analysis,
    lesion_emergence_analysis,
)
from src.mechanism_validation.feature_provenance import audit_feature_provenance
from src.model.config_utils import load_full_config, resolve_runtime_profile
from src.model.slmf_bbdm import SLMFBBDM


def _select_checkpoint_state(checkpoint: dict, weights: str = "ema"):
    """Same logic as scripts/evaluate.py._select_checkpoint_state."""
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
    raise KeyError(
        "EMA weights requested, but checkpoint has no ema.shadow, ema_model, or model_ema"
    )


def _write_cloud_run_manifest(out_dir: Path, payload: dict) -> None:
    write_json(out_dir / "cloud_run_manifest.json", payload)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-A representation audit")
    ap.add_argument("--config", required=True, help="resolved experiment YAML")
    ap.add_argument("--checkpoint", required=True, help="full model checkpoint .pt")
    ap.add_argument("--split", default="val", help="train/val/test")
    ap.add_argument("--output", required=True, help="results/lesion_feature_emergence_v1/<run_id>/")
    ap.add_argument("--weights", default="ema", choices=("ema", "raw"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-t", type=int, default=0, help="timestep used for bundle extraction")
    ap.add_argument("--fake-data", action="store_true", help="use FakeDataset, no real cache")
    ap.add_argument("--skip-cka", action="store_true", help="skip A1 (e.g. no PET in batch)")
    ap.add_argument("--skip-emergence", action="store_true", help="skip A2")
    ap.add_argument(
        "--frozen-quartiles-json",
        default=None,
        help="JSON file with lesion-area quartiles frozen on the TRAIN split "
        "(e.g. {\"q1\": N, \"q2\": N, \"q3\": N}). Required when --split is "
        "val or test to avoid validation leakage.",
    )
    args = ap.parse_args()

    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = Path(args.output).name
    out_dir = Path(args.output)

    # Fail-closed on output-directory reuse: a non-empty run directory may hold
    # a stale COMPLETE.json that a sync process could mistake for THIS run.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"[feature-emergence] Refusing to write into non-empty run directory "
            f"{out_dir}. Use a unique run id or clear the directory first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_full_config(args.config)
    config = resolve_runtime_profile(config)
    if args.fake_data:
        config.setdefault("data", {})["use_fake_data"] = True
        config.setdefault("data", {})["image_size"] = 64
        # Strict lineage conflicts with fake data; drop it for smoke runs.
        config["data"]["require_cache_lineage"] = False
        config["data"].pop("cache_lineage", None)
        # Disable modules that require real cloud checkpoints.
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"[feature-emergence] run_id={run_id} device={device} split={args.split}")

    model = SLMFBBDM.from_config(config)
    expected_lineage = load_checkpoint_data_lineage(config)

    print(f"[feature-emergence] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    validate_checkpoint_data_lineage(
        ckpt,
        expected_lineage,
        required=bool(config.get("data", {}).get("require_cache_lineage", False)),
        context=f"feature-emergence checkpoint {args.checkpoint}",
    )
    state, state_source = _select_checkpoint_state(ckpt, weights=args.weights)
    # In fake-data mode the config is deliberately reduced (modules disabled to
    # avoid cloud checkpoints), so the real checkpoint has extra keys.  Load
    # loosely for smoke runs; real runs must match strictly.
    model.load_state_dict(state, strict=not args.fake_data)
    model.to(device)
    model.eval()
    print(f"[feature-emergence] weights={state_source}")

    # A0: provenance audit (static).
    provenance = audit_feature_provenance(model, config, args.checkpoint, out_dir)
    provenance["checkpoint"]["weights_are_ema"] = args.weights == "ema"
    provenance["checkpoint"]["state_source"] = state_source
    train_code_sha = str(ckpt.get("git_sha", "") or "").strip()
    provenance["checkpoint"]["train_code_sha"] = train_code_sha or "unknown"
    provenance["timestep_policy"]["fixed_t_for_extraction"] = args.fixed_t
    write_json(out_dir / "feature_provenance.json", provenance)
    print(f"[feature-emergence] A0 provenance written to {out_dir}")

    # Build dataloader.
    data_cfg = config.get("data", {})
    if args.fake_data:
        ds = FakeDataset(16, data_cfg.get("image_size", 64), seed=args.seed)
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

    print(f"[feature-emergence] loaded {len(ds)} samples for split={args.split}")

    results = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "stage": "A0+A1+A2",
    }
    errors: list[str] = []

    if not args.skip_cka:
        try:
            cka = cross_modal_similarity_analysis(
                model, loader, device, out_dir, seed=args.seed
            )
            results["A1_cross_modal"] = {
                "s1_gate": cka["s1_gate"],
                "layer_metrics": cka["layer_rows"],
            }
            print("[feature-emergence] A1 CKA complete")
        except Exception as exc:
            print(f"[feature-emergence] A1 FAILED: {exc!r}")
            errors.append(f"A1_cross_modal: {exc!r}")
            results["A1_cross_modal"] = {"error": repr(exc)}

    if not args.skip_emergence:
        try:
            frozen_quartiles = _load_frozen_quartiles(args, config, loader)
            emergence = lesion_emergence_analysis(
                model,
                loader,
                device,
                out_dir,
                fixed_t=args.fixed_t,
                frozen_quartiles=frozen_quartiles,
            )
            results["A2_emergence"] = {
                "m1_gate": emergence["m1_gate"],
                "layer_summary": emergence["layer_summary"],
            }
            print("[feature-emergence] A2 emergence complete")
        except Exception as exc:
            print(f"[feature-emergence] A2 FAILED: {exc!r}")
            errors.append(f"A2_emergence: {exc!r}")
            results["A2_emergence"] = {"error": repr(exc)}

    # Fail-closed: a stage that ran but errored means the run is NOT COMPLETE,
    # and no formal decision summary may be emitted for the failed stages.
    status = "FAILED" if errors else "COMPLETE"
    finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
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
        "config_sha256": provenance["config"]["sha256"],
        "cache_lineage_sha256": str(
            (expected_lineage or {}).get("cache_metadata_sha256", "")
        ),
        "split_manifest_sha256": _split_manifest_sha256(config),
        "patient_set_hash": _patient_set_hash(config, loader),
        "data_mode": "fake" if args.fake_data else "real",
        "eligibility": eligibility,
        "seeds": [args.seed],
        "command": " ".join(sys.argv),
        "started_at": started,
        "finished_at": finished,
        "exit_code": 0 if not errors else 1,
        "status": status,
        "errors": errors,
    }
    _write_cloud_run_manifest(out_dir, manifest)
    if errors:
        write_json(out_dir / "error_report.json", {"status": status, "errors": errors})
        print(f"[feature-emergence] FAILED stages: {errors}")
    else:
        write_json(out_dir / "audit_summary.json", results)
        write_json(
            out_dir / "COMPLETE.json",
            {"run_id": run_id, "status": "COMPLETE", "finished_at": finished},
        )
    print(f"[feature-emergence] status={status} Outputs in {out_dir}")
    return 0 if not errors else 1


def _git_sha() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


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


def _patient_set_hash(config, loader) -> str:
    """Canonical hash of the sorted patient set actually loaded for this run."""
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


def _load_frozen_quartiles(args, config, loader) -> list[float] | None:
    """Resolve train-frozen lesion-area quartiles for the emergence analysis.

    Fail-closed rule: on val/test (i.e. any split that is not the training
    split), frozen quartiles MUST come from a provided JSON file.  Freezing from
    the evaluation split itself is validation leakage and is rejected.  On the
    train split (or fake-data smoke), the quartiles may be computed from the
    loader and are labelled accordingly.
    """
    from src.mechanism_validation.common import load_json

    if args.frozen_quartiles_json:
        payload = load_json(Path(args.frozen_quartiles_json))
        keys = ("q1", "q2", "q3")
        if not all(k in payload for k in keys):
            raise ValueError(
                f"frozen-quartiles JSON must contain q1/q2/q3, got keys {sorted(payload)}"
            )
        q = [float(payload[k]) for k in keys]
        if not (q[0] <= q[1] <= q[2]):
            raise ValueError(f"frozen quartiles must be non-decreasing, got {q}")
        print(f"[feature-emergence] using train-frozen quartiles {q} from {args.frozen_quartiles_json}")
        return q

    if args.fake_data or args.split == "train":
        # Computed from the (train) loader; labelled frozen_on=train_loader by
        # the analysis function.
        return None

    raise RuntimeError(
        f"split={args.split!r} requires --frozen-quartiles-json with thresholds "
        f"frozen on the TRAIN split. Freezing quartiles from the evaluation "
        f"split is validation leakage."
    )


if __name__ == "__main__":
    sys.exit(main())
