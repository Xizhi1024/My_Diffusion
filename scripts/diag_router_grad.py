"""Diagnostic: is the prior-anchored router gradient-deadlock structural?

Experiment A (minimal, falsifiable) for the prior-anchored spectral router.
Loads a trained prior-anchored checkpoint, freezes EVERYTHING except the two
router heads (`prior_active_heads` + `prior_destination_heads`), disables all
`spectral_router_regularization` terms, and measures on ONE fixed batch:

  reg_on  — sanity measurement reproducing the audit (~1e-6 grad on active head).
  A0      — pure data-loss gradient floor on each router head, with regularizers
            OFF (so the only gradient source is base diffusion + lesion losses).
            No optimizer step. Answers: "does the data loss push the route at all?"
  A1      — overfit the single batch with router-only AdamW at high LR.
            Answers: "can active_delta / shallow_mass grow at all?"

Verdict (see THRESHOLDS below):
  PASS  constraints/regularizers were the limiter
        -> A0 grad_active >= GRAD_FLOOR_PASS  AND  A1 delta grows to >= DELTA_PASS
  FAIL  amplitude/magnitude contract (computational graph) is the root cause
        -> A0 grad_active <  GRAD_FLOOR_FAIL OR  A1 delta stays   <  DELTA_FAIL
  else  INCONCLUSIVE

Run on the CLOUD box (the prior-anchored checkpoint + PNG cache live there; the
local dev box has neither — see memory [[run-env-cloud-win]]):

  python -u scripts/diag_router_grad.py \
      --config configs/experiments/slmf_png_prior_anchored_router_100e.yaml \
      --checkpoint results/prior_anchored_router_100e/runs/run-20260726T133123.363531Z/checkpoints/ckpt_epoch0100.pt \
      --steps 50 --lr 1e-2 --device cuda \
      --out results/router_diag

This script is READ-ONLY w.r.t. model code: it does not modify any source file.
The checkpoint is loaded weights_only; router regularizer weights are zeroed
IN-MEMORY only (never persisted). It writes one JSON artifact + prints a
summary table + a one-line verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Dict, Iterable, List, Tuple

import torch
import yaml

# Ensure src/ is on path (matches scripts/train_v2.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import build_dataloaders
from src.model.slmf_bbdm import SLMFBBDM

# --- verdict thresholds (single source of truth) ---------------------------
GRAD_FLOOR_PASS = 1.0e-4      # data-loss grad on active head strong enough to matter
GRAD_FLOOR_FAIL = 1.0e-7      # below this -> data loss contributes nothing
DELTA_PASS = 1.0e-2           # active_delta grew to a meaningful fraction of ±2 range
DELTA_FAIL = 1.0e-3           # delta never moved past this -> route inert
SHALLOW_PASS = 1.0e-2         # shallow mass recovered above this -> branch bootstrapped


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_run_config(template: Dict, checkpoint: str) -> Tuple[Dict, str]:
    """The template has h3_schedule_path=null, but prior_anchored_learned needs
    a real schedule. Prefer the run's resolved_config.yaml (authoritative for
    this checkpoint); else inject the run's own prior artifact + its sha."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    resolved = os.path.join(run_dir, "resolved_config.yaml")
    if os.path.exists(resolved):
        with open(resolved, "r", encoding="utf-8") as f:
            return yaml.safe_load(f), resolved
    prior_path = os.path.join(run_dir, "prior", "h3_prior_preview.json")
    if not os.path.exists(prior_path):
        raise FileNotFoundError(
            f"Neither {resolved} nor {prior_path} exist; cannot build "
            f"prior_anchored_learned router. Re-run scripts/estimate_h3_prior_from_png.py "
            f"into this run dir, or point --checkpoint at a run that still has its prior/."
        )
    cl = template["modules"]["residual_frequency"]["cross_level_router"]
    cl["h3_schedule_path"] = prior_path
    cl["h3_schedule_sha256"] = _sha256(prior_path)
    return template, prior_path


def _unwrap(model):
    return getattr(model, "_orig_mod", model)


def _pick(logs: Dict, *substrings: str) -> Dict[str, float]:
    """Robustly extract the first log key containing each substring."""
    out: Dict[str, float] = {}
    for s in substrings:
        for k, v in logs.items():
            if s in k and s not in out:
                out[s] = float(v.item() if torch.is_tensor(v) else v)
                break
    return out


def _zero_router_regularizers(model) -> List[str]:
    """Zero every spectral_router_regularization weight IN-MEMORY.

    With all weights zero the term contributes no gradient; base diffusion +
    lesion losses are the only remaining signal. Returns the list of zeroed
    attribute names so the artifact records what was neutralized.
    """
    name = "spectral_router_regularization"
    if name not in model.loss_terms:  # nn.ModuleDict: use __contains__, not .get()
        return []
    reg = model.loss_terms[name]
    names = [
        "temporal_weight", "dct_weight", "gabor_weight",
        "active_mass_weight", "prior_anchor_weight", "monotonic_weight",
        "curvature_weight", "budget_weight", "shallow_weight",
    ]
    for n in names:
        if hasattr(reg, n):
            setattr(reg, n, 0.0)
    return names


def _freeze_all_but_router_heads(model) -> Tuple[int, int]:
    """Freeze the whole model, then re-enable only the two prior-anchored heads."""
    m = _unwrap(model)
    router = m.residual_preconditioner
    frozen = trainable = 0
    for _, p in m.named_parameters():
        p.requires_grad_(False)
        frozen += p.numel()
    for head in (router.prior_active_heads, router.prior_destination_heads):
        for p in head.parameters():
            p.requires_grad_(True)
            trainable += p.numel()
            frozen -= p.numel()
    return trainable, frozen


def _head_final_rms_grad(heads: Iterable) -> float:
    """RMS gradient over each head's `.final` submodule (Trainer convention).

    Iterates the ``final`` submodule directly rather than name-matching, because
    ``head.named_parameters()`` yields *relative* names (``final.weight``) where
    the Trainer's ``.final.`` substring (valid only on full model names) matches
    nothing.
    """
    total = None
    count = 0
    for head in heads:
        final = getattr(head, "final", None)
        if final is None:
            continue
        for p in final.parameters(recurse=False):
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            total = g.square().sum() if total is None else total + g.square().sum()
            count += g.numel()
    if total is None or count == 0:
        return 0.0
    return float((total / count).sqrt().item())


def _heads_param_norm(heads: Iterable) -> float:
    total = None
    for head in heads:
        for p in head.parameters():
            x = p.detach().float()
            total = x.square().sum() if total is None else total + x.square().sum()
    return 0.0 if total is None else float(total.sqrt().item())


def _snapshot(heads: Iterable) -> Dict[int, torch.Tensor]:
    snap = {}
    for head in heads:
        for i, p in enumerate(head.parameters()):
            snap[id(p)] = p.detach().clone()
    return snap


def _update_over_param_ratio(heads: Iterable, snap: Dict[int, torch.Tensor]) -> float:
    num = den = 0.0
    for head in heads:
        for p in head.parameters():
            before = snap[id(p)]
            num += (p.detach().float() - before).square().sum().item()
            den += before.square().sum().item()
    if den <= 0:
        return 0.0
    return float((num / den) ** 0.5)


def _route_snapshot(model, logs: Dict) -> Dict[str, float]:
    """Pull the route-mass + delta diagnostics from one forward's logs."""
    return _pick(
        logs,
        "active_delta_abs_mean",   # |delta| averaged over the route
        "prior_active_mae",        # |active - prior| — how far off the prior
        "route_shallow_mass",
        "route_native_mass",
        "route_null_mass",
        "loss/base_diffusion",
        "active_delta_abs_max",
    )


def _set_router_epoch(model, epoch: int) -> None:
    m = _unwrap(model)
    setter = getattr(m.residual_preconditioner, "set_training_epoch", None)
    if callable(setter):
        setter(int(epoch))


def _forward_loss_logs(model, batch) -> Tuple[torch.Tensor, Dict]:
    # fp32 (no autocast) for clean, deterministic gradient accounting.
    return model(batch)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prior-anchored router gradient deadlock probe")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="prior-anchored ckpt; omit for fresh-init (verifies experiment D's de-zeroed projections without a trained ckpt overwriting them — then pass a resolved_config as --config)")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/router_diag")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"[diag] config     = {args.config}")
    if args.checkpoint:
        print(f"[diag] checkpoint = {args.checkpoint}")
        print(f"[diag] sha256     = {_sha256(args.checkpoint)}")
    else:
        print("[diag] checkpoint = (none — fresh init; experiment D de-zeroed projections active)")
    print(f"[diag] device={args.device} steps={args.steps} lr={args.lr}")

    # 1. config + model + checkpoint. The template has h3_schedule_path=null;
    #    with a checkpoint we prefer its run's resolved_config.yaml, else inject
    #    the run's prior artifact. Without a checkpoint (--no-checkpoint) the
    #    --config must already carry a valid h3_schedule (use a resolved_config).
    with open(args.config, "r", encoding="utf-8") as f:
        template = yaml.safe_load(f)
    if args.checkpoint:
        config, config_source = _resolve_run_config(template, args.checkpoint)
    else:
        config, config_source = template, os.path.abspath(args.config)
    config.setdefault("runtime", {})
    print(f"[diag] config_src = {config_source}")
    model = SLMFBBDM.from_config(config).to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.train()
    _set_router_epoch(model, 99)  # active_progress = destination_progress = 1

    # 2. one fixed batch (reuse the project loader so keys/normalization match)
    train_loader, _ = build_dataloaders(config.get("data", {}), config.get("runtime", {}))
    batch = next(iter(train_loader))
    batch = {
        k: (v.to(args.device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }
    print(f"[diag] batch keys = {list(batch.keys())}")
    print(f"[diag] batch size = {next(v.shape[0] for v in batch.values() if torch.is_tensor(v))}")

    # 3. freeze everything except the two router heads
    trainable, frozen = _freeze_all_but_router_heads(model)
    print(f"[diag] trainable = {trainable:,}  frozen = {frozen:,}  (router heads only)")

    artifact: Dict = {
        "meta": {
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": os.path.abspath(args.config),
            "config_resolved_from": config_source,
            "checkpoint": (os.path.abspath(args.checkpoint) if args.checkpoint else None),
            "checkpoint_sha256": (_sha256(args.checkpoint) if args.checkpoint else "fresh_init"),
            "device": args.device,
            "seed": args.seed,
            "steps": args.steps,
            "lr": args.lr,
            "trainable_param_count": trainable,
            "frozen_param_count": frozen,
            "batch_keys": list(batch.keys()),
            "thresholds": {
                "GRAD_FLOOR_PASS": GRAD_FLOOR_PASS,
                "GRAD_FLOOR_FAIL": GRAD_FLOOR_FAIL,
                "DELTA_PASS": DELTA_PASS,
                "DELTA_FAIL": DELTA_FAIL,
                "SHALLOW_PASS": SHALLOW_PASS,
            },
        },
    }

    router = _unwrap(model).residual_preconditioner
    active_heads = router.prior_active_heads
    dest_heads = router.prior_destination_heads

    # --- reg_on baseline (should reproduce the audit ~1e-6 active grad) -----
    model.zero_grad(set_to_none=True)
    loss_on, logs_on = _forward_loss_logs(model, batch)
    loss_on.backward()
    reg_on = {
        "grad_active": _head_final_rms_grad(active_heads),
        "grad_destination": _head_final_rms_grad(dest_heads),
        **_route_snapshot(model, logs_on),
        "loss_total": float(loss_on.item()),
    }
    artifact["reg_on_baseline"] = reg_on
    print(f"[reg_on ] grad_active={reg_on['grad_active']:.3e} "
          f"grad_dest={reg_on['grad_destination']:.3e} "
          f"delta={reg_on.get('active_delta_abs_mean', float('nan')):.3e} "
          f"loss={reg_on['loss_total']:.4e}")

    # --- A0: data-loss gradient floor (regularizers OFF, no optim step) -----
    zeroed = _zero_router_regularizers(model)
    artifact["meta"]["reg_weights_zeroed"] = zeroed
    model.zero_grad(set_to_none=True)
    loss_a0, logs_a0 = _forward_loss_logs(model, batch)
    loss_a0.backward()
    a0 = {
        "grad_active": _head_final_rms_grad(active_heads),
        "grad_destination": _head_final_rms_grad(dest_heads),
        **_route_snapshot(model, logs_a0),
        "loss_total": float(loss_a0.item()),
    }
    artifact["A0_reg_off_grad_floor"] = a0
    print(f"[A0     ] grad_active={a0['grad_active']:.3e} "
          f"grad_dest={a0['grad_destination']:.3e} "
          f"delta={a0.get('active_delta_abs_mean', float('nan')):.3e} "
          f"prior_mae={a0.get('prior_active_mae', float('nan')):.3e} "
          f"shallow={a0.get('route_shallow_mass', float('nan')):.3e} "
          f"base={a0.get('loss/base_diffusion', float('nan')):.4e}")

    # --- A1: router-only overfit, high LR, same batch ----------------------
    router_params = [p for h in (active_heads, dest_heads) for p in h.parameters()]
    opt = torch.optim.AdamW(router_params, lr=args.lr, weight_decay=0.0)
    per_step: List[Dict] = []
    for step in range(args.steps):
        snap = _snapshot([active_heads, dest_heads])
        model.zero_grad(set_to_none=True)
        loss, logs = _forward_loss_logs(model, batch)
        loss.backward()
        grad_active = _head_final_rms_grad(active_heads)
        grad_dest = _head_final_rms_grad(dest_heads)
        opt.step()
        upd_active = _update_over_param_ratio([active_heads], snap)
        upd_dest = _update_over_param_ratio([dest_heads], snap)
        rec = {
            "step": step,
            "loss_total": float(loss.item()),
            "grad_active": grad_active,
            "grad_destination": grad_dest,
            "update_param_active": upd_active,
            "update_param_destination": upd_dest,
            "param_norm_active": _heads_param_norm([active_heads]),
            "param_norm_destination": _heads_param_norm([dest_heads]),
            **_route_snapshot(model, logs),
        }
        per_step.append(rec)
        if step % 5 == 0 or step == args.steps - 1:
            print(f"[A1 {step:3d}] loss={rec['loss_total']:.4e} "
                  f"delta={rec.get('active_delta_abs_mean', float('nan')):.3e} "
                  f"prior_mae={rec.get('prior_active_mae', float('nan')):.3e} "
                  f"shallow={rec.get('route_shallow_mass', float('nan')):.3e} "
                  f"upd/par_act={upd_active:.2e}")
    artifact["A1_per_step"] = per_step

    # --- verdict -----------------------------------------------------------
    final = per_step[-1]
    max_delta = max(r.get("active_delta_abs_mean", 0.0) for r in per_step)
    max_shallow = max(r.get("route_shallow_mass", 0.0) for r in per_step)
    a0_grad = a0["grad_active"]

    if a0_grad >= GRAD_FLOOR_PASS and max_delta >= DELTA_PASS:
        result, reason = "PASS", (
            f"data-loss grad on active head = {a0_grad:.2e} (>= {GRAD_FLOOR_PASS:.0e}) "
            f"and delta grew to {max_delta:.2e} (>= {DELTA_PASS:.0e}); constraints "
            f"were the limiter, not the graph."
        )
    elif a0_grad < GRAD_FLOOR_FAIL or max_delta < DELTA_FAIL:
        result, reason = "FAIL", (
            f"data-loss grad = {a0_grad:.2e} (< {GRAD_FLOOR_FAIL:.0e} suggests near-zero "
            f"signal) OR delta stayed at {max_delta:.2e} (< {DELTA_FAIL:.0e}); root cause "
            f"is the amplitude/magnitude contract (zero-init projections), NOT training "
            f"length or LR. Proceed to experiment D (de-zero shallow projection / aux loss)."
        )
    else:
        result, reason = "INCONCLUSIVE", (
            f"a0_grad={a0_grad:.2e}, max_delta={max_delta:.2e}; between thresholds. "
            f"Re-run with --steps 200 and/or --lr 3e-2 to separate weak-signal from dead-signal."
        )
    artifact["verdict"] = {
        "result": result,
        "reason": reason,
        "a0_grad_active": a0_grad,
        "max_delta": max_delta,
        "max_shallow_mass": max_shallow,
        "shallow_recovered": max_shallow >= SHALLOW_PASS,
    }

    # 4. write artifact
    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = os.path.join(args.out, f"diag_router_grad_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print()
    print("=" * 72)
    print(f"VERDICT: {result}")
    print(reason)
    print(f"max_shallow_mass={max_shallow:.2e} (recovered={'yes' if max_shallow >= SHALLOW_PASS else 'no'})")
    print(f"artifact: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
