#!/usr/bin/env python3
"""Runner for the perceptual-x0 five-arm ablation (Stage A).

Local-safe entry point is ``--dry-run``: it audits the five-arm fairness
contract, reports the resolved command for each requested variant, and marks
BLOCKED when a required real encoder checkpoint is missing.  It never fakes a
runnable state and never starts a real long training run.

A real launch requires:
  * a frozen pretrained PET encoder checkpoint for P2/P3 (P4 uses an explicit
    random encoder, so it needs none), and
  * CUDA (formal few-step evaluation forbids the CPU smoke profile).

Artifacts under ``--output``:
  resolved_runs.json   per-variant resolved config + command + execution status
  execution_metadata.json  repository git commit / dirty status / run owner info
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.config_utils import apply_dotlist_overrides, load_full_config  # noqa: E402

SCHEMA_VERSION = 1
PIPELINE_ID = "PFM_LESION_PERCEPTUAL_X0_V1"
DEFAULT_CONFIG = Path("configs/experiments/perceptual_x0_ablation_plan_v1.yaml")
BASE_CONFIG = Path("configs/experiments/slmf_png_baseline.yaml")
DEFAULT_OUTPUT = Path("results/perceptual_x0_few_step_v1")
ARM_NAMES = ("P0_PIXEL", "P1_SEG_OUT", "P2_FEAT_GLOBAL", "P3_FEAT_LESION_BALANCED", "P4_FEAT_RANDOM")


class RunnerError(RuntimeError):
    """A fail-closed runner contract violation."""


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    temporary.replace(path)


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "dirty_files": [line.split(" ", 2)[-1] for line in status.splitlines()],
    }


def _arm_loss_overrides(arm: Mapping[str, Any]) -> dict[str, Any]:
    """Return dotted-path loss overrides for one arm."""
    overrides: dict[str, Any] = {}
    for loss_name, loss_cfg in arm.get("losses", {}).items():
        for key, value in loss_cfg.items():
            overrides[f"losses.{loss_name}.{key}"] = value
    return overrides


def resolve_variant_config(
    plan: Mapping[str, Any],
    *,
    arm_name: str,
    base_config: Path,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    """Resolve the base config plus one arm's supervision overrides."""
    arms = plan.get("arms", {})
    if arm_name not in arms:
        raise RunnerError(f"Unknown arm {arm_name!r}; available: {sorted(arms)}")
    arm = arms[arm_name]
    config = load_full_config(str(base_config))
    config["experiment"]["name"] = f"perceptual_x0_{arm_name}"
    config["experiment"]["seed"] = int(seed)
    config.setdefault("formal_mechanism", {})["pipeline_id"] = PIPELINE_ID
    config["formal_mechanism"]["schema_version"] = SCHEMA_VERSION
    config["formal_mechanism"]["variant"] = arm_name
    config["formal_mechanism"]["supervision"] = arm.get("supervision", "pixel")
    config["formal_mechanism"]["weight_selection_partition"] = "calibration"
    config["formal_mechanism"]["validation_used_for_selection"] = False
    # Record the immutable base commit the experiment is layered on.  This is
    # the exact SHA the branch was reset to before any PFM change; recording it
    # here makes the base unambiguous even after the worktree is committed.
    base_sha = str(plan.get("base_commit_sha256") or "").strip()
    if not base_sha:
        raise RunnerError(
            "plan.base_commit_sha256 is required so experiment provenance "
            "records the immutable git base"
        )
    config["formal_mechanism"]["base_commit_sha256"] = base_sha

    overrides = _arm_loss_overrides(arm)
    # P1_SEG_OUT: carry the arm's segmenter config so the model actually loads
    # a frozen segmenter instead of running with self.segmenter=None.
    seg_cfg = arm.get("segmenter")
    if isinstance(seg_cfg, dict) and seg_cfg.get("enabled", False):
        config["segmenter"] = dict(seg_cfg)
        config["model"]["segmenter"] = dict(seg_cfg)
        overrides["segmenter.enabled"] = True
        if seg_cfg.get("checkpoint"):
            overrides["segmenter.checkpoint"] = seg_cfg["checkpoint"]
    # Formal ablation: the held-out validation partition must never drive
    # early stopping or best-checkpoint selection.  Training runs a fixed
    # epoch count; the exact final-epoch EMA checkpoint is the evaluation
    # subject.  This is the plan's declared contract enforced in code.
    config.setdefault("runtime", {})["early_stopping"] = {
        "enabled": False
    }
    config["runtime"]["best_checkpoint"] = {"enabled": False}
    config["formal_mechanism"]["checkpoint_policy"] = "fixed_epoch_ema_no_selection"
    config["formal_mechanism"]["early_stopping_enabled"] = False
    config["formal_mechanism"]["best_checkpoint_enabled"] = False
    # Pin the split manifest to the physical path shared by all arms.  The
    # base config spells it `../main_data/...` (legacy) while the plan uses
    # `main_data/...`; both must resolve to the same file.  Resolving against
    # ROOT here makes the resolved config self-contained and unambiguous.
    plan_split = str(plan.get("fairness", {}).get("split_manifest") or "").replace("\\", "/")
    if plan_split:
        manifest_path = Path(plan_split)
        if not manifest_path.is_absolute():
            manifest_path = (root / manifest_path).resolve()
        config["data"]["split_manifest"] = manifest_path.as_posix()
    # Pin the cache dir to the plan's declared value so the encoder and the
    # generation model read the same data version.  The base config uses a
    # different legacy cache dir (`cache/tensors_main`); forcing the plan's
    # canonical cache dir here prevents a silent data-version mismatch.
    plan_cache = str(plan.get("fairness", {}).get("cache_dir") or "").replace("\\", "/")
    if plan_cache:
        cache_path = Path(plan_cache)
        if not cache_path.is_absolute():
            cache_path = (root / cache_path).resolve()
        config["data"]["cache_dir"] = cache_path.as_posix()
    return apply_dotlist_overrides(config, overrides)


def _encoder_checkpoint_for(arm: Mapping[str, Any], root: Path) -> Path | None:
    """Resolve the perceptual encoder checkpoint path for an arm, or None."""
    perceptual = arm.get("losses", {}).get("perceptual_x0", {})
    if not perceptual.get("enabled", False):
        return None
    if perceptual.get("encoder_kind") == "random":
        return None
    checkpoint = perceptual.get("checkpoint")
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def audit_arm(
    plan: Mapping[str, Any],
    *,
    arm_name: str,
    base_config: Path,
    root: Path,
) -> dict[str, Any]:
    """Audit one arm's fairness contract. Returns a report dict."""
    arms = plan.get("arms", {})
    arm = arms[arm_name]
    report: dict[str, Any] = {
        "arm": arm_name,
        "supervision": arm.get("supervision"),
        "checks": {},
        "blockers": [],
        "status": "READY",
    }

    def check(name: str, ok: bool, detail: str) -> None:
        report["checks"][name] = {"ok": bool(ok), "detail": detail}

    # 1. Only-loss-difference: base must equal except for the two supervision
    #    losses.  We compare against the bare base config's losses, ignoring
    #    the formal_mechanism block we add ourselves.
    base = load_full_config(str(base_config))
    base_losses = base.get("losses", {})
    arm_losses = arm.get("losses", {})
    unexpected = set(arm_losses) - {"segmenter_consistency", "perceptual_x0"}
    check(
        "only_supervision_losses_differ",
        not unexpected,
        f"arm declares {sorted(unexpected) or 'none'} unexpected losses",
    )
    if unexpected:
        report["blockers"].append(
            f"Arm {arm_name} mutates non-supervision losses {sorted(unexpected)}; "
            "arms must differ only in segmenter_consistency / perceptual_x0"
        )

    # 2. Baseline losses must not be silently changed by the arm.
    baseline_losses = {
        name: cfg for name, cfg in base_losses.items()
        if name not in ("segmenter_consistency", "perceptual_x0")
    }
    arm_declared = {
        name: cfg for name, cfg in arm_losses.items()
        if name not in ("segmenter_consistency", "perceptual_x0")
    }
    check(
        "baseline_losses_untouched",
        not arm_declared,
        "arm must not override non-supervision losses",
    )

    # 3. Shared data split / seed / epochs.  The base config uses a legacy
    #    `../main_data` spelling while the plan documents the canonical
    #    `main_data` path; both must resolve to the SAME physical manifest file,
    #    and that file must exist.  A missing manifest is a hard blocker, not a
    #    silent fallback to _meta.json splits.
    base_data = base.get("data", {})
    plan_fairness = plan.get("fairness", {})
    base_split = str(base_data.get("split_manifest") or "").replace("\\", "/")
    plan_split = str(plan_fairness.get("split_manifest") or "").replace("\\", "/")

    def _resolve_manifest(value: str) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    base_manifest = _resolve_manifest(base_split)
    plan_manifest = _resolve_manifest(plan_split)
    same_physical = (
        base_manifest is not None
        and plan_manifest is not None
        and base_manifest == plan_manifest
    )
    base_exists = base_manifest is not None and base_manifest.is_file()
    plan_exists = plan_manifest is not None and plan_manifest.is_file()
    check(
        "shared_data_split",
        same_physical,
        f"base={base_manifest} vs plan={plan_manifest}",
    )
    check(
        "manifest_exists",
        bool(base_exists and plan_exists),
        f"base={'found' if base_exists else 'MISSING'}:{base_manifest} "
        f"plan={'found' if plan_exists else 'MISSING'}:{plan_manifest}",
    )
    if not same_physical:
        report["blockers"].append(
            f"Arm {arm_name}: base split_manifest ({base_manifest}) and plan "
            f"split_manifest ({plan_manifest}) resolve to different files; arms "
            "must share one physical manifest"
        )
    if not (base_exists and plan_exists):
        report["blockers"].append(
            f"Arm {arm_name}: split manifest does not exist "
            f"(base={base_manifest} found={base_exists}, "
            f"plan={plan_manifest} found={plan_exists}); a missing manifest "
            "would silently fall back to _meta.json splits in CachedDataset"
        )
    # 3b. Cache dir: the plan's declared cache must exist, and the encoder and
    #     generation model must read the same data version.  The base config's
    #     legacy cache dir is pinned away by resolve_variant_config; the audit
    #     surfaces the divergence so it is never silent.
    base_cache = str(base_data.get("cache_dir") or "").replace("\\", "/")
    plan_cache = str(plan_fairness.get("cache_dir") or "").replace("\\", "/")
    plan_cache_path = (
        _resolve_manifest(plan_cache) if plan_cache else None
    )
    plan_cache_exists = plan_cache_path is not None and plan_cache_path.is_dir()
    check(
        "plan_cache_dir_declared",
        bool(plan_cache),
        f"plan cache_dir={plan_cache!r}",
    )
    check(
        "plan_cache_dir_exists",
        bool(plan_cache_exists),
        f"plan cache_dir={'found' if plan_cache_exists else 'MISSING'}:{plan_cache_path}",
    )
    if plan_cache and not plan_cache_exists:
        report["blockers"].append(
            f"Arm {arm_name}: plan cache_dir {plan_cache_path} does not exist; "
            "training would silently read an empty or missing cache"
        )
    check(
        "shared_cache_dir",
        bool(base_cache)
        and bool(plan_cache)
        and _resolve_manifest(base_cache) == _resolve_manifest(plan_cache),
        f"base cache_dir={base_cache!r} vs plan={plan_cache!r}",
    )
    shared_seed_ok = base.get("experiment", {}).get("seed") == plan_fairness.get(
        "shared_initialization_seed"
    )
    check(
        "shared_seed",
        shared_seed_ok,
        f"seed={base.get('experiment', {}).get('seed')}",
    )
    if not shared_seed_ok:
        report["blockers"].append(
            f"Arm {arm_name}: base seed "
            f"({base.get('experiment', {}).get('seed')}) != plan "
            f"shared_initialization_seed "
            f"({plan_fairness.get('shared_initialization_seed')})"
        )
    shared_epochs_ok = base.get("training", {}).get("num_epochs") == plan_fairness.get(
        "shared_epochs"
    )
    check(
        "shared_epochs",
        shared_epochs_ok,
        f"epochs={base.get('training', {}).get('num_epochs')}",
    )
    if not shared_epochs_ok:
        report["blockers"].append(
            f"Arm {arm_name}: base epochs "
            f"({base.get('training', {}).get('num_epochs')}) != plan "
            f"shared_epochs ({plan_fairness.get('shared_epochs')})"
        )

    # 4. Encoder freeze / checkpoint lineage.
    perceptual = arm.get("losses", {}).get("perceptual_x0", {})
    if perceptual.get("enabled", False):
        encoder_kind = perceptual.get("encoder_kind", "pretrained")
        check("encoder_kind_valid", encoder_kind in {"pretrained", "random"}, encoder_kind)
        if encoder_kind == "random":
            check("random_negative_control", True, "explicit P4 random encoder")
        else:
            ckpt = _encoder_checkpoint_for(arm, root)
            exists = ckpt is not None and ckpt.is_file()
            check("encoder_checkpoint_exists", bool(exists), str(ckpt) if ckpt else "missing")
            check(
                "encoder_lineage_required",
                bool(perceptual.get("require_checkpoint_lineage", False)),
                "strict lineage required for pretrained encoder",
            )
            if not exists:
                report["blockers"].append(
                    f"P2/P3 arm {arm_name} requires a real frozen encoder "
                    f"checkpoint at {ckpt}; refusing to fall back to random weights"
                )
        if encoder_kind == "random" and not arm_name.startswith("P4"):
            report["blockers"].append(
                "random encoder allowed only for the P4_FEAT_RANDOM negative control"
            )
    else:
        check("perceptual_disabled", True, "perceptual_x0 disabled")

    # 5. Segmenter (P1_SEG_OUT): must carry a real frozen checkpoint or the
    #    segmenter_consistency loss is silently zero.
    seg_cfg = arm.get("segmenter") or {}
    seg_enabled = bool(seg_cfg.get("enabled", False))
    seg_ckpt = str(seg_cfg.get("checkpoint") or "").strip()
    if seg_enabled:
        seg_path = Path(seg_ckpt) if seg_ckpt else None
        if seg_path is not None and not seg_path.is_absolute():
            seg_path = root / seg_path
        if seg_path is not None:
            seg_path = seg_path.resolve()
        seg_exists = seg_path is not None and seg_path.is_file()
        check("segmenter_checkpoint_exists", bool(seg_exists), seg_ckpt or "missing")
        if not seg_exists:
            report["blockers"].append(
                f"Arm {arm_name} enables segmenter_consistency but has no "
                f"real segmenter checkpoint at {seg_ckpt!r}; the loss would be "
                "silently zero"
            )
        lineage_required = bool(seg_cfg.get("require_checkpoint_lineage", False))
        check(
            "segmenter_lineage_required",
            lineage_required,
            "strict lineage required for the frozen P1 segmenter",
        )
        if not lineage_required:
            report["blockers"].append(
                f"Arm {arm_name}: segmenter.require_checkpoint_lineage must be true"
            )
        if seg_exists and seg_path is not None:
            sidecar_path = seg_path.with_name(seg_path.name + ".lineage.json")
            sidecar_exists = sidecar_path.is_file()
            check(
                "segmenter_lineage_exists",
                sidecar_exists,
                str(sidecar_path),
            )
            if not sidecar_exists:
                report["blockers"].append(
                    f"Arm {arm_name}: segmenter lineage sidecar missing: {sidecar_path}"
                )
            else:
                try:
                    lineage = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    recorded_sha = str(lineage.get("checkpoint_sha256") or "").lower()
                    actual_sha = sha256_file(seg_path).lower()
                    hash_ok = bool(recorded_sha) and recorded_sha == actual_sha
                    metrics = lineage.get("eval_metrics") or {}
                    recall = float(metrics.get("lesion_recall", -1.0))
                    small_recall = float(metrics.get("small_lesion_recall", -1.0))
                    gates = lineage.get("gates") or {}
                    gates_ok = (
                        bool(gates.get("pass", False))
                        and recall >= float(gates.get("lesion_recall_min", 0.70))
                        and small_recall
                        >= float(gates.get("small_lesion_recall_min", 0.70))
                    )
                    roles = set((lineage.get("train_patient_split") or {}).values())
                    roles_ok = bool(roles) and roles <= {"train", "calibration"}
                    check("segmenter_lineage_hash", hash_ok, actual_sha)
                    check(
                        "segmenter_recall_gate",
                        gates_ok,
                        f"recall={recall:.4f} small_recall={small_recall:.4f}",
                    )
                    check(
                        "segmenter_partition_roles",
                        roles_ok,
                        f"roles={sorted(roles)}",
                    )
                    if not hash_ok:
                        report["blockers"].append(
                            f"Arm {arm_name}: segmenter checkpoint hash does not "
                            "match its lineage sidecar"
                        )
                    if not gates_ok:
                        report["blockers"].append(
                            f"Arm {arm_name}: segmenter recall gates did not pass"
                        )
                    if not roles_ok:
                        report["blockers"].append(
                            f"Arm {arm_name}: segmenter lineage contains forbidden "
                            f"patient roles {sorted(roles)}"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    check("segmenter_lineage_valid", False, str(exc))
                    report["blockers"].append(
                        f"Arm {arm_name}: invalid segmenter lineage: {exc}"
                    )
    else:
        check("segmenter_disabled", True, "segmenter not used")

    # 6. Validation never used for selection.
    validation_for_selection = bool(
        plan.get("fairness", {}).get("validation_touched_for_selection", False)
    )
    check(
        "validation_not_selection",
        not validation_for_selection,
        "validation_touched_for_selection=false",
    )
    if validation_for_selection:
        report["blockers"].append(
            f"Arm {arm_name}: plan declares "
            "fairness.validation_touched_for_selection=true; held-out "
            "validation patients must never participate in selection"
        )

    # 7. Any fairness failure is a hard blocker.
    if report["blockers"]:
        report["status"] = "BLOCKED"
    return report


def run_dry_run(
    plan: Mapping[str, Any],
    *,
    plan_path: Path,
    base_config: Path,
    output_dir: Path,
    root: Path,
) -> dict[str, Any]:
    """Audit all five arms without launching training. Reports BLOCKED honestly."""
    resolved_runs: dict[str, Any] = {}
    any_blocked = False
    for arm_name in ARM_NAMES:
        audit = audit_arm(plan, arm_name=arm_name, base_config=base_config, root=root)
        resolved = resolve_variant_config(
            plan, arm_name=arm_name, base_config=base_config, seed=42, root=root
        )
        resolved_path = output_dir / "configs" / f"{arm_name}.yaml"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_path.write_text(
            yaml.safe_dump(resolved, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        resolved_runs[arm_name] = {
            "audit": audit,
            "resolved_config": resolved_path.as_posix(),
            "command": (
                f"{sys.executable} scripts/train_v2.py --config "
                f"{resolved_path}"
            ),
            "git": git_info(root),
        }
        if audit["status"] == "BLOCKED":
            any_blocked = True

    base_sha = str(plan.get("base_commit_sha256") or "").strip()
    head_sha = git_info(root)["commit"]
    execution = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "mode": "dry_run",
        "plan": plan_path.as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "base_config": base_config.as_posix(),
        "base_config_sha256": sha256_file(base_config),
        "base_commit_sha256": base_sha,
        "head_commit_sha256": head_sha,
        "worktree_is_on_base": base_sha == head_sha,
        "git": git_info(root),
        "dry_run_status": "BLOCKED" if any_blocked else "READY",
        "started_utc": utc_now(),
        "note": "Dry-run only. Real launch requires encoder checkpoints + CUDA.",
    }
    write_json_atomic(output_dir / "resolved_runs.json", resolved_runs)
    write_json_atomic(output_dir / "execution_metadata.json", execution)
    return execution


def launch_training(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    base_config: Path,
    variant: str,
    seed: int,
    output_dir: Path,
    root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Resolve one arm's config, write it, and invoke train_v2.py.

    Real training requires CUDA and the frozen encoder / segmenter checkpoints
    the arm needs; the audit must have passed before this is called.
    """
    resolved = resolve_variant_config(
        plan,
        arm_name=variant,
        base_config=base_config,
        seed=seed,
        root=root,
    )
    if resume:
        # train_v2.py has no --resume flag; resume is config-driven via
        # training.resume_from (the trainer auto-detects the latest checkpoint).
        resolved["training"]["resume_from"] = resolved["training"].get(
            "resume_from"
        ) or resolved["training"].get("init_from")
        resolved["training"]["init_from"] = None
    config_path = output_dir / "configs" / f"{variant}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            resolved,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "scripts/train_v2.py",
        "--config",
        str(config_path),
    ]
    environment = dict(__import__("os").environ)
    environment["PYTHONUNBUFFERED"] = "1"
    log_path = output_dir / "logs" / f"{variant}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="") as log:
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        raise RunnerError(
            f"train_v2.py failed with exit {exit_code} for {variant}: {log_path}"
        )
    return {
        "variant": variant,
        "config": config_path.as_posix(),
        "command": subprocess.list2cmdline(command),
        "log": log_path.as_posix(),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG).replace("\\", "/"))
    parser.add_argument("--base-config", default=str(BASE_CONFIG).replace("\\", "/"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", choices=ARM_NAMES, default="P3_FEAT_LESION_BALANCED")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT).replace("\\", "/"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan_path = (ROOT / args.config).resolve()
    base_path = (ROOT / args.base_config).resolve()
    output_dir = (ROOT / args.output).resolve()

    plan = read_yaml(plan_path)

    if args.dry_run:
        execution = run_dry_run(
            plan,
            plan_path=plan_path,
            base_config=base_path,
            output_dir=output_dir,
            root=ROOT,
        )
        print(json.dumps(execution, indent=2, ensure_ascii=False))
        return 0 if execution["dry_run_status"] == "READY" else 1

    audit = audit_arm(
        plan, arm_name=args.variant, base_config=base_path, root=ROOT
    )
    if audit["status"] == "BLOCKED":
        raise RunnerError(
            f"Variant {args.variant} is BLOCKED: {audit['blockers']}"
        )
    if not torch_cuda_available():
        raise RunnerError(
            "Real training requires CUDA. Run --dry-run locally; launch "
            "training on the cloud."
        )
    result = launch_training(
        plan=plan,
        plan_path=plan_path,
        base_config=base_path,
        variant=args.variant,
        seed=args.seed,
        output_dir=output_dir,
        root=ROOT,
        resume=args.resume,
    )
    write_json_atomic(output_dir / "resolved_runs.json", {args.variant: result})
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
