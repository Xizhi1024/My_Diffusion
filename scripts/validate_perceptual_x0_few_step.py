#!/usr/bin/env python3
"""Patient-level few-step validation for the perceptual-x0 ablation.

Runs the existing evaluation capability at a fixed sampling seed across
``NFE ∈ {20, 8, 4, 2}`` for a pair of arms, aggregates to patient level, and
applies the paired H6-style statistics (paired_patient_effects,
effect_statistics, calibration_absolute_margin) to decide the primary and
safety gates.

Fail-closed rules:
  * fewer than ``--min-paired-patients`` (default 5) matched patients → FAIL
  * a required endpoint missing from either arm's eval → FAIL/BLOCKED
  * weight selection must happen on calibration; validation only ever runs the
    frozen winner

Outputs under ``--output``:
  patient_effects.csv       per-patient effect rows for every comparison
  calibration_margins.json  frozen non-inferiority margins from calibration
  resolved_runs.json        per-arm × NFE eval provenance (checkpoint sha, seed)
  decision.json             primary + safety gate verdict
  execution_metadata.json   git commit / dirty status / timestamps
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (  # noqa: E402
    canonical_json_sha256,
    file_sha256,
    write_csv,
    write_json,
)
from src.mechanism_validation.model_experiments import (  # noqa: E402
    calibration_absolute_margin,
    effect_statistics,
    paired_patient_effects,
)

SCHEMA_VERSION = 1
PIPELINE_ID = "PFM_LESION_PERCEPTUAL_X0_V1"
EVAL_SEED = 42
DEFAULT_PLAN = Path("configs/experiments/perceptual_x0_ablation_plan_v1.yaml")
DEFAULT_OUTPUT = Path("results/perceptual_x0_few_step_v1")
NFE_GRID = (20, 8, 4, 2)


class ValidatorError(RuntimeError):
    """A fail-closed validator contract violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValidatorError(f"JSON root must be an object: {path}")
    return payload


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_files": [line.split(" ", 2)[-1] for line in status.splitlines()],
    }


def run_evaluate(
    *,
    config: Path,
    checkpoint: Path,
    split: str,
    nfe: int,
    output: Path,
    root: Path,
    seed: int = EVAL_SEED,
    log_path: Path,
) -> None:
    """Invoke scripts/evaluate.py at a fixed seed and step count."""
    command = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--weights",
        "ema",
        "--split",
        split,
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--mc-steps",
        str(nfe),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(__import__("os").environ)
    environment["PYTHONUNBUFFERED"] = "1"
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
            log.write(line)
            log.flush()
        exit_code = process.wait()
    if exit_code:
        raise ValidatorError(
            f"evaluate.py failed with exit {exit_code} for {checkpoint} "
            f"NFE={nfe} split={split}: {log_path}"
        )


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_arm_checkpoints(plan_path: Path, root: Path) -> dict[str, Path]:
    """Resolve each arm's checkpoint from the standard train output location.

    The runner trains into ``checkpoints/perceptual_x0_<ARM>/``; this looks up
    ``ckpt_epochNNNN.pt`` there.  Explicit ``--checkpoint ARM=PATH`` overrides
    are the authoritative source when provided.
    """
    plan = load_plan(plan_path)
    arms = plan.get("arms", {})
    epochs = plan.get("fairness", {}).get("shared_epochs", 300)
    checkpoints: dict[str, Path] = {}
    for arm_name in arms:
        candidate = (
            root
            / "checkpoints"
            / f"perceptual_x0_{arm_name}"
            / f"ckpt_epoch{epochs:04d}.pt"
        )
        if candidate.is_file():
            checkpoints[arm_name] = candidate
    return checkpoints


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidatorError(f"Expected a YAML mapping: {path}")
    return payload


def load_plan(plan_path: Path) -> dict[str, Any]:
    if plan_path.suffix == ".json":
        return load_json(plan_path)
    return _load_yaml(plan_path)


def collect_effects(
    *,
    left_eval: dict[str, Any],
    right_eval: dict[str, Any],
    metric: str,
    lower_is_better: bool,
) -> list[dict[str, Any]]:
    return paired_patient_effects(
        left_eval,
        right_eval,
        metric=metric,
        lower_is_better=lower_is_better,
    )


def freeze_calibration_margins(
    *,
    calibration_effects: Mapping[str, Sequence[Mapping[str, Any]]],
    plan: Mapping[str, Any],
) -> dict[str, float]:
    """Freeze q95 absolute-difference non-inferiority margins on calibration."""
    endpoints = plan.get("evaluation", {}).get(
        "non_inferiority_vs_P0_20", []
    )
    margins: dict[str, float] = {}
    for metric in endpoints:
        effects = calibration_effects.get(metric)
        if not effects:
            raise ValidatorError(
                f"Cannot freeze margin for {metric}: no calibration effects"
            )
        margins[metric] = calibration_absolute_margin(effects)
    return margins


def _verify_endpoints_present(
    eval_payload: dict[str, Any],
    endpoints: Sequence[str],
    context: str,
) -> None:
    """Fail closed when any required endpoint is absent from the eval."""
    per_patient = eval_payload.get("per_patient", {})
    if not per_patient:
        raise ValidatorError(f"{context} evaluation has no per-patient data")
    sample = next(iter(per_patient.values()))
    for metric in endpoints:
        key = metric if metric.endswith("_mean") else f"{metric}_mean"
        if key not in sample:
            raise ValidatorError(
                f"{context} evaluation is missing endpoint {metric!r} "
                f"(key {key!r}); refusing to skip"
            )


def _gate_primary(
    primary_effects: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Primary gate: P3@8 vs P0@8 on the main small-lesion endpoint."""
    stats = effect_statistics(primary_effects, seed=seed)
    alpha = float(plan.get("evaluation", {}).get("alpha", 0.05))
    positive = bool(stats["ci95_low"] > 0.0 and stats["sign_flip_p"] < alpha)
    return {
        "endpoint": plan.get("evaluation", {}).get(
            "primary_endpoint", "small_lesion_topq_peak_error_norm"
        ),
        "paired_patients": stats["patients"],
        "median_effect": stats["estimate"],
        "ci95": [stats["ci95_low"], stats["ci95_high"]],
        "sign_flip_p": stats["sign_flip_p"],
        "alpha": alpha,
        "pass": positive,
        "rule": "positive effect with 95% CI lower bound > 0 and p < alpha",
    }


def _gate_non_inferiority(
    *,
    ni_effects: Mapping[str, Sequence[Mapping[str, Any]]],
    margins: Mapping[str, float],
    plan: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Safety gate: P3@8 vs P0@20 must stay within frozen calibration margins.

    ``paired_patient_effects`` always returns ``effect_left_better`` — positive
    when the left arm (P3) is better than the right arm (P0) — for every
    endpoint, including higher-is-better ones like ssim.  So the fail-closed
    non-inferiority condition is the same in both cases: the 95% CI LOWER bound
    of the effect must not fall below ``-margin``.  Using the upper bound
    (the original bug) was optimistic and let a genuinely worse arm pass when
    its CI spanned ``-margin``.
    """
    endpoints = list(margins)
    results: dict[str, Any] = {}
    for metric in endpoints:
        effects = ni_effects.get(metric)
        if not effects:
            raise ValidatorError(f"No non-inferiority effects for {metric}")
        stats = effect_statistics(effects, seed=seed)
        margin = margins[metric]
        lower = stats["ci95_low"]
        upper = stats["ci95_high"]
        lower_is_better = _endpoint_direction(metric)
        # P3 is worse when its effect_left_better goes negative beyond -margin.
        # Guard the CI lower bound (the pessimistic side) in both directions.
        within = bool(lower >= -margin)
        results[metric] = {
            "margin": margin,
            "effect_ci95": [lower, upper],
            "lower_is_better": lower_is_better,
            "pass": within,
        }
    all_pass = all(item["pass"] for item in results.values())
    return {"endpoints": results, "pass": all_pass}


def _gate_learned_representation(
    effects: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """P3@8 vs P4@8: P3 must beat the random-encoder control."""
    stats = effect_statistics(effects, seed=seed)
    alpha = float(plan.get("evaluation", {}).get("alpha", 0.05))
    pass_ = bool(stats["ci95_low"] > 0.0 and stats["sign_flip_p"] < alpha)
    return {
        "control": "P4_FEAT_RANDOM",
        "paired_patients": stats["patients"],
        "ci95": [stats["ci95_low"], stats["ci95_high"]],
        "sign_flip_p": stats["sign_flip_p"],
        "pass": pass_,
    }


def _resolve_arm(plan: Mapping[str, Any], short: str) -> str:
    """Map a short arm label (P0/P1/P2/P3/P4) to its full arm name."""
    arms = plan.get("arms", {})
    if short in arms:
        return short
    prefix = f"{short}_"
    matches = [name for name in arms if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValidatorError(
            f"Cannot resolve arm label {short!r} to a unique arm in {sorted(arms)}"
        )
    return matches[0]


def _safety_delta_gate(
    *,
    p3_8: dict[str, Any],
    p0_8: dict[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """P3@8 must not add small-lesion underestimate or false-hotspot failures."""
    per_patient = p3_8.get("per_patient", {})
    baseline = p0_8.get("per_patient", {})
    underestimate_metric = "small_lesion_underestimate_mean"
    hotspot_metric = "target_relative_false_hotspot_density_mean"
    extra_underestimate = 0
    extra_hotspot = 0
    for patient in per_patient:
        if patient not in baseline:
            continue
        row = per_patient[patient]
        base_row = baseline[patient]
        for key in (underestimate_metric, hotspot_metric):
            if key not in row or key not in base_row:
                raise ValidatorError(f"Missing safety endpoint {key} for patient {patient}")
        extra_underestimate += max(
            0.0, float(row[underestimate_metric]) - float(base_row[underestimate_metric])
        )
        extra_hotspot += max(
            0.0, float(row[hotspot_metric]) - float(base_row[hotspot_metric])
        )
    pass_ = bool(extra_underestimate == 0.0 and extra_hotspot == 0.0)
    return {
        "extra_small_lesion_underestimate": extra_underestimate,
        "extra_false_hotspot_density": extra_hotspot,
        "pass": pass_,
    }


# Endpoints where a HIGHER value is better (e.g. SSIM).  Everything not listed
# here is lower-is-better, which is the default direction.
HIGHER_IS_BETTER_ENDPOINTS = ("ssim",)


def _endpoint_direction(metric: str) -> bool:
    """Return ``lower_is_better`` for a metric name.

    ``ssim`` is higher-is-better: a larger SSIM means the prediction is closer
    to the target.  Using ``lower_is_better=True`` for it would sign the paired
    effect backwards and silently invert the non-inferiority gate.
    """
    return metric not in HIGHER_IS_BETTER_ENDPOINTS


def derive_mechanism_partition(
    manifest_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, str], dict[str, dict[str, list[str]]]]:
    """Derive the locked calibration/validation partition from the manifest.

    Reuses ``patient_partition`` (seed 42, 20% calibration) so calibration is a
    deterministic subset of the training patients and validation is exactly the
    held-out ``val`` patients.  Returns (patient->role, per-role sample lists).
    """
    from src.mechanism_validation.common import patient_partition as _partition

    rows = []
    manifest_path = Path(manifest_path)
    path = manifest_path if manifest_path.is_absolute() else (root / manifest_path).resolve()
    if not path.is_file():
        raise ValidatorError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    partition = _partition(rows)
    samples_by_role: dict[str, dict[str, list[str]]] = {
        role: {"patients": [], "samples": []} for role in ("calibration", "validation")
    }
    for row in rows:
        role = partition.get(row["patient_id"])
        if role in ("calibration", "validation"):
            samples_by_role[role]["samples"].append(row["sample_id"])
    for patient, role in partition.items():
        if role in ("calibration", "validation"):
            samples_by_role[role]["patients"].append(patient)
    return partition, samples_by_role


def _filter_eval_to_partition(
    payload: dict[str, Any],
    patient_ids: Sequence[str],
) -> dict[str, Any]:
    """Restrict an eval payload's per_patient records to a patient list.

    evaluate.py runs on a whole split (e.g. all train patients); the mechanism
    calibration partition is a deterministic subset of those, so the validator
    filters to exactly the calibration patient_ids before computing effects.
    """
    allowed = set(patient_ids)
    per_patient = payload.get("per_patient", {})
    filtered = {pid: row for pid, row in per_patient.items() if pid in allowed}
    out = dict(payload)
    out["per_patient"] = filtered
    out["num_patients"] = len(filtered)
    return out


def run_validation(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    checkpoints: Mapping[str, Path],
    config_for: dict[str, Path],
    output_dir: Path,
    root: Path,
    manifest_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run evaluations, aggregate, apply gates, and persist all artifacts."""
    nfe_grid = tuple(int(n) for n in plan.get("evaluation", {}).get("nfe_grid", NFE_GRID))
    primary = plan.get("evaluation", {}).get("primary_endpoint", "small_lesion_topq_peak_error_norm")
    primary_cmp = plan.get("evaluation", {}).get("primary_comparison", "P3@8 vs P0@8")
    primary_short = primary_cmp.split("@")[0].strip()          # P3
    primary_arm = _resolve_arm(plan, primary_short)
    primary_nfe = int(primary_cmp.split("@")[1].split()[0])  # 8
    p0_arm = _resolve_arm(plan, "P0")
    p0_nfe = 20
    p4_arm = _resolve_arm(plan, "P4")
    min_paired = int(plan.get("evaluation", {}).get("minimum_paired_patients", 5))
    ni_endpoints = list(plan.get("evaluation", {}).get("non_inferiority_vs_P0_20", []))

    # The manifest drives the partition: calibration = deterministic subset of
    # training patients, validation = held-out val patients.  evaluate.py is
    # told which split label to use; the partition is verified per-patient below.
    _, samples_by_role = derive_mechanism_partition(manifest_path, root=root)
    split_cal = "train"
    split_val = "val"

    # ---- Resolve eval provenance records ----
    evals: dict[tuple[str, str], dict[str, Any]] = {}  # (arm, nfe) -> eval payload
    provenance: dict[str, Any] = {}
    for arm_name, ckpt in checkpoints.items():
        config = config_for[arm_name]
        for nfe in nfe_grid:
            for role, split in (("calibration", split_cal), ("validation", split_val)):
                output = output_dir / "evaluations" / f"{arm_name}" / f"nfe{nfe:02d}" / f"{role}.json"
                prov_path = output.with_name(f"{role}_provenance.json")
                expected_prov = {
                    "schema_version": SCHEMA_VERSION,
                    "arm": arm_name,
                    "nfe": nfe,
                    "split": split,
                    "checkpoint": ckpt.as_posix(),
                    "checkpoint_sha256": file_sha256(ckpt),
                    "config": config.as_posix(),
                    "config_sha256": file_sha256(config),
                    "eval_seed": EVAL_SEED,
                    "weights": "ema",
                }
                reusable = (
                    output.is_file()
                    and prov_path.is_file()
                    and load_json(prov_path) == expected_prov
                )
                if not reusable or force:
                    run_evaluate(
                        config=config,
                        checkpoint=ckpt,
                        split=split,
                        nfe=nfe,
                        output=output,
                        root=root,
                        log_path=output_dir / "logs" / f"{arm_name}_nfe{nfe:02d}_{role}.log",
                    )
                    write_json(prov_path, expected_prov)
                payload = load_json(output)
                if int(payload.get("num_patients", 0)) <= 0:
                    raise ValidatorError(
                        f"{arm_name} NFE={nfe} {role} evaluation has no patients"
                    )
                payload = _filter_eval_to_partition(
                    payload,
                    samples_by_role[role]["patients"],
                )
                if int(payload.get("num_patients", 0)) < min_paired:
                    raise ValidatorError(
                        f"{arm_name} NFE={nfe} {role} evaluation has only "
                        f"{payload.get('num_patients')} partition patients; "
                        f"need >= {min_paired}"
                    )
                evals[(arm_name, nfe, role)] = payload
                provenance[f"{arm_name}_nfe{nfe:02d}_{role}"] = expected_prov

    # ---- Verify required endpoints ----
    endpoints = [primary] + list(
        plan.get("evaluation", {}).get("non_inferiority_vs_P0_20", [])
    )
    for key, payload in evals.items():
        _verify_endpoints_present(payload, endpoints, context=f"{key}")

    # ---- Patient-level effects ----
    all_effects: list[dict[str, Any]] = []
    primary_effects_cal: list[Mapping[str, Any]] = collect_effects(
        left_eval=evals[(primary_arm, primary_nfe, "calibration")],
        right_eval=evals[(p0_arm, primary_nfe, "calibration")],
        metric=primary,
        lower_is_better=_endpoint_direction(primary),
    )
    calibration_effects: dict[str, list[Mapping[str, Any]]] = {}
    ni_effects_val: dict[str, list[Mapping[str, Any]]] = {}
    for metric in ni_endpoints:
        lower_is_better = _endpoint_direction(metric)
        calibration_effects[metric] = collect_effects(
            left_eval=evals[(primary_arm, primary_nfe, "calibration")],
            right_eval=evals[(p0_arm, p0_nfe, "calibration")],
            metric=metric,
            lower_is_better=lower_is_better,
        )
        ni_effects_val[metric] = collect_effects(
            left_eval=evals[(primary_arm, primary_nfe, "validation")],
            right_eval=evals[(p0_arm, p0_nfe, "validation")],
            metric=metric,
            lower_is_better=lower_is_better,
        )
    learned_effects_val = collect_effects(
        left_eval=evals[(primary_arm, primary_nfe, "validation")],
        right_eval=evals[(p4_arm, primary_nfe, "validation")],
        metric=primary,
        lower_is_better=_endpoint_direction(primary),
    )
    primary_effects_val = collect_effects(
        left_eval=evals[(primary_arm, primary_nfe, "validation")],
        right_eval=evals[(p0_arm, primary_nfe, "validation")],
        metric=primary,
        lower_is_better=_endpoint_direction(primary),
    )

    all_effects.extend(
        {"comparison": "primary_calibration", **row} for row in primary_effects_cal
    )
    all_effects.extend(
        {"comparison": "primary_validation", **row} for row in primary_effects_val
    )
    all_effects.extend(
        {"comparison": "learned_representation_validation", **row}
        for row in learned_effects_val
    )
    for metric, rows in ni_effects_val.items():
        all_effects.extend(
            {"comparison": f"non_inferiority_{metric}_validation", **row}
            for row in rows
        )
    write_csv(output_dir / "patient_effects.csv", all_effects)

    # ---- Gates ----
    if len(primary_effects_cal) < min_paired:
        raise ValidatorError(
            f"Only {len(primary_effects_cal)} calibration paired patients; "
            f"need >= {min_paired}"
        )
    margins = freeze_calibration_margins(
        calibration_effects=calibration_effects, plan=plan
    )
    write_json(output_dir / "calibration_margins.json", margins)

    if len(primary_effects_val) < min_paired:
        raise ValidatorError(
            f"Only {len(primary_effects_val)} validation paired patients; "
            f"need >= {min_paired}"
        )
    primary_gate = _gate_primary(primary_effects_val, plan, seed=EVAL_SEED)
    ni_gate = _gate_non_inferiority(
        ni_effects=ni_effects_val,
        margins=margins,
        plan=plan,
        seed=EVAL_SEED + 1,
    )
    learned_gate = _gate_learned_representation(
        learned_effects_val, plan, seed=EVAL_SEED + 2
    )
    safety_delta = _safety_delta_gate(
        p3_8=evals[(primary_arm, primary_nfe, "validation")],
        p0_8=evals[(p0_arm, primary_nfe, "validation")],
        plan=plan,
    )

    passed = (
        primary_gate["pass"]
        and ni_gate["pass"]
        and learned_gate["pass"]
        and safety_delta["pass"]
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "decision": "PASS" if passed else "FAIL",
        "primary": primary_gate,
        "non_inferiority_vs_P0_20": ni_gate,
        "learned_representation_control": learned_gate,
        "safety_delta_vs_P0_8": safety_delta,
        "intersection_rule": [
            "P3@8 better than P0@8 on primary",
            "P3@8 non-inferior to P0@20 on all safety endpoints",
            "P3@8 beats P4@8 (learned representation)",
            "P3@8 adds no small-lesion underestimate / false-hotspot",
        ],
    }
    write_json(output_dir / "decision.json", decision)

    resolved_runs = {
        arm: {
            "checkpoint": ckpt.as_posix(),
            "checkpoint_sha256": file_sha256(ckpt),
            "config": config_for[arm].as_posix(),
            "evaluations": {
                "calibration": {str(n): evals[(arm, n, "calibration")].get("num_patients", 0) for n in nfe_grid},
                "validation": {str(n): evals[(arm, n, "validation")].get("num_patients", 0) for n in nfe_grid},
            },
        }
        for arm, ckpt in checkpoints.items()
    }
    write_json(output_dir / "resolved_runs.json", resolved_runs)

    execution = {
        "schema_version": SCHEMA_VERSION,
        "pipeline_id": PIPELINE_ID,
        "plan": plan_path.as_posix(),
        "plan_sha256": file_sha256(plan_path),
        "eval_seed": EVAL_SEED,
        "nfe_grid": list(nfe_grid),
        "git": git_info(root),
        "completed_utc": utc_now(),
    }
    write_json(output_dir / "execution_metadata.json", execution)
    return decision


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN).replace("\\", "/"))
    parser.add_argument("--manifest", default=str(Path("main_data/split_manifest.csv")).replace("\\", "/"))
    parser.add_argument("--checkpoint", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--config", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT).replace("\\", "/"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan_path = (ROOT / args.plan).resolve()
    output_dir = (ROOT / args.output).resolve()
    manifest_path = (ROOT / args.manifest).resolve()
    plan = load_plan(plan_path)

    checkpoints: dict[str, Path] = {}
    config_for: dict[str, Path] = {}
    for pair in args.checkpoint:
        arm, _, path = pair.partition("=")
        if not arm or not path:
            parser.error(f"--checkpoint must be ARM=PATH, got {pair!r}")
        checkpoints[arm] = _resolve_path(ROOT, path)
    for pair in args.config:
        arm, _, path = pair.partition("=")
        if not arm or not path:
            parser.error(f"--config must be ARM=PATH, got {pair!r}")
        config_for[arm] = _resolve_path(ROOT, path)
    if not checkpoints:
        checkpoints = load_arm_checkpoints(plan_path, ROOT)
    for arm_name in checkpoints:
        if arm_name not in config_for:
            default = (
                ROOT
                / "configs"
                / "experiments"
                / f"perceptual_x0_{arm_name}.yaml"
            )
            if default.is_file():
                config_for[arm_name] = default
            else:
                raise ValidatorError(
                    f"No resolved model config found for arm {arm_name}. "
                    f"Run scripts/run_perceptual_x0_ablation.py --dry-run "
                    f"--output {output_dir} first, or pass --config {arm_name}=PATH. "
                    "Refusing to pass the experiment plan YAML as a model config."
                )

    missing = [
        arm for arm, ckpt in checkpoints.items() if not ckpt.is_file()
    ]
    if missing:
        raise ValidatorError(
            f"Missing arm checkpoints: {missing}. Pass --checkpoint ARM=PATH."
        )

    decision = run_validation(
        plan=plan,
        plan_path=plan_path,
        checkpoints=checkpoints,
        config_for=config_for,
        output_dir=output_dir,
        root=ROOT,
        manifest_path=manifest_path,
        force=args.force,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    return 0 if decision["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
