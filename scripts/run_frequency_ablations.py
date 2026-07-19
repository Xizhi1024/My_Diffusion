"""Screen residual-frequency blocks, apply artifact gates, and retrain winners.

The runner deliberately launches every promoted model as a fresh training
process with ``resume_from=null`` and a distinct experiment name. It never
continues a short-screen optimizer or scheduler state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import yaml


def metric_value(metrics: Mapping[str, Any], name: str) -> float:
    if name not in metrics:
        raise KeyError(f"Required evaluation metric {name!r} is missing")
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"Required evaluation metric {name!r} is not finite: {value}")
    return value


def passes_hard_gates(
    metrics: Mapping[str, Any],
    reference: Mapping[str, Any],
    gates: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, List[str]]:
    reasons: List[str] = []
    for name, rule in gates.items():
        try:
            candidate = metric_value(metrics, name)
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(str(exc))
            continue

        direction = str(rule.get("direction", "lower"))
        epsilon = float(rule.get("epsilon", 0.0))
        if direction == "lower":
            limits = []
            if "max_value" in rule:
                limits.append(float(rule["max_value"]))
            if "max_delta" in rule:
                try:
                    baseline = metric_value(reference, name)
                    limits.append(baseline + float(rule["max_delta"]))
                except (KeyError, TypeError, ValueError) as exc:
                    reasons.append(str(exc))
            if "max_ratio" in rule:
                try:
                    baseline = metric_value(reference, name)
                    limits.append(baseline * float(rule["max_ratio"]) + epsilon)
                except (KeyError, TypeError, ValueError) as exc:
                    reasons.append(str(exc))
            if not limits:
                raise ValueError(f"Lower-is-better gate {name!r} has no limit")
            limit = min(limits)
            if candidate > limit:
                reasons.append(f"{name}={candidate:.6g} exceeds limit {limit:.6g}")
        elif direction == "higher":
            limits = []
            if "min_value" in rule:
                limits.append(float(rule["min_value"]))
            if "max_delta" in rule:
                try:
                    baseline = metric_value(reference, name)
                    limits.append(baseline - float(rule["max_delta"]))
                except (KeyError, TypeError, ValueError) as exc:
                    reasons.append(str(exc))
            if "min_ratio" in rule:
                try:
                    baseline = metric_value(reference, name)
                    limits.append(baseline * float(rule["min_ratio"]) - epsilon)
                except (KeyError, TypeError, ValueError) as exc:
                    reasons.append(str(exc))
            if not limits:
                raise ValueError(f"Higher-is-better gate {name!r} has no limit")
            limit = max(limits)
            if candidate < limit:
                reasons.append(f"{name}={candidate:.6g} falls below limit {limit:.6g}")
        else:
            raise ValueError(f"Unknown gate direction {direction!r} for {name!r}")
    return not reasons, reasons


def composite_score(metrics: Mapping[str, Any]) -> float:
    """Lesion-heavy score used only after artifact gates have passed."""
    peak = metric_value(metrics, "lesion_peak_error_norm_mean")
    centroid = metric_value(metrics, "lesion_centroid_distance_mean")
    failure = metric_value(metrics, "failure_any_mean")
    hotspot = metric_value(metrics, "false_hotspot_density_mean")
    stripe = max(metric_value(metrics, "stripe_excess_mean"), 0.0)
    ssim = metric_value(metrics, "ssim_mean")
    mae = metric_value(metrics, "mae_mean")
    lesion_boundary = metric_value(
        metrics, "lesion_boundary_gradient_mae_norm_mean"
    )
    anatomy_boundary = metric_value(
        metrics, "anatomy_edge_gradient_mae_norm_mean"
    )
    direction_error = metric_value(
        metrics, "directional_spectrum_error_norm_mean"
    )

    if "lesion_topq_peak_error_norm_mean" in metrics:
        topq_peak = metric_value(metrics, "lesion_topq_peak_error_norm_mean")
        lesion = -(
            topq_peak
            + 0.50 * peak
            + 0.02 * centroid
            + 0.50 * failure
            + 0.20 * lesion_boundary
        )
    else:
        lesion = -(
            peak + 0.02 * centroid + 0.50 * failure + 0.20 * lesion_boundary
        )
    image = (
        ssim
        - mae
        - 0.20 * stripe
        - 100.0 * hotspot
        - 0.10 * anatomy_boundary
        - 0.10 * direction_error
    )
    return 0.70 * lesion + 0.30 * image


def select_promotions(
    records: Sequence[Mapping[str, Any]],
    reference_id: str,
    gates: Mapping[str, Mapping[str, Any]],
    top_k: int,
) -> tuple[List[str], List[Dict[str, Any]]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    reference_row = next((row for row in records if row.get("id") == reference_id), None)
    if reference_row is None:
        raise KeyError(f"Reference variant {reference_id!r} was not evaluated")
    reference_metrics = reference_row["metrics"]

    ranked: List[Dict[str, Any]] = []
    for row in records:
        variant_id = str(row.get("id"))
        if variant_id == reference_id:
            continue
        metrics = row["metrics"]
        gate_passed, reasons = passes_hard_gates(metrics, reference_metrics, gates)
        try:
            score = composite_score(metrics)
        except (KeyError, TypeError, ValueError) as exc:
            gate_passed = False
            reasons = [*reasons, str(exc)]
            score = -1e30
        ranked.append({
            "id": variant_id,
            "preset": row.get("preset"),
            "experiment": row.get("experiment"),
            "checkpoint_epoch": row.get("checkpoint_epoch"),
            "metrics": dict(metrics),
            "gate_passed": gate_passed,
            "gate_reasons": reasons,
            "score": score,
        })
    ranked.sort(key=lambda row: (bool(row["gate_passed"]), float(row["score"])), reverse=True)
    promoted = [row["id"] for row in ranked if row["gate_passed"]][:top_k]
    return promoted, ranked


def build_train_command(
    *,
    python: str,
    config: str,
    ablation_config: str,
    preset: str,
    experiment: str,
    epochs: int,
    seed: int,
    eval_interval: int,
    save_interval: int | None = None,
    sample_interval: int | None = None,
    early_stopping_enabled: bool = False,
    extra_overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> List[str]:
    _save = save_interval if save_interval is not None else eval_interval
    _sample = sample_interval if sample_interval is not None else eval_interval
    overrides = [
        f"experiment.name={experiment}",
        f"experiment.seed={seed}",
        f"training.num_epochs={epochs}",
        "training.resume_from=null",
        "training.init_from=null",
        f"runtime.early_stopping.enabled={'true' if early_stopping_enabled else 'false'}",
        f"runtime.eval_interval={eval_interval}",
        f"runtime.save_interval={_save}",
        f"runtime.sample_interval={_sample}",
    ]
    overrides.extend(_normalise_overrides(extra_overrides))
    command = [
        python,
        "scripts/train_v2.py",
        "--config", config,
        "--ablation", preset,
        "--ablation-config", ablation_config,
    ]
    for override in overrides:
        command.extend(["--override", override])
    return command


def _override_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return yaml.safe_dump(
            value,
            default_flow_style=True,
            sort_keys=False,
        ).strip().replace(", ", ",")
    return str(value)


def _normalise_overrides(
    overrides: Mapping[str, Any] | Sequence[str] | None,
) -> List[str]:
    if overrides is None:
        return []
    if isinstance(overrides, Mapping):
        return [f"{key}={_override_value(value)}" for key, value in overrides.items()]
    return [str(override) for override in overrides]


def _merge_overrides(
    common: Mapping[str, Any] | Sequence[str] | None,
    variant: Mapping[str, Any] | Sequence[str] | None,
) -> Mapping[str, Any] | List[str] | None:
    if common is None:
        return variant
    if variant is None:
        return common
    if isinstance(common, Mapping) and isinstance(variant, Mapping):
        return {**common, **variant}
    return [*_normalise_overrides(common), *_normalise_overrides(variant)]


def build_mean_pretrain_command(
    *,
    python: str,
    config: str,
    output_dir: str,
    epochs: int,
    seed: int,
    overrides: Mapping[str, Any] | Sequence[str] | None = None,
) -> List[str]:
    command = [
        python,
        "scripts/pretrain_conditional_mean.py",
        "--config", config,
        "--output-dir", output_dir,
        "--epochs", str(epochs),
        "--seed", str(seed),
    ]
    for override in _normalise_overrides(overrides):
        command.extend(["--override", override])
    return command


def build_eval_command(
    *,
    python: str,
    config: str,
    checkpoint: str,
    output: str,
    split: str,
    max_samples: int,
    seed: int,
    mc_steps: int,
) -> List[str]:
    return [
        python,
        "scripts/evaluate.py",
        "--config", config,
        "--checkpoint", checkpoint,
        "--weights", "ema",
        "--split", split,
        "--max-samples", str(max_samples),
        "--seed", str(seed),
        "--mc-steps", str(mc_steps),
        "--output", output,
    ]


def _run_manifest_entry(
    plan: Mapping[str, Any],
    variant: Mapping[str, Any],
    settings: Mapping[str, Any],
    python: str,
    phase: str,
) -> Dict[str, Any]:
    variant_id = variant["id"]
    experiment = f"{settings['experiment_prefix']}_{variant_id.lower()}"
    checkpoint_dir = Path("checkpoints") / experiment
    output_dir = Path(str(plan.get("output_dir", "results/frequency_ablations"))) / phase
    checkpoint = checkpoint_dir / "ckpt_best_combined.pt"
    resolved = checkpoint_dir / "resolved_config.yaml"
    result = output_dir / f"{variant_id.lower()}.json"
    completion_checkpoint = None
    if settings.get("require_final_checkpoint", False):
        completion_checkpoint = checkpoint_dir / f"ckpt_epoch{int(settings['epochs']):04d}.pt"
    return {
        "id": variant_id,
        "preset": variant["preset"],
        "experiment": experiment,
        "checkpoint": str(checkpoint),
        "resolved_config": str(resolved),
        "result": str(result),
        "completion_checkpoint": (
            str(completion_checkpoint) if completion_checkpoint is not None else None
        ),
        "train_command": build_train_command(
            python=python,
            config=str(plan["base_config"]),
            ablation_config=str(plan["ablation_config"]),
            preset=variant["preset"],
            experiment=experiment,
            epochs=int(settings["epochs"]),
            seed=int(settings["seed"]),
            eval_interval=int(settings["eval_interval"]),
            save_interval=(
                int(settings["save_interval"])
                if "save_interval" in settings
                else None
            ),
            sample_interval=(
                int(settings["sample_interval"])
                if "sample_interval" in settings
                else None
            ),
            early_stopping_enabled=bool(settings.get("early_stopping", False)),
            extra_overrides=_merge_overrides(
                plan.get("common_train_overrides"), variant.get("overrides")
            ),
        ),
        "eval_command": build_eval_command(
            python=python,
            config=str(resolved),
            checkpoint=str(checkpoint),
            output=str(result),
            split=str(settings.get("split", "val")),
            max_samples=int(settings["max_samples"]),
            seed=int(settings["seed"]),
            mc_steps=int(settings["mc_steps"]),
        ),
    }


def build_execution_manifest(
    plan: Mapping[str, Any],
    *,
    python: str,
    stage: str,
    promoted_ids: Sequence[str] | None = None,
) -> Dict[str, Any]:
    variants = list(plan.get("variants", []))
    mean_run = None
    screen_runs = []
    promotion_runs = []
    mean_settings = plan.get("mean_pretrain", {})
    if (
        stage in {"mean", "all"}
        and isinstance(mean_settings, Mapping)
        and mean_settings.get("enabled", False)
    ):
        experiment = str(mean_settings.get("experiment", "freq_mean_pretrain"))
        output_dir = str(
            mean_settings.get("output_dir", Path("checkpoints") / experiment)
        )
        checkpoint = str(
            mean_settings.get("checkpoint", Path(output_dir) / "mean_best.pt")
        )
        mean_overrides = dict(mean_settings.get("overrides", {}))
        mean_overrides.setdefault("experiment.name", experiment)
        mean_run = {
            "experiment": experiment,
            "output_dir": output_dir,
            "checkpoint": checkpoint,
            "command": build_mean_pretrain_command(
                python=python,
                config=str(plan["base_config"]),
                output_dir=output_dir,
                epochs=int(mean_settings.get("epochs", 30)),
                seed=int(mean_settings.get("seed", 42)),
                overrides=mean_overrides,
            ),
        }
    if stage in {"screen", "all"}:
        screen_runs = [
            _run_manifest_entry(plan, variant, plan["screen"], python, "screen")
            for variant in variants
        ]
    if stage in {"promote", "all"} and promoted_ids:
        by_id = {variant["id"]: variant for variant in variants}
        promotion_runs = [
            _run_manifest_entry(plan, by_id[variant_id], plan["promote"], python, "promote")
            for variant_id in promoted_ids
        ]
    return {
        "stage": stage,
        "mean_run": mean_run,
        "screen_runs": screen_runs,
        "promotion_runs": promotion_runs,
    }


def _execute(command: Sequence[str]) -> None:
    print("\n+ " + subprocess.list2cmdline(list(command)), flush=True)
    subprocess.run(list(command), check=True)


def _run_mean(entry: Mapping[str, Any], force: bool) -> None:
    checkpoint = Path(str(entry["checkpoint"]))
    if force or not checkpoint.exists():
        _execute(entry["command"])
    else:
        print(f"[skip mean] checkpoint exists: {checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Mean pretraining did not produce {checkpoint}"
        )


def _require_planned_mean_checkpoint(plan: Mapping[str, Any]) -> None:
    mean_settings = plan.get("mean_pretrain", {})
    if not isinstance(mean_settings, Mapping):
        return
    if mean_settings.get("enabled", False):
        experiment = str(mean_settings.get("experiment", "freq_mean_pretrain"))
        output_dir = Path(
            str(mean_settings.get("output_dir", Path("checkpoints") / experiment))
        )
        default_checkpoint = output_dir / "mean_best.pt"
    else:
        default_checkpoint = plan.get("common_train_overrides", {}).get(
            "modules.conditional_mean.checkpoint"
        )
    configured = mean_settings.get("checkpoint", default_checkpoint)
    if not configured:
        return
    checkpoint = Path(str(configured))
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Required frozen mean checkpoint is missing: {checkpoint}"
        )


def checkpoint_epoch(path: Path) -> int:
    """Read the actual selected checkpoint epoch for the experiment record."""
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint {path} is not a mapping")
    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise ValueError(f"Checkpoint {path} is missing integer epoch metadata")
    return epoch


def _run_entries(
    entries: Iterable[Mapping[str, Any]],
    force: bool,
    checkpoint_validator: Callable[[Path, Mapping[str, Any]], int | None]
    | None = None,
) -> List[Dict[str, Any]]:
    records = []
    for entry in entries:
        checkpoint = Path(entry["checkpoint"])
        completion_path_value = entry.get("completion_checkpoint")
        completion_checkpoint = (
            Path(str(completion_path_value)) if completion_path_value else None
        )
        result = Path(entry["result"])
        result.parent.mkdir(parents=True, exist_ok=True)
        training_complete = checkpoint.exists() and (
            completion_checkpoint is None or completion_checkpoint.exists()
        )
        if force or not training_complete:
            _execute(entry["train_command"])
        else:
            print(f"[skip train] required checkpoints exist: {checkpoint}")
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Training did not produce {checkpoint}; check eval_interval and best_checkpoint settings"
            )
        if completion_checkpoint is not None and not completion_checkpoint.exists():
            raise FileNotFoundError(
                "Training stopped before the required final checkpoint was written: "
                f"{completion_checkpoint}"
            )
        validated_epoch = None
        if checkpoint_validator is not None:
            validated_epoch = checkpoint_validator(checkpoint, entry)
        if force or not result.exists():
            _execute(entry["eval_command"])
        else:
            print(f"[skip eval] result exists: {result}")
        with result.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        records.append({
            "id": entry["id"],
            "preset": entry["preset"],
            "experiment": entry["experiment"],
            "checkpoint_epoch": (
                validated_epoch
                if validated_epoch is not None
                else checkpoint_epoch(checkpoint)
            ),
            "metrics": metrics,
        })
    return records


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_rankings(output_dir: Path, promoted: Sequence[str], ranked: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = _json_safe({"promoted": list(promoted), "ranked": list(ranked)})
    with (output_dir / "promotion_decision.json").open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, ensure_ascii=False, allow_nan=False)

    metric_names = [
        "lesion_peak_error_norm_mean",
        "lesion_topq_peak_error_norm_mean",
        "lesion_centroid_distance_mean",
        "failure_any_mean",
        "false_hotspot_density_mean",
        "stripe_excess_mean",
        "ssim_mean",
        "mae_mean",
        "lesion_boundary_gradient_mae_norm_mean",
        "anatomy_edge_gradient_mae_norm_mean",
        "directional_spectrum_error_norm_mean",
    ]
    with (output_dir / "leaderboard.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "id", "preset", "checkpoint_epoch", "gate_passed", "score", "gate_reasons", *metric_names],
        )
        writer.writeheader()
        for rank, row in enumerate(ranked, start=1):
            metrics = row["metrics"]
            writer.writerow({
                "rank": rank,
                "id": row["id"],
                "preset": row.get("preset"),
                "checkpoint_epoch": row.get("checkpoint_epoch"),
                "gate_passed": row["gate_passed"],
                "score": row["score"],
                "gate_reasons": "; ".join(row["gate_reasons"]),
                **{name: metrics.get(name) for name in metric_names},
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default="configs/experiments/frequency_ablation_plan.yaml",
        help="Ablation plan YAML",
    )
    parser.add_argument(
        "--stage",
        choices=["mean", "screen", "promote", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run existing train/eval outputs")
    args = parser.parse_args()

    with Path(args.plan).open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    output_dir = Path(str(plan.get("output_dir", "results/frequency_ablations")))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = build_execution_manifest(
            plan, python=sys.executable, stage=args.stage, promoted_ids=None
        )
        destination = output_dir / "dry_run_manifest.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(f"Dry-run manifest saved to {destination}")
        if manifest["mean_run"] is not None:
            print(subprocess.list2cmdline(manifest["mean_run"]["command"]))
        for entry in [*manifest["screen_runs"], *manifest["promotion_runs"]]:
            print(subprocess.list2cmdline(entry["train_command"]))
            print(subprocess.list2cmdline(entry["eval_command"]))
        return

    if args.stage in {"mean", "all"}:
        mean_manifest = build_execution_manifest(
            plan,
            python=sys.executable,
            stage="mean",
        )
        if mean_manifest["mean_run"] is not None:
            _run_mean(mean_manifest["mean_run"], force=args.force)
        if args.stage == "mean":
            return

    promoted: List[str] = []
    if args.stage in {"screen", "all"}:
        _require_planned_mean_checkpoint(plan)
        screen_manifest = build_execution_manifest(plan, python=sys.executable, stage="screen")
        records = _run_entries(screen_manifest["screen_runs"], force=args.force)
        promoted, ranked = select_promotions(
            records,
            reference_id=str(plan.get("reference_id", "R0")),
            gates=plan["hard_gates"],
            top_k=int(plan.get("top_k", 2)),
        )
        _write_rankings(output_dir, promoted, ranked)
        print(f"Promoted variants: {promoted or '(none)'}")

    if args.stage == "promote":
        decision_path = output_dir / "promotion_decision.json"
        if not decision_path.exists():
            raise FileNotFoundError("Run --stage screen first; promotion_decision.json is missing")
        with decision_path.open("r", encoding="utf-8") as handle:
            promoted = list(json.load(handle).get("promoted", []))

    if args.stage in {"promote", "all"}:
        if not promoted:
            print("No variants passed promotion; full retraining was not started.")
            return
        promotion_manifest = build_execution_manifest(
            plan, python=sys.executable, stage="promote", promoted_ids=promoted
        )
        promotion_records = _run_entries(
            promotion_manifest["promotion_runs"], force=args.force
        )
        with (output_dir / "promotion_results.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                _json_safe(promotion_records),
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )


if __name__ == "__main__":
    main()
