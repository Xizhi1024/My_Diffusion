"""Exact same-checkpoint oracle over per-level/per-band route actions.

The six action slots are::

    L2-LH, L2-HL, L2-HH, L1-LH, L1-HL, L1-HH

For every complete action map, this script resets the evaluation RNG, samples
the same checkpoint on the same ordered dataset, and records per-sample
metrics.  It then selects the best complete action map independently for each
sample using a predeclared target-based objective.

With the default ``native,shallow,null`` choices, the exact search contains
``3**6 = 729`` complete action maps.  This is an oracle upper-bound analysis:
the target PET is used only after inference to select a route map, so the
selected route is not deployable at inference.

``null`` changes active availability as well as destination.  To isolate only
native-vs-shallow destination while preserving non-null operation, run with
``--choices native,shallow`` (``2**6 = 64`` maps).

Example::

    pixi run python -u -m scripts.eval_oracle_destination `
      --config configs/experiments/slmf_png_prior_anchored_router_identifiable_full_100e.yaml `
      --checkpoint results/prior_anchored_router_identifiable_full_100e/checkpoints/ckpt_best_combined.pt `
      --split val --max-samples 64 --steps 50 `
      --objective lesion_topq_peak_error_norm `
      --choices native,shallow,null `
      --output-dir results/oracle_destination/identifiable_full_e100

The run is resumable.  Each completed action map is stored under
``variants/`` and is skipped on the next invocation when ``--resume`` is set.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_router_background_suite import (  # noqa: E402
    _json_safe,
    _load_dataset,
    _numeric_summary,
    _paired_effect,
    _seed_evaluation,
    _select_checkpoint_state,
    _sha256,
    _write_csv,
    _write_json,
    evaluate_variant,
)
from src.data.lineage import (  # noqa: E402
    load_checkpoint_data_lineage,
    validate_checkpoint_data_lineage,
)
from src.model.config_utils import load_full_config, resolve_runtime_profile  # noqa: E402
from src.model.slmf_bbdm import SLMFBBDM  # noqa: E402


SLOTS = (
    "l2_lh",
    "l2_hl",
    "l2_hh",
    "l1_lh",
    "l1_hl",
    "l1_hh",
)
ALLOWED_ACTIONS = ("native", "shallow", "null")
ACTION_CODE = {
    "native": "N",
    "shallow": "S",
    "null": "0",
}


def _parse_choices(value: str) -> tuple[str, ...]:
    choices = tuple(
        item.strip().lower()
        for item in str(value).split(",")
        if item.strip()
    )
    if not choices:
        raise ValueError("--choices must contain at least one route action")
    if len(set(choices)) != len(choices):
        raise ValueError("--choices must not contain duplicates")
    unknown = sorted(set(choices).difference(ALLOWED_ACTIONS))
    if unknown:
        raise ValueError(
            f"Unknown oracle actions {unknown}; expected actions from "
            f"{list(ALLOWED_ACTIONS)}"
        )
    return choices


def enumerate_action_maps(
    choices: Sequence[str],
) -> list[tuple[str, ...]]:
    normalized = _parse_choices(",".join(str(item) for item in choices))
    return list(itertools.product(normalized, repeat=len(SLOTS)))


def action_map_code(actions: Sequence[str]) -> str:
    if len(actions) != len(SLOTS):
        raise ValueError(f"action map must contain {len(SLOTS)} actions")
    encoded = []
    for action in actions:
        normalized = str(action).strip().lower()
        if normalized not in ACTION_CODE:
            raise ValueError(f"Unknown action {action!r}")
        encoded.append(ACTION_CODE[normalized])
    return "".join(encoded[:3]) + "-" + "".join(encoded[3:])


def action_map_payload(actions: Sequence[str]) -> dict[str, str]:
    if len(actions) != len(SLOTS):
        raise ValueError(f"action map must contain {len(SLOTS)} actions")
    return {
        slot: str(action).strip().lower()
        for slot, action in zip(SLOTS, actions)
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            rows.append(payload)
    return rows


def _finite_metric(row: Mapping[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"sample {row.get('sample_id')} has no numeric metric {metric!r}"
        )
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(
            f"sample {row.get('sample_id')} has non-finite metric {metric!r}"
        )
    return converted


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError(f"{label} contains a row without sample_id")
        if sample_id in indexed:
            raise ValueError(f"{label} repeats sample_id {sample_id!r}")
        indexed[sample_id] = dict(row)
    return indexed


def select_oracle_rows(
    baseline_rows: Sequence[Mapping[str, Any]],
    variant_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    action_maps: Mapping[str, Sequence[str]],
    *,
    objective: str,
    lower_is_better: bool,
    learned_fallback: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select the best complete route map independently for every sample."""

    baseline = _index_rows(baseline_rows, label="learned baseline")
    best: dict[str, tuple[float, str, dict[str, Any]]] = {}
    expected_ids = set(baseline)
    for code in sorted(variant_rows):
        if code not in action_maps:
            raise ValueError(f"missing action map for variant {code!r}")
        indexed = _index_rows(variant_rows[code], label=f"variant {code}")
        if set(indexed) != expected_ids:
            missing = sorted(expected_ids.difference(indexed))
            extra = sorted(set(indexed).difference(expected_ids))
            raise ValueError(
                f"variant {code} sample identity mismatch: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        for sample_id, row in indexed.items():
            value = _finite_metric(row, objective)
            current = best.get(sample_id)
            better = (
                current is None
                or (value < current[0] if lower_is_better else value > current[0])
            )
            if better:
                best[sample_id] = (value, code, row)

    if not best:
        raise ValueError("no oracle variants were provided")

    forced_rows: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []
    for sample_id in baseline:
        baseline_row = baseline[sample_id]
        baseline_value = _finite_metric(baseline_row, objective)
        forced_value, code, selected = best[sample_id]
        actions = action_map_payload(action_maps[code])
        forced = dict(selected)
        forced.update(
            {
                "oracle_selected_code": code,
                "oracle_used_learned_fallback": 0.0,
                "oracle_objective": objective,
                "oracle_objective_value": forced_value,
                "learned_objective_value": baseline_value,
                "oracle_objective_improvement": (
                    baseline_value - forced_value
                    if lower_is_better
                    else forced_value - baseline_value
                ),
                **{
                    f"oracle_action_{slot}": action
                    for slot, action in actions.items()
                },
            }
        )
        forced_rows.append(forced)

        forced_better = (
            forced_value <= baseline_value
            if lower_is_better
            else forced_value >= baseline_value
        )
        if learned_fallback and not forced_better:
            fallback = dict(baseline_row)
            fallback.update(
                {
                    "oracle_selected_code": "LEARNED",
                    "oracle_used_learned_fallback": 1.0,
                    "oracle_objective": objective,
                    "oracle_objective_value": baseline_value,
                    "learned_objective_value": baseline_value,
                    "oracle_objective_improvement": 0.0,
                    **{
                        f"oracle_action_{slot}": "learned"
                        for slot in SLOTS
                    },
                }
            )
        else:
            fallback = dict(forced)
        fallback_rows.append(fallback)
    return forced_rows, fallback_rows


def _selection_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    combination_counts = Counter(
        str(row["oracle_selected_code"])
        for row in rows
    )
    slot_counts = {
        slot: Counter(
            str(row[f"oracle_action_{slot}"])
            for row in rows
        )
        for slot in SLOTS
    }
    total = max(len(rows), 1)
    return {
        "combination_counts": dict(sorted(combination_counts.items())),
        "combination_fractions": {
            key: count / total
            for key, count in sorted(combination_counts.items())
        },
        "slot_counts": {
            slot: dict(sorted(counts.items()))
            for slot, counts in slot_counts.items()
        },
        "slot_fractions": {
            slot: {
                action: count / total
                for action, count in sorted(counts.items())
            }
            for slot, counts in slot_counts.items()
        },
    }


def _load_or_evaluate(
    *,
    path: Path,
    resume: bool,
    model: SLMFBBDM,
    loader: DataLoader,
    eval_kwargs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    if resume and path.is_file():
        return _read_jsonl(path), True
    rows, _ = evaluate_variant(
        model,
        loader,
        **eval_kwargs,
        grid_samples=0,
        verbose=False,
    )
    _write_jsonl(path, rows)
    return rows, False


def _manifest_identity(args: argparse.Namespace, choices: Sequence[str]) -> dict[str, Any]:
    config = Path(args.config)
    checkpoint = Path(args.checkpoint)
    return {
        "schema_version": 1,
        "scope": "target_selected_exact_route_action_oracle",
        "config": str(config),
        "config_sha256": _sha256(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "weights": args.weights,
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "seed": args.seed,
        "choices": list(choices),
        "slots": list(SLOTS),
        "objective": args.objective,
        "lower_is_better": not args.maximize_objective,
        "frequency_mode": args.frequency_mode,
    }


def _validate_resume_manifest(path: Path, identity: Mapping[str, Any]) -> None:
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8-sig"))
    existing_identity = existing.get("identity")
    if existing_identity != identity:
        raise RuntimeError(
            "Existing oracle output has a different run identity. "
            "Use another --output-dir or remove the partial run explicitly."
        )


def _resolve_conditional_mean_checkpoint(
    config: dict[str, Any],
    *,
    evaluated_checkpoint_path: Path,
    evaluated_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Make a full-model evaluation checkpoint self-contained when possible."""

    modules = config.get("modules")
    if not isinstance(modules, dict):
        return {"mode": "not_configured"}
    mean = modules.get("conditional_mean")
    if not isinstance(mean, dict) or not bool(mean.get("enabled", False)):
        return {"mode": "disabled"}
    configured_value = mean.get("checkpoint")
    if not configured_value:
        return {"mode": "not_configured"}

    configured_path = Path(str(configured_value))
    candidates = (configured_path, ROOT / configured_path)
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            mean["checkpoint"] = str(resolved)
            return {
                "mode": "configured_checkpoint",
                "configured_path": str(configured_value),
                "resolved_path": str(resolved),
            }

    state = evaluated_checkpoint.get("model")
    if not isinstance(state, Mapping) or not any(
        str(key).startswith("mean_predictor.")
        for key in state
    ):
        raise FileNotFoundError(
            "Configured conditional-mean checkpoint does not exist and the "
            "evaluated checkpoint contains no mean_predictor.* tensors: "
            f"{configured_path}"
        )

    resolved = evaluated_checkpoint_path.resolve()
    mean["checkpoint"] = str(resolved)
    mean["checkpoint_format"] = "full_model"
    return {
        "mode": "evaluated_checkpoint_self_bootstrap",
        "configured_path": str(configured_value),
        "resolved_path": str(resolved),
    }


def _main(args: argparse.Namespace) -> int:
    choices = _parse_choices(args.choices)
    action_maps = enumerate_action_maps(choices)
    if len(action_maps) > args.max_combinations:
        raise ValueError(
            f"search contains {len(action_maps)} combinations, exceeding "
            f"--max-combinations={args.max_combinations}"
        )
    print(
        f"Exact oracle search: {len(action_maps)} complete maps "
        f"({len(choices)}^{len(SLOTS)})",
        flush=True,
    )
    if args.plan_only:
        for index, actions in enumerate(action_maps):
            print(
                f"{index:04d} {action_map_code(actions)} "
                f"{action_map_payload(actions)}"
            )
        return 0

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _manifest_identity(args, choices)
    manifest_path = output_dir / "run_manifest.json"
    _validate_resume_manifest(manifest_path, identity)
    _write_json(
        manifest_path,
        {
            "identity": identity,
            "num_combinations": len(action_maps),
            "oracle_scope": (
                "per-sample target-selected best complete six-slot action map; "
                "actions are constant across diffusion timesteps and pixels"
            ),
            "warning": (
                "This is an upper-bound analysis and is not deployable: target "
                "PET selects the route after inference."
            ),
            "null_semantics": (
                "null sets exact [native, shallow, null]=[0,0,1] and therefore "
                "changes availability; omit null for destination-only search"
            ),
        },
    )

    config = resolve_runtime_profile(load_full_config(str(config_path)))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_evaluation(args.seed)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Evaluation checkpoint must contain a mapping")
    mean_resolution = _resolve_conditional_mean_checkpoint(
        config,
        evaluated_checkpoint_path=checkpoint_path,
        evaluated_checkpoint=checkpoint,
    )
    validate_checkpoint_data_lineage(
        checkpoint,
        load_checkpoint_data_lineage(config),
        required=bool(
            config.get("data", {}).get("require_cache_lineage", False)
        ),
        context=f"oracle destination checkpoint {checkpoint_path}",
    )
    print(
        "Building model "
        f"(conditional mean: {mean_resolution['mode']})...",
        flush=True,
    )
    model = SLMFBBDM.from_config(config)
    state, state_source = _select_checkpoint_state(
        checkpoint,
        weights=args.weights,
    )
    model.load_state_dict(state)
    model = model.to(device)
    router = getattr(model, "residual_preconditioner", None)
    if router is None or not hasattr(
        router,
        "set_inference_route_action_intervention",
    ):
        raise RuntimeError(
            "Checkpoint model has no per-band route-action intervention API"
        )
    if hasattr(router, "set_inference_destination_intervention"):
        router.set_inference_destination_intervention("learned")
    if hasattr(router, "set_inference_frequency_intervention"):
        router.set_inference_frequency_intervention(args.frequency_mode)

    dataset, selected_indices = _load_dataset(
        config,
        split=args.split,
        max_samples=args.max_samples,
        allow_train_fallback=args.allow_train_fallback,
    )
    data_config = config.get("data", {})
    batch_size = int(
        args.batch_size
        or data_config.get(
            "val_batch_size",
            data_config.get("batch_size", 4),
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest.update(
        {
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint_weights": state_source,
            "selected_indices": selected_indices,
            "num_samples": len(dataset),
            "resolved_batch_size": batch_size,
            "device": device,
            "conditional_mean_resolution": mean_resolution,
        }
    )
    _write_json(manifest_path, manifest)

    eval_kwargs = {
        "device": device,
        "amp": not args.no_amp,
        "steps": args.steps,
        "seed": args.seed,
        "lowpass_sigma": args.lowpass_sigma,
        "body_threshold": args.body_threshold,
        "lesion_exclusion_radius": args.lesion_exclusion_radius,
        "hotspot_hit_radius": args.hotspot_hit_radius,
        "recovery_ratio": args.recovery_ratio,
        "small_lesion_quantile": args.small_lesion_quantile,
        "small_lesion_underestimate_tolerance": (
            args.small_lesion_underestimate_tolerance
        ),
    }
    started = time.perf_counter()
    router.set_inference_route_action_intervention()
    print("Evaluating learned baseline...", flush=True)
    baseline_rows, baseline_cached = _load_or_evaluate(
        path=output_dir / "learned_baseline.jsonl",
        resume=args.resume,
        model=model,
        loader=loader,
        eval_kwargs=eval_kwargs,
    )
    print(
        "  loaded cached baseline"
        if baseline_cached
        else "  completed learned baseline",
        flush=True,
    )

    variant_dir = output_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)
    rows_by_code: dict[str, list[dict[str, Any]]] = {}
    action_by_code: dict[str, tuple[str, ...]] = {}
    variant_summaries: list[dict[str, Any]] = []
    completed_fresh = 0
    for index, actions in enumerate(action_maps, start=1):
        code = action_map_code(actions)
        action_by_code[code] = tuple(actions)
        router.set_inference_route_action_intervention(
            actions_l2=actions[:3],
            actions_l1=actions[3:],
        )
        rows, cached = _load_or_evaluate(
            path=variant_dir / f"{code}.jsonl",
            resume=args.resume,
            model=model,
            loader=loader,
            eval_kwargs=eval_kwargs,
        )
        rows_by_code[code] = rows
        summary = _numeric_summary(rows)
        summary.update(
            {
                "combination_index": index - 1,
                "code": code,
                "cached": cached,
                **action_map_payload(actions),
            }
        )
        variant_summaries.append(summary)
        if not cached:
            completed_fresh += 1
        elapsed = time.perf_counter() - started
        fresh_rate = elapsed / max(completed_fresh, 1)
        remaining = sum(
            not (variant_dir / f"{action_map_code(item)}.jsonl").is_file()
            for item in action_maps[index:]
        )
        eta = fresh_rate * remaining if completed_fresh else 0.0
        print(
            f"[{index:03d}/{len(action_maps):03d}] {code} "
            f"{'cached' if cached else 'done'}; ETA {eta / 60.0:.1f} min",
            flush=True,
        )

    forced_rows, fallback_rows = select_oracle_rows(
        baseline_rows,
        rows_by_code,
        action_by_code,
        objective=args.objective,
        lower_is_better=not args.maximize_objective,
        learned_fallback=True,
    )
    _write_jsonl(output_dir / "oracle_forced_samples.jsonl", forced_rows)
    _write_jsonl(output_dir / "oracle_with_learned_fallback_samples.jsonl", fallback_rows)
    _write_csv(output_dir / "variant_summary.csv", variant_summaries)
    _write_csv(output_dir / "oracle_forced_samples.csv", forced_rows)
    _write_csv(
        output_dir / "oracle_with_learned_fallback_samples.csv",
        fallback_rows,
    )

    lower_is_better = not args.maximize_objective
    forced_effect = _paired_effect(
        forced_rows,
        baseline_rows,
        metric=args.objective,
        lower_is_better=lower_is_better,
        seed=args.seed + 7001,
    )
    fallback_effect = _paired_effect(
        fallback_rows,
        baseline_rows,
        metric=args.objective,
        lower_is_better=lower_is_better,
        seed=args.seed + 7002,
    )
    total_seconds = time.perf_counter() - started
    summary = {
        "manifest": manifest,
        "baseline": _numeric_summary(baseline_rows),
        "forced_oracle": {
            "metrics": _numeric_summary(forced_rows),
            "paired_effect_vs_learned": forced_effect,
            "selection": _selection_summary(forced_rows),
        },
        "oracle_with_learned_fallback": {
            "metrics": _numeric_summary(fallback_rows),
            "paired_effect_vs_learned": fallback_effect,
            "selection": _selection_summary(fallback_rows),
        },
        "total_seconds": total_seconds,
        "interpretation": {
            "forced_oracle": (
                "best target-selected categorical action map among the "
                "enumerated native/shallow/null combinations"
            ),
            "learned_fallback": (
                "best of the forced oracle and the checkpoint learned output; "
                "guarantees a non-worse per-sample upper envelope"
            ),
            "positive_paired_effect": "oracle better than learned",
        },
    }
    _write_json(output_dir / "oracle_summary.json", summary)
    router.set_inference_route_action_intervention()
    print(
        f"Oracle complete in {total_seconds / 60.0:.1f} min. "
        f"Results: {output_dir}",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an exact target-selected oracle over six per-band "
            "native/shallow/null route actions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument(
        "--choices",
        default="native,shallow,null",
        help=(
            "Comma-separated route actions. Use native,shallow for the exact "
            "destination-only 2^6 oracle."
        ),
    )
    parser.add_argument(
        "--objective",
        default="lesion_topq_peak_error_norm",
        help="Per-sample numeric metric used to select the oracle action map.",
    )
    parser.add_argument(
        "--maximize-objective",
        action="store_true",
        help="Select the largest objective value instead of the smallest.",
    )
    parser.add_argument(
        "--frequency-mode",
        default="full",
        choices=["full", "ll_off"],
        help=(
            "Keep LL fixed across variants (full) or suppress it for a "
            "detail-only oracle (ll_off)."
        ),
    )
    parser.add_argument("--weights", choices=["ema", "raw"], default="ema")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-train-fallback", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=729)
    parser.add_argument("--lowpass-sigma", type=float, default=4.0)
    parser.add_argument("--body-threshold", type=float, default=0.03)
    parser.add_argument("--lesion-exclusion-radius", type=int, default=8)
    parser.add_argument("--hotspot-hit-radius", type=int, default=3)
    parser.add_argument("--recovery-ratio", type=float, default=0.6)
    parser.add_argument("--small-lesion-quantile", type=float, default=0.25)
    parser.add_argument(
        "--small-lesion-underestimate-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output-dir",
        default="results/oracle_destination",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.max_combinations <= 0:
        raise ValueError("--max-combinations must be positive")
    if args.lowpass_sigma <= 0:
        raise ValueError("--lowpass-sigma must be positive")
    if not 0.0 <= args.body_threshold <= 1.0:
        raise ValueError("--body-threshold must be in [0, 1]")
    if args.lesion_exclusion_radius < 0 or args.hotspot_hit_radius < 0:
        raise ValueError("mask radii must be non-negative")
    if not 0.0 < args.recovery_ratio <= 1.0:
        raise ValueError("--recovery-ratio must be in (0, 1]")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
