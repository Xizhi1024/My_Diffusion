"""Artifact-safety gate for the fixed full router versus hard-null reference.

Both models use identical data, conditional mean, initialization seed,
optimizer, epoch budget, and exact final EMA endpoint.  Calibration fixes a
95th-percentile paired absolute non-inferiority margin for each predeclared
safety endpoint.  Held-out validation is evaluated once with patient
bootstrap confidence intervals.  Every endpoint must pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import (
    decision_guardrails,
    file_sha256,
    write_csv,
    write_json,
)
from src.mechanism_validation.model_experiments import (
    DEFAULT_EPOCHS,
    DEFAULT_MC_STEPS,
    calibration_absolute_margin,
    effect_statistics,
    ensure_variant,
    load_evaluation,
    load_formal_context,
    paired_patient_effects,
    resolve,
)


SCHEMA_VERSION = 1
ANALYSIS_SEED = 20260727
SAFETY_ENDPOINTS = {
    "target_relative_false_hotspot_density": {
        "lower_is_better": True,
        "role": "false hotspots",
    },
    "stripe_abs_excess": {
        "lower_is_better": True,
        "role": "directional stripes",
    },
    "lesion_boundary_gradient_mae_norm": {
        "lower_is_better": True,
        "role": "mask-boundary fidelity",
    },
    "failure_any": {
        "lower_is_better": True,
        "role": "composite failure",
    },
    "mae": {
        "lower_is_better": True,
        "role": "whole-image error",
    },
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    context = load_formal_context(
        root=root,
        contract_path=resolve(root, args.contract),
        h2_decision_path=resolve(root, args.h2_decision),
        curriculum_decision_path=resolve(root, args.curriculum_decision),
        cache_dir=resolve(root, args.cache_dir),
        cache_lineage=resolve(root, args.cache_lineage),
        base_config=resolve(root, args.base_config),
    )
    run_dir = resolve(root, args.formal_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = {
        variant: ensure_variant(
            root=root,
            work_dir=run_dir,
            base_config=context["base_config"],
            cache_dir=context["cache_dir"],
            cache_lineage=context["cache_lineage"],
            contract=context["contract_path"],
            mechanism_manifest=context["mechanism_manifest"],
            mean_checkpoint=context["mean_checkpoint"],
            curriculum=context["curriculum"],
            variant=variant,
            epochs=args.epochs,
            mc_steps=args.mc_steps,
            force=args.force,
        )
        for variant in ("null_reference", "full")
    }
    evaluations = {
        variant: {
            role: load_evaluation(record, role)
            for role in ("calibration", "validation")
        }
        for variant, record in records.items()
    }

    endpoint_results: dict[str, Any] = {}
    all_patient_rows: list[dict[str, Any]] = []
    for index, (metric, spec) in enumerate(SAFETY_ENDPOINTS.items()):
        calibration = paired_patient_effects(
            evaluations["full"]["calibration"],
            evaluations["null_reference"]["calibration"],
            metric=metric,
            lower_is_better=bool(spec["lower_is_better"]),
        )
        validation = paired_patient_effects(
            evaluations["full"]["validation"],
            evaluations["null_reference"]["validation"],
            metric=metric,
            lower_is_better=bool(spec["lower_is_better"]),
        )
        if len(calibration) < 5 or len(validation) < 5:
            raise RuntimeError(f"Insufficient patient pairs for {metric}")
        margin = calibration_absolute_margin(calibration)
        statistics = effect_statistics(
            validation,
            seed=ANALYSIS_SEED + index * 100,
        )
        passed = bool(statistics["ci95_low"] >= -margin)
        endpoint_results[metric] = {
            "role": spec["role"],
            "effect_definition": "positive means full is safer/better",
            "calibration_absolute_difference_q95_margin": margin,
            "validation": statistics,
            "status": "PASS" if passed else "FAIL",
        }
        for partition, rows in (
            ("calibration", calibration),
            ("validation", validation),
        ):
            all_patient_rows.extend(
                {
                    **row,
                    "partition": partition,
                    "endpoint": metric,
                }
                for row in rows
            )

    passed = all(
        result["status"] == "PASS"
        for result in endpoint_results.values()
    )
    write_csv(output / "patient_safety_effects.csv", all_patient_rows)
    write_json(
        output / "calibration_margins.json",
        {
            "source": "calibration paired patient absolute differences only",
            "quantile": 0.95,
            "endpoints": {
                metric: result[
                    "calibration_absolute_difference_q95_margin"
                ]
                for metric, result in endpoint_results.items()
            },
        },
    )
    write_json(
        output / "formal_runs.json",
        {
            "null_reference": records["null_reference"],
            "full": records["full"],
        },
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "06_artifact_safety",
        "decision": "PASS" if passed else "FAIL",
        "dataset_contract_sha256": context["contract_sha256"],
        "cache_metadata_sha256": records["full"]["checkpoint"][
            "cache_metadata_sha256"
        ],
        "mechanism_partition_sha256": context["h2"].get(
            "mechanism_partition_sha256"
        ),
        "guardrails": decision_guardrails(),
        "artifact_safety": {
            "status": "PASS" if passed else "FAIL",
            "reference": "hard-all-null spectral injection",
            "candidate": "full learned router",
            "fixed_endpoint_epochs": args.epochs,
            "weights": "EMA",
            "endpoints": endpoint_results,
            "intersection_gate": "every predeclared endpoint must pass",
        },
        "formal_runs": {
            key: {
                "checkpoint": value["checkpoint"],
                "evaluations": value["evaluations"],
            }
            for key, value in records.items()
        },
        "next_stage_allowed": passed,
        "stop_rule": (
            "Artifact safety passed; router role ablations may start."
            if passed
            else "Stop router ablations and final integration; do not relax "
            "calibration margins from held-out validation."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "training_endpoint": "fixed final epoch",
            "model_selection": False,
            "validation_used_to_drive_training_or_course": False,
        },
    )
    return decision


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("configs/dataset_contract_stage0a_v1.json"),
    )
    parser.add_argument(
        "--h2-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/01_h2_residual_enrichment/decision.json"
        ),
    )
    parser.add_argument(
        "--curriculum-decision",
        type=Path,
        default=Path("results/mechanism_validation/05_curriculum/decision.json"),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path(
            "configs/experiments/slmf_png_spectral_router_v5.yaml"
        ),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("cache/tensors_main")
    )
    parser.add_argument(
        "--cache-lineage",
        type=Path,
        default=Path("cache/tensors_main/cache_lineage.json"),
    )
    parser.add_argument(
        "--formal-run-dir",
        type=Path,
        default=Path("results/mechanism_validation/formal_model_runs_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mechanism_validation/06_artifact_safety"),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--mc-steps", type=int, default=DEFAULT_MC_STEPS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.root = args.root.resolve()
    if args.epochs <= 0 or args.mc_steps <= 0:
        parser.error("--epochs and --mc-steps must be positive")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        decision = run(args)
    except Exception as exc:
        output = resolve(args.root, args.output)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "stage": "06_artifact_safety",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(
                checkpoint_lineage="NOT_EVALUATED"
            ),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop H5/H6 formal model claims.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
