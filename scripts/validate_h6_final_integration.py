"""H6: final small-lesion benefit with artifact/global non-inferiority.

The full router and hard-null reference are reused from the exact fixed
endpoint formal runs.  Calibration supplies only non-inferiority margins.
Held-out validation is evaluated once.  H6 requires a positive small-lesion
benefit and simultaneous artifact/global safety; no external-test
generalization claim is made because the locked dataset has no test cohort.
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
    load_json,
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
ANALYSIS_SEED = 20260729
PRIMARY_SMALL_LESION = ("small_lesion_topq_peak_error_norm", True)
SECONDARY_SMALL_LESION = ("small_lesion_underestimate", True)
SAFETY_ENDPOINTS = {
    "target_relative_false_hotspot_density": True,
    "stripe_abs_excess": True,
    "lesion_boundary_gradient_mae_norm": True,
    "failure_any": True,
    "mae": True,
    "ssim": False,
}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    h5_path = resolve(root, args.h5_decision)
    h5 = load_json(h5_path)
    if h5.get("decision") != "PASS" or not h5.get("next_stage_allowed"):
        raise RuntimeError("H5 is not PASS; H6 is blocked")
    context = load_formal_context(
        root=root,
        contract_path=resolve(root, args.contract),
        h2_decision_path=resolve(root, args.h2_decision),
        curriculum_decision_path=resolve(root, args.curriculum_decision),
        cache_dir=resolve(root, args.cache_dir),
        cache_lineage=resolve(root, args.cache_lineage),
        base_config=resolve(root, args.base_config),
    )
    if h5.get("dataset_contract_sha256") != context["contract_sha256"]:
        raise RuntimeError("H5 dataset contract mismatch")
    run_dir = resolve(root, args.formal_run_dir)
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

    patient_rows: list[dict[str, Any]] = []
    primary_metric, primary_lower = PRIMARY_SMALL_LESION
    primary_validation = paired_patient_effects(
        evaluations["full"]["validation"],
        evaluations["null_reference"]["validation"],
        metric=primary_metric,
        lower_is_better=primary_lower,
    )
    if len(primary_validation) < 5:
        raise RuntimeError("Insufficient small-lesion validation patient pairs")
    primary_statistics = effect_statistics(
        primary_validation,
        seed=ANALYSIS_SEED,
    )
    primary_pass = bool(
        primary_statistics["ci95_low"] > 0.0
        and primary_statistics["sign_flip_p"] < 0.05
    )
    patient_rows.extend(
        {**row, "partition": "validation", "gate": "primary_small_lesion"}
        for row in primary_validation
    )

    secondary_metric, secondary_lower = SECONDARY_SMALL_LESION
    secondary_calibration = paired_patient_effects(
        evaluations["full"]["calibration"],
        evaluations["null_reference"]["calibration"],
        metric=secondary_metric,
        lower_is_better=secondary_lower,
    )
    secondary_validation = paired_patient_effects(
        evaluations["full"]["validation"],
        evaluations["null_reference"]["validation"],
        metric=secondary_metric,
        lower_is_better=secondary_lower,
    )
    secondary_margin = calibration_absolute_margin(
        secondary_calibration
    )
    secondary_statistics = effect_statistics(
        secondary_validation,
        seed=ANALYSIS_SEED + 100,
    )
    secondary_pass = bool(
        secondary_statistics["ci95_low"] >= -secondary_margin
    )

    safety_results: dict[str, Any] = {}
    for index, (metric, lower_is_better) in enumerate(
        SAFETY_ENDPOINTS.items()
    ):
        calibration = paired_patient_effects(
            evaluations["full"]["calibration"],
            evaluations["null_reference"]["calibration"],
            metric=metric,
            lower_is_better=lower_is_better,
        )
        validation = paired_patient_effects(
            evaluations["full"]["validation"],
            evaluations["null_reference"]["validation"],
            metric=metric,
            lower_is_better=lower_is_better,
        )
        margin = calibration_absolute_margin(calibration)
        statistics = effect_statistics(
            validation,
            seed=ANALYSIS_SEED + 1000 + index * 100,
        )
        passed = bool(statistics["ci95_low"] >= -margin)
        safety_results[metric] = {
            "status": "PASS" if passed else "FAIL",
            "effect_definition": "positive means full is better",
            "calibration_absolute_difference_q95_margin": margin,
            "validation": statistics,
        }
        for partition, rows in (
            ("calibration", calibration),
            ("validation", validation),
        ):
            patient_rows.extend(
                {
                    **row,
                    "partition": partition,
                    "gate": "safety_noninferiority",
                }
                for row in rows
            )

    safety_pass = all(
        result["status"] == "PASS"
        for result in safety_results.values()
    )
    passed = primary_pass and secondary_pass and safety_pass
    write_csv(output / "patient_final_effects.csv", patient_rows)
    write_json(
        output / "calibration_margins.json",
        {
            "source": "calibration paired absolute differences only",
            "small_lesion_underestimate": secondary_margin,
            "safety": {
                metric: result[
                    "calibration_absolute_difference_q95_margin"
                ]
                for metric, result in safety_results.items()
            },
        },
    )
    write_json(output / "formal_runs.json", records)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "08_H6_final_integration",
        "decision": "PASS" if passed else "FAIL",
        "dataset_contract_sha256": context["contract_sha256"],
        "cache_metadata_sha256": records["full"]["checkpoint"][
            "cache_metadata_sha256"
        ],
        "mechanism_partition_sha256": context["h2"].get(
            "mechanism_partition_sha256"
        ),
        "guardrails": decision_guardrails(),
        "H6": {
            "status": "PASS" if passed else "FAIL",
            "candidate": "full learned router",
            "reference": "hard-all-null spectral injection",
            "small_lesion_primary": {
                "metric": primary_metric,
                "status": "PASS" if primary_pass else "FAIL",
                "effect_definition": "positive means lower error for full",
                "validation": primary_statistics,
            },
            "small_lesion_underestimate_noninferiority": {
                "metric": secondary_metric,
                "status": "PASS" if secondary_pass else "FAIL",
                "calibration_margin": secondary_margin,
                "validation": secondary_statistics,
            },
            "artifact_and_global_noninferiority": {
                "status": "PASS" if safety_pass else "FAIL",
                "endpoints": safety_results,
            },
            "physical_SUV_claim": False,
            "external_test_generalization_claim": False,
            "claim_scope": (
                "locked PNG cohort held-out patient validation only"
            ),
            "intersection_gate": (
                "primary small-lesion superiority and every safety/global "
                "non-inferiority endpoint must pass"
            ),
        },
        "formal_final_performance_claims_allowed": passed,
        "next_stage_allowed": passed,
        "stop_rule": (
            "H6 passed for the locked held-out PNG validation cohort."
            if passed
            else "Do not promote the complete method or alter failed gates "
            "using held-out validation."
        ),
    }
    write_json(output / "decision.json", decision)
    write_json(
        output / "execution_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": " ".join(sys.argv),
            "script_sha256": file_sha256(Path(__file__).resolve()),
            "fixed_endpoint_epochs": args.epochs,
            "validation_used_to_drive_training_or_course": False,
            "external_test_cohort_available": False,
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
        "--h5-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/07_h5_router_roles/decision.json"
        ),
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
        default=Path(
            "results/mechanism_validation/08_h6_final_integration"
        ),
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
            "stage": "08_H6_final_integration",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(
                checkpoint_lineage="NOT_EVALUATED"
            ),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "formal_final_performance_claims_allowed": False,
            "next_stage_allowed": False,
            "stop_rule": "Do not make a complete-method performance claim.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
