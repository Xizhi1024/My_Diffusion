"""H5: ablate CT support, recoverability, and null-safety separately.

The full fixed-endpoint router is compared with three single-factor ablations.
Each role owns a predeclared outcome family.  Calibration may select one
primary endpoint within that family; held-out validation may not change it.
H5 passes only if all three validation effects favor the full router with a
patient-bootstrap CI above zero and one-sided sign-flip p < 0.05.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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
    effect_statistics,
    ensure_variant,
    load_evaluation,
    load_formal_context,
    paired_patient_effects,
    resolve,
)


SCHEMA_VERSION = 1
ANALYSIS_SEED = 20260728
ROLE_SPECS = {
    "ct_spatial_support": {
        "ablation": "no_ct_support",
        "candidates": [
            ("lesion_centroid_distance", True),
            ("lesion_peak_to_boundary_distance", False),
        ],
        "interpretation": "spatial localization/support, not PET amplitude",
    },
    "residual_recoverability": {
        "ablation": "no_recoverability",
        "candidates": [
            ("directional_spectrum_error_norm", True),
            ("lesion_ring_core_ratio_error", True),
        ],
        "interpretation": "time/band recoverability fidelity",
    },
    "artifact_safety": {
        "ablation": "no_artifact_safety",
        "candidates": [
            ("target_relative_false_hotspot_density", True),
            ("stripe_abs_excess", True),
            ("lesion_boundary_gradient_mae_norm", True),
            ("failure_any", True),
        ],
        "interpretation": "null-route suppression of unsafe injection",
    },
}


def select_calibration_metric(
    full: Mapping[str, Any],
    ablation: Mapping[str, Any],
    candidates: Sequence[tuple[str, bool]],
) -> tuple[str, bool, list[dict[str, Any]], dict[str, Any]]:
    """Select the largest standardized paired effect on calibration only."""

    ranked = []
    for order, (metric, lower_is_better) in enumerate(candidates):
        rows = paired_patient_effects(
            full,
            ablation,
            metric=metric,
            lower_is_better=lower_is_better,
        )
        values = np.asarray(
            [float(row["effect_left_better"]) for row in rows],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size < 5:
            continue
        scale = max(float(values.std(ddof=1)), 1e-8)
        ranked.append(
            {
                "metric": metric,
                "lower_is_better": lower_is_better,
                "patients": int(values.size),
                "mean_effect": float(values.mean()),
                "standardized_effect": float(values.mean() / scale),
                "order": order,
                "rows": rows,
            }
        )
    if not ranked:
        raise RuntimeError("No role candidate has enough calibration patients")
    selected = max(
        ranked,
        key=lambda row: (row["standardized_effect"], -row["order"]),
    )
    audit = {
        row["metric"]: {
            key: value
            for key, value in row.items()
            if key not in {"rows", "order"}
        }
        for row in ranked
    }
    return (
        str(selected["metric"]),
        bool(selected["lower_is_better"]),
        list(selected["rows"]),
        audit,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output = resolve(root, args.output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = resolve(root, args.artifact_decision)
    artifact = load_json(artifact_path)
    if artifact.get("decision") != "PASS" or not artifact.get(
        "next_stage_allowed"
    ):
        raise RuntimeError("Artifact gate is not PASS; H5 is blocked")
    context = load_formal_context(
        root=root,
        contract_path=resolve(root, args.contract),
        h2_decision_path=resolve(root, args.h2_decision),
        curriculum_decision_path=resolve(root, args.curriculum_decision),
        cache_dir=resolve(root, args.cache_dir),
        cache_lineage=resolve(root, args.cache_lineage),
        base_config=resolve(root, args.base_config),
    )
    if artifact.get("dataset_contract_sha256") != context[
        "contract_sha256"
    ]:
        raise RuntimeError("Artifact gate dataset contract mismatch")
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
        for variant in (
            "full",
            "no_ct_support",
            "no_recoverability",
            "no_artifact_safety",
        )
    }
    evaluations = {
        variant: {
            role: load_evaluation(record, role)
            for role in ("calibration", "validation")
        }
        for variant, record in records.items()
    }
    role_results: dict[str, Any] = {}
    patient_rows: list[dict[str, Any]] = []
    frozen_metrics: dict[str, Any] = {}
    for index, (role, spec) in enumerate(ROLE_SPECS.items()):
        ablation_name = str(spec["ablation"])
        (
            metric,
            lower_is_better,
            calibration_rows,
            calibration_audit,
        ) = select_calibration_metric(
            evaluations["full"]["calibration"],
            evaluations[ablation_name]["calibration"],
            spec["candidates"],
        )
        validation_rows = paired_patient_effects(
            evaluations["full"]["validation"],
            evaluations[ablation_name]["validation"],
            metric=metric,
            lower_is_better=lower_is_better,
        )
        if len(validation_rows) < 5:
            raise RuntimeError(
                f"Insufficient validation patient pairs for {role}/{metric}"
            )
        statistics = effect_statistics(
            validation_rows,
            seed=ANALYSIS_SEED + index * 100,
        )
        passed = bool(
            statistics["ci95_low"] > 0.0
            and statistics["sign_flip_p"] < 0.05
        )
        frozen_metrics[role] = {
            "selected_metric": metric,
            "lower_is_better": lower_is_better,
            "candidate_audit": calibration_audit,
        }
        role_results[role] = {
            "status": "PASS" if passed else "FAIL",
            "ablation": ablation_name,
            "interpretation": spec["interpretation"],
            "selected_metric_from_calibration": metric,
            "effect_definition": "positive means full outperforms ablation",
            "validation": statistics,
        }
        for partition, rows in (
            ("calibration", calibration_rows),
            ("validation", validation_rows),
        ):
            patient_rows.extend(
                {
                    **row,
                    "partition": partition,
                    "role": role,
                    "ablation": ablation_name,
                }
                for row in rows
            )
    passed = all(
        result["status"] == "PASS" for result in role_results.values()
    )
    write_csv(output / "patient_role_effects.csv", patient_rows)
    write_json(
        output / "frozen_role_metrics.json",
        {
            "source": "calibration only",
            "selection": "largest standardized paired effect within each predeclared role family",
            "roles": frozen_metrics,
        },
    )
    write_json(output / "formal_runs.json", records)
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "07_H5_router_role_separation",
        "decision": "PASS" if passed else "FAIL",
        "dataset_contract_sha256": context["contract_sha256"],
        "cache_metadata_sha256": records["full"]["checkpoint"][
            "cache_metadata_sha256"
        ],
        "mechanism_partition_sha256": context["h2"].get(
            "mechanism_partition_sha256"
        ),
        "guardrails": decision_guardrails(),
        "H5": {
            "status": "PASS" if passed else "FAIL",
            "roles": role_results,
            "mask_participates_in_router": False,
            "wording": (
                "paired predictive/ablation evidence; no causal or "
                "mutual-information claim"
            ),
            "intersection_gate": "all three role contrasts must pass",
        },
        "formal_runs": {
            key: {
                "checkpoint": value["checkpoint"],
                "evaluations": value["evaluations"],
            }
            for key, value in records.items()
        },
        "artifact_gate": {
            "path": artifact_path.as_posix(),
            "sha256": file_sha256(artifact_path),
        },
        "next_stage_allowed": passed,
        "stop_rule": (
            "H5 passed; final integration may be evaluated once."
            if passed
            else "Stop failed router roles; do not change role metrics from "
            "held-out validation."
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
            "validation_used_for_metric_selection": False,
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
        "--artifact-decision",
        type=Path,
        default=Path(
            "results/mechanism_validation/06_artifact_safety/decision.json"
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
        default=Path("results/mechanism_validation/07_h5_router_roles"),
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
            "stage": "07_H5_router_role_separation",
            "decision": "FAIL",
            "dataset_contract_sha256": None,
            "guardrails": decision_guardrails(
                checkpoint_lineage="NOT_EVALUATED"
            ),
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "next_stage_allowed": False,
            "stop_rule": "Stop H5 and H6.",
        }
        write_json(output / "decision.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return 2
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    return 0 if decision["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
