"""Run V5 evidence screening, route screening, and exact-epoch promotion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_v4_results import compare_all_results
from scripts.run_boundary_reliable_v4 import _load_decision, _run_stage
from scripts.run_frequency_ablations import (
    _json_safe,
    _require_planned_mean_checkpoint,
    _run_entries,
    _run_manifest_entry,
    _write_rankings,
    composite_score,
    passes_hard_gates,
)


MODE_KEY = "modules.residual_frequency.mode"
CROSS_ENABLED_KEY = "modules.residual_frequency.cross_level_router.enabled"
HARD_NULL_KEY = "modules.residual_frequency.cross_level_router.hard_all_null"
DCT_ENABLED_KEY = "modules.residual_frequency.dct_descriptor.enabled"
GABOR_ENABLED_KEY = "modules.residual_frequency.gabor_descriptor.enabled"


def _stage_a_variant(
    plan: Mapping[str, Any], variant_id: str
) -> Mapping[str, Any]:
    match = next(
        (row for row in plan["stage_a"]["variants"] if row["id"] == variant_id),
        None,
    )
    if match is None:
        raise KeyError(f"Unknown Stage A evidence variant {variant_id!r}")
    return match


def build_stage_b_variants(
    plan: Mapping[str, Any], selected_evidence_id: str
) -> list[Dict[str, Any]]:
    """Pair every V5 route with the exact Stage A evidence preset."""
    selected = _stage_a_variant(plan, selected_evidence_id)
    variants: list[Dict[str, Any]] = []
    expected_ids = ("N0", "T0", "C0", "C1")
    templates = {str(row["id"]): row for row in plan["stage_b"]["variants"]}
    missing = [variant_id for variant_id in expected_ids if variant_id not in templates]
    if missing:
        raise KeyError(f"Missing Stage B route templates: {missing}")

    for variant_id in expected_ids:
        overrides = dict(templates[variant_id].get("overrides", {}))
        overrides[MODE_KEY] = "spectral_evidence_router"
        if variant_id == "N0":
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = True
        elif variant_id == "T0":
            overrides[CROSS_ENABLED_KEY] = False
            overrides[HARD_NULL_KEY] = False
        elif variant_id == "C0":
            overrides[DCT_ENABLED_KEY] = False
            overrides[GABOR_ENABLED_KEY] = False
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = False
        else:
            # S0 disables both descriptors, but C1 is the complete V5 route;
            # descriptor-bearing S1-S3 continue to inherit their evidence preset.
            if selected_evidence_id == "S0":
                overrides[DCT_ENABLED_KEY] = True
                overrides[GABOR_ENABLED_KEY] = True
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = False
        variants.append({
            "id": variant_id,
            "preset": str(selected["preset"]),
            "overrides": overrides,
            "inherited_evidence": selected_evidence_id,
        })
    return variants


def _exact_final_checkpoint(entry: Mapping[str, Any]) -> Dict[str, Any]:
    exact_entry = dict(entry)
    completion_checkpoint = exact_entry.get("completion_checkpoint")
    if not completion_checkpoint:
        raise ValueError("V5 promotion requires an exact final checkpoint")
    exact_entry["checkpoint"] = str(completion_checkpoint)
    eval_command = list(exact_entry["eval_command"])
    checkpoint_index = eval_command.index("--checkpoint") + 1
    eval_command[checkpoint_index] = str(completion_checkpoint)
    exact_entry["eval_command"] = eval_command
    return exact_entry


def _build_entries(
    plan: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    python: str,
    phase: str,
) -> list[Dict[str, Any]]:
    entries = [
        _run_manifest_entry(plan, variant, settings, python, phase)
        for variant in variants
    ]
    if phase == "promote":
        return [_exact_final_checkpoint(entry) for entry in entries]
    return entries


def build_v5_dry_run_manifest(
    plan: Mapping[str, Any], *, python: str
) -> Dict[str, Any]:
    """Build a deterministic two-stage V5 manifest without prior results."""
    selected_evidence = str(
        plan["stage_a"].get("dry_run_selected_evidence", "S3")
    )
    stage_b_variants = build_stage_b_variants(plan, selected_evidence)
    promotion_count = min(int(plan["stage_b"].get("top_k", 2)), 2)
    return {
        "selected_evidence_template": selected_evidence,
        "stage_a_runs": _build_entries(
            plan,
            plan["stage_a"]["variants"],
            plan["stage_a"],
            python,
            "stage_a",
        ),
        "stage_b_runs": _build_entries(
            plan, stage_b_variants, plan["stage_b"], python, "stage_b"
        ),
        "promotion_runs": _build_entries(
            plan,
            stage_b_variants[:promotion_count],
            plan["promote"],
            python,
            "promote",
        ),
    }


def _write_promotion_outputs(
    plan: Mapping[str, Any],
    promotion_records: Sequence[Mapping[str, Any]],
    stage_b_decision: Mapping[str, Any],
) -> None:
    output_dir = Path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_id = str(plan["stage_b"]["reference_id"])
    reference = next(
        row for row in stage_b_decision["ranked"] if row["id"] == reference_id
    )

    final_rows = []
    accepted = []
    for row in promotion_records:
        passed, reasons = passes_hard_gates(
            row["metrics"], reference["metrics"], plan["stage_b"]["hard_gates"]
        )
        final_rows.append({
            **dict(row),
            "gate_passed": passed,
            "gate_reasons": reasons,
        })
        if passed:
            accepted.append(row["id"])

    (output_dir / "final_gate_decision.json").write_text(
        json.dumps(
            _json_safe({"accepted": accepted, "ranked": final_rows}),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "promotion_results.json").write_text(
        json.dumps(
            _json_safe(list(promotion_records)), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    result_paths = {
        str(row["id"]): output_dir / "promote" / f"{str(row['id']).lower()}.json"
        for row in promotion_records
    }
    stage_b_reference = output_dir / "stage_b" / f"{reference_id.lower()}.json"
    if stage_b_reference.exists():
        result_paths[f"{reference_id}_stage_b"] = stage_b_reference
    comparison_cfg = plan.get("paired_comparison", {})
    paired = compare_all_results(
        result_paths,
        comparison_cfg.get("metrics", {}),
        seed=int(comparison_cfg.get("seed", 42)),
        resamples=int(comparison_cfg.get("resamples", 10_000)),
    )
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(_json_safe(paired), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default="configs/experiments/spectral_router_ablation_plan_v5.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=["stage-a", "stage-b", "promote", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with Path(args.plan).open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    output_dir = Path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = build_v5_dry_run_manifest(plan, python=sys.executable)
        destination = output_dir / "dry_run_manifest.json"
        destination.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Dry-run manifest saved to {destination}")
        for phase in ("stage_a_runs", "stage_b_runs", "promotion_runs"):
            for entry in manifest[phase]:
                print(" ".join(entry["train_command"]))
                print(" ".join(entry["eval_command"]))
        return

    _require_planned_mean_checkpoint(plan)
    selected_evidence: list[str]
    if args.stage in {"stage-a", "all"}:
        selected_evidence, _ = _run_stage(
            plan,
            plan["stage_a"]["variants"],
            plan["stage_a"],
            phase="stage_a",
            python=sys.executable,
            force=args.force,
        )
        if args.stage == "stage-a":
            return
    else:
        selected_evidence = list(
            _load_decision(output_dir, "stage_a").get("promoted", [])
        )
    if not selected_evidence:
        print("No Stage A evidence candidate passed; Stage B and promotion stopped.")
        return

    stage_b_variants = build_stage_b_variants(plan, selected_evidence[0])
    promoted_routes: list[str]
    if args.stage in {"stage-b", "all"}:
        promoted_routes, _ = _run_stage(
            plan,
            stage_b_variants,
            plan["stage_b"],
            phase="stage_b",
            python=sys.executable,
            force=args.force,
        )
        if args.stage == "stage-b":
            return
    else:
        promoted_routes = list(
            _load_decision(output_dir, "stage_b").get("promoted", [])
        )
    if not promoted_routes:
        print("No Stage B route passed; 300-epoch promotion stopped.")
        return

    promoted_ids = set(promoted_routes[:2])
    promotion_variants = [
        row for row in stage_b_variants if row["id"] in promoted_ids
    ]
    promotion_records = _run_entries(
        _build_entries(
            plan,
            promotion_variants,
            plan["promote"],
            sys.executable,
            "promote",
        ),
        force=args.force,
    )
    _write_promotion_outputs(
        plan, promotion_records, _load_decision(output_dir, "stage_b")
    )


if __name__ == "__main__":
    main()
