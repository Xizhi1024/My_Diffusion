"""Run BR-D peak/CT screening, then dynamic Gabor screening and promotion."""

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
from scripts.run_frequency_ablations import (
    _json_safe,
    _require_planned_mean_checkpoint,
    _run_entries,
    _run_manifest_entry,
    _write_rankings,
    composite_score,
    passes_hard_gates,
)


def _stage_a_variant(plan: Mapping[str, Any], variant_id: str) -> Mapping[str, Any]:
    variants = plan["stage_a"]["variants"]
    match = next((row for row in variants if row["id"] == variant_id), None)
    if match is None:
        raise KeyError(f"Unknown Stage A variant {variant_id!r}")
    return match


def build_stage_b_variants(
    plan: Mapping[str, Any], selected_d_id: str
) -> list[Dict[str, Any]]:
    """Attach every Gabor route to the exact Stage A winner preset."""
    selected = _stage_a_variant(plan, selected_d_id)
    variants = []
    for route in plan["stage_b"]["variants"]:
        variants.append({
            "id": str(route["id"]),
            "preset": str(selected["preset"]),
            "overrides": dict(route.get("overrides", {})),
            "inherited_d": selected_d_id,
        })
    return variants


def _build_entries(
    plan: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    python: str,
    phase: str,
) -> list[Dict[str, Any]]:
    return [
        _run_manifest_entry(plan, variant, settings, python, phase)
        for variant in variants
    ]


def build_v4_dry_run_manifest(
    plan: Mapping[str, Any], *, python: str
) -> Dict[str, Any]:
    selected_d = str(plan["stage_b"].get("dry_run_selected_d", "D3"))
    stage_b_variants = build_stage_b_variants(plan, selected_d)
    promotion_count = min(int(plan["stage_b"].get("top_k", 2)), 2)
    return {
        "selected_d_template": selected_d,
        "stage_a_runs": _build_entries(
            plan, plan["stage_a"]["variants"], plan["stage_a"], python, "stage_a"
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


def _rank_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_id: str,
    gates: Mapping[str, Mapping[str, Any]],
    top_k: int,
) -> tuple[list[str], list[Dict[str, Any]]]:
    reference = next((row for row in records if row["id"] == reference_id), None)
    if reference is None:
        raise KeyError(f"Reference variant {reference_id!r} was not evaluated")
    ranked = []
    for row in records:
        gate_passed, reasons = passes_hard_gates(
            row["metrics"], reference["metrics"], gates
        )
        try:
            score = composite_score(row["metrics"])
        except (KeyError, TypeError, ValueError) as exc:
            gate_passed = False
            reasons = [*reasons, str(exc)]
            score = -1e30
        ranked.append({
            **dict(row),
            "metrics": dict(row["metrics"]),
            "gate_passed": gate_passed,
            "gate_reasons": reasons,
            "score": score,
        })
    ranked.sort(
        key=lambda row: (bool(row["gate_passed"]), float(row["score"])),
        reverse=True,
    )
    promoted = [row["id"] for row in ranked if row["gate_passed"]][:top_k]
    return promoted, ranked


def _load_decision(output_dir: Path, phase: str) -> Mapping[str, Any]:
    path = output_dir / phase / "promotion_decision.json"
    if not path.exists():
        raise FileNotFoundError(f"Run --stage {phase.replace('_', '-')} first: {path} is missing")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_stage(
    plan: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    *,
    phase: str,
    python: str,
    force: bool,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    records = _run_entries(
        _build_entries(plan, variants, settings, python, phase), force=force
    )
    top_k = min(int(settings.get("top_k", 1)), 2)
    promoted, ranked = _rank_candidates(
        records,
        reference_id=str(settings["reference_id"]),
        gates=settings["hard_gates"],
        top_k=top_k,
    )
    _write_rankings(Path(plan["output_dir"]) / phase, promoted, ranked)
    print(f"{phase} selected: {promoted or '(none)'}")
    return promoted, ranked


def _write_promotion_outputs(
    plan: Mapping[str, Any],
    promotion_records: Sequence[Mapping[str, Any]],
    stage_b_decision: Mapping[str, Any],
) -> None:
    output_dir = Path(plan["output_dir"])
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
        final_row = {**dict(row), "gate_passed": passed, "gate_reasons": reasons}
        final_rows.append(final_row)
        if passed:
            accepted.append(row["id"])
    final_decision = _json_safe({"accepted": accepted, "ranked": final_rows})
    (output_dir / "final_gate_decision.json").write_text(
        json.dumps(final_decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "promotion_results.json").write_text(
        json.dumps(_json_safe(list(promotion_records)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    comparison_cfg = plan.get("paired_comparison", {})
    result_paths = {
        str(row["id"]): output_dir / "promote" / f"{str(row['id']).lower()}.json"
        for row in promotion_records
    }
    paired = compare_all_results(
        result_paths,
        comparison_cfg.get("metrics", {}),
        seed=int(comparison_cfg.get("seed", 42)),
        resamples=int(comparison_cfg.get("resamples", 10_000)),
    )
    (output_dir / "paired_comparison.json").write_text(
        json.dumps(_json_safe(paired), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default="configs/experiments/boundary_reliable_ablation_plan_v4.yaml",
    )
    parser.add_argument(
        "--stage", choices=["stage-a", "stage-b", "promote", "all"], default="all"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with Path(args.plan).open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = build_v4_dry_run_manifest(plan, python=sys.executable)
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
    selected_d = []
    if args.stage in {"stage-a", "all"}:
        selected_d, _ = _run_stage(
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
        selected_d = list(_load_decision(output_dir, "stage_a").get("promoted", []))
    if not selected_d:
        print("No Stage A candidate passed hard gates; Stage B and promotion stopped.")
        return

    stage_b_variants = build_stage_b_variants(plan, selected_d[0])
    promoted_g = []
    if args.stage in {"stage-b", "all"}:
        promoted_g, _ = _run_stage(
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
        promoted_g = list(_load_decision(output_dir, "stage_b").get("promoted", []))
    if not promoted_g:
        print("No Stage B candidate passed hard gates; 300-epoch promotion stopped.")
        return

    promotion_variants = [
        row for row in stage_b_variants if row["id"] in set(promoted_g[:2])
    ]
    promotion_records = _run_entries(
        _build_entries(
            plan, promotion_variants, plan["promote"], sys.executable, "promote"
        ),
        force=args.force,
    )
    stage_b_decision = _load_decision(output_dir, "stage_b")
    _write_promotion_outputs(plan, promotion_records, stage_b_decision)


if __name__ == "__main__":
    main()
