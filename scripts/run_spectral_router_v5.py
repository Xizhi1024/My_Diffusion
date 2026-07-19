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
from scripts.run_boundary_reliable_v4 import (
    _load_decision,
    _run_stage as _shared_run_stage,
)
from scripts.run_frequency_ablations import (
    _json_safe,
    _require_planned_mean_checkpoint,
    _run_entries,
    _run_manifest_entry,
    _write_rankings,
    checkpoint_epoch,
    passes_hard_gates,
)


MODE_KEY = "modules.residual_frequency.mode"
CROSS_ENABLED_KEY = "modules.residual_frequency.cross_level_router.enabled"
HARD_NULL_KEY = "modules.residual_frequency.cross_level_router.hard_all_null"
POLICY_KEY = "modules.residual_frequency.cross_level_router.policy"
FIXED_PRIOR_KEY = "modules.residual_frequency.cross_level_router.fixed_prior"


def _evidence_slug(evidence_id: str) -> str:
    return f"evidence-{evidence_id.lower()}"


def _evidence_phase(phase: str, evidence_id: str) -> str:
    return str(Path(phase) / _evidence_slug(evidence_id))


def _evidence_settings(
    settings: Mapping[str, Any], evidence_id: str
) -> Dict[str, Any]:
    adapted = dict(settings)
    adapted["experiment_prefix"] = (
        f"{settings['experiment_prefix']}_{_evidence_slug(evidence_id)}"
    )
    return adapted


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
    # New fair-control variant IDs (supersede old N0/T0/C0/C1)
    expected_ids = ("N0", "T_legacy", "T_native", "T_fixed", "C1", "C_no_null")
    templates = {str(row["id"]): row for row in plan["stage_b"]["variants"]}
    available_ids = [vid for vid in expected_ids if vid in templates]
    if not available_ids:
        raise KeyError(f"No Stage B route templates found. Expected any of: {expected_ids}")

    for variant_id in available_ids:
        overrides = dict(templates[variant_id].get("overrides", {}))
        overrides[MODE_KEY] = "spectral_evidence_router"
        # Parse policy from overrides or infer legacy defaults
        policy = overrides.get(POLICY_KEY)
        if variant_id == "N0":
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = True
        elif variant_id == "T_legacy":
            overrides[CROSS_ENABLED_KEY] = False
            overrides[HARD_NULL_KEY] = False
        elif variant_id == "T_native":
            overrides[CROSS_ENABLED_KEY] = True
            if not policy:
                overrides[POLICY_KEY] = "native_only"
        elif variant_id == "T_fixed":
            overrides[CROSS_ENABLED_KEY] = True
            if not policy:
                overrides[POLICY_KEY] = "fixed_prior"
            if FIXED_PRIOR_KEY not in overrides:
                overrides[FIXED_PRIOR_KEY] = [0.05, 0.05, 0.90]
        elif variant_id == "C1":
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = False
            if not policy:
                overrides[POLICY_KEY] = "learned"
        elif variant_id == "C_no_null":
            overrides[CROSS_ENABLED_KEY] = True
            overrides[HARD_NULL_KEY] = False
            if not policy:
                overrides[POLICY_KEY] = "learned_no_null"
            if FIXED_PRIOR_KEY not in overrides:
                overrides[FIXED_PRIOR_KEY] = [0.5, 0.5]
        variants.append({
            "id": variant_id,
            "preset": str(selected["preset"]),
            "overrides": overrides,
            "inherited_evidence": selected_evidence_id,
        })
    return variants


def _exact_final_checkpoint(
    entry: Mapping[str, Any], *, expected_epoch: int
) -> Dict[str, Any]:
    exact_entry = dict(entry)
    completion_checkpoint = exact_entry.get("completion_checkpoint")
    if not completion_checkpoint:
        raise ValueError("V5 promotion requires an exact final checkpoint")
    exact_entry["checkpoint"] = str(completion_checkpoint)
    eval_command = list(exact_entry["eval_command"])
    checkpoint_index = eval_command.index("--checkpoint") + 1
    eval_command[checkpoint_index] = str(completion_checkpoint)
    exact_entry["eval_command"] = eval_command
    exact_entry["required_checkpoint_epoch"] = expected_epoch
    return exact_entry


def validate_checkpoint_epoch(
    path: str | Path, *, expected_epoch: int = 300
) -> int:
    """Require checkpoint metadata to identify the exact requested epoch."""
    actual_epoch = checkpoint_epoch(Path(path))
    if actual_epoch != expected_epoch:
        raise ValueError(
            f"Checkpoint {path} expected exact epoch {expected_epoch}, "
            f"found {actual_epoch}"
        )
    return actual_epoch


def _validate_promotion_checkpoint(
    checkpoint: Path, entry: Mapping[str, Any]
) -> int:
    expected_epoch = int(entry.get("required_checkpoint_epoch", 300))
    return validate_checkpoint_epoch(checkpoint, expected_epoch=expected_epoch)


def _build_entries(
    plan: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    python: str,
    phase: str,
    *,
    evidence_id: str | None = None,
) -> list[Dict[str, Any]]:
    effective_settings = settings
    effective_phase = phase
    if evidence_id is not None:
        _stage_a_variant(plan, evidence_id)
        effective_settings = _evidence_settings(settings, evidence_id)
        effective_phase = _evidence_phase(phase, evidence_id)
    entries = [
        _run_manifest_entry(
            plan, variant, effective_settings, python, effective_phase
        )
        for variant in variants
    ]
    for entry in entries:
        entry["selected_evidence_id"] = evidence_id
    if phase == "promote":
        expected_epoch = int(settings["epochs"])
        return [
            _exact_final_checkpoint(entry, expected_epoch=expected_epoch)
            for entry in entries
        ]
    return entries


def build_v5_dry_run_manifest(
    plan: Mapping[str, Any],
    *,
    python: str,
    stage: str = "all",
    selected_evidence_id: str | None = None,
) -> Dict[str, Any]:
    """Build a deterministic two-stage V5 manifest without prior results."""
    if stage not in {"stage-a", "stage-b", "promote", "all"}:
        raise ValueError(f"Unknown V5 stage {stage!r}")
    selected_evidence = str(
        selected_evidence_id
        or plan["stage_a"].get("dry_run_selected_evidence", "S3")
    )
    stage_b_variants = build_stage_b_variants(plan, selected_evidence)
    promotion_count = min(int(plan["stage_b"].get("top_k", 3)), 4)
    reference_id = str(plan["stage_b"]["reference_id"])

    # Always include the reference (T_native) in the dry-run manifest so it
    # matches the real training budget.
    reference_variant = next(
        row for row in stage_b_variants if row["id"] == reference_id
    )
    candidate_variants = [
        row for row in stage_b_variants
        if (
            row["id"] != reference_id
            and str(row["id"]) not in _NON_PROMOTABLE_IDS
        )
    ][:promotion_count]
    eligible_promotion_variants = [reference_variant] + candidate_variants
    return {
        "promotion_mode": "maximum_budget",
        "selected_evidence_template": selected_evidence,
        "stage_a_runs": (
            _build_entries(
                plan,
                plan["stage_a"]["variants"],
                plan["stage_a"],
                python,
                "stage_a",
            )
            if stage in {"stage-a", "all"}
            else []
        ),
        "stage_b_runs": (
            _build_entries(
                plan,
                stage_b_variants,
                plan["stage_b"],
                python,
                "stage_b",
                evidence_id=selected_evidence,
            )
            if stage in {"stage-b", "all"}
            else []
        ),
        "promotion_runs": (
            _build_entries(
                plan,
                eligible_promotion_variants,
                plan["promote"],
                python,
                "promote",
                evidence_id=selected_evidence,
            )
            if stage in {"promote", "all"}
            else []
        ),
    }


# Variants that are diagnostics-only and must never be promoted to 300-epoch training.
_NON_PROMOTABLE_IDS: frozenset[str] = frozenset({"N0", "T_legacy"})


def _select_eligible_stage_b_routes(
    ranked: Sequence[Mapping[str, Any]],
    *,
    reference_id: str,
    top_k: int,
) -> list[str]:
    """Filter the shared gate/score ranking to promotable V5 routes.

    Excludes the reference architecture, diagnostic-only variants (N0, T_legacy),
    and any variant that failed its hard gates.
    """
    limit = min(max(int(top_k), 0), 4)
    return [
        str(row["id"])
        for row in ranked
        if (
            row["id"] != reference_id
            and str(row["id"]) not in _NON_PROMOTABLE_IDS
            and bool(row["gate_passed"])
        )
    ][:limit]


def _promotion_route_ids(
    promoted_routes: Sequence[str],
    *,
    reference_id: str,
    top_k: int,
) -> set[str]:
    """Return the reference plus up to ``top_k`` eligible candidates."""
    candidate_ids: list[str] = []
    for route_id_value in promoted_routes:
        route_id = str(route_id_value)
        if (
            route_id == reference_id
            or route_id in _NON_PROMOTABLE_IDS
            or route_id in candidate_ids
        ):
            continue
        candidate_ids.append(route_id)
    limit = max(int(top_k), 0)
    return {reference_id, *candidate_ids[:limit]}


def _record_evidence_provenance(
    decision_dir: Path, evidence_id: str
) -> None:
    decision_path = decision_dir / "promotion_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["selected_evidence_id"] = evidence_id
    decision_path.write_text(
        json.dumps(_json_safe(decision), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_v5_stage(
    plan: Mapping[str, Any],
    variants: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    *,
    phase: str,
    python: str,
    force: bool,
    evidence_id: str | None = None,
) -> tuple[list[str], list[Mapping[str, Any]]]:
    effective_phase = phase
    effective_settings = settings
    if evidence_id is not None:
        _stage_a_variant(plan, evidence_id)
        effective_phase = _evidence_phase(phase, evidence_id)
        effective_settings = _evidence_settings(settings, evidence_id)
    promoted, ranked = _shared_run_stage(
        plan,
        variants,
        effective_settings,
        phase=effective_phase,
        python=python,
        force=force,
    )
    if phase == "stage_b":
        reference_id = str(settings["reference_id"])
        promoted = _select_eligible_stage_b_routes(
            ranked,
            reference_id=reference_id,
            top_k=int(settings.get("top_k", 3)),
        )
        decision_dir = Path(str(plan["output_dir"])) / effective_phase
        _write_rankings(decision_dir, promoted, ranked)
        if evidence_id is None:
            raise ValueError("Stage B requires a selected evidence identity")
        _record_evidence_provenance(decision_dir, evidence_id)
        print(f"stage_b eligible routes: {promoted or '(none)'}")
    return promoted, ranked


def _selected_evidence_from_decision(
    plan: Mapping[str, Any], decision: Mapping[str, Any]
) -> str | None:
    promoted = list(decision.get("promoted", []))
    if not promoted:
        return None
    selected_id = str(promoted[0])
    selected = _stage_a_variant(plan, selected_id)
    ranked_row = next(
        (row for row in decision.get("ranked", []) if row.get("id") == selected_id),
        None,
    )
    if ranked_row is None or ranked_row.get("preset") != selected.get("preset"):
        raise ValueError(
            f"Stage A decision provenance for {selected_id} does not match the plan"
        )
    return selected_id


def _load_selected_evidence(
    plan: Mapping[str, Any], output_dir: Path
) -> str | None:
    return _selected_evidence_from_decision(
        plan, _load_decision(output_dir, "stage_a")
    )


def _load_stage_b_decision(
    plan: Mapping[str, Any], output_dir: Path, evidence_id: str
) -> Mapping[str, Any]:
    phase = _evidence_phase("stage_b", evidence_id)
    decision = _load_decision(output_dir, phase)
    if decision.get("selected_evidence_id") != evidence_id:
        raise ValueError(
            "Stage B decision evidence provenance mismatch: "
            f"expected {evidence_id}, found "
            f"{decision.get('selected_evidence_id')!r}"
        )
    expected_preset = str(_stage_a_variant(plan, evidence_id)["preset"])
    expected_experiment_token = _evidence_slug(evidence_id)
    for row in decision.get("ranked", []):
        if row.get("preset") != expected_preset or expected_experiment_token not in str(
            row.get("experiment", "")
        ):
            raise ValueError(
                f"Stage B decision mixes artifacts outside {expected_experiment_token}"
            )
    return decision


def _write_promotion_outputs(
    plan: Mapping[str, Any],
    promotion_records: Sequence[Mapping[str, Any]],
    stage_b_decision: Mapping[str, Any],
) -> None:
    output_dir = Path(str(plan["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_id = str(plan["stage_b"]["reference_id"])

    # The reference (T_native) must be present in 300-epoch records so the
    # final comparison is fair: candidate-300 vs T_native-300, not vs a
    # 50-epoch Stage-B baseline.
    promotion_by_id: Dict[str, Any] = {
        str(row["id"]): row for row in promotion_records
    }
    if reference_id not in promotion_by_id:
        raise ValueError(
            f"Reference route {reference_id} was not trained to 300 epochs; "
            f"it must be included in the promotion batch for a fair comparison."
        )
    reference = promotion_by_id[reference_id]
    evidence_id_value = stage_b_decision.get("selected_evidence_id")
    evidence_id = str(evidence_id_value) if evidence_id_value else None

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
        if passed and str(row["id"]) != reference_id:
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

    promote_phase = (
        _evidence_phase("promote", evidence_id) if evidence_id else "promote"
    )
    stage_b_phase = (
        _evidence_phase("stage_b", evidence_id) if evidence_id else "stage_b"
    )
    result_paths = {
        str(row["id"]): output_dir / promote_phase / f"{str(row['id']).lower()}.json"
        for row in promotion_records
    }
    # Final paired comparison uses only 300-epoch results — all entries in
    # promotion_records (including the reference) are at the same epoch budget.
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
        selected_evidence_id = None
        stage_a_decision = output_dir / "stage_a" / "promotion_decision.json"
        if args.stage in {"stage-b", "promote"} and stage_a_decision.is_file():
            selected_evidence_id = _load_selected_evidence(plan, output_dir)
        if args.stage == "promote" and selected_evidence_id is not None:
            stage_b_decision = (
                output_dir
                / _evidence_phase("stage_b", selected_evidence_id)
                / "promotion_decision.json"
            )
            if stage_b_decision.is_file():
                _load_stage_b_decision(plan, output_dir, selected_evidence_id)
        manifest = build_v5_dry_run_manifest(
            plan,
            python=sys.executable,
            stage=args.stage,
            selected_evidence_id=selected_evidence_id,
        )
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
    selected_evidence: list[str] = []
    if args.stage in {"stage-a", "all"}:
        selected_evidence, _ = _run_v5_stage(
            plan,
            plan["stage_a"]["variants"],
            plan["stage_a"],
            phase="stage_a",
            python=sys.executable,
            force=args.force,
        )
        if args.stage == "stage-a":
            return
    selected_evidence_id = _load_selected_evidence(plan, output_dir)
    if selected_evidence and selected_evidence_id != selected_evidence[0]:
        raise ValueError("Fresh Stage A selection does not match its decision file")
    if selected_evidence_id is None:
        print("No Stage A evidence candidate passed; Stage B and promotion stopped.")
        return

    stage_b_variants = build_stage_b_variants(plan, selected_evidence_id)
    promoted_routes: list[str]
    if args.stage in {"stage-b", "all"}:
        promoted_routes, _ = _run_v5_stage(
            plan,
            stage_b_variants,
            plan["stage_b"],
            phase="stage_b",
            python=sys.executable,
            force=args.force,
            evidence_id=selected_evidence_id,
        )
        if args.stage == "stage-b":
            return
    stage_b_decision = _load_stage_b_decision(
        plan, output_dir, selected_evidence_id
    )
    promoted_routes = list(stage_b_decision.get("promoted", []))
    if not promoted_routes:
        print(
            "No Stage B candidate passed; promoting the T_native reference "
            "alone to preserve the matched 300-epoch baseline."
        )

    # Always include the reference architecture (T_native) in the 300-epoch
    # batch so the final paired comparison uses matched-budget endpoints.
    reference_id = str(plan["stage_b"]["reference_id"])
    promoted_ids = _promotion_route_ids(
        promoted_routes,
        reference_id=reference_id,
        top_k=int(plan["stage_b"].get("top_k", 3)),
    )
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
            evidence_id=selected_evidence_id,
        ),
        force=args.force,
        checkpoint_validator=_validate_promotion_checkpoint,
    )
    _write_promotion_outputs(
        plan, promotion_records, stage_b_decision
    )


if __name__ == "__main__":
    main()
