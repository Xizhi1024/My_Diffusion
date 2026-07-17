#!/usr/bin/env python3
"""Run the matched-budget V5 T0/C1 300-epoch rescue experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_v4_results import compare_all_results
from scripts.run_frequency_ablations import (
    _require_planned_mean_checkpoint,
    _run_entries,
)
from scripts.run_spectral_router_v5 import (
    _build_entries,
    _validate_promotion_checkpoint,
    build_stage_b_variants,
)


DEFAULT_PLAN = Path(
    "configs/experiments/spectral_router_ablation_plan_v5.yaml"
)
DEFAULT_EVIDENCE_ID = "S3"
RESCUE_ROUTE_IDS = ("T0", "C1")


def load_plan(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if not isinstance(plan, dict):
        raise TypeError(f"Expected a mapping in V5 plan: {path}")
    return plan


def build_rescue_entries(
    plan: Mapping[str, Any],
    *,
    python: str = sys.executable,
    evidence_id: str = DEFAULT_EVIDENCE_ID,
) -> list[dict[str, Any]]:
    """Build matched from-scratch T0/C1 entries with exact epoch-300 outputs."""

    variants = {
        str(variant["id"]): variant
        for variant in build_stage_b_variants(plan, evidence_id)
    }
    missing = [route_id for route_id in RESCUE_ROUTE_IDS if route_id not in variants]
    if missing:
        raise KeyError(f"Missing V5 rescue routes: {missing}")
    selected = [variants[route_id] for route_id in RESCUE_ROUTE_IDS]
    entries = _build_entries(
        plan,
        selected,
        plan["promote"],
        python,
        "promote",
        evidence_id=evidence_id,
    )
    expected_epoch = int(plan["promote"]["epochs"])
    if expected_epoch != 300:
        raise ValueError(
            f"T0/C1 rescue requires exactly 300 epochs, got {expected_epoch}"
        )
    return entries


def _print_manifest(entries: Sequence[Mapping[str, Any]]) -> None:
    for entry in entries:
        print(f"\n[{entry['id']}] train")
        print(subprocess.list2cmdline(list(entry["train_command"])))
        print(f"[{entry['id']}] evaluate")
        print(subprocess.list2cmdline(list(entry["eval_command"])))


def _comparison_output(entries: Sequence[Mapping[str, Any]]) -> Path:
    if not entries:
        raise ValueError("T0/C1 rescue produced no entries")
    return Path(str(entries[0]["result"])).parent / "paired_t0_c1_epoch300.json"


def run_rescue(
    plan: Mapping[str, Any],
    *,
    force: bool = False,
    dry_run: bool = False,
    python: str = sys.executable,
    evidence_id: str = DEFAULT_EVIDENCE_ID,
) -> Path:
    entries = build_rescue_entries(
        plan,
        python=python,
        evidence_id=evidence_id,
    )
    output = _comparison_output(entries)
    if dry_run:
        _print_manifest(entries)
        print(f"\nPaired comparison output: {output}")
        return output

    _require_planned_mean_checkpoint(plan)
    _run_entries(
        entries,
        force=force,
        checkpoint_validator=_validate_promotion_checkpoint,
    )

    result_paths = {
        str(entry["id"]): Path(str(entry["result"])) for entry in entries
    }
    report = compare_all_results(
        result_paths,
        plan["paired_comparison"]["metrics"],
        seed=int(plan["paired_comparison"]["seed"]),
        resamples=int(plan["paired_comparison"]["resamples"]),
    )
    report.update({
        "selected_evidence_id": evidence_id,
        "checkpoint_epoch": 300,
        "comparison": list(RESCUE_ROUTE_IDS),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nT0/C1 epoch-300 paired comparison saved to {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--evidence-id", default=DEFAULT_EVIDENCE_ID)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and reevaluate even when outputs already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact train/evaluation commands without executing them",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = load_plan(args.plan)
    run_rescue(
        plan,
        force=args.force,
        dry_run=args.dry_run,
        evidence_id=str(args.evidence_id),
    )


if __name__ == "__main__":
    main()
