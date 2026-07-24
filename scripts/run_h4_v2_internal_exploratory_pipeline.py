"""One-command frozen internal exploratory H4-v2 nested-CV pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import load_json, write_json


SCHEMA_VERSION = 2
PIPELINE_ROOT = (
    "results/mechanism_validation_v2/"
    "02_h4_v2_internal_exploratory_nested_cv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze patient-only nested partitions and run the existing "
            "155-patient H4-v2 internal exploratory evaluation."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--config",
        default="configs/h4_v2_internal_exploratory_nested_cv_v1.json",
    )
    parser.add_argument("--plan", action="store_true")
    return parser


def _run_command(command: list[str], *, root: Path) -> int:
    print("[H4-v2 internal pipeline] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    config_path = (
        Path(args.config).resolve()
        if Path(args.config).is_absolute()
        else (root / args.config).resolve()
    )
    config = load_json(config_path)
    plan_output = root / PIPELINE_ROOT / "00_frozen_plan"
    evaluation_output = root / PIPELINE_ROOT / "01_outer_evaluation"
    pipeline_output = root / PIPELINE_ROOT / "99_pipeline"
    freeze_command = [
        sys.executable,
        "scripts/freeze_h4_v2_internal_cv_plan.py",
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--output",
        str(plan_output),
    ]
    evaluation_command = [
        sys.executable,
        "scripts/validate_h4_v2_internal_cv.py",
        "--root",
        str(root),
        "--config",
        str(config_path),
        "--plan",
        str(plan_output / "frozen_plan.json"),
        "--output",
        str(evaluation_output),
    ]
    if args.plan:
        print(
            json.dumps(
                {
                    "pipeline_id": config["pipeline_id"],
                    "scope": config["scope"],
                    "commands": [freeze_command, evaluation_command],
                    "outer_evaluation_has_run": (
                        evaluation_output / "decision.json"
                    ).exists(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    freeze_code = _run_command(freeze_command, root=root)
    pipeline_output.mkdir(parents=True, exist_ok=True)
    if freeze_code != 0:
        decision = {
            "schema_version": SCHEMA_VERSION,
            "stage": "99_H4_v2_internal_exploratory_pipeline",
            "pipeline_id": config["pipeline_id"],
            "decision": "FAIL",
            "failure_phase": "PLAN_FREEZE",
            "outer_evaluation_run": False,
            "recommended_for_subsequent_internal_use": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(pipeline_output / "decision.json", decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return freeze_code

    evaluation_code = _run_command(evaluation_command, root=root)
    evaluation_decision = load_json(evaluation_output / "decision.json")
    completed = (
        "conclusion" in evaluation_decision
        and evaluation_decision.get("decision") in {"PASS", "FAIL"}
    )
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage": "99_H4_v2_internal_exploratory_pipeline",
        "pipeline_id": config["pipeline_id"],
        "decision": evaluation_decision.get("decision", "FAIL"),
        "plan_sha256": evaluation_decision.get("plan_sha256"),
        "partition_fingerprint": evaluation_decision.get(
            "partition_fingerprint"
        ),
        "conclusion": evaluation_decision.get("conclusion"),
        "completed_outer_evaluation": completed,
        "H4_v1_preserved": evaluation_decision.get(
            "H4_v1_preserved",
            "FAIL",
        ),
        "recommended_for_subsequent_internal_use": evaluation_decision.get(
            "recommended_for_subsequent_internal_use",
            False,
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if decision["conclusion"] is None:
        decision.pop("conclusion")
    write_json(pipeline_output / "decision.json", decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return evaluation_code


if __name__ == "__main__":
    raise SystemExit(main())
