"""One-command H4-v2 development and external confirmation pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mechanism_validation.common import load_json, write_json


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _run(command: list[str], *, root: Path) -> int:
    print("[H4-v2 pipeline] " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=root, check=False).returncode


def _finish(
    output: Path,
    *,
    decision: str,
    reason: str,
    stages: dict[str, Any],
    next_stage: str | None = None,
) -> int:
    payload = {
        "schema_version": 2,
        "stage": "99_H4_v2_pipeline",
        "decision": decision,
        "reason": reason,
        "stages": stages,
        "next_stage_allowed": decision == "PASS",
        "next_stage": next_stage if decision == "PASS" else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "decision.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if decision in {"PASS", "DEFERRED_NEW_COHORT"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze H4-v2 locally; optionally audit, seal, and confirm a "
            "new external cohort in the same command."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-root",
        default="results/mechanism_validation_v2",
    )
    parser.add_argument("--external-raw-root")
    parser.add_argument("--external-manifest")
    parser.add_argument("--external-cache-dir")
    parser.add_argument("--confirmation-cohort-id")
    parser.add_argument("--confirmation-split", default="test")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the frozen pipeline definition without running it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    output_root = _resolve(root, args.output_root)
    pipeline_output = output_root / "99_pipeline"
    pipeline_output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs" / "mechanism_validation_pipeline_v2.json"
    if args.plan:
        print(config_path.read_text(encoding="utf-8"))
        return 0

    stages: dict[str, Any] = {}
    development_output = output_root / "00_h4_v2_development"
    external_values = {
        "external_raw_root": args.external_raw_root,
        "external_manifest": args.external_manifest,
        "external_cache_dir": args.external_cache_dir,
        "confirmation_cohort_id": args.confirmation_cohort_id,
    }
    supplied = [name for name, value in external_values.items() if value]
    if supplied and not (development_output / "frozen_probe.json").is_file():
        stages["frozen_probe"] = {
            "status": "FAIL",
            "required_path": str(
                development_output / "frozen_probe.json"
            ),
        }
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason=(
                "External confirmation requires the previously frozen "
                "development probe; cloud-side refreezing is forbidden"
            ),
            stages=stages,
        )
    code = _run(
        [
            sys.executable,
            "scripts/develop_h4_v2_uncertainty_aware.py",
            "--root",
            str(root),
            "--output",
            str(development_output),
        ],
        root=root,
    )
    if code != 0:
        stages["development"] = {"status": "FAIL", "exit_code": code}
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason="H4-v2 development/freeze failed",
            stages=stages,
        )
    development = load_json(development_output / "decision.json")
    stages["development"] = {
        "status": development["decision"],
        "probe_sha256": development["probe_sha256"],
    }

    if not supplied:
        return _finish(
            pipeline_output,
            decision="DEFERRED_NEW_COHORT",
            reason=(
                "H4-v2 is frozen; formal confirmation requires a new "
                "zero-overlap patient cohort"
            ),
            stages=stages,
        )
    missing = [name for name, value in external_values.items() if not value]
    if missing:
        stages["external_inputs"] = {
            "status": "FAIL",
            "missing": missing,
        }
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason="External confirmation inputs are incomplete",
            stages=stages,
        )

    raw_root = _resolve(root, args.external_raw_root)
    manifest = _resolve(root, args.external_manifest)
    cache_dir = _resolve(root, args.external_cache_dir)
    audit_output = output_root / "01_external_stage0a"
    code = _run(
        [
            sys.executable,
            "scripts/run_mechanism_stage0_data_audit.py",
            "--root",
            str(root),
            "--raw-root",
            str(raw_root),
            "--manifest",
            str(manifest),
            "--output",
            str(audit_output),
            "--skip-checkpoint-inventory",
        ],
        root=root,
    )
    if code != 0:
        stages["external_stage0a"] = {
            "status": "FAIL",
            "exit_code": code,
        }
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason="External Stage 0A data gate failed",
            stages=stages,
        )
    audit = load_json(audit_output / "decision.json")
    stages["external_stage0a"] = {"status": audit["data_gate"]}
    contract = audit_output / "dataset_contract.json"

    stage0c_output = output_root / "02_external_stage0c"
    code = _run(
        [
            sys.executable,
            "scripts/run_cloud_stage0c_gate.py",
            "--root",
            str(root),
            "--contract",
            str(contract),
            "--manifest",
            str(manifest),
            "--raw-root",
            str(raw_root),
            "--cache-dir",
            str(cache_dir),
            "--output",
            str(stage0c_output),
            "--seal-cache",
        ],
        root=root,
    )
    stage0c = (
        load_json(stage0c_output / "decision.json")
        if (stage0c_output / "decision.json").is_file()
        else {}
    )
    stages["external_stage0c"] = {
        "status": stage0c.get("cloud_training_gate", "FAIL"),
        "exit_code": code,
    }
    if code != 0 or stage0c.get("cloud_training_gate") != "PASS":
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason="External Stage 0C cache/lineage gate failed",
            stages=stages,
        )

    confirmation_output = output_root / "03_h4_v2_confirmation"
    code = _run(
        [
            sys.executable,
            "scripts/validate_h4_v2_external_confirmation.py",
            "--root",
            str(root),
            "--probe",
            str(development_output / "frozen_probe.json"),
            "--external-contract",
            str(contract),
            "--external-manifest",
            str(manifest),
            "--external-cache-dir",
            str(cache_dir),
            "--external-cache-lineage",
            str(cache_dir / "cache_lineage.json"),
            "--confirmation-cohort-id",
            str(args.confirmation_cohort_id),
            "--confirmation-split",
            str(args.confirmation_split),
            "--output",
            str(confirmation_output),
            "--num-workers",
            str(args.num_workers),
            "--device",
            str(args.device),
        ],
        root=root,
    )
    confirmation = (
        load_json(confirmation_output / "decision.json")
        if (confirmation_output / "decision.json").is_file()
        else {}
    )
    stages["external_confirmation"] = {
        "status": confirmation.get("decision", "FAIL"),
        "exit_code": code,
    }
    if code != 0 or confirmation.get("decision") != "PASS":
        return _finish(
            pipeline_output,
            decision="FAIL",
            reason=(
                "Frozen H4-v2 failed formal external confirmation; "
                "downstream mechanisms remain blocked"
            ),
            stages=stages,
        )
    return _finish(
        pipeline_output,
        decision="PASS",
        reason="Frozen H4-v2 passed all external confirmation gates",
        stages=stages,
        next_stage="CT_support_head",
    )


if __name__ == "__main__":
    raise SystemExit(main())
